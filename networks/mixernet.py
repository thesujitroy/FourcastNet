"""MixerNet: FourCastNet's AFNONet with the token mixer chosen by config.

A copy of networks/afnonet.py::AFNONet in which each block's `filter` is built by
`networks.mixers.build_mixer(params.mixer)`. Module names are AFNONet's, so with
`mixer: {type: afno}` (or no `mixer` at all) the model is AFNONet: same state_dict
keys and the same function; released FourCastNet checkpoints load strictly.

Initialization is AFNONet's (trunc_normal 0.02 on every Linear, zero biases, unit
LayerNorms, trunc_normal pos_embed), with one difference: each block's mixer is built
and initialized from its own random stream, seeded from a single draw of the global
one. Every mixer therefore consumes the global stream identically, so for a given
seed the patch embedding, position embedding, MLPs and head start from the SAME
weights whatever the mixer, and runs differ only in the mixer. (Upstream AFNONet
draws its AFNO weights inline, so its initial weights are not reproduced bit for bit.)

Block, for every mixer:

    n = norm1(x);  x = x + mix(n);  x = x + drop_path(mlp(norm2(x)))
    mix(n) = filter(n) + n      (AFNO2D adds the n itself; others via inner_skip)
"""
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange

try:
    from timm.layers import DropPath, trunc_normal_
except ImportError:  # timm < 0.9
    from timm.models.layers import DropPath, trunc_normal_

from networks.mixers import build_mixer


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def init_weights(m):
    """AFNONet._init_weights."""
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class MixerBlock(nn.Module):
    def __init__(self, dim, grid_size, mixer, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, double_skip=True, mixer_seed=None, **afno_defaults):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # The mixer draws from its own stream (see the module docstring). Its Linear layers get
        # AFNONet's init here, and the adaptive Tucker gate is re-zeroed after it.
        with torch.random.fork_rng(devices=[]):
            if mixer_seed is not None:
                torch.default_generator.manual_seed(mixer_seed)   # CPU only; torch.manual_seed would reseed CUDA too
            self.filter, self.add_input = build_mixer(mixer, dim, grid_size, **afno_defaults)
            self.filter.apply(init_weights)
            if hasattr(self.filter, "reset_conditioning_"):
                self.filter.reset_conditioning_()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.double_skip = double_skip

    def mix(self, x):
        y = self.filter(x)
        return y + x if self.add_input else y

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.mix(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class MixerNet(nn.Module):
    """AFNONet's signature; `params.patch_size`, `N_in_channels`, `N_out_channels`,
    `num_blocks` and (optionally) `mixer` are read from params, as AFNONet does."""

    def __init__(
            self,
            params,
            img_size=(720, 1440),
            embed_dim=768,
            depth=12,
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
        ):
        super().__init__()
        self.params = params
        self.img_size = img_size
        self.patch_size = (params.patch_size, params.patch_size)
        self.in_chans = params.N_in_channels
        self.out_chans = params.N_out_channels
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = params.num_blocks
        self.mixer = getattr(params, "mixer", None) or {"type": "afno"}
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.h = img_size[0] // self.patch_size[0]
        self.w = img_size[1] // self.patch_size[1]

        mixer_seed = int(torch.randint(0, 2**31 - depth, ()).item())  # one draw, whatever the mixer
        self.blocks = nn.ModuleList([
            MixerBlock(dim=embed_dim, grid_size=(self.h, self.w), mixer=self.mixer, mlp_ratio=mlp_ratio,
                       drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer, mixer_seed=mixer_seed + i,
                       num_blocks=self.num_blocks, sparsity_threshold=sparsity_threshold,
                       hard_thresholding_fraction=hard_thresholding_fraction)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)  # unused in forward, kept for AFNONet state_dict parity

        self.head = nn.Linear(embed_dim, self.out_chans*self.patch_size[0]*self.patch_size[1], bias=False)

        trunc_normal_(self.pos_embed, std=.02)
        # AFNONet's self.apply(self._init_weights), minus the mixers (initialized in their blocks)
        in_mixers = {id(m) for block in self.blocks for m in block.filter.modules()}
        for m in self.modules():
            if id(m) not in in_mixers:
                init_weights(m)

    _init_weights = staticmethod(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = x.reshape(B, self.h, self.w, self.embed_dim)
        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        x = rearrange(
            x,
            "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            h=self.img_size[0] // self.patch_size[0],
            w=self.img_size[1] // self.patch_size[1],
        )
        return x

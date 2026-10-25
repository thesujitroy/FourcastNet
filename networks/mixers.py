"""Token mixers for MixerNet, all mapping (B, H, W, C) -> (B, H, W, C).

`build_mixer` is the only entry point. It validates the options of one mixer
config (unknown keys are rejected, not ignored) and returns the module plus a flag
saying whether the host block must add the mixer input to its output.

FourCastNet's AFNO2D returns `F(x) + x`: the `+ x` is part of FourCastNet's block,
so for a like-for-like swap every other mixer gets it too (`inner_skip`, default
true). AFNO2D keeps its own and cannot switch it off.

LS attention is not offered: the existing implementation needs a square token grid
and the ERA5 grid is 90 x 180 at patch 8.

The spectral and adaptive Tucker maps both run in fp32 under autocast (the spectral map
does it itself; the adaptive map is wrapped here).
"""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from attention.adaptive_spectral_tucker import AdaptiveSpectralTuckerMap
from attention.basic_tucker_map import FreeTuckerMap
from attention.spectral_tucker_map import SpectralTuckerMap

KINDS = ("afno", "vanilla", "tucker_free", "tucker_spectral", "tucker_adaptive")


class AFNO2D(nn.Module):
    """Verbatim copy of networks/afnonet.py::AFNO2D (that file no longer imports
    under numpy 2; tests/test_mixers.py checks this copy against it)."""

    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, H, W // 2 + 1, C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        x = x.type(dtype)

        return x + bias


class AdaptiveSpectralTuckerMapFP32(AdaptiveSpectralTuckerMap):
    """AdaptiveSpectralTuckerMap with autocast off, as SpectralTuckerMap does internally.
    Under bf16 autocast the latitude profile `1 + basis @ coeff` and the gate MLP would run
    in bf16, rounding the profile to steps of 2^-7 near 1 and handicapping only this mixer.
    Same parameters and state_dict keys as the parent."""

    def forward(self, x):
        dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x if dtype in (torch.float32, torch.float64) else x.float()
            return super().forward(x).to(dtype)


class GridAttention(nn.Module):
    """Global multi-head self-attention over all H*W tokens (Helio's VanillaAttention)."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        q, k, v = self.qkv(x.reshape(B, H * W, C)).reshape(
            B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop if self.training else 0.)
        return self.proj_drop(self.proj(y.transpose(1, 2).reshape(B, H * W, C))).reshape(B, H, W, C)


_OPTIONS = {
    "afno": {"num_blocks", "sparsity_threshold", "hard_thresholding_fraction", "hidden_size_factor"},
    "vanilla": {"num_heads", "qkv_bias", "attn_drop", "proj_drop"},
    "tucker_free": {"L", "p", "q", "r", "zero_init"},
    "tucker_spectral": {"L", "kh", "kw", "r", "basis", "zero_init"},
    "tucker_adaptive": {"L", "kh", "kw", "r", "ctx_hidden", "lat_basis", "gate_range", "zero_init"},
}


def build_mixer(config, dim, grid_size, **afno_defaults):
    """Return (mixer, add_input). `afno_defaults` are AFNONet's arguments
    (num_blocks, sparsity_threshold, hard_thresholding_fraction), used when the
    afno config does not set them."""
    cfg = deepcopy(config or {"type": "afno"})
    kind = cfg.pop("type", "afno")
    if kind not in KINDS:
        raise ValueError(f"Unknown mixer type {kind!r}; choose from {KINDS}")
    inner_skip = cfg.pop("inner_skip", True)
    extra = set(cfg) - _OPTIONS[kind]
    if extra:
        raise ValueError(f"Unsupported {kind} mixer options: {sorted(extra)}")
    H, W = grid_size

    if kind == "afno":
        if not inner_skip:
            raise ValueError("AFNO2D adds its input internally; inner_skip cannot be disabled for afno")
        cfg = {**afno_defaults, **cfg}
        if dim % cfg.get("num_blocks", 8):
            raise ValueError(f"embed_dim {dim} must be divisible by AFNO num_blocks {cfg['num_blocks']}")
        return AFNO2D(dim, **cfg), False

    if kind == "vanilla":
        heads = cfg.get("num_heads", 8)
        if heads < 1 or dim % heads:
            raise ValueError(f"embed_dim {dim} must be divisible by num_heads {heads}")
        return GridAttention(dim, **cfg), inner_skip

    cfg = {"L": 4, "r": 16, "zero_init": False, **cfg}
    if min(cfg["L"], cfg["r"]) < 1 or cfg["r"] > dim:
        raise ValueError("Tucker requires L >= 1 and 1 <= r <= embed_dim")
    if kind == "tucker_free":
        cfg = {"p": 4, "q": 4, **cfg}
        if not (1 <= cfg["p"] <= H and 1 <= cfg["q"] <= W):
            raise ValueError(f"tucker_free needs 1 <= p <= {H} and 1 <= q <= {W}")
        return FreeTuckerMap(dim, (H, W), **cfg), inner_skip

    cfg = {"kh": 4, "kw": 4, **cfg}
    limits = (H, W) if cfg.get("basis") == "dct" else ((H - 1) // 2 + 1, (W - 1) // 2 + 1)
    if not (1 <= cfg["kh"] <= limits[0] and 1 <= cfg["kw"] <= limits[1]):
        raise ValueError(f"retained modes (kh, kw) = ({cfg['kh']}, {cfg['kw']}) exceed the "
                         f"grid limits {limits} of a {H} x {W} token grid")
    if kind == "tucker_spectral":
        return SpectralTuckerMap(dim, **cfg), inner_skip
    return AdaptiveSpectralTuckerMapFP32(dim, **cfg), inner_skip

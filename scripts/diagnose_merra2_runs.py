#!/usr/bin/env python3
"""Checks on the trained MERRA-2 mixer runs (one GPU, a few minutes).

1. Reference check: the afno run's best checkpoint is loaded strictly into upstream
   networks/afnonet.py::AFNONet and into MixerNet; both must give the same forecast.
2. One 6 h step from N evenly spaced samples of the out-of-sample year, for every config:
   latitude-weighted RMSE (physical units) in fp32 and under bf16 autocast (how the runs
   were trained and validated), the relative L2 error (the training loss), persistence,
   and the share of the squared error that lies inside the 8 x 8 patches (what is left
   after each patch's mean error is removed; the latitude weight is constant per patch,
   so "inside" and "patch mean" add up to the total exactly).
3. The dtype AFNO's Fourier-space einsum runs in under bf16 autocast.

    python scripts/diagnose_merra2_runs.py --output results/merra2_diagnose.json
"""
import argparse
import glob
import json
import os
import sys
import types
from collections import OrderedDict

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path[:0] = [os.path.join(HERE, '..'), os.path.join(HERE, '..', '..')]  # FourCastNet, repo root (attention/)

import numpy
if 'numpy.lib.arraypad' not in sys.modules:   # upstream afnonet.py's dead import (as tests/test_mixers.py)
    _shim = types.ModuleType('numpy.lib.arraypad')
    _shim.pad = numpy.pad
    sys.modules['numpy.lib.arraypad'] = _shim

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from utils.mixer_params import Params, disable_transparent_hugepages
from utils.weighted_acc_rmse import weighted_rmse_torch_channels
from networks.afnonet import AFNONet
from networks.mixernet import MixerNet

CONFIGS = ['afno', 'tucker_free', 'tucker_spectral', 'tucker_spectral_dct', 'tucker_adaptive']
SHOW = ['h_l51', 'h_l34', 'sp', 'slp', 't2m', 't_l63', 'u10', 'u_l51', 'qv_l63', 'tqv']


def load_state(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = OrderedDict((k[7:] if k.startswith('module.') else k, v) for k, v in checkpoint['model_state'].items())
    return state, checkpoint.get('mixer'), checkpoint.get('epoch')


def patch_weight(H, patch, device):
    """Latitude weight of weighted_acc_rmse.py, averaged over each patch (mean 1)."""
    j = torch.arange(H, device=device, dtype=torch.float)
    w = torch.cos(3.1416 / 180. * (90. - j * 180. / float(H - 1)))
    w = w.reshape(H // patch, patch).mean(1).repeat_interleave(patch)
    return (w / w.mean()).reshape(1, 1, H, 1)


def inside_patch_split(err, weight, patch):
    """Per channel: (weighted squared error summed over the batch, the part inside patches)."""
    mean = F.interpolate(F.avg_pool2d(err, patch), scale_factor=patch, mode='nearest')
    total = (weight * err ** 2).mean(dim=(-1, -2)).sum(0)
    inside = (weight * (err - mean) ** 2).mean(dim=(-1, -2)).sum(0)
    return total.double(), inside.double()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--yaml_config', default='config/mixer_comparison_merra2.yaml')
    parser.add_argument('--configs', nargs='+', default=CONFIGS)
    parser.add_argument('--run_num', default='0')
    parser.add_argument('--n_samples', type=int, default=64)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--split_patch', type=int, default=None,
                        help='block size of the inside/patch-mean error split (default: the patch size of the config)')
    parser.add_argument('--output', default=None)
    args = parser.parse_args(argv)
    disable_transparent_hugepages()
    device = torch.device('cuda', 0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    base = Params(os.path.abspath(args.yaml_config), args.configs[0])
    names = [base.channel_names[c] for c in base.out_channels]
    out = np.array(base.out_channels)
    data_file = sorted(glob.glob(base.inf_data_path + '/*.h5'))[0]
    fields = h5py.File(data_file, 'r')['fields']
    H, W = fields.shape[2] - 1, fields.shape[3]   # the training loader drops the last latitude row
    means = np.load(base.global_means_path)[0, out]
    stds = np.load(base.global_stds_path)[0, out]
    std = torch.as_tensor(stds[:, 0, 0], device=device, dtype=torch.float)
    picks = np.linspace(0, fields.shape[0] - 2, args.n_samples).round().astype(int)
    print(f'data {data_file}: {fields.shape}, {len(picks)} samples, first/last index {picks[0]}/{picks[-1]}, '
          f'grid {H} x {W}', flush=True)
    pairs = np.stack([(fields[i:i + 2, out, 0:H] - means) / stds for i in picks]).astype(np.float32)
    inp_all, tar_all = torch.as_tensor(pairs[:, 0]), torch.as_tensor(pairs[:, 1])
    split = args.split_patch or int(base.patch_size)
    weight = patch_weight(H, split, device)
    report = {'data_file': data_file, 'samples': picks.tolist(), 'channels': names, 'split_patch': split,
              'model_patch': int(base.patch_size), 'runs': {}}

    # persistence and the size of the true 6 h change
    rmse, tot, ins, rel = 0., 0., 0., 0.
    for s in range(0, len(picks), args.batch):
        x, y = inp_all[s:s + args.batch].to(device), tar_all[s:s + args.batch].to(device)
        rmse = rmse + (weighted_rmse_torch_channels(x, y) * std).sum(0).double()
        t, i = inside_patch_split(x - y, weight, split)
        tot, ins = tot + t, ins + i
        rel = rel + ((x - y).flatten(1).norm(dim=1) / y.flatten(1).norm(dim=1)).sum().double()
    report['persistence'] = {'rmse': dict(zip(names, (rmse / len(picks)).tolist())),
                             'inside_patch_share': dict(zip(names, (ins / tot).tolist())),
                             'relative_l2': float(rel / len(picks))}

    for config in args.configs:
        params = Params(os.path.abspath(args.yaml_config), config)
        params['N_in_channels'] = len(params.in_channels)
        params['N_out_channels'] = len(out)
        ckpt = os.path.join(params.exp_dir, config, str(args.run_num), 'training_checkpoints/best_ckpt.tar')
        state, trained_mixer, epoch = load_state(ckpt, device)
        assert trained_mixer == params.mixer, (trained_mixer, params.mixer)
        model = MixerNet(params, img_size=(H, W), embed_dim=params.embed_dim, depth=params.depth).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        run = {'checkpoint': ckpt, 'epoch': epoch, 'mixer': params.mixer}

        if params.mixer['type'] == 'afno':
            ref = AFNONet(params, img_size=(H, W), embed_dim=params.embed_dim, depth=params.depth).to(device)
            ref.load_state_dict(state, strict=True)   # upstream FourCastNet class, same weights
            ref.eval()
            with torch.no_grad():
                x = inp_all[:args.batch].to(device)
                a, b = ref(x), model(x)
            run['upstream_afnonet'] = {'state_dict_keys_equal': list(ref.state_dict()) == list(model.state_dict()),
                                       'outputs_bit_identical': bool(torch.equal(a, b)),
                                       'max_abs_difference': float((a - b).abs().max()),
                                       'output_rms': float(b.pow(2).mean().sqrt())}
            print(f"[{config}] upstream AFNONet vs MixerNet on the trained weights: {run['upstream_afnonet']}", flush=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                filt = model.blocks[0].filter
                probe = torch.randn(1, 4, 5, filt.num_blocks, filt.block_size, device=device)
                run['afno_einsum_dtype_under_bf16_autocast'] = str(torch.einsum('...bi,bio->...bo', probe, filt.w1[0]).dtype)
            print(f"[{config}] AFNO Fourier einsum dtype under bf16 autocast: "
                  f"{run['afno_einsum_dtype_under_bf16_autocast']}", flush=True)
            del ref

        if params.mixer.get('basis') == 'dct' and params.mixer.get('backend', 'reference') != 'gemm':
            # The DCT runs were trained on the reference path. The GEMM path is the same linear
            # map, so the trained weights must give the same forecasts through it: check in fp32.
            twin_params = Params(os.path.abspath(args.yaml_config), config)
            twin_params['N_in_channels'] = len(twin_params.in_channels)
            twin_params['N_out_channels'] = len(out)
            twin_params['mixer'] = {**params.mixer, 'backend': 'gemm'}
            twin = MixerNet(twin_params, img_size=(H, W), embed_dim=params.embed_dim, depth=params.depth).to(device)
            twin.load_state_dict(state, strict=True)
            twin.eval()
            diff, peak, rel_ref, rel_twin = 0., 0., 0., 0.
            with torch.no_grad():
                for s in range(0, len(picks), args.batch):
                    x, y = inp_all[s:s + args.batch].to(device), tar_all[s:s + args.batch].to(device)
                    a, b = model(x).float(), twin(x).float()
                    diff, peak = max(diff, float((a - b).abs().max())), max(peak, float(a.abs().max()))
                    rel_ref += float(((a - y).flatten(1).norm(dim=1) / y.flatten(1).norm(dim=1)).sum())
                    rel_twin += float(((b - y).flatten(1).norm(dim=1) / y.flatten(1).norm(dim=1)).sum())
            run['gemm_backend_twin'] = {'state_dict_keys_equal': list(twin.state_dict()) == list(model.state_dict()),
                                        'max_abs_difference': diff, 'max_abs_output': peak,
                                        'relative_l2_reference': rel_ref / len(picks),
                                        'relative_l2_gemm': rel_twin / len(picks)}
            print(f"[{config}] trained weights through the GEMM DCT path vs the reference path (fp32): "
                  f"{run['gemm_backend_twin']}", flush=True)
            del twin

        for mode in ('fp32', 'bf16'):
            rmse, tot, ins, rel = 0., 0., 0., 0.
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=mode == 'bf16'):
                for s in range(0, len(picks), args.batch):
                    x, y = inp_all[s:s + args.batch].to(device), tar_all[s:s + args.batch].to(device)
                    pred = model(x).float()
                    rmse = rmse + (weighted_rmse_torch_channels(pred, y) * std).sum(0).double()
                    t, i = inside_patch_split(pred - y, weight, split)
                    tot, ins = tot + t, ins + i
                    rel = rel + ((pred - y).flatten(1).norm(dim=1) / y.flatten(1).norm(dim=1)).sum().double()
            run[mode] = {'rmse': dict(zip(names, (rmse / len(picks)).tolist())),
                         'inside_patch_share': dict(zip(names, (ins / tot).tolist())),
                         'relative_l2': float(rel / len(picks))}
        report['runs'][config] = run
        print(f"[{config}] epoch {epoch}: relative L2 fp32 {run['fp32']['relative_l2']:.5f}, "
              f"bf16 {run['bf16']['relative_l2']:.5f}", flush=True)
        del model
        torch.cuda.empty_cache()

    show = [n for n in SHOW if n in names]
    print('\n6 h RMSE, fp32 (bf16 autocast in brackets), physical units, latitude weighted')
    print('| channel | persistence | ' + ' | '.join(args.configs) + ' |')
    print('|---|---|' + '---|' * len(args.configs))
    for n in show:
        cells = [f"{report['runs'][c]['fp32']['rmse'][n]:.4g} ({report['runs'][c]['bf16']['rmse'][n]:.4g})"
                 for c in args.configs]
        print(f"| {n} | {report['persistence']['rmse'][n]:.4g} | " + ' | '.join(cells) + ' |')
    print(f'\nShare of the squared 6 h error inside {split} x {split} blocks (fp32); "change" = the true 6 h change')
    print('| channel | change | ' + ' | '.join(args.configs) + ' |')
    print('|---|---|' + '---|' * len(args.configs))
    for n in show:
        cells = [f"{report['runs'][c]['fp32']['inside_patch_share'][n]:.2f}" for c in args.configs]
        print(f"| {n} | {report['persistence']['inside_patch_share'][n]:.2f} | " + ' | '.join(cells) + ' |')
    print(f"\nrelative L2 of persistence: {report['persistence']['relative_l2']:.5f}")
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=1)


if __name__ == '__main__':
    main()

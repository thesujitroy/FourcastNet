#!/usr/bin/env python3
"""Spectral band diagnostic for the mixer branch of a trained MixerNet (FourCastNet on MERRA-2).

The FourCastNet counterpart of HelioFM-surya_1.1/scripts/spectral_band_diagnostic.py (whose
pure-torch core it imports). The spectral Tucker map can only write inside the retained band
rho_H(m) < kh, rho_W(n) < kw (rho(m) = min(m, H - m), the folded mode index; on this grid the
index is the wavenumber on the globe, so it means the same thing at patch 8 and patch 2).
Attention is free to emit anything: where does the energy of what it emits go, and does the
part outside a band matter for the loss?

Two measurements in one pass over the run's own validation samples (the fixed, evenly spaced
`n_valid_samples` of the test years, the set every epoch of training was validated on):

(1) Energy. For every block j the 2-D spectral energy of the mixer OUTPUT  filter_j(norm1(x))
    and of its INPUT, folded onto (rho_H, rho_W) bins, DC excluded. The block's own `+ n` inner
    skip is not part of the output (it is not band-limited for the Tucker maps either).
    Reported: the fraction inside the spectral Tucker band of this patch size, the fraction
    inside every k x k square, the smallest square holding 90 / 95 / 99 %, and the white-noise
    value of each band.
(2) Ablation. The same samples are run again once per band with every mixer output projected
    onto the band (fft2 -> zero outside -> ifft2). Band 0 switches the mixer branch off. The
    paired change of the validation loss (relative L2, the training loss) says whether the
    out-of-band output is inert or load-bearing.

AFNO2D adds its input inside the module, so for the `afno` config only the branch-off ablation
(band 0: the module returns its input) is run and no energy is collected.

    python scripts/band_diagnostic_mixers.py --yaml_config config/mixer_comparison_merra2.yaml \
        --config vanilla --bands 0 3 6 12 23 --out results/band_diagnostic/p8/vanilla
    python scripts/band_diagnostic_mixers.py --self-test      # no data, CPU

Outputs in --out: summary.json (energy), ablation.json (paired losses), band_energy.npz (the
folded energy grids). One GPU. Figures: plot_diagnostic_figures.py at the repository root.
"""
import argparse
import glob
import importlib.util
import json
import math
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent)]  # FourCastNet (networks/, utils/), repo root (attention/)

import numpy as np
import torch


def _load_core():
    """The pure-torch core of the solar diagnostic (BandEnergy, band_mask, ...). That script puts
    HelioFM on sys.path when imported; undo it, HelioFM has its own top-level `utils`."""
    path = ROOT.parent / 'HelioFM-surya_1.1' / 'scripts' / 'spectral_band_diagnostic.py'
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location('helio_band_diagnostic_core', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path[:] = saved
    return module


core = _load_core()


# ---------------------------------------------------------------- band helpers (rectangular bands)
def parse_band(token):
    """'23' -> (23, 23); '90x91' -> (90, 91)."""
    kh, _, kw = str(token).partition('x')
    return int(kh), int(kw or kh)


def modes_in_band(n, k):
    """Number of DFT indices m of an n-point axis with rho(m) = min(m, n - m) < k."""
    return max(0, min(2 * k - 1, n))


def white_noise_fraction(H, W, kh, kw):
    """Expected band fraction, DC excluded, of a spatially white signal."""
    return max(0, modes_in_band(H, kh) * modes_in_band(W, kw) - 1) / (H * W - 1)


def is_full_band(H, W, kh, kw):
    return kh >= H // 2 + 1 and kw >= W // 2 + 1


def k_for_fraction(curve, frac):
    """Smallest square k whose band holds `frac` of the energy; None when no square of the
    curve does (on a non-square grid the largest square leaves out the longer axis's top modes)."""
    idx = np.nonzero(np.asarray(curve) >= frac - 1e-12)[0]
    return int(idx[0]) if len(idx) else None


@torch.no_grad()
def band_limit(x, kh, kw, chunk=128):
    """Project (B, H, W, C) onto the band rho_H < kh, rho_W < kw (DC kept). kh = 0 or kw = 0
    returns zeros; a band holding every mode returns x. Output has the dtype of x."""
    if kh <= 0 or kw <= 0:
        return torch.zeros_like(x)
    B, H, W, C = x.shape
    if is_full_band(H, W, kh, kw):
        return x
    mask = core.band_mask(H, W, kh, kw, x.device)
    g = x.float().permute(0, 3, 1, 2)                           # (B, C, H, W)
    out = torch.empty(B, C, H, W, dtype=torch.float32, device=x.device)
    for c0 in range(0, C, chunk):
        X = torch.fft.fft2(g[:, c0:c0 + chunk], dim=(-2, -1), norm='ortho')
        out[:, c0:c0 + chunk] = torch.fft.ifft2(X * mask, dim=(-2, -1), norm='ortho').real
    return out.permute(0, 2, 3, 1).to(x.dtype)


def self_test():
    torch.manual_seed(0)
    H, W, C, B = 9, 16, 8, 2                                    # odd H, as the 45-row patch-8 grid
    x = torch.randn(B, H, W, C)
    assert parse_band('23') == (23, 23) and parse_band('90x91') == (90, 91)
    assert modes_in_band(45, 23) == 45 and modes_in_band(72, 23) == 45 and modes_in_band(180, 90) == 179
    assert modes_in_band(288, 91) == 181 and modes_in_band(16, 9) == 16 and modes_in_band(16, 0) == 0
    # projection: idempotent, identity on the full band, zero on the empty band, dtype kept
    y = band_limit(x, 3, 4)
    assert torch.allclose(band_limit(y, 3, 4), y, atol=1e-5)
    assert band_limit(x, H // 2 + 1, W // 2 + 1) is x
    assert band_limit(x, 0, 0).abs().max() == 0
    assert band_limit(x.to(torch.bfloat16), 3, 4).dtype == torch.bfloat16
    # a band that holds every row (kh = (H - 1) / 2 + 1 on an odd axis) limits the width only
    rows_all = band_limit(x, H // 2 + 1, 4)
    assert torch.allclose(rows_all.mean(2), x.mean(2), atol=1e-5)   # the n = 0 column is untouched
    # the kept energy equals the folded-grid band fraction (DC kept on both sides)
    acc = core.BandEnergy(H, W, C, 3, 4, 'cpu', keep_dc=True)
    acc.update(x.reshape(B, H * W, C))
    E2 = acc.E2_grid().numpy()
    assert abs(float(y.double().square().sum() / x.double().square().sum()) - core.band_fraction(E2, 3, 4)) < 1e-6
    # white noise: the measured fraction matches the mode count
    acc = core.BandEnergy(H, W, C, 3, 4, 'cpu')
    for _ in range(200):
        acc.update(torch.randn(B, H * W, C))
    f, expect = core.band_fraction(acc.E2_grid().numpy(), 3, 4), white_noise_fraction(H, W, 3, 4)
    assert abs(f - expect) < 0.05 * expect, (f, expect)
    # the spectral Tucker map writes inside its band only (the band definition used here is its own)
    from attention.spectral_tucker_map import SpectralTuckerMap
    for backend in ('reference', 'gemm'):
        tucker = SpectralTuckerMap(C, L=2, kh=3, kw=4, r=4, backend=backend)
        out = tucker(x)
        assert torch.allclose(band_limit(out, 3, 4), out, atol=1e-5), backend
        assert not torch.allclose(band_limit(out, 2, 4), out, atol=1e-5), backend
        assert not torch.allclose(band_limit(out, 3, 3), out, atol=1e-5), backend
    print('self-test passed')


# ---------------------------------------------------------------- model and data
def load_state(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state = OrderedDict((k[7:] if k.startswith('module.') else k, v) for k, v in checkpoint['model_state'].items())
    return state, checkpoint


def validation_pairs(params, n_samples):
    """(input, target) of the validation samples train_mixers.py uses (get_data_loader,
    valid_indices): evenly spaced over the test years, the last step of each year skipped, one
    6 h step ahead, z-scored, last latitude row dropped. Read straight from the year files."""
    import h5py
    files = sorted(glob.glob(os.path.join(params.valid_data_path, '*.h5')))
    if not files:
        raise FileNotFoundError(f'no .h5 files in {params.valid_data_path}')
    handles = [h5py.File(f, 'r')['fields'] for f in files]
    per_year, rows = handles[0].shape[0], handles[0].shape[2] - 1
    dt = int(params.dt)
    candidates = [i for i in range(len(files) * per_year) if i % per_year < per_year - dt]
    if n_samples is None or n_samples >= len(candidates):
        picks = candidates
    else:
        picks = [candidates[i] for i in np.linspace(0, len(candidates) - 1, n_samples).round().astype(int)]
    cin, cout = np.array(params.in_channels), np.array(params.out_channels)
    means, stds = np.load(params.global_means_path)[0], np.load(params.global_stds_path)[0]
    inp = np.empty((len(picks), len(cin), rows, handles[0].shape[3]), dtype=np.float32)
    tar = np.empty((len(picks), len(cout), rows, handles[0].shape[3]), dtype=np.float32)
    for n, g in enumerate(picks):
        year, local = divmod(g, per_year)
        pair = handles[year][local:local + dt + 1:dt, :, 0:rows]        # (2, all channels, rows, W)
        inp[n] = (pair[0][cin] - means[cin]) / stds[cin]
        tar[n] = (pair[1][cout] - means[cout]) / stds[cout]
    return torch.from_numpy(inp), torch.from_numpy(tar), picks, files


def relative_l2(pred, target):
    """Per-sample training loss (utils/darcy_loss.py::LpLoss.rel, p = 2)."""
    return ((pred - target).flatten(1).norm(dim=1) / target.flatten(1).norm(dim=1)).tolist()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--yaml_config', default='config/mixer_comparison_merra2.yaml')
    parser.add_argument('--config', default='vanilla', help='named config of --yaml_config')
    parser.add_argument('--run_num', default='0')
    parser.add_argument('--n_samples', type=int, default=None, help='default: the config n_valid_samples')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--bands', nargs='*', default=['0'],
                        help="ablation bands: 'k' (square) or 'KHxKW'; 0 = mixer branch off")
    parser.add_argument('--kh', type=int, default=None, help='reported band; default: the tucker_spectral config')
    parser.add_argument('--kw', type=int, default=None)
    parser.add_argument('--dtype', choices=['auto', 'float32', 'bfloat16'], default='auto',
                        help='autocast dtype of the forward; auto = the config amp_dtype (as trained)')
    parser.add_argument('--out', default=None)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return

    from networks.mixernet import MixerNet
    from utils.mixer_params import Params, disable_transparent_hugepages

    disable_transparent_hugepages()
    t0 = time.time()
    device = torch.device('cuda', 0)
    yaml_path = os.path.abspath(args.yaml_config)
    params = Params(yaml_path, args.config)
    params['N_in_channels'] = len(params.in_channels)
    params['N_out_channels'] = len(params.out_channels)
    band_cfg = Params(yaml_path, 'tucker_spectral').mixer
    kh, kw = args.kh or band_cfg['kh'], args.kw or band_cfg['kw']
    out_dir = Path(args.out or ROOT / 'results' / 'band_diagnostic' / f'p{params.patch_size}' / args.config)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir = os.path.join(params.exp_dir, args.config, str(args.run_num))
    ckpt = os.path.join(run_dir, 'training_checkpoints', 'best_ckpt.tar')
    state, checkpoint = load_state(ckpt)
    assert checkpoint.get('mixer') == params.mixer, (checkpoint.get('mixer'), params.mixer)

    inp_all, tar_all, picks, files = validation_pairs(params, args.n_samples or params.n_valid_samples)
    n = len(picks)
    rows, cols = inp_all.shape[-2:]
    patch = int(params.patch_size)
    H, W, C = rows // patch, cols // patch, int(params.embed_dim)
    print(f'data: {n} validation samples from {len(files)} files in {params.valid_data_path} '
          f'({time.time() - t0:.0f}s); fields {rows} x {cols}, token grid {H} x {W}', flush=True)

    model = MixerNet(params, img_size=(rows, cols), embed_dim=params.embed_dim, depth=params.depth)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    blocks = list(model.blocks)
    J = len(blocks)
    kind = params.mixer['type']
    measure_energy = all(b.add_input for b in blocks)   # AFNO2D returns F(x) + x: no clean branch output
    print(f"model: {args.config} ({kind}), {J} blocks, C = {C}, checkpoint epoch {checkpoint.get('epoch')}, "
          f"best_valid_loss {checkpoint.get('best_valid_loss')}; reported band {kh} x {kw}", flush=True)

    amp = getattr(torch, params.amp_dtype if args.dtype == 'auto' else args.dtype)
    acc_out = [core.BandEnergy(H, W, C, kh, kw, device) for _ in range(J)] if measure_energy else []
    acc_in = [core.BandEnergy(H, W, C, kh, kw, device) for _ in range(J)] if measure_energy else []
    bands = [(str(b), *parse_band(b)) for b in args.bands]
    if not measure_energy and any(k_h > 0 and k_w > 0 for _, k_h, k_w in bands):
        raise SystemExit(f'{kind}: only the branch-off ablation (band 0) is defined for this mixer')
    losses = {'none': []}
    losses.update({label: [] for label, _, _ in bands})

    def energy_hook(j):
        def hook(module, inputs, output):
            acc_in[j].update(inputs[0].reshape(-1, H * W, C))
            acc_out[j].update(output.reshape(-1, H * W, C))
        return hook

    def limit_hook(block, k_h, k_w):
        if block.add_input:
            return lambda module, inputs, output: band_limit(output, k_h, k_w)
        return lambda module, inputs, output: inputs[0].to(output.dtype)      # afno, band 0: F(x) + x -> x

    def run(x, hooks):
        handles = [b.filter.register_forward_hook(h) for b, h in zip(blocks, hooks)]
        try:
            with torch.no_grad(), torch.autocast('cuda', dtype=amp, enabled=amp != torch.float32):
                return model(x).float()
        finally:
            for handle in handles:
                handle.remove()

    n_batches = math.ceil(n / args.batch)
    for b, s in enumerate(range(0, n, args.batch)):
        x, y = inp_all[s:s + args.batch].to(device), tar_all[s:s + args.batch].to(device)
        base_hooks = [energy_hook(j) for j in range(J)] if measure_energy else []
        losses['none'] += relative_l2(run(x, base_hooks), y)
        for label, k_h, k_w in bands:
            losses[label] += relative_l2(run(x, [limit_hook(blk, k_h, k_w) for blk in blocks]), y)
        if (b + 1) % max(1, n_batches // 8) == 0 or b == n_batches - 1:
            line = f"[{b + 1}/{n_batches}] {time.time() - t0:6.0f}s  loss(none) = {np.mean(losses['none']):.5f}"
            if measure_energy:
                f = [core.band_fraction(a.E2_grid().cpu().numpy(), kh, kw) for a in acc_out]
                line += f'  f_out({kh}x{kw}) per block: ' + ' '.join(f'{v:.3f}' for v in f)
            print(line, flush=True)

    # ---------------------------------------------------------------- energy summary
    base = np.array(losses['none'])
    summary = {'yaml_config': yaml_path, 'config': args.config, 'mixer': params.mixer, 'checkpoint': ckpt,
               'checkpoint_epoch': checkpoint.get('epoch'), 'logged_best_valid_loss': checkpoint.get('best_valid_loss'),
               'patch_size': patch, 'H': H, 'W': W, 'C': C, 'kh': kh, 'kw': kw, 'n_samples': n,
               'samples': [int(p) for p in picks], 'autocast': str(amp),
               'white_noise_f_khkw': white_noise_fraction(H, W, kh, kw),
               'white_noise_curve': [white_noise_fraction(H, W, k, k) for k in range(min(H // 2 + 1, W // 2 + 1) + 1)],
               'baseline_loss': float(base.mean()), 'blocks': []}
    if measure_energy:
        report_ks = sorted({(k_h, k_w) for _, k_h, k_w in bands if k_h > 0 and k_w > 0} | {(kh, kw)})
        for j in range(J):
            so, si = acc_out[j].state(), acc_in[j].state()
            co, ci = core.square_fraction_curve(so['E2']), core.square_fraction_curve(si['E2'])
            fc = so['Ec_band'] / np.maximum(so['Ec_tot'], 1e-30)
            summary['blocks'].append({
                'index': j, 'f_out_khkw': core.band_fraction(so['E2'], kh, kw),
                'f_in_khkw': core.band_fraction(si['E2'], kh, kw),
                'f_out_channel_median': float(np.median(fc)),
                'f_out_channel_q25': float(np.percentile(fc, 25)), 'f_out_channel_q75': float(np.percentile(fc, 75)),
                'f_out_at_band': {f'{a}x{b_}': core.band_fraction(so['E2'], a, b_) for a, b_ in report_ks},
                'f_in_at_band': {f'{a}x{b_}': core.band_fraction(si['E2'], a, b_) for a, b_ in report_ks},
                'k_out_90': k_for_fraction(co, 0.90), 'k_out_95': k_for_fraction(co, 0.95),
                'k_out_99': k_for_fraction(co, 0.99),
                'k_in_90': k_for_fraction(ci, 0.90), 'k_in_95': k_for_fraction(ci, 0.95),
                'k_in_99': k_for_fraction(ci, 0.99),
                'curve_out': co.tolist(), 'curve_in': ci.tolist()})
        pooled = np.sum([a.state()['E2'] for a in acc_out], axis=0)
        pooled_curve = core.square_fraction_curve(pooled)
        summary.update({'pooled_f_out_khkw': core.band_fraction(pooled, kh, kw),
                        'pooled_f_out_at_band': {f'{a}x{b_}': core.band_fraction(pooled, a, b_) for a, b_ in report_ks},
                        'pooled_curve_out': pooled_curve.tolist(),
                        'pooled_k_out_90_95_99': [k_for_fraction(pooled_curve, q) for q in (0.90, 0.95, 0.99)],
                        'white_noise_at_band': {f'{a}x{b_}': white_noise_fraction(H, W, a, b_) for a, b_ in report_ks}})
        np.savez_compressed(out_dir / 'band_energy.npz',
                            E2_out=np.stack([a.state()['E2'] for a in acc_out]),
                            E2_in=np.stack([a.state()['E2'] for a in acc_in]),
                            Ec_tot_out=np.stack([a.state()['Ec_tot'] for a in acc_out]),
                            Ec_band_out=np.stack([a.state()['Ec_band'] for a in acc_out]),
                            kh=kh, kw=kw, H=H, W=W, n_samples=n)
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=1))

    # ---------------------------------------------------------------- ablation summary
    sem = float(base.std(ddof=1) / math.sqrt(len(base))) if len(base) > 1 else 0.0
    ablation = {'bands': [{'label': label, 'kh': k_h, 'kw': k_w,
                           'modes_kept_fraction': modes_in_band(H, k_h) * modes_in_band(W, k_w) / (H * W)}
                          for label, k_h, k_w in bands],
                'n_samples': len(base), 'loss': 'relative L2 (training loss), one 6 h step',
                'baseline': {'mean': float(base.mean()), 'sem': sem},
                'per_band': {label: core.paired_stats(base, np.array(losses[label])) for label, _, _ in bands},
                'per_sample': losses}
    (out_dir / 'ablation.json').write_text(json.dumps(ablation, indent=1))

    # ---------------------------------------------------------------- console tables
    logged = checkpoint.get('best_valid_loss')
    print(f"\n{args.config}, patch {patch}: validation loss {base.mean():.5f} +- {sem:.5f} on {n} samples"
          + (f' (logged in training: {logged:.5f}, difference {base.mean() / logged - 1:+.3%})' if logged else ''))
    if measure_energy:
        print(f"\nband fraction of mixer OUTPUT energy, {kh} x {kw} band "
              f"(white noise = {summary['white_noise_f_khkw']:.4f}), DC excluded")
        print(f"{'block':>5} {'f_out':>7} {'f_in':>7} {'ch-med':>7} {'k90':>5} {'k95':>5} {'k99':>5}")
        beyond = f">{min(H // 2 + 1, W // 2 + 1)}"          # not reached inside the largest square
        for blk in summary['blocks']:
            ks = ' '.join(f"{beyond if blk[key] is None else blk[key]:>5}" for key in ('k_out_90', 'k_out_95', 'k_out_99'))
            print(f"{blk['index']:>5} {blk['f_out_khkw']:7.3f} {blk['f_in_khkw']:7.3f} {blk['f_out_channel_median']:7.3f} {ks}")
        print(f"pooled f_out = {summary['pooled_f_out_khkw']:.3f};  k90/95/99 = {summary['pooled_k_out_90_95_99']}")
        print('pooled f_out per band (white noise): '
              + '  '.join(f"{key} {val:.3f} ({summary['white_noise_at_band'][key]:.3f})"
                          for key, val in summary['pooled_f_out_at_band'].items()))
    print('\nvalidation loss with every mixer output limited to the band (0 = mixer branch off):')
    print(f"{'band':>8} {'modes':>7} {'loss':>9} {'sem':>8} {'delta':>9} {'d-sem':>8} {'rel':>9}")
    print(f"{'none':>8} {'100.0%':>7} {base.mean():9.5f} {sem:8.5f}")
    for entry in ablation['bands']:
        s = ablation['per_band'][entry['label']]
        print(f"{entry['label']:>8} {entry['modes_kept_fraction']:7.1%} {s['mean']:9.5f} {s['sem']:8.5f} "
              f"{s['delta_mean']:+9.5f} {s['delta_sem']:8.5f} {s['delta_rel']:+9.2%}")
    print(f'\nwrote {out_dir}/  ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()

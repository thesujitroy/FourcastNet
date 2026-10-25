#!/usr/bin/env python3
"""Synthetic cost benchmark of the FourCastNet mixers. No data or trained weights needed.

For each named config of config/mixer_comparison.yaml, seed and batch size, at two
scopes (one mixer on the token grid, `attention`; the whole MixerNet, `model`):
parameters, forward FLOPs (FlopCounterMode, FFTs added by formula), p50/p95
inference and training-step latency, images/s and peak GPU memory. Inputs are
already on the device, so data loading and DDP communication are excluded.

Rows use HelioFM's benchmark schema, so HelioFM's seed summary reads them as is:
    python scripts/benchmark_mixers.py --seeds 0 1 2 --output results/mixer_benchmark/run1
    python ../HelioFM-surya_1.1/scripts/summarize_benchmark_seeds.py results/mixer_benchmark/run1 \
        --output results/mixer_benchmark/run1
"""
import argparse
import csv
import gc
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent)]

import torch
from torch.utils.flop_counter import FlopCounterMode, sdpa_flop_count

from networks.mixernet import MixerNet
from networks.mixers import build_mixer
from utils.mixer_params import Params, parse_overrides

KINDS = ['afno', 'vanilla', 'tucker_free', 'tucker_spectral', 'tucker_adaptive']
FLOP_CONVENTION = ("multiply-add = 2 FLOPs (complex 8); FFT of n points 5 n log2 n, halved for "
                   "real-to-complex/complex-to-real; elementwise, norms, softmax not counted")


# FlopCounterMode has no FFT formulas and counts complex matmuls as real (as HelioFM's
# scripts/evaluate_attention_comparison.py).
def _fft_flops(n, batch, real):
    return int(batch * 5 * n * math.log2(n) / (2 if real else 1)) if n > 1 else 0


def fft_r2c_flops(x, dim, *args, out_val=None, **kwargs):
    n = math.prod(x.shape[d] for d in dim)
    return _fft_flops(n, x.numel() // n, real=True)


def fft_c2r_flops(x, dim, *args, out_val=None, **kwargs):
    n = math.prod(out_val.shape[d] for d in dim)
    return _fft_flops(n, out_val.numel() // n, real=True)


def fft_c2c_flops(x, dim, *args, out_val=None, **kwargs):
    n = math.prod(x.shape[d] for d in dim)
    return _fft_flops(n, x.numel() // n, real=False)


def matmul_flops(a, b, *args, out_val=None, **kwargs):
    return 2 * out_val.numel() * a.shape[-1] * (4 if a.is_complex() else 1)


def cpu_sdpa_flops(query, key, value, *args, out_val=None, **kwargs):
    # PyTorch counts the CUDA SDPA kernels but has no formula for the CPU one.
    return sdpa_flop_count(query.shape, key.shape, value.shape)


for _f in (fft_r2c_flops, fft_c2r_flops, fft_c2c_flops, matmul_flops, cpu_sdpa_flops):
    _f._get_raw = True
FLOP_FORMULAS = {torch.ops.aten._fft_r2c: fft_r2c_flops, torch.ops.aten._fft_c2r: fft_c2r_flops,
                 torch.ops.aten._fft_c2c: fft_c2c_flops, torch.ops.aten.mm: matmul_flops,
                 torch.ops.aten.bmm: matmul_flops}
if hasattr(torch.ops.aten, '_scaled_dot_product_flash_attention_for_cpu'):
    FLOP_FORMULAS[torch.ops.aten._scaled_dot_product_flash_attention_for_cpu] = cpu_sdpa_flops


def count_flops(model, inputs):
    counter = FlopCounterMode(display=False, custom_mapping=FLOP_FORMULAS)
    with counter, torch.no_grad():
        model(inputs)
    by_op = {str(op): int(v) for op, v in counter.get_flop_counts()["Global"].items()}
    return int(counter.get_total_flops()), dict(sorted(by_op.items(), key=lambda kv: -kv[1]))


def percentile(values, q):
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo = int(index)
    return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (index - lo)


def environment_info():
    return {"torch": str(torch.__version__), "cuda": torch.version.cuda, "python": platform.python_version(),
            "platform": platform.platform(),
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"}


def build(params, scope, img_size, batch_size, device):
    patch = params.patch_size
    if img_size[0] % patch or img_size[1] % patch:
        raise ValueError(f"image size {img_size} must be divisible by patch_size {patch}")
    grid = (img_size[0] // patch, img_size[1] // patch)
    if scope == 'attention':
        afno = {'num_blocks': params.num_blocks, 'sparsity_threshold': 0.01, 'hard_thresholding_fraction': 1.0}
        model, _ = build_mixer(params.mixer, params.embed_dim, grid, **afno)
        inputs = torch.randn(batch_size, *grid, params.embed_dim, device=device)
    else:
        model = MixerNet(params, img_size=tuple(img_size), embed_dim=params.embed_dim, depth=params.depth)
        inputs = torch.randn(batch_size, params.N_in_channels, *img_size, device=device)
    return model.to(device), inputs, grid


def benchmark_case(params, scope, img_size, batch_size, device, dtype, warmup, repeats):
    torch.manual_seed(params.seed)
    model, inputs, grid = build(params, scope, img_size, batch_size, device)
    filters = [model] if scope == 'attention' else [b.filter for b in model.blocks]
    result = {'tokens': grid[0] * grid[1],
              'parameters/total': sum(p.numel() for p in model.parameters()),
              'parameters/trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'parameters/attention': sum(p.numel() for f in filters for p in f.parameters()),
              'storage/parameter_bytes': sum(p.numel() * p.element_size() for p in model.parameters())}
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
        flops, by_op = count_flops(model.eval(), inputs)
    result.update({'flops/forward': flops, 'flops/forward_per_sample': flops / batch_size,
                   'flops/forward_by_op': by_op, 'flops/convention': FLOP_CONVENTION,
                   'precision/parameters': 'float32', 'precision/autocast': str(dtype)})
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == 'cuda' else (lambda: None)

    for training in (False, True):
        model.train(training)
        label = 'train' if training else 'inference'
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) if training else None
        scaler = torch.amp.GradScaler(device.type, enabled=training and dtype == torch.float16)

        def step():
            if training:
                optimizer.zero_grad(set_to_none=True)
            with (nullcontext() if training else torch.no_grad()), \
                    torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
                output = model(inputs)
                loss = output.float().square().mean() if training else None
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            if not torch.isfinite(output).all():
                raise FloatingPointError('non-finite output during benchmark')

        for _ in range(warmup):
            step()
        sync()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        times = []
        for _ in range(repeats):
            sync()
            start = time.perf_counter()
            step()
            sync()
            times.append((time.perf_counter() - start) * 1000)
        result[f'{label}/latency_ms_p50'] = statistics.median(times)
        result[f'{label}/latency_ms_p95'] = percentile(times, .95)
        result[f'{label}/images_per_second'] = batch_size * 1000 / statistics.mean(times)
        result[f'{label}/peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
        result[f'{label}/peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None
        if training:
            result['storage/gradient_bytes'] = sum(p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None)
            result['storage/optimizer_bytes'] = sum(v.numel() * v.element_size() for state in optimizer.state.values()
                                                    for v in state.values() if isinstance(v, torch.Tensor))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--yaml-config', type=Path, default=ROOT / 'config' / 'mixer_comparison.yaml')
    parser.add_argument('--configs', nargs='+', default=KINDS, help='named configs in --yaml-config')
    parser.add_argument('--img-size', nargs=2, type=int, default=[720, 1440], metavar=('H', 'W'))
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1])
    parser.add_argument('--scope', choices=['attention', 'model', 'both'], default='both')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--dtype', choices=['float32', 'bfloat16', 'float16'],
                        help='autocast dtype; default: the config amp_dtype on cuda, float32 on cpu')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--seeds', nargs='+', type=int, help='one row per seed (default: the config seed)')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'mixer_benchmark')
    parser.add_argument('--override', nargs='*', default=[], metavar='KEY=VALUE')
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1 or min(args.batch_sizes) < 1:
        parser.error('warmup, repeats and batch sizes must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; --device cpu gives smoke measurements only')
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    overrides = parse_overrides(args.override)
    image_size = f'{args.img_size[0]}x{args.img_size[1]}'

    records = []
    for name in args.configs:
        params = Params(args.yaml_config, name, overrides)
        params['N_in_channels'] = len(params.in_channels) + (1 if params.orography else 0)
        params['N_out_channels'] = len(params.out_channels)
        dtype = getattr(torch, args.dtype or ('float32' if args.device == 'cpu' else params.amp_dtype))
        resolved = {k: v for k, v in params.params.items()}
        for seed in args.seeds or [params.seed]:
            params['seed'] = seed
            for batch_size in args.batch_sizes:
                for scope in ['attention', 'model'] if args.scope == 'both' else [args.scope]:
                    row = {'attention': params.mixer['type'].replace('_', '-'), 'config': name, 'scope': scope,
                           'image_size': image_size, 'batch_size': batch_size, 'dtype': str(dtype), 'seed': seed,
                           'config_path': str(args.yaml_config.resolve()), 'warmup': args.warmup,
                           'repeats': args.repeats, 'cpu_threads': args.threads,
                           'environment': environment_info(), 'resolved_config': {**resolved, 'seed': seed},
                           'config_hash': hashlib.sha256(json.dumps({**resolved, 'seed': seed}, sort_keys=True,
                                                                    default=str).encode()).hexdigest()[:16]}
                    try:
                        row.update(benchmark_case(params, scope, args.img_size, batch_size, device, dtype,
                                                  args.warmup, args.repeats))
                        row['status'] = 'ok'
                    except torch.OutOfMemoryError as exc:
                        row.update(status='oom', error=str(exc))
                    except Exception as exc:
                        row.update(status='error', error=f'{type(exc).__name__}: {exc}')
                    records.append(row)
                    with (args.output / 'measurements.jsonl').open('a') as stream:
                        stream.write(json.dumps(row, default=str) + '\n')
                    print(f"{row['attention']} {scope} image={image_size} batch={batch_size} seed={seed}: {row['status']}",
                          flush=True)
                    gc.collect()
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

    keys = sorted(set().union(*(r.keys() for r in records)) - {'environment', 'resolved_config', 'flops/forward_by_op'})
    with (args.output / 'measurements.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    if any(row['status'] == 'error' for row in records):
        raise SystemExit('Some benchmark cases failed; see measurements.jsonl')


if __name__ == '__main__':
    main()

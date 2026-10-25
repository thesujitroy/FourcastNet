"""CPU tests for the FourCastNet mixer comparison (run from FourCastNet/: python -m pytest)."""
from copy import deepcopy
from types import SimpleNamespace

import sys
import types

import numpy
import pytest
import torch

# Upstream networks/afnonet.py has a dead `from numpy.lib.arraypad import pad`, a module
# numpy 2 removed. The upstream file stays untouched; this test-only shim lets it import
# so MixerNet can be checked against it.
if 'numpy.lib.arraypad' not in sys.modules:
    _shim = types.ModuleType('numpy.lib.arraypad')
    _shim.pad = numpy.pad
    sys.modules['numpy.lib.arraypad'] = _shim

from networks.afnonet import AFNONet  # noqa: E402
from networks.mixernet import MixerNet  # noqa: E402

# 32 x 64 pixels at patch 8 -> a 4 x 8 token grid, small enough for CPU.
SMALL = dict(img_size=(32, 64), embed_dim=32, depth=2)
MIXERS = {
    'afno': {'type': 'afno'},
    'vanilla': {'type': 'vanilla', 'num_heads': 4},
    'tucker_free': {'type': 'tucker_free', 'L': 2, 'p': 2, 'q': 3, 'r': 4},
    'tucker_spectral': {'type': 'tucker_spectral', 'L': 2, 'kh': 2, 'kw': 3, 'r': 4},
    'tucker_adaptive': {'type': 'tucker_adaptive', 'L': 2, 'kh': 2, 'kw': 3, 'r': 4,
                        'ctx_hidden': 8, 'lat_basis': 4},
}


def params(mixer=None):
    return SimpleNamespace(patch_size=8, N_in_channels=3, N_out_channels=3, num_blocks=8,
                           mixer=deepcopy(mixer))


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('mixer', [None, {'type': 'afno'}])
def test_afno_mixernet_is_afnonet(mixer):
    ref = AFNONet(params(), **SMALL).eval()
    new = MixerNet(params(mixer), **SMALL).eval()
    assert list(ref.state_dict()) == list(new.state_dict())
    new.load_state_dict(ref.state_dict(), strict=True)   # released FourCastNet checkpoints load
    x = torch.randn(2, 3, 32, 64)
    assert torch.equal(ref(x), new(x))


def test_one_seed_gives_identical_non_mixer_weights_for_every_mixer():
    states = {}
    for kind, cfg in MIXERS.items():
        torch.manual_seed(0)
        states[kind] = {k: v for k, v in MixerNet(params(cfg), **SMALL).state_dict().items() if '.filter.' not in k}
    for kind, state in states.items():
        assert state.keys() == states['afno'].keys()
        for key, value in states['afno'].items():
            assert torch.equal(state[key], value), (kind, key)


def test_initialization_follows_the_seed_and_blocks_differ():
    def build(seed):
        torch.manual_seed(seed)
        return MixerNet(params(MIXERS['vanilla']), **SMALL).state_dict()
    a, b, c = build(1), build(1), build(2)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not torch.equal(a['blocks.0.filter.qkv.weight'], c['blocks.0.filter.qkv.weight'])
    assert not torch.equal(a['blocks.0.mlp.fc1.weight'], c['blocks.0.mlp.fc1.weight'])
    assert not torch.equal(a['blocks.0.filter.qkv.weight'], a['blocks.1.filter.qkv.weight'])


def test_adaptive_mixer_ignores_bf16_autocast():
    # Under bf16 the latitude profile 1 + sum_j c_j cos(...) would round to steps of 2^-7.
    from networks.mixers import build_mixer
    torch.manual_seed(0)
    mixer, _ = build_mixer({'type': 'tucker_adaptive', 'L': 2, 'kh': 4, 'kw': 3, 'r': 4, 'ctx_hidden': 8,
                            'lat_basis': 4}, 32, (90, 8))
    with torch.no_grad():
        mixer.lat_in.fill_(1e-3)
        mixer.lat_out.fill_(-1e-3)
        mixer.ctx[-1].weight.normal_(std=0.1)
    x = torch.randn(2, 90, 8, 32)
    reference = mixer(x)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        mixed = mixer(x)
    assert torch.equal(mixed, reference)


@pytest.mark.parametrize('kind', MIXERS)
def test_every_mixer_trains(kind):
    torch.manual_seed(0)
    model = MixerNet(params(MIXERS[kind]), **SMALL)
    x = torch.randn(2, 3, 32, 64)
    y = model(x)
    assert y.shape == (2, 3, 32, 64)
    y.square().mean().backward()
    for name, p in model.named_parameters():
        if name.startswith('norm.'):
            continue  # AFNONet's final norm is defined but never used in forward
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    before = model(x).detach()
    torch.optim.Adam(model.parameters(), lr=1e-2).step()
    assert not torch.equal(before, model(x).detach())


@pytest.mark.parametrize('kind', MIXERS)
def test_state_dict_round_trip(kind):
    model = MixerNet(params(MIXERS[kind]), **SMALL).eval()
    restored = MixerNet(params(MIXERS[kind]), **SMALL).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    x = torch.randn(1, 3, 32, 64)
    torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)


def test_each_block_uses_the_configured_mixer():
    from attention.spectral_tucker_map import SpectralTuckerMap
    model = MixerNet(params(MIXERS['tucker_spectral']), **SMALL)
    assert all(isinstance(block.filter, SpectralTuckerMap) for block in model.blocks)


@pytest.mark.parametrize('inner_skip', [True, False])
def test_inner_skip_adds_the_mixer_input(inner_skip):
    cfg = {**MIXERS['tucker_spectral'], 'zero_init': True, 'inner_skip': inner_skip}
    block = MixerNet(params(cfg), **SMALL).blocks[0]
    n = torch.randn(2, 4, 8, 32)
    # zero_init zeroes B_C, so the Tucker map itself outputs exactly 0
    expected = n if inner_skip else torch.zeros_like(n)
    torch.testing.assert_close(block.mix(n), expected, rtol=0, atol=0)


def test_inner_skip_is_on_by_default():
    block = MixerNet(params({**MIXERS['tucker_free'], 'zero_init': True}), **SMALL).blocks[0]
    n = torch.randn(1, 4, 8, 32)
    torch.testing.assert_close(block.mix(n), n, rtol=0, atol=0)


def test_adaptive_gate_survives_global_init():
    model = MixerNet(params(MIXERS['tucker_adaptive']), **SMALL)
    for block in model.blocks:
        assert torch.count_nonzero(block.filter.ctx[-1].weight) == 0
        assert torch.count_nonzero(block.filter.ctx[-1].bias) == 0
        assert torch.count_nonzero(block.filter.lat_in) == 0
        assert torch.count_nonzero(block.filter.lat_out) == 0


@pytest.mark.parametrize('cfg', [
    {'type': 'unknown'},
    {'type': 'ls'},                                    # square-grid only, not offered
    {'type': 'tucker_spectral', 'L': 2, 'r': 4, 'foo': 1},
    {'type': 'tucker_spectral', 'L': 2, 'r': 4, 'kh': 3, 'kw': 3},   # 4-row grid allows kh <= 2
    {'type': 'tucker_adaptive', 'L': 2, 'r': 4, 'kh': 2, 'kw': 5},   # 8-col grid allows kw <= 4
    {'type': 'tucker_free', 'r': 64},                  # r > embed_dim
    {'type': 'tucker_free', 'p': 5},                   # p > grid rows
    {'type': 'vanilla', 'num_heads': 5},               # 32 % 5 != 0
    {'type': 'afno', 'inner_skip': False},             # AFNO2D has the skip built in
    {'type': 'afno', 'num_blocks': 3},                 # 32 % 3 != 0
])
def test_invalid_mixer_options_fail(cfg):
    with pytest.raises(ValueError):
        MixerNet(params(cfg), **SMALL)


# ---------------------------------------------------------------- configs
from pathlib import Path  # noqa: E402

from utils.mixer_params import Params, parse_overrides  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config' / 'mixer_comparison.yaml'
COMPARED = ['afno', 'vanilla', 'tucker_free', 'tucker_spectral', 'tucker_adaptive']


def test_comparison_configs_differ_only_in_the_mixer():
    loaded = {kind: Params(CONFIG, kind) for kind in COMPARED}
    base = loaded['afno'].params
    for kind, p in loaded.items():
        assert p.mixer['type'] == kind
        differing = {k for k in set(base) | set(p.params) if base.get(k) != p.params.get(k)}
        assert differing <= {'mixer'}, (kind, differing)


def test_config_values_have_python_types():
    p = Params(CONFIG, 'afno')
    assert isinstance(p.lr, float) and p.lr == 5e-4      # PyYAML reads "5E-4" as a string
    assert p.orography is False and p.crop_size_x is None  # 'None' strings become None, like YParams
    assert len(p.in_channels) == len(p.channel_names) == 20
    assert p.channel_names[14] == 'z500' and p.channel_names[5] == 't850'


def test_params_behave_like_yparams():
    p = Params(CONFIG, 'tucker_spectral')
    assert 'lr' in p and 'precip' not in p
    p['N_in_channels'] = 20
    assert p.N_in_channels == 20 and p['N_in_channels'] == 20
    p.update_params({'batch_size': 4})
    assert p.batch_size == 4 and p.params['batch_size'] == 4


def test_overrides_are_parsed_as_yaml():
    assert parse_overrides(['max_epochs=1', 'lr=1e-3', 'exp_dir=/tmp/x', 'max_iters_per_epoch=None',
                            'mixer={type: afno}']) == {
        'max_epochs': 1, 'lr': 1e-3, 'exp_dir': '/tmp/x', 'max_iters_per_epoch': None,
        'mixer': {'type': 'afno'}}
    with pytest.raises(ValueError):
        parse_overrides(['no_equals_sign'])


@pytest.mark.parametrize('kind', COMPARED)
def test_full_size_comparison_models_build(kind):
    p = Params(CONFIG, kind)
    p['N_in_channels'] = p['N_out_channels'] = len(p.in_channels)
    model = MixerNet(p, img_size=(720, 1440), embed_dim=p.embed_dim, depth=p.depth)
    assert (model.h, model.w) == (90, 180)
    assert all(type(b.filter).__name__ == type(model.blocks[0].filter).__name__ for b in model.blocks)
    assert len(model.blocks) == 12


# ---------------------------------------------------------------- end to end on synthetic ERA5
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402

# 721 x 32 pixels (the loader drops the last row) at patch 8 -> a 90 x 4 token grid, embed_dim 16
GRID_MIXERS = {
    'afno': '{type: afno}',
    'vanilla': '{type: vanilla, num_heads: 4}',
    'tucker_free': '{type: tucker_free, L: 2, p: 4, q: 2, r: 4}',
    'tucker_spectral': '{type: tucker_spectral, L: 2, kh: 4, kw: 2, r: 4}',
    'tucker_adaptive': '{type: tucker_adaptive, L: 2, kh: 4, kw: 2, r: 4, ctx_hidden: 8, lat_basis: 4}',
}


@pytest.fixture(scope='session')
def era5_like(tmp_path_factory):
    """FourCastNet's preprocessed layout: year files with `fields` (T, C, 721, W) plus stats."""
    root = tmp_path_factory.mktemp('era5')
    rng = np.random.default_rng(0)
    lat = np.cos(np.linspace(0, np.pi, 721))[None, None, :, None]
    offset = 10.0 * np.arange(3)[None, :, None, None]

    def year(path, steps):
        path.parent.mkdir(exist_ok=True)
        fields = lat + offset + 0.1 * rng.standard_normal((steps, 3, 721, 32))
        with h5py.File(path, 'w') as f:
            f['fields'] = fields.astype(np.float32)

    year(root / 'train' / '1990.h5', 8)
    year(root / 'train' / '1991.h5', 8)
    year(root / 'test' / '2016.h5', 8)
    year(root / 'out_of_sample' / '2018.h5', 12)
    (root / 'stats').mkdir()
    np.save(root / 'stats' / 'global_means.npy', offset.astype(np.float32))
    np.save(root / 'stats' / 'global_stds.npy', np.full((1, 3, 1, 1), 0.5, np.float32))
    np.save(root / 'stats' / 'time_means.npy', np.broadcast_to(lat + offset, (1, 3, 721, 32)).astype(np.float32))
    return root


def overrides(data, exp_dir, kind, **extra):
    values = {'train_data_path': data / 'train', 'valid_data_path': data / 'test',
              'inf_data_path': data / 'out_of_sample', 'exp_dir': exp_dir,
              'global_means_path': data / 'stats' / 'global_means.npy',
              'global_stds_path': data / 'stats' / 'global_stds.npy',
              'time_means_path': data / 'stats' / 'time_means.npy',
              'in_channels': '[0, 1, 2]', 'out_channels': '[0, 1, 2]', 'channel_names': '[a, b, c]',
              'embed_dim': 16, 'depth': 2, 'batch_size': 2, 'max_epochs': 1, 'num_data_workers': 0,
              'log_to_wandb': 'false', 'n_valid_samples': 4, 'mixer': GRID_MIXERS[kind], **extra}
    return ['--override', *[f'{k}={v}' for k, v in values.items()]]


def run(*args):
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(ROOT), str(ROOT.parent)]),
           'OMP_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': '', 'WANDB_MODE': 'disabled'}
    result = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    return result


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.mark.parametrize('kind', GRID_MIXERS)
def test_training_runs_end_to_end(kind, era5_like, tmp_path):
    run('train_mixers.py', '--config', kind, '--run_num', 0, *overrides(era5_like, tmp_path, kind))
    run_dir = tmp_path / kind / '0'
    model = jsonl(run_dir / 'model.jsonl')[-1]
    assert model['mixer'] == kind and model['parameters/mixer'] > 0
    epoch = jsonl(run_dir / 'epochs.jsonl')[-1]
    assert epoch['epoch'] == 1 and epoch['steps'] == 8 and epoch['step'] == 8  # 16 samples, batch 2
    assert np.isfinite(epoch['train_epoch_loss']) and epoch['performance/train_epoch_seconds'] > 0
    valid = jsonl(run_dir / 'validation.jsonl')[-1]
    assert valid['n_valid_samples'] == 4 and np.isfinite(valid['valid_loss'])
    for name in 'abc':
        assert valid[f'valid_rmse/{name}'] > 0
        assert -1 <= valid[f'valid_acc/{name}'] <= 1
    assert (run_dir / 'training_checkpoints' / 'best_ckpt.tar').is_file()


def test_training_resumes_and_caps_steps(era5_like, tmp_path):
    common = ['train_mixers.py', '--config', 'tucker_adaptive', '--run_num', 0]
    run(*common, *overrides(era5_like, tmp_path, 'tucker_adaptive', max_iters_per_epoch=3))
    run(*common, *overrides(era5_like, tmp_path, 'tucker_adaptive', max_iters_per_epoch=3, max_epochs=2))
    run_dir = tmp_path / 'tucker_adaptive' / '0'
    epochs = jsonl(run_dir / 'epochs.jsonl')
    assert [(e['epoch'], e['steps'], e['step']) for e in epochs] == [(1, 3, 3), (2, 3, 6)]
    # GPU hours carry over from the first job
    assert epochs[1]['compute/cumulative_gpu_hours'] >= (
        epochs[0]['compute/cumulative_gpu_hours'] + epochs[1]['performance/train_epoch_seconds'] / 3600)
    # the best checkpoint is the best epoch over both jobs, not whatever ran first after the resume.
    # Ties go to the later epoch (`<=`, as train.py). Here they happen: job 1's 1-epoch cosine
    # schedule ends at lr 0, which job 2 inherits, so epoch 2 does not change the weights.
    valid = jsonl(run_dir / 'validation.jsonl')
    lowest = min(v['valid_loss'] for v in valid)
    best_epoch = max(v['epoch'] for v in valid if v['valid_loss'] == lowest)
    ckpt = torch.load(run_dir / 'training_checkpoints' / 'ckpt.tar', weights_only=False)
    assert ckpt['best_valid_loss'] == lowest
    assert torch.load(run_dir / 'training_checkpoints' / 'best_ckpt.tar', weights_only=False)['epoch'] == best_epoch
    assert not list((run_dir / 'training_checkpoints').glob('*.tmp'))


@pytest.mark.parametrize('script', ['train_mixers.py', 'inference/inference_mixers.py'])
def test_residual_field_target_is_refused(script, era5_like, tmp_path):
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(ROOT), str(ROOT.parent)]),
           'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1'}
    result = subprocess.run([sys.executable, script, '--config', 'afno',
                             *overrides(era5_like, tmp_path, 'afno', target='residual_field')],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode != 0 and 'target residual_field is not supported' in result.stderr


def test_inference_scores_a_trained_run_against_persistence(era5_like, tmp_path):
    kind = 'tucker_spectral'
    run('train_mixers.py', '--config', kind, '--run_num', 0, *overrides(era5_like, tmp_path, kind))
    # 12 steps in 2018.h5, 3-step rollouts, one IC every 4 steps -> ICs 0, 4, 8
    run('inference/inference_mixers.py', '--config', kind, '--run_num', 0, '--ic_spacing', 4,
        *overrides(era5_like, tmp_path, kind, prediction_length=3))
    run_dir = tmp_path / kind / '0'
    summary = json.loads((run_dir / 'inference_summary.json').read_text())
    assert summary['mixer'] == kind and summary['initial_conditions'] == [0, 4, 8]
    assert summary['lead_hours'] == [0, 6, 12]
    for name in 'abc':
        assert summary['rmse'][name][0] == 0 and summary['acc'][name][0] == pytest.approx(1)
        assert summary['rmse'][name][1] > 0 and summary['rmse_persistence'][name][1] > 0
    with h5py.File(run_dir / 'autoregressive_predictions.h5', 'r') as f:
        assert f['rmse'].shape == f['acc'].shape == f['rmse_persistence'].shape == (3, 3, 3)


def test_inference_refuses_a_checkpoint_from_another_mixer(era5_like, tmp_path):
    run('train_mixers.py', '--config', 'afno', '--run_num', 0, *overrides(era5_like, tmp_path, 'afno'))
    weights = tmp_path / 'afno' / '0' / 'training_checkpoints' / 'best_ckpt.tar'
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(ROOT), str(ROOT.parent)]),
           'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1'}
    result = subprocess.run([sys.executable, 'inference/inference_mixers.py', '--config', 'tucker_spectral',
                             '--weights', str(weights), '--override_dir', str(tmp_path / 'out'),
                             *overrides(era5_like, tmp_path, 'tucker_spectral', prediction_length=3)],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode != 0 and 'trained with mixer' in result.stderr


def test_benchmark_rows_feed_the_helio_seed_summary(tmp_path):
    out = tmp_path / 'bench'
    run('scripts/benchmark_mixers.py', '--configs', 'afno', 'vanilla', '--device', 'cpu', '--img-size', 32, 64,
        '--warmup', 1, '--repeats', 2, '--seeds', 0, 1, '--output', out,
        '--override', 'embed_dim=24', 'depth=2', 'in_channels=[0, 1, 2]', 'out_channels=[0, 1, 2]')
    rows = jsonl(out / 'measurements.jsonl')
    assert all(r['status'] == 'ok' for r in rows), [r.get('error') for r in rows]
    assert sorted((r['attention'], r['scope'], r['seed']) for r in rows) == sorted(
        (a, s, seed) for a in ('afno', 'vanilla') for s in ('attention', 'model') for seed in (0, 1))
    for r in rows:
        assert r['flops/forward'] > 0 and r['train/latency_ms_p50'] > 0 and r['inference/latency_ms_p50'] > 0
        assert r['tokens'] == 32 and r['image_size'] == '32x64'
        if r['scope'] == 'model':
            assert r['parameters/total'] > r['parameters/attention'] > 0
        if (r['attention'], r['scope']) == ('vanilla', 'attention'):
            N, C = 32, 24   # qkv + proj matmuls 8 N C^2, plus QK^T and AV 4 N^2 C (SDPA must be counted on CPU too)
            assert r['flops/forward'] == 8 * N * C * C + 4 * N * N * C
    # the summary groups rows by the folder of config_path, i.e. config/ for every row here
    summary = run(ROOT.parent / 'HelioFM-surya_1.1' / 'scripts' / 'summarize_benchmark_seeds.py', out)
    assert '| config | vanilla | model | 2 |' in summary.stdout and '| config | afno | model | 2 |' in summary.stdout


def test_rollout_table_compares_runs_and_persistence(tmp_path):
    def summary(kind, rmse, acc):
        return {'mixer': kind, 'config': kind, 'lead_hours': [0, 6, 12], 'initial_conditions': [0, 36],
                'rmse': {'z500': [0, rmse, 2 * rmse]}, 'acc': {'z500': [1, acc, acc - .1]},
                'rmse_persistence': {'z500': [0, 50, 90]}, 'acc_persistence': {'z500': [1, .9, .8]}}
    for kind, rmse, acc in [('afno', 30, .99), ('tucker_spectral', 25, .995)]:
        (tmp_path / kind / '0').mkdir(parents=True)
        (tmp_path / kind / '0' / 'inference_summary.json').write_text(json.dumps(summary(kind, rmse, acc)))
    out = run('scripts/summarize_rollouts.py', tmp_path, '--channels', 'z500', '--leads', 6, 12,
              '--output', tmp_path / 'rollouts.md').stdout
    assert '| z500 | 6 h | 50 | 30 | **25** |' in out          # RMSE, best per row in bold
    assert '| z500 | 12 h | 0.8 | 0.89 | **0.895** |' in out    # ACC
    assert out.index('afno') < out.index('tucker_spectral')
    assert (tmp_path / 'rollouts.md').read_text().strip() == out.strip()


def test_training_under_torchrun_shards_data_and_validation(era5_like, tmp_path):
    """The NAS launch path: torchrun, 2 ranks (gloo on CPU), bf16 autocast, global batch 2 -> 1 per rank."""
    run('-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2', 'train_mixers.py',
        '--config', 'tucker_adaptive', '--run_num', 0, '--enable_amp', *overrides(era5_like, tmp_path, 'tucker_adaptive', n_valid_samples=5))
    run_dir = tmp_path / 'tucker_adaptive' / '0'
    epoch = jsonl(run_dir / 'epochs.jsonl')
    assert len(epoch) == 1 and epoch[0]['steps'] == 8      # 16 samples over 2 ranks, 1 per rank per step
    assert jsonl(run_dir / 'model.jsonl')[-1]['world_size'] == 2
    assert jsonl(run_dir / 'validation.jsonl')[-1]['n_valid_samples'] == 5   # uneven shards, all counted once

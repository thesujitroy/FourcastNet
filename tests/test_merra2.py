"""CPU tests for the MERRA-2 path of the mixer comparison (run from FourCastNet/: python -m pytest).

    * data_process/merra2_to_fcn.py on synthetic MERRA-2-like daily files
    * the loader and trainer on a grid with an odd number of rows and no climatology
    * config/mixer_comparison_merra2.yaml: the configs differ only in the mixer, and every
      model builds and runs at the real 360 x 576 size
"""
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from data_process import merra2_to_fcn as m2f
from networks.mixernet import MixerNet
from utils.mixer_params import Params

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config' / 'mixer_comparison_merra2.yaml'
COMPARED = ['afno', 'tucker_free', 'tucker_spectral', 'tucker_spectral_dct', 'tucker_adaptive']


def run(*args):
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(ROOT), str(ROOT.parent)]),
           'OMP_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': '', 'WANDB_MODE': 'disabled'}
    result = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    return result


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------- the converter
# Synthetic files in the Prithvi layout on a 5 x 8 grid. Every value is an integer that names
# its grid cell, its (variable, level) and its 3-hourly step of the year, exactly in float32:
#     value = cell * 120000 + code * 3000 + step
NLAT, NLON = 5, 8
LEVELS = [34.0, 39.0, 51.0, 63.0, 72.0]
SURFACE_VARS = ['EFLUX', 'PS', 'SLP', 'T2M', 'TQV', 'U10M', 'V10M']
VERTICAL_VARS = ['H', 'OMEGA', 'QV', 'T', 'U', 'V']
CODES = {(v, None): i for i, v in enumerate(SURFACE_VARS)}
CODES.update({(v, lev): len(SURFACE_VARS) + i * len(LEVELS) + k
              for i, v in enumerate(VERTICAL_VARS) for k, lev in enumerate(LEVELS)})
CELL = (np.arange(NLAT * NLON).reshape(NLAT, NLON) * 120000).astype(np.float64)
YEAR = 1980   # a leap year in the training split


def write_day(surface_dir, vertical_dir, day):
    import pandas as pd
    import xarray as xr
    steps = (day - dt.date(day.year, 1, 1)).days * 8 + np.arange(8)
    coords = {'time': pd.date_range(str(day), periods=8, freq='3h'),
              'lat': np.linspace(-90, 90, NLAT), 'lon': np.linspace(-180, 135, NLON)}
    surface = {v: (('time', 'lat', 'lon'), (CELL[None] + CODES[v, None] * 3000 + steps[:, None, None]).astype(np.float32))
               for v in SURFACE_VARS}
    xr.Dataset(surface, coords=coords).to_netcdf(surface_dir / f'MERRA2_sfc_{day:%Y%m%d}.nc', engine='h5netcdf')
    codes = {v: np.array([CODES[v, lev] for lev in LEVELS]) for v in VERTICAL_VARS}
    vertical = {v: (('time', 'lev', 'lat', 'lon'),
                    (CELL[None, None] + codes[v][None, :, None, None] * 3000 + steps[:, None, None, None]).astype(np.float32))
                for v in VERTICAL_VARS}
    xr.Dataset(vertical, coords={**coords, 'lev': LEVELS}).to_netcdf(
        vertical_dir / f'MERRA_pres_{day:%Y%m%d}.nc', engine='h5netcdf')


@pytest.fixture(scope='session')
def merra2_like(tmp_path_factory):
    root = tmp_path_factory.mktemp('merra2')
    surface, vertical = root / 'sfc', root / 'pres'
    surface.mkdir(), vertical.mkdir()
    day = dt.date(YEAR, 1, 1)
    while day.year == YEAR:      # all 366 days
        write_day(surface, vertical, day)
        day += dt.timedelta(days=1)
    return surface, vertical


def expected_fields():
    """(1460, 20, NLAT, NLON): step t is 6-hourly step t of the year = 3-hourly step 2t."""
    codes = np.array([CODES[variable, level] for _, _, variable, level in m2f.CHANNELS])
    steps = 2 * np.arange(m2f.STEPS_PER_YEAR)
    return CELL[None, None] + codes[None, :, None, None] * 3000 + steps[:, None, None, None]


def convert(out, surface, vertical, *extra):
    return run('data_process/merra2_to_fcn.py', '--out', out, '--surface', surface, '--vertical', vertical,
               '--years', YEAR, '--jobs', 1, *extra)


@pytest.fixture(scope='session')
def converted(merra2_like, tmp_path_factory):
    out = tmp_path_factory.mktemp('fcn_merra2')
    convert(out, *merra2_like)
    return out


def test_converter_writes_fourcastnets_layout(converted):
    path = converted / 'train' / f'{YEAR}.h5'
    assert path.is_file() and not list(converted.rglob('*.partial'))
    with h5py.File(path, 'r') as f:
        fields = f['fields'][:]
        assert fields.dtype == np.float32 and fields.shape == (1460, 20, NLAT, NLON)
        assert f['fields'].chunks is None                       # contiguous
        np.testing.assert_array_equal(fields, expected_fields())  # channel order, levels, hours, grid
        assert f['time'][0] == 1980010100 and f['time'][1] == 1980010106
        assert f['time'][-1] == 1980123018                      # 31 December of the leap year is dropped
        assert json.loads(f.attrs['channel_names']) == m2f.CHANNEL_NAMES
        assert f['lat'][0] == -90 and f['lat'][-1] == 90


def test_converter_statistics_are_the_training_means_and_stds(converted):
    with h5py.File(converted / 'train' / f'{YEAR}.h5', 'r') as f:
        fields = f['fields'][:].astype(np.float64)
    means = np.load(converted / 'stats' / 'global_means.npy')
    stds = np.load(converted / 'stats' / 'global_stds.npy')
    assert means.shape == stds.shape == (1, 20, 1, 1) and means.dtype == np.float32
    np.testing.assert_allclose(means[0, :, 0, 0], fields.mean(axis=(0, 2, 3)), rtol=1e-6)
    np.testing.assert_allclose(stds[0, :, 0, 0], fields.std(axis=(0, 2, 3)), rtol=1e-5)
    summary = json.loads((converted / 'stats' / 'channels.json').read_text())
    assert [c['name'] for c in summary['channels']] == m2f.CHANNEL_NAMES and summary['years'] == [str(YEAR)]


def test_converter_resumes_a_partial_year(merra2_like, converted, tmp_path):
    final, partial = m2f.year_paths(tmp_path, YEAR)
    m2f.convert_month(YEAR, 2, partial, *merra2_like)        # as if a job died after one month
    assert partial.is_file() and not final.exists()
    log = convert(tmp_path, *merra2_like).stdout
    assert f'{YEAR}: ok' in log and final.is_file() and not partial.exists()
    assert '1980-02: already done' in (tmp_path / 'logs' / f'{YEAR}.log').read_text()
    with h5py.File(final, 'r') as a, h5py.File(converted / 'train' / f'{YEAR}.h5', 'r') as b:
        np.testing.assert_array_equal(a['fields'][:], b['fields'][:])
    assert f'{YEAR}: ok (present)' in convert(tmp_path, *merra2_like).stdout   # nothing left to do


def test_converter_refuses_bad_values(merra2_like, tmp_path):
    import shutil
    surface, vertical = tmp_path / 'sfc', tmp_path / 'pres'
    surface.mkdir(), vertical.mkdir()
    day = dt.date(YEAR, 3, 1)
    shutil.copy(merra2_like[0] / f'MERRA2_sfc_{day:%Y%m%d}.nc', surface)
    shutil.copy(merra2_like[1] / f'MERRA_pres_{day:%Y%m%d}.nc', vertical)
    fields, lat, lon = m2f.read_day(day, surface, vertical)
    assert fields.shape == (4, 20, NLAT, NLON) and lat.shape == (NLAT,) and lon.shape == (NLON,)
    with h5py.File(surface / f'MERRA2_sfc_{day:%Y%m%d}.nc', 'r+') as f:
        f['T2M'][2, 1, 1] = np.nan       # 06 UTC
    with pytest.raises(ValueError, match='t2m'):
        m2f.read_day(day, surface, vertical)


def test_calendar_and_splits():
    assert [d for d, _ in m2f.month_days(1980, 2)][-1] == dt.date(1980, 2, 29)
    assert m2f.month_days(1980, 12)[-1] == (dt.date(1980, 12, 30), 364)   # leap year: 365 days kept
    assert m2f.month_days(1981, 12)[-1] == (dt.date(1981, 12, 31), 364)
    assert sum(len(m2f.month_days(y, mth)) for y in (1980, 1981) for mth in range(1, 13)) == 730
    assert [m2f.split_of(y) for y in (1980, 2019, 2020, 2022, 2023)] == ['train', 'train', 'test', 'test', 'out_of_sample']
    assert m2f.parse_years('1980-1982,2020') == [1980, 1981, 1982, 2020]
    assert len(m2f.CHANNELS) == 20 and len(set(m2f.CHANNEL_NAMES)) == 20


# ---------------------------------------------------------------- loader and trainer, odd rows
# 41 x 32 pixels (the loader drops the last row -> 40 x 32) at patch 8: a 5 x 4 token grid with
# an odd number of rows, like MERRA-2's 45 x 72. No time_means file.
ODD_GRID_MIXERS = {
    'afno': '{type: afno}',
    'tucker_free': '{type: tucker_free, L: 2, p: 4, q: 2, r: 4, backend: gemm}',
    'tucker_spectral': '{type: tucker_spectral, L: 2, kh: 3, kw: 2, r: 4, backend: gemm}',
    'tucker_spectral_dct': '{type: tucker_spectral, L: 2, kh: 5, kw: 4, r: 4, basis: dct}',
    'tucker_adaptive': '{type: tucker_adaptive, L: 2, kh: 3, kw: 2, r: 4, ctx_hidden: 8, lat_basis: 4, backend: gemm}',
}


@pytest.fixture(scope='session')
def odd_rows(tmp_path_factory):
    root = tmp_path_factory.mktemp('odd_rows')
    rng = np.random.default_rng(0)
    lat = np.cos(np.linspace(0, np.pi, 41))[None, None, :, None]
    offset = 10.0 * np.arange(3)[None, :, None, None]
    for split, year, steps in (('train', 1990, 8), ('train', 1991, 8), ('test', 2020, 8)):
        (root / split).mkdir(exist_ok=True)
        with h5py.File(root / split / f'{year}.h5', 'w') as f:
            f['fields'] = (lat + offset + 0.1 * rng.standard_normal((steps, 3, 41, 32))).astype(np.float32)
    (root / 'stats').mkdir()
    np.save(root / 'stats' / 'global_means.npy', offset.astype(np.float32))
    np.save(root / 'stats' / 'global_stds.npy', np.full((1, 3, 1, 1), 0.5, np.float32))
    return root


@pytest.mark.parametrize('kind', ODD_GRID_MIXERS)
def test_training_on_an_odd_row_grid_without_climatology(kind, odd_rows, tmp_path):
    values = {'train_data_path': odd_rows / 'train', 'valid_data_path': odd_rows / 'test',
              'exp_dir': tmp_path, 'global_means_path': odd_rows / 'stats' / 'global_means.npy',
              'global_stds_path': odd_rows / 'stats' / 'global_stds.npy',
              'in_channels': '[0, 1, 2]', 'out_channels': '[0, 1, 2]', 'channel_names': '[a, b, c]',
              'embed_dim': 16, 'depth': 2, 'batch_size': 2, 'max_epochs': 1, 'num_data_workers': 0,
              'log_to_wandb': 'false', 'n_valid_samples': 4, 'mixer': ODD_GRID_MIXERS[kind]}
    run('train_mixers.py', '--yaml_config', CONFIG, '--config', kind, '--run_num', 0,
        '--override', *[f'{k}={v}' for k, v in values.items()])
    run_dir = tmp_path / kind / '0'
    model = jsonl(run_dir / 'model.jsonl')[-1]
    assert model['grid'] == [5, 4] and model['parameters/mixer'] > 0
    epoch = jsonl(run_dir / 'epochs.jsonl')[-1]
    assert epoch['steps'] == 8 and np.isfinite(epoch['train_epoch_loss'])
    valid = jsonl(run_dir / 'validation.jsonl')[-1]
    assert valid['n_valid_samples'] == 4 and np.isfinite(valid['valid_loss'])
    assert all(valid[f'valid_rmse/{name}'] > 0 for name in 'abc')
    assert not [k for k in valid if k.startswith('valid_acc')]     # no climatology, no ACC


# ---------------------------------------------------------------- the MERRA-2 config
def test_merra2_configs_differ_only_in_the_mixer():
    loaded = {kind: Params(CONFIG, kind) for kind in COMPARED}
    base = loaded['afno'].params
    for kind, p in loaded.items():
        differing = {k for k in set(base) | set(p.params) if base.get(k) != p.params.get(k)}
        assert differing <= {'mixer'}, (kind, differing)
    assert loaded['afno'].mixer == {'type': 'afno'}
    assert all(loaded[k].mixer.get('backend') == 'gemm' for k in ('tucker_free', 'tucker_spectral', 'tucker_adaptive'))
    assert loaded['tucker_spectral_dct'].mixer['basis'] == 'dct'


def test_merra2_config_matches_the_converter_and_fourcastnet():
    p = Params(CONFIG, 'afno')
    assert p.channel_names == m2f.CHANNEL_NAMES and len(p.in_channels) == len(p.out_channels) == 20
    assert p.time_means_path is None and p.dataset_tag == 'merra2'
    fcn = Params(ROOT / 'config' / 'mixer_comparison.yaml', 'afno')
    for key in ('lr', 'scheduler', 'batch_size', 'num_blocks', 'patch_size', 'embed_dim', 'depth', 'loss',
                'optimizer_type', 'amp_dtype', 'dt', 'n_history', 'max_epochs', 'seed', 'mixer'):
        assert p[key] == fcn[key], key           # FourCastNet's architecture and recipe, unchanged
    assert [fcn.channel_names[i] for i in range(20)] == [c[1] for c in m2f.CHANNELS]


@pytest.mark.parametrize('kind', COMPARED)
def test_merra2_models_run_at_full_size(kind):
    """360 x 576 at patch 8 -> the 45 x 72 token grid (odd rows) with the real mixer settings."""
    p = Params(CONFIG, kind)
    p['N_in_channels'] = p['N_out_channels'] = len(p.in_channels)
    torch.manual_seed(0)
    model = MixerNet(p, img_size=(360, 576), embed_dim=p.embed_dim, depth=p.depth).eval()
    assert (model.h, model.w) == (45, 72) and len(model.blocks) == 12
    with torch.no_grad():
        y = model(torch.randn(1, 20, 360, 576))
    assert y.shape == (1, 20, 360, 576) and torch.isfinite(y).all()

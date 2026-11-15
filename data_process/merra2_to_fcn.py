#!/usr/bin/env python
"""MERRA-2 (the daily netCDF files Prithvi WxC trains on) -> FourCastNet's year-file layout.

FourCastNet's loader (utils/data_loader_multifiles.py) wants one HDF5 file per year with a
dataset `fields` of shape (T, C, H + 1, W) at 6 h, split into train/ test/ out_of_sample/,
plus global_means.npy and global_stds.npy of shape (1, C, 1, 1). This script writes exactly
that from

    <surface>/MERRA2_sfc_YYYYMMDD.nc    (time, lat, lon), 3-hourly
    <vertical>/MERRA_pres_YYYYMMDD.nc   (time, lev, lat, lon), 3-hourly, 14 model levels

so that train_mixers.py runs on MERRA-2 without a new data loader:

    <out>/train/1980.h5 ... 2019.h5      fields (1460, 20, 361, 576) float32
    <out>/test/2020.h5 ... 2022.h5
    <out>/out_of_sample/2023.h5
    <out>/stats/global_means.npy, global_stds.npy, channels.json

Choices
  * 20 channels, the MERRA-2 counterparts of FourCastNet's 20 ERA5 variables in the same
    order (CHANNELS below). The upper-air fields sit on MERRA-2 model levels, not pressure
    levels: 72 ~ 985 hPa, 63 ~ 850 hPa, 51 ~ 525 hPa, 34 ~ 50 hPa (nominal). Humidity is
    specific humidity QV (MERRA-2 has no relative humidity) and H is geopotential height in m.
  * 6-hourly: the 00, 06, 12 and 18 UTC steps of each daily file.
  * every year has 365 x 4 = 1460 steps, as the loader assumes one length for all files:
    31 December of a leap year is dropped.
  * rows stay in MERRA-2 order (south to north, 361 rows); the loader drops the last row
    (the north pole), which leaves 360 x 576.
  * means and standard deviations are over all grid points and steps of the training years.
    No climatology (time_means) is written.

Robustness: each (year, month) is converted by a fresh interpreter (xarray/h5netcdf leaked
memory in long-lived readers during the 2026-10 NaN scan), months of one year run one after
another so a year file has a single writer, and years run in parallel. A year is written to
`<year>.h5.partial` and renamed when all twelve months are in, so the loader's `*.h5` glob
never sees an incomplete file and a killed job resumes where it stopped.

    python data_process/merra2_to_fcn.py --out /nobackupnfs1/sroy14/FourCastNet_MERRA2 \
        --years 1980-2023 --jobs 40          # convert (resumable), then write the stats
    python data_process/merra2_to_fcn.py --out ... --finalize   # stats only

CPU only; run it on a compute node.
"""
import argparse
import calendar
import datetime as dt
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SURFACE_DIR = '/nobackupnfs1/sroy14/Prithvi_WxC_training_data/processed_data_sfc'
VERTICAL_DIR = '/nobackupnfs1/sroy14/Prithvi_WxC_training_data/merra_pressuredata'

# (channel name, FourCastNet's variable at this position, MERRA-2 variable, model level or None)
CHANNELS = [
    ('u10',    'u10',   'U10M', None),
    ('v10',    'v10',   'V10M', None),
    ('t2m',    't2m',   'T2M',  None),
    ('sp',     'sp',    'PS',   None),
    ('slp',    'msl',   'SLP',  None),
    ('t_l63',  't850',  'T',    63.0),
    ('u_l72',  'u1000', 'U',    72.0),
    ('v_l72',  'v1000', 'V',    72.0),
    ('h_l72',  'z1000', 'H',    72.0),
    ('u_l63',  'u850',  'U',    63.0),
    ('v_l63',  'v850',  'V',    63.0),
    ('h_l63',  'z850',  'H',    63.0),
    ('u_l51',  'u500',  'U',    51.0),
    ('v_l51',  'v500',  'V',    51.0),
    ('h_l51',  'z500',  'H',    51.0),
    ('t_l51',  't500',  'T',    51.0),
    ('h_l34',  'z50',   'H',    34.0),
    ('qv_l51', 'r500',  'QV',   51.0),
    ('qv_l63', 'r850',  'QV',   63.0),
    ('tqv',    'tcwv',  'TQV',  None),
]
CHANNEL_NAMES = [c[0] for c in CHANNELS]
HOURS = (0, 6, 12, 18)
DAYS_PER_YEAR = 365
STEPS_PER_YEAR = DAYS_PER_YEAR * len(HOURS)
SPLITS = (('train', 1980, 2019), ('test', 2020, 2022), ('out_of_sample', 2023, 2023))
OPEN = {'chunks': None, 'cache': False, 'engine': 'h5netcdf'}


def split_of(year, splits=SPLITS):
    for name, first, last in splits:
        if first <= year <= last:
            return name
    raise ValueError(f'year {year} is in no split of {splits}')


def month_days(year, month):
    """The days of `month` among the first 365 days of `year`, with their day index."""
    first = dt.date(year, 1, 1)
    days = [dt.date(year, month, d) for d in range(1, calendar.monthrange(year, month)[1] + 1)]
    return [(day, (day - first).days) for day in days if (day - first).days < DAYS_PER_YEAR]


def parse_years(text):
    years = []
    for part in text.split(','):
        first, _, last = part.partition('-')
        years.extend(range(int(first), int(last or first) + 1))
    return years


def year_paths(out, year, splits=SPLITS):
    folder = Path(out) / split_of(year, splits)
    return folder / f'{year}.h5', folder / f'{year}.h5.partial'


# ----------------------------------------------------------------------------- reading one day
def _time_indices(ds, day, path):
    import pandas as pd
    stamps = pd.DatetimeIndex(ds['time'].values)
    wanted = [pd.Timestamp(day.year, day.month, day.day, hour) for hour in HOURS]
    index = [int(np.flatnonzero(stamps == stamp)[0]) if (stamps == stamp).sum() == 1 else None for stamp in wanted]
    if None in index:
        raise ValueError(f'{path}: expected one step at each of {HOURS} UTC on {day}, found times {list(stamps)}')
    return index


def read_day(day, surface_dir, vertical_dir):
    """(4, C, H, W) float32 for one day, plus the latitudes and longitudes of the files."""
    import xarray as xr
    stamp = day.strftime('%Y%m%d')
    sfc_path = Path(surface_dir) / f'MERRA2_sfc_{stamp}.nc'
    pres_path = Path(vertical_dir) / f'MERRA_pres_{stamp}.nc'
    out = None
    with xr.open_dataset(sfc_path, **OPEN) as sfc, xr.open_dataset(pres_path, **OPEN) as pres:
        lat, lon = np.asarray(sfc['lat'].values, np.float64), np.asarray(sfc['lon'].values, np.float64)
        if not (np.array_equal(lat, pres['lat'].values) and np.array_equal(lon, pres['lon'].values)):
            raise ValueError(f'{sfc_path} and {pres_path} are on different grids')
        if not (np.all(np.diff(lat) > 0) and np.all(np.diff(lon) > 0)):
            raise ValueError(f'{sfc_path}: latitudes and longitudes must be ascending')
        out = np.empty((len(HOURS), len(CHANNELS), lat.size, lon.size), np.float32)
        sfc_times, pres_times = _time_indices(sfc, day, sfc_path), _time_indices(pres, day, pres_path)

        # one read per MERRA-2 variable (a variable's levels share compressed chunks)
        variables = {}
        for c, (_, _, variable, level) in enumerate(CHANNELS):
            variables.setdefault(variable, []).append((c, level))
        for variable, uses in variables.items():
            if uses[0][1] is None:
                (c, _), = uses
                out[:, c] = sfc[variable].isel(time=sfc_times).transpose('time', 'lat', 'lon').values
            else:
                levels = [level for _, level in uses]
                block = pres[variable].isel(time=pres_times).sel(lev=levels).transpose('time', 'lev', 'lat', 'lon').values
                for k, (c, _) in enumerate(uses):
                    out[:, c] = block[:, k]
    bad = ~np.isfinite(out)
    if bad.any() or np.abs(out).max() > 1e30:
        channels = sorted({CHANNEL_NAMES[c] for c in np.argwhere(bad | (np.abs(out) > 1e30))[:, 1]})
        raise ValueError(f'{stamp}: NaN, inf or |x| > 1e30 in channels {channels}')
    return out, lat, lon


# ----------------------------------------------------------------------------- one month, one writer
def create_partial(path, lat, lon, year):
    import h5py
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as f:
        # contiguous and uncompressed: one sample is one 16.6 MB sequential read for the loader
        f.create_dataset('fields', shape=(STEPS_PER_YEAR, len(CHANNELS), lat.size, lon.size), dtype='f4')
        f['lat'], f['lon'] = lat, lon
        first = dt.datetime(year, 1, 1)
        stamps = [first + dt.timedelta(hours=6 * i) for i in range(STEPS_PER_YEAR)]
        f['time'] = np.array([int(s.strftime('%Y%m%d%H')) for s in stamps], np.int64)
        f.attrs['channel_names'] = json.dumps(CHANNEL_NAMES)
        f.attrs['channels'] = json.dumps(CHANNELS)
        f.attrs['source'] = 'MERRA-2 (Prithvi WxC training files); data_process/merra2_to_fcn.py'
        stats = f.create_group('stats')
        stats.create_dataset('done', data=np.zeros(12, bool))
        stats.create_dataset('count', data=np.zeros(12, np.int64))          # values per channel
        for name in ('sum', 'sumsq'):
            stats.create_dataset(name, data=np.zeros((12, len(CHANNELS)), np.float64))
        stats.create_dataset('min', data=np.full((12, len(CHANNELS)), np.inf))
        stats.create_dataset('max', data=np.full((12, len(CHANNELS)), -np.inf))


def convert_month(year, month, partial, surface_dir, vertical_dir):
    """Write one month of `year` into the partial year file (created on first use)."""
    import h5py
    partial = Path(partial)
    if partial.exists():
        with h5py.File(partial, 'r') as f:
            if f['stats/done'][month - 1]:
                print(f'{year}-{month:02d}: already done', flush=True)
                return
    days = month_days(year, month)
    start = time.time()
    block = total = total_sq = lo = hi = None
    for i, (day, _) in enumerate(days):
        fields, lat, lon = read_day(day, surface_dir, vertical_dir)
        if block is None:
            block = np.empty((len(days) * len(HOURS), *fields.shape[1:]), np.float32)
            total, total_sq = np.zeros(len(CHANNELS)), np.zeros(len(CHANNELS))
            lo, hi = np.full(len(CHANNELS), np.inf), np.full(len(CHANNELS), -np.inf)
        block[i * len(HOURS):(i + 1) * len(HOURS)] = fields
        wide = fields.astype(np.float64)
        total += wide.sum(axis=(0, 2, 3))
        total_sq += np.square(wide).sum(axis=(0, 2, 3))
        lo, hi = np.minimum(lo, wide.min(axis=(0, 2, 3))), np.maximum(hi, wide.max(axis=(0, 2, 3)))
    if not partial.exists():
        create_partial(partial, lat, lon, year)
    first_step = days[0][1] * len(HOURS)
    with h5py.File(partial, 'r+') as f:
        if not (np.array_equal(f['lat'][:], lat) and np.array_equal(f['lon'][:], lon)):
            raise ValueError(f'{year}-{month:02d}: grid differs from the one in {partial}')
        f['fields'][first_step:first_step + block.shape[0]] = block
        m = month - 1
        f['stats/sum'][m], f['stats/sumsq'][m] = total, total_sq
        f['stats/min'][m], f['stats/max'][m] = lo, hi
        f['stats/count'][m] = block.shape[0] * block.shape[2] * block.shape[3]
        f.flush()
        f['stats/done'][m] = True
    print(f'{year}-{month:02d}: {len(days)} days, steps {first_step}..{first_step + block.shape[0] - 1}, '
          f'{time.time() - start:.0f} s', flush=True)


def seal(partial, final):
    """Rename a complete partial file to its final name."""
    import h5py
    with h5py.File(partial, 'r') as f:
        done, count = f['stats/done'][:], int(f['stats/count'][:].sum())
        shape = f['fields'].shape
    if not done.all():
        raise ValueError(f'{partial}: months {list(np.flatnonzero(~done) + 1)} are missing')
    if count != shape[0] * shape[2] * shape[3]:
        raise ValueError(f'{partial}: {count} values written per channel, expected {shape[0] * shape[2] * shape[3]}')
    os.replace(partial, final)
    print(f'sealed {final}', flush=True)


# ----------------------------------------------------------------------------- statistics
def finalize(out, splits=SPLITS):
    """global_means.npy / global_stds.npy (1, C, 1, 1) over every training year present."""
    import h5py
    out = Path(out)
    files = sorted((out / 'train').glob('*.h5'))
    if not files:
        raise ValueError(f'no training year files under {out / "train"}')
    count, total, total_sq = 0, np.zeros(len(CHANNELS)), np.zeros(len(CHANNELS))
    lo, hi = np.full(len(CHANNELS), np.inf), np.full(len(CHANNELS), -np.inf)
    for path in files:
        with h5py.File(path, 'r') as f:
            if json.loads(f.attrs['channel_names']) != CHANNEL_NAMES:
                raise ValueError(f'{path}: channels differ from this script')
            count += int(f['stats/count'][:].sum())
            total += f['stats/sum'][:].sum(0)
            total_sq += f['stats/sumsq'][:].sum(0)
            lo, hi = np.minimum(lo, f['stats/min'][:].min(0)), np.maximum(hi, f['stats/max'][:].max(0))
    means = total / count
    stds = np.sqrt(total_sq / count - means ** 2)
    (out / 'stats').mkdir(exist_ok=True)
    for name, values in (('global_means.npy', means), ('global_stds.npy', stds)):
        tmp = out / 'stats' / f'.tmp_{os.getpid()}_{name}'     # readers never see a half-written file
        np.save(tmp, values.astype(np.float32).reshape(1, -1, 1, 1))
        os.replace(tmp, out / 'stats' / name)
    summary = {'years': [p.stem for p in files], 'values_per_channel': count,
               'channels': [{'name': n, 'fourcastnet': fcn, 'merra2': var, 'level': lev, 'mean': float(mu),
                             'std': float(sd), 'min': float(a), 'max': float(b)}
                            for (n, fcn, var, lev), mu, sd, a, b in zip(CHANNELS, means, stds, lo, hi)]}
    with open(out / 'stats' / 'channels.json', 'w') as stream:
        json.dump(summary, stream, indent=1)
    print(f'stats over {len(files)} training years ({files[0].stem}-{files[-1].stem}) -> {out / "stats"}')
    for row in summary['channels']:
        print('  {name:8s} mean {mean:12.5g}  std {std:12.5g}  min {min:12.5g}  max {max:12.5g}'.format(**row))
    if not np.all(np.isfinite(stds)) or np.any(stds <= 0):
        raise ValueError('a channel has a non-finite or zero standard deviation')


# ----------------------------------------------------------------------------- driver
def convert_year(year, args):
    """Run the twelve month workers of one year in order, then seal. Returns (year, ok)."""
    final, partial = year_paths(args.out, year)
    if final.exists():
        return year, True, 'present'
    log = Path(args.out) / 'logs' / f'{year}.log'
    base = [sys.executable, os.path.abspath(__file__), '--out', str(args.out), '--surface', args.surface,
            '--vertical', args.vertical, '--worker-year', str(year)]
    start = time.time()
    with open(log, 'a') as stream:
        for month in [*range(1, 13), 0]:          # 0 = seal
            rc = subprocess.run([*base, '--worker-month', str(month)], stdout=stream, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                return year, False, f'month {month} failed (rc {rc}), see {log}'
    return year, True, f'{time.time() - start:.0f} s'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--surface', default=SURFACE_DIR)
    p.add_argument('--vertical', default=VERTICAL_DIR)
    p.add_argument('--years', default='1980-2023', help='e.g. 1980-2019,2023')
    p.add_argument('--jobs', type=int, default=8, help='years converted in parallel')
    p.add_argument('--finalize', action='store_true', help='only write the statistics')
    p.add_argument('--no-stats', action='store_true', help='convert without writing the statistics')
    p.add_argument('--worker-year', type=int, help=argparse.SUPPRESS)
    p.add_argument('--worker-month', type=int, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.worker_year is not None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from utils.mixer_params import disable_transparent_hugepages
        disable_transparent_hugepages()
        final, partial = year_paths(args.out, args.worker_year)
        if args.worker_month == 0:
            seal(partial, final)
        else:
            convert_month(args.worker_year, args.worker_month, partial, args.surface, args.vertical)
        return 0

    if not args.finalize:
        years = parse_years(args.years)
        for year in years:
            year_paths(args.out, year)[0].parent.mkdir(parents=True, exist_ok=True)
        (args.out / 'logs').mkdir(parents=True, exist_ok=True)
        print(f'converting {len(years)} years with {args.jobs} in parallel -> {args.out}', flush=True)
        failed = []
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for year, ok, note in pool.map(lambda y: convert_year(y, args), years):
                print(f'{year}: {"ok" if ok else "FAILED"} ({note})', flush=True)
                if not ok:
                    failed.append(year)
        if failed:
            print(f'FAILED years: {failed}; rerun the same command to resume')
            return 1
    if not args.no_stats:
        finalize(args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())

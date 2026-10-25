#!/usr/bin/env python3
"""Rollout skill of the mixer runs side by side, with persistence as the reference.

Reads every inference_summary.json (written by inference/inference_mixers.py) under the
given directories and prints markdown tables of latitude-weighted RMSE (physical units)
and ACC per channel and lead time, mean over initial conditions. The best mixer in
each row is in bold; persistence is shown but not ranked.

    python scripts/summarize_rollouts.py expts/mixer_comparison --output results/rollouts.md
"""
import argparse
import json
from pathlib import Path

ORDER = ['afno', 'vanilla', 'tucker_free', 'tucker_spectral', 'tucker_adaptive']


def load(paths):
    summaries = []
    for path in paths:
        path = Path(path)
        files = sorted(path.rglob('inference_summary*.json')) if path.is_dir() else [path]
        summaries += [json.loads(f.read_text()) | {'_path': str(f)} for f in files]
    summaries.sort(key=lambda s: (ORDER.index(s['mixer']) if s['mixer'] in ORDER else len(ORDER), s['config']))
    labels = [s['config'] for s in summaries]
    for s in summaries:  # several runs of one config: tell them apart by run directory
        if labels.count(s['config']) > 1:
            s['config'] = f"{s['config']} ({Path(s['_path']).parent.name})"
    return summaries


def fmt(value):
    return f'{value:.4g}'


def table(summaries, metric, channels, leads, best):
    names = [s['config'] for s in summaries]
    lines = ['| channel | lead | persistence | ' + ' | '.join(names) + ' |',
             '|---|---|---|' + '---|' * len(names)]
    ref = summaries[0]
    for channel in channels:
        for lead in leads:
            if lead not in ref['lead_hours'] or channel not in ref[metric]:
                continue
            i = ref['lead_hours'].index(lead)
            values = [s[metric][channel][i] for s in summaries]
            top = best(values)
            cells = [f'**{fmt(v)}**' if v == top and len(values) > 1 else fmt(v) for v in values]
            lines.append(f"| {channel} | {lead} h | {fmt(ref[metric + '_persistence'][channel][i])} | "
                         + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('inputs', nargs='+', type=Path, help='summary files or directories to search')
    parser.add_argument('--channels', nargs='+', default=['z500', 't850', 't2m', 'u10', 'v10'])
    parser.add_argument('--leads', nargs='+', type=int, default=[24, 72, 120], help='hours')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    summaries = load(args.inputs)
    if not summaries:
        raise SystemExit('no inference_summary.json found')
    ics = {tuple(s['initial_conditions']) for s in summaries}
    note = ('' if len(ics) == 1 else
            '\n> Warning: runs were scored on different initial conditions; persistence is from the first run.\n')
    text = (f"Mean over {len(summaries[0]['initial_conditions'])} initial conditions.{note}\n\n"
            '## RMSE (physical units, latitude weighted; lower is better)\n\n'
            + table(summaries, 'rmse', args.channels, args.leads, min)
            + '\n\n## ACC (latitude weighted; higher is better)\n\n'
            + table(summaries, 'acc', args.channels, args.leads, max) + '\n')
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)


if __name__ == '__main__':
    main()

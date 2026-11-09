"""Aggregate chunked result CSVs under Results/ into aligned summary tables.

Usage:
    python aggregate_results.py                 # all runs
    python aggregate_results.py --type basic    # one constraint type
    python aggregate_results.py --run pilot     # runs whose name contains "pilot"
    python aggregate_results.py --detail RUN    # per-instance view of a single run

Output is plain ASCII with fixed-width columns.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Results')

PROF_COLS = ['total_time(s)', 'init_time(s)', 'sample_time(s)', 'rl_time(s)', 'reward_time(s)']


def load_run(run_dir):
    """Merge every chunk CSV of one run, dedupe by sample_idx (keep last write)."""
    frames = []
    for path in sorted(glob.glob(os.path.join(run_dir, '*.csv'))):
        try:
            frames.append(pd.read_csv(path))
        except Exception as e:  # noqa: BLE001 - a broken chunk should not kill the report
            print(f'[warn] unreadable csv skipped: {path} ({e})', file=sys.stderr)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    if 'sample_idx' in df.columns:
        df = df.drop_duplicates(subset='sample_idx', keep='last').sort_values('sample_idx')
    return df


def parse_n_cities(run_dir):
    for path in glob.glob(os.path.join(run_dir, '*.csv')):
        m = re.search(r'tsp(\d+)_', os.path.basename(path))
        if m:
            return int(m.group(1))
    return None


def summarize(df, run_name, constraint, n_cities):
    row = {
        'type': constraint,
        'N': n_cities if n_cities is not None else -1,
        'run': run_name,
        'inst': len(df),
        'obj': df['solved_cost'].mean(),
        'gt': df['gt_cost'].mean(),
        'gap%': df['final_gap(%)'].mean(),
        'med_gap%': df['final_gap(%)'].median(),
    }
    if 'penalty_count' in df.columns:
        row['feas%'] = 100.0 * (df['penalty_count'] == 0).mean()
    if 'basic_cost' in df.columns and (df['basic_cost'] != df['solved_cost']).any():
        # basic_cost is the tour length of the first evaluated candidate (the initial solution)
        init_gap = 100.0 * (df['basic_cost'] - df['gt_cost']) / df['gt_cost']
        row['init_gap%'] = init_gap.mean()
        row['improved%'] = 100.0 * (df['solved_cost'] < df['basic_cost'] - 1e-9).mean()
    if 'total_time(s)' in df.columns:
        row['s/inst'] = df['total_time(s)'].mean()
    if 'early_stopped' in df.columns:
        row['estop%'] = 100.0 * df['early_stopped'].mean()
    return row


NUMERIC_FMT = {
    'N': '{:d}', 'inst': '{:d}',
    'obj': '{:.3f}', 'gt': '{:.3f}',
    'gap%': '{:.3f}', 'med_gap%': '{:.3f}', 'init_gap%': '{:.3f}',
    'feas%': '{:.1f}', 'improved%': '{:.1f}', 's/inst': '{:.1f}', 'estop%': '{:.1f}',
}
COL_ORDER = ['type', 'N', 'run', 'inst', 'obj', 'gt', 'gap%', 'med_gap%',
             'init_gap%', 'improved%', 'feas%', 's/inst', 'estop%']


def render(rows):
    cols = [c for c in COL_ORDER if any(c in r for r in rows)]
    cells = []
    for r in rows:
        line = []
        for c in cols:
            v = r.get(c)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                line.append('-')
            elif c in NUMERIC_FMT:
                line.append(NUMERIC_FMT[c].format(v))
            else:
                line.append(str(v))
        cells.append(line)
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
    def fmt(row):
        out = []
        for i, cell in enumerate(row):
            # left-align text columns, right-align numbers
            out.append(cell.ljust(widths[i]) if cols[i] in ('type', 'run') else cell.rjust(widths[i]))
        return '  '.join(out)
    sep = '  '.join('-' * w for w in widths)
    return '\n'.join([fmt(cols), sep] + [fmt(row) for row in cells])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--type', default=None, help='filter by constraint type (basic/box/path/cluster)')
    ap.add_argument('--run', default=None, help='substring filter on run name')
    ap.add_argument('--detail', default=None, help='print per-instance table for runs matching this substring')
    ap.add_argument('--sort', default='type,N,run', help='comma-separated sort keys')
    args = ap.parse_args()

    rows = []
    for constraint in sorted(os.listdir(RESULTS_DIR)):
        cdir = os.path.join(RESULTS_DIR, constraint)
        if not os.path.isdir(cdir):
            continue
        if args.type and constraint != args.type:
            continue
        for run_name in sorted(os.listdir(cdir)):
            run_dir = os.path.join(cdir, run_name)
            if not os.path.isdir(run_dir):
                continue
            if args.run and args.run not in run_name:
                continue
            df = load_run(run_dir)
            if df is None or df.empty:
                continue
            if args.detail and args.detail in run_name:
                keep = [c for c in ['sample_idx', 'best_epoch', 'solved_cost', 'gt_cost',
                                    'basic_cost', 'penalty_count', 'final_gap(%)',
                                    'total_time(s)', 'early_stopped'] if c in df.columns]
                print(f'== {constraint}/{run_name} ({len(df)} instances) ==')
                print(df[keep].to_string(index=False,
                                         float_format=lambda x: f'{x:.4f}'))
                print()
            rows.append(summarize(df, run_name, constraint, parse_n_cities(run_dir)))

    if not rows:
        print('no matching runs under', RESULTS_DIR)
        return
    keys = [k.strip() for k in args.sort.split(',')]
    rows.sort(key=lambda r: tuple(str(r.get(k, '')) if k not in ('N', 'inst') else r.get(k, 0) for k in keys))
    print(render(rows))


if __name__ == '__main__':
    main()

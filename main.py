"""DDRL: A Diffusion-Driven Reinforcement Learning Approach for Enhanced TSP Solutions.

Runs the instance-wise search on instances [start_idx, end_idx) of one test file
and writes one CSV row per instance to
    Results/<constraint_type>/<run_name>/ours_tsp<N>_<constraint>_<start>_<end>.csv

Example (the paper's budget is the default):
    python main.py --constraint_type box --num_cities 20 --start_idx 0 --end_idx 8

Code map (paper -> code):
    A(phi), f(phi)                  ddrl/latent.py
    Eq. (1), Supp. Def. D.1         ddrl/sampling.py
    Eq. (2), reward r(phi)          ddrl/reward.py, ddrl/decoding.py, ddrl/solver.py
    Eq. (3), Algorithms 1 and 2     ddrl/search.py
    Supp. Eq. (A1) initialization   ddrl/prior_init.py
"""
import gc
import json
import os
import subprocess
import sys
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ddrl.config import load_config
from ddrl.dataset import TSPDataset
from ddrl.prior_init import build_prior
from ddrl.sampling import build_scheduler
from ddrl.search import InstanceContext, optimize_instance
from ddrl.utils import enable_determinism, seed_everything

# generation date suffix of the released constrained benchmark files (data/README.md)
DATASET_DATE = {'box': '240710', 'path': '240711', 'cluster': '240721'}


def data_file_name(cfg):
    if cfg.constraint_type == 'basic':
        return f'tsp{cfg.num_cities}_test_concorde.txt'
    return f'tsp{cfg.num_cities}_{cfg.constraint_type}_constraint_{DATASET_DATE[cfg.constraint_type]}.txt'


def result_file_name(cfg):
    kind = 'basic' if cfg.constraint_type == 'basic' else f'{cfg.constraint_type}_constraint'
    return f'ours_tsp{cfg.num_cities}_{kind}_{cfg.start_idx}_{cfg.end_idx}.csv'


def save_run_meta(cfg, result_dir):
    """Write the full configuration next to the results, so every result directory is self-describing."""
    repo = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                                cwd=repo).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True,
                                    cwd=repo).stdout.strip())
    except Exception:
        commit, dirty = 'unknown', None
    meta = {
        'git_commit': commit,
        'git_dirty': dirty,
        'argv': sys.argv,
        'config': dict(cfg),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'torch': torch.__version__,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
    }
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, f'run_meta_{cfg.start_idx}_{cfg.end_idx}.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=str)


def main():
    start_time = time.time()
    cfg = load_config()
    enable_determinism(cfg.deterministic)
    seed_everything(cfg.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    result_dir = os.path.join('./Results', cfg.constraint_type, cfg.run_name)
    result_path = os.path.join(result_dir, result_file_name(cfg))
    print(json.dumps(cfg, indent=4))
    save_run_meta(cfg, result_dir)
    print(f'Result file : {result_path}')

    # eps_prior: fixed pre-trained U-Net (random weights with --no-use_prior)
    prior, ckpt = build_prior(cfg.img_size, device, ckpt_dir=cfg.ckpt_path if cfg.use_prior else None,
                              num_cities=cfg.num_cities, ckpt_name=cfg.prior_ckpt)
    if ckpt:
        print(f'num_cities : {cfg.num_cities}, prior : {ckpt}')
    scheduler = build_scheduler()
    if cfg.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    dataset = TSPDataset(
        data_file=os.path.join(cfg.data_path, data_file_name(cfg)),
        img_size=cfg.img_size,
        constraint_type=cfg.constraint_type,
        point_radius=cfg.point_radius,
        point_color=cfg.point_color,
        point_circle=cfg.point_circle,
        line_thickness=cfg.line_thickness,
        line_color=cfg.line_color,
    )
    indices = range(cfg.start_idx, min(cfg.end_idx, len(dataset)))
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, pin_memory=True,
                        num_workers=1 if sys.gettrace() is not None else cfg.num_workers)

    dphi_trace = None
    if cfg.dphi_trace:
        with open(cfg.dphi_trace) as f:
            dphi_trace = {int(k): v for k, v in json.load(f).items()}
    ctx = InstanceContext(cfg, device, prior, scheduler, dphi_trace)

    rows, dphi_dump = [], {}

    def save():
        if cfg.save_result:
            os.makedirs(result_dir, exist_ok=True)
            pd.DataFrame(rows).to_csv(result_path, index=False)

    for img, points, gt_tour, sample_idx, constraint in tqdm(loader, desc="Optimize test samples",
                                                             dynamic_ncols=True):
        points, gt_tour = points.cpu().numpy()[0], gt_tour.cpu().numpy()[0]
        constraint = None if cfg.constraint_type == 'basic' else constraint.cpu().numpy()[0]

        row, dphi = optimize_instance(ctx, img, points, gt_tour, sample_idx, constraint)
        rows.append(row)
        if cfg.dump_dphi:
            dphi_dump[int(sample_idx)] = dphi
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if sample_idx % cfg.save_freq == 0:
            save()

    save()
    if cfg.save_result and cfg.dump_dphi:
        path = os.path.join(result_dir, f'dphi_{cfg.start_idx}_{cfg.end_idx}.json')
        with open(path, 'w') as f:
            json.dump(dphi_dump, f)
        print(f'dphi trace  : {path}')

    hours, rem = divmod(time.time() - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total execution time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
    if rows:
        df = pd.DataFrame(rows)
        print(f"N={cfg.num_cities} {cfg.constraint_type} | instances={len(df)} | "
              f"mean gap {df['final_gap(%)'].mean():.3f}% | "
              f"time/instance {df['total_time(s)'].mean():.1f}s "
              f"(init {df['init_time(s)'].mean():.1f}, sample {df['sample_time(s)'].mean():.1f}, "
              f"rl {df['rl_time(s)'].mean():.1f}, reward {df['reward_time(s)'].mean():.1f}) | "
              f"reward evals {df['num_reward_evals'].mean():.1f} | "
              f"peak GPU mem {df['peak_gpu_mem(MB)'].max():.0f} MB")


if __name__ == '__main__':
    main()

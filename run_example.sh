#!/usr/bin/env bash
# DDRL-prior on the first 8 instances of the N=20 obstacle (box) benchmark, with the
# evaluation budget used throughout the paper: 10 initial samples, 5 outer rounds,
# 5 inner epochs in round 1 (68 reward evaluations per instance).
#
# Needs: ckpt/unet50_64_8.pth, data/tsp20_box_constraint_240710.txt (both described in
# ckpt/README.md and data/README.md) and the built ddrl/cython_merge extension (README.md).
# Results land in Results/box/<run_name>/ours_tsp20_box_constraint_0_8.csv.
set -euo pipefail
cd "$(dirname "$0")"
# The budget is the default of configs/default.yaml; it is spelled out here for clarity.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python main.py \
  --constraint_type box --num_cities 20 --img_size 64 \
  --start_idx 0 --end_idx 8 \
  --num_init_sample 10 --num_epochs 5 --num_inner_epochs 5 --no-use_early_stop \
  --init_mode prior --num_workers 1 \
  --run_name example_box20 "$@"

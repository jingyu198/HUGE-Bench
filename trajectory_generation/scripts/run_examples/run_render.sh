#!/usr/bin/env bash
set -euo pipefail

: "${HUGE_DATA_ROOT:?Set HUGE_DATA_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python trajectory_generation/scripts/render/my_render_traj_overall.py \
  --data_root "$HUGE_DATA_ROOT" \
  --env_id 1_office \
  --task_id hl \
  --poses_txt_name traj_random.txt \
  --intr_scale 0.065

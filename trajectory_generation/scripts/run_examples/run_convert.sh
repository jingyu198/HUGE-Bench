#!/usr/bin/env bash
set -euo pipefail

: "${HUGE_DATA_TRAJ_ROOT:?Set HUGE_DATA_TRAJ_ROOT}"
: "${HUGE_LEROBOT_ROOT:?Set HUGE_LEROBOT_ROOT}"

python trajectory_generation/scripts/convert/build_instruction_splits.py \
  --data_root "$HUGE_DATA_TRAJ_ROOT" \
  --task_id hl \
  --env_ids 1_office,2_city,3_road,4_lake \
  --unseen_env_ids 4_lake

uv run trajectory_generation/scripts/convert/convert_and_merge.py \
  --data_root "$HUGE_DATA_TRAJ_ROOT" \
  --task_id hl \
  --repo_id_prefix task_hl \
  --splits train,test_seen,test_unseen

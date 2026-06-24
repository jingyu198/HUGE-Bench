#!/usr/bin/env bash
set -euo pipefail

: "${HUGE_DATA_3D_ROOT:?Set HUGE_DATA_3D_ROOT}"
: "${HUGE_DATA_TRAJ_ROOT:?Set HUGE_DATA_TRAJ_ROOT}"

python trajectory_generation/scripts/generate/traj_gen_hl.py --env_id 1_office --traj_per_loc 9
python trajectory_generation/scripts/generate/traj_gen_orbit.py --env_id 1_office --traj_per_loc 9
python trajectory_generation/scripts/generate/traj_gen_building.py --env_id 1_office --traj_per_loc 15

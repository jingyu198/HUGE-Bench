# Trajectory Generation

This directory contains the released trajectory collection pipeline used by
HUGE-Bench. The pipeline has four stages:

1. Generate camera trajectories and instructions.
2. Render RGB frames, and optionally mesh depth, from the 3DGS-Mesh scenes.
3. Build train / test_seen / test_unseen instruction splits.
4. Convert the rendered data into LeRobot format for PI0/OpenPI training.

The scripts are adapted from the internal data-generation workspace and are kept
close to the original task logic. Paths are configurable through environment
variables and command-line arguments.

## Data Layout

Expected dataset root:

```text
<HUGE_DATA_ROOT>/
├── data_3d/
│   └── <env_id>/
│       ├── BlocksExchangeUndistortAT_WithoutTiePoints.xml
│       ├── 3dgs_ply/
│       │   ├── point_cloud_utm50.ply
│       │   └── metadata.xml
│       ├── terra_ply/
│       │   ├── simplified_mesh.obj
│       │   └── metadata.xml
│       ├── location_gen/landmark_merged*.txt
│       ├── building_coords/*.txt
│       ├── farm_coords/*.txt
│       └── road_coords/*.txt
└── data_traj/
    └── task_<task_id>/
        └── <env_id>/
            ├── traj_random.txt
            ├── traj_meta.txt
            ├── subtask.txt
            ├── instruction.txt
            ├── render_img/
            ├── render_depth/
            └── wash_res.txt
```

The public `HUGE_Environment` release contains the 3DGS point clouds and mesh
assets. The trajectory-generation scripts also need the small XML / metadata /
annotation files listed above. If they are distributed separately, place them
under the same `data_3d/<env_id>/` folders.

## Environment Variables

```bash
export HUGE_DATA_ROOT=/path/to/HUGE_data
export HUGE_DATA_3D_ROOT=$HUGE_DATA_ROOT/data_3d
export HUGE_DATA_TRAJ_ROOT=$HUGE_DATA_ROOT/data_traj
export HUGE_LEROBOT_ROOT=/path/to/lerobot_output
```

`HUGE_DATA_ROOT` is used by rendering scripts. `HUGE_DATA_3D_ROOT` and
`HUGE_DATA_TRAJ_ROOT` are used by trajectory-generation scripts.
`HUGE_LEROBOT_ROOT` is used by conversion and merge scripts.

## Task Scripts

| Task ID | Script | Main Environments |
| --- | --- | --- |
| `0` | `scripts/generate/traj_gen_0.py` | `1_office`, `2_city`, `3_road`, `4_lake` |
| `hl` | `scripts/generate/traj_gen_hl.py` | `1_office`, `2_city`, `3_road`, `4_lake` |
| `orbit` | `scripts/generate/traj_gen_orbit.py` | `1_office`, `2_city`, `3_road`, `4_lake` |
| `orbit_multi` | `scripts/generate/traj_gen_orbit_multi.py` | `1_office`, `2_city`, `3_road`, `4_lake` |
| `building` | `scripts/generate/traj_gen_building.py` | `1_office`, `2_city` |
| `road` | `scripts/generate/traj_gen_road.py` | `3_road`, `real_road` |
| `farm` | `scripts/generate/traj_gen_farm.py` | `3_road`, `4_lake` |
| `obstacle` | `scripts/generate/traj_gen_obstacle.py` | `2_city`, `no1_building`, `no3_door`, `overhead_bridge` |

Example generation command:

```bash
python trajectory_generation/scripts/generate/traj_gen_hl.py \
  --env_id 1_office \
  --traj_per_loc 9
```

For obstacle trajectories:

```bash
python trajectory_generation/scripts/generate/traj_gen_obstacle.py \
  --env_id overhead_bridge \
  --num_traj 100
```

## Rendering

Run render scripts from inside a Gaussian Splatting checkout so imports such as
`gaussian_renderer`, `scene`, `arguments`, and `utils` are available.

```bash
CUDA_VISIBLE_DEVICES=0 python trajectory_generation/scripts/render/my_render_traj_overall.py \
  --data_root "$HUGE_DATA_ROOT" \
  --env_id 1_office \
  --task_id hl \
  --poses_txt_name traj_random.txt \
  --intr_scale 0.065
```

For obstacle environments, use `my_render_traj_overall_obs.py`.

The renderer writes `render_img/`, optional `render_depth/`, and `wash_res.txt`
beside the trajectory file. `wash_res.txt` records valid trajectories after
render-quality filtering.

## Instruction Splits

The conversion script expects:

```text
data_traj/task_<task_id>/split_res_merged/
├── instruction_train.txt
├── instruction_test_seen.txt
└── instruction_test_unseen.txt
```

Each line uses:

```text
env_id traj_id pose_start pose_end instruction text...
```

These files should be built from each environment's `instruction.txt` and
`wash_res.txt`, using the train / seen / unseen protocol for the release.

For a deterministic split helper:

```bash
python trajectory_generation/scripts/convert/build_instruction_splits.py \
  --data_root "$HUGE_DATA_TRAJ_ROOT" \
  --task_id hl \
  --env_ids 1_office,2_city,3_road,4_lake \
  --unseen_env_ids 4_lake
```

## Convert to LeRobot

Run from the OpenPI environment with LeRobot installed:

```bash
cd /path/to/openpi
uv run /path/to/HUGE-Bench/trajectory_generation/scripts/convert/convert_and_merge.py \
  --data_root "$HUGE_DATA_TRAJ_ROOT" \
  --task_id hl \
  --repo_id_prefix task_hl \
  --splits train,test_seen,test_unseen
```

The converter creates shard datasets and then merges them into:

```text
$HUGE_LEROBOT_ROOT/task_<task_id>/<split>/
```

To merge task-level LeRobot datasets into the overall benchmark split:

```bash
python trajectory_generation/scripts/convert/merge_lerobot_overall.py \
  --src_root "$HUGE_LEROBOT_ROOT" \
  --dst_root "$HUGE_LEROBOT_ROOT" \
  --out_repo_prefix task_overall
```

To convert high-level instructions into concatenated subtask instructions for a
low-level variant:

```bash
python trajectory_generation/scripts/convert/to_low_level.py \
  --task_root "$HUGE_DATA_TRAJ_ROOT/task_building" \
  --src_repo_dir "$HUGE_LEROBOT_ROOT/task_building/train" \
  --dst_repo_dir "$HUGE_LEROBOT_ROOT/task_building_low/train" \
  --split train
```

## Notes

- `traj_random.txt` is the default pose file for synthetic trajectories.
- `traj_random_global.txt` is still supported for real-pose/global rendering by
  passing `--poses_txt_name traj_random_global.txt`.
- Rendering scripts first look for `terra_ply/merged_mesh.obj` and fall back to
  `terra_ply/simplified_mesh.obj`, matching the public environment release.
- `label_gen.py` is an optional visualization helper for projecting building
  contour points onto rendered frames.

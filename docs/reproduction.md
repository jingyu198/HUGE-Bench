# Reproduction Guide

This guide describes the shortest path for reproducing the public HUGE-Bench
benchmark with the released trajectories, 3DGS-Mesh environments, PI0 checkpoint,
and evaluation code.

## 1. External Repositories

HUGE-Bench reuses two upstream codebases. Install them first, following their
official instructions:

- [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) for RGB rendering.
- [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) for PI0 training and rollout inference.

Then copy or merge the adapter files from this repository:

```bash
# Inside an official gaussian-splatting checkout.
cp /path/to/HUGE-Bench/gaussian_splatting/3dgs_renderer.py .
cp /path/to/HUGE-Bench/gaussian_splatting/my_render_traj.py .
cp /path/to/HUGE-Bench/gaussian_splatting/utils/graphics_utils.py utils/graphics_utils.py

# Inside an official openpi checkout.
cp /path/to/HUGE-Bench/openpi/scripts/action_infer.py scripts/action_infer.py
cp /path/to/HUGE-Bench/openpi/src/openpi/policies/drone_policy.py src/openpi/policies/drone_policy.py
cp /path/to/HUGE-Bench/openpi/src/openpi/training/config.py src/openpi/training/config.py
```

If your upstream checkout has local modifications, merge the files manually
instead of overwriting them.

For the standalone preprocessing, annotation, trajectory-generation, and metric
utilities in this repository, install the lightweight dependencies:

```bash
cd /path/to/HUGE-Bench
python -m pip install -r requirements-utils.txt
```

The renderer still needs the Gaussian Splatting environment, and PI0 training or
inference still needs the OpenPI environment.

## 2. Public Assets

Download the 3DGS-Mesh environments, LeRobot trajectories, and checkpoint:

```bash
export HUGE_DATA_ROOT=/path/to/HUGE_data
export HUGE_DATA_3D_ROOT=$HUGE_DATA_ROOT/data_3d
export HF_LEROBOT_HOME=/path/to/lerobot_datasets
export HUGE_PI0_DIR=/path/to/checkpoints/HUGE_PI0

mkdir -p "$HUGE_DATA_3D_ROOT" "$HF_LEROBOT_HOME" "$HUGE_PI0_DIR"

huggingface-cli download yu781986168/3DGS_Mesh_Envs \
  --repo-type dataset \
  --local-dir /path/to/downloads/3DGS_Mesh_Envs

huggingface-cli download yu781986168/HUGE_Dataset_v0 \
  --repo-type dataset \
  --local-dir "$HF_LEROBOT_HOME/task_overall"

huggingface-cli download yu781986168/HUGE_PI0 \
  --local-dir "$HUGE_PI0_DIR"
```

Extract the environment archives from `3DGS_Mesh_Envs` so that each environment
has this layout:

```text
$HUGE_DATA_3D_ROOT/
|-- 1_office/
|   |-- 3dgs_ply/point_cloud_utm50.ply
|   `-- terra_ply/simplified_mesh.obj
|-- 2_city/
|-- 3_road/
|-- 4_lake/
|-- no1_building/
|-- no3_door/
`-- overhead_bridge/
```

Overlay the released metadata and coordinate annotations from this repository:

```bash
cp -r /path/to/HUGE-Bench/trajectory_generation/scene_annotations/data_3d/* \
  "$HUGE_DATA_3D_ROOT/"
```

The annotation overlay adds files such as
`BlocksExchangeUndistortAT_WithoutTiePoints.xml`, `3dgs_ply/metadata.xml`,
`terra_ply/metadata.xml`, `location_gen/landmark_merged*.txt`,
`building_coords/*.txt`, `road_coords/*.txt`, and `farm_coords/*.txt`. These
small files are required by the trajectory generators and some debug utilities;
the large point clouds and meshes come from the Hugging Face environment
release.

The released LeRobot dataset is downloaded under
`$HF_LEROBOT_HOME/task_overall/{train,test_seen,test_unseen}`. This matches the
`pi0_overall` config and the rollout script's default repo ids for
`--task_id overall`.

The released checkpoint directory should contain:

```text
$HUGE_PI0_DIR/
|-- params/
|-- assets/
`-- train_state/
```

Pass `$HUGE_PI0_DIR` as `--checkpoint_dir`; the rollout script also accepts a
parent directory containing numeric checkpoint subfolders and will pick the
largest step.

## 3. Rollout Inference

Start the render server from the Gaussian Splatting checkout:

```bash
cd /path/to/gaussian-splatting

CUDA_VISIBLE_DEVICES=0 python 3dgs_renderer.py \
  --host 127.0.0.1 \
  --port 5550 \
  --ply_template "$HUGE_DATA_3D_ROOT/{env_id}/3dgs_ply/point_cloud_utm50.ply"
```

Run PI0 rollout from the OpenPI checkout. For the released overall dataset:

```bash
cd /path/to/openpi

CUDA_VISIBLE_DEVICES=1 uv run scripts/action_infer.py \
  --task_id overall \
  --config_name pi0_overall \
  --checkpoint_dir "$HUGE_PI0_DIR" \
  --splits test_seen,test_unseen \
  --out_dir "$HUGE_DATA_ROOT/rollout_outputs" \
  --host 127.0.0.1 \
  --port 5550
```

For a task-specific regenerated dataset, use the matching task id and config,
for example `--task_id obstacle --config_name pi0_obstacle`. The script then
expects LeRobot split directories such as
`$HF_LEROBOT_HOME/task_obstacle/test_seen` unless you pass `--repo_id`
explicitly.

## 4. Evaluation

Evaluate saved rollouts from this repository:

```bash
cd /path/to/HUGE-Bench

python metric.py \
  --out_dir "$HUGE_DATA_ROOT/rollout_outputs" \
  --mesh_root "$HUGE_DATA_3D_ROOT" \
  --mesh_rel terra_ply/simplified_mesh.obj \
  --tcr_thresholds 1,2,5
```

The script reports average TCR, nDTW, NSP, CR, and CSPL. If `--mesh_root` is
omitted, it still reports trajectory metrics but leaves collision-based metrics
as `nan`.

## 5. Training PI0

Use the OpenPI checkout with the released config file copied into
`src/openpi/training/config.py`.

Before training, check three local paths in the config:

- `repo_id`, for example `task_overall/train` or `task_obstacle/train`.
- The base PI0 weight loader path, replacing `/path/to/pi0_base/params`.
- `assets_base_dir` and `checkpoint_base_dir`, if you want them outside the
  OpenPI checkout.

Then compute dataset normalization statistics:

```bash
cd /path/to/openpi
uv run scripts/compute_norm_stats.py --config-name pi0_drone
```

This writes the normalization assets consumed by the PI0 data config. Replace
`pi0_drone` with the exact config name you train if you use `pi0_overall` or a
task-specific config. After that step, train with the same config name using the
official OpenPI training command.

## 6. Generating New Trajectories

The released trajectory pipeline is documented in
[`trajectory_generation/README.md`](../trajectory_generation/README.md). It
contains task-specific generators, 3DGS rendering scripts, instruction split
helpers, and LeRobot conversion scripts.

The minimum generated-data loop is:

```bash
export HUGE_DATA_ROOT=/path/to/HUGE_data
export HUGE_DATA_3D_ROOT=$HUGE_DATA_ROOT/data_3d
export HUGE_DATA_TRAJ_ROOT=$HUGE_DATA_ROOT/data_traj
export HUGE_LEROBOT_ROOT=/path/to/lerobot_output

python trajectory_generation/scripts/generate/traj_gen_hl.py --env_id 1_office

CUDA_VISIBLE_DEVICES=0 python trajectory_generation/scripts/render/my_render_traj_overall.py \
  --data_root "$HUGE_DATA_ROOT" \
  --env_id 1_office \
  --task_id hl \
  --poses_txt_name traj_random.txt

python trajectory_generation/scripts/convert/build_instruction_splits.py \
  --data_root "$HUGE_DATA_TRAJ_ROOT" \
  --task_id hl \
  --env_ids 1_office,2_city,3_road,4_lake \
  --unseen_env_ids 4_lake

uv run trajectory_generation/scripts/convert/convert_and_merge.py \
  --data_root "$HUGE_DATA_TRAJ_ROOT" \
  --task_id hl \
  --repo_id_prefix task_hl
```

## 7. Building New 3DGS-Mesh Scenes

The aligned 3DGS-Mesh construction pipeline is documented in
[`aligned_3dgs_mesh/README.md`](../aligned_3dgs_mesh/README.md). The recommended
workflow is to collect raw aerial images with a DJI drone, reconstruct with DJI
Terra, export both 3DGS PLY blocks and mesh OBJ blocks, merge them, convert the
3DGS coordinates into the mesh-local metric frame, and simplify the mesh for
collision checks.

The final scene should expose the same two files consumed by the benchmark:

```text
data_3d/<env_id>/
|-- 3dgs_ply/point_cloud_utm50.ply
`-- terra_ply/simplified_mesh.obj
```

If you annotate buildings, roads, regions, or landmarks in DJI Terra, convert
those coordinates to the same local frame before using the trajectory
generators.

## Reproduction Checklist

Before running a full benchmark, verify:

- `data_3d/<env_id>/3dgs_ply/point_cloud_utm50.ply` exists for every evaluated environment.
- `data_3d/<env_id>/terra_ply/simplified_mesh.obj` exists for collision metrics.
- The annotation overlay from `trajectory_generation/scene_annotations/data_3d` has been copied.
- `$HF_LEROBOT_HOME/task_overall/test_seen` and `test_unseen` contain LeRobot `meta/`, `data/`, and video/image assets.
- `$HUGE_PI0_DIR/params` exists.
- The render server is running before `action_infer.py`.
- `metric.py` receives the same `--mesh_root` that contains the evaluated environments.

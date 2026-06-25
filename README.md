<div align="center">

# [ECCV 2026] HUGE-Bench: A Benchmark for High-Level UAV Vision-Language-Action Tasks

[![Project Page](https://img.shields.io/badge/Project-Page-2d7ff9?style=for-the-badge&logo=googlechrome&logoColor=white)](https://jingyu198.github.io/HUGE_Bench/)
[![Paper](https://img.shields.io/badge/Paper-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.19822)

</div>

<p align="center">
  <img src="overview.png" alt="HUGE-Bench overview" width="100%" />
</p>

## Overview

HUGE-Bench targets high-level UAV vision-language-action tasks, where agents ground brief, potentially ambiguous commands into safe, multi-stage behaviors. HUGE-Bench contains 4 real-world scenes, 8 high-level tasks, and 2.56M meters of trajectories. It is built on aligned 3DGS-Mesh representations that combines photorealistic rendering with collision-capable geometry, enabling scalable data generation and collision-aware evaluation.

We open-source HUGE-Bench to provide the community with a UAV simulation platform that is easy to configure, high-fidelity, and collision-aware. We hope researchers can build on this benchmark to train and evaluate UAV VLA tasks in a practical and reproducible setting.

## ToDo

- ✅ Release `HUGE_Trajectory`, including all trajectory data (train and test) and one 3DGS environment (for debug the pipeline).
- ✅ Release checkpoint.
- ✅ Release `HUGE_Environment`, including 4 3DGS-Mesh environments (and 3 more finely reconstructed subregions for low-altitude forward obstacle avoidance).
- ✅ Release trajectory collection scripts, including RGB, depth, subtask, and instruction generation.
- ✅ Release the 3DGS-Mesh construction pipeline

## Dataset

[`HUGE_Trajectory`](https://huggingface.co/datasets/yu781986168/HUGE_Dataset_v0) is released in LeRobot format, it can be used directly with `pi0` training pipeline.

[`HUGE_Environment`](https://huggingface.co/datasets/yu781986168/3DGS_Mesh_Envs) includes all 3DGS-Mesh environments.

For a complete local setup, including the expected data layout, external
Gaussian Splatting/OpenPI checkouts, checkpoint paths, and evaluation commands,
see the [reproduction guide](docs/reproduction.md).


| Task ID | Task |
| --- | --- |
| `0` | Landing |
| `hl` | Orbit-H |
| `orbit` | Orbit-R |
| `building` | Inspection-B |
| `road` | Inspection-R |
| `farm` | Mapping |
| `obstacle` | Traversal |
| `orbit_multi` | Spiral Down |

Please refer to the paper for the detailed task definitions.

## Training with PI0

Our checkpoint is at [HUGE_PI0](https://huggingface.co/yu781986168/HUGE_PI0). You can also train PI0 using your own data:

Please set up the training environment by following the official [OpenPI repository](https://github.com/Physical-Intelligence/openpi). Then you can train on `HUGE_Trajectory`, and the dataset already follows the LeRobot format expected by the pipeline.

1. Copy `drone_policy.py` from this repository to `openpi/src/openpi/policies/drone_policy.py`.
   It defines the data mapping from the UAV environment to the model and back, and is used for both training and inference.
2. Replace `openpi/src/openpi/training/config.py` with the `config.py` provided in this repository.
   It defines the fine-tuning hyperparameters, data config, and weight loader for UAV training.
3. Update the config paths for your local setup.
   Set the LeRobot `repo_id` to the dataset you are training on, such as `task_overall/train` or `task_obstacle/train`, and replace `/path/to/pi0_base/params` with the PI0 base checkpoint `params` directory.
4. Do not forget to compute the dataset normalization statistics before training:
   `uv run scripts/compute_norm_stats.py --config-name pi0_drone`.
   This calculates the normalization statistics for the selected LeRobot dataset and writes the assets consumed by the training config. Replace `pi0_drone` with the exact config name you train if you use `pi0_overall` or a task-specific config.




## 3DGS-Based Environment

For 3DGS-based rendering (used for RGB collection and model inference), please set up the official [Gaussian Splatting repository](https://github.com/graphdeco-inria/gaussian-splatting) first:

Note:
1. Copy `gaussian_splatting/3dgs_renderer.py` into the official `gaussian-splatting/` project as the render-server entrypoint.
2. Copy `gaussian_splatting/my_render_traj.py` into the official `gaussian-splatting/` project for trajectory rendering.
3. Merge or replace `gaussian_splatting/utils/graphics_utils.py` with the corresponding file in `gaussian-splatting/utils/`.
4. Copy `openpi/scripts/action_infer.py` into `openpi/scripts/` as the rollout entrypoint.
5. `metric.py` is provided at the repository root for trajectory evaluation.

## Inference and Evaluation

Start the 3DGS render server in the Gaussian Splatting environment:

```bash
CUDA_VISIBLE_DEVICES=0 python 3dgs_renderer.py \
  --host 127.0.0.1 \
  --port 5550 \
  --ply_template /path/to/3dgs_root/{env_id}/3dgs_ply/point_cloud_utm50.ply
```

Then run rollout inference in the OpenPI environment:

```bash
CUDA_VISIBLE_DEVICES=1 uv run scripts/action_infer.py \
  --task_id obstacle \
  --config_name pi0_obstacle \
  --checkpoint_dir /path/to/checkpoint \
  --out_dir /path/to/rollout_outputs \
  --host 127.0.0.1 \
  --port 5550
```

For the released overall benchmark dataset, place the downloaded LeRobot split
folders under `$HF_LEROBOT_HOME/task_overall/` and run with
`--task_id overall --config_name pi0_overall`; the script then resolves split
repo ids such as `task_overall/test_seen`. For task-specific datasets generated
by the released scripts, use the matching task id such as `obstacle`,
`building`, or `hl`.

You will likely need to adapt dataset paths, checkpoint paths, and rendering templates to your local setup. The checkpoint directory should point to a directory containing `params/`, such as the downloaded [HUGE_PI0](https://huggingface.co/yu781986168/HUGE_PI0) model root.

Then evaluate the saved trajectories:

```bash
python metric.py \
  --out_dir /path/to/rollout_outputs \
  --mesh_root /path/to/data_3d \
  --mesh_rel terra_ply/simplified_mesh.obj \
  --tcr_thresholds 1,2,5
```

The evaluation script reports the main HUGE-Bench metrics used in the paper: average TCR, nDTW, NSP, CR, and CSPL. If `--mesh_root` is omitted, the script still reports trajectory metrics but leaves collision-based metrics (`CR` and `CSPL`) as `nan`.


## Output Structure

The rollout script saves results by task, split, environment, and episode:

```text
<out_dir>/
└── task_<task_token>/
    └── <split>/
        └── <env_id>/
            └── episode_<episode_index>/
                ├── compare_gt_vs_pred_3d.png
                ├── traj_gt_pred_xyzk.npz
                ├── instruction.txt
                └── gt_pred_side_by_side.mp4
```

Where:

- `compare_gt_vs_pred_3d.png` visualizes the predicted and ground-truth trajectories in 3D.
- `traj_gt_pred_xyzk.npz` stores the aligned trajectory arrays, including `gt_xyzk` and `pred_xyzk`.
- `instruction.txt` records the prompt, split, environment id, checkpoint, and evaluation settings used for that episode.
- `gt_pred_side_by_side.mp4` shows the rendered ground-truth trajectory and predicted trajectory side by side for quick visual comparison.

## Aligned 3DGS-Mesh Construction

The 3D environment construction pipeline is released under [`aligned_3dgs_mesh/`](aligned_3dgs_mesh/README.md). It documents the recommended fast collection workflow: capture raw aerial images with a DJI drone, reconstruct in DJI Terra, export 3DGS point-cloud blocks and mesh blocks, then align both products into one local metric coordinate frame.

The released utilities include:

- DJI Terra 3DGS PLY block merging;
- DJI Terra OBJ mesh block merging;
- ENU-to-projected local coordinate conversion for 3DGS point clouds;
- mesh simplification for collision and depth queries;
- optional DJI Terra coordinate annotation conversion;
- an alignment sanity checker for the final point cloud and mesh.

Basic workflow:

```bash
export SCENE_ROOT=/path/to/scene_export

python aligned_3dgs_mesh/scripts/merge_3dgs_blocks.py \
  --input-dir "$SCENE_ROOT/3dgs_ply" \
  --output "$SCENE_ROOT/3dgs_ply/merged_3dgs.ply"

python aligned_3dgs_mesh/scripts/merge_terra_mesh_blocks.py \
  --input-dir "$SCENE_ROOT/terra_ply" \
  --output "$SCENE_ROOT/terra_ply/merged_mesh.obj"

python aligned_3dgs_mesh/scripts/convert_enu_ply_to_utm.py \
  --input-ply "$SCENE_ROOT/3dgs_ply/merged_3dgs.ply" \
  --output-ply "$SCENE_ROOT/3dgs_ply/point_cloud_utm50.ply" \
  --source-metadata "$SCENE_ROOT/3dgs_ply/metadata.xml" \
  --target-metadata "$SCENE_ROOT/terra_ply/metadata.xml"

python aligned_3dgs_mesh/scripts/simplify_mesh.py \
  --input "$SCENE_ROOT/terra_ply/merged_mesh.obj" \
  --output "$SCENE_ROOT/terra_ply/simplified_mesh.obj" \
  --ratio 0.05
```

The expected outputs are `3dgs_ply/point_cloud_utm50.ply` for RGB rendering and `terra_ply/simplified_mesh.obj` for collision-aware trajectory generation and evaluation.

## Generate Your Own Trajectory Dataset

The trajectory collection pipeline is released under [`trajectory_generation/`](trajectory_generation/README.md). It includes:

- task-specific trajectory generators for landing, orbiting, inspection, mapping, and traversal tasks;
- 3DGS-based RGB rendering and optional mesh-depth rendering scripts;
- helpers for subtask visualization, LeRobot conversion, and train/test split merging.

The generation scripts expect the public 3DGS-Mesh assets plus the small scene metadata and annotation files released under [`trajectory_generation/scene_annotations/`](trajectory_generation/scene_annotations/). The basic workflow is:

```bash
export HUGE_DATA_ROOT=/path/to/HUGE_data
export HUGE_DATA_3D_ROOT=$HUGE_DATA_ROOT/data_3d
export HUGE_DATA_TRAJ_ROOT=$HUGE_DATA_ROOT/data_traj
export HUGE_LEROBOT_ROOT=/path/to/lerobot_output

python -m pip install -r requirements-utils.txt
cp -r trajectory_generation/scene_annotations/data_3d/* "$HUGE_DATA_3D_ROOT/"

python trajectory_generation/scripts/generate/traj_gen_hl.py --env_id 1_office

CUDA_VISIBLE_DEVICES=0 python trajectory_generation/scripts/render/my_render_traj_overall.py \
  --data_root "$HUGE_DATA_ROOT" \
  --env_id 1_office \
  --task_id hl \
  --poses_txt_name traj_random.txt

uv run trajectory_generation/scripts/convert/convert_and_merge.py \
  --data_root "$HUGE_DATA_TRAJ_ROOT" \
  --task_id hl \
  --repo_id_prefix task_hl
```

## Acknowledgements

- [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
- [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

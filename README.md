<div align="center">

# HUGE-Bench: A Benchmark for High-Level UAV Vision-Language-Action Tasks

<h2>[ECCV 2026]</h2>

[![Project Page](https://img.shields.io/badge/Project-Page-2d7ff9?style=for-the-badge&logo=googlechrome&logoColor=white)](https://jingyu198.github.io/HUGE_Bench/)
[![Paper](https://img.shields.io/badge/Paper-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.19822)

</div>

<p align="center">
  <img src="overview.png" alt="HUGE-Bench overview" width="100%" />
</p>

## Overview

HUGE-Bench targets high-level UAV vision-language-action tasks, where agents must ground brief, potentially ambiguous commands into safe, multi-stage behaviors. HUGE-Bench contains 4 real-world digital twin scenes, 8 high-level tasks, and 2.56M meters of trajectories. It is built on an aligned 3DGS-Mesh representation that combines photorealistic rendering with collision-capable geometry, enabling scalable data generation and collision-aware evaluation.

We open-source HUGE-Bench to provide the community with a UAV simulation learning platform that is easy to configure, high-fidelity, and collision-aware. We hope researchers can build on this benchmark to train and evaluate UAV VLA tasks in a practical and reproducible setting.

## ToDo

- ✅ Release `HUGE_Trajectory`, including all trajectory data (train and test) and one 3DGS environment (for test).
- ✅ Release checkpoint.
- ✅ Release `HUGE_Environment`, including all 3DGS-Mesh environments (four environments and three refined smaller environments used for low-altitude forward obstacle-avoidance tasks).
- [ ] Release trajectory collection scripts, including RGB, depth, subtask, and instruction collection and generation.

## Dataset

[`HUGE_Trajectory`](https://huggingface.co/datasets/yu781986168/HUGE_Dataset_v0) is released in LeRobot format, it can be used directly with `pi0` training pipeline.

[`HUGE_Environment`](https://huggingface.co/datasets/yu781986168/3DGS_Mesh_Envs) includes all 3DGS-Mesh environments.


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

Please set up the training environment by following the official [OpenPi repository](https://github.com/Physical-Intelligence/openpi).

Once the OpenPi environment is ready, you can train directly on `HUGE_Trajetory` because the dataset already follows the LeRobot format expected by the pipeline.

1. Set up the `pi0` / `OpenPi` environment by following the official OpenPi installation instructions.
2. Copy `drone_policy.py` from this repository to `openpi/src/openpi/policies/drone_policy.py`.
   It defines the data mapping from the UAV environment to the model and back, and is used for both training and inference.
3. Replace `openpi/src/openpi/training/config.py` with the `config.py` provided in this repository.
   It defines the fine-tuning hyperparameters, data config, and weight loader for UAV training.

Our checkpoint is at [HUGE_PI0](https://huggingface.co/yu781986168/HUGE_PI0).


## 3DGS-Based Environment

For 3DGS-based rendering and inference, please set up the official [Gaussian Splatting repository](https://github.com/graphdeco-inria/gaussian-splatting) first:

Note:
1. `3dgs_renderer.py` should be used inside the `gaussian-splatting/` project as the render-server entrypoint.
2. `action_infer.py` should be used inside the `openpi/scripts/` directory as the rollout entrypoint.
3. `metric.py` is provided at the repository root for trajectory evaluation.

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

You will likely need to adapt dataset paths, checkpoint paths, and rendering templates to your local setup.

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

## Acknowledgements

- [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
- [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

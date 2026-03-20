<div align="center">

# HUGE-Bench: A Benchmark for High-Level UAV Vision-Language-Action Tasks

![Project Page](https://img.shields.io/badge/Project-Page-2d7ff9?style=for-the-badge&logo=googlechrome&logoColor=white)
![Paper](https://img.shields.io/badge/Paper-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)

Official repository for the HUGE-Bench benchmark and the `HUGE_PI` dataset.

</div>

<p align="center">
  <img src="overview.png" alt="HUGE-Bench overview" width="100%" />
</p>

## Overview

Existing UAV vision-language navigation benchmarks mainly emphasize long, step-wise route descriptions and goal-centric evaluation. HUGE-Bench instead targets high-level UAV vision-language-action tasks, where agents must ground brief, potentially ambiguous commands into safe, multi-stage behaviors.

HUGE-Bench contains 4 real-world digital twin scenes, 8 high-level tasks, and 2.56M meters of trajectories. It is built on an aligned 3DGS-Mesh representation that combines photorealistic rendering with collision-capable geometry, enabling scalable data generation and collision-aware evaluation.

We further introduce process-oriented and collision-aware metrics to measure process fidelity, terminal accuracy, and safety. Experiments with representative state-of-the-art VLA models reveal clear gaps in high-level semantic completion and safe execution, making HUGE-Bench a useful diagnostic benchmark for high-level UAV autonomy.

## Contributions

- We introduce an HL-VLA benchmark setting where UAVs must interpret short commands and execute multi-stage semantic behaviors.
- We build a real-to-sim benchmark from real-world scenes with an aligned 3DGS-Mesh digital twin for scalable generation and realistic safety-aware evaluation.
- We propose process-oriented and collision-aware metrics covering process fidelity, terminal accuracy, and safety.

Dataset release: [yu781986168/HUGE_PI](https://huggingface.co/datasets/yu781986168/HUGE_PI)

Note: the dataset upload is currently in progress on Hugging Face.

## What Is Included

- A Hugging Face dataset link for `HUGE_PI`
- Documentation for training with `pi0` / `openpi`
- 3DGS helper scripts for rendering and rollout inference
- A repository layout that mirrors where the helper files should be placed in the upstream codebases

## Dataset

`HUGE_PI` is released in LeRobot format, so it can be used directly with the `pi0` training pipeline.

- Dataset: [yu781986168/HUGE_PI](https://huggingface.co/datasets/yu781986168/HUGE_PI)
- Format: `LeRobot`
- Recommended training codebase: [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)

## Training With pi0

Please set up the training environment by following the official `openpi` repository:

- `openpi`: [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)

Once the `openpi` environment is ready, users can train directly on `HUGE_PI` because the dataset already follows the LeRobot format expected by the pipeline.

1. Set up the `openpi` / `pi0` environment from the official repository.
2. Point the training pipeline to the `HUGE_PI` dataset on Hugging Face.
3. Train or fine-tune your `pi0` checkpoint as usual.

This repository does not replace the official `openpi` installation instructions. It is meant to provide the dataset release and the helper scripts needed for our workflow.

## Gaussian Splatting Environment

For 3DGS-based rendering and inference, please set up the official Gaussian Splatting repository first:

- `gaussian-splatting`: [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

Follow the official environment setup from that repository before using the helper scripts included here.

## Helper Script Placement

The helper scripts in this repository are stored in paths that mirror where they should be placed in the upstream projects:

- `gaussian_splatting/3dgs_renderer.py` -> `<gaussian-splatting-root>/3dgs_renderer.py`
- `gaussian_splatting/my_render_traj.py` -> `<gaussian-splatting-root>/my_render_traj.py`
- `openpi/scripts/action_infer.py` -> `<openpi-root>/scripts/action_infer.py`

`3dgs_renderer.py` and `my_render_traj.py` depend on the Gaussian Splatting codebase and should be used inside the `gaussian-splatting` project. `action_infer.py` depends on `openpi` and should be used inside the `openpi/scripts/` directory.

## Minimal Workflow

1. Clone and configure `openpi` by following the official repository.
2. Clone and configure `gaussian-splatting` by following the official repository.
3. Copy the helper scripts from this repository into the matching locations in those two codebases.
4. Train your model with `HUGE_PI`.
5. After training, run inference on our platform. In this initial open-source release, we provide a 3DGS-based inference path.

## Example Setup

```bash
git clone https://github.com/Physical-Intelligence/openpi.git
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git
```

Then copy the helper files into the corresponding locations:

```text
<gaussian-splatting-root>/3dgs_renderer.py
<gaussian-splatting-root>/my_render_traj.py
<openpi-root>/scripts/action_infer.py
```

## Example Inference Flow

Start the 3DGS render server in the Gaussian Splatting environment:

```bash
python 3dgs_renderer.py --host 127.0.0.1 --port 5550
```

Then run rollout inference in the OpenPI environment:

```bash
python scripts/action_infer.py \
  --task_id obstacle \
  --config_name pi0_obstacle \
  --checkpoint_dir /path/to/checkpoint \
  --host 127.0.0.1 \
  --port 5550
```

You will likely need to adapt dataset paths, checkpoint paths, and rendering templates to your local setup.

## Repository Layout

```text
HUGE-Bench/
|-- overview.png
|-- README.md
|-- gaussian_splatting/
|   |-- 3dgs_renderer.py
|   `-- my_render_traj.py
`-- openpi/
    `-- scripts/
        `-- action_infer.py
```

## Inference Availability

After training is completed, users can run inference on our platform. For this first public release, we open-source:

- a 3DGS-based renderer
- a 3DGS environment integration path
- an `openpi` inference entry script

## Acknowledgements

- [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
- [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [Hugging Face Datasets](https://huggingface.co/datasets)

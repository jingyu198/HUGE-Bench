# Pi0.5 Training on HUGE-Bench

This page describes the reference fine-tuning setup for the released
`task_overall/train` split. It uses the HUGE-Bench data pipeline, including
`MyDroneDataConfig` and `drone_policy.py`; the upstream OpenPI Pi0.5 example
alone is not sufficient for this dataset.

## 1. Prepare OpenPI

Use the OpenPI checkout and Python environment described by the official
[OpenPI repository](https://github.com/Physical-Intelligence/openpi). Copy the
following files from HUGE-Bench into that checkout:

```text
HUGE-Bench/openpi/src/openpi/policies/drone_policy.py
HUGE-Bench/openpi/src/openpi/training/config.py
```

Use the same OpenPI revision as the code in this repository. Mixing the
HUGE-Bench files with an unrelated OpenPI revision can change model fields,
data transforms, or checkpoint loading behavior.

## 2. Prepare the data

Download `HUGE_Dataset_v0` and make the overall training split available under
the LeRobot home directory. For example:

```bash
export HF_LEROBOT_HOME=/path/to/lerobot_data
```

The config expects the split repo id:

```text
task_overall/train
```

Do not replace this with an arbitrary parent directory unless that directory
has the LeRobot dataset layout expected by `LeRobotDataset`.

## 3. Compute normalization statistics

From the OpenPI checkout, run:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_overall
```

The command creates the assets for `pi05_overall`, including the normalization
statistics for `task_overall/train`. The released [HUGE_PI05 checkpoint](https://huggingface.co/yu781986168/HUGE_PI05)
also contains the normalization asset used by the trained model.

## 4. Reference config

The reproducible config is `pi05_overall` in
`openpi/src/openpi/training/config.py`:

```python
TrainConfig(
    name="pi05_overall",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=20,
        discrete_state_input=False,
    ),
    data=MyDroneDataConfig(
        repo_id="task_overall/train",
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "/path/to/pi05_base/params"
    ),
    num_train_steps=25000,
    batch_size=512,
    save_interval=5000,
    fsdp_devices=8,
)
```

The reference run used a global batch size of 512, or 64 samples per GPU on
8 GPUs. `batch_size` is global in OpenPI; it should not be set to 512 per GPU.

The `weight_loader` path must point to the `params` directory of the official
Pi0.5 base checkpoint. The `pytorch_weight_path` field is for a PyTorch
checkpoint and should be left unset when using the OpenPI `params` checkpoint.
Likewise, do not set a custom `AssetsConfig(asset_id=...)` unless the matching
normalization asset has been generated at that exact asset path.

## 5. Start training

Run the official OpenPI training command with the same config name:

```bash
cd /path/to/openpi
uv run scripts/train.py pi05_overall \
  --exp-name task_overall \
  --overwrite \
  --num-workers 8
```

The final released checkpoint is available at
[yu781986168/HUGE_PI05](https://huggingface.co/yu781986168/HUGE_PI05). It is
provided for inference and evaluation; it does not include the large
training-resumption `train_state` directory.

## 6. Reproduction notes

Training quality can still vary with the OpenPI revision, GPU count, effective
global batch size, data version, normalization assets, and random seed. Before
comparing results, verify that the data loader reports the expected dataset
split and that the normalization asset is loaded rather than skipped.

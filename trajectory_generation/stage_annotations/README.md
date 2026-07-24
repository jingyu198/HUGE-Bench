# Multi-Stage Annotations

This directory releases multi-stage trajectory annotations and an episode
mapping for the public
[`HUGE_Dataset_v0`](https://huggingface.co/datasets/yu781986168/HUGE_Dataset_v0)
LeRobot dataset.

The current LeRobot parquet files do not contain a `subtask_phase` column. The
files here are a lightweight sidecar release: they preserve the available
original trajectory-level `subtask.txt` annotations and map stages to the final
`task_overall/<split>` episode indices without changing the existing dataset.

## Contents

```text
stage_annotations/
|-- raw_subtasks/task_<task_id>/<env_id>/subtask.txt
|-- episode_mapping/{train,test_seen,test_unseen}.jsonl
|-- stage_segments/{train,test_seen,test_unseen}.jsonl
`-- manifest.json
```

The release covers all 6,168 episodes in the public overall dataset: 5,175
train, 576 test-seen, and 417 test-unseen episodes. It contains 27,539 stage
segments and 20 original `subtask.txt` files.

Each original `subtask.txt` row has the following whitespace-separated format:

```text
traj_id subtask_id pose_id_start pose_id_end subtask_text
```

Each `episode_mapping/<split>.jsonl` row maps a released overall LeRobot episode
to its source trajectory. Important fields are:

- `episode_index`: episode index in `HUGE_Dataset_v0/<split>`;
- `task_id` and `task_episode_index`: source task and its task-level episode index;
- `env_id` and `traj_id`: source environment and trajectory;
- `pose_start`, `pose_end`, and `length`: inclusive source pose range;
- `subtask_file`: path to the corresponding original annotation file.

Each `stage_segments/<split>.jsonl` row is one stage intersected with one
released episode. `pose_start` and `pose_end` are inclusive global pose ids;
`frame_start` and `frame_end` are inclusive zero-based offsets inside that
LeRobot episode.

`annotation_provenance` distinguishes two cases:

- `original_raw`: the released episode was matched to the original
  `subtask.txt`, including its source `traj_id` and global pose range;
- `recovered_from_released_actions`: used for the obstacle task. Its raw source
  trajectories were regenerated after the LeRobot release, so the old global
  pose ids are no longer available. The turn/fly boundary is recovered from
  the released action sequence, and unavailable `traj_id`/pose fields are
  `null`. This is kept explicit rather than linking an episode to a newer,
  incompatible raw trajectory.

Example lookup:

```python
import json
from pathlib import Path

root = Path("trajectory_generation/stage_annotations")
split = "train"
episode_index = 0

with (root / "episode_mapping" / f"{split}.jsonl").open(encoding="utf-8") as f:
    episode = next(row for row in map(json.loads, f) if row["episode_index"] == episode_index)

with (root / "stage_segments" / f"{split}.jsonl").open(encoding="utf-8") as f:
    stages = [row for row in map(json.loads, f) if row["episode_index"] == episode_index]

print(episode)
print(stages)
```

To construct a per-frame phase array, assign each segment's `subtask_id` to the
inclusive range `frame_start:frame_end + 1`. The segments for an episode are
validated to be contiguous, non-overlapping, and to cover its complete
`0:length` frame range.

The mappings are generated and strictly checked by
[`export_stage_annotations.py`](../scripts/convert/export_stage_annotations.py).
For every released episode, the exporter verifies task id, environment,
instruction, episode length, final overall episode order, and stage coverage.

These annotations are provided as additional supervision and were not required
by the released PI0 baseline, which was trained with high-level instructions.

#!/usr/bin/env python3
"""Export raw subtask annotations and mappings for the released LeRobot data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_TASK_IDS = ("0", "hl", "building", "farm", "obstacle", "orbit", "road", "orbit_multi")
DEFAULT_SPLITS = ("train", "test_seen", "test_unseen")


@dataclass(frozen=True)
class EpisodeSource:
    env_id: str
    traj_id: int
    pose_start: int
    pose_end: int
    instruction: str

    @property
    def length(self) -> int:
        return self.pose_end - self.pose_start + 1


@dataclass(frozen=True)
class Stage:
    subtask_id: int
    pose_start: int
    pose_end: int
    text: str


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def parse_instruction_file(path: Path) -> list[EpisodeSource]:
    episodes: list[EpisodeSource] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=4)
            if len(parts) != 5:
                raise ValueError(f"Malformed instruction row at {path}:{line_number}: {line}")
            episodes.append(
                EpisodeSource(
                    env_id=parts[0],
                    traj_id=int(parts[1]),
                    pose_start=int(parts[2]),
                    pose_end=int(parts[3]),
                    instruction=parts[4],
                )
            )
    return episodes


def parse_subtask_file(path: Path) -> dict[int, list[Stage]]:
    stages: dict[int, list[Stage]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=4)
            if len(parts) != 5:
                raise ValueError(f"Malformed subtask row at {path}:{line_number}: {line}")
            traj_id = int(parts[0])
            stages.setdefault(traj_id, []).append(
                Stage(
                    subtask_id=int(parts[1]),
                    pose_start=int(parts[2]),
                    pose_end=int(parts[3]),
                    text=parts[4],
                )
            )
    for traj_stages in stages.values():
        traj_stages.sort(key=lambda item: (item.pose_start, item.pose_end, item.subtask_id))
    return stages


def episode_instruction(meta_row: dict) -> str:
    tasks = meta_row.get("tasks")
    if isinstance(tasks, list) and tasks:
        return str(tasks[0])
    for key in ("task", "instruction"):
        if isinstance(meta_row.get(key), str):
            return meta_row[key]
    raise ValueError(f"Cannot read instruction from episode metadata: {meta_row}")


def source_signature(source: EpisodeSource) -> tuple[str, str, int]:
    return source.env_id, source.instruction, source.length


def meta_signature(row: dict) -> tuple[str, str, int]:
    return str(row.get("env_id", "")), episode_instruction(row), int(row["length"])


def shard_order(episodes: list[EpisodeSource], num_shards: int) -> list[EpisodeSource]:
    return [episode for shard_id in range(num_shards) for episode in episodes[shard_id::num_shards]]


def float32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def wrap_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def load_required_states(traj_path: Path, pose_ids: set[int], task_id: str) -> dict[int, tuple[float, ...]]:
    states: dict[int, tuple[float, ...]] = {}
    obstacle = task_id == "obstacle"
    with traj_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            pose_id = int(parts[0])
            if pose_id not in pose_ids:
                continue
            angle_degrees = float(parts[5] if obstacle else parts[6])
            states[pose_id] = (
                float32(float(parts[1])),
                float32(float(parts[2])),
                float32(float(parts[3])),
                float32(math.radians(wrap_degrees(angle_degrees))),
            )
    missing = pose_ids - states.keys()
    if missing:
        raise KeyError(f"Missing {len(missing)} required poses in {traj_path}; first missing={min(missing)}")
    return states


def parquet_endpoint_states(path: Path) -> tuple[tuple[float, ...], tuple[float, ...]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required only when the raw split differs from the released task dataset. "
            "Install pyarrow and rerun the exporter."
        ) from exc
    state_rows = pq.read_table(path, columns=["state"])["state"].to_pylist()
    if not state_rows:
        raise ValueError(f"Empty episode parquet: {path}")
    return tuple(float(value) for value in state_rows[0]), tuple(float(value) for value in state_rows[-1])


def recover_obstacle_stages(path: Path, expected_length: int) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to recover the released obstacle stage boundaries") from exc
    actions = pq.read_table(path, columns=["actions"])["actions"].to_pylist()
    if len(actions) != expected_length:
        raise ValueError(f"Obstacle parquet length mismatch at {path}: {len(actions)} != {expected_length}")
    fly_start = next(
        (
            index
            for index, action in enumerate(actions)
            if max(abs(float(action[0])), abs(float(action[1])), abs(float(action[2]))) > 1e-6
        ),
        None,
    )
    if fly_start is None:
        raise ValueError(f"Cannot find obstacle flight phase in {path}")
    stages: list[dict] = []
    if fly_start > 0:
        stages.append(
            {
                "subtask_id": 0,
                "pose_start": None,
                "pose_end": None,
                "frame_start": 0,
                "frame_end": fly_start - 1,
                "subtask_text": "Turn to face the target.",
            }
        )
    stages.append(
        {
            "subtask_id": 1,
            "pose_start": None,
            "pose_end": None,
            "frame_start": fly_start,
            "frame_end": expected_length - 1,
            "subtask_text": "Fly to the target while avoiding obstacles.",
        }
    )
    return stages


def states_close(left: tuple[float, ...], right: tuple[float, ...], tolerance: float = 1e-5) -> bool:
    return len(left) == len(right) and all(abs(a - b) <= tolerance for a, b in zip(left, right))


def recover_by_endpoint_states(
    raw_episodes: list[EpisodeSource],
    meta_rows: list[dict],
    allowed_envs: set[str],
    raw_task_root: Path,
    lerobot_task_split: Path,
    task_id: str,
    context: str,
) -> list[EpisodeSource]:
    # A raw split can be rewritten after the LeRobot release. Load every pose
    # when needed so released episodes can still be recovered by exact states.
    states_by_env: dict[str, dict[int, tuple[float, ...]]] = {}
    for env_id in allowed_envs:
        traj_path = raw_task_root / env_id / "traj_random.txt"
        all_pose_ids: set[int] = set()
        with traj_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line and not line.startswith("#"):
                    all_pose_ids.add(int(line.split(maxsplit=1)[0]))
        states_by_env[env_id] = load_required_states(traj_path, all_pose_ids, task_id)

    candidates_by_env_length: dict[tuple[str, int], list[EpisodeSource]] = {}
    for env_id, states in states_by_env.items():
        pose_to_traj: dict[int, tuple[int, int, int]] = {}
        subtask_path = raw_task_root / env_id / "subtask.txt"
        for traj_id, stages in parse_subtask_file(subtask_path).items():
            pose_start = min(stage.pose_start for stage in stages)
            pose_end = max(stage.pose_end for stage in stages)
            pose_to_traj[pose_start] = (traj_id, pose_start, pose_end)
        for traj_id, pose_start, pose_end in pose_to_traj.values():
            if pose_start in states and pose_end in states:
                candidates_by_env_length.setdefault((env_id, pose_end - pose_start + 1), []).append(
                    EpisodeSource(env_id, traj_id, pose_start, pose_end, "")
                )
    aligned: list[EpisodeSource] = []
    used: set[tuple[str, int, int, int]] = set()
    for episode_index, meta_row in enumerate(meta_rows):
        parquet_path = (
            lerobot_task_split
            / "data"
            / f"chunk-{episode_index // 1000:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )
        first_state, last_state = parquet_endpoint_states(parquet_path)
        matches: list[EpisodeSource] = []
        env_id = str(meta_row["env_id"])
        length = int(meta_row["length"])
        for candidate in candidates_by_env_length.get((env_id, length), []):
            identity = (candidate.env_id, candidate.traj_id, candidate.pose_start, candidate.pose_end)
            if identity in used:
                continue
            source_states = states_by_env[candidate.env_id]
            if states_close(first_state, source_states[candidate.pose_start]) and states_close(
                last_state, source_states[candidate.pose_end]
            ):
                matches.append(candidate)
        if len(matches) != 1:
            raise ValueError(
                f"{context}: endpoint-state recovery found {len(matches)} matches for task episode "
                f"{episode_index}, signature={meta_signature(meta_row)!r}"
            )
        selected = matches[0]
        selected = EpisodeSource(
            env_id=selected.env_id,
            traj_id=selected.traj_id,
            pose_start=selected.pose_start,
            pose_end=selected.pose_end,
            instruction=episode_instruction(meta_row),
        )
        used.add((selected.env_id, selected.traj_id, selected.pose_start, selected.pose_end))
        aligned.append(selected)
    print(f"[WARN] {context}: recovered {len(aligned)} released episodes by exact endpoint states")
    return aligned


def align_task_episodes(
    raw_episodes: list[EpisodeSource],
    meta_rows: list[dict],
    allowed_envs: set[str],
    num_shards: int,
    raw_task_root: Path,
    lerobot_task_split: Path,
    task_id: str,
    context: str,
) -> list[EpisodeSource]:
    candidates = shard_order([ep for ep in raw_episodes if ep.env_id in allowed_envs], num_shards)
    if len(candidates) != len(meta_rows):
        return recover_by_endpoint_states(
            raw_episodes=raw_episodes,
            meta_rows=meta_rows,
            allowed_envs=allowed_envs,
            raw_task_root=raw_task_root,
            lerobot_task_split=lerobot_task_split,
            task_id=task_id,
            context=context,
        )
    for index, (source, meta_row) in enumerate(zip(candidates, meta_rows)):
        if int(meta_row.get("episode_index", index)) != index:
            raise ValueError(f"{context}: non-contiguous task episode_index at row {index}")
        if source_signature(source) != meta_signature(meta_row):
            return recover_by_endpoint_states(
                raw_episodes=raw_episodes,
                meta_rows=meta_rows,
                allowed_envs=allowed_envs,
                raw_task_root=raw_task_root,
                lerobot_task_split=lerobot_task_split,
                task_id=task_id,
                context=context,
            )
    return candidates


def intersect_stages(source: EpisodeSource, stages: list[Stage], context: str) -> list[dict]:
    selected: list[dict] = []
    coverage = [0] * source.length
    for stage in stages:
        start = max(source.pose_start, stage.pose_start)
        end = min(source.pose_end, stage.pose_end)
        if start > end:
            continue
        frame_start = start - source.pose_start
        frame_end = end - source.pose_start
        for frame in range(frame_start, frame_end + 1):
            coverage[frame] += 1
        selected.append(
            {
                "subtask_id": stage.subtask_id,
                "pose_start": start,
                "pose_end": end,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "subtask_text": stage.text,
            }
        )
    uncovered = sum(value == 0 for value in coverage)
    multiply_covered = sum(value > 1 for value in coverage)
    if uncovered or multiply_covered:
        raise ValueError(
            f"{context}: invalid stage coverage for poses [{source.pose_start}, {source.pose_end}]: "
            f"uncovered={uncovered}, multiply_covered={multiply_covered}"
        )
    return selected


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True, help="Root containing task_<id>/ raw trajectories.")
    parser.add_argument("--lerobot-root", type=Path, required=True, help="Root containing task_<id>/ and task_overall/.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-ids", default=",".join(DEFAULT_TASK_IDS))
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--num-shards", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    task_ids = parse_csv(args.task_ids)
    splits = parse_csv(args.splits)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    task_meta: dict[tuple[str, str], list[dict]] = {}
    task_envs: dict[str, set[str]] = {task_id: set() for task_id in task_ids}
    for task_id in task_ids:
        for split in splits:
            path = args.lerobot_root / f"task_{task_id}" / split / "meta" / "episodes.jsonl"
            rows = read_jsonl(path)
            task_meta[(task_id, split)] = rows
            task_envs[task_id].update(str(row["env_id"]) for row in rows)

    subtask_cache: dict[tuple[str, str], dict[int, list[Stage]]] = {}
    raw_files: list[Path] = []
    for task_id in task_ids:
        if task_id == "obstacle":
            continue
        for env_id in sorted(task_envs[task_id]):
            source = args.raw_root / f"task_{task_id}" / env_id / "subtask.txt"
            if not source.exists():
                raise FileNotFoundError(source)
            destination = output_dir / "raw_subtasks" / f"task_{task_id}" / env_id / "subtask.txt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            raw_files.append(destination)
            subtask_cache[(task_id, env_id)] = parse_subtask_file(source)

    summary: dict[str, dict] = {}
    total_episodes = 0
    total_segments = 0
    for split in splits:
        overall_path = args.lerobot_root / "task_overall" / split / "meta" / "episodes.jsonl"
        overall_rows = read_jsonl(overall_path)
        expected_overall_index = 0
        mapping_rows: list[dict] = []
        segment_rows: list[dict] = []

        for task_id in task_ids:
            aligned: list[EpisodeSource] | None = None
            if task_id != "obstacle":
                instruction_path = (
                    args.raw_root / f"task_{task_id}" / "split_res_merged" / f"instruction_{split}.txt"
                )
                aligned = align_task_episodes(
                    raw_episodes=parse_instruction_file(instruction_path),
                    meta_rows=task_meta[(task_id, split)],
                    allowed_envs=task_envs[task_id],
                    num_shards=args.num_shards,
                    raw_task_root=args.raw_root / f"task_{task_id}",
                    lerobot_task_split=args.lerobot_root / f"task_{task_id}" / split,
                    task_id=task_id,
                    context=f"task={task_id}, split={split}",
                )

            for task_episode_index, task_row in enumerate(task_meta[(task_id, split)]):
                if expected_overall_index >= len(overall_rows):
                    raise ValueError(f"split={split}: task episodes exceed overall metadata")
                overall_row = overall_rows[expected_overall_index]
                if int(overall_row.get("episode_index", -1)) != expected_overall_index:
                    raise ValueError(f"split={split}: non-contiguous overall episode_index at {expected_overall_index}")
                if str(overall_row.get("task_id", "")) != task_id:
                    raise ValueError(
                        f"split={split}, episode={expected_overall_index}: expected task_id={task_id}, "
                        f"found {overall_row.get('task_id')!r}"
                    )
                if meta_signature(overall_row) != meta_signature(task_row):
                    raise ValueError(
                        f"split={split}, episode={expected_overall_index}: task/overall metadata mismatch"
                    )

                if task_id == "obstacle":
                    task_parquet = (
                        args.lerobot_root
                        / f"task_{task_id}"
                        / split
                        / "data"
                        / f"chunk-{task_episode_index // 1000:03d}"
                        / f"episode_{task_episode_index:06d}.parquet"
                    )
                    stages = recover_obstacle_stages(task_parquet, int(task_row["length"]))
                    source = None
                    provenance = "recovered_from_released_actions"
                else:
                    assert aligned is not None
                    source = aligned[task_episode_index]
                    traj_stages = subtask_cache[(task_id, source.env_id)].get(source.traj_id)
                    if not traj_stages:
                        raise KeyError(
                            f"No subtask annotations for task={task_id}, env={source.env_id}, traj={source.traj_id}"
                        )
                    stages = intersect_stages(
                        source,
                        traj_stages,
                        context=f"split={split}, episode={expected_overall_index}, task={task_id}, "
                        f"env={source.env_id}, traj={source.traj_id}",
                    )
                    provenance = "original_raw"

                mapping_rows.append(
                    {
                        "episode_index": expected_overall_index,
                        "task_id": task_id,
                        "task_episode_index": task_episode_index,
                        "env_id": str(task_row["env_id"]),
                        "traj_id": source.traj_id if source is not None else None,
                        "pose_start": source.pose_start if source is not None else None,
                        "pose_end": source.pose_end if source is not None else None,
                        "length": int(task_row["length"]),
                        "instruction": episode_instruction(task_row),
                        "num_stages": len(stages),
                        "annotation_provenance": provenance,
                        "subtask_file": (
                            f"raw_subtasks/task_{task_id}/{source.env_id}/subtask.txt"
                            if source is not None
                            else None
                        ),
                    }
                )
                for stage in stages:
                    segment_rows.append(
                        {
                            "episode_index": expected_overall_index,
                            "task_id": task_id,
                            "env_id": str(task_row["env_id"]),
                            "traj_id": source.traj_id if source is not None else None,
                            "annotation_provenance": provenance,
                            **stage,
                        }
                    )
                expected_overall_index += 1

        if expected_overall_index != len(overall_rows):
            raise ValueError(
                f"split={split}: mapped {expected_overall_index} episodes, overall metadata has {len(overall_rows)}"
            )
        mapping_count = write_jsonl(output_dir / "episode_mapping" / f"{split}.jsonl", mapping_rows)
        segment_count = write_jsonl(output_dir / "stage_segments" / f"{split}.jsonl", segment_rows)
        summary[split] = {"episodes": mapping_count, "stage_segments": segment_count}
        total_episodes += mapping_count
        total_segments += segment_count

    released_files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    manifest = {
        "format_version": 1,
        "source_dataset": "yu781986168/HUGE_Dataset_v0",
        "task_order": task_ids,
        "splits": summary,
        "total_episodes": total_episodes,
        "total_stage_segments": total_segments,
        "raw_subtask_files": len(raw_files),
        "provenance": {
            "original_raw": "All non-obstacle tasks.",
            "recovered_from_released_actions": "Obstacle turn/fly boundaries recovered from released actions.",
        },
        "files": [
            {
                "path": path.relative_to(output_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in released_files
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build train/test instruction files for LeRobot conversion.

Input per environment:
    data_traj/task_<task_id>/<env_id>/instruction.txt
    data_traj/task_<task_id>/<env_id>/wash_res.txt  (optional)

Output:
    data_traj/task_<task_id>/split_res_merged/instruction_train.txt
    data_traj/task_<task_id>/split_res_merged/instruction_test_seen.txt
    data_traj/task_<task_id>/split_res_merged/instruction_test_unseen.txt

Each output line has:
    env_id traj_id pose_start pose_end instruction text...
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Episode:
    env_id: str
    traj_id: int
    pose_start: int
    pose_end: int
    instruction: str

    def to_line(self) -> str:
        return f"{self.env_id} {self.traj_id} {self.pose_start} {self.pose_end} {self.instruction}"


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def read_valid_traj_ids(path: Path) -> set[int] | None:
    if not path.exists():
        return None
    ids: set[int] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            ids.add(int(text.split()[0]))
    return ids


def read_instruction_file(env_id: str, path: Path, valid_ids: set[int] | None) -> list[Episode]:
    episodes: list[Episode] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            parts = text.split(maxsplit=3)
            if len(parts) < 4:
                raise ValueError(f"Bad instruction line in {path}: {line.rstrip()}")
            traj_id = int(parts[0])
            if valid_ids is not None and traj_id not in valid_ids:
                continue
            episodes.append(
                Episode(
                    env_id=env_id,
                    traj_id=traj_id,
                    pose_start=int(parts[1]),
                    pose_end=int(parts[2]),
                    instruction=parts[3],
                )
            )
    return episodes


def discover_env_ids(task_root: Path) -> list[str]:
    env_ids = []
    for child in sorted(task_root.iterdir()):
        if child.is_dir() and (child / "instruction.txt").exists():
            env_ids.append(child.name)
    return env_ids


def write_split(path: Path, episodes: Iterable[Episode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# env_id traj_id pose_start pose_end instruction\n")
        for ep in episodes:
            f.write(ep.to_line() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build merged instruction split files from per-env generated data.")
    parser.add_argument("--data_root", default=os.environ.get("HUGE_DATA_TRAJ_ROOT", "./data_traj"))
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--env_ids", default="", help="Comma-separated env ids. If empty, auto-discovers env folders.")
    parser.add_argument(
        "--unseen_env_ids",
        default="",
        help="Comma-separated env ids to place fully into test_unseen. Others are split into train/test_seen.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dirname", default="split_res_merged")
    args = parser.parse_args()

    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train_ratio must be between 0 and 1")

    task_root = Path(args.data_root) / f"task_{args.task_id}"
    if not task_root.exists():
        raise FileNotFoundError(f"task root not found: {task_root}")

    env_ids = parse_csv(args.env_ids) or discover_env_ids(task_root)
    unseen_env_ids = set(parse_csv(args.unseen_env_ids))
    rng = random.Random(args.seed)

    train: list[Episode] = []
    test_seen: list[Episode] = []
    test_unseen: list[Episode] = []

    for env_id in env_ids:
        env_root = task_root / env_id
        instr_path = env_root / "instruction.txt"
        if not instr_path.exists():
            print(f"[WARN] skip {env_id}: missing {instr_path}")
            continue
        valid_ids = read_valid_traj_ids(env_root / "wash_res.txt")
        episodes = read_instruction_file(env_id, instr_path, valid_ids)
        rng.shuffle(episodes)

        if env_id in unseen_env_ids:
            test_unseen.extend(episodes)
            continue

        cut = int(round(len(episodes) * args.train_ratio))
        train.extend(episodes[:cut])
        test_seen.extend(episodes[cut:])

    out_dir = task_root / args.output_dirname
    write_split(out_dir / "instruction_train.txt", train)
    write_split(out_dir / "instruction_test_seen.txt", test_seen)
    write_split(out_dir / "instruction_test_unseen.txt", test_unseen)

    print(f"[OK] wrote splits under {out_dir}")
    print(f"[OK] train={len(train)} test_seen={len(test_seen)} test_unseen={len(test_unseen)}")


if __name__ == "__main__":
    main()

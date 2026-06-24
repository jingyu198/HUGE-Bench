import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import tyro


@dataclass
class Args:
    # raw data root that contains env folders and split_res_merged/
    task_root: str = os.path.join(os.environ.get("HUGE_DATA_TRAJ_ROOT", "./data_traj"), "task_building")

    # existing merged lerobot repo (read-only), e.g. <HUGE_LEROBOT_ROOT>/task_building/train
    src_repo_dir: str = os.path.join(os.environ.get("HUGE_LEROBOT_ROOT", "./lerobot_datasets"), "task_building", "train")

    # new merged lerobot repo to create, e.g. <HUGE_LEROBOT_ROOT>/task_building_low/train
    dst_repo_dir: str = os.path.join(os.environ.get("HUGE_LEROBOT_ROOT", "./lerobot_datasets"), "task_building_low", "train")

    # which split this repo corresponds to
    split: str = "train"

    # must match the num_shards used when creating the merged repo
    num_shards: int = 16

    instruction_dirname: str = "split_res_merged"
    instruction_pattern: str = "instruction_{split}.txt"
    subtask_filename: str = "subtask.txt"

    # how to concatenate subtasks into one low-level instruction
    joiner: str = " "
    strict: bool = True

    # for all non-parquet assets copied from src -> dst
    # hardlink / symlink / copy
    link_mode: str = "copy"

    # stats.json may contain task_index stats and become stale after remapping
    copy_stats_json: bool = False

    # safety: by default do NOT overwrite destination
    overwrite_dst: bool = False

    # optional: write extra debug mapping file for inspection
    write_episode_mapping_jsonl: bool = True


# =========================
# basic utils
# =========================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def deep_copy_jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x, ensure_ascii=False))


def link_or_copy(src: Path, dst: Path, mode: str = "hardlink") -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        os.symlink(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


# =========================
# source parsing
# =========================
def parse_instruction_file(path: Path) -> List[Tuple[str, int, int, int, str]]:
    rows: List[Tuple[str, int, int, int, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=4)
            if len(parts) < 5:
                raise ValueError(f"Bad instruction line: {line}")
            env_id = parts[0]
            traj_id = int(parts[1])
            pose_start = int(parts[2])
            pose_end = int(parts[3])
            instruction = parts[4]
            rows.append((env_id, traj_id, pose_start, pose_end, instruction))
    return rows


def parse_subtask_file(path: Path) -> Dict[int, List[Tuple[int, int, int, str]]]:
    """
    subtask.txt format:
      traj_id subtask_id pose_id_start pose_id_end subtask_text...
    returns:
      out[traj_id] = [(subtask_id, pose_start, pose_end, subtask_text), ...]
    """
    out: Dict[int, List[Tuple[int, int, int, str]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=4)
            if len(parts) < 5:
                raise ValueError(f"Bad subtask line: {line}")
            traj_id = int(parts[0])
            subtask_id = int(parts[1])
            pose_start = int(parts[2])
            pose_end = int(parts[3])
            subtask = parts[4]
            out.setdefault(traj_id, []).append((subtask_id, pose_start, pose_end, subtask))
    for traj_id in out:
        out[traj_id].sort(key=lambda x: (x[0], x[1], x[2]))
    return out


def build_subtasks_by_env(task_root: Path, env_ids: List[str], subtask_filename: str) -> Dict[str, Dict[int, List[Tuple[int, int, int, str]]]]:
    out: Dict[str, Dict[int, List[Tuple[int, int, int, str]]]] = {}
    for env_id in sorted(set(env_ids)):
        subtask_path = task_root / env_id / subtask_filename
        if not subtask_path.exists():
            raise FileNotFoundError(f"subtask file not found: {subtask_path}")
        out[env_id] = parse_subtask_file(subtask_path)
    return out


def compose_low_level_instruction(
    env_id: str,
    traj_id: int,
    pose_start: int,
    pose_end: int,
    high_level_instruction: str,
    subtasks_by_env: Dict[str, Dict[int, List[Tuple[int, int, int, str]]]],
    joiner: str,
    strict: bool,
) -> str:
    traj_rows = subtasks_by_env.get(env_id, {}).get(traj_id, [])
    selected: List[str] = []

    for subtask_id, st, ed, text in traj_rows:
        # keep any subtask overlapping the episode range
        if ed < pose_start or st > pose_end:
            continue
        text = text.strip()
        if text and (not selected or selected[-1] != text):
            selected.append(text)

    if not selected:
        if strict:
            raise KeyError(
                f"No subtask found for env={env_id}, traj_id={traj_id}, pose_range=[{pose_start}, {pose_end}]"
            )
        return high_level_instruction

    return joiner.join(selected)


# =========================
# merged ordering reconstruction
# =========================
def build_merged_episode_specs(
    task_root: Path,
    split: str,
    num_shards: int,
    instruction_dirname: str,
    instruction_pattern: str,
    subtask_filename: str,
    joiner: str,
    strict: bool,
) -> List[Dict[str, Any]]:
    """
    Rebuild the FINAL merged episode order from instruction_{split}.txt.

    Original pipeline order was:
      1) each shard gets episodes_all[sid::num_shards]
      2) merge concatenates shard0, shard1, ..., shard{N-1}

    We reproduce the same order here, so episode i aligns with merged repo episode_i.
    """
    instruction_path = task_root / instruction_dirname / instruction_pattern.format(split=split)
    if not instruction_path.exists():
        raise FileNotFoundError(f"instruction file not found: {instruction_path}")

    raw_eps = parse_instruction_file(instruction_path)
    subtasks_by_env = build_subtasks_by_env(task_root, [x[0] for x in raw_eps], subtask_filename)

    by_shard = [raw_eps[sid::num_shards] for sid in range(num_shards)]

    merged: List[Dict[str, Any]] = []
    for sid in range(num_shards):
        for env_id, traj_id, pose_start, pose_end, high in by_shard[sid]:
            low = compose_low_level_instruction(
                env_id=env_id,
                traj_id=traj_id,
                pose_start=pose_start,
                pose_end=pose_end,
                high_level_instruction=high,
                subtasks_by_env=subtasks_by_env,
                joiner=joiner,
                strict=strict,
            )
            merged.append(
                {
                    "env_id": env_id,
                    "traj_id": traj_id,
                    "pose_start": pose_start,
                    "pose_end": pose_end,
                    "high_level_task": high,
                    "low_level_task": low,
                }
            )
    return merged


# =========================
# parquet patching
# =========================
def make_constant_column(value: int, n: int, typ: pa.DataType = pa.int64()) -> pa.Array:
    return pa.array([value] * n, type=typ)


def patch_parquet_task_index_to_new_file(src_parquet: Path, dst_parquet: Path, new_task_index: int) -> None:
    table = pq.read_table(src_parquet)
    if table.num_rows > 0 and "task_index" in table.column_names:
        idx = table.column_names.index("task_index")
        typ = table.schema.field("task_index").type
        table = table.set_column(idx, "task_index", make_constant_column(new_task_index, table.num_rows, typ))
    ensure_dir(dst_parquet.parent)
    pq.write_table(table, dst_parquet)


def set_stats_task_index_const(stats_obj: Any, new_task_index: int) -> Any:
    if not isinstance(stats_obj, dict):
        return stats_obj
    out = deep_copy_jsonable(stats_obj)
    task_stat = out.get("task_index")
    if not isinstance(task_stat, dict):
        return out

    def same_shape_fill(template, value):
        if isinstance(template, list):
            return [same_shape_fill(v, value) for v in template]
        return value

    for k in ("min", "max", "mean"):
        if k in task_stat:
            task_stat[k] = same_shape_fill(task_stat[k], int(new_task_index))
    if "std" in task_stat:
        task_stat["std"] = same_shape_fill(task_stat["std"], 0.0)
    out["task_index"] = task_stat
    return out


# =========================
# repo writing
# =========================
def should_skip_raw_copy(rel: Path, copy_stats_json: bool) -> bool:
    rel_posix = rel.as_posix()
    if rel_posix.startswith("data/") and rel.suffix == ".parquet":
        return True
    if rel_posix in {
        "meta/tasks.jsonl",
        "meta/episodes.jsonl",
        "meta/episodes_stats.jsonl",
    }:
        return True
    if rel_posix == "meta/stats.json" and not copy_stats_json:
        return True
    return False


def mirror_non_parquet_assets(src_repo_dir: Path, dst_repo_dir: Path, link_mode: str, copy_stats_json: bool) -> None:
    for src in sorted(src_repo_dir.rglob("*"), key=lambda p: natural_key(p.as_posix())):
        if src.is_dir():
            continue
        rel = src.relative_to(src_repo_dir)
        if should_skip_raw_copy(rel, copy_stats_json=copy_stats_json):
            continue
        dst = dst_repo_dir / rel
        link_or_copy(src, dst, mode=link_mode)


def build_new_task_rows_and_indices(merged_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    task_to_index: Dict[str, int] = {}
    rows: List[Dict[str, Any]] = []
    for spec in merged_specs:
        task = spec["low_level_task"]
        if task not in task_to_index:
            tidx = len(task_to_index)
            task_to_index[task] = tidx
            rows.append({"task_index": tidx, "task": task})
        spec["new_task_index"] = task_to_index[task]
    return rows


def synthesize_episode_rows_if_missing(merged_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ep_idx, spec in enumerate(merged_specs):
        rows.append(
            {
                "episode_index": ep_idx,
                "task_index": int(spec["new_task_index"]),
                "env_id": spec["env_id"],
                "traj_id": spec["traj_id"],
                "pose_start": spec["pose_start"],
                "pose_end": spec["pose_end"],
                "task": spec["low_level_task"],
                "high_level_task": spec["high_level_task"],
            }
        )
    return rows


def create_new_repo_from_src(args: Args) -> None:
    task_root = Path(args.task_root)
    src_repo_dir = Path(args.src_repo_dir)
    dst_repo_dir = Path(args.dst_repo_dir)

    if not src_repo_dir.exists():
        raise FileNotFoundError(f"src_repo_dir not found: {src_repo_dir}")
    if not (src_repo_dir / "data").exists():
        raise FileNotFoundError(f"src repo missing data dir: {src_repo_dir / 'data'}")
    if dst_repo_dir.exists():
        if not args.overwrite_dst:
            raise FileExistsError(
                f"dst_repo_dir already exists: {dst_repo_dir}\n"
                f"Refusing to overwrite because overwrite_dst=False."
            )
        shutil.rmtree(dst_repo_dir)

    merged_specs = build_merged_episode_specs(
        task_root=task_root,
        split=args.split,
        num_shards=args.num_shards,
        instruction_dirname=args.instruction_dirname,
        instruction_pattern=args.instruction_pattern,
        subtask_filename=args.subtask_filename,
        joiner=args.joiner,
        strict=args.strict,
    )

    src_episode_rows = read_jsonl(src_repo_dir / "meta" / "episodes.jsonl")
    src_episode_stats_rows = read_jsonl(src_repo_dir / "meta" / "episodes_stats.jsonl")
    src_parquet_paths = sorted((src_repo_dir / "data").rglob("episode_*.parquet"), key=lambda p: natural_key(p.as_posix()))

    n_src_eps = len(src_parquet_paths)
    if len(merged_specs) != n_src_eps:
        raise RuntimeError(
            f"Episode count mismatch: rebuilt_specs={len(merged_specs)}, src_parquets={n_src_eps}.\n"
            f"Usually this means split or num_shards is wrong."
        )
    if src_episode_rows and len(src_episode_rows) != n_src_eps:
        raise RuntimeError(
            f"episodes.jsonl row count mismatch: src_episodes={len(src_episode_rows)}, src_parquets={n_src_eps}"
        )
    if src_episode_stats_rows and len(src_episode_stats_rows) != n_src_eps:
        raise RuntimeError(
            f"episodes_stats.jsonl row count mismatch: src_stats={len(src_episode_stats_rows)}, src_parquets={n_src_eps}"
        )

    ensure_dir(dst_repo_dir)
    ensure_dir(dst_repo_dir / "meta")
    ensure_dir(dst_repo_dir / "data")

    # 1) copy/link all unchanged files first (videos/media/info/readme/...)
    mirror_non_parquet_assets(
        src_repo_dir=src_repo_dir,
        dst_repo_dir=dst_repo_dir,
        link_mode=args.link_mode,
        copy_stats_json=args.copy_stats_json,
    )

    # 2) build new global tasks
    new_task_rows = build_new_task_rows_and_indices(merged_specs)

    # 3) write patched parquet files into new repo
    for ep_idx, src_parquet in enumerate(src_parquet_paths):
        rel = src_parquet.relative_to(src_repo_dir)
        dst_parquet = dst_repo_dir / rel
        patch_parquet_task_index_to_new_file(
            src_parquet=src_parquet,
            dst_parquet=dst_parquet,
            new_task_index=int(merged_specs[ep_idx]["new_task_index"]),
        )

    # 4) write meta/tasks.jsonl
    write_jsonl(dst_repo_dir / "meta" / "tasks.jsonl", new_task_rows)

    # 5) write patched meta/episodes.jsonl
    if src_episode_rows:
        new_episode_rows: List[Dict[str, Any]] = []
        for ep_idx, row in enumerate(src_episode_rows):
            row = deep_copy_jsonable(row)
            spec = merged_specs[ep_idx]
            row["task_index"] = int(spec["new_task_index"])
            row["env_id"] = spec["env_id"]
            row["traj_id"] = spec["traj_id"]
            row["pose_start"] = spec["pose_start"]
            row["pose_end"] = spec["pose_end"]
            row["task"] = spec["low_level_task"]
            row["high_level_task"] = spec["high_level_task"]
            new_episode_rows.append(row)
    else:
        new_episode_rows = synthesize_episode_rows_if_missing(merged_specs)
    write_jsonl(dst_repo_dir / "meta" / "episodes.jsonl", new_episode_rows)

    # 6) write patched meta/episodes_stats.jsonl if source has it
    if src_episode_stats_rows:
        new_episode_stats_rows: List[Dict[str, Any]] = []
        for ep_idx, row in enumerate(src_episode_stats_rows):
            row = deep_copy_jsonable(row)
            spec = merged_specs[ep_idx]
            new_tidx = int(spec["new_task_index"])
            if isinstance(row.get("task_index"), int):
                row["task_index"] = new_tidx
            row["episode_index"] = ep_idx
            row["stats"] = set_stats_task_index_const(row.get("stats", {}), new_tidx)
            new_episode_stats_rows.append(row)
        write_jsonl(dst_repo_dir / "meta" / "episodes_stats.jsonl", new_episode_stats_rows)

    # 7) optional debug mapping for auditing
    if args.write_episode_mapping_jsonl:
        mapping_rows: List[Dict[str, Any]] = []
        for ep_idx, spec in enumerate(merged_specs):
            mapping_rows.append(
                {
                    "episode_index": ep_idx,
                    "env_id": spec["env_id"],
                    "traj_id": spec["traj_id"],
                    "pose_start": spec["pose_start"],
                    "pose_end": spec["pose_end"],
                    "high_level_task": spec["high_level_task"],
                    "low_level_task": spec["low_level_task"],
                    "task_index": int(spec["new_task_index"]),
                }
            )
        write_jsonl(dst_repo_dir / "meta" / "episode_task_mapping_lowlevel.jsonl", mapping_rows)

    print(f"[OK] source repo kept untouched: {src_repo_dir}")
    print(f"[OK] created new repo: {dst_repo_dir}")
    print(f"[OK] total episodes: {len(merged_specs)}")
    print(f"[OK] unique low-level tasks: {len(new_task_rows)}")
    print(f"[OK] link_mode for unchanged assets: {args.link_mode}")
    print(f"[OK] copy_stats_json: {args.copy_stats_json}")


def main() -> None:
    args = tyro.cli(Args)
    create_new_repo_from_src(args)


if __name__ == "__main__":
    main()

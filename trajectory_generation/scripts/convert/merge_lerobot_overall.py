#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge multiple existing LeRobot repos into one overall repo per split.

Expected source layout (already converted/merged):
    <src_root>/task_<task_id>/<split>/
        meta/
        data/
        videos/

Default output layout:
    <dst_root>/<out_repo_prefix>/<split>/
        meta/
        data/
        videos/

Notes
-----
- This script does not re-convert raw data. It only re-merges existing LeRobot repos.
- Dataset structure stays unchanged: parquet/videos/meta are preserved.
- episode_index/task_index are re-numbered globally in the merged output.
- env_id is preserved in parquet and is also copied into merged meta/episodes.jsonl
  when present.
- Tasks are kept distinct per source task repo, even if two repos share the same
  instruction text. This avoids accidental cross-task task_index collapsing.
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_TASK_IDS = ("0", "hl", "building", "farm", "obstacle", "orbit", "road", "orbit_multi")
DEFAULT_SPLITS = ("train", "test_seen", "test_unseen")


def normalize_split(s: str) -> str:
    x = (s or "").strip().lower()
    x = {
        "seen": "test_seen",
        "unseen": "test_unseen",
        "testseen": "test_seen",
        "testunseen": "test_unseen",
    }.get(x, x)
    if x not in ("train", "test_seen", "test_unseen"):
        raise ValueError(f"Unknown split={s}. Expected train/test_seen/test_unseen.")
    return x


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_episode_id_from_name(name: str) -> Optional[int]:
    m = re.search(r"episode_(\d+)\.parquet$", name)
    if m:
        return int(m.group(1))
    m = re.search(r"episode_(\d+)\.[A-Za-z0-9]+$", name)
    if m:
        return int(m.group(1))
    return None


def format_chunk_dir(ep_id: int, chunk_size: int) -> str:
    return f"chunk-{ep_id // chunk_size:03d}"


def format_episode_stem(ep_id: int) -> str:
    return f"episode_{ep_id:06d}"


def link_or_copy(src: Path, dst: Path, mode: str = "hardlink") -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        return
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


def replace_episode_and_chunk_in_relpath(relpath: Path, new_ep: int, chunk_size: int) -> Path:
    s = relpath.as_posix()
    s = re.sub(r"chunk-\d{3,}", format_chunk_dir(new_ep, chunk_size), s)
    s = re.sub(r"episode_\d{1,}", format_episode_stem(new_ep), s)
    return Path(s)


def list_episode_parquets(repo_dir: Path) -> List[Path]:
    data_dir = repo_dir / "data"
    if not data_dir.exists():
        return []
    return sorted(data_dir.rglob("episode_*.parquet"), key=lambda p: natural_key(p.as_posix()))


def list_episode_media_files(repo_dir: Path, old_ep: int) -> List[Path]:
    out: List[Path] = []
    for media_root_name in ("videos", "video", "media"):
        media_root = repo_dir / media_root_name
        if not media_root.exists():
            continue
        pattern = f"**/{format_episode_stem(old_ep)}.*"
        out.extend(media_root.glob(pattern))
    return sorted(set(out), key=lambda p: natural_key(p.as_posix()))


def make_constant_column(value: int, n: int, typ: pa.DataType = pa.int64()) -> pa.Array:
    return pa.array([value] * n, type=typ)


def patch_parquet_episode_and_task(
    src_parquet: Path,
    dst_parquet: Path,
    new_episode_index: int,
    new_task_index: Optional[int] = None,
) -> int:
    table = pq.read_table(src_parquet)
    n = table.num_rows
    if n <= 0:
        ensure_dir(dst_parquet.parent)
        pq.write_table(table, dst_parquet)
        return n

    col_names = table.column_names
    if "episode_index" in col_names:
        idx = col_names.index("episode_index")
        typ = table.schema.field("episode_index").type
        table = table.set_column(idx, "episode_index", make_constant_column(new_episode_index, n, typ))

    if new_task_index is not None and "task_index" in col_names:
        idx = col_names.index("task_index")
        typ = table.schema.field("task_index").type
        table = table.set_column(idx, "task_index", make_constant_column(new_task_index, n, typ))

    ensure_dir(dst_parquet.parent)
    pq.write_table(table, dst_parquet)
    return n


def find_task_text_key(obj: Dict[str, Any]) -> Optional[str]:
    for k in ("task", "text", "instruction", "name"):
        if k in obj and isinstance(obj[k], str):
            return k
    return None


def build_local_task_map(tasks_rows: List[Dict[str, Any]]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for i, row in enumerate(tasks_rows):
        local_idx = None
        for k in ("task_index", "id", "index"):
            if k in row and isinstance(row[k], int):
                local_idx = row[k]
                break
        if local_idx is None:
            local_idx = i
        text_key = find_task_text_key(row)
        if text_key is None:
            continue
        mapping[int(local_idx)] = row[text_key]
    return mapping


def build_episode_meta_index(rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for i, row in enumerate(rows):
        ep = None
        for k in ("episode_index", "id", "index"):
            if k in row and isinstance(row[k], int):
                ep = row[k]
                break
        if ep is None:
            ep = i
        out[int(ep)] = row
    return out


def build_episode_stats_index(rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        ep = row.get("episode_index", None)
        if isinstance(ep, int):
            out[int(ep)] = row
    return out


def _deepcopy_jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x, ensure_ascii=False))


def _same_shape_fill(template: Any, value: Any) -> Any:
    if isinstance(template, list):
        return [_same_shape_fill(item, value) for item in template]
    return value


def _set_stat_const(stat_obj: Dict[str, Any], value: float) -> Dict[str, Any]:
    if not isinstance(stat_obj, dict):
        return stat_obj
    out = _deepcopy_jsonable(stat_obj)
    for k in ("min", "max", "mean"):
        if k in out:
            out[k] = _same_shape_fill(out[k], float(value) if isinstance(value, float) else value)
    if "std" in out:
        out["std"] = _same_shape_fill(out["std"], 0.0)
    return out


def _shift_stat(stat_obj: Dict[str, Any], delta: float) -> Dict[str, Any]:
    if not isinstance(stat_obj, dict):
        return stat_obj
    out = _deepcopy_jsonable(stat_obj)

    def _add_same_shape(x: Any, d: float) -> Any:
        if isinstance(x, list):
            return [_add_same_shape(v, d) for v in x]
        if isinstance(x, (int, float)):
            return x + d
        return x

    for k in ("min", "max", "mean"):
        if k in out:
            out[k] = _add_same_shape(out[k], delta)
    return out


def patch_episode_stats_row(
    row: Dict[str, Any],
    new_episode_index: int,
    frame_index_offset: int,
    new_task_index: Optional[int] = None,
) -> Dict[str, Any]:
    out = _deepcopy_jsonable(row)
    out["episode_index"] = int(new_episode_index)
    stats = out.get("stats", {})
    if isinstance(stats, dict):
        if "episode_index" in stats:
            stats["episode_index"] = _set_stat_const(stats["episode_index"], int(new_episode_index))
        if "index" in stats:
            stats["index"] = _shift_stat(stats["index"], int(frame_index_offset))
        if new_task_index is not None and "task_index" in stats:
            stats["task_index"] = _set_stat_const(stats["task_index"], int(new_task_index))
        out["stats"] = stats
    return out


def deep_patch_meta_obj(
    obj: Any,
    old_ep: Optional[int],
    new_ep: Optional[int],
    old_task: Optional[int],
    new_task: Optional[int],
    chunk_size: int,
) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "episode_index" and isinstance(v, int) and new_ep is not None:
                out[k] = new_ep
            elif k == "task_index" and isinstance(v, int) and new_task is not None:
                out[k] = new_task
            else:
                out[k] = deep_patch_meta_obj(v, old_ep, new_ep, old_task, new_task, chunk_size)
        return out
    if isinstance(obj, list):
        return [deep_patch_meta_obj(v, old_ep, new_ep, old_task, new_task, chunk_size) for v in obj]
    if isinstance(obj, str):
        s = obj
        if old_ep is not None and new_ep is not None:
            s = s.replace(format_episode_stem(old_ep), format_episode_stem(new_ep))
        if new_ep is not None and "chunk-" in s:
            s = re.sub(r"chunk-\d{3,}", format_chunk_dir(new_ep, chunk_size), s)
        return s
    return obj


def patch_info_json_counts(info_obj: Any, total_episodes: int, total_frames: int) -> Any:
    episode_keys = {"total_episodes", "num_episodes", "episodes_count", "n_episodes"}
    frame_keys = {"total_frames", "num_frames", "frames_count", "n_frames"}

    def _rec(x: Any) -> Any:
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                if k in episode_keys and isinstance(v, int):
                    out[k] = int(total_episodes)
                elif k in frame_keys and isinstance(v, int):
                    out[k] = int(total_frames)
                else:
                    out[k] = _rec(v)
            return out
        if isinstance(x, list):
            return [_rec(v) for v in x]
        return x

    return _rec(info_obj)


def infer_local_task_from_episode_row_or_parquet(ep_row: Optional[Dict[str, Any]], parquet_path: Path) -> Optional[int]:
    if ep_row is not None and "task_index" in ep_row and isinstance(ep_row["task_index"], int):
        return int(ep_row["task_index"])
    try:
        table = pq.read_table(parquet_path, columns=["task_index"])
        if table.num_rows > 0:
            return int(table["task_index"][0].as_py())
    except Exception:
        pass
    return None


def infer_env_id_from_parquet(parquet_path: Path) -> Optional[str]:
    try:
        table = pq.read_table(parquet_path, columns=["env_id"])
        if table.num_rows <= 0:
            return None
        v = table["env_id"][0].as_py()
        if isinstance(v, str):
            return v
        if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], str):
            return v[0]
    except Exception:
        return None
    return None


def copy_file_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)


def extract_info_signature(info_obj: Any) -> str:
    if not isinstance(info_obj, dict):
        return ""
    key_obj = {
        "features": info_obj.get("features", None),
        "fps": info_obj.get("fps", None),
        "robot_type": info_obj.get("robot_type", None),
        "format_version": info_obj.get("format_version", None),
    }
    return json.dumps(key_obj, ensure_ascii=False, sort_keys=True)


def task_id_from_source_name(source_name: str) -> str:
    source_name = str(source_name).strip()
    if source_name.startswith("task_"):
        return source_name[len("task_"):]
    return source_name


def build_source_repo(src_root: Path, task_id: str, split: str) -> Tuple[str, Path]:
    repo_name = f"task_{task_id}"
    repo_dir = src_root / repo_name / split
    return repo_name, repo_dir


def build_dst_repo(dst_root: Path, split: str, out_repo_prefix: str) -> Path:
    prefix = str(out_repo_prefix).strip().strip("/\\")
    if prefix:
        return dst_root / prefix / split
    return dst_root / split


def merge_repos_for_split(
    source_repos: List[Tuple[str, Path]],
    dst_repo: Path,
    chunk_size: int,
    overwrite: bool,
    link_mode: str,
    copy_stats_json: bool,
    strict: bool,
) -> Path:
    print(f"\n[Merge] output={dst_repo}")
    print("[Merge] sources:")
    for source_name, repo_dir in source_repos:
        print(f"  - {source_name}: {repo_dir}")

    existing_sources: List[Tuple[str, Path]] = []
    missing_sources: List[str] = []
    for source_name, repo_dir in source_repos:
        if repo_dir.exists():
            existing_sources.append((source_name, repo_dir))
        else:
            missing_sources.append(f"{source_name}: {repo_dir}")

    if missing_sources:
        msg = "Missing source repos:\n" + "\n".join(missing_sources)
        if strict:
            raise FileNotFoundError(msg)
        print("[Merge][WARN]", msg)

    if not existing_sources:
        raise FileNotFoundError("No existing source repos found for this split.")

    if dst_repo.exists():
        if not overwrite:
            raise FileExistsError(f"Target merged repo already exists: {dst_repo}")
        print(f"[Merge] Removing existing target: {dst_repo}")
        shutil.rmtree(dst_repo)

    ensure_dir(dst_repo / "meta")
    ensure_dir(dst_repo / "data")
    ensure_dir(dst_repo / "videos")

    global_task_to_idx: Dict[Tuple[str, int], int] = {}
    global_tasks_rows: List[Dict[str, Any]] = []
    merged_episodes_rows: List[Dict[str, Any]] = []
    merged_episode_stats_rows: List[Dict[str, Any]] = []

    total_frames = 0
    total_episodes = 0

    first_info_obj = None
    first_info_signature = None
    first_stats_path = None

    for source_name, repo_dir in existing_sources:
        print(f"\n[Merge][{source_name}] Scanning: {repo_dir}")
        source_task_id = task_id_from_source_name(source_name)

        info_path = repo_dir / "meta" / "info.json"
        if info_path.exists():
            try:
                info_obj = read_json(info_path)
                info_sig = extract_info_signature(info_obj)
                if first_info_obj is None:
                    first_info_obj = info_obj
                    first_info_signature = info_sig
                elif info_sig != first_info_signature:
                    msg = f"[{source_name}] info.json feature signature differs from the first source."
                    if strict:
                        raise RuntimeError(msg)
                    print(f"[Merge][WARN] {msg}")
            except Exception as e:
                msg = f"[{source_name}] failed to read info.json: {e}"
                if strict:
                    raise RuntimeError(msg) from e
                print(f"[Merge][WARN] {msg}")

        stats_path = repo_dir / "meta" / "stats.json"
        if first_stats_path is None and stats_path.exists():
            first_stats_path = stats_path

        source_tasks_rows = read_jsonl(repo_dir / "meta" / "tasks.jsonl")
        source_episodes_rows = read_jsonl(repo_dir / "meta" / "episodes.jsonl")
        source_episode_stats_rows = read_jsonl(repo_dir / "meta" / "episodes_stats.jsonl")

        local_task_map = build_local_task_map(source_tasks_rows)
        local_episode_meta = build_episode_meta_index(source_episodes_rows)
        local_episode_stats = build_episode_stats_index(source_episode_stats_rows)

        local_taskidx_to_global: Dict[int, int] = {}
        for local_tidx, task_text in local_task_map.items():
            task_key = (source_name, int(local_tidx))
            if task_key not in global_task_to_idx:
                gidx = len(global_task_to_idx)
                global_task_to_idx[task_key] = gidx
                global_tasks_rows.append(
                    {
                        "task_index": gidx,
                        "task": task_text,
                        "task_id": source_task_id,
                    }
                )
            local_taskidx_to_global[local_tidx] = global_task_to_idx[task_key]

        ep_parquets = list_episode_parquets(repo_dir)
        if not ep_parquets:
            msg = f"[{source_name}] no episode parquet under {repo_dir / 'data'}"
            if strict:
                raise RuntimeError(msg)
            print(f"[Merge][WARN] {msg}")
            continue

        print(f"[Merge][{source_name}] Found {len(ep_parquets)} episode parquet files")

        for src_parquet in ep_parquets:
            old_ep = parse_episode_id_from_name(src_parquet.name)
            if old_ep is None:
                msg = f"[{source_name}] cannot parse episode id from {src_parquet.name}"
                if strict:
                    raise RuntimeError(msg)
                print(f"[Merge][WARN] {msg}")
                continue

            new_ep = total_episodes
            new_chunk = format_chunk_dir(new_ep, chunk_size)

            ep_row_src = local_episode_meta.get(old_ep)
            local_task_idx = infer_local_task_from_episode_row_or_parquet(ep_row_src, src_parquet)
            global_task_idx = local_taskidx_to_global.get(local_task_idx) if local_task_idx is not None else None

            env_id_val = infer_env_id_from_parquet(src_parquet)

            dst_parquet = dst_repo / "data" / new_chunk / f"{format_episode_stem(new_ep)}.parquet"
            n_rows = patch_parquet_episode_and_task(
                src_parquet=src_parquet,
                dst_parquet=dst_parquet,
                new_episode_index=new_ep,
                new_task_index=global_task_idx,
            )

            frame_index_offset = total_frames

            media_files = list_episode_media_files(repo_dir, old_ep)
            for src_media in media_files:
                rel_media = src_media.relative_to(repo_dir)
                new_rel_media = replace_episode_and_chunk_in_relpath(rel_media, new_ep, chunk_size)
                dst_media = dst_repo / new_rel_media
                link_or_copy(src_media, dst_media, mode=link_mode)

            if ep_row_src is not None:
                ep_row_dst = deep_patch_meta_obj(
                    ep_row_src,
                    old_ep=old_ep,
                    new_ep=new_ep,
                    old_task=local_task_idx,
                    new_task=global_task_idx,
                    chunk_size=chunk_size,
                )
                if isinstance(ep_row_dst, dict):
                    if "length" in ep_row_dst and isinstance(ep_row_dst["length"], int):
                        ep_row_dst["length"] = int(n_rows)
                    elif "num_frames" in ep_row_dst and isinstance(ep_row_dst["num_frames"], int):
                        ep_row_dst["num_frames"] = int(n_rows)
            else:
                ep_row_dst = {
                    "episode_index": int(new_ep),
                    "task_index": int(global_task_idx) if global_task_idx is not None else 0,
                    "length": int(n_rows),
                }

            if isinstance(ep_row_dst, dict):
                ep_row_dst["task_id"] = source_task_id
                if env_id_val is not None:
                    ep_row_dst["env_id"] = env_id_val

            merged_episodes_rows.append(ep_row_dst)

            ep_stats_src = local_episode_stats.get(old_ep)
            if ep_stats_src is not None:
                merged_episode_stats_rows.append(
                    patch_episode_stats_row(
                        row=ep_stats_src,
                        new_episode_index=new_ep,
                        frame_index_offset=frame_index_offset,
                        new_task_index=global_task_idx,
                    )
                )

            total_frames += int(n_rows)
            total_episodes += 1

        print(
            f"[Merge][{source_name}] Done -> merged episodes so far: "
            f"{total_episodes}, frames: {total_frames}"
        )

    if not global_tasks_rows:
        global_tasks_rows = [{"task_index": 0, "task": "", "task_id": ""}]
        print("[Merge][WARN] No tasks extracted; wrote fallback empty task.")

    write_jsonl(dst_repo / "meta" / "tasks.jsonl", global_tasks_rows)
    write_jsonl(dst_repo / "meta" / "episodes.jsonl", merged_episodes_rows)

    if merged_episode_stats_rows:
        write_jsonl(dst_repo / "meta" / "episodes_stats.jsonl", merged_episode_stats_rows)
        print(f"[Merge] Wrote meta/episodes_stats.jsonl ({len(merged_episode_stats_rows)} rows)")
    else:
        print("[Merge][WARN] No episodes_stats.jsonl found in source repos; skipped.")

    if first_info_obj is not None:
        info_obj = patch_info_json_counts(first_info_obj, total_episodes=total_episodes, total_frames=total_frames)
        write_json(dst_repo / "meta" / "info.json", info_obj)
    else:
        write_json(
            dst_repo / "meta" / "info.json",
            {"total_episodes": int(total_episodes), "total_frames": int(total_frames), "note": "Minimal info.json"},
        )
        print("[Merge][WARN] No info.json found in source repos; wrote minimal info.json.")

    if copy_stats_json and first_stats_path is not None and first_stats_path.exists():
        copy_file_if_exists(first_stats_path, dst_repo / "meta" / "stats.json")
        print("[Merge][WARN] Copied stats.json from the first source repo only.")
    else:
        stats_dst = dst_repo / "meta" / "stats.json"
        if stats_dst.exists():
            stats_dst.unlink()

    print("\n[Merge] DONE")
    print(f"[Merge] merged repo: {dst_repo}")
    print(f"[Merge] total_episodes={total_episodes}, total_frames={total_frames}")
    return dst_repo


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple existing LeRobot task repos into overall train/test splits.")
    parser.add_argument("--src_root", type=str, default=os.environ.get("HUGE_LEROBOT_ROOT", "./lerobot_datasets"))
    parser.add_argument("--dst_root", type=str, default=os.environ.get("HUGE_LEROBOT_ROOT", "./lerobot_datasets"))
    parser.add_argument("--task_ids", type=str, default=",".join(DEFAULT_TASK_IDS))
    parser.add_argument("--splits", type=str, default=",".join(DEFAULT_SPLITS))
    parser.add_argument(
        "--out_repo_prefix",
        type=str,
        default="task_overall",
        help="Optional extra directory under dst_root, e.g. 'overall' -> <dst_root>/overall/<split>.",
    )
    parser.add_argument("--merge_chunk_size", type=int, default=1000)
    parser.add_argument("--link_mode", type=str, default="hardlink", choices=("hardlink", "copy", "symlink"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy_stats_json", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    task_ids = parse_csv_list(args.task_ids)
    splits = [normalize_split(s) for s in parse_csv_list(args.splits)]

    if not task_ids:
        raise ValueError("No task_ids provided.")
    if not splits:
        raise ValueError("No splits provided.")

    print("\n========== GLOBAL CONFIG ==========")
    print(f"[INFO] src_root={src_root}")
    print(f"[INFO] dst_root={dst_root}")
    print(f"[INFO] task_ids={task_ids}")
    print(f"[INFO] splits={splits}")
    print(f"[INFO] out_repo_prefix={args.out_repo_prefix or '(none)'}")
    print(f"[INFO] merge_chunk_size={args.merge_chunk_size}")
    print(f"[INFO] link_mode={args.link_mode}")
    print(f"[INFO] overwrite={args.overwrite} strict={args.strict}")

    for split in splits:
        source_repos = [build_source_repo(src_root, task_id, split) for task_id in task_ids]
        dst_repo = build_dst_repo(dst_root, split, args.out_repo_prefix)
        merge_repos_for_split(
            source_repos=source_repos,
            dst_repo=dst_repo,
            chunk_size=int(args.merge_chunk_size),
            overwrite=bool(args.overwrite),
            link_mode=str(args.link_mode),
            copy_stats_json=bool(args.copy_stats_json),
            strict=bool(args.strict),
        )

    print("\n========== ALL SPLITS DONE ==========")
    for split in splits:
        print(build_dst_repo(dst_root, split, args.out_repo_prefix))


if __name__ == "__main__":
    main()

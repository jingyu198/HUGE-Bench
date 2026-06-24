# -*- coding: utf-8 -*-
"""
Convert custom drone dataset -> LeRobotDataset SHARDS -> Fast merge to final repo
===============================================================================

改动（按你最新要求）：
1) train / test_seen / test_unseen 全部保留完整轨迹（不再只写第一帧）
2) convert 时为每条轨迹（每帧）写入 env_id 标签（parquet 列 + features）
   merge 时保留该列，并额外把 env_id 写进 merged 的 meta/episodes.jsonl（方便不读帧也能查到）

其它逻辑保持不变：
- actions：action_t = state_{t+1} - state_t（dyaw wrap），最后一帧 action=0
- scenes：1_office, 2_city, 3_road, 4_lake
- 默认输出 repo_id：task_{task_id}/{split}
  shards repo_id：{repo_id}_shard{sid}

数据结构（与你原脚本一致）：
{data_root}/task_{task_id}/
  1_office/
    traj_random.txt
    render_img/*.png
  2_city/...
  3_road/...
  4_lake/...
  split_res_merged/
    instruction_train.txt
    instruction_test_seen.txt
    instruction_test_unseen.txt
"""

import os
import re
import json
import time
import shutil
import multiprocessing as mp
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
import tyro
import pyarrow as pa
import pyarrow.parquet as pq

# ---- lerobot imports (compatible with older/newer paths) ----
os.environ.setdefault(
    "HF_LEROBOT_HOME",
    str(Path(os.environ.get("HUGE_LEROBOT_ROOT", "./lerobot_datasets")).expanduser()),
)

try:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
except Exception:
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


# =========================
# Args
# =========================
@dataclass
class Args:
    # dataset root
    data_root: str = os.environ.get("HUGE_DATA_TRAJ_ROOT", "./data_traj")
    task_id: str = "0"

    # multiple splits in one run
    splits: Tuple[str, ...] = ("train", "test_seen", "test_unseen")

    # scenes
    env_ids: Tuple[str, ...] = ("1_office", "2_city", "3_road", "4_lake","no1_building","no3_door","overhead_bridge")

    # instruction resolution
    instruction_dirname: str = "split_res_merged"
    instruction_pattern: str = "instruction_{split}.txt"

    # per-split repo_id base (default auto): task_{task_id}/{split}
    # if set non-empty, will be used as prefix; final per split: {repo_id_prefix}/{split}
    repo_id_prefix: str = ""  # "" => auto task_{task_id}

    # convert settings
    traj_filename: str = "traj_random.txt"
    image_dirname: str = "render_img"
    image_size: Tuple[int, int] = (256, 256)
    image_ext: str = "png"
    image_name_format: str = "{pose_id:06d}"

    state_dim: int = 4
    action_dim: int = 4
    fps: int = 5
    max_episodes: int = -1

    # writer concurrency for SHARD writing (auto-scaled in multi-process)
    image_writer_threads: int = 256
    image_writer_processes: int = 128

    # multiprocess sharding
    num_shards: int = 16
    mp_start_method: str = "spawn"

    # merge settings
    merge_chunk_size: int = 1000
    merge_overwrite: bool = False
    merge_link_mode: str = "hardlink"  # hardlink / copy / symlink
    merge_copy_stats_json: bool = False
    merge_strict: bool = False

    # optional cleanup
    delete_shards_after_merge: bool = True

    # For debug: run only one split (empty => all splits)
    only_split: str = ""
    only_merge: bool = False


# =========================
# Split normalization
# =========================
def normalize_split(s: str) -> str:
    x = (s or "").strip().lower()
    x = {
        "seen": "test_seen",
        "unseen": "test_unseen",
        "testseen": "test_seen",
        "testunseen": "test_unseen",
    }.get(x, x)
    if x not in ("train", "test_seen", "test_unseen"):
        raise ValueError(f"Unknown split={s}. Expected: train/test_seen/test_unseen (or seen/unseen).")
    return x


def default_repo_id_base(task_id: int, split: str, repo_id_prefix: str) -> str:
    split = normalize_split(split)
    prefix = (repo_id_prefix or "").strip()
    if prefix:
        return f"{prefix}/{split}"
    return f"task_{task_id}/{split}"


def resolve_instruction_path(task_root: Path, instruction_dirname: str, pattern: str, split: str) -> Path:
    split = normalize_split(split)
    return task_root / instruction_dirname / pattern.format(split=split)


# =========================
# Convert helpers
# =========================
def wrap_pi(a: float) -> float:
    """Wrap radians to [-pi, pi)."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi

def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def is_obstacle_task(task_id: str) -> bool:
    s = str(task_id).strip().lower()
    return s in ("obstacle", "task_obstacle")


def parse_traj_file(traj_path: Path, task_id: str) -> Dict[int, np.ndarray]:
    states: Dict[int, np.ndarray] = {}
    use_phi_angle = is_obstacle_task(task_id)

    with traj_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 7:
                raise ValueError(f"Bad traj line (need 7 columns): {line}")

            pose_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            phi_deg = float(parts[5])
            kappa_deg = float(parts[6])

            if use_phi_angle:
                # Obstacle trajectories are labeled with fixed omega/kappa and
                # phi already represents the render-driving horizontal view.
                angle_deg = wrap_deg(phi_deg)
            else:
                angle_deg = wrap_deg(kappa_deg)

            angle_rad = float(np.deg2rad(angle_deg))
            states[pose_id] = np.array([x, y, z, angle_rad], dtype=np.float32)

    return states

def parse_instruction_file(instr_path: Path) -> List[Tuple[str, int, int, int, str]]:
    """
    instruction_{split}.txt line format:
      env_id traj_id pose_start pose_end instruction_text...
    """
    episodes: List[Tuple[str, int, int, int, str]] = []
    with instr_path.open("r", encoding="utf-8") as f:
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
            episodes.append((env_id, traj_id, pose_start, pose_end, instruction))
    return episodes


def load_image_for_pose(
    image_dir: Path,
    pose_id: int,
    image_size: Optional[Tuple[int, int]],
    image_ext: str,
    image_name_format: str,
) -> np.ndarray:
    stem = image_name_format.format(pose_id=pose_id)
    img_path = image_dir / f"{stem}.{image_ext}"
    if not img_path.exists():
        raise FileNotFoundError(f"Image for pose_id={pose_id} not found at {img_path}")
    img = Image.open(img_path).convert("RGB")
    if image_size is not None:
        img = img.resize(image_size, Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def run_one_shard_convert(args: Args, split: str, repo_id_base: str, instruction_path: Path, shard_id: int) -> None:
    """
    Convert ONE shard for ONE split. Exceptions bubble up -> shard process exits non-zero.
    Fail-fast is handled by convert_shards_failfast().
    """
    split = normalize_split(split)
    task_root = Path(args.data_root) / f"task_{args.task_id}"

    shard_repo_id = f"{repo_id_base}_shard{shard_id}"
    output_path = HF_LEROBOT_HOME / shard_repo_id

    # Clean shard output only
    if output_path.exists():
        shutil.rmtree(output_path)

    # 1) 先扫描哪些 env 真有数据（traj + render_img）
    available_envs: List[str] = []
    for env_id in args.env_ids:
        traj_path = task_root / env_id / args.traj_filename
        img_dir = task_root / env_id / args.image_dirname
        if not traj_path.exists() or not img_dir.exists():
            print(
                f"[Convert/{split}][Shard {shard_id}] WARN skip env={env_id} "
                f"(traj_exists={traj_path.exists()}, img_dir_exists={img_dir.exists()})"
            )
            continue
        available_envs.append(env_id)

    available_env_set = set(available_envs)
    if not available_envs:
        print(f"[Convert/{split}][Shard {shard_id}] No available env found. Nothing to do.")
        return

    # 2) 解析 episodes，然后只保留 “既在 args.env_ids 且本地真实存在数据” 的 env
    print(f"[Convert/{split}][Shard {shard_id}] Parsing instructions: {instruction_path}")
    episodes_all = parse_instruction_file(instruction_path)
    episodes_all = [ep for ep in episodes_all if ep[0] in available_env_set]

    if args.max_episodes and args.max_episodes > 0:
        episodes_all = episodes_all[: args.max_episodes]

    episodes = episodes_all[shard_id :: int(args.num_shards)]

    print(
        f"[Convert/{split}][Shard {shard_id}] Using {len(episodes)} episodes "
        f"(of total {len(episodes_all)}) from available envs={sorted(available_envs)}"
    )
    if len(episodes) == 0:
        print(f"[Convert/{split}][Shard {shard_id}] No episodes assigned, exiting.")
        return

    # 3) 只解析本 shard 真正用到的 env（更省内存/时间）
    envs_used = sorted({ep[0] for ep in episodes})

    states_by_env: Dict[str, Dict[int, np.ndarray]] = {}
    image_dir_by_env: Dict[str, Path] = {}
    for env_id in envs_used:
        traj_path = task_root / env_id / args.traj_filename
        img_dir = task_root / env_id / args.image_dirname
        # 这里理论上一定存在（因为 episodes_all 已过滤），但保底一下
        if not traj_path.exists() or not img_dir.exists():
            print(f"[Convert/{split}][Shard {shard_id}] WARN env disappeared? skip env={env_id}")
            continue

        print(f"[Convert/{split}][Shard {shard_id}] Parsing traj for env={env_id}: {traj_path}")
        states_by_env[env_id] = parse_traj_file(traj_path, args.task_id)
        image_dir_by_env[env_id] = img_dir

    width, height = args.image_size

    # NOTE: 增加 env_id feature（dtype=string, shape=(1,)）
    dataset = LeRobotDataset.create(
        repo_id=shard_repo_id,
        robot_type="drone",
        fps=args.fps,
        features={
            "image": {"dtype": "image", "shape": (height, width, 3), "names": ["height", "width", "channel"]},
            "first_image": {"dtype": "image", "shape": (height, width, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (args.state_dim,), "names": ["x", "y", "z", "yaw_rad"]},
            "actions": {"dtype": "float32", "shape": (args.action_dim,), "names": ["dx", "dy", "dz", "dyaw"]},
            "env_id": {"dtype": "string", "shape": (1,)},
        },
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )

    for env_id, traj_id, pose_start, pose_end, instruction in episodes:
        print(f"[Convert/{split}][Shard {shard_id}] Episode env={env_id} traj_id={traj_id}, poses [{pose_start}, {pose_end}]")
        try:
            states = states_by_env[env_id]
            image_dir = image_dir_by_env[env_id]

            first_image = load_image_for_pose(
                image_dir=image_dir,
                pose_id=pose_start,
                image_size=args.image_size,
                image_ext=args.image_ext,
                image_name_format=args.image_name_format,
            )

            # ✅ 改动点：所有 split 都写完整轨迹
            pose_ids = list(range(pose_start, pose_end + 1))

            # actions = delta-to-next-frame, last action = 0
            for idx, pose_id in enumerate(pose_ids):
                if pose_id not in states:
                    raise KeyError(f"[{env_id}] pose_id={pose_id} not found in traj file states")
                state = states[pose_id]

                if idx < len(pose_ids) - 1:
                    next_pose_id = pose_ids[idx + 1]
                    if next_pose_id not in states:
                        raise KeyError(f"[{env_id}] pose_id={next_pose_id} not found in traj file states")
                    next_state = states[next_pose_id]
                    dxyz = (next_state[:3] - state[:3]).astype(np.float32)
                    dyaw = wrap_pi(float(next_state[3] - state[3]))
                    action = np.array([dxyz[0], dxyz[1], dxyz[2], dyaw], dtype=np.float32)
                else:
                    action = np.zeros((args.action_dim,), dtype=np.float32)

                image = load_image_for_pose(
                    image_dir=image_dir,
                    pose_id=pose_id,
                    image_size=args.image_size,
                    image_ext=args.image_ext,
                    image_name_format=args.image_name_format,
                )

                dataset.add_frame(
                    {
                        "image": image,
                        "first_image": first_image,
                        "state": state,
                        "actions": action,
                        "task": instruction,   # 保持不变（你原脚本这样用）
                        "env_id": env_id,       # ✅ 新增
                    }
                )

            dataset.save_episode()

        except Exception as e:
            print(
                f"[Convert/{split}][Shard {shard_id}] ERROR episode: env={env_id}, traj_id={traj_id}, "
                f"poses=[{pose_start}, {pose_end}] | {type(e).__name__}: {e}"
            )
            raise

    print(f"[Convert/{split}][Shard {shard_id}] Done. Output: {output_path}")


def convert_shards_failfast(args: Args, split: str, repo_id_base: str, instruction_path: Path) -> None:
    """
    Launch all shards for one split; fail-fast if any shard exits non-zero.
    """
    num_shards = int(args.num_shards)
    if num_shards <= 0:
        raise ValueError("num_shards must be > 0")

    per_shard_writer_procs = max(1, args.image_writer_processes // num_shards)
    per_shard_writer_threads = max(1, args.image_writer_threads // num_shards)

    print(f"\n[Convert/{split}] Multiprocess shards: {num_shards}")
    print(f"[Convert/{split}] Writer per shard: procs={per_shard_writer_procs}, threads={per_shard_writer_threads}")
    print(f"[Convert/{split}] repo_id_base={repo_id_base}")
    print(f"[Convert/{split}] HF_LEROBOT_HOME={HF_LEROBOT_HOME}")

    ctx = mp.get_context(args.mp_start_method)
    procs: List[mp.Process] = []

    for sid in range(num_shards):
        shard_args = replace(
            args,
            image_writer_processes=per_shard_writer_procs,
            image_writer_threads=per_shard_writer_threads,
        )
        p = ctx.Process(
            target=run_one_shard_convert,
            args=(shard_args, split, repo_id_base, instruction_path, sid),
        )
        p.start()
        procs.append(p)

    failed: Optional[Tuple[int, int]] = None  # (sid, exit_code)

    try:
        while True:
            all_done = True
            for sid, p in enumerate(procs):
                ec = p.exitcode
                if ec is None:
                    all_done = False
                    continue
                if ec != 0 and failed is None:
                    failed = (sid, ec)
                    print(f"[Convert/{split}] Detected shard {sid} failed (exit={ec}). Terminating others...")
                    for j, q in enumerate(procs):
                        if j != sid and q.is_alive():
                            try:
                                q.terminate()
                            except Exception as te:
                                print(f"[Convert/{split}] Warning: terminate shard {j} failed: {te}")
                    break

            if failed is not None or all_done:
                break
            time.sleep(0.2)

    finally:
        for sid, p in enumerate(procs):
            if p.is_alive():
                p.join(timeout=2.0)
            if p.is_alive():
                try:
                    p.kill()
                except Exception as ke:
                    print(f"[Convert/{split}] Warning: kill shard {sid} failed: {ke}")
            p.join()

    exit_codes = [p.exitcode for p in procs]
    if any(ec != 0 for ec in exit_codes):
        raise RuntimeError(f"[Convert/{split}] Conversion stopped due to shard failure. Exit codes: {exit_codes}")

    print(f"[Convert/{split}] All shards finished successfully: {repo_id_base}_shard0..{num_shards-1}")


# =========================
# Merge helpers (fast merge)
# =========================
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


def _set_stat_const(stat_obj: Dict[str, Any], value: float, keep_count: bool = True) -> Dict[str, Any]:
    if not isinstance(stat_obj, dict):
        return stat_obj
    out = _deepcopy_jsonable(stat_obj)

    def _same_shape_fill(template, v):
        if isinstance(template, list):
            return [_same_shape_fill(item, v) for item in template]
        return v

    for k in ("min", "max", "mean"):
        if k in out:
            out[k] = _same_shape_fill(out[k], float(value) if isinstance(value, float) else value)
    if "std" in out:
        out["std"] = _same_shape_fill(out["std"], 0.0)
    if not keep_count and "count" in out:
        out.pop("count", None)
    return out


def _shift_stat(stat_obj: Dict[str, Any], delta: float) -> Dict[str, Any]:
    if not isinstance(stat_obj, dict):
        return stat_obj
    out = _deepcopy_jsonable(stat_obj)

    def _add_same_shape(x, d):
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
    # 方便 merge 时把 env_id 也写进 episodes.jsonl（不影响 parquet 列本身）
    try:
        table = pq.read_table(parquet_path, columns=["env_id"])
        if table.num_rows <= 0:
            return None
        v = table["env_id"][0].as_py()
        if isinstance(v, str):
            return v
        # 有些实现可能返回 ['1_office'] 之类
        if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], str):
            return v[0]
    except Exception:
        return None
    return None


def copy_file_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)


def fast_merge_shards(
    repo_id_base: str,
    num_shards: int,
    chunk_size: int,
    overwrite: bool,
    link_mode: str,
    copy_stats_json: bool,
    strict: bool,
) -> Path:
    hf_home = Path(os.environ.get("HF_LEROBOT_HOME", str(HF_LEROBOT_HOME)))
    dst_repo = hf_home / repo_id_base

    shard_ids = list(range(int(num_shards)))
    shard_repo_ids = [f"{repo_id_base}_shard{sid}" for sid in shard_ids]
    shard_dirs = [hf_home / rid for rid in shard_repo_ids]

    print(f"\n[Merge] repo_id(output)={repo_id_base}")
    print(f"[Merge] HF_LEROBOT_HOME={hf_home}")
    print(f"[Merge] shard_dirs:")
    for p in shard_dirs:
        print(f"  - {p}")

    missing = [str(p) for p in shard_dirs if not p.exists()]
    if missing:
        msg = "Missing shard directories:\n" + "\n".join(missing)
        if strict:
            raise FileNotFoundError(msg)
        print("[Merge][WARN]", msg)
        # 只保留存在的 shard
        shard_pairs = [(sid, d) for sid, d in zip(shard_ids, shard_dirs) if d.exists()]
        shard_ids = [sid for sid, _ in shard_pairs]
        shard_dirs = [d for _, d in shard_pairs]

    if not shard_dirs:
        print(f"[Merge][WARN] No shard directories available for {repo_id_base}; skipping merge.")
        return dst_repo

    if dst_repo.exists():
        if not overwrite:
            raise FileExistsError(f"Target merged repo already exists: {dst_repo}\nUse --merge_overwrite True")
        print(f"[Merge] Removing existing target: {dst_repo}")
        shutil.rmtree(dst_repo)

    ensure_dir(dst_repo / "meta")
    ensure_dir(dst_repo / "data")
    ensure_dir(dst_repo / "videos")

    global_task_to_idx: Dict[str, int] = {}
    global_tasks_rows: List[Dict[str, Any]] = []

    merged_episodes_rows: List[Dict[str, Any]] = []
    merged_episode_stats_rows: List[Dict[str, Any]] = []

    total_frames = 0
    total_episodes = 0

    first_info_obj = None
    first_info_path = shard_dirs[0] / "meta" / "info.json"
    if first_info_path.exists():
        try:
            first_info_obj = read_json(first_info_path)
        except Exception as e:
            print(f"[Merge][WARN] Failed to read first shard info.json: {e}")

    for sid, shard_dir in zip(shard_ids, shard_dirs):
        print(f"\n[Merge][Shard {sid}] Scanning: {shard_dir}")

        shard_tasks_rows = read_jsonl(shard_dir / "meta" / "tasks.jsonl")
        shard_episodes_rows = read_jsonl(shard_dir / "meta" / "episodes.jsonl")
        shard_episode_stats_rows = read_jsonl(shard_dir / "meta" / "episodes_stats.jsonl")

        local_task_map = build_local_task_map(shard_tasks_rows)
        local_episode_meta = build_episode_meta_index(shard_episodes_rows)
        local_episode_stats = build_episode_stats_index(shard_episode_stats_rows)

        local_taskidx_to_global: Dict[int, int] = {}
        for local_tidx, task_text in local_task_map.items():
            if task_text not in global_task_to_idx:
                gidx = len(global_task_to_idx)
                global_task_to_idx[task_text] = gidx
                global_tasks_rows.append({"task_index": gidx, "task": task_text})
            local_taskidx_to_global[local_tidx] = global_task_to_idx[task_text]

        ep_parquets = list_episode_parquets(shard_dir)
        if not ep_parquets:
            msg = f"[Merge][Shard {sid}] No episode parquet under {shard_dir / 'data'}"
            if strict:
                raise RuntimeError(msg)
            print("[Merge][WARN]", msg)
            continue

        print(f"[Merge][Shard {sid}] Found {len(ep_parquets)} episode parquet files")

        for src_parquet in ep_parquets:
            old_ep = parse_episode_id_from_name(src_parquet.name)
            if old_ep is None:
                msg = f"[Merge][Shard {sid}] Cannot parse episode id from {src_parquet.name}"
                if strict:
                    raise RuntimeError(msg)
                print("[Merge][WARN]", msg)
                continue

            new_ep = total_episodes
            new_chunk = format_chunk_dir(new_ep, chunk_size)

            ep_row_src = local_episode_meta.get(old_ep)
            local_task_idx = infer_local_task_from_episode_row_or_parquet(ep_row_src, src_parquet)
            global_task_idx = local_taskidx_to_global.get(local_task_idx) if local_task_idx is not None else None

            # ✅ 读取 env_id，写入 merged episodes.jsonl（parquet 列会原样保留，无需 patch）
            env_id_val = infer_env_id_from_parquet(src_parquet)

            dst_parquet = dst_repo / "data" / new_chunk / f"{format_episode_stem(new_ep)}.parquet"
            n_rows = patch_parquet_episode_and_task(
                src_parquet=src_parquet,
                dst_parquet=dst_parquet,
                new_episode_index=new_ep,
                new_task_index=global_task_idx,
            )

            frame_index_offset = total_frames

            media_files = list_episode_media_files(shard_dir, old_ep)
            for src_media in media_files:
                rel_media = src_media.relative_to(shard_dir)
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

            # ✅ merge 时把 env_id 也放进 episodes.jsonl（不改变其它结构）
            if isinstance(ep_row_dst, dict) and env_id_val is not None:
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

        print(f"[Merge][Shard {sid}] Done -> merged episodes so far: {total_episodes}, frames: {total_frames}")

    if not global_tasks_rows:
        global_tasks_rows = [{"task_index": 0, "task": ""}]
        print("[Merge][WARN] No tasks extracted; wrote fallback empty task.")

    write_jsonl(dst_repo / "meta" / "tasks.jsonl", global_tasks_rows)
    write_jsonl(dst_repo / "meta" / "episodes.jsonl", merged_episodes_rows)

    if merged_episode_stats_rows:
        write_jsonl(dst_repo / "meta" / "episodes_stats.jsonl", merged_episode_stats_rows)
        print(f"[Merge] Wrote meta/episodes_stats.jsonl ({len(merged_episode_stats_rows)} rows)")
    else:
        print("[Merge][WARN] No episodes_stats.jsonl found in shards; skipped.")

    if first_info_obj is not None:
        info_obj = patch_info_json_counts(first_info_obj, total_episodes=total_episodes, total_frames=total_frames)
        write_json(dst_repo / "meta" / "info.json", info_obj)
    else:
        write_json(
            dst_repo / "meta" / "info.json",
            {"total_episodes": int(total_episodes), "total_frames": int(total_frames), "note": "Minimal info.json"},
        )
        print("[Merge][WARN] first shard info.json not found; wrote minimal info.json.")

    if copy_stats_json:
        first_stats = shard_dirs[0] / "meta" / "stats.json"
        if first_stats.exists():
            copy_file_if_exists(first_stats, dst_repo / "meta" / "stats.json")
            print("[Merge][WARN] Copied stats.json from shard0 only (likely stale after merge).")
    else:
        stats_dst = dst_repo / "meta" / "stats.json"
        if stats_dst.exists():
            stats_dst.unlink()

    print("\n[Merge] DONE")
    print(f"[Merge] merged repo: {dst_repo}")
    print(f"[Merge] total_episodes={total_episodes}, total_frames={total_frames}")
    return dst_repo


# =========================
# Orchestration (multi-split)
# =========================
def convert_and_merge_one_split(args: Args, split: str) -> None:
    split = normalize_split(split)
    task_root = Path(args.data_root) / f"task_{args.task_id}"
    if not task_root.exists():
        raise FileNotFoundError(f"task_root not found: {task_root}")

    instruction_path = resolve_instruction_path(task_root, args.instruction_dirname, args.instruction_pattern, split)
    if not instruction_path.exists():
        raise FileNotFoundError(f"instruction file not found: {instruction_path}")

    repo_id_base = default_repo_id_base(args.task_id, split, args.repo_id_prefix)

    print("\n========== CONFIG ==========")
    print(f"[INFO] split={split}")
    print(f"[INFO] repo_id_base={repo_id_base}")
    print(f"[INFO] instruction_path={instruction_path}")
    print(f"[INFO] env_ids={args.env_ids}")
    print(f"[INFO] traj_filename={args.traj_filename}")
    print(f"[INFO] image_dirname={args.image_dirname}")

    available_envs = [
        env_id
        for env_id in args.env_ids
        if (task_root / env_id / args.traj_filename).exists() and (task_root / env_id / args.image_dirname).exists()
    ]
    available_env_set = set(available_envs)

    episodes_all = parse_instruction_file(instruction_path)
    episodes_all = [ep for ep in episodes_all if ep[0] in available_env_set]
    if args.max_episodes and args.max_episodes > 0:
        episodes_all = episodes_all[: args.max_episodes]

    print(f"[INFO] available_envs={tuple(available_envs)}")
    print(f"[INFO] matched_episode_count={len(episodes_all)}")
    if len(episodes_all) == 0:
        print(f"[SKIP] Split {split} has no runnable episodes for env_ids={args.env_ids}. Skipping convert+merge.")
        return

    # 1) convert -> shards
    if not args.only_merge:
        convert_shards_failfast(args, split=split, repo_id_base=repo_id_base, instruction_path=instruction_path)

    dst_repo = fast_merge_shards(
        repo_id_base=repo_id_base,
        num_shards=int(args.num_shards),
        chunk_size=int(args.merge_chunk_size),
        overwrite=bool(args.merge_overwrite),
        link_mode=str(args.merge_link_mode),
        copy_stats_json=bool(args.merge_copy_stats_json),
        strict=bool(args.merge_strict),
    )

    # 3) optional cleanup
    if args.delete_shards_after_merge:
        hf_home = Path(os.environ.get("HF_LEROBOT_HOME", str(HF_LEROBOT_HOME)))
        for sid in range(int(args.num_shards)):
            shard_dir = hf_home / f"{repo_id_base}_shard{sid}"
            if shard_dir.exists():
                print(f"[Cleanup] Removing shard: {shard_dir}")
                shutil.rmtree(shard_dir)

    print(f"[OK] Split {split} finished. Merged repo at: {dst_repo}")


def main(args: Args) -> None:
    splits = [normalize_split(s) for s in args.splits]
    if args.only_split.strip():
        splits = [normalize_split(args.only_split)]

    print("\n========== GLOBAL CONFIG ==========")
    print(f"[INFO] data_root={args.data_root}")
    print(f"[INFO] task_id={args.task_id}")
    print(f"[INFO] splits={splits}")
    print(f"[INFO] HF_LEROBOT_HOME={HF_LEROBOT_HOME}")
    print(f"[INFO] num_shards={args.num_shards} mp_start_method={args.mp_start_method}")
    print(f"[INFO] merge_overwrite={args.merge_overwrite} link_mode={args.merge_link_mode}")
    print(f"[INFO] repo_id_prefix={args.repo_id_prefix or '(auto task_{task_id})'}")

    # 任一 split 失败直接停止（fail-fast across splits）
    for split in splits:
        convert_and_merge_one_split(args, split)

    print("\n========== ALL SPLITS DONE ==========")
    for split in splits:
        print(f"  - {default_repo_id_base(args.task_id, split, args.repo_id_prefix)}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)

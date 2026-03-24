import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


DEFAULT_TASKS = [
    "0",
    "obstacle",
    "hl",
    "orbit",
    "road",
    "building",
    "orbit_multi",
    "farm",
]


def wrap_rad(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def maybe_deg_to_rad(angle: float) -> float:
    value = float(angle)
    if abs(value) > 3.5 * math.pi:
        return math.radians(value)
    return value


def sanitize_traj_xyzk(traj: np.ndarray) -> np.ndarray:
    arr = np.asarray(traj, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError(f"Expected trajectory with shape (T,4+), got {arr.shape}")

    arr = arr[:, :4].copy()
    for i in range(arr.shape[0]):
        arr[i, 3] = wrap_rad(float(maybe_deg_to_rad(arr[i, 3])))
    return arr.astype(np.float32)


def parse_thresholds(text: str) -> List[float]:
    if text is None:
        return []

    normalized = str(text).replace("\uFF0C", ",").replace(";", ",")
    parts: List[str] = []
    for chunk in normalized.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts.extend([piece for piece in chunk.split() if piece.strip()])

    thresholds: List[float] = []
    seen = set()
    for piece in parts:
        try:
            value = float(piece)
        except Exception:
            continue
        if value <= 0 or value in seen:
            continue
        thresholds.append(value)
        seen.add(value)
    return thresholds


def parse_tasks(text: str, default_tasks: List[str]) -> List[str]:
    if text is None:
        return list(default_tasks)

    normalized = str(text).strip()
    if not normalized:
        return list(default_tasks)

    normalized = normalized.replace("\uFF0C", ",").replace(";", ",")
    parts: List[str] = []
    for chunk in normalized.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts.extend([piece for piece in chunk.split() if piece.strip()])

    tasks: List[str] = []
    seen = set()
    for piece in parts:
        if piece in seen:
            continue
        tasks.append(piece)
        seen.add(piece)
    return tasks if tasks else list(default_tasks)


def compute_tcr_xyz(gt_t4: np.ndarray, pred_t4: np.ndarray, thresholds_m: List[float]) -> Dict[float, float]:
    gt = np.asarray(gt_t4, dtype=np.float32)
    pred = np.asarray(pred_t4, dtype=np.float32)
    output: Dict[float, float] = {float(thr): float("nan") for thr in thresholds_m}

    if gt.shape[0] == 0 or pred.shape[0] == 0:
        return output

    gt_xyz = gt[:, :3]
    pred_xyz = pred[:, :3]
    min_d2 = np.empty((gt_xyz.shape[0],), dtype=np.float32)

    for i in range(gt_xyz.shape[0]):
        delta = pred_xyz - gt_xyz[i]
        d2 = np.sum(delta * delta, axis=1)
        min_d2[i] = np.min(d2) if d2.size > 0 else np.float32(np.inf)

    for thr in thresholds_m:
        thr2 = np.float32(float(thr) * float(thr))
        output[float(thr)] = float(np.mean(min_d2 <= thr2))

    return output


def compute_success_rate(gt_t4: np.ndarray, pred_t4: np.ndarray, success_thresh_m: float) -> float:
    gt = np.asarray(gt_t4, dtype=np.float32)
    pred = np.asarray(pred_t4, dtype=np.float32)
    if gt.shape[0] == 0 or pred.shape[0] == 0:
        return float("nan")
    end_dist = float(np.linalg.norm(pred[-1, :3] - gt[-1, :3]))
    return 1.0 if end_dist <= float(success_thresh_m) else 0.0


def find_all_npz(out_dir: str) -> List[Path]:
    root = Path(out_dir)
    if not root.exists():
        return []
    return sorted(root.rglob("traj_gt_pred_xyzk.npz"))


def parse_task_split_from_path(path: Path) -> Tuple[Optional[str], Optional[str]]:
    parts = path.parts
    task_id = None
    split = None
    for i in range(len(parts) - 1):
        if parts[i].startswith("task_"):
            task_id = parts[i].replace("task_", "", 1)
            if i + 1 < len(parts):
                split = parts[i + 1]
            break
    return task_id, split


def canonical_seen_unseen(split: str) -> Optional[str]:
    value = (split or "").lower()
    if "unseen" in value:
        return "unseen"
    if "seen" in value:
        return "seen"
    return None


def mean_of(values: List[float]) -> float:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def format_metric(value: float) -> str:
    return f"{value:.6f}" if np.isfinite(value) else "nan"


def evaluate_one_task(
    npz_files: List[Path],
    task_id: str,
    thresholds_m: List[float],
    success_thresh_m: float,
    stride: int,
    max_len: int,
    per_episode: bool,
) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[Dict[str, float]]] = {"seen": [], "unseen": []}

    for npz_path in npz_files:
        parsed_task, split = parse_task_split_from_path(npz_path)
        if parsed_task is None or split is None or str(parsed_task) != str(task_id):
            continue

        seen_key = canonical_seen_unseen(split)
        if seen_key is None:
            continue

        data = np.load(str(npz_path))
        gt = sanitize_traj_xyzk(data["gt_xyzk"])
        pred = sanitize_traj_xyzk(data["pred_xyzk"])

        if max_len > 0:
            gt = gt[:max_len]
            pred = pred[:max_len]
        if stride > 1:
            gt = gt[::stride]
            pred = pred[::stride]

        tcr = compute_tcr_xyz(gt, pred, thresholds_m=thresholds_m)
        sr = compute_success_rate(gt, pred, success_thresh_m=success_thresh_m)

        record: Dict[str, float] = {"SR": float(sr)}
        for threshold in thresholds_m:
            record[f"TCR@{float(threshold):g}"] = float(tcr[float(threshold)])
        buckets[seen_key].append(record)

        if per_episode:
            episode_name = npz_path.parent.name
            env_id = npz_path.parent.parent.name
            tcr_str = "  ".join(
                f"TCR@{float(threshold):g}={record[f'TCR@{float(threshold):g}']:.6f}"
                for threshold in thresholds_m
            )
            print(
                f"task={task_id} split={split} ({seen_key}) env={env_id} {episode_name}  "
                f"SR={record['SR']:.0f}  {tcr_str}"
            )

    summary: Dict[str, Dict[str, float]] = {"seen": {}, "unseen": {}}
    for seen_key in ["seen", "unseen"]:
        summary[seen_key]["SR"] = mean_of([record["SR"] for record in buckets[seen_key]])
        for threshold in thresholds_m:
            metric_name = f"TCR@{float(threshold):g}"
            summary[seen_key][metric_name] = mean_of([record[metric_name] for record in buckets[seen_key]])
    return summary


def print_summary_table(tasks: List[str], summaries: Dict[str, Dict[str, Dict[str, float]]], thresholds_m: List[float]):
    metric_names = ["SR"] + [f"TCR@{float(threshold):g}" for threshold in thresholds_m]
    task_width = max(len("task"), max((len(task) for task in tasks), default=0))
    metric_width = 10
    sep = "  "

    line1 = (
        f"{'task':<{task_width}}{sep}"
        f"{'seen':^{len(metric_names) * metric_width + (len(metric_names) - 1) * len(sep)}}{sep}"
        f"{'unseen':^{len(metric_names) * metric_width + (len(metric_names) - 1) * len(sep)}}"
    )
    print(line1)

    header_cols = [f"{'':<{task_width}}"]
    header_cols.extend([f"{name:^{metric_width}}" for name in metric_names])
    header_cols.extend([f"{name:^{metric_width}}" for name in metric_names])
    print(sep.join(header_cols))
    print("-" * len(line1))

    for task in tasks:
        seen = summaries.get(task, {}).get("seen", {})
        unseen = summaries.get(task, {}).get("unseen", {})
        cols = [f"{task:<{task_width}}"]
        cols.extend([f"{format_metric(seen.get(name, float('nan'))):>{metric_width}}" for name in metric_names])
        cols.extend([f"{format_metric(unseen.get(name, float('nan'))):>{metric_width}}" for name in metric_names])
        print(sep.join(cols))


def main():
    parser = argparse.ArgumentParser("Evaluate SR/TCR from saved traj_gt_pred_xyzk.npz outputs.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./outputs",
        help="Root output directory that contains task_<task_id>/<split>/.../traj_gt_pred_xyzk.npz",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated task ids. Leave empty to use the default benchmark task list.",
    )
    parser.add_argument(
        "--tcr_thresholds",
        type=str,
        default="1,2,5",
        help="Comma-separated TCR thresholds in meters.",
    )
    parser.add_argument(
        "--success_thresh",
        type=float,
        default=10.0,
        help="Success threshold in meters: success if endpoint distance is within this value.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Subsample trajectories by stride.")
    parser.add_argument("--max_len", type=int, default=0, help="Optional maximum trajectory length.")
    parser.add_argument("--per_episode", action="store_true", help="Print per-episode metrics.")

    args = parser.parse_args()

    thresholds_m = parse_thresholds(args.tcr_thresholds)
    if not thresholds_m:
        thresholds_m = [1.0, 2.0, 5.0]

    tasks = parse_tasks(args.tasks, default_tasks=DEFAULT_TASKS)
    npz_files = find_all_npz(args.out_dir)
    if not npz_files:
        print(f"[ERR] No traj_gt_pred_xyzk.npz found under: {args.out_dir}")
        return

    print(f"[INFO] out_dir={args.out_dir}")
    print(f"[INFO] tasks={tasks}")
    print(f"[INFO] thresholds={thresholds_m}  success_thresh={float(args.success_thresh):g}")

    summaries: Dict[str, Dict[str, Dict[str, float]]] = {}
    for task in tasks:
        summaries[task] = evaluate_one_task(
            npz_files=npz_files,
            task_id=task,
            thresholds_m=thresholds_m,
            success_thresh_m=float(args.success_thresh),
            stride=max(int(args.stride), 1),
            max_len=max(int(args.max_len), 0),
            per_episode=bool(args.per_episode),
        )

    print("\n==================== FINAL SUMMARY ====================")
    print_summary_table(tasks, summaries, thresholds_m)


if __name__ == "__main__":
    main()

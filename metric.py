#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import trimesh  # type: ignore
except Exception:
    trimesh = None


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

DEFAULT_TCR_THRESHOLDS = [1.0, 2.0, 5.0]


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
        raise ValueError(f"Expected trajectory with shape (T, 4+), got {arr.shape}")

    arr = arr[:, :4].copy()
    for i in range(arr.shape[0]):
        arr[i, 3] = wrap_rad(float(maybe_deg_to_rad(arr[i, 3])))
    return arr.astype(np.float32)


def parse_number_list(text: Optional[str], default: List[float]) -> List[float]:
    if text is None or not str(text).strip():
        return list(default)

    normalized = str(text).replace("\uFF0C", ",").replace(";", ",")
    values: List[float] = []
    seen = set()
    for chunk in normalized.split(","):
        for piece in chunk.split():
            try:
                value = float(piece)
            except Exception:
                continue
            if value <= 0 or value in seen:
                continue
            values.append(value)
            seen.add(value)
    return values if values else list(default)


def parse_task_list(text: Optional[str]) -> List[str]:
    if text is None or not str(text).strip():
        return list(DEFAULT_TASKS)

    normalized = str(text).replace("\uFF0C", ",").replace(";", ",")
    tasks: List[str] = []
    seen = set()
    for chunk in normalized.split(","):
        for piece in chunk.split():
            task = piece.strip()
            if task.startswith("task_"):
                task = task.replace("task_", "", 1)
            if not task or task in seen:
                continue
            tasks.append(task)
            seen.add(task)
    return tasks if tasks else list(DEFAULT_TASKS)


def canonical_task_name(task: str) -> str:
    task = str(task)
    return task.replace("task_", "", 1) if task.startswith("task_") else task


def canonical_split(split: Optional[str]) -> Optional[str]:
    value = (split or "").lower()
    if "unseen" in value:
        return "unseen"
    if "seen" in value:
        return "seen"
    return None


def find_npz_files(out_dir: Path) -> List[Path]:
    if not out_dir.exists():
        return []
    return sorted(out_dir.rglob("traj_gt_pred_xyzk.npz"))


def parse_task_split_env(path: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    parts = list(path.parts)
    task_id = None
    split = None
    env_id = None

    for i, part in enumerate(parts):
        if part.startswith("task_"):
            task_id = canonical_task_name(part)
            if i + 1 < len(parts):
                split = parts[i + 1]
            break

    if split and split in parts:
        idx = parts.index(split)
        if idx + 2 < len(parts) and parts[idx + 1].lower() in {"openvla", "fastvla", "fastvlm"}:
            env_id = parts[idx + 2]
        elif idx + 1 < len(parts):
            env_id = parts[idx + 1]

    if env_id is None and len(parts) >= 3:
        env_id = path.parent.parent.name

    return task_id, canonical_split(split), env_id


def finite_mean(values: Iterable[float]) -> float:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def format_float(value: float, ndigits: int = 4) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{float(value):.{ndigits}f}"


def path_length_xyz(points: np.ndarray) -> float:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 2:
        return float("nan")
    return float(np.linalg.norm(arr[1:] - arr[:-1], axis=1).sum())


def compute_tcr_xyz(gt: np.ndarray, pred: np.ndarray, thresholds_m: List[float], chunk_size: int = 2048) -> Dict[float, float]:
    output: Dict[float, float] = {float(thr): float("nan") for thr in thresholds_m}
    if gt.shape[0] == 0 or pred.shape[0] == 0:
        return output

    gt_xyz = np.asarray(gt[:, :3], dtype=np.float32)
    pred_xyz = np.asarray(pred[:, :3], dtype=np.float32)
    min_d2 = np.empty((gt_xyz.shape[0],), dtype=np.float32)

    for start in range(0, gt_xyz.shape[0], chunk_size):
        end = min(start + chunk_size, gt_xyz.shape[0])
        delta = gt_xyz[start:end, None, :] - pred_xyz[None, :, :]
        d2 = np.sum(delta * delta, axis=2)
        min_d2[start:end] = np.min(d2, axis=1)

    for thr in thresholds_m:
        output[float(thr)] = float(np.mean(min_d2 <= float(thr) * float(thr)))
    return output


def compute_avg_tcr(gt: np.ndarray, pred: np.ndarray, thresholds_m: List[float]) -> float:
    tcr = compute_tcr_xyz(gt, pred, thresholds_m)
    return finite_mean(tcr.values())


def point_cost_xyzk(a: np.ndarray, b: np.ndarray, yaw_weight: float) -> float:
    dx = float(a[0] - b[0])
    dy = float(a[1] - b[1])
    dz = float(a[2] - b[2])
    da = wrap_rad(float(a[3] - b[3]))
    return math.sqrt(dx * dx + dy * dy + dz * dz + float(yaw_weight) * float(yaw_weight) * da * da)


def softmin3(a: float, b: float, c: float, gamma: float) -> float:
    m = min(a, b, c)
    if not math.isfinite(m):
        return float("inf")
    return -gamma * math.log(
        math.exp(-(a - m) / gamma) + math.exp(-(b - m) / gamma) + math.exp(-(c - m) / gamma)
    ) + m


def compute_soft_dtw_xyzk(gt: np.ndarray, pred: np.ndarray, yaw_weight: float, gamma: float) -> float:
    if gt.shape[0] == 0 or pred.shape[0] == 0:
        return float("nan")

    n = int(pred.shape[0])
    m = int(gt.shape[0])
    prev = np.full((m + 1,), np.inf, dtype=np.float64)
    curr = np.full((m + 1,), np.inf, dtype=np.float64)
    prev[0] = 0.0

    for i in range(1, n + 1):
        curr[0] = np.inf
        pred_i = pred[i - 1]
        for j in range(1, m + 1):
            cost = point_cost_xyzk(pred_i, gt[j - 1], yaw_weight)
            curr[j] = cost + softmin3(curr[j - 1], prev[j], prev[j - 1], gamma)
        prev, curr = curr, prev
    return float(prev[m])


def compute_ndtw(gt: np.ndarray, pred: np.ndarray, eta: float, yaw_weight: float, softdtw_gamma: float) -> float:
    dtw = compute_soft_dtw_xyzk(gt[:, :4], pred[:, :4], yaw_weight=yaw_weight, gamma=softdtw_gamma)
    if not np.isfinite(dtw):
        return float("nan")
    score = math.exp(-dtw / max(float(eta) * float(len(gt)), 1e-6))
    return float(np.clip(score, 0.0, 1.0))


def compute_nsp(gt_xyz: np.ndarray, pred_xyz: np.ndarray) -> float:
    """Normalized progress of the predicted endpoint along the GT polyline."""
    if gt_xyz.shape[0] < 2 or pred_xyz.shape[0] < 1:
        return float("nan")

    endpoint = np.asarray(pred_xyz[-1], dtype=np.float64)
    seg0 = np.asarray(gt_xyz[:-1], dtype=np.float64)
    seg1 = np.asarray(gt_xyz[1:], dtype=np.float64)
    vec = seg1 - seg0
    seg_len2 = np.einsum("ij,ij->i", vec, vec)
    valid = seg_len2 > 1e-12
    if not np.any(valid):
        return 0.0

    seg0_valid = seg0[valid]
    vec_valid = vec[valid]
    seg_len2_valid = seg_len2[valid]
    t = np.clip(np.einsum("ij,ij->i", endpoint[None, :] - seg0_valid, vec_valid) / seg_len2_valid, 0.0, 1.0)
    proj = seg0_valid + t[:, None] * vec_valid
    dist2 = np.einsum("ij,ij->i", proj - endpoint[None, :], proj - endpoint[None, :])
    best_local = int(np.argmin(dist2))
    best_seg = int(np.nonzero(valid)[0][best_local])

    seg_lengths = np.linalg.norm(vec, axis=1)
    total = float(seg_lengths.sum())
    if total <= 1e-12:
        return 0.0

    progress = float(seg_lengths[:best_seg].sum() + t[best_local] * seg_lengths[best_seg])
    return float(np.clip(progress / total, 0.0, 1.0))


def compute_success(gt_xyz: np.ndarray, pred_xyz: np.ndarray, success_thresh_m: float) -> float:
    if gt_xyz.shape[0] == 0 or pred_xyz.shape[0] == 0:
        return float("nan")
    end_dist = float(np.linalg.norm(pred_xyz[-1] - gt_xyz[-1]))
    return 1.0 if end_dist <= float(success_thresh_m) else 0.0


def load_as_single_mesh(path: Path):
    if trimesh is None:
        raise RuntimeError("trimesh is not available; install it to evaluate collision metrics")

    geom = trimesh.load(str(path), force=None, process=False)
    if isinstance(geom, trimesh.Scene):
        dumped = geom.dump(concatenate=True)
        if isinstance(dumped, trimesh.Trimesh):
            mesh = dumped
        else:
            mesh = trimesh.util.concatenate([m for m in dumped if isinstance(m, trimesh.Trimesh)])
    elif isinstance(geom, trimesh.Trimesh):
        mesh = geom
    else:
        raise TypeError(f"Unsupported mesh type: {type(geom)}")
    mesh.remove_unreferenced_vertices()
    return mesh


class CollisionCache:
    def __init__(self, mesh_root: Optional[Path], mesh_rel: str):
        self.mesh_root = mesh_root
        self.mesh_rel = mesh_rel
        self.cache = {}
        self.errors: Dict[str, str] = {}

    @property
    def enabled(self) -> bool:
        return self.mesh_root is not None

    def get_ray(self, env_id: str):
        if self.mesh_root is None:
            raise RuntimeError("mesh_root is not set")
        if env_id not in self.cache:
            mesh_path = self.mesh_root / env_id / self.mesh_rel
            mesh = load_as_single_mesh(mesh_path)
            try:
                from trimesh.ray.ray_pyembree import RayMeshIntersector  # type: ignore

                ray = RayMeshIntersector(mesh)
            except Exception:
                ray = mesh.ray
            self.cache[env_id] = ray
        return self.cache[env_id]

    def collision_flag(self, env_id: Optional[str], pred_xyz: np.ndarray) -> float:
        if not self.enabled:
            return float("nan")
        if not env_id:
            return float("nan")
        if pred_xyz.shape[0] < 2:
            return 0.0

        try:
            ray = self.get_ray(env_id)
            p0 = pred_xyz[:-1].astype(np.float64)
            p1 = pred_xyz[1:].astype(np.float64)
            vec = p1 - p0
            lengths = np.linalg.norm(vec, axis=1)
            valid = lengths > 1e-12
            if not np.any(valid):
                return 0.0

            origins = p0[valid]
            directions = vec[valid] / lengths[valid, None]
            valid_lengths = lengths[valid]
            locations = None
            index_ray = None

            variants = [
                ((), {"origins": origins, "directions": directions, "multiple_hits": False}),
                ((), {"ray_origins": origins, "ray_directions": directions, "multiple_hits": False}),
                ((origins, directions), {"multiple_hits": False}),
                ((), {"origins": origins, "directions": directions}),
                ((), {"ray_origins": origins, "ray_directions": directions}),
                ((origins, directions), {}),
            ]
            for args, kwargs in variants:
                try:
                    locations, index_ray, _ = ray.intersects_location(*args, **kwargs)
                    break
                except Exception:
                    continue

            if locations is None or index_ray is None or len(locations) == 0:
                return 0.0

            t = np.einsum("ij,ij->i", locations - origins[index_ray], directions[index_ray])
            return 1.0 if bool(np.any((t >= -1e-8) & (t <= valid_lengths[index_ray] + 1e-8))) else 0.0
        except Exception as exc:
            self.errors[str(env_id)] = repr(exc)
            return float("nan")


def compute_episode_metrics(
    npz_path: Path,
    thresholds_m: List[float],
    success_thresh_m: float,
    eta: float,
    yaw_weight: float,
    softdtw_gamma: float,
    collision_cache: CollisionCache,
    stride: int,
    max_len: int,
) -> Dict[str, float]:
    task_id, split, env_id = parse_task_split_env(npz_path)
    data = np.load(str(npz_path))
    gt = sanitize_traj_xyzk(data["gt_xyzk"])
    pred = sanitize_traj_xyzk(data["pred_xyzk"])

    if max_len > 0:
        gt = gt[:max_len]
        pred = pred[:max_len]
    if stride > 1:
        gt = gt[::stride]
        pred = pred[::stride]

    gt_xyz = gt[:, :3]
    pred_xyz = pred[:, :3]
    tcr = compute_tcr_xyz(gt, pred, thresholds_m)
    avg_tcr = finite_mean(tcr.values())
    ndtw = compute_ndtw(gt, pred, eta=eta, yaw_weight=yaw_weight, softdtw_gamma=softdtw_gamma)
    nsp = compute_nsp(gt_xyz, pred_xyz)
    sr = compute_success(gt_xyz, pred_xyz, success_thresh_m)
    cr = collision_cache.collision_flag(env_id, pred_xyz)

    gt_len = path_length_xyz(gt_xyz)
    pred_len = path_length_xyz(pred_xyz)
    ratio = gt_len / max(gt_len, pred_len) if np.isfinite(gt_len) and np.isfinite(pred_len) and gt_len > 1e-12 else float("nan")
    spl = sr * ratio if np.isfinite(sr) and np.isfinite(ratio) else float("nan")
    cspl = sr * (1.0 - cr) * ratio if np.isfinite(sr) and np.isfinite(cr) and np.isfinite(ratio) else float("nan")

    record: Dict[str, float] = {
        "task": task_id or "",
        "split": split or "",
        "env_id": env_id or "",
        "avg_tcr": avg_tcr,
        "ndtw": ndtw,
        "nsp": nsp,
        "sr": sr,
        "cr": cr,
        "spl": spl,
        "cspl": cspl,
        "gt_length": gt_len,
        "pred_length": pred_len,
        "num_gt": float(gt.shape[0]),
        "num_pred": float(pred.shape[0]),
    }
    for threshold, value in tcr.items():
        record[f"tcr@{float(threshold):g}"] = value
    return record


def summarize(records: List[Dict[str, float]], keys: List[str]) -> Dict[str, float]:
    summary: Dict[str, float] = {"n": float(len(records))}
    for key in keys:
        summary[key] = finite_mean(float(record.get(key, float("nan"))) for record in records)
    return summary


def grouped_summary(records: List[Dict[str, float]], keys: List[str], group_key: str) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[Dict[str, float]]] = {}
    for record in records:
        group = str(record.get(group_key, "") or "unknown")
        groups.setdefault(group, []).append(record)
    return {group: summarize(items, keys) for group, items in sorted(groups.items())}


def print_table(title: str, rows: Dict[str, Dict[str, float]], metric_keys: List[str]) -> None:
    name_width = max(len(title), max((len(name) for name in rows), default=4))
    headers = ["group", "#traj"] + metric_keys
    widths = [name_width, 7] + [10 for _ in metric_keys]
    print(f"\n{title}")
    print("  ".join(f"{header:>{width}}" for header, width in zip(headers, widths)))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for name, values in rows.items():
        cols = [f"{name:>{name_width}}", f"{int(values.get('n', 0)):>7}"]
        cols.extend(f"{format_float(values.get(key, float('nan'))):>10}" for key in metric_keys)
        print("  ".join(cols))


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate HUGE-Bench rollout outputs with Avg. TCR, nDTW, NSP, CR, and CSPL."
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("./outputs"),
        help="Root output directory containing task_<task_id>/<split>/<env_id>/episode_*/traj_gt_pred_xyzk.npz.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated task ids. Use ids without or with the task_ prefix.",
    )
    parser.add_argument("--tcr_thresholds", type=str, default="1,2,5", help="Comma-separated TCR thresholds in meters.")
    parser.add_argument("--success_thresh", type=float, default=20.0, help="Endpoint threshold in meters for SR/SPL/CSPL.")
    parser.add_argument("--eta", type=float, default=3.0, help="nDTW length normalization factor.")
    parser.add_argument("--yaw_weight", type=float, default=0.1, help="Yaw weight used in XYZA DTW distance.")
    parser.add_argument("--softdtw_gamma", type=float, default=0.5, help="Soft-DTW smoothing parameter used for nDTW.")
    parser.add_argument("--mesh_root", type=Path, default=None, help="Optional root containing <env_id>/<mesh_rel> for CR/CSPL.")
    parser.add_argument("--mesh_rel", type=str, default="terra_ply/simplified_mesh.obj", help="Mesh path relative to each env id.")
    parser.add_argument("--stride", type=int, default=1, help="Optional trajectory subsampling stride.")
    parser.add_argument("--max_len", type=int, default=0, help="Optional maximum trajectory length after loading.")
    parser.add_argument("--per_episode", action="store_true", help="Print per-episode metric rows.")
    parser.add_argument("--json_out", type=Path, default=None, help="Optional path to save detailed JSON results.")
    args = parser.parse_args()

    thresholds_m = parse_number_list(args.tcr_thresholds, DEFAULT_TCR_THRESHOLDS)
    tasks = set(parse_task_list(args.tasks))
    npz_files = [
        path
        for path in find_npz_files(args.out_dir)
        if (parse_task_split_env(path)[0] in tasks and parse_task_split_env(path)[1] in {"seen", "unseen"})
    ]

    if not npz_files:
        print(f"[ERR] No matching traj_gt_pred_xyzk.npz files found under: {args.out_dir}")
        return 1

    collision_cache = CollisionCache(args.mesh_root, args.mesh_rel)
    if args.mesh_root is None:
        print("[INFO] --mesh_root not provided; CR and CSPL will be reported as nan.")
    elif trimesh is None:
        print("[WARN] trimesh is not installed; CR and CSPL will be reported as nan.")

    metric_keys = [f"tcr@{float(thr):g}" for thr in thresholds_m] + ["avg_tcr", "ndtw", "nsp", "sr", "cr", "cspl"]
    records: List[Dict[str, float]] = []
    for index, npz_path in enumerate(npz_files, 1):
        record = compute_episode_metrics(
            npz_path=npz_path,
            thresholds_m=thresholds_m,
            success_thresh_m=float(args.success_thresh),
            eta=float(args.eta),
            yaw_weight=float(args.yaw_weight),
            softdtw_gamma=float(args.softdtw_gamma),
            collision_cache=collision_cache,
            stride=max(int(args.stride), 1),
            max_len=max(int(args.max_len), 0),
        )
        record["path"] = str(npz_path)
        records.append(record)

        if args.per_episode:
            print(
                f"{index:05d} task={record['task']} split={record['split']} env={record['env_id']} "
                f"avg_tcr={format_float(record['avg_tcr'])} ndtw={format_float(record['ndtw'])} "
                f"nsp={format_float(record['nsp'])} sr={format_float(record['sr'])} "
                f"cr={format_float(record['cr'])} cspl={format_float(record['cspl'])}"
            )

    overall = {"overall": summarize(records, metric_keys)}
    by_split = grouped_summary(records, metric_keys, "split")
    by_task = grouped_summary(records, metric_keys, "task")

    print(f"[INFO] out_dir={args.out_dir}")
    print(f"[INFO] episodes={len(records)} tasks={sorted(tasks)} thresholds={thresholds_m}")
    print_table("Overall", overall, metric_keys)
    print_table("By Split", by_split, metric_keys)
    print_table("By Task", by_task, metric_keys)

    if collision_cache.errors:
        print("\n[WARN] Collision metrics failed for some environments:")
        for env_id, error in sorted(collision_cache.errors.items()):
            print(f"  {env_id}: {error}")

    if args.json_out is not None:
        payload = {
            "config": {
                "out_dir": str(args.out_dir),
                "tasks": sorted(tasks),
                "tcr_thresholds": thresholds_m,
                "success_thresh": float(args.success_thresh),
                "eta": float(args.eta),
                "yaw_weight": float(args.yaw_weight),
                "softdtw_gamma": float(args.softdtw_gamma),
                "mesh_root": str(args.mesh_root) if args.mesh_root is not None else None,
                "mesh_rel": args.mesh_rel,
                "stride": max(int(args.stride), 1),
                "max_len": max(int(args.max_len), 0),
            },
            "overall": overall["overall"],
            "by_split": by_split,
            "by_task": by_task,
            "episodes": records,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")
        print(f"\n[INFO] Saved JSON results to: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

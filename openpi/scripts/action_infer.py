# -*- coding: utf-8 -*-
import os
import json
import socket
import math
from pathlib import Path
from argparse import ArgumentParser
from typing import Dict, Any, List, Tuple

import numpy as np
from PIL import Image

# matplotlib (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import imageio.v2 as imageio

# ----------------------------
# Env
# ----------------------------
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_CACHE", "/mnt/hf_cache")
os.environ["HF_LEROBOT_HOME"] = "/mnt/jingyu/lerobot_small"

from openpi.training import config as train_config
from openpi.policies import policy_config
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset


# ----------------------------
# Small utils
# ----------------------------
def _as_str(x) -> str:
    if isinstance(x, (str, np.str_)):
        return str(x)
    a = np.asarray(x)
    if a.dtype == object:
        return str(a.reshape(-1)[0])
    return str(a)


def is_obstacle_task(task_id: str) -> bool:
    return str(task_id).strip().lower() in ("obstacle", "task_obstacle")


def wrap_rad(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def step_state(state_xyza: np.ndarray, action_dxyza: np.ndarray):
    """
    Simple add in SAME frame (GLOBAL frame in this script).

    state/action 第4维含义：
      - 普通任务: kappa/yaw(rad)
      - obstacle: phi(rad)
    """
    s = np.asarray(state_xyza, dtype=np.float32).reshape(4)
    a = np.asarray(action_dxyza, dtype=np.float32).reshape(4)
    s2 = s + a
    s2[3] = wrap_rad(float(s2[3]))
    return s2.astype(np.float32)


def resize_rgb_to_256(rgb_hwc_uint8: np.ndarray):
    img = Image.fromarray(np.asarray(rgb_hwc_uint8, dtype=np.uint8))
    img = img.resize((256, 256), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def save_video_mp4(frames_rgb: List[np.ndarray], out_mp4: str, fps: float = 10.0):
    frames = [np.asarray(fr, dtype=np.uint8) for fr in frames_rgb]
    if len(frames) == 0:
        raise ValueError(f"No frames to save: {out_mp4}")

    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)

    with imageio.get_writer(
        out_mp4,
        fps=max(float(fps), 1.0),
        macro_block_size=2,   # 关键：至少保证宽高为偶数
    ) as writer:
        for fr in frames:
            writer.append_data(fr)


def render_states_to_frames(
    rc,
    states_T4: np.ndarray,
    env_id: str,
    task_id: str,
    tmp_png_path: str,
    obstacle_angle_mode: str = "phi",
) -> List[np.ndarray]:
    frames = []
    for i, st in enumerate(np.asarray(states_T4, dtype=np.float32)):
        render_st = convert_state_for_render(
            state_xyza=st,
            task_id=task_id,
            obstacle_angle_mode=obstacle_angle_mode,
        )
        rc.render(
            t=i,
            state_xyzk=render_st,
            out_path=tmp_png_path,
            env_id=env_id,
            task_id=task_id,
        )
        rgb = np.array(Image.open(tmp_png_path).convert("RGB"))
        frames.append(rgb)
        try:
            os.remove(tmp_png_path)
        except Exception:
            pass
    return frames


# ----------------------------
# Angle utils
# ----------------------------
def maybe_deg_to_rad(yaw_or_phi: float) -> float:
    """
    If the last angle is very likely degrees (large magnitude), convert to rad.
    Otherwise treat as rad.

    普通任务第4维是 yaw/kappa，obstacle 第4维是 phi。
    """
    y = float(yaw_or_phi)
    if abs(y) > 3.5 * math.pi:
        return math.radians(y)
    return y


def use_legacy_obstacle_yaw(task_id: str, obstacle_angle_mode: str) -> bool:
    return is_obstacle_task(task_id) and str(obstacle_angle_mode).strip().lower() == "yaw_legacy"


def convert_state_for_render(
    state_xyza: np.ndarray,
    task_id: str,
    obstacle_angle_mode: str,
) -> np.ndarray:
    """
    The render server expects obstacle state[3] to be phi(rad).
    Legacy converted datasets/checkpoints may instead store yaw(rad), where:
        phi = pi/2 - yaw
    """
    st = np.asarray(state_xyza, dtype=np.float32).reshape(4).copy()
    st[3] = wrap_rad(float(maybe_deg_to_rad(st[3])))
    if use_legacy_obstacle_yaw(task_id, obstacle_angle_mode):
        st[3] = wrap_rad(float(0.5 * math.pi - st[3]))
    return st.astype(np.float32)


# ----------------------------
# Dataset image parsing (robust)
# ----------------------------
def _parse_ds_image_to_hwc_uint8(img) -> np.ndarray:
    """
    Make dataset image into uint8 HWC.
    Accepts float [0,1] or uint8, CHW or HWC.
    """
    a = np.asarray(img)
    if np.issubdtype(a.dtype, np.floating):
        if a.max() <= 1.5:
            a = (255.0 * a).astype(np.uint8)
        else:
            a = np.clip(a, 0, 255).astype(np.uint8)
    else:
        a = a.astype(np.uint8, copy=False)

    # CHW -> HWC
    if a.ndim == 3 and a.shape[0] == 3 and a.shape[-1] != 3:
        a = np.transpose(a, (1, 2, 0))
    return a


def _pick_prompt_and_debug(sample: dict, default_prompt: str):
    """Prefer 'prompt' (training-consistent), then task/instruction, else default."""
    if "prompt" in sample:
        src = "sample[prompt]"
        prompt = _as_str(sample["prompt"])
        used_default = False
    elif "task" in sample:
        src = "sample[task]"
        prompt = _as_str(sample["task"])
        used_default = False
    elif "instruction" in sample:
        src = "sample[instruction]"
        prompt = _as_str(sample["instruction"])
        used_default = False
    else:
        src = "default_prompt"
        prompt = _as_str(default_prompt)
        used_default = True

    if prompt.strip() == "":
        src = "default_prompt(empty_prompt_fallback)"
        prompt = _as_str(default_prompt)
        used_default = True

    return prompt, src, used_default


def _get_first_image_from_sample(sample: dict):
    """
    Try common keys for first frame image in your dataset.
    Returns (image_hwc_uint8 or None, key_used or None)
    """
    candidate_keys = [
        "observation/first_image",
        "first_image",
        "observation/first_rgb",
        "first_rgb",
    ]
    for k in candidate_keys:
        if k in sample:
            try:
                im = _parse_ds_image_to_hwc_uint8(sample[k])
                return im, k
            except Exception:
                pass
    return None, None


def _get_env_id_from_sample(sample: dict) -> str:
    """
    Your LeRobot info.json shows 'env_id' is a feature (string, shape [1]).
    In dataset sample it is typically accessible as sample['env_id'].
    """
    if "env_id" in sample:
        v = _as_str(sample["env_id"]).strip()
        if v:
            return v

    for k in ["observation/env_id", "metadata/env_id", "info/env_id"]:
        if k in sample:
            v = _as_str(sample[k]).strip()
            if v:
                return v

    for k in sample.keys():
        if str(k).endswith("env_id"):
            try:
                v = _as_str(sample[k]).strip()
                if v:
                    return v
            except Exception:
                pass

    return "unknown_env"


def _get_state4_from_sample(sample: dict) -> np.ndarray:
    """
    Robustly read state from dataset sample and return [x,y,z,a] (float32).
    第4维 a:
      - 普通任务: yaw/kappa(rad)
      - obstacle: phi(rad)
    """
    for k in ["state", "observation/state", "obs/state"]:
        if k in sample:
            s = np.asarray(sample[k], dtype=np.float32).reshape(-1)
            if s.size >= 4:
                out = s[:4].copy()
                out[3] = wrap_rad(float(maybe_deg_to_rad(out[3])))
                return out.astype(np.float32)
    raise KeyError(f"state not found; keys={sorted(list(sample.keys()))}")


def _get_step_index_from_sample(sample: dict, fallback_i: int) -> int:
    """Try to recover timestep index from sample; fallback to running index."""
    for k in ["step_index", "frame_index", "t", "time_index", "timestamp_index"]:
        if k in sample:
            try:
                return int(np.asarray(sample[k]).reshape(-1)[0])
            except Exception:
                pass
    return int(fallback_i)


def _get_episode_index(sample: dict) -> int:
    try:
        return int(np.asarray(sample.get("episode_index", -1)).reshape(-1)[0])
    except Exception:
        return -1


# ----------------------------
# NDTW (4D: x,y,z,angle) using DTW with wrapped angle cost
# ----------------------------
def _point_cost_xyza(a4: np.ndarray, b4: np.ndarray, angle_weight: float = 1.0) -> float:
    a = np.asarray(a4, dtype=np.float32).reshape(4)
    b = np.asarray(b4, dtype=np.float32).reshape(4)
    dpos = a[:3] - b[:3]
    dangle = wrap_rad(float(a[3]) - float(b[3]))
    return float(math.sqrt(float(np.dot(dpos, dpos)) + (angle_weight * dangle) ** 2))


def dtw_distance_xyza(pred_T4: np.ndarray, gt_T4: np.ndarray, angle_weight: float = 1.0) -> float:
    """
    DTW distance with custom cost:
        sqrt(||dpos||^2 + (angle_weight * wrap(dangle))^2)
    Uses two-row DP to save memory.
    """
    P = np.asarray(pred_T4, dtype=np.float32)
    G = np.asarray(gt_T4, dtype=np.float32)
    N, M = P.shape[0], G.shape[0]
    inf = np.float32(np.inf)

    prev = np.full((M + 1,), inf, dtype=np.float32)
    curr = np.full((M + 1,), inf, dtype=np.float32)
    prev[0] = np.float32(0.0)

    def softmin(a, b, c, gamma=1.0):
        m = min(a, b, c)
        return -gamma * math.log(
            math.exp(-(a - m) / gamma)
            + math.exp(-(b - m) / gamma)
            + math.exp(-(c - m) / gamma)
        ) + m

    for i in range(1, N + 1):
        curr[0] = inf
        pi = P[i - 1]
        for j in range(1, M + 1):
            cost = np.float32(_point_cost_xyza(pi, G[j - 1], angle_weight=angle_weight))
            curr[j] = cost + softmin(curr[j - 1], prev[j], prev[j - 1], gamma=0.5)
        prev, curr = curr, prev

    return float(prev[M])


def compute_ndtw_xyza(
    gt_T4: np.ndarray, pred_T4: np.ndarray, eta: float = 1.0, angle_weight: float = 1.0
) -> float:
    """
    NDTW = exp(-DTW / (eta * len(gt)))
    eta: distance scale (larger -> more tolerant)
    """
    dtw = dtw_distance_xyza(pred_T4, gt_T4, angle_weight=angle_weight)
    denom = max(float(eta) * float(len(gt_T4)), 1e-6)
    return float(math.exp(-dtw / denom))


# ----------------------------
# 3D plot (GT vs Pred)
# ----------------------------
def _set_axes_equal_3d(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range, 1e-6])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_gt_vs_pred_3d(gt_T4: np.ndarray, pred_T4: np.ndarray, out_png: str, title: str = ""):
    gt = np.asarray(gt_T4, dtype=np.float32)
    pr = np.asarray(pred_T4, dtype=np.float32)
    L = min(len(gt), len(pr))
    gt = gt[:L]
    pr = pr[:L]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], label="GT")
    ax.plot(pr[:, 0], pr[:, 1], pr[:, 2], label="Pred")

    ax.scatter([gt[0, 0]], [gt[0, 1]], [gt[0, 2]], marker="o", s=40)
    ax.scatter([gt[-1, 0]], [gt[-1, 1]], [gt[-1, 2]], marker="x", s=60)
    ax.scatter([pr[0, 0]], [pr[0, 1]], [pr[0, 2]], marker="o", s=40)
    ax.scatter([pr[-1, 0]], [pr[-1, 1]], [pr[-1, 2]], marker="x", s=60)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(True)
    ax.legend()

    all_xyz = np.concatenate([gt[:, :3], pr[:, :3]], axis=0)
    xyz_min = all_xyz.min(axis=0)
    xyz_max = all_xyz.max(axis=0)
    pad = 0.05 * max(float(np.linalg.norm(xyz_max - xyz_min)), 1e-6)
    ax.set_xlim(float(xyz_min[0] - pad), float(xyz_max[0] + pad))
    ax.set_ylim(float(xyz_min[1] - pad), float(xyz_max[1] + pad))
    ax.set_zlim(float(xyz_min[2] - pad), float(xyz_max[2] + pad))
    _set_axes_equal_3d(ax)

    if title:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


# ----------------------------
# Render client (render server writes png)
# ----------------------------
class RenderClient:
    def __init__(self, host="127.0.0.1", port=5555):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self.f = self.sock.makefile("rwb")

    def render(self, t: int, state_xyzk, out_path: str, env_id: str = None, task_id: str = None) -> str:
        req = {
            "cmd": "render",
            "t": int(t),
            "state": [float(x) for x in np.asarray(state_xyzk).reshape(4)],
            "out_path": out_path,
        }
        if env_id is not None:
            req["env_id"] = str(env_id)
        if task_id is not None:
            req["task_id"] = str(task_id)

        self.f.write((json.dumps(req) + "\n").encode("utf-8"))
        self.f.flush()
        resp = json.loads(self.f.readline().decode("utf-8"))
        if not resp.get("ok", False):
            raise RuntimeError(resp.get("err", "render_failed"))
        return resp["out_path"]

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass


# ----------------------------
# Smooth overlapping predictions
# ----------------------------
class ActionPlanAverager:
    """Smooth overlapping predictions (GLOBAL frame)."""

    def __init__(self):
        self.sum = {}  # step -> np.ndarray(4,)
        self.cnt = {}  # step -> int

    def add_chunk(self, start_step: int, actions_h4: np.ndarray):
        acts = np.asarray(actions_h4, dtype=np.float32)
        if acts.ndim == 1:
            acts = acts.reshape(1, 4)
        for i in range(acts.shape[0]):
            step = int(start_step + i)
            a = acts[i].reshape(4).astype(np.float32)
            if step in self.sum:
                self.sum[step] += a
                self.cnt[step] += 1
            else:
                self.sum[step] = a.copy()
                self.cnt[step] = 1

    def has(self, step: int) -> bool:
        return step in self.sum

    def get_mean(self, step: int) -> np.ndarray:
        a = (self.sum[step] / float(self.cnt[step])).astype(np.float32)
        a[3] = wrap_rad(float(a[3]))
        return a

    def prune_before(self, step_threshold: int):
        for k in list(self.sum.keys()):
            if k < step_threshold:
                self.sum.pop(k, None)
                self.cnt.pop(k, None)


# ----------------------------
# One-pass scan: select episodes + collect GT trajs + ep info
# ----------------------------
def scan_dataset_collect(dataset, num_trajs: int, default_prompt: str):
    """
    Returns:
      selected_eps: list[int] (in dataset order)
      ep_info: dict[ep] -> {env_id, prompt, prompt_src, used_default, first_img_hwc_uint8 or None, first_img_key or None}
      gt_trajs: dict[ep] -> np.ndarray(T,4) sorted by step
    """
    want_all = (num_trajs <= 0)
    selected_eps = []
    selected_set = set()
    ep_info = {}
    buf = {}  # ep -> list[(step, state4)]

    last_ep = None
    episode_monotonic = True

    N = len(dataset)
    for i in range(N):
        sample = dataset[i]
        ep = _get_episode_index(sample)
        if ep < 0:
            continue

        if last_ep is not None and ep < last_ep:
            episode_monotonic = False
        last_ep = ep

        if want_all:
            if ep not in selected_set:
                selected_eps.append(ep)
                selected_set.add(ep)
                buf[ep] = []
                env_id = _get_env_id_from_sample(sample)
                prompt, prompt_src, used_default = _pick_prompt_and_debug(sample, default_prompt=default_prompt)
                first_img, first_key = _get_first_image_from_sample(sample)
                ep_info[ep] = {
                    "env_id": env_id,
                    "prompt": prompt,
                    "prompt_src": prompt_src,
                    "used_default": used_default,
                    "first_img": first_img,
                    "first_img_key": first_key,
                }
        else:
            if ep not in selected_set and len(selected_eps) < num_trajs:
                selected_eps.append(ep)
                selected_set.add(ep)
                buf[ep] = []
                env_id = _get_env_id_from_sample(sample)
                prompt, prompt_src, used_default = _pick_prompt_and_debug(sample, default_prompt=default_prompt)
                first_img, first_key = _get_first_image_from_sample(sample)
                ep_info[ep] = {
                    "env_id": env_id,
                    "prompt": prompt,
                    "prompt_src": prompt_src,
                    "used_default": used_default,
                    "first_img": first_img,
                    "first_img_key": first_key,
                }

        if ep in selected_set:
            st4 = _get_state4_from_sample(sample)
            step = _get_step_index_from_sample(sample, fallback_i=len(buf[ep]))
            buf[ep].append((step, st4))

        if (not want_all) and episode_monotonic and (len(selected_eps) == num_trajs):
            last_sel = selected_eps[-1]
            if ep > last_sel:
                break

    gt_trajs = {}
    missing = []
    for ep in selected_eps:
        lst = buf.get(ep, [])
        if len(lst) == 0:
            missing.append(ep)
            continue
        lst.sort(key=lambda x: x[0])
        gt_trajs[ep] = np.stack([s for _, s in lst], axis=0).astype(np.float32)

    if missing:
        raise RuntimeError(f"GT trajectories missing for episodes: {missing}")

    return selected_eps, ep_info, gt_trajs


# ----------------------------
# Checkpoint dir resolution (pick largest numeric subdir)
# ----------------------------
def _find_largest_numeric_subdir(root: Path) -> Path:
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"checkpoint root not found: {root}")
    best = None
    best_n = None
    for p in root.iterdir():
        if not p.is_dir():
            continue
        name = p.name.strip()
        if not name.isdigit():
            continue
        n = int(name)
        if best is None or n > best_n:
            best = p
            best_n = n
    if best is None:
        raise FileNotFoundError(f"no numeric subfolders under: {root}")
    return best


def resolve_checkpoint_dir(task_id: str, checkpoint_dir_arg: str) -> str:
    """
    If checkpoint_dir_arg points to a directory that contains numeric subfolders,
    use the largest numeric subfolder.
    Otherwise, use checkpoint_dir_arg as-is.

    If checkpoint_dir_arg is empty, use:
      /mnt/jingyu/openpi/checkpoints/pi0_{task_id}/task_{task_id}/<largest_numeric_subfolder>
    """
    task_id = str(task_id)

    if checkpoint_dir_arg and checkpoint_dir_arg.strip():
        p = Path(checkpoint_dir_arg).expanduser()
        if p.exists() and p.is_dir():
            try:
                best = _find_largest_numeric_subdir(p)
                return str(best)
            except FileNotFoundError:
                return str(p)
        return str(p)

    base = Path(f"/mnt/jingyu/openpi/checkpoints/pi0_{task_id}/task_{task_id}")
    best = _find_largest_numeric_subdir(base)
    return str(best)


# ----------------------------
# Dataset cache / download status (best-effort)
# ----------------------------
def _list_snapshots(snap_root: Path) -> List[Tuple[float, str]]:
    snaps = []
    if not snap_root.exists():
        return snaps
    for d in snap_root.iterdir():
        if d.is_dir():
            try:
                snaps.append((d.stat().st_mtime, str(d)))
            except Exception:
                snaps.append((0.0, str(d)))
    snaps.sort(key=lambda x: x[0], reverse=True)
    return snaps


def get_dataset_cache_state(repo_id: str) -> Dict[str, Any]:
    repo_id = repo_id.strip()

    home = os.environ.get("HF_LEROBOT_HOME", "").strip()
    cache = os.environ.get("HF_HUB_CACHE", "").strip()

    home_path = (Path(home) / repo_id).expanduser() if home else None
    home_exists = bool(home_path and home_path.exists())

    key_dir = None
    snap_root = None
    snaps = []
    key_exists = False

    if cache:
        key = "datasets--" + repo_id.replace("/", "--")
        key_dir = (Path(cache).expanduser() / key)
        key_exists = key_dir.exists()
        snap_root = key_dir / "snapshots"
        snaps = _list_snapshots(snap_root)

    latest_snap = snaps[0][1] if snaps else ""
    latest_mtime = snaps[0][0] if snaps else 0.0

    return {
        "repo_id": repo_id,
        "home": home,
        "home_path": str(home_path) if home_path is not None else "",
        "home_exists": home_exists,
        "cache": cache,
        "cache_key_dir": str(key_dir) if key_dir is not None else "",
        "cache_key_exists": key_exists,
        "snap_root": str(snap_root) if snap_root is not None else "",
        "snap_count": len(snaps),
        "latest_snapshot": latest_snap,
        "latest_snapshot_mtime": latest_mtime,
    }


def infer_dataset_load_status(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    if after.get("home_exists", False):
        mode = "local(HF_LEROBOT_HOME)"
    elif after.get("cache_key_exists", False):
        mode = "cache(HF_HUB_CACHE)"
    else:
        mode = "unknown"

    downloaded = False
    reasons = []
    if after.get("snap_count", 0) > before.get("snap_count", 0):
        downloaded = True
        reasons.append("snapshot_count_increased")
    if after.get("latest_snapshot_mtime", 0.0) > before.get("latest_snapshot_mtime", 0.0) + 1e-6:
        downloaded = True
        reasons.append("latest_snapshot_mtime_increased")
    if (not before.get("cache_key_exists", False)) and after.get("cache_key_exists", False):
        downloaded = True
        reasons.append("cache_key_dir_created")

    status = "download_or_update" if downloaded else "cache_hit_or_local"

    return {
        "load_mode": mode,
        "load_status": status,
        "load_reasons": ",".join(reasons) if reasons else "",
    }


def resolve_eval_dataset_abs_path(repo_id: str, cache_state_after: Dict[str, Any]) -> str:
    if cache_state_after.get("home_exists", False):
        p = Path(cache_state_after["home_path"]).expanduser()
        try:
            return str(p.resolve())
        except Exception:
            return str(p)

    latest = cache_state_after.get("latest_snapshot", "").strip()
    if latest:
        p = Path(latest).expanduser()
        if p.exists():
            try:
                return str(p.resolve())
            except Exception:
                return str(p)

    key_dir = cache_state_after.get("cache_key_dir", "").strip()
    if key_dir:
        p = Path(key_dir).expanduser()
        try:
            return str(p.resolve()) if p.exists() else str(p)
        except Exception:
            return str(p)

    home = cache_state_after.get("home", "").strip()
    if home:
        return str((Path(home) / repo_id).expanduser())
    return repo_id


def debug_print_dataset_paths(dataset, meta):
    cand_attrs = ["dataset_dir", "root", "data_dir", "local_dir", "cache_dir", "path", "repo_dir"]
    for name in cand_attrs:
        for obj, tag in [(dataset, "dataset"), (meta, "meta")]:
            if hasattr(obj, name):
                v = getattr(obj, name)
                try:
                    p = Path(str(v)).expanduser()
                    print(f"[{tag}.{name}] {p} (abs={p.is_absolute()} exists={p.exists()})")
                except Exception:
                    print(f"[{tag}.{name}] {v}")


# ----------------------------
# Main
# ----------------------------
def main():
    parser = ArgumentParser(
        "OpenPI rollout + save plot/npz/instruction + GT/Pred videos + print config + dataset load status + metrics"
    )

    parser.add_argument("--task_id", type=str, default="obstacle")
    parser.add_argument("--config_name", type=str, default="pi0_obstacle")
    parser.add_argument("--splits", type=str, default="test_seen, test_unseen")
    parser.add_argument("--checkpoint_dir", type=str, default="")

    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5550)

    parser.add_argument("--out_dir", type=str, default="/mnt/jingyu/ECCV_outputs_pi0")

    #  - if <=0: infer from GT trajectory length (per-episode)
    #  - if >0: use this cap (optionally truncate_to_gt)
    parser.add_argument("--max_steps", type=int, default=0)

    # >0: rollout first N episodes; <=0: all episodes
    parser.add_argument("--num_trajs", type=int, default=0)

    parser.add_argument("--default_prompt", type=str, default="fly the drone")
    parser.add_argument("--exec_steps", type=int, default=10)

    parser.add_argument("--smooth_overlap", dest="smooth_overlap", action="store_true", default=True)
    parser.add_argument("--no_smooth_overlap", dest="smooth_overlap", action="store_false")

    # repo_id defaults to task_{task_id}/{split} (per split)
    parser.add_argument("--repo_id", type=str, default="")

    parser.add_argument("--truncate_to_gt", dest="truncate_to_gt", action="store_true", default=True)

    # NDTW params
    parser.add_argument("--ndtw_eta", type=float, default=3.0)
    parser.add_argument("--ndtw_yaw_weight", type=float, default=0.1)
    parser.add_argument(
        "--obstacle_angle_mode",
        type=str,
        default="phi",
        choices=("phi", "yaw_legacy"),
        help="For obstacle task, interpret dataset/model angle as phi (new/correct) or yaw_legacy (old converted data).",
    )

    parser.add_argument("--debug_dataset_paths", action="store_true", default=False)

    args = parser.parse_args()

    if not args.config_name.strip():
        args.config_name = f"pi05_{args.task_id}"

    splits_to_run = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits_to_run:
        splits_to_run = ["test_seen", "test_unseen"]

    args.checkpoint_dir = resolve_checkpoint_dir(args.task_id, args.checkpoint_dir)

    print("========== RUN CONFIG ==========")
    print(f"task_id: {args.task_id}")
    print(f"config_name: {args.config_name}")
    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"splits_to_run: {','.join(splits_to_run)}")
    print(f"angle_mode: {'phi(rad)' if is_obstacle_task(args.task_id) else 'kappa/yaw(rad)'}")
    print(f"obstacle_angle_mode: {args.obstacle_angle_mode}")
    print("---- ENV ----")
    print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', '')}")
    print(f"HF_HUB_CACHE: {os.environ.get('HF_HUB_CACHE', '')}")
    print(f"HF_LEROBOT_HOME: {os.environ.get('HF_LEROBOT_HOME', '')}")
    print("================================")

    config = train_config.get_config(args.config_name)
    policy = policy_config.create_trained_policy(config, checkpoint_dir=Path(args.checkpoint_dir))
    data_config = config.data.create(config.assets_dirs, config.model)

    rc = RenderClient(args.host, args.port)

    try:
        for split in splits_to_run:
            repo_id = args.repo_id.strip() if args.repo_id.strip() else f"task_{args.task_id}/{split}"

            out_root = os.path.join(args.out_dir, f"task_{args.task_id}", str(split))
            os.makedirs(out_root, exist_ok=True)

            before_state = get_dataset_cache_state(repo_id)

            print("---------- DATASET LOAD (BEFORE) ----------")
            print(f"split: {split}")
            print(f"repo_id: {repo_id}")
            print(f"home_exists_before: {before_state['home_exists']}  home_path: {before_state['home_path']}")
            print(f"cache_key_exists_before: {before_state['cache_key_exists']}  cache_key_dir: {before_state['cache_key_dir']}")
            print(f"snap_count_before: {before_state['snap_count']}  latest_snapshot_before: {before_state['latest_snapshot']}")
            print("-------------------------------------------")

            meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
            dataset = lerobot_dataset.LeRobotDataset(
                repo_id,
                delta_timestamps={
                    key: [t / meta.fps for t in range(config.model.action_horizon)]
                    for key in data_config.action_sequence_keys
                },
            )

            after_state = get_dataset_cache_state(repo_id)
            load_info = infer_dataset_load_status(before_state, after_state)
            ds_abs = resolve_eval_dataset_abs_path(repo_id, after_state)

            print("---------- DATASET LOAD (AFTER) -----------")
            print(f"split: {split}")
            print(f"repo_id: {repo_id}")
            print(f"dataset_abs_path: {ds_abs} (exists={Path(ds_abs).exists()})")
            print(f"load_mode: {load_info['load_mode']}")
            print(f"load_status: {load_info['load_status']}")
            if load_info["load_reasons"]:
                print(f"load_reasons: {load_info['load_reasons']}")
            print(f"home_exists_after: {after_state['home_exists']}  home_path: {after_state['home_path']}")
            print(f"cache_key_exists_after: {after_state['cache_key_exists']}  cache_key_dir: {after_state['cache_key_dir']}")
            print(f"snap_count_after: {after_state['snap_count']}  latest_snapshot_after: {after_state['latest_snapshot']}")
            try:
                print(f"dataset_len: {len(dataset)}  meta_fps: {getattr(meta, 'fps', 'NA')}")
            except Exception:
                pass
            if args.debug_dataset_paths:
                debug_print_dataset_paths(dataset, meta)
            print("-------------------------------------------")

            selected_eps, ep_info, gt_trajs = scan_dataset_collect(
                dataset=dataset,
                num_trajs=int(args.num_trajs),
                default_prompt=args.default_prompt,
            )

            n_done = 0
            sum_end_dist = 0.0
            sum_ndtw = 0.0

            for ep in selected_eps:
                info = ep_info[ep]
                env_id = info["env_id"]
                prompt = info["prompt"]
                prompt_src = info["prompt_src"]
                used_default = info["used_default"]
                ds_first_img_hwc = info["first_img"]
                ds_first_key = info["first_img_key"]

                gt_states = gt_trajs[ep]  # (T,4)
                gt_len = int(gt_states.shape[0])

                if args.max_steps <= 0:
                    max_steps_ep = gt_len
                else:
                    max_steps_ep = int(args.max_steps)
                    if args.truncate_to_gt:
                        max_steps_ep = min(max_steps_ep, gt_len)

                if max_steps_ep < 2:
                    continue

                ep_dir = os.path.join(out_root, str(env_id), f"episode_{ep:04d}")
                os.makedirs(ep_dir, exist_ok=True)

                plot_path = os.path.join(ep_dir, "compare_gt_vs_pred_3d.png")
                traj_npz_path = os.path.join(ep_dir, "traj_gt_pred_xyzk.npz")
                instr_path = os.path.join(ep_dir, "instruction.txt")
                pred_video_path = os.path.join(ep_dir, "pred_video.mp4")
                gt_video_path = os.path.join(ep_dir, "gt_video.mp4")

                tmp_png_pred = os.path.join(ep_dir, "_tmp_pred.png")
                tmp_png_gt = os.path.join(ep_dir, "_tmp_gt.png")

                plan = ActionPlanAverager()

                # init state from GT t=0
                st_w = gt_states[0].copy().astype(np.float32)
                st_w[3] = wrap_rad(float(maybe_deg_to_rad(st_w[3])))

                pred_states = [st_w.copy()]

                # render pred t=0
                t = 0
                render_st = convert_state_for_render(
                    state_xyza=st_w,
                    task_id=args.task_id,
                    obstacle_angle_mode=args.obstacle_angle_mode,
                )
                rc.render(t=t, state_xyzk=render_st, out_path=tmp_png_pred, env_id=env_id, task_id=args.task_id)
                rgb_pred = np.array(Image.open(tmp_png_pred).convert("RGB"))
                try:
                    os.remove(tmp_png_pred)
                except Exception:
                    pass

                pred_frames = [rgb_pred.copy()]
                obs_rgb_model = resize_rgb_to_256(rgb_pred)

                if ds_first_img_hwc is not None:
                    first_obs_rgb_model = resize_rgb_to_256(ds_first_img_hwc)
                    first_src = f"dataset[{ds_first_key}]"
                else:
                    first_obs_rgb_model = obs_rgb_model.copy()
                    first_src = "render_t0_fallback"

                with open(instr_path, "w", encoding="utf-8") as f:
                    f.write(f"prompt: {prompt}\n")
                    f.write(f"prompt_src: {prompt_src}\n")
                    f.write(f"used_default: {used_default}\n")
                    f.write(f"first_image_src: {first_src}\n")
                    f.write(f"task_id: {args.task_id}\n")
                    f.write(f"angle_mode: {'phi(rad)' if is_obstacle_task(args.task_id) else 'kappa/yaw(rad)'}\n")
                    f.write(f"obstacle_angle_mode: {args.obstacle_angle_mode}\n")
                    f.write(f"split: {split}\n")
                    f.write(f"repo_id: {repo_id}\n")
                    f.write(f"dataset_abs_path: {ds_abs}\n")
                    f.write(f"dataset_load_mode: {load_info['load_mode']}\n")
                    f.write(f"dataset_load_status: {load_info['load_status']}\n")
                    if load_info["load_reasons"]:
                        f.write(f"dataset_load_reasons: {load_info['load_reasons']}\n")
                    f.write(f"env_id: {env_id}\n")
                    f.write(f"episode_index: {ep}\n")
                    f.write(f"config_name: {args.config_name}\n")
                    f.write(f"checkpoint_dir: {args.checkpoint_dir}\n")
                    f.write(f"max_steps_ep: {max_steps_ep}\n")
                    f.write(f"exec_steps: {args.exec_steps}\n")
                    f.write(f"smooth_overlap: {args.smooth_overlap}\n")
                    f.write(f"truncate_to_gt: {args.truncate_to_gt}\n")
                    f.write(f"ndtw_eta: {args.ndtw_eta}\n")
                    f.write(f"ndtw_yaw_weight: {args.ndtw_yaw_weight}\n")
                    f.write("saved_files: compare_gt_vs_pred_3d.png, traj_gt_pred_xyzk.npz, instruction.txt, pred_video.mp4, gt_video.mp4\n")

                # rollout
                while t < max_steps_ep - 1:
                    example = {
                        "observation/state": st_w,
                        "observation/image": obs_rgb_model,
                        "observation/first_image": first_obs_rgb_model,
                        "prompt": prompt,
                    }
                    out = policy.infer(example)

                    pred = np.asarray(out["actions"], dtype=np.float32)
                    if pred.ndim == 1:
                        pred = pred.reshape(1, 4)

                    horizon = pred.shape[0]
                    if args.smooth_overlap:
                        plan.add_chunk(start_step=t, actions_h4=pred)

                    n_exec = max(1, min(args.exec_steps, horizon, (max_steps_ep - 1 - t)))
                    base_step = t

                    for i in range(n_exec):
                        step_idx = base_step + i

                        if args.smooth_overlap and plan.has(step_idx):
                            act_w = plan.get_mean(step_idx)
                        else:
                            act_w = pred[i].reshape(4).astype(np.float32)
                            act_w[3] = wrap_rad(float(act_w[3]))

                        st_w = step_state(st_w, act_w)
                        st_w[3] = wrap_rad(float(maybe_deg_to_rad(st_w[3])))

                        t += 1
                        if args.smooth_overlap:
                            plan.prune_before(t)

                        render_st = convert_state_for_render(
                            state_xyza=st_w,
                            task_id=args.task_id,
                            obstacle_angle_mode=args.obstacle_angle_mode,
                        )
                        rc.render(
                            t=t,
                            state_xyzk=render_st,
                            out_path=tmp_png_pred,
                            env_id=env_id,
                            task_id=args.task_id,
                        )
                        rgb_pred = np.array(Image.open(tmp_png_pred).convert("RGB"))
                        try:
                            os.remove(tmp_png_pred)
                        except Exception:
                            pass

                        pred_frames.append(rgb_pred.copy())
                        obs_rgb_model = resize_rgb_to_256(rgb_pred)
                        pred_states.append(st_w.copy())

                        if t >= max_steps_ep - 1:
                            break

                pred_states_arr = np.stack(pred_states, axis=0).astype(np.float32)

                # Align GT / Pred length
                L = min(int(gt_states.shape[0]), int(pred_states_arr.shape[0]))
                gt_al = np.asarray(gt_states[:L], dtype=np.float32)
                pr_al = np.asarray(pred_states_arr[:L], dtype=np.float32)
                pred_frames_al = pred_frames[:L]

                # file #2: trajectories
                np.savez_compressed(traj_npz_path, gt_xyzk=gt_al, pred_xyzk=pr_al)

                # save GT video
                gt_frames_al = render_states_to_frames(
                    rc=rc,
                    states_T4=gt_al,
                    env_id=env_id,
                    task_id=args.task_id,
                    tmp_png_path=tmp_png_gt,
                    obstacle_angle_mode=args.obstacle_angle_mode,
                )

                fps_video = float(getattr(meta, "fps", 10))
                save_video_mp4(pred_frames_al, pred_video_path, fps=fps_video)
                save_video_mp4(gt_frames_al, gt_video_path, fps=fps_video)

                # file #1: visualization
                ndtw = compute_ndtw_xyza(
                    gt_al,
                    pr_al,
                    eta=float(args.ndtw_eta),
                    angle_weight=float(args.ndtw_yaw_weight),
                )
                end_dist = float(np.linalg.norm(pr_al[-1, :3] - gt_al[-1, :3]))
                plot_gt_vs_pred_3d(
                    gt_al,
                    pr_al,
                    plot_path,
                    title=f"ep={ep} | end_dist={end_dist:.3f} | ndtw={ndtw:.3f}",
                )

                n_done += 1
                sum_end_dist += end_dist
                sum_ndtw += ndtw
                avg_end = sum_end_dist / max(n_done, 1)
                avg_nd = sum_ndtw / max(n_done, 1)

                print(f"end_dist:{end_dist:.6f} ndtw:{ndtw:.6f}")
                print(f"avg_end_dist:{avg_end:.6f} avg_ndtw:{avg_nd:.6f}")

    finally:
        rc.close()


if __name__ == "__main__":
    main()

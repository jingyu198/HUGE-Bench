# -*- coding: utf-8 -*-
# 渲染 DJI 3DGS 的 PLY + 随机生成的 pose（traj_random.txt）
#
# 路径与运行方式：
# - 用户只需要提供：--env_id 和 --task_id
# - 自动推导路径：
#   * traj_random.txt: <data_root>/data_traj/task_<task_id>/<env_id>/traj_random.txt
#   * traj_meta.txt  : <data_root>/data_traj/task_<task_id>/<env_id>/traj_meta.txt（若不存在则 fallback）
#   * 3DGS PLY       : <data_root>/data_3d/<env_id>/3dgs_ply/point_cloud_utm50.ply
#   * mesh           : <data_root>/data_3d/<env_id>/terra_ply/merged_mesh.obj
#   * RGB 输出目录    : <data_root>/data_traj/task_<task_id>/<env_id>/render_img
#   * Depth 输出目录  : <data_root>/data_traj/task_<task_id>/<env_id>/render_depth
#   * wash_res.txt   : 与 traj_random.txt 同级目录
#
# 渲染逻辑（修改后）：
# - 删除原先“RGB/Depth 黑像素占比判无效轨迹”的所有计算/清理逻辑
# - wash_res.txt 默认记录“全部轨迹”（每条轨迹一个 traj_id）
# - 增加天空背景合成：
#   * Optional sky image path is configured with HUGE_SKY_IMAGE_PATH.
#   * 采用简单屏幕空间背景：不考虑天空随相机旋转/平移变化，仅作为背景
#   * 合成方式优先使用 renderer 输出的 alpha/accumulation（若存在），否则用“接近背景色”的像素做 mask
#
# 保持不变：
# 0) 原有 3DGS 渲染逻辑保持不变（输出文件名/缩放/模式均不变，默认 scale=1.0）
# 1) 多卡时只让 rank0 打印（其余 GPU 静默）
# 2) 增加 ETA 预估（rank0 每隔 args.log_interval 秒打印一次）
# 3) 同时渲染 mesh 深度（Open3D OffscreenRenderer），保存为 16-bit PNG（单位：cm）
# 4) 可调 RGB 输出分辨率 --img_scale
# 5) 3DGS 渲染可对内参做缩放 --intr_scale（fx/fy/cx/cy 同比例缩放）
# 6) mesh depth 渲染也应用同一个 --intr_scale（确保 3DGS 与 mesh 使用同一内参缩放）
#
# 依赖：
#   open3d, opencv-python（cv2）
# 若环境中缺失 open3d/cv2，会自动跳过深度渲染，但不影响原有 3DGS 渲染。

import torch.multiprocessing as mp
import os
import gc
import math
import time
import numpy as np
import torch

from torch import nn
import xml.etree.ElementTree as ET  # 可以保留，不影响
from argparse import ArgumentParser
from typing import Optional
from plyfile import PlyData
import torchvision
from PIL import Image

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from arguments import PipelineParams
from scene.cameras import Camera
from utils.graphics_utils import getProjectionMatrix
from utils.graphics_utils import getProjectionMatrix_with_principal
import torch.nn.functional as F

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


# =========================================================
# Open3D Mesh Depth Render (optional)
# =========================================================
MESH_RENDER_AVAILABLE = True
try:
    import open3d as o3d
    import cv2
except Exception:
    MESH_RENDER_AVAILABLE = False
    o3d = None
    cv2 = None


# =========================================================
# 天空背景（固定路径）
# =========================================================
SKY_IMAGE_PATH = os.environ.get(
    "HUGE_SKY_IMAGE_PATH",
    os.path.join(os.environ.get("HUGE_DATA_ROOT", "."), "data_3d", "sky.jpg"),
)
SKY_MASK_EPS = 1e-3  # fallback 用：与背景色接近则判为背景


# =========================================================
# 日志控制：只让 rank0 输出 + ETA 控制
# =========================================================
WORKER_RANK = 0
ONLY_RANK0_LOG = True


def rank_print(msg: str, rank: int = 0, only_rank0: bool = True):
    """only_rank0=True 时，只允许 rank==0 打印。"""
    if (not only_rank0) or (rank == 0):
        print(msg, flush=True)


def log(msg: str):
    """Worker 内部统一用 log()，自动遵循 only rank0 打印规则。"""
    global WORKER_RANK, ONLY_RANK0_LOG
    rank_print(msg, rank=WORKER_RANK, only_rank0=ONLY_RANK0_LOG)


def fmt_seconds(sec: float) -> str:
    if sec is None or not np.isfinite(sec) or sec < 0:
        return "N/A"
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h:d}h{m:02d}m{s:02d}s"
    return f"{m:02d}m{s:02d}s"


# =========================================================
# 小工具：路径解析
# =========================================================
def _maybe_prefix_base(arg_value: str, base_dir: str) -> str:
    """
    如果 arg_value 本身已经是路径（含 / 或 \\ 或 以 . 开头 或 以 ~ 开头 或绝对路径），就不动；
    否则拼到 base_dir 下。
    """
    s = (arg_value or "").strip()
    if not s:
        return s
    if os.path.isabs(s) or s.startswith("./") or s.startswith("../") or s.startswith("~") or ("/" in s) or ("\\" in s):
        return s
    return os.path.join(base_dir, s)


def normalize_task_id(task_id: str) -> str:
    """支持传 0/hl/building 或 task_0/task_hl/task_building，统一转成目录名 task_xxx"""
    s = str(task_id).strip()
    if not s:
        raise ValueError("task_id 不能为空")
    if s.startswith("task_"):
        return s
    return f"task_{s}"


# =========================================================
# 轨迹 meta：加载 traj_id, start, end
# =========================================================
def load_traj_meta(traj_meta_path: str):
    """
    traj_meta.txt:
      # traj_id loc_name pose_id_start pose_id_end
      0 basketball court 0 107
      ...
    loc_name 可能带空格，所以 start/end 从最后两列取
    """
    metas = []
    if not traj_meta_path or (not os.path.isfile(traj_meta_path)):
        return metas

    with open(traj_meta_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            try:
                traj_id = int(parts[0])
                pose_id_start = int(parts[-2])
                pose_id_end = int(parts[-1])
                metas.append({"traj_id": traj_id, "start": pose_id_start, "end": pose_id_end})
            except Exception as e:
                rank_print(f"[WARN] traj_meta parse failed: {s} ({e})", rank=0, only_rank0=False)
                continue

    metas.sort(key=lambda x: (x["start"], x["traj_id"]))
    return metas


def build_trajectories_from_meta(poses_all: list, traj_metas: list):
    """
    返回：list of dict:
      {"traj_id": int, "poses": [pose_dict...], "start": int, "end": int}
    若 traj_metas 为空：fallback -> 每个 pose 当作一个“单帧轨迹”，保证仍可运行
    """
    pose_by_id = {int(p["id"]): p for p in poses_all}

    trajs = []
    if traj_metas:
        for m in traj_metas:
            st, ed = int(m["start"]), int(m["end"])
            plist = []
            for pid in range(st, ed + 1):
                p = pose_by_id.get(pid, None)
                if p is not None:
                    plist.append(p)
            if len(plist) == 0:
                continue
            trajs.append({"traj_id": int(m["traj_id"]), "poses": plist, "start": st, "end": ed})
    else:
        for p in poses_all:
            pid = int(p["id"])
            trajs.append({"traj_id": pid, "poses": [p], "start": pid, "end": pid})

    return trajs


# =========================================================
# 多卡渲染 worker（按“轨迹”分配任务）
# - 不再做黑像素/深度无效检测
# - wash_res 默认包含全部轨迹
# =========================================================
def render_worker(
    rank,
    world_size,
    args,
    intrinsics,
    trajs,             # 轨迹列表（每条含 poses）
    pipeline,
    counter,
    lock,
    total_units: int,  # 进度总量（按 pose 计数）
    start_time_main: float,
    tmp_wash_dir: str, # 每个 rank 写一个临时 traj_id 列表（最终合并成 wash_res）
):
    global WORKER_RANK, ONLY_RANK0_LOG
    WORKER_RANK = rank
    ONLY_RANK0_LOG = True  # 多卡时只让 rank0 打印

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    log(f"[RANK {rank}] device = {device}")

    # 每个进程自己的 cache（进程间不共享）
    global GAUSSIANS_CACHE, PLY_PATH_CACHE, DEVICE_CACHE
    GAUSSIANS_CACHE = None
    PLY_PATH_CACHE = None
    DEVICE_CACHE = None

    global O3D_RENDERER_CACHE, O3D_MESH_PATH_CACHE, O3D_INTR_KEY_CACHE
    O3D_RENDERER_CACHE = None
    O3D_MESH_PATH_CACHE = None
    O3D_INTR_KEY_CACHE = None

    global SKY_TENSOR_CACHE
    SKY_TENSOR_CACHE = {}

    os.makedirs(args.out_dir, exist_ok=True)
    if args.enable_mesh_depth and MESH_RENDER_AVAILABLE:
        os.makedirs(args.depth_dir, exist_ok=True)

    traj_ids_this_rank = []
    last_log_t = time.time()

    n_traj = len(trajs)
    for t_i in range(rank, n_traj, world_size):
        traj = trajs[t_i]
        traj_id = int(traj["traj_id"])
        poses = traj["poses"]
        if not poses:
            continue

        # 记录：默认全部轨迹都写入 wash_res（不做筛选）
        traj_ids_this_rank.append(traj_id)

        for idx, p in enumerate(poses):
            pid = int(p["id"])
            C_local = p["C_local"]
            omega = p["omega"]
            phi = p["phi"]
            kappa = p["kappa"]

            R_w2c = opk_to_R_world2cam(omega, phi, kappa)
            t_w2c = -R_w2c @ C_local

            cam_info = {
                **intrinsics,
                "R_w2c": R_w2c.astype(np.float32),
                "T_w2c": t_w2c.astype(np.float32),
                "img_scale": float(args.img_scale),
                "intr_scale": float(args.intr_scale),
            }

            out_name = f"{pid:06d}.png"
            out_path = os.path.join(args.out_dir, out_name)

            # RGB render（带天空合成）
            try:
                render_single_view_from_cam_info(
                    ply_path=args.ply_path,
                    cam_info=cam_info,
                    pipeline=pipeline,
                    out_path=out_path,
                    white_background=args.white_bg,
                    device=device,
                    sky_path=SKY_IMAGE_PATH,
                )
            except Exception as e:
                log(f"[RANK {rank}] [WARN] render failed traj={traj_id} pose_id={pid}: {e}")

            # Mesh depth render（可选）
            if args.enable_mesh_depth and MESH_RENDER_AVAILABLE:
                try:
                    depth_out_path = os.path.join(args.depth_dir, out_name)
                    render_mesh_depth_single_view(
                        mesh_path=args.mesh_path,
                        intrinsics=intrinsics,
                        intr_scale=float(args.intr_scale),
                        C_local=C_local.astype(np.float64),
                        R_w2c=R_w2c.astype(np.float64),
                        out_depth_path=depth_out_path,
                        znear=args.mesh_znear,
                        zfar=args.mesh_zfar,
                        scale=args.depth_scale,
                        save_vis=args.depth_save_vis,
                        rank=rank,
                        return_depth_u16_cm=False,
                    )
                except Exception as e:
                    log(f"[RANK {rank}] [WARN] mesh depth render failed traj={traj_id} pose_id={pid}: {e}")

            # 更新进度（当前帧已处理）
            with lock:
                counter.value += 1
                done = counter.value

            # ETA log（rank0 节流）
            now = time.time()
            if rank == 0 and args.log_interval > 0 and (now - last_log_t) >= args.log_interval:
                elapsed = now - start_time_main
                speed = done / max(elapsed, 1e-6)
                remain = total_units - done
                eta = remain / max(speed, 1e-6)
                log(f"[RANK 0] progress {done}/{total_units} | {speed:.3f} unit/s | ETA {fmt_seconds(eta)}")
                last_log_t = now

    # 每个 rank 写临时 traj_id 列表
    os.makedirs(tmp_wash_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_wash_dir, f"wash_rank{rank}.txt")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            for tid in traj_ids_this_rank:
                f.write(f"{int(tid)}\n")
    except Exception as e:
        if rank == 0:
            log(f"[WARN] write tmp wash file failed: {tmp_path} ({e})")

    # rank0 最后补一条完成信息
    if rank == 0:
        with lock:
            done = counter.value
        elapsed = time.time() - start_time_main
        speed = done / max(elapsed, 1e-6)
        log(f"[RANK 0] done {done}/{total_units} | avg {speed:.3f} unit/s | elapsed {fmt_seconds(elapsed)}")


# --------------------------
# 全局缓存：避免重复读取 PLY
# --------------------------
GAUSSIANS_CACHE = None
PLY_PATH_CACHE = None
DEVICE_CACHE = None


# --------------------------
# Open3D renderer cache（每个进程一份）
# --------------------------
O3D_RENDERER_CACHE = None
O3D_MESH_PATH_CACHE = None
O3D_INTR_KEY_CACHE = None


# --------------------------
# Sky tensor cache（每个进程一份）
# --------------------------
SKY_TENSOR_CACHE = {}  # key: (W,H,str(device), sky_path) -> torch.Tensor [3,H,W] on device


def _cuda_empty_cache_if_needed(device: Optional[torch.device] = None):
    if not torch.cuda.is_available():
        return
    if device is not None and str(device).startswith("cpu"):
        return
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _release_gaussians_cache(reason: str = ""):
    global GAUSSIANS_CACHE, PLY_PATH_CACHE, DEVICE_CACHE

    old_gaussians = GAUSSIANS_CACHE
    old_ply_path = PLY_PATH_CACHE
    old_device = DEVICE_CACHE

    GAUSSIANS_CACHE = None
    PLY_PATH_CACHE = None
    DEVICE_CACHE = None

    if old_gaussians is not None:
        try:
            del old_gaussians
        except Exception:
            pass

    gc.collect()
    _cuda_empty_cache_if_needed(old_device)

    if reason:
        log(f"[INFO] released gaussians cache ({reason}) from {old_ply_path}")


def _get_sky_tensor(sky_path: str, W: int, H: int, device: torch.device):
    """
    读取 sky_path，并 resize 到 (W,H)，返回 [3,H,W] float32 in [0,1] on device。
    缓存到 SKY_TENSOR_CACHE，避免每帧重复 IO。
    """
    global SKY_TENSOR_CACHE
    key = (int(W), int(H), str(device), str(sky_path))
    if key in SKY_TENSOR_CACHE:
        return SKY_TENSOR_CACHE[key]

    if (not sky_path) or (not os.path.isfile(sky_path)):
        log(f"[WARN] sky image not found: {sky_path} -> disable sky compositing")
        SKY_TENSOR_CACHE[key] = None
        return None

    img = Image.open(sky_path).convert("RGB")
    img = img.resize((int(W), int(H)), Image.BILINEAR)
    sky = torchvision.transforms.functional.to_tensor(img).to(device=device, dtype=torch.float32)  # [3,H,W]
    SKY_TENSOR_CACHE[key] = sky
    return sky


# -------------------------------------------------------
# 1. 读取 PLY -> GaussianModel
# -------------------------------------------------------
def load_dji_ply_into_gaussians(gaussians: GaussianModel, ply_path: str, device: torch.device):
    log(f"[INFO] loading PLY: {ply_path}")
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data
    n_pts = v.shape[0]

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
    opacity = np.asarray(v["opacity"], dtype=np.float32)[..., None]
    scale0 = np.asarray(v["scale_0"], dtype=np.float32)

    xyz_t = torch.tensor(xyz, dtype=torch.float32, device=device)
    features_dc = torch.tensor(f_dc, dtype=torch.float32, device=device)[:, :, None]
    features_rest = torch.zeros((n_pts, 3, 0), dtype=torch.float32, device=device)
    opacities_t = torch.tensor(opacity, dtype=torch.float32, device=device)

    scales = np.stack([scale0, scale0, scale0], axis=1).astype(np.float32)
    scales_t = torch.tensor(scales, dtype=torch.float32, device=device)

    rots = torch.zeros((n_pts, 4), dtype=torch.float32, device=device)
    rots[:, 0] = 1.0

    gaussians.max_sh_degree = 0
    gaussians.active_sh_degree = 0

    gaussians._xyz = nn.Parameter(xyz_t.requires_grad_(False))
    gaussians._features_dc = nn.Parameter(features_dc.transpose(1, 2).contiguous().requires_grad_(False))
    gaussians._features_rest = nn.Parameter(features_rest.transpose(1, 2).contiguous().requires_grad_(False))
    gaussians._opacity = nn.Parameter(opacities_t.requires_grad_(False))
    gaussians._scaling = nn.Parameter(scales_t.requires_grad_(False))
    gaussians._rotation = nn.Parameter(rots.requires_grad_(False))

    gaussians.max_radii2D = torch.zeros((n_pts,), device=device)
    log(f"[INFO] gaussians loaded: {n_pts}")


# -------------------------------------------------------
# 2. OPK -> R_w2c
# -------------------------------------------------------
def opk_to_R_world2cam(omega_deg: float, phi_deg: float, kappa_deg: float) -> np.ndarray:
    om = math.radians(omega_deg)
    ph = math.radians(phi_deg)
    ka = math.radians(kappa_deg)

    cos_o, sin_o = math.cos(om), math.sin(om)
    cos_p, sin_p = math.cos(ph), math.sin(ph)
    cos_k, sin_k = math.cos(ka), math.sin(ka)

    Rx = np.array(
        [[1.0, 0.0, 0.0],
         [0.0, cos_o, sin_o],
         [0.0, -sin_o, cos_o]],
        dtype=np.float32,
    )
    Ry = np.array(
        [[cos_p, 0.0, -sin_p],
         [0.0, 1.0, 0.0],
         [sin_p, 0.0, cos_p]],
        dtype=np.float32,
    )
    Rz = np.array(
        [[cos_k, sin_k, 0.0],
         [-sin_k, cos_k, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return Rz @ Ry @ Rx


# -------------------------------------------------------
# 2.1 从 txt 读取内参
# -------------------------------------------------------
def parse_intrinsics_from_txt(txt_path: str):
    width = height = fx = fy = cx = cy = None

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if line.startswith("#"):
            content = line.lstrip("#").strip()
            if content.lower().startswith("intrinsics"):
                j = i + 1
                while j < n:
                    nl = lines[j].strip()
                    if not nl:
                        j += 1
                        continue
                    if nl.startswith("#"):
                        nl = nl.lstrip("#").strip()
                    parts = nl.split()
                    if len(parts) >= 6:
                        width = int(float(parts[0]))
                        height = int(float(parts[1]))
                        fx = float(parts[2])
                        fy = float(parts[3])
                        cx = float(parts[4])
                        cy = float(parts[5])
                        break
                    j += 1
                break
        i += 1

    if width is None:
        raise RuntimeError(f"在 {txt_path} 中未找到 intrinsics 信息，请确认文件头部注释已写入内参。")

    FoVx = 2.0 * math.atan(0.5 * width / fx)
    FoVy = 2.0 * math.atan(0.5 * height / fy)

    rank_print(
        f"[INFO] intrinsics: width={width}, height={height}, fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}",
        rank=0,
        only_rank0=False,
    )
    return {"width": width, "height": height, "fx": fx, "fy": fy, "cx": cx, "cy": cy, "FoVx": FoVx, "FoVy": FoVy}


# -------------------------------------------------------
# 2.2 从 txt 读取 pose
# -------------------------------------------------------
def load_random_poses(txt_path: str):
    poses = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            pid = int(parts[0])
            x_rel = float(parts[1])
            y_rel = float(parts[2])
            z_rel = float(parts[3])
            omega = float(parts[4])
            phi = float(parts[5])
            kappa = float(parts[6])

            C_local = np.array([x_rel, y_rel, z_rel], dtype=np.float32)
            poses.append({"id": pid, "C_local": C_local, "omega": omega, "phi": phi, "kappa": kappa})

    rank_print(f"[INFO] poses loaded: {len(poses)} from {txt_path}", rank=0, only_rank0=False)
    return poses


def _safe_scale(x: float, default: float = 1.0) -> float:
    x = float(x)
    if (not np.isfinite(x)) or x <= 0:
        return float(default)
    return x


def _scaled_wh(W: int, H: int, s: float):
    W2 = max(1, int(round(int(W) * float(s))))
    H2 = max(1, int(round(int(H) * float(s))))
    return W2, H2


def make_3dgs_camera_from_info(cam_info: dict, device: torch.device) -> Camera:
    W0, H0 = int(cam_info["width"]), int(cam_info["height"])

    intr_scale = _safe_scale(cam_info.get("intr_scale", 1.0), default=1.0)
    W, H = _scaled_wh(W0, H0, intr_scale)

    fx = float(cam_info["fx"]) * intr_scale
    fy = float(cam_info["fy"]) * intr_scale
    cx = float(cam_info["cx"]) * intr_scale
    cy = float(cam_info["cy"]) * intr_scale

    FoVx = 2.0 * math.atan(0.5 * W / fx)
    FoVy = 2.0 * math.atan(0.5 * H / fy)

    R_w2c = cam_info["R_w2c"]
    T_w2c = cam_info["T_w2c"]

    R = R_w2c.T
    T = T_w2c

    dummy_image = Image.new("RGB", (W, H), (0, 0, 0))

    cam = Camera(
        resolution=[W, H],
        colmap_id=0,
        R=R,
        T=T,
        FoVx=FoVx,
        FoVy=FoVy,
        depth_params=None,
        image=dummy_image,
        invdepthmap=None,
        image_name="DJI_View",
        uid=0,
        trans=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        scale=1.0,
        data_device=str(device),
        train_test_exp=False,
        is_test_dataset=False,
        is_test_view=False,
    )

    cam.znear = 0.1
    cam.zfar = 1000.0

    cam.projection_matrix = (
        getProjectionMatrix_with_principal(
            znear=cam.znear,
            zfar=cam.zfar,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=W,
            height=H,
        )
        .transpose(0, 1)
        .contiguous()
        .to(device)
    )

    cam.world_view_transform = cam.world_view_transform.to(device).contiguous()
    cam.full_proj_transform = cam.world_view_transform @ cam.projection_matrix
    return cam


def _alpha_from_two_pass(render_black: torch.Tensor, render_white: torch.Tensor) -> torch.Tensor:
    """
    由黑底/白底两次渲染反推 alpha。
    render_black/render_white: [3,H,W] float in [0,1]
    返回 alpha: [1,H,W] in [0,1]
    """
    diff = (render_white - render_black).clamp(0.0, 1.0)     # ~ (1-alpha)
    one_minus_alpha = diff.mean(dim=0, keepdim=True)         # [1,H,W]
    alpha = (1.0 - one_minus_alpha).clamp(0.0, 1.0)
    return alpha


def _composite_with_sky(render_black: torch.Tensor, alpha: torch.Tensor, sky: torch.Tensor,
                        alpha_cut: float = 0.0, alpha_gamma: float = 1.0) -> torch.Tensor:
    """
    sky: [3,H,W]
    alpha: [1,H,W]
    alpha_cut: 可选，把很小的 alpha 当作 0（去掉灰雾/黑雾），例如 0.02~0.08
    alpha_gamma: 可选，调整边缘软硬，>1 更“硬”，<1 更“软”
    """
    a = alpha
    if alpha_cut > 0:
        a = ((a - alpha_cut) / (1.0 - alpha_cut)).clamp(0.0, 1.0)
    if alpha_gamma != 1.0:
        a = a.clamp(0.0, 1.0) ** float(alpha_gamma)

    out = render_black + (1.0 - a) * sky
    return out.clamp(0.0, 1.0)


def render_single_view_from_cam_info(
    ply_path: str,
    cam_info: dict,
    pipeline,
    out_path: str,
    white_background: bool = False,   # 这个参数对“最终背景”不再重要，最终背景是 sky
    device: torch.device = None,
    sky_path: str = SKY_IMAGE_PATH,
):
    global GAUSSIANS_CACHE, PLY_PATH_CACHE, DEVICE_CACHE

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if GAUSSIANS_CACHE is not None and (PLY_PATH_CACHE != ply_path or DEVICE_CACHE != device):
        _release_gaussians_cache(reason="scene/device switch")

    if GAUSSIANS_CACHE is None or PLY_PATH_CACHE != ply_path or DEVICE_CACHE != device:
        log(f"[INFO] init gaussians on {device}")
        gaussians = GaussianModel(sh_degree=0, optimizer_type="default")
        load_dji_ply_into_gaussians(gaussians, ply_path, device)
        GAUSSIANS_CACHE = gaussians
        PLY_PATH_CACHE = ply_path
        DEVICE_CACHE = device
    else:
        gaussians = GAUSSIANS_CACHE

    cam = None
    bg_black = None
    bg_white = None
    out_b = None
    out_w = None
    rb = None
    rw = None
    sky = None
    alpha = None
    comp = None
    fallback = None

    try:
        cam = make_3dgs_camera_from_info(cam_info, device)

        # 先准备天空（按保存分辨率来）
        # 注意：render 的分辨率由 intr_scale 决定，save 的分辨率由 img_scale 决定
        # 我们做两次 render 后再统一 resize，保证 alpha 和 sky 对齐
        scale = _safe_scale(cam_info.get("img_scale", 1.0), default=1.0)

        # 两次渲染：黑底 & 白底
        bg_black = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        bg_white = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        with torch.no_grad():
            out_b = render(
                cam, gaussians, pipeline, bg_black,
                use_trained_exp=False, separate_sh=SPARSE_ADAM_AVAILABLE,
            )
            out_w = render(
                cam, gaussians, pipeline, bg_white,
                use_trained_exp=False, separate_sh=SPARSE_ADAM_AVAILABLE,
            )

            rb = out_b["render"]  # [3,H,W]
            rw = out_w["render"]  # [3,H,W]

            # resize 到保存分辨率（两张都一致 resize）
            if scale != 1.0:
                H, W = rb.shape[1], rb.shape[2]
                new_H = max(1, int(round(H * scale)))
                new_W = max(1, int(round(W * scale)))
                rb = F.interpolate(rb.unsqueeze(0), size=(new_H, new_W), mode="bilinear", align_corners=False)[0]
                rw = F.interpolate(rw.unsqueeze(0), size=(new_H, new_W), mode="bilinear", align_corners=False)[0]
            else:
                new_H, new_W = rb.shape[1], rb.shape[2]

            # sky resize 到保存分辨率
            sky = _get_sky_tensor(sky_path=sky_path, W=new_W, H=new_H, device=device)
            if sky is None:
                # No sky requested/found: preserve the caller's original background choice.
                fallback = rw if bool(white_background) else rb
                torchvision.utils.save_image(fallback.detach().cpu(), out_path)
                return

            # 反推 alpha + 合成
            alpha = _alpha_from_two_pass(rb, rw)

            # 可选：去掉灰雾/黑雾（你截图那种灰雾，建议从 0.03 开始试）
            alpha_cut = getattr(cam_info, "alpha_cut", 0.03) if False else 0.03
            alpha_gamma = 1.8

            comp = _composite_with_sky(rb, alpha, sky, alpha_cut=alpha_cut, alpha_gamma=alpha_gamma)

            torchvision.utils.save_image(comp.detach().cpu(), out_path)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            _cuda_empty_cache_if_needed(device)
        raise
    finally:
        out_b = None
        out_w = None
        rb = None
        rw = None
        sky = None
        alpha = None
        comp = None
        fallback = None
        bg_black = None
        bg_white = None
        cam = None
        gc.collect()
        _cuda_empty_cache_if_needed(device)


# =========================================================
# Mesh Depth Rendering (Open3D OffscreenRenderer)
# =========================================================
def _o3d_intr_key(intrinsics: dict, intr_scale: float):
    s = _safe_scale(intr_scale, default=1.0)
    W0 = int(intrinsics["width"])
    H0 = int(intrinsics["height"])
    W, H = _scaled_wh(W0, H0, s)
    FX = float(intrinsics["fx"]) * s
    FY = float(intrinsics["fy"]) * s
    CX = float(intrinsics["cx"]) * s
    CY = float(intrinsics["cy"]) * s
    return (W, H, FX, FY, CX, CY)


def _get_o3d_renderer(mesh_path: str, intrinsics: dict, intr_scale: float, rank: int = 0):
    global O3D_RENDERER_CACHE, O3D_MESH_PATH_CACHE, O3D_INTR_KEY_CACHE

    if not MESH_RENDER_AVAILABLE:
        raise RuntimeError("Open3D/cv2 not available, cannot render mesh depth.")

    key = _o3d_intr_key(intrinsics, intr_scale)
    need_init = (
        O3D_RENDERER_CACHE is None
        or O3D_MESH_PATH_CACHE != mesh_path
        or O3D_INTR_KEY_CACHE != key
    )

    if not need_init:
        return O3D_RENDERER_CACHE

    W, H, *_ = key

    if rank == 0:
        log(f"[INFO] init Open3D OffscreenRenderer: {W}x{H}")
        log(f"[INFO] loading mesh: {mesh_path}")

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise RuntimeError(f"mesh is empty: {mesh_path}")
    mesh.compute_vertex_normals()
    if not mesh.has_vertex_colors():
        mesh.paint_uniform_color([0.8, 0.8, 0.8])

    renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"

    renderer.scene.clear_geometry()
    renderer.scene.add_geometry("mesh", mesh, mat)

    O3D_RENDERER_CACHE = renderer
    O3D_MESH_PATH_CACHE = mesh_path
    O3D_INTR_KEY_CACHE = key
    return renderer


def _depth_m_to_uint16_cm(depth_m: np.ndarray, scale: float = 1.0):
    """
    depth_m: float32 meters, invalid already set to 0
    返回：uint16 cm（并按 scale 做保存前 resize）
    """
    depth_cm = np.clip(depth_m * 100.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    if scale != 1.0:
        H, W = depth_cm.shape[:2]
        new_w = max(1, int(round(W * scale)))
        new_h = max(1, int(round(H * scale)))
        depth_cm = cv2.resize(depth_cm, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return depth_cm


def _save_depth_uint16_cm(depth_m: np.ndarray, out_path: str, scale: float = 1.0, save_vis: bool = False):
    """
    depth_m: float32 meters, invalid already set to 0
    保存：
      - out_path: uint16 png in centimeters
      - optional: <stem>_vis.png pseudo-color (8-bit)
    返回：保存尺寸对应的 uint16 depth_cm
    """
    depth_cm = _depth_m_to_uint16_cm(depth_m, scale=scale)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(str(out_path), depth_cm)

    if save_vis:
        valid = depth_cm > 0
        vis = np.zeros_like(depth_cm, dtype=np.uint8)
        if np.any(valid):
            d = depth_cm.astype(np.float32)
            dv = d[valid]
            d_min = float(dv.min())
            d_max = float(dv.max())
            denom = (d_max - d_min) + 1e-6
            vis_f = (d - d_min) / denom * 255.0
            vis = vis_f.astype(np.uint8)
            vis[~valid] = 0
        vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        stem, _ = os.path.splitext(out_path)
        out_vis = stem + "_vis.png"
        cv2.imwrite(str(out_vis), vis_color)

    return depth_cm


def render_mesh_depth_single_view(
    mesh_path: str,
    intrinsics: dict,
    intr_scale: float,
    C_local: np.ndarray,
    R_w2c: np.ndarray,
    out_depth_path: str,
    znear: float = 0.1,
    zfar: float = 1000.0,
    scale: float = 1.0,
    save_vis: bool = False,
    rank: int = 0,
    return_depth_u16_cm: bool = False,
):
    """
    Render depth from mesh using Open3D OffscreenRenderer.
    Depth is saved in centimeters (uint16 PNG). Invalid pixels -> 0.
    """
    s = _safe_scale(intr_scale, default=1.0)
    renderer = _get_o3d_renderer(mesh_path, intrinsics, s, rank=rank)

    W0 = int(intrinsics["width"])
    H0 = int(intrinsics["height"])
    W, H = _scaled_wh(W0, H0, s)

    FX = float(intrinsics["fx"]) * s
    FY = float(intrinsics["fy"]) * s
    CX = float(intrinsics["cx"]) * s
    CY = float(intrinsics["cy"]) * s

    intrinsic = np.array(
        [
            [FX, 0.0, CX],
            [0.0, FY, CY],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    renderer.scene.camera.set_projection(intrinsic, float(znear), float(zfar), W, H)

    cam_pos = C_local.astype(np.float64)
    forward = (R_w2c.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64))
    up = (R_w2c.T @ np.array([0.0, -1.0, 0.0], dtype=np.float64))
    renderer.scene.camera.look_at(cam_pos + forward, cam_pos, up)

    depth_o3d = renderer.render_to_depth_image(z_in_view_space=True)
    depth = np.asarray(depth_o3d, dtype=np.float32)  # (H,W)

    invalid = ~np.isfinite(depth) | (depth <= 0)
    depth_clean = depth.copy()
    depth_clean[invalid] = 0.0

    depth_u16_cm = _save_depth_uint16_cm(depth_clean, out_depth_path, scale=scale, save_vis=save_vis)
    if return_depth_u16_cm:
        return depth_u16_cm
    return None


# =========================================================
# main
# =========================================================
if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = ArgumentParser(description="Render DJI 3DGS PLY views from traj_random.txt (auto path by env_id + task_id)")

    # ========= 你要求的核心参数（用户运行时只需传这两个） =========
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.environ.get("HUGE_DATA_ROOT", "."),
        help="Dataset root containing data_3d/ and data_traj/.",
    )
    parser.add_argument(
        "--env_id",
        type=str,
        required=True,
        choices=["no1_building", "no3_door", "overhead_bridge","2_city"],
        help="Environment ID.",
    )
    parser.add_argument(
        "--task_id",
        type=str,
        required=True,
        help="Task ID suffix, e.g. 0 / hl / building (will be normalized to task_<task_id>)",
    )
    parser.add_argument(
        "--poses_txt_name",
        type=str,
        default="traj_random.txt",
        help="Pose txt filename under the task/env directory. Use traj_random_global.txt for real-pose/global rendering.",
    )

    # ========= 以下路径参数保留为“可选覆盖”，默认自动推导 =========
    parser.add_argument(
        "--ply_path",
        type=str,
        default="",
        help="Optional override for 3DGS PLY path. If empty, auto uses: <data_root>/data_3d/<env_id>/3dgs_ply/point_cloud_utm50.ply",
    )
    parser.add_argument(
        "--mesh_path",
        type=str,
        default="",
        help="Optional override for mesh path. If empty, auto uses: <data_root>/data_3d/<env_id>/terra_ply/merged_mesh.obj",
    )

    # 输出目录：默认放到 traj_random.txt 同级目录下
    parser.add_argument(
        "--out_dir",
        type=str,
        default="render_img",
        help="RGB output directory name or path. If only a name, it will be placed beside traj_random.txt",
    )
    parser.add_argument(
        "--depth_dir",
        type=str,
        default="render_depth",
        help="Depth output directory name or path. If only a name, it will be placed beside traj_random.txt",
    )

    parser.add_argument("--white_bg", action="store_true")

    parser.add_argument(
        "--img_scale",
        type=float,
        default=1.0,
        help="Post-scale for SAVED 3DGS RGB (resize before save). 1.0 keeps render size. "
             "NOTE: render size is controlled by --intr_scale.",
    )

    parser.add_argument(
        "--intr_scale",
        type=float,
        default=0.065,
        help="Scale intrinsics AND render resolution for BOTH 3DGS and mesh depth. "
             "W,H and fx/fy/cx/cy are all multiplied by this. 1.0 keeps original.",
    )

    parser.add_argument(
        "--log_interval",
        type=float,
        default=10.0,
        help="Seconds between progress logs (rank0 only). Set <=0 to disable.",
    )

    # =========================
    # mesh depth 输出
    # =========================
    parser.add_argument(
        "--enable_mesh_depth",
        action="store_true",
        default=True,
        help="Enable rendering mesh depth with Open3D (saved as uint16 PNG in cm).",
    )
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=1.0,
        help="Post-scale for SAVED depth (resize before save). 1.0 keeps render size. "
             "NOTE: render size is controlled by --intr_scale.",
    )
    parser.add_argument(
        "--depth_save_vis",
        action="store_true",
        default=False,
        help="Also save pseudo-colored depth visualization PNG (*_vis.png).",
    )
    parser.add_argument("--mesh_znear", type=float, default=0.1)
    parser.add_argument("--mesh_zfar", type=float, default=1000.0)

    pipeline_params = PipelineParams(parser)
    args = parser.parse_args()
    pipeline = pipeline_params.extract(args)

    # 固定为“完整轨迹渲染逻辑”（等价原 train 模式）
    args.mode = "train"

    if args.enable_mesh_depth and not MESH_RENDER_AVAILABLE:
        rank_print(
            "[WARN] open3d/cv2 not available in this environment -> mesh depth rendering disabled.",
            rank=0,
            only_rank0=False,
        )
        args.enable_mesh_depth = False

    # -----------------------------
    # 按 env_id + task_id 自动推导路径
    # -----------------------------
    task_dir_name = normalize_task_id(args.task_id)

    base_dir = os.path.join(args.data_root, "data_traj", task_dir_name, args.env_id)
    os.makedirs(base_dir, exist_ok=True)

    poses_txt_path = os.path.join(base_dir, args.poses_txt_name)
    traj_meta_path = os.path.join(base_dir, "traj_meta.txt")  # 自动尝试读取（不存在则 fallback）

    if (args.ply_path or "").strip() == "":
        args.ply_path = os.path.join(args.data_root, "data_3d", args.env_id, "3dgs_ply", "point_cloud_utm50.ply")

    if (args.mesh_path or "").strip() == "":
        mesh_dir = os.path.join(args.data_root, "data_3d", args.env_id, "terra_ply")
        args.mesh_path = os.path.join(mesh_dir, "merged_mesh.obj")
        if not os.path.isfile(args.mesh_path):
            args.mesh_path = os.path.join(mesh_dir, "simplified_mesh.obj")

    out_dir = _maybe_prefix_base(args.out_dir, base_dir)
    os.makedirs(out_dir, exist_ok=True)

    depth_dir = _maybe_prefix_base(args.depth_dir, base_dir)
    if args.enable_mesh_depth:
        os.makedirs(depth_dir, exist_ok=True)

    wash_res_path = os.path.join(base_dir, "wash_res.txt")
    tmp_wash_dir = os.path.join(base_dir, "wash_tmp")

    rank_print(f"[INFO] data_root      = {args.data_root}", rank=0, only_rank0=False)
    rank_print(f"[INFO] env_id        = {args.env_id}", rank=0, only_rank0=False)
    rank_print(f"[INFO] task_id       = {args.task_id} -> {task_dir_name}", rank=0, only_rank0=False)
    rank_print(f"[INFO] render mode    = FULL trajectory (fixed, no train/test split)", rank=0, only_rank0=False)
    rank_print(f"[INFO] base_dir       = {base_dir}", rank=0, only_rank0=False)
    rank_print(f"[INFO] poses_txt_path = {poses_txt_path}", rank=0, only_rank0=False)
    rank_print(f"[INFO] traj_meta_path = {traj_meta_path}", rank=0, only_rank0=False)
    rank_print(f"[INFO] ply_path       = {args.ply_path}", rank=0, only_rank0=False)
    rank_print(f"[INFO] out_dir        = {out_dir}", rank=0, only_rank0=False)
    rank_print(f"[INFO] wash_res_path  = {wash_res_path}", rank=0, only_rank0=False)
    rank_print(f"[INFO] img_scale      = {args.img_scale}", rank=0, only_rank0=False)
    rank_print(f"[INFO] intr_scale     = {args.intr_scale}", rank=0, only_rank0=False)
    rank_print(f"[INFO] sky_path       = {SKY_IMAGE_PATH}", rank=0, only_rank0=False)

    if args.enable_mesh_depth:
        rank_print(f"[INFO] mesh_path      = {args.mesh_path}", rank=0, only_rank0=False)
        rank_print(f"[INFO] depth_dir      = {depth_dir}", rank=0, only_rank0=False)
        rank_print(f"[INFO] depth_scale    = {args.depth_scale}", rank=0, only_rank0=False)
        rank_print(f"[INFO] depth_save_vis = {args.depth_save_vis}", rank=0, only_rank0=False)

    if not os.path.isfile(poses_txt_path):
        raise FileNotFoundError(f"poses_txt_path not found: {poses_txt_path}")

    if not os.path.isfile(args.ply_path):
        raise FileNotFoundError(f"ply_path not found: {args.ply_path}")

    if args.enable_mesh_depth and (not os.path.isfile(args.mesh_path)):
        raise FileNotFoundError(f"mesh_path not found: {args.mesh_path}")

    # 1) read intrinsics
    intrinsics = parse_intrinsics_from_txt(poses_txt_path)

    # 2) read poses
    poses_all = load_random_poses(poses_txt_path)

    # 3) load traj meta + build trajectories
    traj_metas = load_traj_meta(traj_meta_path)
    if len(traj_metas) == 0:
        rank_print(
            "[WARN] traj_meta missing/empty -> fallback: treat each pose as a single-frame trajectory.",
            rank=0,
            only_rank0=False,
        )

    trajs = build_trajectories_from_meta(poses_all, traj_metas)
    rank_print(f"[INFO] trajectories built: {len(trajs)}", rank=0, only_rank0=False)

    # 4) progress unit definition: 固定按“pose”为单位（完整轨迹渲染）
    total_units = int(sum(len(t["poses"]) for t in trajs))

    def print_resolution_summary(intrinsics_, args_, rank=0):
        W0, H0 = int(intrinsics_["width"]), int(intrinsics_["height"])
        intr_s = _safe_scale(args_.intr_scale, 1.0)
        img_s = _safe_scale(args_.img_scale, 1.0)
        dep_s = _safe_scale(args_.depth_scale, 1.0)

        Wr, Hr = _scaled_wh(W0, H0, intr_s)
        W_rgb, H_rgb = _scaled_wh(Wr, Hr, img_s)
        W_dep, H_dep = _scaled_wh(Wr, Hr, dep_s)

        rank_print(
            f"[RES] base(W0,H0)=({W0},{H0}) | intr_scale={intr_s}\n"
            f"      3DGS render  = ({Wr},{Hr})  -> save(img_scale={img_s}) = ({W_rgb},{H_rgb})\n"
            f"      Mesh render  = ({Wr},{Hr})  -> save(depth_scale={dep_s}) = ({W_dep},{H_dep})",
            rank=rank,
            only_rank0=False,
        )

    print_resolution_summary(intrinsics, args, rank=0)
    rank_print(f"[INFO] total_units to process = {total_units} (full trajectories)", rank=0, only_rank0=False)

    # 5) multi-gpu / single
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    start_time_main = time.time()

    # 写回 args 供 worker 使用
    args.out_dir = out_dir
    args.depth_dir = depth_dir

    # 清理/创建 tmp wash dir
    try:
        os.makedirs(tmp_wash_dir, exist_ok=True)
        for fn in os.listdir(tmp_wash_dir):
            if fn.startswith("wash_rank") and fn.endswith(".txt"):
                try:
                    os.remove(os.path.join(tmp_wash_dir, fn))
                except Exception:
                    pass
    except Exception:
        pass

    if num_gpus > 1:
        rank_print(f"[INFO] detected {num_gpus} GPUs -> multi-GPU rendering", rank=0, only_rank0=False)
        world_size = num_gpus

        counter = mp.Value("i", 0)
        lock = mp.Lock()

        mp.spawn(
            render_worker,
            args=(world_size, args, intrinsics, trajs, pipeline, counter, lock, total_units, start_time_main, tmp_wash_dir),
            nprocs=world_size,
            join=True,
        )
        rank_print("[INFO] multi-GPU rendering finished.", rank=0, only_rank0=False)

        # merge wash files -> 默认包含全部 traj_id
        traj_ids = set()
        for r in range(world_size):
            p = os.path.join(tmp_wash_dir, f"wash_rank{r}.txt")
            if not os.path.isfile(p):
                continue
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        traj_ids.add(int(s))
                    except Exception:
                        pass

        traj_sorted = sorted(traj_ids)
        with open(wash_res_path, "w", encoding="utf-8") as f:
            for tid in traj_sorted:
                f.write(f"{tid}\n")

        rank_print(f"[INFO] wash_res written: {wash_res_path} | traj_count={len(traj_sorted)}", rank=0, only_rank0=False)

    else:
        rank_print("[INFO] single GPU / CPU rendering", rank=0, only_rank0=False)

        WORKER_RANK = 0
        ONLY_RANK0_LOG = False

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # reset caches
        GAUSSIANS_CACHE = None
        PLY_PATH_CACHE = None
        DEVICE_CACHE = None

        O3D_RENDERER_CACHE = None
        O3D_MESH_PATH_CACHE = None
        O3D_INTR_KEY_CACHE = None

        SKY_TENSOR_CACHE = {}

        done = 0
        last_log_t = time.time()

        # 默认：wash_res 记录全部轨迹
        all_traj_ids = []

        for traj in trajs:
            traj_id = int(traj["traj_id"])
            poses = traj["poses"]
            if not poses:
                continue

            all_traj_ids.append(traj_id)

            for idx, p in enumerate(poses):
                pid = int(p["id"])
                C_local = p["C_local"]
                omega = p["omega"]
                phi = p["phi"]
                kappa = p["kappa"]

                R_w2c = opk_to_R_world2cam(omega, phi, kappa)
                t_w2c = -R_w2c @ C_local

                cam_info = {
                    **intrinsics,
                    "R_w2c": R_w2c.astype(np.float32),
                    "T_w2c": t_w2c.astype(np.float32),
                    "img_scale": float(args.img_scale),
                    "intr_scale": float(args.intr_scale),
                }

                out_name = f"{pid:06d}.png"
                out_path = os.path.join(out_dir, out_name)

                try:
                    render_single_view_from_cam_info(
                        ply_path=args.ply_path,
                        cam_info=cam_info,
                        pipeline=pipeline,
                        out_path=out_path,
                        white_background=args.white_bg,
                        device=device,
                        sky_path=SKY_IMAGE_PATH,
                    )
                except Exception as e:
                    rank_print(f"[WARN] render failed traj={traj_id} pose_id={pid}: {e}", rank=0, only_rank0=False)

                if args.enable_mesh_depth and MESH_RENDER_AVAILABLE:
                    try:
                        depth_out_path = os.path.join(depth_dir, out_name)
                        render_mesh_depth_single_view(
                            mesh_path=args.mesh_path,
                            intrinsics=intrinsics,
                            intr_scale=float(args.intr_scale),
                            C_local=C_local.astype(np.float64),
                            R_w2c=R_w2c.astype(np.float64),
                            out_depth_path=depth_out_path,
                            znear=args.mesh_znear,
                            zfar=args.mesh_zfar,
                            scale=args.depth_scale,
                            save_vis=args.depth_save_vis,
                            rank=0,
                            return_depth_u16_cm=False,
                        )
                    except Exception as e:
                        rank_print(f"[WARN] mesh depth render failed traj={traj_id} pose_id={pid}: {e}", rank=0, only_rank0=False)

                done += 1

                # ETA log（节流）
                now = time.time()
                if args.log_interval > 0 and (now - last_log_t) >= args.log_interval:
                    elapsed = now - start_time_main
                    speed = done / max(elapsed, 1e-6)
                    remain = total_units - done
                    eta = remain / max(speed, 1e-6)
                    rank_print(
                        f"[progress] {done}/{total_units} | {speed:.3f} unit/s | ETA {fmt_seconds(eta)}",
                        rank=0,
                        only_rank0=False,
                    )
                    last_log_t = now

        elapsed = time.time() - start_time_main
        speed = done / max(elapsed, 1e-6)
        rank_print(
            f"[INFO] done {done}/{total_units} | avg {speed:.3f} unit/s | elapsed {fmt_seconds(elapsed)}",
            rank=0,
            only_rank0=False,
        )

        # 写 wash_res.txt（默认全部轨迹）
        traj_sorted = sorted(set(int(x) for x in all_traj_ids))
        with open(wash_res_path, "w", encoding="utf-8") as f:
            for tid in traj_sorted:
                f.write(f"{tid}\n")
        rank_print(f"[INFO] wash_res written: {wash_res_path} | traj_count={len(traj_sorted)}", rank=0, only_rank0=False)

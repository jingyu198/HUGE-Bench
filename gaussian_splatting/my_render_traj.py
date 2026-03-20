# -*- coding: utf-8 -*-
#
# 渲染 DJI 3DGS 的 PLY + 随机生成的 pose（poses_random.txt）
# 不再依赖 XML 读取内参，内参从 poses_random.txt 的头部注释读取
#
import torch.multiprocessing as mp
import os
import math
import numpy as np
import torch
from torch import nn
import xml.etree.ElementTree as ET  # 可以保留，不影响
from argparse import ArgumentParser
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

def render_worker(rank, world_size, args, intrinsics, poses, pipeline):
    """
    多卡渲染的子进程：每个 rank 对应一块 GPU，负责一部分 pose。
    """
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)   # 关键：设置当前进程默认 GPU
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    print(f"[RANK {rank}] 使用设备: {device}")

    # 每个进程自己的全局 cache（进程之间不会共享内存）
    global GAUSSIANS_CACHE, PLY_PATH_CACHE, DEVICE_CACHE
    GAUSSIANS_CACHE = None
    PLY_PATH_CACHE = None
    DEVICE_CACHE = None

    os.makedirs(args.out_dir, exist_ok=True)

    num_poses = len(poses)
    # 采用简单的 stride 拆分：rank 0 处理 0, world_size, 2*world_size, ...
    for idx in range(rank, num_poses, world_size):
        p = poses[idx]
        pid = p["id"]
        C_local = p["C_local"]
        omega = p["omega"]
        phi = p["phi"]
        kappa = p["kappa"]

        # 构造 R_w2c, t_w2c（局部坐标）
        R_w2c = opk_to_R_world2cam(omega, phi, kappa)
        t_w2c = -R_w2c @ C_local

        cam_info = {
            **intrinsics,
            "R_w2c": R_w2c.astype(np.float32),
            "T_w2c": t_w2c.astype(np.float32),
        }

        out_name = f"{pid:06d}.png"
        out_path = os.path.join(args.out_dir, out_name)

        print(f"[RANK {rank}] 渲染随机 pose id={pid} -> {out_name}")
        try:
            # 这里不用显式传 device，因为 render_single_view_from_cam_info 里
            # 用的是 "cuda"，而我们已经 set_device(rank) 了
            render_single_view_from_cam_info(
                ply_path=args.ply_path,
                cam_info=cam_info,
                pipeline=pipeline,
                out_path=out_path,
                white_background=args.white_bg,
            )
        except Exception as e:
            print(f"[RANK {rank}] [WARN] 渲染 pose id={pid} 失败: {e}")

# --------------------------
# 全局缓存：避免重复读取 PLY
# --------------------------
GAUSSIANS_CACHE = None
PLY_PATH_CACHE = None
DEVICE_CACHE = None


# -------------------------------------------------------
# 1. 把 DJI 3DGS PLY 填进 GaussianModel（sh_degree=0）
# -------------------------------------------------------
def load_dji_ply_into_gaussians(gaussians: GaussianModel, ply_path: str, device: torch.device):
    """
    读取 DJI 导出的 3DGS PLY：
        x, y, z, nx, ny, nz,
        f_dc_0, f_dc_1, f_dc_2,
        opacity,
        scale_0
    """
    print(f"[INFO] 读取 PLY: {ply_path}")
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data
    n_pts = v.shape[0]

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
    opacity = np.asarray(v["opacity"], dtype=np.float32)[..., None]  # [N,1]
    scale0 = np.asarray(v["scale_0"], dtype=np.float32)             # [N]

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
    gaussians._features_dc = nn.Parameter(
        features_dc.transpose(1, 2).contiguous().requires_grad_(False)
    )
    gaussians._features_rest = nn.Parameter(
        features_rest.transpose(1, 2).contiguous().requires_grad_(False)
    )
    gaussians._opacity = nn.Parameter(opacities_t.requires_grad_(False))
    gaussians._scaling = nn.Parameter(scales_t.requires_grad_(False))
    gaussians._rotation = nn.Parameter(rots.requires_grad_(False))

    gaussians.max_radii2D = torch.zeros((n_pts,), device=device)

    print(f"[INFO] 加载高斯数量: {n_pts}")


# -------------------------------------------------------
# 2. BlocksExchange: Omega/Phi/Kappa -> R_w2c
# -------------------------------------------------------
def opk_to_R_world2cam(omega_deg: float, phi_deg: float, kappa_deg: float) -> np.ndarray:
    """
    R_w2c = Rz(kappa) * Ry(phi) * Rx(omega)
    X_cam = R_w2c * (X_world - C)
    """
    om = math.radians(omega_deg)
    ph = math.radians(phi_deg)
    ka = math.radians(kappa_deg)

    cos_o, sin_o = math.cos(om), math.sin(om)
    cos_p, sin_p = math.cos(ph), math.sin(ph)
    cos_k, sin_k = math.cos(ka), math.sin(ka)

    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_o, sin_o],
            [0.0, -sin_o, cos_o],
        ],
        dtype=np.float32,
    )
    Ry = np.array(
        [
            [cos_p, 0.0, -sin_p],
            [0.0, 1.0, 0.0],
            [sin_p, 0.0, cos_p],
        ],
        dtype=np.float32,
    )
    Rz = np.array(
        [
            [cos_k, sin_k, 0.0],
            [-sin_k, cos_k, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    R = Rz @ Ry @ Rx
    return R


# === NEW ===
# 2.1 从 poses_random.txt 读取相机内参（分辨率 + fx, fy, cx, cy）
def parse_intrinsics_from_txt(txt_path: str):
    """
    解析 poses_random.txt 头部注释中的内参信息：
    期望格式类似：
        # intrinsics: width height fx fy cx cy
        # 5280 3956 3480.185791 3600.000000 2690.748469 1960.206342
    """
    width = height = fx = fy = cx = cy = None

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if line.startswith("#"):
            content = line.lstrip("#").strip()
            # 找到 intrinsics 这一行
            if content.lower().startswith("intrinsics"):
                # 下一行应该是具体数值，可能也带 '#'
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

    print(f"[INFO] 从 txt 读取内参: width={width}, height={height}, fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")
    return {
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "FoVx": FoVx,
        "FoVy": FoVy,
    }


# === 仍然保留 ===
# 2.2 从 poses_random.txt 读取随机相机 pose
def load_random_poses(txt_path: str):
    """
    txt 格式:
    # id  x_rel  y_rel  z_rel  omega_deg  phi_deg  kappa_deg
    0 -106.812500 39.250000 -5.210571 6.525151 -1.264588 0.144632
    ...
    x_rel,y_rel,z_rel 已经是 C_world - OFFSET（也就是点云局部坐标系）
    """
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
            poses.append(
                {
                    "id": pid,
                    "C_local": C_local,   # 直接就是局部坐标
                    "omega": omega,
                    "phi": phi,
                    "kappa": kappa,
                }
            )

    print(f"[INFO] 从 {txt_path} 读取到 {len(poses)} 个随机相机 pose")
    return poses


def make_3dgs_camera_from_info(cam_info: dict, device: torch.device) -> Camera:
    """
    使用:
      - R_w2c, T_w2c
      - FoVx, FoVy
      - width, height, fx, fy, cx, cy
    构造 3DGS Camera
    """
    W, H = cam_info["width"], cam_info["height"]
    FoVx, FoVy = cam_info["FoVx"], cam_info["FoVy"]
    R_w2c = cam_info["R_w2c"]
    T_w2c = cam_info["T_w2c"]

    # 3DGS 期望的是 camera-to-world 旋转矩阵
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

    cam.projection_matrix = getProjectionMatrix_with_principal(
        znear=cam.znear,
        zfar=cam.zfar,
        fx=cam_info["fx"],
        fy=cam_info["fy"],
        cx=cam_info["cx"],
        cy=cam_info["cy"],
        width=W,
        height=H,
    ).transpose(0, 1).cuda()

    cam.full_proj_transform = (
        cam.world_view_transform.unsqueeze(0)
        .bmm(cam.projection_matrix.unsqueeze(0))
        .squeeze(0)
    )

    return cam


# -------------------------------------------------------
# 3. 渲染单个视角（使用随机 pose）
# -------------------------------------------------------
def render_single_view_from_cam_info(
    ply_path: str,
    cam_info: dict,
    pipeline,
    out_path: str,
    white_background: bool = False,
):
    global GAUSSIANS_CACHE, PLY_PATH_CACHE, DEVICE_CACHE

    # 1) 构建 / 复用 GaussianModel 并加载 DJI PLY
    if GAUSSIANS_CACHE is None or PLY_PATH_CACHE != ply_path:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] 使用设备: {device}")
        gaussians = GaussianModel(sh_degree=0, optimizer_type="default")
        load_dji_ply_into_gaussians(gaussians, ply_path, device)
        GAUSSIANS_CACHE = gaussians
        PLY_PATH_CACHE = ply_path
        DEVICE_CACHE = device
    else:
        gaussians = GAUSSIANS_CACHE
        device = DEVICE_CACHE
        print(f"[INFO] 复用已加载的 3DGS PLY: {ply_path}")

    cam = make_3dgs_camera_from_info(cam_info, device)

    bg_color = [1.0, 1.0, 1.0] if white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with torch.no_grad():
        out = render(
            cam,
            gaussians,
            pipeline,
            background,
            use_trained_exp=False,
            separate_sh=SPARSE_ADAM_AVAILABLE,
        )
        rendering = out["render"]  # [3, H, W]

        scale = 1.0  #choose the resolution scale of output
        H, W = rendering.shape[1], rendering.shape[2]
        new_H = max(1, int(H * scale))
        new_W = max(1, int(W * scale))

        rendering_small = F.interpolate(
            rendering.unsqueeze(0),
            size=(new_H, new_W),
            mode="bilinear",
            align_corners=False,
        )[0]

        torchvision.utils.save_image(rendering_small.cpu(), out_path)
        #print(f"[INFO] 渲染完成（0.2x 分辨率），保存到: {out_path}")


# -------------------------------------------------------
# 4. 命令行入口：遍历渲染 poses_random.txt 中的所有视角
# -------------------------------------------------------
if __name__ == "__main__":
    # 对于多进程 + CUDA，推荐使用 spawn
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # 在某些环境中可能已经设置过 start_method，这里直接忽略
        pass

    parser = ArgumentParser(description="Render DJI 3DGS PLY views from poses_random.txt")

    parser.add_argument("--ply_path", type=str,
                        default='/mnt/jingyu/DJI_3dgs/3dgs_ply/point_cloud_utm50.ply')
    parser.add_argument("--poses_txt_path", type=str,
                        default='/mnt/jingyu/DJI_3dgs/pi_data/traj_random.txt')
    parser.add_argument("--out_dir", type=str,
                        default='/mnt/jingyu/DJI_3dgs/pi_data/render_img')
    parser.add_argument("--white_bg", action="store_true")

    pipeline_params = PipelineParams(parser)
    args = parser.parse_args()
    pipeline = pipeline_params.extract(args)

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) 从 txt 读一次内参（而不是 XML）
    intrinsics = parse_intrinsics_from_txt(args.poses_txt_path)

    # 2) 从 txt 读随机 pose
    poses = load_random_poses(args.poses_txt_path)

    print(f"[INFO] 总共有 {len(poses)} 个随机 pose。")

    # 3) 判断 GPU 数量：>1 就多卡并行，否则沿用原来的单卡循环
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if num_gpus > 1:
        print(f"[INFO] 检测到 {num_gpus} 张 GPU，启用多卡并行渲染。")
        world_size = num_gpus

        # mp.spawn 会自动传入 rank [0, world_size-1]
        mp.spawn(
            render_worker,
            args=(world_size, args, intrinsics, poses, pipeline),
            nprocs=world_size,
            join=True,
        )
        print("[INFO] 多卡渲染完成。")

    else:
        print("[INFO] 只检测到单卡或无 GPU，使用原来的单线程渲染逻辑。")

        for p in poses:
            pid = p["id"]
            C_local = p["C_local"]
            omega = p["omega"]
            phi = p["phi"]
            kappa = p["kappa"]

            # 构造 R_w2c, t_w2c（局部坐标）
            R_w2c = opk_to_R_world2cam(omega, phi, kappa)
            t_w2c = -R_w2c @ C_local

            cam_info = {
                **intrinsics,
                "R_w2c": R_w2c.astype(np.float32),
                "T_w2c": t_w2c.astype(np.float32),
            }

            out_name = f"{pid:06d}.png"
            out_path = os.path.join(args.out_dir, out_name)

            print(f"\n[INFO] 渲染随机 pose id={pid}")
            try:
                render_single_view_from_cam_info(
                    ply_path=args.ply_path,
                    cam_info=cam_info,
                    pipeline=pipeline,
                    out_path=out_path,
                    white_background=args.white_bg,
                )
            except Exception as e:
                print(f"[WARN] 渲染 pose id={pid} 失败: {e}")

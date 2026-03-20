# -*- coding: utf-8 -*-
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from plyfile import PlyData
from torch import nn

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.graphics_utils import getProjectionMatrix_with_principal

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


GAUSSIANS_CACHE = None
PLY_PATH_CACHE = None
DEVICE_CACHE = None


def load_dji_ply_into_gaussians(gaussians: GaussianModel, ply_path: str, device: torch.device):
    """Load a DJI-exported 3DGS PLY into GaussianModel."""
    print(f"[INFO] reading PLY: {ply_path}")
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

    print(f"[INFO] loaded {n_pts} gaussians")


def opk_to_R_world2cam(omega_deg: float, phi_deg: float, kappa_deg: float) -> np.ndarray:
    """Convert omega/phi/kappa angles to a world-to-camera rotation matrix."""
    om = math.radians(omega_deg)
    ph = math.radians(phi_deg)
    ka = math.radians(kappa_deg)

    cos_o, sin_o = math.cos(om), math.sin(om)
    cos_p, sin_p = math.cos(ph), math.sin(ph)
    cos_k, sin_k = math.cos(ka), math.sin(ka)

    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_o, sin_o],
            [0.0, -sin_o, cos_o],
        ],
        dtype=np.float32,
    )
    ry = np.array(
        [
            [cos_p, 0.0, -sin_p],
            [0.0, 1.0, 0.0],
            [sin_p, 0.0, cos_p],
        ],
        dtype=np.float32,
    )
    rz = np.array(
        [
            [cos_k, sin_k, 0.0],
            [-sin_k, cos_k, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    return rz @ ry @ rx


def make_3dgs_camera_from_info(cam_info: dict, device: torch.device) -> Camera:
    """Build a Gaussian Splatting camera from intrinsics and world-to-camera pose."""
    width = cam_info["width"]
    height = cam_info["height"]
    fovx = cam_info["FoVx"]
    fovy = cam_info["FoVy"]
    r_w2c = cam_info["R_w2c"]
    t_w2c = cam_info["T_w2c"]

    cam = Camera(
        resolution=[width, height],
        colmap_id=0,
        R=r_w2c.T,
        T=t_w2c,
        FoVx=fovx,
        FoVy=fovy,
        depth_params=None,
        image=Image.new("RGB", (width, height), (0, 0, 0)),
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
        width=width,
        height=height,
    ).transpose(0, 1).to(device)

    cam.full_proj_transform = (
        cam.world_view_transform.unsqueeze(0)
        .bmm(cam.projection_matrix.unsqueeze(0))
        .squeeze(0)
    )
    return cam


def render_single_view_from_cam_info(
    ply_path: str,
    cam_info: dict,
    pipeline,
    out_path: str,
    white_background: bool = False,
):
    global GAUSSIANS_CACHE, PLY_PATH_CACHE, DEVICE_CACHE

    if GAUSSIANS_CACHE is None or PLY_PATH_CACHE != ply_path:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] using device: {device}")
        gaussians = GaussianModel(sh_degree=0, optimizer_type="default")
        load_dji_ply_into_gaussians(gaussians, ply_path, device)
        GAUSSIANS_CACHE = gaussians
        PLY_PATH_CACHE = ply_path
        DEVICE_CACHE = device
    else:
        gaussians = GAUSSIANS_CACHE
        device = DEVICE_CACHE
        print(f"[INFO] reusing loaded 3DGS PLY: {ply_path}")

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
        rendering = out["render"]
        height, width = rendering.shape[1], rendering.shape[2]
        rendering_resized = F.interpolate(
            rendering.unsqueeze(0),
            size=(max(1, int(height)), max(1, int(width))),
            mode="bilinear",
            align_corners=False,
        )[0]
        torchvision.utils.save_image(rendering_resized.cpu(), out_path)


__all__ = [
    "opk_to_R_world2cam",
    "render_single_view_from_cam_info",
]

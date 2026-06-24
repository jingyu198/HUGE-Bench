# -*- coding: utf-8 -*-
"""
用 3DGS Camera 的 full_proj_transform 投影建筑轮廓点到渲染图像上，叠加红点并输出 *_masked.png

要求：在你的 3DGS 工程环境里运行（能 import scene.cameras / utils.graphics_utils）
依赖：pip install pillow numpy torch

输入：
  1) traj_random.txt  (poses + intrinsics)
  2) traj_meta.txt    (traj_id loc_name pose_id_start pose_id_end)
  3) building_coords/*.txt
  4) render_img/{pose_id:06d}.png  (渲染输出)

输出：
  masked_dir/traj_{traj_id:04d}_{pose_id:06d}_masked.png
"""

import os
import math
import numpy as np
from argparse import ArgumentParser
from PIL import Image, ImageDraw

import torch

from scene.cameras import Camera
from utils.graphics_utils import getProjectionMatrix_with_principal


# -------------------------
# 1) intrinsics from txt
# -------------------------
def parse_intrinsics_from_txt(txt_path: str):
    width = height = fx = fy = cx = cy = None
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        s = line.strip()
        if not s.startswith("#"):
            continue
        content = s.lstrip("#").strip()
        if content.lower().startswith("intrinsics"):
            # next non-empty line
            for j in range(i + 1, len(lines)):
                nl = lines[j].strip()
                if not nl:
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
            break

    if width is None:
        raise RuntimeError(f"在 {txt_path} 中未找到 intrinsics。")

    FoVx = 2.0 * math.atan(0.5 * width / fx)
    FoVy = 2.0 * math.atan(0.5 * height / fy)

    return {"width": width, "height": height, "fx": fx, "fy": fy, "cx": cx, "cy": cy, "FoVx": FoVx, "FoVy": FoVy}


# -------------------------
# 2) poses dict
# -------------------------
def load_poses_dict(txt_path: str):
    poses = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 7:
                continue
            pid = int(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            omega, phi, kappa = float(parts[4]), float(parts[5]), float(parts[6])
            poses[pid] = {
                "id": pid,
                "C_local": np.array([x, y, z], dtype=np.float32),
                "omega": omega,
                "phi": phi,
                "kappa": kappa,
            }
    if not poses:
        raise RuntimeError(f"{txt_path} 未读取到任何 pose。")
    return poses


# -------------------------
# 3) traj_meta
# -------------------------
def load_traj_meta(meta_path: str):
    metas = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 4:
                continue
            metas.append(
                {
                    "traj_id": int(parts[0]),
                    "loc_name": parts[1],
                    "pose_id_start": int(parts[2]),
                    "pose_id_end": int(parts[3]),
                }
            )
    if not metas:
        raise RuntimeError(f"{meta_path} 未读取到任何轨迹 meta。")
    return metas


# -------------------------
# 4) buildings
# -------------------------
def parse_building_contours_from_txt_folder(contour_dir):
    if not os.path.isdir(contour_dir):
        raise FileNotFoundError(f"contour_dir 不存在: {contour_dir}")
    txts = sorted([fn for fn in os.listdir(contour_dir) if fn.lower().endswith(".txt")])
    if not txts:
        raise FileNotFoundError(f"在目录中未找到 txt: {contour_dir}")

    buildings = {}
    for fn in txts:
        path = os.path.join(contour_dir, fn)
        pts = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.split()
                if len(parts) < 2:
                    continue
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2]) if len(parts) >= 3 else 0.0
                except Exception:
                    continue
                pts.append([x, y, z])
        if len(pts) < 3:
            continue
        name = os.path.splitext(fn)[0]
        buildings[name] = np.asarray(pts, dtype=np.float32)

    if not buildings:
        raise RuntimeError(f"未成功读取任何 building 轮廓: {contour_dir}")
    return buildings


# -------------------------
# 5) EXACT same as renderer: OPK -> R_w2c
# -------------------------
def opk_to_R_world2cam(omega_deg: float, phi_deg: float, kappa_deg: float) -> np.ndarray:
    om = math.radians(float(omega_deg))
    ph = math.radians(float(phi_deg))
    ka = math.radians(float(kappa_deg))

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
    return (Rz @ Ry @ Rx).astype(np.float32)


# -------------------------
# 6) densify
# -------------------------
def densify_polyline_xyz(xyz, step_m=0.5, close=True):
    xyz = np.asarray(xyz, dtype=np.float32)
    if close and np.linalg.norm(xyz[0, :2] - xyz[-1, :2]) > 1e-6:
        xyz = np.vstack([xyz, xyz[0]])

    out = []
    for i in range(len(xyz) - 1):
        p0 = xyz[i]
        p1 = xyz[i + 1]
        vec = p1 - p0
        L = float(np.linalg.norm(vec))
        if L < 1e-6:
            continue
        n = max(1, int(math.floor(L / step_m)))
        for k in range(n):
            t = k / float(n)
            out.append(p0 * (1 - t) + p1 * t)
    return np.asarray(out, dtype=np.float32)


# -------------------------
# 7) build Camera exactly like renderer (no gaussian needed)
# -------------------------
def make_3dgs_camera_from_pose(intr, pose, device="cpu"):
    W, H = int(intr["width"]), int(intr["height"])

    C_local = pose["C_local"]
    omega, phi, kappa = pose["omega"], pose["phi"], pose["kappa"]

    R_w2c = opk_to_R_world2cam(omega, phi, kappa)
    t_w2c = -R_w2c @ C_local

    # renderer uses: R = R_w2c.T (camera-to-world), T = T_w2c
    R = R_w2c.T
    T = t_w2c

    dummy_img = Image.new("RGB", (W, H), (0, 0, 0))
    cam = Camera(
        resolution=[W, H],
        colmap_id=0,
        R=R,
        T=T,
        FoVx=float(intr["FoVx"]),
        FoVy=float(intr["FoVy"]),
        depth_params=None,
        image=dummy_img,
        invdepthmap=None,
        image_name="overlay",
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
            fx=float(intr["fx"]),
            fy=float(intr["fy"]),
            cx=float(intr["cx"]),
            cy=float(intr["cy"]),
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


# -------------------------
# 8) project points using full_proj_transform (same pipeline as render)
# -------------------------
def project_points_fullproj(points_xyz, cam: Camera, W0, H0):
    """
    points_xyz: [N,3] in the SAME coordinate system as PLY/pose (local)
    returns uv (float) in original resolution (W0,H0)
    """
    device = cam.full_proj_transform.device
    pts = torch.from_numpy(points_xyz).to(device=device, dtype=torch.float32)
    ones = torch.ones((pts.shape[0], 1), device=device, dtype=torch.float32)
    pts_h = torch.cat([pts, ones], dim=1)  # [N,4]

    clip = pts_h @ cam.full_proj_transform  # [N,4]  (row-vector convention in 3DGS code)
    w = clip[:, 3:4]
    valid = (w.abs() > 1e-8).squeeze(1)

    ndc = clip[:, 0:3] / w  # [N,3]
    x = ndc[:, 0]
    y = ndc[:, 1]

    # NDC -> pixel (OpenGL: y=+1 top). Convert to image coords (v down).
    u = (x + 1.0) * 0.5 * float(W0)
    v = (1.0 - (y + 1.0) * 0.5) * float(H0)  # = 0.5*(1-y)*H0

    return u.detach().cpu().numpy(), v.detach().cpu().numpy(), valid.detach().cpu().numpy()


def overlay_red_points_on_render(
    base_img_path,
    out_img_path,
    contour_xyz,
    cam: Camera,
    intr,
    densify_step_m=0.5,
    point_r=3,
):
    if not os.path.isfile(base_img_path):
        raise FileNotFoundError(base_img_path)

    img = Image.open(base_img_path).convert("RGBA")
    draw = ImageDraw.Draw(img)

    W0, H0 = int(intr["width"]), int(intr["height"])
    W1, H1 = img.size

    # 用真实尺寸算 scale，避免 0.25 四舍五入误差
    sx = W1 / float(W0)
    sy = H1 / float(H0)

    pts = densify_polyline_xyz(contour_xyz, step_m=densify_step_m, close=True)
    u0, v0, valid = project_points_fullproj(pts, cam, W0, H0)

    red = (255, 0, 0, 255)
    cnt = 0
    for uu, vv, ok in zip(u0, v0, valid):
        if not ok:
            continue
        u = uu * sx
        v = vv * sy
        if 0 <= u < W1 and 0 <= v < H1:
            draw.ellipse((u - point_r, v - point_r, u + point_r, v + point_r), fill=red)
            cnt += 1

    img.save(out_img_path)
    print(f"[OK] {os.path.basename(base_img_path)} -> {os.path.basename(out_img_path)} | drawn_points={cnt}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--poses_txt", type=str, required=True)
    parser.add_argument("--traj_meta", type=str, required=True)
    parser.add_argument("--contour_dir", type=str, required=True)
    parser.add_argument("--render_dir", type=str, required=True)
    parser.add_argument("--masked_dir", type=str, default="masked_img")
    parser.add_argument("--densify_step", type=float, default=0.5)
    parser.add_argument("--point_r", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.masked_dir, exist_ok=True)

    intr = parse_intrinsics_from_txt(args.poses_txt)
    poses = load_poses_dict(args.poses_txt)
    metas = load_traj_meta(args.traj_meta)
    buildings = parse_building_contours_from_txt_folder(args.contour_dir)

    device = torch.device("cpu")  # 叠加不需要 GPU

    miss_building = 0
    miss_img = 0

    for info in metas:
        traj_id = info["traj_id"]
        bname = info["loc_name"]
        pid0 = info["pose_id_start"]

        if bname not in buildings:
            print(f"[WARN] traj {traj_id}: building 不存在: {bname}")
            miss_building += 1
            continue
        if pid0 not in poses:
            print(f"[WARN] traj {traj_id}: pose_id_start 不在 poses_txt 里: {pid0}")
            continue

        img_path = os.path.join(args.render_dir, f"{pid0:06d}.png")
        if not os.path.isfile(img_path):
            print(f"[WARN] traj {traj_id}: 第一帧渲染图不存在: {img_path}")
            miss_img += 1
            continue

        cam = make_3dgs_camera_from_pose(intr, poses[pid0], device=device)

        out_path = os.path.join(args.masked_dir, f"traj_{traj_id:04d}_{pid0:06d}_masked.png")
        overlay_red_points_on_render(
            base_img_path=img_path,
            out_img_path=out_path,
            contour_xyz=buildings[bname],
            cam=cam,
            intr=intr,
            densify_step_m=args.densify_step,
            point_r=args.point_r,
        )

    print(f"[DONE] masked 输出到: {args.masked_dir}")
    print(f"[STAT] miss_building={miss_building}, miss_img={miss_img}")


if __name__ == "__main__":
    main()

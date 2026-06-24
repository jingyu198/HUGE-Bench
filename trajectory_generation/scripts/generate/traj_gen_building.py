# -*- coding: utf-8 -*-
"""
Merged script:
1) Generate DJI 3DGS trajectories around building contours (txt)

IMPORTANT:
    VLN 轨迹不做 global -> local 转换！
    只输出相对坐标系下的轨迹（C_rel = C_world - DJI_OFFSET），不生成 traj_random_local.txt。

新增：
- 生成 subtask.txt：
  # traj_id  subtask_id  pose_id_start  pose_id_end  subtask
  对每条 traj 划分 6 个子任务：
    0) 移动到******label边缘
    1) 下降到巡航高度（按 building 类型配置）
    2) 调整转向
    3) 开始环绕
    4) 上升到原始高度
    5) 移动到起始点

参数改动：
- 新增 DATA_ROOT / OUT_ROOT 常量
- 新增 --env_id（如 1_office / 2_city / 3_road / 4_lake）
- 输入路径默认根据 DATA_ROOT/env_id 自动拼接
- 输出统一写到 OUT_ROOT/env_id/ 下
- 删除 train/test mode 逻辑（不再区分 mode）
- start_h_min / start_h_max / end_height 支持按 env_id 自动使用 ENV_CONFIGS 默认值（也可手动覆盖）
- 新增 --seed，方便复现

本版补充（关键修复）：
- 对 2_city 等场景，按 building 标题/名称（当前默认是 txt 文件名）匹配关键词（如“高楼”“别墅”）
  使用不同的高度配置生成轨迹，而不是全场景统一高度。
"""

import os
import math
import random
import numpy as np
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from pathlib import Path

# 下面这些保留（尽量不动你的其他逻辑）
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ==== offset 不再手动写死，而是从 metadata.xml 里读取 ====
DJI_OFFSET = None

# =========================================================
# 全局路径与场景配置（新增）
# =========================================================
DATA_ROOT = Path(os.environ.get("HUGE_DATA_3D_ROOT", "./data_3d")).expanduser()
OUT_ROOT = Path(os.environ.get("HUGE_DATA_TRAJ_ROOT", "./data_traj")).expanduser() / "task_building"

ENV_CONFIGS = {
    "1_office": {
        "start_h_min": 20.0,
        "start_h_max": 40.0,
        "end_height": -20.0,
    },
    "2_city": {
        # 默认（可理解为“高楼”配置）
        "start_h_min": 0.0,
        "start_h_max": 20.0,
        "end_height": -40.0,

        # 按标题/名称关键词选择高度配置
        "xlsx_title_overrides": {
            "highrise": {
                "start_h_min": 0.0,
                "start_h_max": 20.0,
                "end_height": -40.0,
            },
            "hotel": {
                "start_h_min": 0.0,
                "start_h_max": 20.0,
                "end_height": -40.0,
            },
            "villa": {
                "start_h_min": -50.0,
                "start_h_max": -40.0,
                "end_height": -60.0,
            },
            "高楼": {
                "start_h_min": 0.0,
                "start_h_max": 20.0,
                "end_height": -40.0,
            },
            "别墅": {
                "start_h_min": -50.0,
                "start_h_max": -40.0,
                "end_height": -60.0,
            },
        },
    },
    "3_road": {
        "start_h_min": -10.0,
        "start_h_max": 10.0,
        "end_height": -50.0,
    },
    "4_lake": {
        "start_h_min": 30.0,
        "start_h_max": 50.0,
        "end_height": -10.0,
    },
}


# =========================================================
# 0.0 按 building 标题/名称选择高度配置（新增，真正生效）
# =========================================================
def resolve_height_config_for_building(
    env_id: str,
    building: dict,
    default_start_h_min: float,
    default_start_h_max: float,
    default_end_height: float,
    enable_building_override: bool = True,
):
    """
    根据 building 的 title/name（当前通常为 txt 文件名）匹配 ENV_CONFIGS[env_id]["xlsx_title_overrides"]，
    返回该 building 应使用的高度配置。

    若未匹配或未启用 override，则返回默认值。
    """
    cfg = {
        "start_h_min": float(default_start_h_min),
        "start_h_max": float(default_start_h_max),
        "end_height": float(default_end_height),
        "matched_rule": None,
    }

    if not enable_building_override:
        return cfg

    env_cfg = ENV_CONFIGS.get(env_id, {})
    overrides = env_cfg.get("xlsx_title_overrides", None)
    if not overrides:
        return cfg

    # 优先 title（若未来接入 xlsx），否则退化到 name（txt 文件名）
    title_text = str(building.get("title") or building.get("name") or "")
    title_text_lower = title_text.lower()

    for keyword, hcfg in overrides.items():
        keyword_text = str(keyword)
        keyword_lower = keyword_text.lower()
        if keyword_text and (keyword_text in title_text or keyword_lower in title_text_lower):
            if "start_h_min" in hcfg:
                cfg["start_h_min"] = float(hcfg["start_h_min"])
            if "start_h_max" in hcfg:
                cfg["start_h_max"] = float(hcfg["start_h_max"])
            if "end_height" in hcfg:
                cfg["end_height"] = float(hcfg["end_height"])
            cfg["matched_rule"] = keyword
            return cfg

    return cfg


# =========================================================
# 0.1 从 metadata.xml 解析 DJI_OFFSET（SRSOrigin）
# =========================================================
def parse_dji_offset_from_metadata(metadata_path):
    """
    从 DJI Terra 导出的 metadata.xml 中读取 SRSOrigin 作为 offset：
      <SRSOrigin>x,y,z</SRSOrigin>
    """
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"metadata.xml 未找到: {metadata_path}")

    tree = ET.parse(metadata_path)
    root = tree.getroot()

    srs_origin_node = root.find("SRSOrigin")
    if srs_origin_node is None or not srs_origin_node.text:
        raise RuntimeError(f"metadata.xml 中未找到有效的 <SRSOrigin> 节点: {metadata_path}")

    text = srs_origin_node.text.strip()
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise RuntimeError(f"SRSOrigin 格式错误，应为 'x,y,z'，实际为: {text}")

    offset = np.array([float(p) for p in parts], dtype=np.float64)
    print(f"[INFO] 从 metadata 读取 DJI_OFFSET = {offset.tolist()}")
    return offset


# =========================================================
# 1. 从 XML 解析相机内参（width, height, fx, fy, cx, cy）
# =========================================================
def parse_intrinsics_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    block = root.find("Block")
    if block is None:
        raise RuntimeError("XML 中未找到 <Block> 节点")

    photogroups = block.find("Photogroups")
    if photogroups is None:
        raise RuntimeError("XML 中未找到 <Photogroups> 节点")

    pg = photogroups.find("Photogroup")
    if pg is None:
        raise RuntimeError("XML 中未找到 <Photogroup> 节点")

    dim = pg.find("ImageDimensions")
    width = int(dim.find("Width").text)
    height = int(dim.find("Height").text)

    focal_px = float(pg.find("FocalLengthPixels").text)

    pp = pg.find("PrincipalPoint")
    cx = float(pp.find("x").text)
    cy = float(pp.find("y").text)

    aspect_node = pg.find("AspectRatio")
    aspect = float(aspect_node.text) if aspect_node is not None else 1.0

    fx = focal_px
    fy = focal_px * aspect

    print(f"[INFO] 内参: width={width}, height={height}, fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")
    return {"width": width, "height": height, "fx": fx, "fy": fy, "cx": cx, "cy": cy}


# =========================================================
# 2. 从 XML 解析所有相机中心 (world 坐标)
# =========================================================
def parse_all_cameras_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    block = root.find("Block")
    if block is None:
        raise RuntimeError("XML 中未找到 <Block> 节点")

    photogroups = block.find("Photogroups")
    if photogroups is None:
        raise RuntimeError("XML 中未找到 <Photogroups> 节点")

    cameras = []
    for pg in photogroups.findall("Photogroup"):
        for photo in pg.findall("Photo"):
            pid = int(photo.find("Id").text)
            img_path = photo.find("ImagePath").text

            center_node = photo.find("Pose/Center")
            C_world = np.array(
                [
                    float(center_node.find("x").text),
                    float(center_node.find("y").text),
                    float(center_node.find("z").text),
                ],
                dtype=np.float32,
            )
            cameras.append({"id": pid, "img_path": img_path, "C_world": C_world})

    if not cameras:
        raise RuntimeError("XML 中未找到任何 <Photo> 相机")
    return cameras


# =========================================================
# 3. OPK <-> R
# =========================================================
def R_to_opk(R):
    """
    输入: R_w2c (3x3)
    输出: (omega, phi, kappa) radians
    约定: R = Rz(kappa) @ Ry(phi) @ Rx(omega)
    """
    r20 = np.clip(R[2, 0], -1.0, 1.0)
    phi = math.asin(r20)

    cos_phi = math.cos(phi)
    if abs(cos_phi) < 1e-8:
        omega = 0.0
        kappa = math.atan2(-R[1, 2], R[1, 1])
    else:
        omega = math.atan2(-R[2, 1], R[2, 2])
        kappa = math.atan2(-R[1, 0], R[0, 0])

    return omega, phi, kappa


def _Rx(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=np.float64)


def _Ry(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=np.float64)


def _Rz(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=np.float64)


def opk_to_R(omega_rad, phi_rad, kappa_rad):
    return (_Rz(kappa_rad) @ _Ry(phi_rad) @ _Rx(omega_rad)).astype(np.float64)


# =========================================================
# 4. 俯视姿态：光轴向下；飞行方向对应画面上方
# =========================================================
def build_nadir_R_from_dir(dir_xy):
    z_c_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    if dir_xy is None or np.linalg.norm(dir_xy) < 1e-6:
        dir_xy = np.array([1.0, 0.0], dtype=np.float32)
    else:
        dir_xy = np.asarray(dir_xy, dtype=np.float32)
        dir_xy /= np.linalg.norm(dir_xy) + 1e-8

    # 飞行方向 -> 图像的“向上”，对应 -y_c
    y_c_w = np.array([-dir_xy[0], -dir_xy[1], 0.0], dtype=np.float32)
    y_c_w /= np.linalg.norm(y_c_w) + 1e-8

    # 右手系：x_c = y_c × z_c
    x_c_w = np.cross(y_c_w, z_c_w)
    x_c_w /= np.linalg.norm(x_c_w) + 1e-8

    R = np.stack([x_c_w, y_c_w, z_c_w], axis=0)
    return R.astype(np.float32)


# =========================================================
# 4.5 只在偏航角上插值，用于“原地转向”
# =========================================================
def _angle_diff(yaw_to, yaw_from):
    return (yaw_to - yaw_from + math.pi) % (2.0 * math.pi) - math.pi


def sample_yaw_only_dirs(dir_from, dir_to, step_deg=10.0):
    """
    返回一串“中间方向向量”(xy)，用来在原地做 yaw 插值。
    """
    dir_from = np.asarray(dir_from, dtype=np.float32)
    dir_to = np.asarray(dir_to, dtype=np.float32)

    if np.linalg.norm(dir_from) < 1e-6:
        dir_from = np.array([1.0, 0.0], dtype=np.float32)
    else:
        dir_from /= np.linalg.norm(dir_from) + 1e-8

    if np.linalg.norm(dir_to) < 1e-6:
        return []
    dir_to /= np.linalg.norm(dir_to) + 1e-8

    yaw_from = math.atan2(dir_from[1], dir_from[0])
    yaw_to = math.atan2(dir_to[1], dir_to[0])
    d_yaw = _angle_diff(yaw_to, yaw_from)

    total_deg = abs(math.degrees(d_yaw))
    if total_deg < 1e-3:
        return []

    n_steps = max(1, int(total_deg // step_deg))
    dirs = []
    for i in range(1, n_steps + 1):
        alpha = i / float(n_steps)
        yaw_i = yaw_from + d_yaw * alpha
        dirs.append(np.array([math.cos(yaw_i), math.sin(yaw_i)], dtype=np.float32))
    return dirs


# =========================================================
# 5. 在两点之间按固定步长采样直线路径
# =========================================================
def sample_line_points(p0, p1, step=1.0, include_endpoint=True, skip_first=False):
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    vec = p1 - p0
    length = float(np.linalg.norm(vec))

    if length < 1e-6:
        if include_endpoint and not skip_first:
            return [p0]
        if include_endpoint and skip_first:
            return [p1]
        return []

    dir_vec = vec / length
    num_full_steps = int(math.floor(length / step))

    points = []
    start_idx = 0 if not skip_first else 1
    for i in range(start_idx, num_full_steps + 1):
        t = i * step
        if t > length:
            break
        points.append(p0 + dir_vec * t)

    if include_endpoint:
        if len(points) == 0 or np.linalg.norm(points[-1] - p1) > 1e-4:
            points.append(p1)
    return points


# =========================================================
# 6. 读取 building 轮廓（txt）：每栋楼一个文件
# =========================================================
def parse_building_contours_from_txt_folder(contour_dir):
    if not os.path.isdir(contour_dir):
        raise FileNotFoundError(f"contour_dir 不存在: {contour_dir}")

    txts = sorted([fn for fn in os.listdir(contour_dir) if fn.lower().endswith(".txt")])
    if not txts:
        raise FileNotFoundError(f"在目录中未找到 txt: {contour_dir}")

    buildings = []
    for fn in txts:
        path = os.path.join(contour_dir, fn)

        pts_xyz = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2]) if len(parts) >= 3 else 0.0
                except Exception:
                    continue
                pts_xyz.append([x, y, z])

        if len(pts_xyz) < 3:
            print(f"[WARN] {fn} 轮廓点 < 3，跳过")
            continue

        xyz = np.asarray(pts_xyz, dtype=np.float32)

        # 去掉连续重复点（按 xy 判断）
        cleaned = [xyz[0]]
        for i in range(1, xyz.shape[0]):
            if np.linalg.norm(xyz[i, :2] - cleaned[-1][:2]) > 1e-6:
                cleaned.append(xyz[i])
        xyz = np.asarray(cleaned, dtype=np.float32)

        target_z = float(np.median(xyz[:, 2]))
        name = os.path.splitext(fn)[0]

        buildings.append(
            {
                "name": name,          # 常规名称（txt 文件名）
                "title": name,         # 预留：后续若接 xlsx，可替换为真实标题
                "contour_xyz": xyz,
                "contour_xy": xyz[:, :2],
                "target_z": target_z,
            }
        )

    if not buildings:
        raise RuntimeError(f"未成功读取任何 building 轮廓: {contour_dir}")

    print(f"[INFO] 从 {contour_dir} 读取到 {len(buildings)} 个 building 轮廓")
    return buildings


def close_polyline_xy(xy):
    xy = np.asarray(xy, dtype=np.float32)
    if np.linalg.norm(xy[0] - xy[-1]) > 1e-6:
        xy = np.vstack([xy, xy[0]])
    return xy


def resample_closed_polyline(xy_closed, step=1.0):
    """
    输入：闭合 polyline（首尾相同） [M,2]
    输出：沿周长按 step 等间距采样的点 [N,2]，且最后不会重复首点
    """
    xy = np.asarray(xy_closed, dtype=np.float32)
    if xy.shape[0] < 4:
        raise ValueError("闭合轮廓点过少")

    seg = xy[1:] - xy[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    total = float(np.sum(seg_len))
    if total < 1e-6:
        raise ValueError("轮廓周长为 0")

    cum = np.concatenate([[0.0], np.cumsum(seg_len)], axis=0)
    n = max(3, int(math.floor(total / step)))
    s = np.linspace(0.0, total, num=n, endpoint=False, dtype=np.float32)

    out = []
    j = 0
    for si in s:
        while j + 1 < len(cum) and cum[j + 1] <= si:
            j += 1
        denom = max(1e-8, (cum[j + 1] - cum[j]))
        t = (si - cum[j]) / denom
        p = xy[j] * (1 - t) + xy[j + 1] * t
        out.append(p)

    return np.asarray(out, dtype=np.float32)


def _closest_index(points_xy, query_xy):
    d = np.linalg.norm(points_xy - query_xy[None, :], axis=1)
    return int(np.argmin(d))


# =========================================================
# 6.5 轮廓平滑（位置平滑）：减少尖角导致的方向抖动
# =========================================================
def smooth_closed_points_xy_moving_average(xy_ring, window=5, iters=1):
    """
    对闭环点列做 circular moving average 平滑（不改变点数）。
    xy_ring: [N,2]，不重复首点，视为环
    """
    xy = np.asarray(xy_ring, dtype=np.float32)
    n = xy.shape[0]
    if n < 3 or window <= 1 or iters <= 0:
        return xy

    w = int(window)
    half = w // 2
    out = xy.copy()

    for _ in range(int(iters)):
        padded = np.concatenate([out[-half:], out, out[:half]], axis=0)
        new = np.zeros_like(out)
        for i in range(n):
            new[i] = np.mean(padded[i: i + w], axis=0)
        out = new

    return out.astype(np.float32)


# =========================================================
# 7. 轮廓外扩（2D）- shapely buffer
# =========================================================
def buffer_polygon_xy_shapely(xy_ring, expand_m, join_style="round", resolution=16, simplify_tol=0.0):
    """
    用 shapely 的 polygon.buffer 做真正的外扩（最稳），避免凹多边形外扩自交。
    输入:
      xy_ring: [N,2] 不重复首点（视为闭环）
      expand_m: 外扩距离（米）
      join_style: "round" | "mitre" | "bevel"
      resolution: 圆角近似分段数（越大越圆）
      simplify_tol: 可选，简化容差（米），0 表示不简化
    输出:
      out_xy: [M,2] 不重复首点（外环）
    """
    if expand_m <= 1e-8:
        return np.asarray(xy_ring, dtype=np.float32)

    try:
        from shapely.geometry import Polygon
    except Exception as e:
        raise RuntimeError("未安装 shapely，请先 pip install shapely") from e

    xy = np.asarray(xy_ring, dtype=np.float64)
    if xy.shape[0] < 3:
        return xy.astype(np.float32)

    poly = Polygon(xy)

    # 修复不合法情况
    if not poly.is_valid:
        poly = poly.buffer(0.0)

    if poly.is_empty:
        raise RuntimeError("输入轮廓无法构成有效多边形（poly.is_empty）")

    join_map = {"round": 1, "mitre": 2, "bevel": 3}
    js = join_map.get(join_style, 1)

    buf = poly.buffer(
        float(expand_m),
        resolution=int(resolution),
        join_style=js,
        mitre_limit=5.0,
    )

    if buf.is_empty:
        raise RuntimeError("buffer 结果为空，可能 expand_m 太小或输入有问题")

    # MultiPolygon: 取面积最大块
    geom = buf
    if geom.geom_type == "MultiPolygon":
        geom = max(list(geom.geoms), key=lambda g: g.area)

    coords = np.array(geom.exterior.coords, dtype=np.float64)
    if coords.shape[0] < 4:
        raise RuntimeError("buffer 外环点过少，无法用于轨迹")

    out = coords[:-1, :2].astype(np.float32)

    if simplify_tol > 1e-8:
        simp = geom.simplify(float(simplify_tol), preserve_topology=True)
        coords2 = np.array(simp.exterior.coords, dtype=np.float64)
        if coords2.shape[0] >= 4:
            out = coords2[:-1, :2].astype(np.float32)

    return out


# =========================================================
# 8. 投影与“画面位置标签”
# =========================================================
def project_point_to_image(P_rel, C_rel, omega_deg, phi_deg, kappa_deg, intr):
    omega = math.radians(float(omega_deg))
    phi = math.radians(float(phi_deg))
    kappa = math.radians(float(kappa_deg))
    R = opk_to_R(omega, phi, kappa)  # w2c

    P = np.asarray(P_rel, dtype=np.float64).reshape(3)
    C = np.asarray(C_rel, dtype=np.float64).reshape(3)
    pc = R @ (P - C)

    zc = float(pc[2])
    if abs(zc) < 1e-12:
        zc = 1e-12

    u = intr["fx"] * (pc[0] / zc) + intr["cx"]
    v = intr["fy"] * (pc[1] / zc) + intr["cy"]
    return float(u), float(v), float(pc[2])


def uv_to_region_label(u, v, width, height):
    W = float(width)
    H = float(height)

    if u < W / 3.0:
        x = "left"
    elif u < 2.0 * W / 3.0:
        x = "center"
    else:
        x = "right"

    if v < H / 3.0:
        y = "top"
    elif v < 2.0 * H / 3.0:
        y = "center"
    else:
        y = "bottom"

    if x == "center" and y == "center":
        return "center"
    if x == "center":
        return y
    if y == "center":
        return x
    return f"{y}-{x}"


def make_instruction_from_region(region_label, lang="zh"):
    if lang.lower() == "en":
        if region_label == "center":
            return "Orbit the building in the center of the view once."
        return f"Orbit the building in the {region_label} of the view once."

    mapping = {
        "top-left": "左上角",
        "top": "上方",
        "top-right": "右上角",
        "left": "左侧",
        "center": "中央",
        "right": "右侧",
        "bottom-left": "左下角",
        "bottom": "下方",
        "bottom-right": "右下角",
    }
    pos = mapping.get(region_label, "某处")
    return f"巡视画面{pos}的建筑"


# =========================================================
# 9. 生成绕楼轨迹（保留你修改后的逻辑） + 记录 subtask pose 范围
#    修复：按 building 类型使用不同高度配置
# =========================================================
def _append_pose(poses, pose_id, C_rel, dir_xy):
    R_w2c = build_nadir_R_from_dir(dir_xy)
    om, ph, ka = R_to_opk(R_w2c)
    poses.append(
        {
            "id": pose_id,
            "C_rel": np.asarray(C_rel, dtype=np.float64),
            "omega": math.degrees(om),
            "phi": math.degrees(ph),
            "kappa": math.degrees(ka),
        }
    )
    return pose_id + 1


def _append_yaw_interp_at_same_pos(poses, pose_id, C_rel, dir_from, dir_to, yaw_step_deg):
    yaw_dirs = sample_yaw_only_dirs(dir_from, dir_to, step_deg=yaw_step_deg)
    for dxy in yaw_dirs:
        pose_id = _append_pose(poses, pose_id, C_rel, dxy)
    return pose_id


def _normalize_seg_range(seg_start, seg_end, fallback_pose_id):
    """
    seg_end < seg_start 表示该段没有新增 pose（例如 yaw 插值为 0）。
    这种情况下用 fallback_pose_id 作为 [start,end]，保证 subtask 有有效范围。
    """
    if seg_end < seg_start:
        return int(fallback_pose_id), int(fallback_pose_id)
    return int(seg_start), int(seg_end)


def generate_poses_from_building_contours(
    buildings,
    num_traj_per_building=3,
    start_radius=60.0,
    start_h_min=20.0,
    start_h_max=40.0,
    end_height=-20.0,
    step=1.0,
    yaw_step_deg=10.0,
    contour_expand=10.0,
    pos_smooth_window=7,
    pos_smooth_iters=2,
    yaw_smooth_window=9,
    rng=None,
    env_id: Optional[str] = None,
    enable_building_height_override: bool = True,
):
    if rng is None:
        rng = np.random.default_rng()

    poses = []
    traj_infos = []
    pose_id = 0
    traj_global_id = 0

    for b in buildings:
        if traj_global_id % 50 == 0:
            print(traj_global_id)

        # ====== 每栋楼单独确定高度配置（关键修复）======
        height_cfg = resolve_height_config_for_building(
            env_id=env_id or "",
            building=b,
            default_start_h_min=start_h_min,
            default_start_h_max=start_h_max,
            default_end_height=end_height,
            enable_building_override=enable_building_height_override,
        )
        b_start_h_min = float(height_cfg["start_h_min"])
        b_start_h_max = float(height_cfg["start_h_max"])
        b_end_height = float(height_cfg["end_height"])
        matched_rule = height_cfg.get("matched_rule", None)

        name = b["name"]
        target_z = float(b.get("target_z", b_end_height))

        # if traj_global_id < 10:  # 前几条打印一下方便检查
        #     print(
        #         f"[INFO] building={name} "
        #         f"(rule={matched_rule if matched_rule else 'default'}) "
        #         f"-> start_h=[{b_start_h_min}, {b_start_h_max}], end_height={b_end_height}"
        #     )

        contour_xy = close_polyline_xy(b["contour_xy"])
        contour_xy_rs = resample_closed_polyline(contour_xy, step=step)

        centroid_xy = np.mean(contour_xy_rs, axis=0).astype(np.float32)

        contour_xy_rs_s = smooth_closed_points_xy_moving_average(
            contour_xy_rs, window=pos_smooth_window, iters=pos_smooth_iters
        )

        contour_xy_buf = buffer_polygon_xy_shapely(
            contour_xy_rs_s,
            contour_expand,
            join_style="round",
            resolution=16,
        )

        contour_xy_rs_exp_s = resample_closed_polyline(
            close_polyline_xy(contour_xy_buf),
            step=step
        )

        for _ in range(num_traj_per_building):
            # ---- 随机起点 ----
            u = rng.random()
            r = start_radius * math.sqrt(u)
            theta = 2.0 * math.pi * rng.random()
            start_xy = centroid_xy + np.array([math.cos(theta), math.sin(theta)], dtype=np.float32) * r
            z_start = float(rng.uniform(b_start_h_min, b_start_h_max))
            start = np.array([start_xy[0], start_xy[1], z_start], dtype=np.float32)

            # ---- entry: start 最近的外扩轮廓点 ----
            idx0 = _closest_index(contour_xy_rs_exp_s, start_xy)
            entry_xy = contour_xy_rs_exp_s[idx0].astype(np.float32)
            entry = np.array([entry_xy[0], entry_xy[1], z_start], dtype=np.float32)
            entry_low = np.array([entry_xy[0], entry_xy[1], b_end_height], dtype=np.float32)

            # ---- 随机初始航向：保持到 entry_low 才转弯 ----
            theta0 = 2.0 * math.pi * rng.random()
            init_dir_xy = np.array([math.cos(theta0), math.sin(theta0)], dtype=np.float32)

            traj_pose_start_id = pose_id

            # ========== subtask 0: 平移 start -> entry ==========
            seg0_start = pose_id
            pts_to_entry = sample_line_points(start, entry, step=step, include_endpoint=True, skip_first=False)
            for p in pts_to_entry:
                pose_id = _append_pose(poses, pose_id, p, init_dir_xy)
            seg0_end = pose_id - 1

            # ========== subtask 1: 下降 entry -> entry_low ==========
            seg1_start = pose_id
            pts_drop = sample_line_points(entry, entry_low, step=step, include_endpoint=True, skip_first=True)
            for p in pts_drop:
                pose_id = _append_pose(poses, pose_id, p, init_dir_xy)
            seg1_end = pose_id - 1

            # 这一段结束时，已有的最后一个 pose
            last_pose_after_seg1 = seg1_end if seg1_end >= seg1_start else seg0_end

            # ---- 到轨道起点后：转弯到绕楼切线方向 ----
            loop_xy = np.concatenate([contour_xy_rs_exp_s[idx0:], contour_xy_rs_exp_s[:idx0]], axis=0)
            loop_z = float(b_end_height)
            n_loop = loop_xy.shape[0]

            yaws = []
            for i in range(n_loop):
                p_xy = loop_xy[i]
                p_next = loop_xy[(i + 1) % n_loop]
                t = p_next - p_xy
                if np.linalg.norm(t) < 1e-6:
                    t = np.array([1.0, 0.0], dtype=np.float32)
                else:
                    t = t / (np.linalg.norm(t) + 1e-8)
                yaws.append(math.atan2(float(t[1]), float(t[0])))

            dirs_raw = np.stack([np.cos(yaws), np.sin(yaws)], axis=1).astype(np.float32)
            dirs_sm = smooth_closed_points_xy_moving_average(dirs_raw, window=yaw_smooth_window, iters=1)
            norm = np.linalg.norm(dirs_sm, axis=1, keepdims=True)
            dirs_s = dirs_sm / (norm + 1e-8)

            first_dir = dirs_s[0]

            # ========== subtask 2: init_dir -> first_dir 原地转向插值 ==========
            seg2_start = pose_id
            pose_id = _append_yaw_interp_at_same_pos(
                poses, pose_id, entry_low, init_dir_xy, first_dir, yaw_step_deg=yaw_step_deg
            )
            seg2_end = pose_id - 1
            seg2_s, seg2_e = _normalize_seg_range(seg2_start, seg2_end, last_pose_after_seg1)

            # ========== subtask 3: 正式绕楼（包含方向突变处原地转向 + 轨迹点） ==========
            seg3_start = pose_id
            prev_dir = first_dir.copy()
            for i in range(n_loop):
                p_xy = loop_xy[i]
                p = np.array([p_xy[0], p_xy[1], loop_z], dtype=np.float32)
                cur_dir = dirs_s[i]

                pose_id = _append_yaw_interp_at_same_pos(
                    poses, pose_id, p, prev_dir, cur_dir, yaw_step_deg=yaw_step_deg
                )
                pose_id = _append_pose(poses, pose_id, p, cur_dir)
                prev_dir = cur_dir
            seg3_end = pose_id - 1
            seg3_s, seg3_e = _normalize_seg_range(seg3_start, seg3_end, seg2_e)

            # ========== subtask 4: 上升回原始高度 ==========
            last_xy = loop_xy[-1].astype(np.float32)
            last_low = np.array([last_xy[0], last_xy[1], b_end_height], dtype=np.float32)
            last_high = np.array([last_xy[0], last_xy[1], z_start], dtype=np.float32)

            seg4_start = pose_id
            pts_up = sample_line_points(last_low, last_high, step=step, include_endpoint=True, skip_first=True)
            for p in pts_up:
                pose_id = _append_pose(poses, pose_id, p, prev_dir)
            seg4_end = pose_id - 1
            seg4_s, seg4_e = _normalize_seg_range(seg4_start, seg4_end, seg3_e)

            # ========== subtask 5: 平移回 start ==========
            seg5_start = pose_id
            pts_back = sample_line_points(last_high, start, step=step, include_endpoint=True, skip_first=True)
            for p in pts_back:
                pose_id = _append_pose(poses, pose_id, p, prev_dir)
            seg5_end = pose_id - 1
            seg5_s, seg5_e = _normalize_seg_range(seg5_start, seg5_end, seg4_e)

            traj_pose_end_id = seg5_e
            if traj_pose_end_id >= traj_pose_start_id:
                # 归一化 seg0/seg1（以防极端情况）
                seg0_s, seg0_e = _normalize_seg_range(seg0_start, seg0_end, traj_pose_start_id)
                seg1_s, seg1_e = _normalize_seg_range(seg1_start, seg1_end, seg0_e)

                subtasks = [
                    {"subtask_id": 0, "pose_id_start": seg0_s, "pose_id_end": seg0_e},
                    {"subtask_id": 1, "pose_id_start": seg1_s, "pose_id_end": seg1_e},
                    {"subtask_id": 2, "pose_id_start": seg2_s, "pose_id_end": seg2_e},
                    {"subtask_id": 3, "pose_id_start": seg3_s, "pose_id_end": seg3_e},
                    {"subtask_id": 4, "pose_id_start": seg4_s, "pose_id_end": seg4_e},
                    {"subtask_id": 5, "pose_id_start": seg5_s, "pose_id_end": seg5_e},
                ]

                traj_infos.append(
                    {
                        "traj_id": traj_global_id,
                        "loc_name": name,
                        "title": b.get("title", name),
                        "pose_id_start": traj_pose_start_id,
                        "pose_id_end": traj_pose_end_id,
                        "target_xyz": np.array([centroid_xy[0], centroid_xy[1], target_z], dtype=np.float64),
                        "subtasks": subtasks,
                        # 记录该轨迹实际使用的高度配置（供 subtask 文本 / 调试）
                        "start_h_min_used": b_start_h_min,
                        "start_h_max_used": b_start_h_max,
                        "end_height_used": b_end_height,
                        "height_rule": matched_rule,
                    }
                )
                traj_global_id += 1

    print(f"[INFO] 总共生成 {len(poses)} 个轨迹点（pose），共 {len(traj_infos)} 条轨迹")
    return poses, traj_infos


# =========================================================
# 10. 写 PLY
# =========================================================
def write_camera_points_ply(cameras, ply_path, color=(255, 0, 0), use_offset=True):
    global DJI_OFFSET
    if use_offset and DJI_OFFSET is None:
        raise RuntimeError("DJI_OFFSET 还未初始化，请先调用主函数或手动设置 DJI_OFFSET")

    verts = []
    for c in cameras:
        C_world = c["C_world"]
        C = C_world.astype(np.float64) - DJI_OFFSET if use_offset else C_world
        verts.append(C)

    verts = np.array(verts, dtype=np.float32)
    r, g, b = color
    N = verts.shape[0]

    os.makedirs(os.path.dirname(ply_path) or ".", exist_ok=True)
    with open(ply_path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(N):
            x, y, z = verts[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


def write_cameras_and_contours_ply(poses, buildings, ply_path, cam_color=(0, 255, 0), contour_color=(0, 0, 255)):
    cam_pts = np.array([np.asarray(p["C_rel"], dtype=np.float64) for p in poses], dtype=np.float64)

    contour_pts_list = []
    for b in buildings:
        xyz = np.asarray(b["contour_xyz"], dtype=np.float64)
        contour_pts_list.append(xyz)
    contour_pts = np.concatenate(contour_pts_list, axis=0) if contour_pts_list else np.zeros((0, 3), dtype=np.float64)

    N_cam = cam_pts.shape[0]
    N_cont = contour_pts.shape[0]
    N = N_cam + N_cont

    cr, cg, cb = cam_color
    br, bg, bb = contour_color

    os.makedirs(os.path.dirname(ply_path) or ".", exist_ok=True)
    with open(ply_path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")

        for i in range(N_cam):
            x, y, z = cam_pts[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {cr} {cg} {cb}\n")

        for i in range(N_cont):
            x, y, z = contour_pts[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {br} {bg} {bb}\n")


# =========================================================
# 11. 输出路径逻辑：按 env_id 输出到 OUT_ROOT/env_id/（替代 train/test）
# =========================================================
def resolve_outputs_by_env(
    env_id: str,
    out_txt: str,
    out_ply_xml: str,
    out_ply_random: str,
    out_instr: str,
    out_meta: str,
    out_subtask: str,
):
    """
    所有输出统一写到:
      OUT_ROOT / env_id / ...
    """
    base = OUT_ROOT / env_id
    return (
        base / out_txt,
        base / out_ply_xml,
        base / out_ply_random,
        base / out_instr,
        base / out_meta,
        base / out_subtask,
    )


def resolve_input_paths_by_env(
    env_id: str,
    xml_path: Optional[str],
    metadata_path: Optional[str],
    contour_dir: Optional[str],
):
    """
    若未显式传入路径，则按 DATA_ROOT/env_id 自动拼接：
      xml_path      = DATA_ROOT/env_id/BlocksExchangeUndistortAT_WithoutTiePoints.xml
      metadata_path = DATA_ROOT/env_id/terra_ply/metadata.xml
      contour_dir   = DATA_ROOT/env_id/building_coords
    """
    env_root = DATA_ROOT / env_id

    xml_path = Path(xml_path) if xml_path else (env_root / "BlocksExchangeUndistortAT_WithoutTiePoints.xml")
    metadata_path = Path(metadata_path) if metadata_path else (env_root / "terra_ply" / "metadata.xml")
    contour_dir = Path(contour_dir) if contour_dir else (env_root / "building_coords")

    return str(xml_path), str(metadata_path), str(contour_dir)


# =========================================================
# 12. 主入口
# =========================================================
def main():
    parser = ArgumentParser(
        description="Generate DJI 3DGS trajectories around building contours (No global->local conversion). "
                    "Inputs default to DATA_ROOT/env_id, outputs go to OUT_ROOT/env_id."
    )

    # -------- env / root config (新增) --------
    parser.add_argument(
        "--env_id",
        type=str,
        required=True,
        choices=list(ENV_CONFIGS.keys()),
        help="场景 ID，例如: 1_office / 2_city / 3_road / 4_lake",
    )

    # -------- inputs (默认按 DATA_ROOT + env_id 自动拼接；也可手动覆盖) --------
    parser.add_argument(
        "--xml_path",
        type=str,
        default=None,
        help="默认: DATA_ROOT/env_id/BlocksExchangeUndistortAT_WithoutTiePoints.xml",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default=None,
        help="默认: DATA_ROOT/env_id/terra_ply/metadata.xml（用于读取 SRSOrigin）",
    )
    parser.add_argument(
        "--contour_dir",
        type=str,
        default=None,
        help="默认: DATA_ROOT/env_id/building_coords（每栋楼一个 txt）",
    )

    # -------- outputs (都放在 OUT_ROOT/env_id/ 下面；这里填写文件名即可) --------
    parser.add_argument("--out_txt", type=str, default="traj_random.txt")
    parser.add_argument("--out_ply_xml", type=str, default="cameras_xml.ply")
    parser.add_argument("--out_ply_random", type=str, default="cameras_random.ply")
    parser.add_argument("--out_instr", type=str, default="instruction.txt", help="每条轨迹的指令与 id 范围")
    parser.add_argument("--out_meta", type=str, default="traj_meta.txt", help="traj_id 对应 building 和 pose id 范围")
    parser.add_argument("--out_subtask", type=str, default="subtask.txt", help="每条轨迹的子任务分段标签")

    # -------- hyperparams --------
    parser.add_argument("--traj_per_loc", type=int, default=15, help="每个 building 生成的轨迹数量")
    parser.add_argument("--start_radius", type=float, default=60.0, help="起点相对 building 中心的水平半径（m）")

    # 默认 None，parse 后按 ENV_CONFIGS[env_id] 自动填充；仍允许命令行手动覆盖
    parser.add_argument("--start_h_min", type=float, default=None, help="起点高度范围下界（m），默认随 env_id")
    parser.add_argument("--start_h_max", type=float, default=None, help="起点高度范围上界（m），默认随 env_id")
    parser.add_argument("--end_height", type=float, default=None, help="巡航高度（m），默认随 env_id")

    parser.add_argument("--sample_step", type=float, default=1.0, help="轨迹采样间隔（米）")
    parser.add_argument("--yaw_step_deg", type=float, default=5.0, help="原地转向时的偏航插值步长（度）")

    parser.add_argument("--contour_expand", type=float, default=8.0, help="轮廓外扩距离（m）")

    parser.add_argument("--instr_lang", type=str, default="en", help="instruction 语言：zh 或 en")

    parser.add_argument("--pos_smooth_window", type=int, default=9, help="巡航轮廓位置平滑窗口(奇数更好)")
    parser.add_argument("--pos_smooth_iters", type=int, default=2, help="巡航轮廓位置平滑迭代次数")
    parser.add_argument("--yaw_smooth_window", type=int, default=9, help="绕楼切线yaw平滑窗口(奇数更好)")

    # 新增：随机种子（复现）
    parser.add_argument("--seed", type=int, default=42, help="随机种子，方便复现")

    args = parser.parse_args()

    # ---------- seed for reproducibility ----------
    random.seed(args.seed)
    np.random.seed(args.seed)  # 防止后续混用 legacy np.random
    print(f"[INFO] seed = {args.seed}")

    # ---------- track whether user manually overrides heights ----------
    user_overrode_heights = any([
        args.start_h_min is not None,
        args.start_h_max is not None,
        args.end_height is not None,
    ])

    # ---------- env-specific defaults ----------
    if args.env_id not in ENV_CONFIGS:
        raise ValueError(f"未知 env_id: {args.env_id}. 可选: {list(ENV_CONFIGS.keys())}")
    env_cfg = ENV_CONFIGS[args.env_id]

    if args.start_h_min is None:
        args.start_h_min = float(env_cfg["start_h_min"])
    if args.start_h_max is None:
        args.start_h_max = float(env_cfg["start_h_max"])
    if args.end_height is None:
        args.end_height = float(env_cfg["end_height"])

    # 若用户手动覆盖高度参数，则不再使用 building 类型自动 override（避免冲突）
    enable_building_height_override = not user_overrode_heights
    print(f"[INFO] enable_building_height_override = {enable_building_height_override} "
          f"(user_overrode_heights={user_overrode_heights})")

    # ---------- resolve input paths by env_id ----------
    args.xml_path, args.metadata_path, args.contour_dir = resolve_input_paths_by_env(
        env_id=args.env_id,
        xml_path=args.xml_path,
        metadata_path=args.metadata_path,
        contour_dir=args.contour_dir,
    )

    # ---------- resolve output paths by env_id ----------
    out_txt, out_ply_xml, out_ply_random, out_instr, out_meta, out_subtask = resolve_outputs_by_env(
        env_id=args.env_id,
        out_txt=args.out_txt,
        out_ply_xml=args.out_ply_xml,
        out_ply_random=args.out_ply_random,
        out_instr=args.out_instr,
        out_meta=args.out_meta,
        out_subtask=args.out_subtask,
    )

    # Ensure output folder exists
    for p in [out_txt, out_ply_xml, out_ply_random, out_instr, out_meta, out_subtask]:
        os.makedirs(str(p.parent) if str(p.parent) else ".", exist_ok=True)

    print(f"[INFO] env_id = {args.env_id}")
    print(f"[INFO] DATA_ROOT = {DATA_ROOT}")
    print(f"[INFO] OUT_ROOT  = {OUT_ROOT}")
    print(f"[INFO] xml_path      = {args.xml_path}")
    print(f"[INFO] metadata_path = {args.metadata_path}")
    print(f"[INFO] contour_dir   = {args.contour_dir}")
    print(
        f"[INFO] base env config -> start_h_min={args.start_h_min}, "
        f"start_h_max={args.start_h_max}, end_height={args.end_height}"
    )

    # ---------- generate ----------
    global DJI_OFFSET
    DJI_OFFSET = parse_dji_offset_from_metadata(args.metadata_path)

    intr = parse_intrinsics_from_xml(args.xml_path)
    cameras_xml = parse_all_cameras_from_xml(args.xml_path)
    print(f"[INFO] 从 XML 中读取到 {len(cameras_xml)} 个原始相机")

    buildings = parse_building_contours_from_txt_folder(args.contour_dir)

    # 使用固定 seed 的 Generator，确保复现
    rng = np.random.default_rng(args.seed)
    poses, traj_infos = generate_poses_from_building_contours(
        buildings,
        num_traj_per_building=args.traj_per_loc,
        start_radius=args.start_radius,
        start_h_min=args.start_h_min,
        start_h_max=args.start_h_max,
        end_height=args.end_height,
        step=args.sample_step,
        yaw_step_deg=args.yaw_step_deg,
        contour_expand=args.contour_expand,
        pos_smooth_window=args.pos_smooth_window,
        pos_smooth_iters=args.pos_smooth_iters,
        yaw_smooth_window=args.yaw_smooth_window,
        rng=rng,
        env_id=args.env_id,
        enable_building_height_override=enable_building_height_override,
    )

    for p in poses:
        C_rel = p["C_rel"]
        p["C_world"] = C_rel + DJI_OFFSET

    # ---------- write traj_random.txt ----------
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(
            "# id  x_rel  y_rel  z_rel  omega_deg  phi_deg  kappa_deg\n"
            "# 注意: x_rel,y_rel,z_rel = C_world - DJI_OFFSET\n"
            f"# env_id = {args.env_id}\n"
            f"# seed = {args.seed}\n"
            f"# DJI_OFFSET = {DJI_OFFSET.tolist()}\n"
        )
        f.write(
            "# intrinsics: width height fx fy cx cy\n"
            f"# {intr['width']} {intr['height']} "
            f"{intr['fx']:.6f} {intr['fy']:.6f} {intr['cx']:.6f} {intr['cy']:.6f}\n"
        )
        f.write(
            f"# base traj_per_loc={args.traj_per_loc}, start_radius={args.start_radius}, "
            f"base_start_h=[{args.start_h_min},{args.start_h_max}], base_end_height={args.end_height}, "
            f"sample_step={args.sample_step}, yaw_step_deg={args.yaw_step_deg}, contour_expand={args.contour_expand}, "
            f"pos_smooth(window={args.pos_smooth_window},iters={args.pos_smooth_iters}), yaw_smooth_window={args.yaw_smooth_window}, "
            f"seed={args.seed}, enable_building_height_override={enable_building_height_override}\n"
            "# 注意：若启用 building height override，不同 building（如高楼/别墅）会使用不同高度配置生成轨迹\n"
            "# 每一行一个 pose，id 为全局递增\n"
        )
        for p in poses:
            c = p["C_rel"]
            f.write(
                f"{p['id']} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f} "
                f"{p['omega']:.6f} {p['phi']:.6f} {p['kappa']:.6f}\n"
            )
    print(f"[INFO] 轨迹 pose 已写入 txt: {out_txt}")

    # ---------- write instruction.txt ----------
    with open(out_instr, "w", encoding="utf-8") as f:
        f.write("# traj_id  pose_id_start  pose_id_end  instruction\n")

        for info in traj_infos:
            pid0 = int(info["pose_id_start"])
            p0 = poses[pid0]

            C0 = np.asarray(p0["C_rel"], dtype=np.float64)
            P_t = np.asarray(info["target_xyz"], dtype=np.float64)

            u, v, _ = project_point_to_image(
                P_t,
                C0,
                p0["omega"],
                p0["phi"],
                p0["kappa"],
                intr,
            )

            region = uv_to_region_label(u, v, intr["width"], intr["height"])
            instr = make_instruction_from_region(region, lang=args.instr_lang)

            f.write(f"{info['traj_id']} {info['pose_id_start']} {info['pose_id_end']} {instr}\n")
    print(f"[INFO] 轨迹 instruction 已写入: {out_instr}")

    # ---------- write subtask.txt ----------
    with open(out_subtask, "w", encoding="utf-8") as f:
        f.write("# traj_id  subtask_id  pose_id_start  pose_id_end  subtask\n")
        for info in traj_infos:
            traj_id = int(info["traj_id"])
            pid0 = int(info["pose_id_start"])
            p0 = poses[pid0]

            C0 = np.asarray(p0["C_rel"], dtype=np.float64)
            P_t = np.asarray(info["target_xyz"], dtype=np.float64)
            u, v, _ = project_point_to_image(
                P_t,
                C0,
                p0["omega"],
                p0["phi"],
                p0["kappa"],
                intr,
            )
            region = uv_to_region_label(u, v, intr["width"], intr["height"])

            end_h_used = float(info.get("end_height_used", args.end_height))

            def _move_to_region_edge_en(region_label: str) -> str:
                if region_label == "center":
                    return "Move to the building in the center of the view."
                return f"Move to the building in the {region_label} of the view."

            subtask_texts = [
                _move_to_region_edge_en(region),
                f"Descend to above the target building.",
                "Adjust heading.",
                "Orbit the target building.",
                "Ascend back to the original altitude.",
                "Move back to the starting point.",
            ]

            subtasks = info.get("subtasks", [])
            if not subtasks or len(subtasks) != 6:
                continue

            for st in subtasks:
                sid = int(st["subtask_id"])
                s0 = int(st["pose_id_start"])
                s1 = int(st["pose_id_end"])
                f.write(f"{traj_id} {sid} {s0} {s1} {subtask_texts[sid]}\n")

    print(f"[INFO] 轨迹 subtask 已写入: {out_subtask}")

    # ---------- write plys ----------
    write_camera_points_ply(cameras_xml, str(out_ply_xml), color=(255, 0, 0), use_offset=True)
    write_cameras_and_contours_ply(
        poses, buildings, str(out_ply_random), cam_color=(0, 255, 0), contour_color=(0, 0, 255)
    )
    print(f"[INFO] PLY 已写入: {out_ply_xml}, {out_ply_random}")

    # ---------- write traj_meta.txt ----------
    with open(out_meta, "w", encoding="utf-8") as f:
        f.write("# traj_id loc_name pose_id_start pose_id_end\n")
        for info in traj_infos:
            f.write(f"{info['traj_id']} {info['loc_name']} {info['pose_id_start']} {info['pose_id_end']}\n")
    print(f"[INFO] 轨迹 meta 已写入: {out_meta}")

    print("[INFO] 已禁用 global->local 转换：不生成 traj_random_local.txt")


if __name__ == "__main__":
    main()

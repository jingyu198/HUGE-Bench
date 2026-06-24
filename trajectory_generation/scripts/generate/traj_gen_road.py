# -*- coding: utf-8 -*-
"""
Road inspection trajectory generator (DJI 3DGS) from road txt polylines.

关键改动（相对原 building 版本）:
1) 输入坐标目录从 building_coords 改为 road_coords（每条道路一个 txt，格式同 building_coords）
2) 对每条道路，从“道路两端”分别开始生成巡检轨迹
3) 起点随机半径默认改小为 30 米
4) 新增 --offset_z：对 start_h_min / start_h_max / end_height 统一减去 offset_z
5) instruction 改为“巡视画面中的马路”（并按文件名关键词替换主语）
6) subtask 改为 5 段（并按文件名关键词替换主语）：
   0) 平移到XX上方
   1) 下降到巡航高度
   2) 调整转向
   3) 沿XX巡检
   4) 上升回原始高度

新增（本次改动）:
7) 新增 --end_inset_m：对每条道路开放折线两端各向内缩 end_inset_m 米作为“新的起止端点”
   - 若道路总长度 < 2*end_inset_m：直接跳过并警告（按你的要求）

IMPORTANT:
    VLN 轨迹不做 global -> local 转换！
    只输出相对坐标系下的轨迹（C_rel = C_world - DJI_OFFSET），不生成 traj_random_local.txt。
"""

import os
import math
import random
import numpy as np
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

# ==== offset 不再手动写死，而是从 metadata.xml 里读取 ====
DJI_OFFSET = None

# =========================================================
# 全局路径与场景配置
# =========================================================
DATA_ROOT = Path(os.environ.get("HUGE_DATA_3D_ROOT", "./data_3d")).expanduser()
OUT_ROOT = Path(os.environ.get("HUGE_DATA_TRAJ_ROOT", "./data_traj")).expanduser() / "task_road"

ENV_CONFIGS = {
    "1_office": {
        "start_h_min": 0.0,
        "start_h_max": 20.0,
        "end_height": -20.0,
    },
    "2_city": {
        "start_h_min": -20.0,
        "start_h_max": 0.0,
        "end_height": -40.0,
    },
    "3_road": {
        "start_h_min": -40.0,
        "start_h_max": -30.0,
        "end_height": -50.0,
    },
    "4_lake": {
        "start_h_min": 0.0,
        "start_h_max": 10.0,
        "end_height": -10.0,
    },
    "real_road": {
        "start_h_min": -38.0,
        "start_h_max": -32.0,
        "end_height": -35.0,
    },
}


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

    # 飞行方向 -> 图像“向上”，对应 -y_c
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
# 6. 读取 road 折线（txt）：每条道路一个文件
# =========================================================
def parse_road_polylines_from_txt_folder(road_dir):
    if not os.path.isdir(road_dir):
        raise FileNotFoundError(f"road_dir 不存在: {road_dir}")

    txts = sorted([fn for fn in os.listdir(road_dir) if fn.lower().endswith(".txt")])
    if not txts:
        raise FileNotFoundError(f"在目录中未找到 txt: {road_dir}")

    roads = []
    for fn in txts:
        path = os.path.join(road_dir, fn)

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

        if len(pts_xyz) < 2:
            print(f"[WARN] {fn} 道路点 < 2，跳过")
            continue

        xyz = np.asarray(pts_xyz, dtype=np.float32)

        # 去掉连续重复点（按 xy 判断）
        cleaned = [xyz[0]]
        for i in range(1, xyz.shape[0]):
            if np.linalg.norm(xyz[i, :2] - cleaned[-1][:2]) > 1e-6:
                cleaned.append(xyz[i])
        xyz = np.asarray(cleaned, dtype=np.float32)

        if xyz.shape[0] < 2:
            print(f"[WARN] {fn} 去重后道路点 < 2，跳过")
            continue

        # 若首尾重复（很多标注会闭合），对于道路巡检这里当作开放折线，去掉尾部重复点
        if xyz.shape[0] >= 3 and np.linalg.norm(xyz[0, :2] - xyz[-1, :2]) <= 1e-6:
            xyz = xyz[:-1]

        if xyz.shape[0] < 2:
            print(f"[WARN] {fn} 去掉闭合尾点后道路点 < 2，跳过")
            continue

        name = os.path.splitext(fn)[0]
        target_z = float(np.median(xyz[:, 2]))
        roads.append(
            {
                "name": name,
                "title": name,
                "polyline_xyz": xyz,
                "polyline_xy": xyz[:, :2],
                "target_z": target_z,
            }
        )

    if not roads:
        raise RuntimeError(f"未成功读取任何 road 折线: {road_dir}")

    print(f"[INFO] 从 {road_dir} 读取到 {len(roads)} 条 road 折线")
    return roads


# =========================================================
# 6.5 开放折线重采样与平滑
# =========================================================
def resample_open_polyline(xy, step=1.0):
    """
    输入：开放 polyline [M,2]
    输出：沿长度按 step 等间距采样的点 [N,2]，包含终点
    """
    xy = np.asarray(xy, dtype=np.float32)
    if xy.shape[0] < 2:
        raise ValueError("开放折线点过少")

    seg = xy[1:] - xy[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    total = float(np.sum(seg_len))
    if total < 1e-6:
        raise ValueError("折线总长度为 0")

    cum = np.concatenate([[0.0], np.cumsum(seg_len)], axis=0)

    s = list(np.arange(0.0, total, step, dtype=np.float32))
    if len(s) == 0 or abs(float(s[-1]) - total) > 1e-4:
        s.append(np.float32(total))

    out = []
    j = 0
    for si in s:
        si = float(si)
        while j + 1 < len(cum) and cum[j + 1] < si:
            j += 1
        if j + 1 >= len(cum):
            out.append(xy[-1])
            continue
        denom = max(1e-8, (cum[j + 1] - cum[j]))
        t = (si - cum[j]) / denom
        p = xy[j] * (1 - t) + xy[j + 1] * t
        out.append(p)

    out = np.asarray(out, dtype=np.float32)

    # 去连续重复
    cleaned = [out[0]]
    for i in range(1, out.shape[0]):
        if np.linalg.norm(out[i] - cleaned[-1]) > 1e-6:
            cleaned.append(out[i])
    out = np.asarray(cleaned, dtype=np.float32)

    return out


def smooth_open_points_xy_moving_average(xy, window=5, iters=1):
    """
    对开放点列做 edge-padded moving average 平滑（不改变点数）。
    xy: [N,2]
    """
    xy = np.asarray(xy, dtype=np.float32)
    n = xy.shape[0]
    if n < 3 or window <= 1 or iters <= 0:
        return xy

    w = int(window)
    if w % 2 == 0:
        w += 1
    half = w // 2

    out = xy.copy()
    for _ in range(int(iters)):
        padded = np.pad(out, ((half, half), (0, 0)), mode="edge")
        new = np.zeros_like(out)
        for i in range(n):
            new[i] = np.mean(padded[i : i + w], axis=0)
        out = new

    return out.astype(np.float32)


def trim_open_polyline_by_distance(xy, inset_start_m=20.0, inset_end_m=20.0, eps=1e-6):
    """
    对开放折线 xy[N,2]，从起点向内缩 inset_start_m，从终点向内缩 inset_end_m，
    返回内缩后的开放折线（仍为 [K,2]，K>=2）。
    若总长度不足以裁剪（<= inset_start_m + inset_end_m），返回 None。
    """
    xy = np.asarray(xy, dtype=np.float32)
    if xy.shape[0] < 2:
        return None

    seg = xy[1:] - xy[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    total = float(np.sum(seg_len))
    if total < eps:
        return None

    s0 = float(max(0.0, inset_start_m))
    s1 = float(total - max(0.0, inset_end_m))
    if s1 - s0 < 1e-3:
        return None

    cum = np.concatenate([[0.0], np.cumsum(seg_len)], axis=0)  # len = N

    def point_at_s(s):
        s = float(np.clip(s, 0.0, total))
        j = int(np.searchsorted(cum, s, side="right") - 1)
        j = max(0, min(j, len(cum) - 2))
        denom = max(eps, float(cum[j + 1] - cum[j]))
        t = (s - float(cum[j])) / denom
        return (xy[j] * (1.0 - t) + xy[j + 1] * t).astype(np.float32)

    p_start = point_at_s(s0)
    p_end = point_at_s(s1)

    i0 = int(np.searchsorted(cum, s0, side="left"))
    i1 = int(np.searchsorted(cum, s1, side="right") - 1)

    mid = xy[i0 : i1 + 1] if i1 >= i0 else np.zeros((0, 2), dtype=np.float32)

    out = [p_start]
    for p in mid:
        if np.linalg.norm(p - out[-1]) > 1e-6:
            out.append(p)
    if np.linalg.norm(p_end - out[-1]) > 1e-6:
        out.append(p_end)

    out = np.asarray(out, dtype=np.float32)
    if out.shape[0] < 2:
        return None
    return out


# =========================================================
# 7. 道路主语识别（用于 instruction / subtask 文本）
# =========================================================
def infer_road_subject_from_name(name: str) -> str:
    """
    按优先级匹配，避免“小马路”先被“马路”截获。
    """
    name = str(name or "")
    keywords = ["小马路", "马路", "小路", "水沟"]
    for kw in keywords:
        if kw in name:
            return kw
    return "马路"


def make_instruction_for_subject(subject_zh: str, lang: str = "zh") -> str:
    if str(lang).lower() == "en":
        zh2en = {
            "马路": "road",
            "小马路": "small road",
            "小路": "dirt road",
            "水沟": "canal",
        }
        subj = zh2en.get(subject_zh, "road")
        return f"Inspect the {subj} in the view"
    return f"巡视画面中的{subject_zh}"


def make_subtasks_for_subject(subject_zh: str, lang: str = "zh"):
    if str(lang).lower() == "en":
        zh2en = {
            "马路": "road",
            "小马路": "small road",
            "小路": "dirt road",
            "水沟": "canal",
        }
        subj = zh2en.get(subject_zh, "road")
        return [
            f"Move above the {subj}",
            "Descend to cruise altitude",
            "Adjust heading",
            f"Inspect along the {subj}",
            "Ascend back to the original altitude",
        ]

    return [
        f"平移到{subject_zh}上方",
        "下降到巡航高度",
        "调整转向",
        f"沿{subject_zh}巡检",
        "上升回原始高度",
    ]


# =========================================================
# 8. 生成沿路巡检轨迹（从道路两端出发）
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
    if seg_end < seg_start:
        return int(fallback_pose_id), int(fallback_pose_id)
    return int(seg_start), int(seg_end)


def _compute_open_path_dirs(path_xy, yaw_smooth_window=9):
    """
    给开放折线每个点分配一个前进方向（xy 单位向量）
    """
    path_xy = np.asarray(path_xy, dtype=np.float32)
    n = path_xy.shape[0]
    if n < 2:
        return np.array([[1.0, 0.0]], dtype=np.float32)

    dirs = []
    prev_valid = np.array([1.0, 0.0], dtype=np.float32)
    for i in range(n):
        if i < n - 1:
            d = path_xy[i + 1] - path_xy[i]
        else:
            d = path_xy[i] - path_xy[i - 1]

        norm = float(np.linalg.norm(d))
        if norm < 1e-6:
            d = prev_valid.copy()
        else:
            d = d / (norm + 1e-8)
            prev_valid = d.copy()
        dirs.append(d)

    dirs = np.asarray(dirs, dtype=np.float32)

    if yaw_smooth_window > 1 and n >= 3:
        dirs_sm = smooth_open_points_xy_moving_average(dirs, window=yaw_smooth_window, iters=1)
        norm = np.linalg.norm(dirs_sm, axis=1, keepdims=True)
        dirs = dirs_sm / (norm + 1e-8)

    return dirs.astype(np.float32)


def generate_poses_from_road_polylines(
    roads,
    num_traj_per_road_end=1,
    start_radius=30.0,
    start_h_min=20.0,
    start_h_max=40.0,
    end_height=-20.0,
    step=1.0,
    yaw_step_deg=10.0,
    pos_smooth_window=7,
    pos_smooth_iters=1,
    yaw_smooth_window=9,
    end_inset_m=20.0,  # <-- 新增：两端内缩距离（m）
    rng=None,
):
    """
    每条 road 从两端各生成 num_traj_per_road_end 条轨迹。
    子任务共 5 段：
      0 平移到XX上方
      1 下降到巡航高度
      2 调整转向
      3 沿XX巡检
      4 上升回原始高度

    end_inset_m:
      对道路开放折线两端各向内缩 end_inset_m 米，作为新的 head/tail 端点。
      若道路总长度 < 2*end_inset_m：直接跳过并警告。
    """
    if rng is None:
        rng = np.random.default_rng()

    poses = []
    traj_infos = []
    pose_id = 0
    traj_global_id = 0

    for road_idx, r in enumerate(roads):
        if road_idx % 50 == 0:
            print(f"[INFO] road index = {road_idx}")

        name = r["name"]
        subject = infer_road_subject_from_name(name)

        try:
            road_xy_rs = resample_open_polyline(r["polyline_xy"], step=step)
        except Exception as e:
            print(f"[WARN] 道路 {name} 重采样失败，跳过: {e}")
            continue

        if road_xy_rs.shape[0] < 2:
            print(f"[WARN] 道路 {name} 重采样后点数不足，跳过")
            continue

        road_xy_sm = smooth_open_points_xy_moving_average(
            road_xy_rs, window=pos_smooth_window, iters=pos_smooth_iters
        )

        # 再次去重，防止平滑后局部重复
        cleaned = [road_xy_sm[0]]
        for i in range(1, road_xy_sm.shape[0]):
            if np.linalg.norm(road_xy_sm[i] - cleaned[-1]) > 1e-6:
                cleaned.append(road_xy_sm[i])
        road_xy_sm = np.asarray(cleaned, dtype=np.float32)
        if road_xy_sm.shape[0] < 2:
            print(f"[WARN] 道路 {name} 平滑后点数不足，跳过")
            continue

        # ---- 两端内缩（关键新增）----
        if end_inset_m is not None and float(end_inset_m) > 1e-6:
            trimmed = trim_open_polyline_by_distance(
                road_xy_sm,
                inset_start_m=float(end_inset_m),
                inset_end_m=float(end_inset_m),
            )
            if trimmed is None or trimmed.shape[0] < 2:
                print(
                    f"[WARN] 道路 {name} 总长度不足以两端内缩 {end_inset_m}m（需要 >= {2*end_inset_m}m），跳过"
                )
                continue
            road_xy_sm = trimmed

        target_z = float(r.get("target_z", 0.0))
        z_cruise = float(end_height)

        # 从两端开始巡检：0 -> 正向，1 -> 反向
        for end_flag in [0, 1]:
            if end_flag == 0:
                path_xy = road_xy_sm.copy()
                start_end_name = "head"
            else:
                path_xy = road_xy_sm[::-1].copy()
                start_end_name = "tail"

            dirs_s = _compute_open_path_dirs(path_xy, yaw_smooth_window=yaw_smooth_window)
            first_dir = dirs_s[0]
            final_dir = dirs_s[-1]

            start_anchor_xy = path_xy[0]
            end_anchor_xy = path_xy[-1]

            for _ in range(int(num_traj_per_road_end)):
                # ---- 随机起点（围绕道路起点端）----
                u = rng.random()
                rr = float(start_radius) * math.sqrt(float(u))
                theta = 2.0 * math.pi * float(rng.random())
                start_xy = start_anchor_xy + np.array(
                    [math.cos(theta), math.sin(theta)], dtype=np.float32
                ) * rr

                z_start = float(rng.uniform(start_h_min, start_h_max))

                start = np.array([start_xy[0], start_xy[1], z_start], dtype=np.float32)
                entry = np.array([start_anchor_xy[0], start_anchor_xy[1], z_start], dtype=np.float32)
                entry_low = np.array([start_anchor_xy[0], start_anchor_xy[1], z_cruise], dtype=np.float32)

                # 初始随机航向（到达 entry_low 前保持）
                theta0 = 2.0 * math.pi * float(rng.random())
                init_dir_xy = np.array([math.cos(theta0), math.sin(theta0)], dtype=np.float32)

                traj_pose_start_id = pose_id

                # ========== subtask 0: 平移到XX上方（start -> entry）==========
                seg0_start = pose_id
                pts_to_entry = sample_line_points(start, entry, step=step, include_endpoint=True, skip_first=False)
                for p in pts_to_entry:
                    pose_id = _append_pose(poses, pose_id, p, init_dir_xy)
                seg0_end = pose_id - 1

                # ========== subtask 1: 下降到巡航高度（entry -> entry_low）==========
                seg1_start = pose_id
                pts_drop = sample_line_points(entry, entry_low, step=step, include_endpoint=True, skip_first=True)
                for p in pts_drop:
                    pose_id = _append_pose(poses, pose_id, p, init_dir_xy)
                seg1_end = pose_id - 1

                last_pose_after_seg1 = seg1_end if seg1_end >= seg1_start else seg0_end

                # ========== subtask 2: 原地转向（init_dir -> road first_dir）==========
                seg2_start = pose_id
                pose_id = _append_yaw_interp_at_same_pos(
                    poses, pose_id, entry_low, init_dir_xy, first_dir, yaw_step_deg=yaw_step_deg
                )
                seg2_end = pose_id - 1
                seg2_s, seg2_e = _normalize_seg_range(seg2_start, seg2_end, last_pose_after_seg1)

                # ========== subtask 3: 沿路巡检（沿开放折线前进）==========
                seg3_start = pose_id
                prev_dir = first_dir.copy()
                n_path = path_xy.shape[0]

                for i in range(n_path):
                    p_xy = path_xy[i]
                    p = np.array([p_xy[0], p_xy[1], z_cruise], dtype=np.float32)
                    cur_dir = dirs_s[i]

                    pose_id = _append_yaw_interp_at_same_pos(
                        poses, pose_id, p, prev_dir, cur_dir, yaw_step_deg=yaw_step_deg
                    )
                    pose_id = _append_pose(poses, pose_id, p, cur_dir)
                    prev_dir = cur_dir

                seg3_end = pose_id - 1
                seg3_s, seg3_e = _normalize_seg_range(seg3_start, seg3_end, seg2_e)

                # ========== subtask 4: 上升回原始高度（终点原地上升）==========
                last_low = np.array([end_anchor_xy[0], end_anchor_xy[1], z_cruise], dtype=np.float32)
                last_high = np.array([end_anchor_xy[0], end_anchor_xy[1], z_start], dtype=np.float32)

                seg4_start = pose_id
                pts_up = sample_line_points(last_low, last_high, step=step, include_endpoint=True, skip_first=True)
                for p in pts_up:
                    pose_id = _append_pose(poses, pose_id, p, prev_dir if len(poses) > 0 else final_dir)
                seg4_end = pose_id - 1
                seg4_s, seg4_e = _normalize_seg_range(seg4_start, seg4_end, seg3_e)

                traj_pose_end_id = seg4_e
                if traj_pose_end_id >= traj_pose_start_id:
                    seg0_s, seg0_e = _normalize_seg_range(seg0_start, seg0_end, traj_pose_start_id)
                    seg1_s, seg1_e = _normalize_seg_range(seg1_start, seg1_end, seg0_e)

                    subtasks = [
                        {"subtask_id": 0, "pose_id_start": seg0_s, "pose_id_end": seg0_e},
                        {"subtask_id": 1, "pose_id_start": seg1_s, "pose_id_end": seg1_e},
                        {"subtask_id": 2, "pose_id_start": seg2_s, "pose_id_end": seg2_e},
                        {"subtask_id": 3, "pose_id_start": seg3_s, "pose_id_end": seg3_e},
                        {"subtask_id": 4, "pose_id_start": seg4_s, "pose_id_end": seg4_e},
                    ]

                    mid_idx = int(len(path_xy) // 2)
                    target_xyz = np.array([path_xy[mid_idx, 0], path_xy[mid_idx, 1], target_z], dtype=np.float64)

                    traj_infos.append(
                        {
                            "traj_id": traj_global_id,
                            "loc_name": name,
                            "title": r.get("title", name),
                            "subject": subject,
                            "start_end": start_end_name,  # head / tail
                            "pose_id_start": traj_pose_start_id,
                            "pose_id_end": traj_pose_end_id,
                            "target_xyz": target_xyz,
                            "subtasks": subtasks,
                            "start_h_min_used": float(start_h_min),
                            "start_h_max_used": float(start_h_max),
                            "end_height_used": float(z_cruise),
                        }
                    )
                    traj_global_id += 1

    print(f"[INFO] 总共生成 {len(poses)} 个轨迹点（pose），共 {len(traj_infos)} 条轨迹")
    return poses, traj_infos


# =========================================================
# 9. 写 PLY
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


def write_cameras_and_roads_ply(poses, roads, ply_path, cam_color=(0, 255, 0), road_color=(0, 0, 255)):
    cam_pts = np.array([np.asarray(p["C_rel"], dtype=np.float64) for p in poses], dtype=np.float64)

    road_pts_list = []
    for r in roads:
        xyz = np.asarray(r["polyline_xyz"], dtype=np.float64)
        road_pts_list.append(xyz)
    road_pts = np.concatenate(road_pts_list, axis=0) if road_pts_list else np.zeros((0, 3), dtype=np.float64)

    N_cam = cam_pts.shape[0]
    N_road = road_pts.shape[0]
    N = N_cam + N_road

    cr, cg, cb = cam_color
    rr, rg, rb = road_color

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

        for i in range(N_road):
            x, y, z = road_pts[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {rr} {rg} {rb}\n")


# =========================================================
# 10. 输出路径逻辑：按 env_id 输出到 OUT_ROOT/env_id/
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
      contour_dir   = DATA_ROOT/env_id/road_coords
    """
    env_root = DATA_ROOT / env_id

    xml_path = Path(xml_path) if xml_path else (env_root / "BlocksExchangeUndistortAT_WithoutTiePoints.xml")
    metadata_path = Path(metadata_path) if metadata_path else (env_root / "terra_ply" / "metadata.xml")
    contour_dir = Path(contour_dir) if contour_dir else (env_root / "road_coords")

    return str(xml_path), str(metadata_path), str(contour_dir)


# =========================================================
# 11. 主入口
# =========================================================
def main():
    parser = ArgumentParser(
        description="Generate DJI 3DGS road-inspection trajectories from road polylines (No global->local conversion). "
        "Inputs default to DATA_ROOT/env_id, outputs go to OUT_ROOT/env_id."
    )

    # -------- env / root config --------
    parser.add_argument(
        "--env_id",
        type=str,
        required=True,
        choices=list(ENV_CONFIGS.keys()),
        help="场景 ID，例如: 1_office / 2_city / 3_road / 4_lake",
    )

    # -------- inputs（默认按 DATA_ROOT + env_id 自动拼接；也可手动覆盖）--------
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
        help="默认: DATA_ROOT/env_id/road_coords（每条道路一个 txt）",
    )

    # -------- outputs（都放在 OUT_ROOT/env_id/ 下面；这里填写文件名即可）--------
    parser.add_argument("--out_txt", type=str, default="traj_random.txt")
    parser.add_argument("--out_ply_xml", type=str, default="cameras_xml.ply")
    parser.add_argument("--out_ply_random", type=str, default="cameras_random.ply")
    parser.add_argument("--out_instr", type=str, default="instruction.txt", help="每条轨迹的指令与 id 范围")
    parser.add_argument("--out_meta", type=str, default="traj_meta.txt", help="traj_id 对应 road 和 pose id 范围")
    parser.add_argument("--out_subtask", type=str, default="subtask.txt", help="每条轨迹的子任务分段标签")

    # -------- hyperparams --------
    parser.add_argument(
        "--traj_per_loc",
        type=int,
        default=15,
        help="每条 road 的每个端点生成的轨迹数量（总数约为 2 * road_num * traj_per_loc）",
    )
    parser.add_argument("--start_radius", type=float, default=5.0, help="起点相对道路端点的水平随机半径（m）")

    # 默认 None，parse 后按 ENV_CONFIGS[env_id] 自动填充；仍允许命令行手动覆盖
    parser.add_argument("--start_h_min", type=float, default=None, help="起点高度范围下界（m），默认随 env_id")
    parser.add_argument("--start_h_max", type=float, default=None, help="起点高度范围上界（m），默认随 env_id")
    parser.add_argument("--end_height", type=float, default=None, help="巡航高度（m），默认随 env_id")

    # 新增：统一高度下移偏置
    parser.add_argument(
        "--offset_z",
        type=float,
        default=0,  #45
        help="对 start_h_min / start_h_max / end_height 统一减去的高度偏置值（m）",
    )

    # 新增：两端内缩距离
    parser.add_argument(
        "--end_inset_m",
        type=float,
        default=0,
        help="对每条道路开放折线，两端各向内缩的距离（m）。若道路总长度 < 2*end_inset_m 则跳过。",
    )

    parser.add_argument("--sample_step", type=float, default=0.1, help="轨迹采样间隔（米）")
    parser.add_argument("--yaw_step_deg", type=float, default=0.5, help="原地转向时的偏航插值步长（度）")

    parser.add_argument("--instr_lang", type=str, default="en", help="instruction/subtask 语言：zh 或 en")

    parser.add_argument("--pos_smooth_window", type=int, default=9, help="道路位置平滑窗口(奇数更好)")
    parser.add_argument("--pos_smooth_iters", type=int, default=1, help="道路位置平滑迭代次数")
    parser.add_argument("--yaw_smooth_window", type=int, default=9, help="道路切线yaw平滑窗口(奇数更好)")

    parser.add_argument("--seed", type=int, default=42, help="随机种子，方便复现")

    args = parser.parse_args()

    # ---------- seed for reproducibility ----------
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"[INFO] seed = {args.seed}")

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

    # ---------- apply unified height offset ----------
    if abs(float(args.offset_z)) > 1e-12:
        args.start_h_min -= float(args.offset_z)
        args.start_h_max -= float(args.offset_z)
        args.end_height -= float(args.offset_z)

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

    for p in [out_txt, out_ply_xml, out_ply_random, out_instr, out_meta, out_subtask]:
        os.makedirs(str(p.parent) if str(p.parent) else ".", exist_ok=True)

    print(f"[INFO] env_id = {args.env_id}")
    print(f"[INFO] DATA_ROOT = {DATA_ROOT}")
    print(f"[INFO] OUT_ROOT  = {OUT_ROOT}")
    print(f"[INFO] xml_path      = {args.xml_path}")
    print(f"[INFO] metadata_path = {args.metadata_path}")
    print(f"[INFO] road_dir       = {args.contour_dir}")
    print(
        f"[INFO] heights(after offset_z={args.offset_z}) -> "
        f"start_h_min={args.start_h_min}, start_h_max={args.start_h_max}, end_height={args.end_height}"
    )
    print(f"[INFO] end_inset_m = {args.end_inset_m}")

    # ---------- generate ----------
    global DJI_OFFSET
    DJI_OFFSET = parse_dji_offset_from_metadata(args.metadata_path)

    intr = parse_intrinsics_from_xml(args.xml_path)
    cameras_xml = parse_all_cameras_from_xml(args.xml_path)
    print(f"[INFO] 从 XML 中读取到 {len(cameras_xml)} 个原始相机")

    roads = parse_road_polylines_from_txt_folder(args.contour_dir)

    rng = np.random.default_rng(args.seed)
    poses, traj_infos = generate_poses_from_road_polylines(
        roads,
        num_traj_per_road_end=args.traj_per_loc,
        start_radius=args.start_radius,
        start_h_min=args.start_h_min,
        start_h_max=args.start_h_max,
        end_height=args.end_height,
        step=args.sample_step,
        yaw_step_deg=args.yaw_step_deg,
        pos_smooth_window=args.pos_smooth_window,
        pos_smooth_iters=args.pos_smooth_iters,
        yaw_smooth_window=args.yaw_smooth_window,
        end_inset_m=args.end_inset_m,  # <-- 新增
        rng=rng,
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
            f"# road inspection | traj_per_road_end={args.traj_per_loc}, start_radius={args.start_radius}, "
            f"start_h=[{args.start_h_min},{args.start_h_max}], end_height={args.end_height}, "
            f"offset_z={args.offset_z}, end_inset_m={args.end_inset_m}, sample_step={args.sample_step}, yaw_step_deg={args.yaw_step_deg}, "
            f"pos_smooth(window={args.pos_smooth_window},iters={args.pos_smooth_iters}), "
            f"yaw_smooth_window={args.yaw_smooth_window}, seed={args.seed}\n"
            "# 每条 road 从两端分别开始巡检（端点已按 end_inset_m 内缩）\n"
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
            subject = info.get("subject", "马路")
            instr = make_instruction_for_subject(subject, lang=args.instr_lang)
            f.write(f"{info['traj_id']} {info['pose_id_start']} {info['pose_id_end']} {instr}\n")
    print(f"[INFO] 轨迹 instruction 已写入: {out_instr}")

    # ---------- write subtask.txt ----------
    with open(out_subtask, "w", encoding="utf-8") as f:
        f.write("# traj_id  subtask_id  pose_id_start  pose_id_end  subtask\n")
        for info in traj_infos:
            traj_id = int(info["traj_id"])
            subject = info.get("subject", "马路")
            subtask_texts = make_subtasks_for_subject(subject, lang=args.instr_lang)

            subtasks = info.get("subtasks", [])
            if not subtasks or len(subtasks) != 5:
                continue

            for st in subtasks:
                sid = int(st["subtask_id"])
                s0 = int(st["pose_id_start"])
                s1 = int(st["pose_id_end"])
                f.write(f"{traj_id} {sid} {s0} {s1} {subtask_texts[sid]}\n")

    print(f"[INFO] 轨迹 subtask 已写入: {out_subtask}")

    # ---------- write plys ----------
    write_camera_points_ply(cameras_xml, str(out_ply_xml), color=(255, 0, 0), use_offset=True)
    write_cameras_and_roads_ply(
        poses, roads, str(out_ply_random), cam_color=(0, 255, 0), road_color=(0, 0, 255)
    )
    print(f"[INFO] PLY 已写入: {out_ply_xml}, {out_ply_random}")

    # ---------- write traj_meta.txt ----------
    with open(out_meta, "w", encoding="utf-8") as f:
        f.write("# traj_id loc_name pose_id_start pose_id_end\n")
        for info in traj_infos:
            f.write(
                f"{info['traj_id']} {info['loc_name']} "
                f"{info['pose_id_start']} {info['pose_id_end']}\n"
            )
    print(f"[INFO] 轨迹 meta 已写入: {out_meta}")

    print("[INFO] 已禁用 global->local 转换：不生成 traj_random_local.txt")


if __name__ == "__main__":
    main()

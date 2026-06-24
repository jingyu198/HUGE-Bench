# -*- coding: utf-8 -*-
"""
Merged script (orbit-around-landmark version, updated v3: radius-varying + fixed height)
============================================================================

✅ 不做 global -> local 转换（不生成 traj_random_local.txt）

✅ 参数优化（本次）：
只需要控制 --env_id（例如 1_office / 2_city / 3_road / 4_lake），即可自动映射：
- data_path = HUGE_DATA_3D_ROOT/{env_id}
- xml_path / metadata_path / location_path
- outputs root = HUGE_DATA_TRAJ_ROOT/task_orbit/{env_id}/
- start_h_min / start_h_max / end_height_candidates

✅ 本次改动（核心）：
1) 不再是“环绕高度变化”，而是“环绕半径变化”：
   - 轨道高度固定为 end_height_candidates 里的最小值（每条轨迹相同高度）
   - 每条轨迹的环绕半径从 {20, 40, 60} 中随机选一个
2) 其他过程不变，但 instruction/subtask 的英文叙述同步包含 orbit radius
3) 其他逻辑不变

✅ landmark_merged.txt 为 TSV：
   x \t y \t label \t label_no_pos
   - parse_locations_from_txt 按此格式解析（兼容老格式：x y name）

✅ instruction.txt：
   - 英文总指令示例：
     “Orbit the {label} once at {real_h} meters with a {R}-meter radius.”

✅ subtask.txt：
   - subtask 0 使用 label
   - subtask 1/2/3 使用 label_no_pos
   - 最后一个 subtask 拆成两个：先 turn_to_face_starting_point，再 fly_while_ascending_back
     => subtask_id 为 0..5
   - 并在 subtask 2/3 中加入 radius 描述

轨迹逻辑（不变）：
random start heading -> yaw to face landmark -> slanted descend -> yaw to enter orbit -> orbit once
-> yaw at orbit end to face start -> slanted ascend+return
"""

import os
import math
import numpy as np
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from pathlib import Path


DJI_OFFSET = None


# =========================================================
# 0.0 环境配置（通过 env_id 自动映射）
# =========================================================
ENV_CONFIGS = {
    "1_office": {
        "start_h_min": 20.0,
        "start_h_max": 40.0,
        "end_height_candidates": [-20.0, -10.0, 0.0],
    },
    "2_city": {
        "start_h_min": 0.0,
        "start_h_max": 20.0,
        "end_height_candidates": [-40.0, -30.0, -20.0],
    },
    "3_road": {
        "start_h_min": -10.0,
        "start_h_max": 10.0,
        "end_height_candidates": [-50.0, -40.0, -30.0],
    },
    "4_lake": {
        "start_h_min": 30.0,
        "start_h_max": 50.0,
        "end_height_candidates": [-10.0, 0.0, 10.0],
    },
}

DATA_ROOT = Path(os.environ.get("HUGE_DATA_3D_ROOT", "./data_3d")).expanduser()
OUT_ROOT = Path(os.environ.get("HUGE_DATA_TRAJ_ROOT", "./data_traj")).expanduser() / "task_orbit"


def resolve_env_paths_and_defaults(env_id: str):
    """
    根据 env_id 自动生成输入路径、输出目录、默认高度参数。
    """
    if env_id not in ENV_CONFIGS:
        raise ValueError(f"未知 env_id: {env_id}. 可选: {list(ENV_CONFIGS.keys())}")

    data_path = DATA_ROOT / env_id
    out_dir = OUT_ROOT / env_id
    cfg = ENV_CONFIGS[env_id]

    return {
        "data_path": data_path,
        "out_dir": out_dir,
        "xml_path": data_path / "BlocksExchangeUndistortAT_WithoutTiePoints.xml",
        "metadata_path": data_path / "terra_ply" / "metadata.xml",
        "location_path": data_path / "location_gen" / "landmark_merged_s.txt",
        "start_h_min": cfg["start_h_min"],
        "start_h_max": cfg["start_h_max"],
        "end_height_candidates": cfg["end_height_candidates"],
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
# 2.5 从 landmark_merged.txt 解析地标平面坐标（相对坐标）
#     TSV 格式: x \t y \t label \t label_no_pos
#     兼容老格式: x y name
# =========================================================
def parse_locations_from_txt(loc_path):
    """
    固定格式（空格或 TSV 都行）：
      x  y  z  label...
    只用 x,y；忽略 z；label 取第4列及之后（可含空格）。
    """
    if not os.path.isfile(loc_path):
        raise FileNotFoundError(f"location file 未找到: {loc_path}")

    locations = []

    with open(loc_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # 兼容 tab/空格：split() 会自动处理多个空格和 tab
            parts = line.split()
            if len(parts) < 4:
                continue

            x = float(parts[0])
            y = float(parts[1])
            # z = float(parts[2])  # 忽略
            name = " ".join(parts[3:]).strip(" ,，。；;")
            name_clean = name  # 你只有一种格式，就直接等于 name

            locations.append(
                {
                    "xy": np.array([x, y], dtype=np.float32),
                    "name": name,
                    "name_clean": name_clean,
                }
            )

    if not locations:
        raise RuntimeError(f"location file 中未解析到任何坐标: {loc_path}")

    print(f"[INFO] 从 location file 读取到 {len(locations)} 个地标点")
    return locations


# =========================================================
# 3. 根据 R_w2c 反求 Omega, Phi, Kappa（弧度）
# =========================================================
def R_to_opk(R):
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


# =========================================================
# 4. 构造俯视姿态（相机光轴向下；飞行方向在图像中朝上）
# =========================================================
def build_nadir_R_from_dir(dir_xy):
    z_c_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    if dir_xy is None or np.linalg.norm(dir_xy) < 1e-6:
        dir_xy = np.array([1.0, 0.0], dtype=np.float32)
    else:
        dir_xy = np.asarray(dir_xy, dtype=np.float32)
        dir_xy /= np.linalg.norm(dir_xy) + 1e-8

    y_c_w = np.array([-dir_xy[0], -dir_xy[1], 0.0], dtype=np.float32)
    y_c_w /= np.linalg.norm(y_c_w) + 1e-8

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
# 5. 直线采样（3D）
# =========================================================
def sample_line_points(p0, p1, step=1.0, include_endpoint=True, skip_first=False):
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    vec = p1 - p0
    length = float(np.linalg.norm(vec))

    if length < 1e-6:
        if include_endpoint and not skip_first:
            return [p0]
        elif include_endpoint and skip_first:
            return [p1]
        else:
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
# small 2D helpers
# =========================================================
def _norm2(v):
    return float(np.linalg.norm(v))


def _normalize2(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float32)
    n = _norm2(v)
    if n < eps:
        return v * 0.0
    return v / (n + eps)


def _left_normal(d):
    return np.array([-d[1], d[0]], dtype=np.float32)


def _right_normal(d):
    return np.array([d[1], -d[0]], dtype=np.float32)


# =========================================================
# 6. 轨迹生成（带 return 前的“原地转向对准起点”）
# =========================================================
def generate_poses_from_locations(
    locations,
    num_traj_per_loc=3,
    start_radius=50.0,
    start_h_min=80.0,
    start_h_max=100.0,
    end_height=-20.0,                 # deprecated fallback
    end_height_candidates=None,
    step=1.0,
    yaw_step_deg=10.0,
    orbit_radius=None,                # deprecated fixed radius fallback
    orbit_radius_candidates=None,     # NEW: list/tuple, choose one per traj
    orbit_ccw=True,
    rng=None,
    debug_print=False,
):
    """
    Segments (unchanged):
      seg0: yaw start (random init_dir -> face landmark)
      seg1: slanted descend to entry
      seg2: yaw entry (face landmark -> tangent)
      seg3: orbit once
      seg4: yaw at orbit end (tangent -> face start)
      seg5: slanted ascend+return to start

    Updates (this version):
      - Orbit height is FIXED to min(end_height_candidates)
      - Orbit radius is RANDOMLY chosen from orbit_radius_candidates for each trajectory
    """
    if rng is None:
        rng = np.random.default_rng()

    # ----- height candidates -> fixed min height -----
    if end_height_candidates is None or len(end_height_candidates) == 0:
        end_height_candidates = [float(end_height)]
    fixed_end_height_rel = float(min([float(x) for x in end_height_candidates]))

    # ----- radius candidates -----
    if orbit_radius_candidates is None or len(orbit_radius_candidates) == 0:
        if orbit_radius is not None:
            orbit_radius_candidates = [float(orbit_radius)]
        else:
            orbit_radius_candidates = [20.0, 40.0, 60.0]
    orbit_radius_candidates = [float(x) for x in orbit_radius_candidates]

    poses = []
    traj_infos = []
    pose_id = 0
    traj_global_id = 0

    for loc in locations:
        if traj_global_id % 100 == 0:
            print(traj_global_id)
        O_xy = np.asarray(loc["xy"], dtype=np.float32)
        label = loc.get("label", loc.get("name", "landmark"))
        label_no_pos = loc.get("label_no_pos", label)

        for _ in range(num_traj_per_loc):
            end_height_rel = fixed_end_height_rel
            R = float(rng.choice(orbit_radius_candidates))

            # ---- start point ----
            start = None
            start_xy = None
            z_start = None

            # 为了保证 start 在 orbit 圆外：距离 >= R + 1
            min_start_dist = R + 1.0
            eff_start_radius = float(start_radius)

            for _try in range(50):
                theta = 2.0 * math.pi * rng.random()

                # 在 [min_start_dist, eff_start_radius] 的环形区域内采样（面积均匀）
                if eff_start_radius > min_start_dist + 1e-6:
                    u = rng.random()
                    rr = math.sqrt(
                        (eff_start_radius**2 - min_start_dist**2) * u + min_start_dist**2
                    )
                else:
                    rr = min_start_dist

                offset_xy = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32) * rr
                start_xy = O_xy + offset_xy
                if _norm2(start_xy - O_xy) >= min_start_dist:
                    z_start = float(rng.uniform(start_h_min, start_h_max))
                    start = np.array([start_xy[0], start_xy[1], z_start], dtype=np.float32)
                    break

            if start is None:
                z_start = float(rng.uniform(start_h_min, start_h_max))
                if start_xy is None:
                    start_xy = O_xy + np.array([max(eff_start_radius, min_start_dist), 0.0], dtype=np.float32)
                start = np.array([start_xy[0], start_xy[1], z_start], dtype=np.float32)

            # ---- random initial heading ----
            theta0 = 2.0 * math.pi * rng.random()
            init_dir = np.array([math.cos(theta0), math.sin(theta0)], dtype=np.float32)

            traj_pose_start_id = pose_id

            # ---- approach dir ----
            d_to_center = O_xy - start[:2]
            if _norm2(d_to_center) < 1e-6:
                d_to_center = np.array([1.0, 0.0], dtype=np.float32)
            approach_dir = _normalize2(d_to_center)

            # seg0: yaw at start
            yaw_dirs_start = sample_yaw_only_dirs(init_dir, approach_dir, step_deg=yaw_step_deg)
            seg0 = [start.copy() for _ in yaw_dirs_start]

            # entry point
            E_enter_xy = O_xy - approach_dir * R
            E_enter_low = np.array([E_enter_xy[0], E_enter_xy[1], float(end_height_rel)], dtype=np.float32)

            # seg1: slanted descend
            seg1 = sample_line_points(start, E_enter_low, step=step, include_endpoint=True, skip_first=True)

            # seg2: yaw at entry (approach -> tangent)
            radial = _normalize2(E_enter_xy - O_xy)
            tangent_dir = _left_normal(radial) if orbit_ccw else _right_normal(radial)
            tangent_dir = _normalize2(tangent_dir)

            yaw_dirs_in = sample_yaw_only_dirs(approach_dir, tangent_dir, step_deg=yaw_step_deg)
            seg2 = [E_enter_low.copy() for _ in yaw_dirs_in]

            # seg3: orbit once
            a0 = math.atan2(E_enter_xy[1] - O_xy[1], E_enter_xy[0] - O_xy[0])
            sign = 1.0 if orbit_ccw else -1.0
            orbit_len = 2.0 * math.pi * R
            n_orb = max(1, int(math.ceil(orbit_len / max(step, 1e-6))))

            seg3 = []
            for i in range(1, n_orb + 1):
                a = a0 + sign * 2.0 * math.pi * (i / float(n_orb))
                x = O_xy[0] + R * math.cos(a)
                y = O_xy[1] + R * math.sin(a)
                seg3.append(np.array([x, y, float(end_height_rel)], dtype=np.float32))

            orbit_end_pt = seg3[-1] if len(seg3) else E_enter_low

            # seg4: yaw at orbit end (tangent->face start)
            if len(seg3) >= 2:
                dxy_orb_end = seg3[-1][:2] - seg3[-2][:2]
                dxy_orb_end = tangent_dir if _norm2(dxy_orb_end) < 1e-6 else _normalize2(dxy_orb_end)
            else:
                dxy_orb_end = tangent_dir

            d_to_start = start[:2] - orbit_end_pt[:2]
            dir_to_start = dxy_orb_end if _norm2(d_to_start) < 1e-6 else _normalize2(d_to_start)

            yaw_dirs_return = sample_yaw_only_dirs(dxy_orb_end, dir_to_start, step_deg=yaw_step_deg)
            seg4 = [orbit_end_pt.copy() for _ in yaw_dirs_return]

            # seg5: slanted return ascent
            seg5 = sample_line_points(orbit_end_pt, start, step=step, include_endpoint=True, skip_first=True)

            traj_pts = seg0 + seg1 + seg2 + seg3 + seg4 + seg5

            if debug_print:
                print(
                    f"[DBG] label={label} fixed_end_h_rel={end_height_rel} orbit_ccw={orbit_ccw} "
                    f"n_orb={n_orb} step={step} R={R} "
                    f"yaw0={len(yaw_dirs_start)} yaw_in={len(yaw_dirs_in)} yaw_ret={len(yaw_dirs_return)}"
                )

            # ---- segment index ranges (traj_pts index space) ----
            len0, len1, len2, len3, len4, len5 = map(len, [seg0, seg1, seg2, seg3, seg4, seg5])
            idx0_s, idx0_e = 0, len0
            idx1_s, idx1_e = idx0_e, idx0_e + len1
            idx2_s, idx2_e = idx1_e, idx1_e + len2
            idx3_s, idx3_e = idx2_e, idx2_e + len3
            idx4_s, idx4_e = idx3_e, idx3_e + len4
            idx5_s, idx5_e = idx4_e, idx4_e + len5

            def _pid_range(seg_start_idx, seg_end_idx_excl, base_pose_id):
                if seg_end_idx_excl <= seg_start_idx:
                    return base_pose_id, base_pose_id - 1
                s = base_pose_id + seg_start_idx
                e = base_pose_id + (seg_end_idx_excl - 1)
                return s, e

            turn0_start, turn0_end = _pid_range(idx0_s, idx0_e, traj_pose_start_id)
            descend_start, descend_end = _pid_range(idx1_s, idx1_e, traj_pose_start_id)
            turn1_start, turn1_end = _pid_range(idx2_s, idx2_e, traj_pose_start_id)
            orbit_start, orbit_end_id = _pid_range(idx3_s, idx3_e, traj_pose_start_id)

            # return split
            ret_turn_start, ret_turn_end = _pid_range(idx4_s, idx4_e, traj_pose_start_id)
            ret_fly_start, ret_fly_end = _pid_range(idx5_s, idx5_e, traj_pose_start_id)

            # ---- orientation ----
            yaw0_cursor = 0
            yaw1_cursor = 0
            yaw2_cursor = 0
            last_dir = init_dir.copy()

            for i, p in enumerate(traj_pts):
                p = np.asarray(p, dtype=np.float32)

                in_yaw0 = (idx0_s <= i < idx0_e)
                in_yaw1 = (idx2_s <= i < idx2_e)
                in_yaw2 = (idx4_s <= i < idx4_e)

                if in_yaw0 and len0 > 0:
                    dxy = yaw_dirs_start[yaw0_cursor] if yaw0_cursor < len(yaw_dirs_start) else approach_dir
                    yaw0_cursor += 1
                    last_dir = dxy
                elif in_yaw1 and len2 > 0:
                    dxy = yaw_dirs_in[yaw1_cursor] if yaw1_cursor < len(yaw_dirs_in) else tangent_dir
                    yaw1_cursor += 1
                    last_dir = dxy
                elif in_yaw2 and len4 > 0:
                    dxy = yaw_dirs_return[yaw2_cursor] if yaw2_cursor < len(yaw_dirs_return) else dir_to_start
                    yaw2_cursor += 1
                    last_dir = dxy
                else:
                    if len(traj_pts) >= 2:
                        if i < len(traj_pts) - 1:
                            dxy = np.asarray(traj_pts[i + 1][:2], dtype=np.float32) - p[:2]
                        else:
                            dxy = p[:2] - np.asarray(traj_pts[i - 1][:2], dtype=np.float32)
                    else:
                        dxy = last_dir

                    if _norm2(dxy) < 1e-6:
                        dxy = last_dir
                    else:
                        dxy = _normalize2(dxy)
                        last_dir = dxy

                R_w2c = build_nadir_R_from_dir(dxy)
                omega_rad, phi_rad, kappa_rad = R_to_opk(R_w2c)

                poses.append(
                    {
                        "id": pose_id,
                        "C_rel": p.astype(np.float64),
                        "omega": math.degrees(omega_rad),
                        "phi": math.degrees(phi_rad),
                        "kappa": math.degrees(kappa_rad),
                    }
                )
                pose_id += 1

            traj_pose_end_id = pose_id - 1
            if traj_pose_end_id >= traj_pose_start_id:
                traj_infos.append(
                    {
                        "traj_id": traj_global_id,
                        "label": label,
                        "label_no_pos": label_no_pos,
                        "pose_id_start": traj_pose_start_id,
                        "pose_id_end": traj_pose_end_id,
                        "end_height_rel": end_height_rel,
                        "orbit_radius": R,

                        # subtask boundaries
                        "turn0_start": turn0_start,
                        "turn0_end": turn0_end,
                        "descend_start": descend_start,
                        "descend_end": descend_end,
                        "turn1_start": turn1_start,
                        "turn1_end": turn1_end,
                        "orbit_start": orbit_start,
                        "orbit_end": orbit_end_id,

                        # return split
                        "return_turn_start": ret_turn_start,
                        "return_turn_end": ret_turn_end,
                        "return_fly_start": ret_fly_start,
                        "return_fly_end": ret_fly_end,
                    }
                )
                traj_global_id += 1

    print(f"[INFO] 总共生成 {len(poses)} 个轨迹点（pose），共 {len(traj_infos)} 条轨迹")
    return poses, traj_infos


# =========================================================
# 7. 写 PLY 可视化
# =========================================================
def write_camera_points_ply(cameras, ply_path, color=(255, 0, 0), use_offset=True):
    global DJI_OFFSET
    if use_offset and DJI_OFFSET is None:
        raise RuntimeError("DJI_OFFSET 还未初始化，请先调用主函数或手动设置 DJI_OFFSET")

    os.makedirs(os.path.dirname(ply_path) or ".", exist_ok=True)

    verts = []
    for c in cameras:
        C_world = c["C_world"]
        C = C_world.astype(np.float64) - DJI_OFFSET if use_offset else C_world
        verts.append(C)

    verts = np.array(verts, dtype=np.float32)
    r, g, b = color
    N = verts.shape[0]

    with open(ply_path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(N):
            x, y, z = verts[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


def write_random_points_ply(poses, ply_path, color=(0, 255, 0), use_offset=True):
    global DJI_OFFSET
    if use_offset and DJI_OFFSET is None:
        raise RuntimeError("DJI_OFFSET 还未初始化，请先调用主函数或手动设置 DJI_OFFSET")

    os.makedirs(os.path.dirname(ply_path) or ".", exist_ok=True)

    verts = []
    for p in poses:
        C_world = p["C_world"]
        C = C_world.astype(np.float64) - DJI_OFFSET if use_offset else C_world
        verts.append(C)

    verts = np.array(verts, dtype=np.float32)
    r, g, b = color
    N = verts.shape[0]

    with open(ply_path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(N):
            x, y, z = verts[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


# =========================================================
# 9. 输出路径逻辑：统一输出到 /traj_data/{env_id}/
# =========================================================
def resolve_outputs(
    out_dir: Path,
    out_txt: str,
    out_ply_xml: str,
    out_ply_random: str,
    out_instr: str,
    out_meta: str,
    out_subtask: str,
):
    out_dir = Path(out_dir)
    return (
        out_dir / out_txt,
        out_dir / out_ply_xml,
        out_dir / out_ply_random,
        out_dir / out_instr,
        out_dir / out_meta,
        out_dir / out_subtask,
    )


# =========================================================
# 10. 命令行入口
# =========================================================
def main():
    parser = ArgumentParser(
        description=(
            "Generate DJI 3DGS orbit trajectories over landmarks "
            "(random start heading -> turn -> slanted descend -> turn into orbit -> full orbit "
            "-> turn to face starting point -> slanted return ascent). "
            "Paths and height defaults are auto-resolved from --env_id. "
            "Orbit HEIGHT is fixed to min(end_height_candidates); orbit RADIUS is randomly chosen per trajectory. "
            "No global->local conversion."
        )
    )

    # ----- env_id (唯一主控参数) -----
    parser.add_argument(
        "--env_id",
        type=str,
        required=True,
        choices=list(ENV_CONFIGS.keys()),
        help="场景 ID，例如 1_office / 2_city / 3_road / 4_lake",
    )

    # ----- inputs（默认由 env_id 自动生成；如有需要可手动覆盖） -----
    parser.add_argument(
        "--xml_path",
        type=str,
        default=None,
        help="可选覆盖：BlocksExchange XML 路径；默认由 env_id 推导",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default=None,
        help="可选覆盖：metadata.xml 路径；默认由 env_id 推导",
    )
    parser.add_argument(
        "--location_path",
        type=str,
        default=None,
        help="可选覆盖：landmark_merged.txt 路径；默认由 env_id 推导",
    )

    # ----- outputs（统一放到 /traj_data/{env_id}/） -----
    parser.add_argument("--out_txt", type=str, default="traj_random.txt")
    parser.add_argument("--out_ply_xml", type=str, default="cameras_xml.ply")
    parser.add_argument("--out_ply_random", type=str, default="cameras_random.ply")
    parser.add_argument("--out_instr", type=str, default="instruction.txt", help="每条轨迹的指令与 id 范围")
    parser.add_argument("--out_meta", type=str, default="traj_meta.txt", help="每条轨迹对应 label 和 pose id 范围")
    parser.add_argument("--out_subtask", type=str, default="subtask.txt", help="每条轨迹的 subtask 标签与 pose id 范围")

    # ----- hyperparams -----
    parser.add_argument("--traj_per_loc", type=int, default=9, help="每个 location 生成的轨迹数量")
    parser.add_argument("--start_radius", type=float, default=60.0, help="起点相对地标的最大水平半径（m）")

    # 以下 3 个默认由 env_id 自动决定；如传入则覆盖 env 默认值
    parser.add_argument("--start_h_min", type=float, default=None, help="起点高度范围下界（m，对应 C_rel.z）")
    parser.add_argument("--start_h_max", type=float, default=None, help="起点高度范围上界（m，对应 C_rel.z）")
    parser.add_argument(
        "--end_height_candidates",
        type=str,
        default=None,
        help="轨道高度候选（相对坐标，逗号分隔）；实际高度固定取最小值。默认由 env_id 决定，例如 '-20,-10,0'",
    )

    parser.add_argument("--sample_step", type=float, default=1.0, help="采样间隔（米）")
    parser.add_argument("--yaw_step_deg", type=float, default=5.0, help="原地转向时每隔多少度采一个 yaw")

    # NEW: orbit radius candidates (choose one per trajectory)
    parser.add_argument(
        "--orbit_radius_candidates",
        type=str,
        default="10, 20, 30",
        help="环绕半径候选（逗号分隔）；每条轨迹随机选一个。默认 '20,40,60'",
    )
    # Deprecated fixed radius
    parser.add_argument(
        "--orbit_radius",
        type=float,
        default=None,
        help="(Deprecated) 固定环绕半径；若设置则覆盖 orbit_radius_candidates",
    )

    parser.add_argument("--orbit_ccw", action="store_true", help="绕圈逆时针（不加该参数则顺时针）")
    parser.add_argument("--debug", action="store_true", help="打印调试信息")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（默认 None，不固定）")

    args = parser.parse_args()

    # ----- resolve env defaults -----
    env_cfg = resolve_env_paths_and_defaults(args.env_id)

    # 输入路径：优先使用命令行显式传入，否则用 env_id 自动推导
    if args.xml_path is None:
        args.xml_path = str(env_cfg["xml_path"])
    if args.metadata_path is None:
        args.metadata_path = str(env_cfg["metadata_path"])
    if args.location_path is None:
        args.location_path = str(env_cfg["location_path"])

    # 高度参数：优先命令行覆盖，否则使用 env 默认
    if args.start_h_min is None:
        args.start_h_min = float(env_cfg["start_h_min"])
    if args.start_h_max is None:
        args.start_h_max = float(env_cfg["start_h_max"])
    if args.end_height_candidates is None:
        args.end_height_candidates = ",".join([str(x) for x in env_cfg["end_height_candidates"]])

    # Output path: HUGE_DATA_TRAJ_ROOT/task_orbit/{env_id}/
    out_txt, out_ply_xml, out_ply_random, out_instr, out_meta, out_subtask = resolve_outputs(
        out_dir=env_cfg["out_dir"],
        out_txt=args.out_txt,
        out_ply_xml=args.out_ply_xml,
        out_ply_random=args.out_ply_random,
        out_instr=args.out_instr,
        out_meta=args.out_meta,
        out_subtask=args.out_subtask,
    )

    # parse height candidates
    end_height_candidates = [float(x.strip()) for x in args.end_height_candidates.split(",") if x.strip() != ""]
    if len(end_height_candidates) == 0:
        raise ValueError("--end_height_candidates 解析后为空，请检查输入格式，例如 '-20,-10,0'")
    fixed_end_height_rel = float(min(end_height_candidates))

    # parse radius candidates
    if args.orbit_radius is not None:
        orbit_radius_candidates = [float(args.orbit_radius)]
    else:
        orbit_radius_candidates = [float(x.strip()) for x in args.orbit_radius_candidates.split(",") if x.strip() != ""]
        if len(orbit_radius_candidates) == 0:
            raise ValueError("--orbit_radius_candidates 解析后为空，请检查输入格式，例如 '20,40,60'")

    print(f"[INFO] env_id = {args.env_id}")
    print(f"[INFO] data_path = {env_cfg['data_path']}")
    print(f"[INFO] out_dir = {env_cfg['out_dir']}")
    print(f"[INFO] xml_path = {args.xml_path}")
    print(f"[INFO] metadata_path = {args.metadata_path}")
    print(f"[INFO] location_path = {args.location_path}")
    print(
        f"[INFO] heights: start_h=[{args.start_h_min}, {args.start_h_max}], "
        f"end_height_candidates={end_height_candidates} (FIXED -> {fixed_end_height_rel})"
    )
    print(f"[INFO] orbit_radius_candidates = {orbit_radius_candidates} (choose one per trajectory)")
    if args.seed is not None:
        print(f"[INFO] seed = {args.seed}")

    for p in [out_txt, out_ply_xml, out_ply_random, out_instr, out_meta, out_subtask]:
        os.makedirs(str(p.parent) if str(p.parent) else ".", exist_ok=True)

    # ----- run -----
    global DJI_OFFSET
    DJI_OFFSET = parse_dji_offset_from_metadata(args.metadata_path)

    intr = parse_intrinsics_from_xml(args.xml_path)
    cameras_xml = parse_all_cameras_from_xml(args.xml_path)
    print(f"[INFO] 从 XML 中读取到 {len(cameras_xml)} 个原始相机")

    locations = parse_locations_from_txt(args.location_path)

    rng = np.random.default_rng(args.seed)
    poses, traj_infos = generate_poses_from_locations(
        locations,
        num_traj_per_loc=args.traj_per_loc,
        start_radius=args.start_radius,
        start_h_min=args.start_h_min,
        start_h_max=args.start_h_max,
        end_height=end_height_candidates[0],   # deprecated fallback param
        end_height_candidates=end_height_candidates,
        step=args.sample_step,
        yaw_step_deg=args.yaw_step_deg,
        orbit_radius=args.orbit_radius,        # deprecated
        orbit_radius_candidates=orbit_radius_candidates,
        orbit_ccw=args.orbit_ccw,
        rng=rng,
        debug_print=args.debug,
    )

    # add C_world
    for p in poses:
        p["C_world"] = p["C_rel"] + DJI_OFFSET

    # ----- write traj_random.txt -----
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(
            "# id  x_rel  y_rel  z_rel  omega_deg  phi_deg  kappa_deg\n"
            "# 注意: x_rel,y_rel,z_rel = C_world - DJI_OFFSET\n"
            f"# env_id = {args.env_id}\n"
            f"# DJI_OFFSET = {DJI_OFFSET.tolist()}\n"
        )
        f.write(
            "# intrinsics: width height fx fy cx cy\n"
            f"# {intr['width']} {intr['height']} "
            f"{intr['fx']:.6f} {intr['fy']:.6f} {intr['cx']:.6f} {intr['cy']:.6f}\n"
        )
        f.write(
            f"# traj_per_loc={args.traj_per_loc}, start_radius={args.start_radius}, "
            f"start_h=[{args.start_h_min},{args.start_h_max}], end_height_candidates={end_height_candidates} (fixed-> {fixed_end_height_rel}), "
            f"sample_step={args.sample_step}, orbit_radius_candidates={orbit_radius_candidates}, orbit_ccw={args.orbit_ccw}, "
            f"yaw_step_deg={args.yaw_step_deg}, seed={args.seed}\n"
            "# 每一行一个 pose，id 为全局递增\n"
        )
        for p in poses:
            c = p["C_rel"]
            f.write(
                f"{p['id']} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f} "
                f"{p['omega']:.6f} {p['phi']:.6f} {p['kappa']:.6f}\n"
            )
    print(f"[INFO] 轨迹 pose 已写入 txt: {out_txt}")

    # ----- write instruction.txt -----
    # real_h = end_rel + 80
    with open(out_instr, "w", encoding="utf-8") as f:
        f.write("# traj_id  pose_id_start  pose_id_end  instruction\n")
        for info in traj_infos:
            real_h = int(round(float(info["end_height_rel"]) + 80.0))
            label = info["label"]
            R = int(round(float(info["orbit_radius"])))
            instr = f"Orbit the {label} once at {real_h} meters with a {R}-meter radius."
            f.write(f"{info['traj_id']} {info['pose_id_start']} {info['pose_id_end']} {instr}\n")
    print(f"[INFO] 轨迹 instruction 已写入: {out_instr}")

    # ----- write subtask.txt -----
    with open(out_subtask, "w", encoding="utf-8") as f:
        f.write("# traj_id  subtask_id  pose_id_start  pose_id_end  subtask\n")
        for info in traj_infos:
            traj_id = info["traj_id"]
            label = info["label"]
            label_no_pos = info["label_no_pos"]
            real_h = int(round(float(info["end_height_rel"]) + 80.0))
            R = int(round(float(info["orbit_radius"])))

            # 0) turn to face landmark (use label)
            if info["turn0_end"] >= info["turn0_start"]:
                sub = f"Turn to face the {label}."
                f.write(f"{traj_id} 0 {info['turn0_start']} {info['turn0_end']} {sub}\n")

            # 1) descend while arriving (use label_no_pos)
            if info["descend_end"] >= info["descend_start"]:
                sub = f"Fly while descending to {real_h} meters above the {label_no_pos}."
                f.write(f"{traj_id} 1 {info['descend_start']} {info['descend_end']} {sub}\n")

            # 2) turn to enter orbit (use label_no_pos) + radius
            if info["turn1_end"] >= info["turn1_start"]:
                sub = f"Turn to enter a {R}-meter-radius orbit around the {label_no_pos}."
                f.write(f"{traj_id} 2 {info['turn1_start']} {info['turn1_end']} {sub}\n")

            # 3) orbit once (use label_no_pos) + radius
            if info["orbit_end"] >= info["orbit_start"]:
                sub = f"Orbit the {label_no_pos} once with a {R}-meter radius."
                f.write(f"{traj_id} 3 {info['orbit_start']} {info['orbit_end']} {sub}\n")

            # 4) turn to face starting point
            if info["return_turn_end"] >= info["return_turn_start"]:
                sub = "Turn to face the starting point."
                f.write(f"{traj_id} 4 {info['return_turn_start']} {info['return_turn_end']} {sub}\n")

            # 5) fly while ascending back
            if info["return_fly_end"] >= info["return_fly_start"]:
                sub = "Fly while ascending back to the starting point."
                f.write(f"{traj_id} 5 {info['return_fly_start']} {info['return_fly_end']} {sub}\n")

    print(f"[INFO] subtask 标签已写入: {out_subtask}")

    # ----- write traj_meta.txt -----
    # 用 TSV，避免 label / label_no_pos 中有空格时解析困难
    with open(out_meta, "w", encoding="utf-8") as f:
        f.write("# traj_id\tlabel\tlabel_no_pos\tpose_id_start\tpose_id_end\n")
        for info in traj_infos:
            f.write(
                f"{info['traj_id']}\t{info['label']}\t{info['label_no_pos']}\t"
                f"{info['pose_id_start']}\t{info['pose_id_end']}\n"
            )
    print(f"[INFO] 轨迹 meta 已写入: {out_meta}")

    # ----- write PLYs -----
    write_camera_points_ply(cameras_xml, str(out_ply_xml), color=(255, 0, 0), use_offset=True)
    write_random_points_ply(poses, str(out_ply_random), color=(0, 255, 0), use_offset=True)
    print(f"[INFO] PLY 已写入: {out_ply_xml}, {out_ply_random}")

    print("[INFO] 已禁用 global->local 转换：仅输出相对坐标轨迹，不生成 traj_random_local.txt")


if __name__ == "__main__":
    main()

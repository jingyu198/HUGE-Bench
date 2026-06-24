# -*- coding: utf-8 -*-
"""
Merged script (landmark / location version):

1) Generate DJI 3DGS trajectories over landmarks from locations.txt
IMPORTANT:
    VLN 轨迹不做 global -> local 转换！
    只输出相对坐标系下的轨迹（C_rel = C_world - DJI_OFFSET），不生成 traj_random_local.txt。

CHANGE (minimal):
    原本终点高度固定 end_height=-20（对应现实 60m）。
    现在改为：每条轨迹随机从相对高度候选中选一个，并同步修改 instruction.txt 的高度。

NEW (requested):
    在保持原有 instruction.txt 不变的基础上，
    为每条轨迹新增 subtask 标签，保存到 subtask.txt：
      1)（需要时）调整机头转向（turn）
      2) 飞到目标(label_no_pos)上方（fly_to_above）
      3) 下降到目标(label_no_pos)上方 real_h 米（descend_to_height）
    subtask.txt 以 pose id 范围标注每个子任务对应的片段。

NEW (automation by env_id):
    只需要控制 --env_id，就会自动映射：
      - data_path
      - xml_path / metadata_path / location_path
      - output directory from HUGE_DATA_TRAJ_ROOT/task_0/{env_id}
      - start_h_min / start_h_max / end_height_candidates

LOCATION FILE FORMAT:
- 推荐 TSV（\t 分隔），支持 label 中有空格：
  x<TAB>y<TAB>label<TAB>label_no_pos
- 也兼容旧格式：
  x y label
  （没有 label_no_pos 时，会自动用 label 代替）
"""

import os
import math
import numpy as np
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from pathlib import Path

# ==== offset 不再手动写死，而是从 metadata.xml 里读取 ====
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
OUT_ROOT = Path(os.environ.get("HUGE_DATA_TRAJ_ROOT", "./data_traj")).expanduser() / "task_0"


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
        "location_path": data_path / "location_gen" / "landmark_merged.txt",
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
    """
    读取第一个 Photogroup 的内参：
      - 图像尺寸: width, height
      - 焦距: fx, fy
      - 主点: cx, cy
    """
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

    return {
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
    }


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
# 2.5 从 location.txt / landmark_merged.txt 解析地标平面坐标（相对坐标）
#     推荐 TSV:
#       x<TAB>y<TAB>label<TAB>label_no_pos
#     兼容旧格式:
#       x y label
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
#    对应: R = Rz(kappa) @ Ry(phi) @ Rx(omega)
# =========================================================
def R_to_opk(R):
    """
    输入: R_w2c (3x3 矩阵)
    输出: (omega, phi, kappa) in radians
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


# =========================================================
# 4. 构造俯视姿态：
#    - 光轴垂直向下 (z_c = (0,0,-1))
#    - “飞行方向”在画面中朝上（对应 -y_c）
# =========================================================
def build_nadir_R_from_dir(dir_xy):
    """
    dir_xy: 二维方向向量，表示“飞行方向”（地面上的投影）。
    返回: R_w2c, 行向量为相机坐标系在世界坐标下的 x/y/z 轴
    """
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
    """最小有向角度差（[-pi, pi]）"""
    return (yaw_to - yaw_from + math.pi) % (2.0 * math.pi) - math.pi


def sample_yaw_only_dirs(dir_from, dir_to, step_deg=10.0):
    """
    在 dir_from -> dir_to 之间，只在偏航角上插值，返回一系列中间方向向量。
    返回: List[np.array([2], float32)]
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
    """
    在 [p0, p1] 之间以给定步长（米）采样直线段。
    返回: List[np.array([3], float32)]
    """
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
# 6. 根据 location.txt 生成若干条轨迹（每条按 step 采样）
# =========================================================
def generate_poses_from_locations(
    locations,
    num_traj_per_loc=3,
    start_radius=50.0,
    start_h_min=80.0,
    start_h_max=100.0,
    end_height_candidates=(-20.0, -10.0, 0.0),
    step=1.0,
    yaw_step_deg=10.0,
    rng=None,
):
    """
    返回:
      poses: List[dict]
      traj_infos: List[dict]
        每条轨迹会带 end_height_rel 供 instruction 使用
        额外带 subtask 边界信息（turn / fly / descend）
    """
    if rng is None:
        rng = np.random.default_rng()

    poses = []
    traj_infos = []
    pose_id = 0
    traj_global_id = 0

    for loc in locations:
        if traj_global_id % 100 == 0:
            print(traj_global_id)

        loc_xy_rel = np.asarray(loc["xy"], dtype=np.float32)
        loc_name = loc["name"]  # 原始 label（可能含方位词）
        loc_name_clean = loc.get("name_clean", loc_name)  # 去方位词 label

        for _ in range(num_traj_per_loc):
            # 每条轨迹随机选择终点高度（相对坐标）
            end_height_rel = float(rng.choice(end_height_candidates))

            # 1) 随机起点（圆盘内均匀采样 + 高度范围）
            u = rng.random()
            r = start_radius * math.sqrt(u)
            theta = 2.0 * math.pi * rng.random()
            start_xy = loc_xy_rel + np.array([math.cos(theta), math.sin(theta)], dtype=np.float32) * r
            z_start = float(rng.uniform(start_h_min, start_h_max))
            start = np.array([start_xy[0], start_xy[1], z_start], dtype=np.float32)

            # 1.5) 随机一个初始机头方向
            theta0 = 2.0 * math.pi * rng.random()
            init_dir_xy = np.array([math.cos(theta0), math.sin(theta0)], dtype=np.float32)

            # 2) 地标正上方同高点 + 终点
            mid = np.array([loc_xy_rel[0], loc_xy_rel[1], z_start], dtype=np.float32)
            end = np.array([loc_xy_rel[0], loc_xy_rel[1], end_height_rel], dtype=np.float32)

            # 目标水平方向：从起点指向地标
            dir_target_xy = mid[:2] - start[:2]
            if np.linalg.norm(dir_target_xy) < 1e-6:
                dir_target_xy = init_dir_xy.copy()
            else:
                dir_target_xy = dir_target_xy / (np.linalg.norm(dir_target_xy) + 1e-8)

            # 3) 轨迹采样：先水平段 A，再垂直段 B
            pts_A = sample_line_points(start, mid, step=step, include_endpoint=True, skip_first=True)
            pts_B = sample_line_points(mid, end, step=step, include_endpoint=True, skip_first=True)

            traj_pose_start_id = pose_id

            # 3.5) 起点原地旋转：初始方向 -> 目标方向
            yaw_dirs = sample_yaw_only_dirs(init_dir_xy, dir_target_xy, step_deg=yaw_step_deg)

            turn_start_id = pose_id
            if len(yaw_dirs) > 0:
                for dir_xy in yaw_dirs:
                    R_w2c = build_nadir_R_from_dir(dir_xy)
                    omega_rad, phi_rad, kappa_rad = R_to_opk(R_w2c)
                    poses.append(
                        {
                            "id": pose_id,
                            "C_rel": start.astype(np.float64),
                            "omega": math.degrees(omega_rad),
                            "phi": math.degrees(phi_rad),
                            "kappa": math.degrees(kappa_rad),
                        }
                    )
                    pose_id += 1
                turn_end_id = pose_id - 1
            else:
                turn_end_id = turn_start_id - 1  # 无 turn 段

            # 4) 水平飞到目标上方
            flyA_start_id = pose_id
            for p in pts_A:
                R_w2c = build_nadir_R_from_dir(dir_target_xy)
                omega_rad, phi_rad, kappa_rad = R_to_opk(R_w2c)
                poses.append(
                    {
                        "id": pose_id,
                        "C_rel": np.asarray(p, dtype=np.float64),
                        "omega": math.degrees(omega_rad),
                        "phi": math.degrees(phi_rad),
                        "kappa": math.degrees(kappa_rad),
                    }
                )
                pose_id += 1
            flyA_end_id = pose_id - 1

            # 5) 垂直下降
            descend_start_id = pose_id
            for p in pts_B:
                R_w2c = build_nadir_R_from_dir(dir_target_xy)
                omega_rad, phi_rad, kappa_rad = R_to_opk(R_w2c)
                poses.append(
                    {
                        "id": pose_id,
                        "C_rel": np.asarray(p, dtype=np.float64),
                        "omega": math.degrees(omega_rad),
                        "phi": math.degrees(phi_rad),
                        "kappa": math.degrees(kappa_rad),
                    }
                )
                pose_id += 1
            descend_end_id = pose_id - 1

            traj_pose_end_id = pose_id - 1
            if traj_pose_end_id >= traj_pose_start_id:
                traj_infos.append(
                    {
                        "traj_id": traj_global_id,
                        "loc_name": loc_name,
                        "loc_name_clean": loc_name_clean,
                        "pose_id_start": traj_pose_start_id,
                        "pose_id_end": traj_pose_end_id,
                        "end_height_rel": end_height_rel,

                        # subtask 边界
                        "turn_start": turn_start_id,
                        "turn_end": turn_end_id,
                        "flyA_start": flyA_start_id,
                        "flyA_end": flyA_end_id,
                        "descend_start": descend_start_id,
                        "descend_end": descend_end_id,
                    }
                )
                traj_global_id += 1

    print(f"[INFO] 总共生成 {len(poses)} 个轨迹点（pose），共 {len(traj_infos)} 条轨迹")
    return poses, traj_infos


# =========================================================
# 7. 写 PLY 可视化
# =========================================================
def write_camera_points_ply(cameras, ply_path, color=(255, 0, 0), use_offset=True):
    """
    cameras: parse_all_cameras_from_xml() 的结果
    use_offset=True 表示写入 (C_world - DJI_OFFSET)
    """
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


def write_random_points_ply(poses, ply_path, color=(0, 255, 0), use_offset=True):
    """
    poses: generate_poses_from_locations() 的结果（调用前需补齐 C_world）
    use_offset=True 表示写入 (C_world - DJI_OFFSET)
    """
    global DJI_OFFSET
    if use_offset and DJI_OFFSET is None:
        raise RuntimeError("DJI_OFFSET 还未初始化，请先调用主函数或手动设置 DJI_OFFSET")

    verts = []
    for p in poses:
        C_world = p["C_world"]
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
# 10. 主入口
# =========================================================
def main():
    parser = ArgumentParser(
        description=(
            "Generate DJI 3DGS camera trajectories over landmarks from BlocksExchange XML + locations.txt. "
            "Paths and height defaults are auto-resolved from --env_id. "
            "No global->local conversion."
        )
    )

    # -------- env switch (唯一主控参数) --------
    parser.add_argument(
        "--env_id",
        type=str,
        required=True,
        choices=list(ENV_CONFIGS.keys()),
        help="场景 ID，例如 1_office / 2_city / 3_road / 4_lake",
    )

    # -------- inputs (默认由 env_id 自动生成；如有需要可手动覆盖) --------
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

    # -------- outputs (统一放到 /traj_data/{env_id}/) --------
    parser.add_argument("--out_txt", type=str, default="traj_random.txt")
    parser.add_argument("--out_ply_xml", type=str, default="cameras_xml.ply")
    parser.add_argument("--out_ply_random", type=str, default="cameras_random.ply")
    parser.add_argument("--out_instr", type=str, default="instruction.txt", help="每条轨迹的指令与 id 范围")
    parser.add_argument("--out_meta", type=str, default="traj_meta.txt", help="traj_id 对应 loc_name 和 pose id 范围")
    parser.add_argument("--out_subtask", type=str, default="subtask.txt", help="每条轨迹的 subtask 标签与 pose id 范围")

    # -------- hyperparams --------
    parser.add_argument("--traj_per_loc", type=int, default=15, help="每个 location 生成的轨迹数量")
    parser.add_argument("--start_radius", type=float, default=60.0, help="起点相对地标的水平半径（m）")

    # 这三个参数默认由 env_id 自动决定；如传入则覆盖 env 默认值
    parser.add_argument("--start_h_min", type=float, default=None, help="起点高度范围下界（m，对应 C_rel.z）")
    parser.add_argument("--start_h_max", type=float, default=None, help="起点高度范围上界（m，对应 C_rel.z）")
    parser.add_argument(
        "--end_height_candidates",
        type=str,
        default=None,
        help="终点高度候选（相对坐标，逗号分隔）；默认由 env_id 决定，例如 '-20,-10,0'",
    )

    parser.add_argument("--sample_step", type=float, default=1.0, help="轨迹采样间隔（米）")
    parser.add_argument("--yaw_step_deg", type=float, default=5.0, help="原地转向时的偏航插值步长（度）")

    # 可选：固定随机种子，便于复现
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 None，表示不固定）")

    args = parser.parse_args()

    # ---------- resolve env defaults ----------
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

    # Output path: HUGE_DATA_TRAJ_ROOT/task_0/{env_id}/
    out_txt, out_ply_xml, out_ply_random, out_instr, out_meta, out_subtask = resolve_outputs(
        out_dir=env_cfg["out_dir"],
        out_txt=args.out_txt,
        out_ply_xml=args.out_ply_xml,
        out_ply_random=args.out_ply_random,
        out_instr=args.out_instr,
        out_meta=args.out_meta,
        out_subtask=args.out_subtask,
    )

    print(f"[INFO] env_id = {args.env_id}")
    print(f"[INFO] data_path = {env_cfg['data_path']}")
    print(f"[INFO] out_dir = {env_cfg['out_dir']}")
    print(f"[INFO] xml_path = {args.xml_path}")
    print(f"[INFO] metadata_path = {args.metadata_path}")
    print(f"[INFO] location_path = {args.location_path}")
    print(
        f"[INFO] heights: start_h=[{args.start_h_min}, {args.start_h_max}], "
        f"end_height_candidates={args.end_height_candidates}"
    )
    if args.seed is not None:
        print(f"[INFO] seed = {args.seed}")

    # Ensure output folder exists
    for p in [out_txt, out_ply_xml, out_ply_random, out_instr, out_meta, out_subtask]:
        os.makedirs(str(p.parent) if str(p.parent) else ".", exist_ok=True)

    # ---------- generate ----------
    global DJI_OFFSET
    DJI_OFFSET = parse_dji_offset_from_metadata(args.metadata_path)

    intr = parse_intrinsics_from_xml(args.xml_path)
    cameras_xml = parse_all_cameras_from_xml(args.xml_path)
    print(f"[INFO] 从 XML 中读取到 {len(cameras_xml)} 个原始相机")

    locations = parse_locations_from_txt(args.location_path)

    # 解析候选终点高度
    end_height_candidates = [float(x.strip()) for x in args.end_height_candidates.split(",") if x.strip() != ""]
    if len(end_height_candidates) == 0:
        raise ValueError("--end_height_candidates 解析后为空，请检查输入格式，例如 '-20,-10,0'")

    rng = np.random.default_rng(args.seed)
    poses, traj_infos = generate_poses_from_locations(
        locations,
        num_traj_per_loc=args.traj_per_loc,
        start_radius=args.start_radius,
        start_h_min=args.start_h_min,
        start_h_max=args.start_h_max,
        end_height_candidates=end_height_candidates,
        step=args.sample_step,
        yaw_step_deg=args.yaw_step_deg,
        rng=rng,
    )

    # 补上 C_world（= C_rel + DJI_OFFSET）
    for p in poses:
        C_rel = p["C_rel"]
        p["C_world"] = C_rel + DJI_OFFSET

    # ---------- write traj_random.txt ----------
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("# id  x_rel  y_rel  z_rel  omega_deg  phi_deg  kappa_deg\n")
        f.write("# 注意: x_rel,y_rel,z_rel = C_world - DJI_OFFSET\n")
        f.write(f"# env_id = {args.env_id}\n")
        f.write(f"# DJI_OFFSET = {DJI_OFFSET.tolist()}\n")
        f.write("# intrinsics: width height fx fy cx cy\n")
        f.write(
            f"# {intr['width']} {intr['height']} "
            f"{intr['fx']:.6f} {intr['fy']:.6f} {intr['cx']:.6f} {intr['cy']:.6f}\n"
        )
        f.write(
            f"# traj_per_loc={args.traj_per_loc}, start_radius={args.start_radius}, "
            f"start_h=[{args.start_h_min},{args.start_h_max}], end_height_candidates={end_height_candidates}, "
            f"sample_step={args.sample_step}, yaw_step_deg={args.yaw_step_deg}, seed={args.seed}\n"
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
    # 现实高度映射：real_h = end_rel + 80
    # instruction.txt 保持用原始 loc_name（可能含方位词）
    with open(out_instr, "w", encoding="utf-8") as f:
        f.write("# traj_id  pose_id_start  pose_id_end  instruction\n")
        for info in traj_infos:
            real_h = int(round(float(info["end_height_rel"]) + 80.0))
            instr = f"Fly to {real_h} meters above the {info['loc_name']}."
            f.write(f"{info['traj_id']} {info['pose_id_start']} {info['pose_id_end']} {instr}\n")

    print(f"[INFO] 轨迹 instruction 已写入: {out_instr}")

    # ---------- write subtask.txt ----------
    # subtask 中：
    #   - turn：仍用原始 loc_name
    #   - fly/descend：用 loc_name_clean（去方位词）
    with open(out_subtask, "w", encoding="utf-8") as f:
        f.write("# traj_id  subtask_id  pose_id_start  pose_id_end  subtask\n")
        for info in traj_infos:
            traj_id = info["traj_id"]
            loc_name = info["loc_name"]
            loc_name_clean = info.get("loc_name_clean", loc_name)

            real_h = int(round(float(info["end_height_rel"]) + 80.0))

            # 0) turn（需要时）
            if info["turn_end"] >= info["turn_start"]:
                sub = f"Turn to face the {loc_name}."
                f.write(f"{traj_id} 0 {info['turn_start']} {info['turn_end']} {sub}\n")

            # 1) fly to above（去方位词）
            if info["flyA_end"] >= info["flyA_start"]:
                sub = f"Fly to above the {loc_name_clean}."
                f.write(f"{traj_id} 1 {info['flyA_start']} {info['flyA_end']} {sub}\n")

            # 2) descend（去方位词）
            if info["descend_end"] >= info["descend_start"]:
                sub = f"Descend to {real_h} meters above the {loc_name_clean}."
                f.write(f"{traj_id} 2 {info['descend_start']} {info['descend_end']} {sub}\n")

    print(f"[INFO] subtask 标签已写入: {out_subtask}")

    # ---------- write traj_meta.txt ----------
    # 用 TSV，避免 loc_name 中有空格时解析困难
    with open(out_meta, "w", encoding="utf-8") as f:
        f.write("# traj_id\tloc_name\tpose_id_start\tpose_id_end\n")
        for info in traj_infos:
            f.write(f"{info['traj_id']}\t{info['loc_name']}\t{info['pose_id_start']}\t{info['pose_id_end']}\n")

    print(f"[INFO] 轨迹 meta 已写入: {out_meta}")

    # ---------- write plys ----------
    write_camera_points_ply(cameras_xml, str(out_ply_xml), color=(255, 0, 0), use_offset=True)
    write_random_points_ply(poses, str(out_ply_random), color=(0, 255, 0), use_offset=True)
    print(f"[INFO] PLY 已写入: {out_ply_xml}, {out_ply_random}")

    print("[INFO] 已禁用 global->local 转换：仅输出相对坐标轨迹，不生成 traj_random_local.txt")


if __name__ == "__main__":
    main()

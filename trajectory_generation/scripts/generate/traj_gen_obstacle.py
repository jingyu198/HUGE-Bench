#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Obstacle-avoidance trajectory generator (horizontal-view camera) with ECCV traj outputs.

Changes requested:
1) Start and goal are BOTH randomly sampled (no fixed --start/--goal).
2) Sampling rectangle (a vertical rectangle in a plane) is defined by user input:
   node1 node2 height_l height_h (two endpoints + two heights) for start AND for goal.
   A random point is sampled inside that rectangle.
3) All start points have initial facing direction perpendicular to the START sampling plane.
   Goal only needs to be reached (no facing constraint).
4) instruction meanings (EN, by env_id):
    overhead_bridge: Fly to the area behind the pedestrian bridge.
    no1_building:    Fly to the corner between the two buildings.
    no3_door:        Fly to the back of the building.
5) subtask: first turn to face the target, then fly to target while avoiding obstacles.
6) instruction and subtask are written in English (not Chinese).
"""

import os
import math
import time
import argparse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import trimesh
from scipy.spatial import cKDTree
import xml.etree.ElementTree as ET


# =============================================================================
# 0) env_id mapping (same style as landmark script)
# =============================================================================
ENV_CONFIGS = {
    "no3_door": {
        "mesh_rel": Path("terra_ply") / "simplified_mesh.obj",
        "start_rect": {  # REL by default
            "node1": (150.0, -24.0),
            "node2": (160.0,  -6.0),
            "h_l": -65.0,
            "h_h": -55.0,
        },
        "goal_rect": {
            "node1": (-13.0,  11.0),
            "node2": ( -3.0,  -7.0),
            "h_l": -60.0,
            "h_h": -50.0,
        },
    },
    "no1_building": {
        "mesh_rel": Path("terra_ply") / "simplified_mesh.obj",
        "start_rect": {  # TODO: fill your env-specific numbers
            "node1": (130, 34),
            "node2": (96, 120),
            "h_l": -65.0,
            "h_h": -55.0,
        },
        "goal_rect": {   # TODO: fill
            "node1": (-70,10),
            "node2": ( -73.0, 20),
            "h_l": -65.0,
            "h_h": -60.0,
        },
    },
    "overhead_bridge": {
        "mesh_rel": Path("terra_ply") / "simplified_mesh.obj",
        "start_rect": {  # TODO: fill
            "node1": (-82,84),
            "node2": (-72,102),
            "h_l": -72.0,
            "h_h": -64.0,
        },
        "goal_rect": {   # TODO: fill
            "node1": (-130.0,  114.0),
            "node2": ( -124,127),
            "h_l": -75.0,
            "h_h": -73.0,
        },
    },
    "2_city": {
        "mesh_rel": Path("terra_ply") / "simplified_mesh.obj",
        "start_rect": {  # TODO: fill
            "node1": (128,-80),
            "node2": (276,-34),
            "h_l": -80,
            "h_h": -65,
        },
        "goal_rect": {   # TODO: fill
            "node1": (60,94),
            "node2": (273, 201),
            "h_l": -80,
            "h_h": -65,
        },
    },
    # 如果你确实有第4个 env_id：在这里再加一项，并且下面 argparse 的 choices 也要包含它
}

# Instruction mapping (EN)
ENV_INSTRUCTION_EN = {
    "overhead_bridge": "Fly to the area behind the pedestrian bridge while avoiding obstacles.",
    "no1_building": "Fly to the back of the building while avoiding obstacles.",
    "no3_door": "Fly to the corner between the two buildings while avoiding obstacles.",
    "2_city" : "Fly through the current building complex while avoiding obstacles.",
}

DATA_ROOT = Path(os.environ.get("HUGE_DATA_3D_ROOT", "./data_3d")).expanduser()
OUT_ROOT = Path(os.environ.get("HUGE_DATA_TRAJ_ROOT", "./data_traj")).expanduser() / "task_obstacle"


def resolve_env_paths_and_defaults(env_id: str) -> Dict[str, Path]:
    if env_id not in ENV_CONFIGS:
        raise ValueError(f"Unknown env_id: {env_id}. choices={list(ENV_CONFIGS.keys())}")

    data_path = DATA_ROOT / env_id
    out_dir = OUT_ROOT / env_id

    cfg = ENV_CONFIGS[env_id]
    return {
        "data_path": data_path,
        "out_dir": out_dir,
        "xml_path": data_path / "BlocksExchangeUndistortAT_WithoutTiePoints.xml",
        "metadata_path": data_path / "terra_ply" / "metadata.xml",
        "mesh_path": data_path / cfg["mesh_rel"],
    }


# =============================================================================
# 1) Fixed planner hyper-params (kept fixed)
# =============================================================================
RRT_RADIUS = 3
RRT_STEP_SIZE = 1.0
RRT_GOAL_BIAS = 0.10
RRT_MAX_ITERS = 15000
RRT_GOAL_THRESHOLD = 1.5
RRT_COLLISION_SAMPLE_STEP = 0.2
RRT_KDTREE_REBUILD_EVERY = 200
RRT_SHORTCUT_ITERS = 300

BBX_HALF_WIDTH = 50
BBX_HALF_HEIGHT = 10.0
BBX_PAD = 0.0

SMOOTH_ITERS = 8


# =============================================================================
# Utils
# =============================================================================
def parse_vec2(s: str) -> np.ndarray:
    parts = s.replace(" ", "").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Vector2 must be 'x,y'")
    return np.array([float(parts[0]), float(parts[1])], dtype=np.float64)

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def parse_vec3(s: str) -> np.ndarray:
    parts = s.replace(" ", "").split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Vector must be 'x,y,z'")
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)


def _norm2(v2: np.ndarray) -> float:
    return float(np.linalg.norm(v2))


def sample_point_in_vertical_rect_xy(
    node1_xy: np.ndarray,
    node2_xy: np.ndarray,
    height_l: float,
    height_h: float,
    rng: np.random.Generator,
) -> np.ndarray:
    t = float(rng.random())
    xy = (1.0 - t) * node1_xy + t * node2_xy
    z = float(rng.uniform(min(height_l, height_h), max(height_l, height_h)))
    return np.array([xy[0], xy[1], z], dtype=np.float64)

def perpendicular_dir_to_vertical_plane_xy_toward_goal(
    node1_xy: np.ndarray,
    node2_xy: np.ndarray,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Return a unit vector perpendicular to the vertical plane defined by (node1_xy, node2_xy),
    choosing the sign such that it points from start toward goal in XY.
    """
    u = (node2_xy - node1_xy).astype(np.float64)
    if float(np.linalg.norm(u)) < 1e-9:
        # Degenerate plane definition; default to +x
        n_xy = np.array([1.0, 0.0], dtype=np.float64)
    else:
        ux, uy = float(u[0]), float(u[1])
        n_xy = np.array([uy, -ux], dtype=np.float64)  # one of the normals in XY
        n_xy /= (float(np.linalg.norm(n_xy)) + 1e-12)

    to_goal = (goal_xy - start_xy).astype(np.float64)
    if float(np.linalg.norm(to_goal)) < 1e-9:
        # start == goal (rare), keep as is
        return n_xy

    # choose the direction that faces the goal
    if float(np.dot(n_xy, to_goal)) < 0.0:
        n_xy = -n_xy
    elif abs(float(np.dot(n_xy, to_goal))) < 1e-12 and rng is not None:
        # If exactly perpendicular to to_goal, break tie randomly (optional)
        if float(rng.random()) < 0.5:
            n_xy = -n_xy

    return n_xy

# =============================================================================
# metadata offset (SRSOrigin)
# =============================================================================
def parse_dji_offset_from_metadata(metadata_path: Optional[str]) -> np.ndarray:
    if metadata_path is None:
        return np.zeros(3, dtype=np.float64)
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"metadata.xml not found: {metadata_path}")

    tree = ET.parse(metadata_path)
    root = tree.getroot()
    node = root.find("SRSOrigin")
    if node is None or not node.text:
        raise RuntimeError(f"<SRSOrigin> not found or empty in: {metadata_path}")

    parts = [p.strip() for p in node.text.strip().split(",")]
    if len(parts) != 3:
        raise RuntimeError(f"SRSOrigin should be 'x,y,z', got: {node.text.strip()}")
    offset = np.array([float(p) for p in parts], dtype=np.float64)
    log(f"[INFO] DJI_OFFSET parsed: {offset.tolist()}")
    return offset


# =============================================================================
# Optional: intrinsics + xml cameras
# =============================================================================
def parse_intrinsics_from_xml(xml_path: Optional[str]) -> Optional[Dict[str, float]]:
    if xml_path is None or (not os.path.isfile(xml_path)):
        return None

    tree = ET.parse(xml_path)
    root = tree.getroot()
    block = root.find("Block")
    if block is None:
        return None
    photogroups = block.find("Photogroups")
    if photogroups is None:
        return None
    pg = photogroups.find("Photogroup")
    if pg is None:
        return None

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

    log(f"[INFO] Intrinsics: w={width}, h={height}, fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")
    return {"width": width, "height": height, "fx": fx, "fy": fy, "cx": cx, "cy": cy}


def parse_all_cameras_from_xml(xml_path: Optional[str]) -> List[Dict[str, Any]]:
    if xml_path is None or (not os.path.isfile(xml_path)):
        return []
    tree = ET.parse(xml_path)
    root = tree.getroot()

    block = root.find("Block")
    if block is None:
        return []
    photogroups = block.find("Photogroups")
    if photogroups is None:
        return []

    cams = []
    for pg in photogroups.findall("Photogroup"):
        for photo in pg.findall("Photo"):
            pid = int(photo.find("Id").text)
            center = photo.find("Pose/Center")
            C_world = np.array(
                [float(center.find("x").text), float(center.find("y").text), float(center.find("z").text)],
                dtype=np.float64,
            )
            cams.append({"id": pid, "C_world": C_world})
    return cams


# =============================================================================
# Output writers
# =============================================================================
def write_camera_points_ply_from_world(
    points_world: np.ndarray,
    ply_path: Union[str, Path],
    dji_offset: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    use_offset: bool = True,
):
    ply_path = str(ply_path)
    os.makedirs(os.path.dirname(ply_path) or ".", exist_ok=True)
    r, g, b = color

    if points_world.size == 0:
        with open(ply_path, "w", encoding="utf-8") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write("element vertex 0\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
        return

    verts = points_world.astype(np.float64) - dji_offset.reshape(1, 3) if use_offset else points_world.astype(np.float64)

    with open(ply_path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {verts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for x, y, z in verts:
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


def write_cameras_xml_ply(cameras_xml: List[Dict[str, Any]], out_ply: Union[str, Path], dji_offset: np.ndarray):
    if len(cameras_xml) == 0:
        write_camera_points_ply_from_world(np.zeros((0, 3), dtype=np.float64), out_ply, dji_offset, color=(255, 0, 0))
        return
    pts = np.vstack([c["C_world"] for c in cameras_xml]).astype(np.float64)
    write_camera_points_ply_from_world(pts, out_ply, dji_offset, color=(255, 0, 0), use_offset=True)


# =============================================================================
# Camera orientation: OPK
# =============================================================================
def R_to_opk(R: np.ndarray) -> Tuple[float, float, float]:
    r20 = float(np.clip(R[2, 0], -1.0, 1.0))
    phi = math.asin(r20)

    cos_phi = math.cos(phi)
    if abs(cos_phi) < 1e-8:
        omega = 0.0
        kappa = math.atan2(-R[1, 2], R[1, 1])
    else:
        omega = math.atan2(-R[2, 1], R[2, 2])
        kappa = math.atan2(-R[1, 0], R[0, 0])
    return omega, phi, kappa


def _angle_diff(yaw_to: float, yaw_from: float) -> float:
    return (yaw_to - yaw_from + math.pi) % (2.0 * math.pi) - math.pi


def sample_yaw_only_dirs(dir_from: np.ndarray, dir_to: np.ndarray, step_deg: float) -> List[np.ndarray]:
    dir_from = np.asarray(dir_from, dtype=np.float64)
    dir_to = np.asarray(dir_to, dtype=np.float64)

    if np.linalg.norm(dir_from) < 1e-6:
        dir_from = np.array([1.0, 0.0], dtype=np.float64)
    else:
        dir_from /= (np.linalg.norm(dir_from) + 1e-12)

    if np.linalg.norm(dir_to) < 1e-6:
        return []
    dir_to /= (np.linalg.norm(dir_to) + 1e-12)

    yaw_from = math.atan2(dir_from[1], dir_from[0])
    yaw_to = math.atan2(dir_to[1], dir_to[0])
    d_yaw = _angle_diff(yaw_to, yaw_from)
    total_deg = abs(math.degrees(d_yaw))
    if total_deg < 1e-3:
        return []

    n_steps = max(1, int(total_deg // step_deg))
    out = []
    for i in range(1, n_steps + 1):
        a = i / float(n_steps)
        yaw_i = yaw_from + d_yaw * a
        out.append(np.array([math.cos(yaw_i), math.sin(yaw_i)], dtype=np.float64))
    return out


def build_level_R_from_dir(dir_xy: np.ndarray) -> np.ndarray:
    """
    Horizontal-view camera:
      - camera z-axis forward along dir_xy in world XY
      - camera y-axis image-down aligned with world -Z
      - x = y × z (right-handed)
    """
    dir_xy = np.asarray(dir_xy, dtype=np.float64)
    if np.linalg.norm(dir_xy) < 1e-6:
        dir_xy = np.array([1.0, 0.0], dtype=np.float64)
    else:
        dir_xy = dir_xy / (np.linalg.norm(dir_xy) + 1e-12)

    z_c_w = np.array([dir_xy[0], dir_xy[1], 0.0], dtype=np.float64)
    y_c_w = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    x_c_w = np.cross(y_c_w, z_c_w)
    nx = np.linalg.norm(x_c_w)
    if nx < 1e-9:
        x_c_w = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        x_c_w /= nx

    y_c_w = np.cross(z_c_w, x_c_w)
    y_c_w /= (np.linalg.norm(y_c_w) + 1e-12)

    R = np.stack([x_c_w, y_c_w, z_c_w], axis=0)
    return R.astype(np.float64)


# =============================================================================
# Mesh + collision utilities
# =============================================================================
def load_as_single_mesh(path: str) -> trimesh.Trimesh:
    geom = trimesh.load(path, force=None, process=False)
    if isinstance(geom, trimesh.Scene):
        meshes = []
        for _, g in geom.geometry.items():
            if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0:
                meshes.append(g)
        if not meshes:
            raise ValueError("Scene has no triangle meshes.")
        mesh = trimesh.util.concatenate(meshes)
    elif isinstance(geom, trimesh.Trimesh):
        mesh = geom
    else:
        raise TypeError(f"Unsupported loaded type: {type(geom)}")
    if mesh.vertices is None or mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError("Mesh empty/invalid.")
    mesh.remove_unreferenced_vertices()
    return mesh


def build_ray_intersector(mesh: trimesh.Trimesh):
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector  # type: ignore
        return RayMeshIntersector(mesh)
    except Exception:
        return mesh.ray


def _call_ray_method(method, O, D):
    variants = [
        ((), dict(origins=O, directions=D)),
        ((), dict(ray_origins=O, ray_directions=D)),
        ((O, D), dict()),
    ]
    for args, kwargs in variants:
        try:
            return method(*args, **kwargs)
        except TypeError:
            continue
    return None


def segment_intersects_mesh(ray, p0: np.ndarray, p1: np.ndarray) -> bool:
    d = p1 - p0
    L = float(np.linalg.norm(d))
    if L <= 1e-12:
        return False
    dir_unit = d / L
    O = np.array([p0], dtype=float)
    D = np.array([dir_unit], dtype=float)

    if hasattr(ray, "intersects_first"):
        res = _call_ray_method(ray.intersects_first, O, D)
        if res is not None:
            dist = float(np.asarray(res)[0])
            if not np.isfinite(dist):
                return False
            eps = 1e-8
            return (dist >= -eps) and (dist <= L + eps)

    if hasattr(ray, "intersects_any"):
        ray_len = np.array([L], dtype=float)
        for kwargs in [
            dict(origins=O, directions=D, ray_lengths=ray_len),
            dict(ray_origins=O, ray_directions=D, ray_lengths=ray_len),
        ]:
            try:
                res = ray.intersects_any(**kwargs)
                return bool(np.asarray(res)[0])
            except TypeError:
                pass

    call_variants = [
        ((), dict(origins=O, directions=D, multiple_hits=True)),
        ((), dict(origins=O, directions=D)),
        ((), dict(ray_origins=O, ray_directions=D, multiple_hits=True)),
        ((), dict(ray_origins=O, ray_directions=D)),
        ((O, D), dict(multiple_hits=True)),
        ((O, D), dict()),
    ]
    locations = None
    for args, kwargs in call_variants:
        try:
            locations, _, _ = ray.intersects_location(*args, **kwargs)
            break
        except TypeError:
            continue
    if locations is None or len(locations) == 0:
        return False
    t = np.dot(locations - p0, dir_unit)
    eps = 1e-8
    return bool(np.any((t >= -eps) & (t <= L + eps)))


def make_distance_query(mesh: trimesh.Trimesh, surface_samples: int = 200000) -> Callable[[np.ndarray], np.ndarray]:
    try:
        import rtree  # noqa: F401
        from trimesh.proximity import ProximityQuery
        pq = ProximityQuery(mesh)

        def dist_fn(points: np.ndarray) -> np.ndarray:
            if hasattr(pq, "closest_point"):
                try:
                    _, dist, _ = pq.closest_point(points)
                    return np.asarray(dist)
                except TypeError:
                    pass
            if hasattr(pq, "on_surface"):
                try:
                    _, dist, _ = pq.on_surface(points)
                    return np.asarray(dist)
                except TypeError:
                    pass
            import trimesh.proximity
            _, dist, _ = trimesh.proximity.closest_point(mesh, points)
            return np.asarray(dist)

        return dist_fn
    except Exception:
        samples = mesh.sample(int(surface_samples))
        kdt = cKDTree(samples)

        def dist_fn(points: np.ndarray) -> np.ndarray:
            d, _ = kdt.query(points, k=1)
            return d

        return dist_fn


def _sample_points_on_segment(p0: np.ndarray, p1: np.ndarray, step: float) -> np.ndarray:
    d = p1 - p0
    L = float(np.linalg.norm(d))
    if L <= 1e-12:
        return p0[None, :].copy()
    n = int(math.ceil(L / max(step, 1e-6))) + 1
    ts = np.linspace(0.0, 1.0, n)
    return p0[None, :] + ts[:, None] * d[None, :]


def segment_free(ray, dist_fn: Optional[Callable[[np.ndarray], np.ndarray]],
                 p0: np.ndarray, p1: np.ndarray, radius: float, sample_step: float) -> bool:
    if segment_intersects_mesh(ray, p0, p1):
        return False
    if radius <= 0.0:
        return True
    if dist_fn is None:
        raise RuntimeError("dist_fn required when radius > 0")
    pts = _sample_points_on_segment(p0, p1, sample_step)
    dist = dist_fn(pts)
    return bool(np.all(dist >= radius))


def crop_mesh_to_strip_bbx(mesh: trimesh.Trimesh, start: np.ndarray, goal: np.ndarray,
                           half_width: float, half_height: float, pad: float, margin: float,
                           chunk_faces: int = 300_000) -> trimesh.Trimesh:
    s0 = start.astype(float)
    g0 = goal.astype(float)
    dxy = g0[:2] - s0[:2]
    Lxy = float(np.linalg.norm(dxy))

    faces = mesh.faces
    V = mesh.vertices
    if len(faces) == 0:
        raise ValueError("Mesh has no faces.")

    half_width = float(max(1e-9, half_width))
    half_height = float(max(1e-9, half_height))
    pad = float(max(0.0, pad))
    margin = float(max(0.0, margin))

    z_lo = min(float(s0[2]), float(g0[2])) - half_height - margin
    z_hi = max(float(s0[2]), float(g0[2])) + half_height + margin

    keep = np.zeros(len(faces), dtype=bool)

    if Lxy > 1e-6:
        u = dxy / Lxy
        v = np.array([-u[1], u[0]], dtype=float)
        s0xy = s0[:2]

        s_lo = -pad - margin
        s_hi = Lxy + pad + margin
        t_lo = -half_width - margin
        t_hi = half_width + margin

        for i in range(0, len(faces), chunk_faces):
            fi = faces[i:i + chunk_faces]
            tri = V[fi]
            tri_xy = tri[:, :, :2] - s0xy[None, None, :]
            s = tri_xy @ u
            t = tri_xy @ v
            smin = s.min(axis=1); smax = s.max(axis=1)
            tmin = t.min(axis=1); tmax = t.max(axis=1)
            zmin = tri[:, :, 2].min(axis=1); zmax = tri[:, :, 2].max(axis=1)
            keep[i:i + chunk_faces] = (
                (smax >= s_lo) & (smin <= s_hi) &
                (tmax >= t_lo) & (tmin <= t_hi) &
                (zmax >= z_lo) & (zmin <= z_hi)
            )
    else:
        bbx_min = np.minimum(s0, g0) - np.array([pad + half_width + margin,
                                                 pad + half_width + margin,
                                                 half_height + margin], dtype=float)
        bbx_max = np.maximum(s0, g0) + np.array([pad + half_width + margin,
                                                 pad + half_width + margin,
                                                 half_height + margin], dtype=float)
        for i in range(0, len(faces), chunk_faces):
            fi = faces[i:i + chunk_faces]
            tri = V[fi]
            tri_min = tri.min(axis=1)
            tri_max = tri.max(axis=1)
            keep[i:i + chunk_faces] = np.all(tri_max >= bbx_min, axis=1) & np.all(tri_min <= bbx_max, axis=1)

    idx = np.flatnonzero(keep)
    if len(idx) == 0:
        raise ValueError("Cropped mesh is empty. Increase BBX or check coordinate mismatch.")

    sub = mesh.submesh([idx], append=True, repair=False)
    sub.remove_unreferenced_vertices()
    return sub


def make_strip_bbx_sampler(start: np.ndarray, goal: np.ndarray,
                           half_width: float, half_height: float, pad: float,
                           rng: np.random.Generator) -> Callable[[], np.ndarray]:
    s0 = start.astype(float)
    g0 = goal.astype(float)
    dxy = g0[:2] - s0[:2]
    Lxy = float(np.linalg.norm(dxy))

    half_width = float(max(1e-9, half_width))
    half_height = float(max(1e-9, half_height))
    pad = float(max(0.0, pad))

    if Lxy > 1e-6:
        u = dxy / Lxy
        v = np.array([-u[1], u[0]], dtype=float)
        z0 = float(s0[2]); z1 = float(g0[2])

        def sample() -> np.ndarray:
            s = rng.uniform(-pad, Lxy + pad)
            t = rng.uniform(-half_width, half_width)
            xy = s0[:2] + u * s + v * t
            alpha = float(np.clip(s / Lxy, 0.0, 1.0))
            zc = (1.0 - alpha) * z0 + alpha * z1
            z = zc + rng.uniform(-half_height, half_height)
            return np.array([xy[0], xy[1], z], dtype=float)

        return sample

    def sample_same() -> np.ndarray:
        return s0 + rng.uniform([-pad, -half_width, -half_height], [pad, half_width, half_height])

    return sample_same


# =============================================================================
# Planner function (packed)
# =============================================================================
def plan_obstacle_avoidance_path(
    start: np.ndarray,
    goal: np.ndarray,
    mesh: Union[str, trimesh.Trimesh],
    seed: int,
    resample_step: float,
    debug_mesh_out_rel: Optional[Union[str, Path]] = None,
    debug_mesh_out_world: Optional[Union[str, Path]] = None,
    dji_offset: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    if isinstance(mesh, str):
        if not os.path.exists(mesh):
            raise FileNotFoundError(f"Mesh not found: {mesh}")
        mesh_obj = load_as_single_mesh(mesh)
    else:
        mesh_obj = mesh

    # crop for speed
    margin = float(RRT_RADIUS + RRT_COLLISION_SAMPLE_STEP)
    mesh_obj = crop_mesh_to_strip_bbx(mesh_obj, start, goal,
                                      half_width=BBX_HALF_WIDTH, half_height=BBX_HALF_HEIGHT,
                                      pad=BBX_PAD, margin=margin)

    # ---- DEBUG EXPORT (cropped mesh actually used by RRT) ----
    if debug_mesh_out_rel is not None:
        debug_mesh_out_rel = Path(debug_mesh_out_rel)
        debug_mesh_out_rel.parent.mkdir(parents=True, exist_ok=True)
        mesh_obj.export(str(debug_mesh_out_rel))  # .ply/.obj/.glb 看后缀自动导出
        log(f"[INFO] Exported RRT mesh (REL): {debug_mesh_out_rel}")

    if debug_mesh_out_world is not None:
        if dji_offset is None:
            raise ValueError("dji_offset is required for debug_mesh_out_world")
        debug_mesh_out_world = Path(debug_mesh_out_world)
        debug_mesh_out_world.parent.mkdir(parents=True, exist_ok=True)
        m2 = mesh_obj.copy()
        m2.vertices = m2.vertices + dji_offset.reshape(1, 3)
        m2.export(str(debug_mesh_out_world))
        log(f"[INFO] Exported RRT mesh (WORLD): {debug_mesh_out_world}")

    rng = np.random.default_rng(seed)
    ray = build_ray_intersector(mesh_obj)
    dist_fn = make_distance_query(mesh_obj, surface_samples=200000) if RRT_RADIUS > 0 else None

    # clearance check
    if RRT_RADIUS > 0 and dist_fn is not None:
        if float(dist_fn(start[None, :])[0]) < RRT_RADIUS:
            raise ValueError("Start is in collision / too close to mesh.")
        if float(dist_fn(goal[None, :])[0]) < RRT_RADIUS:
            raise ValueError("Goal is in collision / too close to mesh.")

    # RRT buffers
    max_nodes = int(RRT_MAX_ITERS) + 4
    nodes = np.empty((max_nodes, 3), dtype=np.float64)
    parents = np.empty((max_nodes,), dtype=np.int32)
    nodes[0] = start.astype(np.float64)
    parents[0] = -1
    n_nodes = 1

    tree = cKDTree(nodes[:1])
    tree_size = 1

    sampler = make_strip_bbx_sampler(start, goal, BBX_HALF_WIDTH, BBX_HALF_HEIGHT, BBX_PAD, rng)

    def nearest(q: np.ndarray) -> int:
        nonlocal tree, tree_size, n_nodes
        dist, idx = tree.query(q, k=1)
        best_idx = int(idx)
        best_dist2 = float(dist) * float(dist)
        if tree_size < n_nodes:
            recent = nodes[tree_size:n_nodes]
            diff = recent - q[None, :]
            d2 = np.einsum("ij,ij->i", diff, diff)
            j = int(np.argmin(d2))
            if float(d2[j]) < best_dist2:
                return tree_size + j
        return best_idx

    def maybe_rebuild():
        nonlocal tree, tree_size, n_nodes
        if (n_nodes - tree_size) >= RRT_KDTREE_REBUILD_EVERY:
            tree = cKDTree(nodes[:n_nodes])
            tree_size = n_nodes

    found_goal_idx = None

    for _ in range(RRT_MAX_ITERS):
        q_rand = goal if (rng.random() < RRT_GOAL_BIAS) else sampler()
        idx_near = nearest(q_rand)
        q_near = nodes[idx_near]

        v = q_rand - q_near
        L = float(np.linalg.norm(v))
        if L <= 1e-12:
            continue
        v = v / L
        q_new = q_near + v * min(RRT_STEP_SIZE, L)

        if not segment_free(ray, dist_fn, q_near, q_new, RRT_RADIUS, RRT_COLLISION_SAMPLE_STEP):
            continue

        if n_nodes >= max_nodes:
            raise RuntimeError("Node buffer full.")
        nodes[n_nodes] = q_new
        parents[n_nodes] = idx_near
        n_nodes += 1
        maybe_rebuild()

        if float(np.linalg.norm(q_new - goal)) <= RRT_GOAL_THRESHOLD:
            if segment_free(ray, dist_fn, q_new, goal, RRT_RADIUS, RRT_COLLISION_SAMPLE_STEP):
                if n_nodes >= max_nodes:
                    raise RuntimeError("Node buffer full when appending goal.")
                nodes[n_nodes] = goal.astype(np.float64)
                parents[n_nodes] = n_nodes - 1
                found_goal_idx = n_nodes
                n_nodes += 1
                break

    if found_goal_idx is None:
        raise RuntimeError("RRT failed to find a path. Increase BBX or max iters.")

    # backtrack
    path = []
    cur = int(found_goal_idx)
    while cur != -1:
        path.append(nodes[cur].copy())
        cur = int(parents[cur])
    path.reverse()

    # shortcut
    rng2 = np.random.default_rng(seed + 12345)
    pts = list(path)
    for _ in range(RRT_SHORTCUT_ITERS):
        if len(pts) <= 2:
            break
        i = int(rng2.integers(0, len(pts) - 1))
        j = int(rng2.integers(i + 1, len(pts)))
        if j <= i + 1:
            continue
        if segment_free(ray, dist_fn, pts[i], pts[j], RRT_RADIUS, RRT_COLLISION_SAMPLE_STEP):
            pts = pts[: i + 1] + pts[j:]

    # Chaikin smooth
    def chaikin(points: List[np.ndarray], iters: int) -> List[np.ndarray]:
        if len(points) < 3 or iters <= 0:
            return list(points)
        cur_pts = [p.astype(float) for p in points]
        for _ in range(iters):
            new_pts = [cur_pts[0]]
            for k in range(len(cur_pts) - 1):
                p0, p1 = cur_pts[k], cur_pts[k + 1]
                Q = 0.75 * p0 + 0.25 * p1
                R = 0.25 * p0 + 0.75 * p1
                new_pts.append(Q); new_pts.append(R)
            new_pts.append(cur_pts[-1])
            cur_pts = new_pts
        # dedup
        out = [cur_pts[0]]
        for p in cur_pts[1:]:
            if np.linalg.norm(p - out[-1]) > 1e-9:
                out.append(p)
        return out

    # resample by arc-length
    def resample(points: List[np.ndarray], step: float) -> List[np.ndarray]:
        if len(points) <= 1:
            return [p.copy() for p in points]
        step = float(step)
        if step <= 1e-9:
            return [p.copy() for p in points]

        P = np.vstack(points).astype(float)
        seg = P[1:] - P[:-1]
        seg_len = np.linalg.norm(seg, axis=1)
        total = float(seg_len.sum())
        if total <= 1e-9:
            return [P[0].copy()]

        n_samples = int(np.floor(total / step))
        targets = [i * step for i in range(n_samples + 1)]
        if targets[-1] < total:
            targets.append(total)

        out = []
        cur_seg = 0
        cur_s = 0.0
        for t in targets:
            while cur_seg < len(seg_len) - 1 and (cur_s + seg_len[cur_seg]) < t:
                cur_s += float(seg_len[cur_seg])
                cur_seg += 1
            Ls = float(seg_len[cur_seg])
            if Ls <= 1e-12:
                out.append(P[cur_seg].copy())
                continue
            a = float(np.clip((t - cur_s) / Ls, 0.0, 1.0))
            p = P[cur_seg] + a * (P[cur_seg + 1] - P[cur_seg])
            out.append(p)
        return out

    cand = resample(chaikin(pts, SMOOTH_ITERS), resample_step)

    # verify; fallback to non-chaikin resample if collision
    ok = True
    for i in range(len(cand) - 1):
        if not segment_free(ray, dist_fn, cand[i], cand[i + 1], RRT_RADIUS, RRT_COLLISION_SAMPLE_STEP):
            ok = False
            break
    if not ok:
        cand = resample(pts, resample_step)

    return [p.astype(np.float64) for p in cand]


# =============================================================================
# Convert path -> poses (horizontal camera) with in-place yaw turn
# =============================================================================
def wrap_deg180(a: float) -> float:
    return (float(a) + 180.0) % 360.0 - 180.0


def canonical_obstacle_opk_from_dir_xy(dir_xy: np.ndarray) -> Tuple[float, float, float]:
    """
    Canonical obstacle label:
      omega = -90 deg
      kappa =   0 deg
      phi   = heading encoded from horizontal forward dir

    For build_level_R_from_dir(dir_xy), this exactly reproduces the same camera view,
    but avoids OPK branch switching near phi ~= +/-90 deg.
    """
    d = np.asarray(dir_xy, dtype=np.float64).reshape(2)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        d = np.array([1.0, 0.0], dtype=np.float64)
    else:
        d = d / (n + 1e-12)

    dx, dy = float(d[0]), float(d[1])

    omega = -90.0
    phi = wrap_deg180(math.degrees(math.atan2(dx, dy)))
    kappa = 0.0
    return omega, phi, kappa

def path_to_horizontal_poses(
    path_rel: List[np.ndarray],
    yaw_step_deg: float,
    init_dir_xy: Optional[np.ndarray] = None,
):
    """
    init_dir_xy:
      - if provided, used as initial facing direction at start (XY unit vector)
      - otherwise random
    """
    if len(path_rel) < 2:
        raise ValueError("Path too short.")

    d0 = path_rel[1][:2] - path_rel[0][:2]
    target_dir = d0 / (np.linalg.norm(d0) + 1e-12) if np.linalg.norm(d0) >= 1e-6 else np.array([1.0, 0.0], dtype=np.float64)

    if init_dir_xy is None:
        theta0 = 2.0 * math.pi * float(np.random.random())
        init_dir = np.array([math.cos(theta0), math.sin(theta0)], dtype=np.float64)
    else:
        init_dir = np.asarray(init_dir_xy, dtype=np.float64)
        if np.linalg.norm(init_dir) < 1e-6:
            init_dir = np.array([1.0, 0.0], dtype=np.float64)
        else:
            init_dir = init_dir / (np.linalg.norm(init_dir) + 1e-12)

    yaw_dirs = sample_yaw_only_dirs(init_dir, target_dir, step_deg=float(yaw_step_deg))

    poses: List[Dict[str, Any]] = []
    pose_id = 0

    # turn segment (optional)
    turn_start = pose_id
    if len(yaw_dirs) > 0:
        p0 = path_rel[0].astype(np.float64)
        for dxy in yaw_dirs:
            o_deg, p_deg, k_deg = canonical_obstacle_opk_from_dir_xy(dxy)
            poses.append({
                "id": pose_id,
                "C_rel": p0.copy(),
                "omega": o_deg,
                "phi": p_deg,
                "kappa": k_deg,
            })
            pose_id += 1
        turn_end = pose_id - 1
    else:
        turn_end = turn_start - 1

    # fly segment
    fly_points = path_rel[1:] if (turn_end >= turn_start) else path_rel
    fly_start = pose_id

    # --- smooth heading for fly segment (bisector + EMA low-pass) ---
    # fly_points: List[np.ndarray], each is (x,y,z)
    # Build 2D array of XY
    F = np.vstack([p[:2] for p in fly_points]).astype(np.float64)
    n = int(F.shape[0])

    # If only one fly point, just face target_dir
    if n <= 1:
        dxy = target_dir.copy()
        o_deg, p_deg, k_deg = canonical_obstacle_opk_from_dir_xy(dxy)
        pt = fly_points[0]
        poses.append({
            "id": pose_id,
            "C_rel": np.asarray(pt, dtype=np.float64),
            "omega": o_deg,
            "phi": p_deg,
            "kappa": k_deg,
        })
        pose_id += 1
    else:
        # 1) segment directions (between consecutive points)
        seg_dirs = np.zeros((n - 1, 2), dtype=np.float64)
        prev = target_dir.copy()
        for i in range(n - 1):
            d = F[i + 1] - F[i]
            nd = float(np.linalg.norm(d))
            if nd < 1e-6:
                seg_dirs[i] = prev
            else:
                prev = d / (nd + 1e-12)
                seg_dirs[i] = prev

        # 2) point directions = angle bisector of prev/next segment
        pt_dirs = np.zeros((n, 2), dtype=np.float64)
        pt_dirs[0] = seg_dirs[0]
        pt_dirs[-1] = seg_dirs[-1]
        for i in range(1, n - 1):
            d = seg_dirs[i - 1] + seg_dirs[i]
            nd = float(np.linalg.norm(d))
            if nd < 1e-6:
                pt_dirs[i] = seg_dirs[i]
            else:
                pt_dirs[i] = d / (nd + 1e-12)

        # 3) EMA low-pass on headings to make yaw change gradual
        alpha = 0.75  # 0.6~0.85 常用；越大越“丝滑”但转向会更滞后
        for i in range(1, n):
            d = alpha * pt_dirs[i - 1] + (1.0 - alpha) * pt_dirs[i]
            d /= (float(np.linalg.norm(d)) + 1e-12)
            pt_dirs[i] = d

        # 4) write poses with smoothed heading
        for i, pt in enumerate(fly_points):
            dxy = pt_dirs[i]
            o_deg, p_deg, k_deg = canonical_obstacle_opk_from_dir_xy(dxy)
            poses.append({
                "id": pose_id,
                "C_rel": np.asarray(pt, dtype=np.float64),
                "omega": o_deg,
                "phi": p_deg,
                "kappa": k_deg,
            })
            pose_id += 1
    fly_end = pose_id - 1

    ranges = {
        "turn_start": turn_start, "turn_end": turn_end,
        "fly_start": fly_start, "fly_end": fly_end,
        "pose_start": 0, "pose_end": fly_end,
    }
    return poses, ranges


# =============================================================================
# Write outputs
# =============================================================================
def write_outputs_merged(
    out_dir: Path,
    poses_all: List[Dict[str, Any]],
    traj_infos: List[Dict[str, Any]],
    dji_offset: np.ndarray,
    intr: Optional[Dict[str, float]],
    cameras_xml: List[Dict[str, Any]],
    env_id: str,
    sample_step: float,
    yaw_step_deg: float,
):
    """
    合并输出（参考 merged landmark 脚本）：
      - traj_random.txt: 全部 pose（全局递增 pose_id）
      - instruction.txt: 每条 traj 一行（traj_id + pose range + instruction）
      - traj_meta.txt: TSV（traj_id + loc_name + pose range）
      - subtask.txt: turn / fly 两段
      - cameras_random.ply: 全部 pose 点
      - cameras_xml.ply: 原始相机点（一次）
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    out_txt = out_dir / "traj_random.txt"
    out_ply_xml = out_dir / "cameras_xml.ply"
    out_ply_random = out_dir / "cameras_random.ply"
    out_instr = out_dir / "instruction.txt"
    out_meta = out_dir / "traj_meta.txt"
    out_subtask = out_dir / "subtask.txt"

    # fill C_world
    for p in poses_all:
        p["C_world"] = p["C_rel"] + dji_offset

    # ---------- traj_random.txt ----------
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("# id  x_rel  y_rel  z_rel  omega_deg  phi_deg  kappa_deg\n")
        f.write("# NOTE: x_rel,y_rel,z_rel = C_world - DJI_OFFSET\n")
        f.write(f"# env_id = {env_id}\n")
        f.write(f"# DJI_OFFSET = {dji_offset.tolist()}\n")
        if intr is not None:
            f.write("# intrinsics: width height fx fy cx cy\n")
            f.write(
                f"# {intr['width']} {intr['height']} "
                f"{intr['fx']:.6f} {intr['fy']:.6f} {intr['cx']:.6f} {intr['cy']:.6f}\n"
            )
        else:
            f.write("# intrinsics: (not provided)\n")
        f.write(
            f"# num_traj={len(traj_infos)}, sample_step={sample_step}, yaw_step_deg={yaw_step_deg}\n"
            f"# planner: radius={RRT_RADIUS}, rrt_step={RRT_STEP_SIZE}, max_iters={RRT_MAX_ITERS}, "
            f"bbx=[w={BBX_HALF_WIDTH},h={BBX_HALF_HEIGHT},pad={BBX_PAD}]\n"
            "# Each line is one pose; pose id is global increasing across all trajectories.\n"
        )
        for p in poses_all:
            c = p["C_rel"]
            f.write(
                f"{p['id']} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f} "
                f"{p['omega']:.6f} {p['phi']:.6f} {p['kappa']:.6f}\n"
            )
    log(f"[INFO] Wrote merged: {out_txt}")

    # ---------- cameras_xml.ply ----------
    write_cameras_xml_ply(cameras_xml, out_ply_xml, dji_offset)
    log(f"[INFO] Wrote: {out_ply_xml}")

    # ---------- cameras_random.ply ----------
    pts_world = np.vstack([p["C_world"] for p in poses_all]).astype(np.float64) if len(poses_all) > 0 else np.zeros((0, 3), dtype=np.float64)
    write_camera_points_ply_from_world(pts_world, out_ply_random, dji_offset, color=(0, 255, 0), use_offset=True)
    log(f"[INFO] Wrote merged: {out_ply_random}")

    # ---------- instruction.txt ----------
    instr = ENV_INSTRUCTION_EN.get(env_id, "Fly to the goal point while avoiding obstacles.")
    with open(out_instr, "w", encoding="utf-8") as f:
        f.write("# traj_id  pose_id_start  pose_id_end  instruction\n")
        for info in traj_infos:
            f.write(f"{info['traj_id']} {info['pose_id_start']} {info['pose_id_end']} {instr}\n")
    log(f"[INFO] Wrote merged: {out_instr}")

    # ---------- traj_meta.txt (TSV) ----------
    with open(out_meta, "w", encoding="utf-8") as f:
        f.write("# traj_id\tloc_name\tpose_id_start\tpose_id_end\n")
        for info in traj_infos:
            f.write(f"{info['traj_id']}\tstart_to_goal\t{info['pose_id_start']}\t{info['pose_id_end']}\n")
    log(f"[INFO] Wrote merged: {out_meta}")

    # ---------- subtask.txt ----------
    with open(out_subtask, "w", encoding="utf-8") as f:
        f.write("# traj_id  subtask_id  pose_id_start  pose_id_end  subtask\n")
        for info in traj_infos:
            traj_id = info["traj_id"]
            # 0) turn（需要时）；无 turn 时仍写单 pose 范围占位（与你原 write_outputs 一致）
            if info["turn_end"] >= info["turn_start"]:
                f.write(f"{traj_id} 0 {info['turn_start']} {info['turn_end']} Turn to face the target.\n")
                fly_s = info["fly_start"]
            else:
                f.write(f"{traj_id} 0 {info['pose_id_start']} {info['pose_id_start']} Turn to face the target.\n")
                fly_s = info["pose_id_start"]

            # 1) fly
            f.write(f"{traj_id} 1 {fly_s} {info['pose_id_end']} Fly to the target while avoiding obstacles.\n")
    log(f"[INFO] Wrote merged: {out_subtask}")
# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()

    # env_id (required)
    parser.add_argument("--env_id", type=str, required=True, choices=list(ENV_CONFIGS.keys()),
                        help="scene ID")

    # inputs (can override)
    parser.add_argument("--mesh_path", type=str, default=None, help="override mesh path")
    parser.add_argument("--xml_path", type=str, default=None, help="override BlocksExchange XML")
    parser.add_argument("--metadata_path", type=str, default=None, help="override metadata.xml")

    # Random sampling rectangles for START (env-specific defaults if omitted)
    parser.add_argument("--start_node1", type=parse_vec2, default=None,
                        help="start line: node1_xy (x,y). If omitted, use env default.")
    parser.add_argument("--start_node2", type=parse_vec2, default=None,
                        help="start line: node2_xy (x,y). If omitted, use env default.")
    parser.add_argument("--start_height_l", type=float, default=None,
                        help="start rectangle: height low (Z). If omitted, use env default.")
    parser.add_argument("--start_height_h", type=float, default=None,
                        help="start rectangle: height high (Z). If omitted, use env default.")

    # Random sampling rectangles for GOAL (env-specific defaults if omitted)
    parser.add_argument("--goal_node1", type=parse_vec2, default=None,
                        help="goal line: node1_xy (x,y). If omitted, use env default.")
    parser.add_argument("--goal_node2", type=parse_vec2, default=None,
                        help="goal line: node2_xy (x,y). If omitted, use env default.")
    parser.add_argument("--goal_height_l", type=float, default=None,
                        help="goal rectangle: height low (Z). If omitted, use env default.")
    parser.add_argument("--goal_height_h", type=float, default=None,
                        help="goal rectangle: height high (Z). If omitted, use env default.")

    # Coordinate mode
    parser.add_argument("--input_world", action="store_true",
                        help="treat all nodes/heights as WORLD and convert to REL by subtracting DJI_OFFSET (heights are Z in WORLD)")

    # sampling / yaw
    parser.add_argument("--num_traj", type=int, default=100, help="number of trajectories to sample")
    parser.add_argument("--sample_step", type=float, default=0.3, help="trajectory resample step (meters)")
    parser.add_argument("--yaw_step_deg", type=float, default=5.0, help="yaw interpolation step for in-place turning (deg)")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dump_rrt_mesh", action="store_true",
                    help="export the CROPPED mesh used by RRT for each traj")
    parser.add_argument("--dump_rrt_mesh_world", action="store_true",
                        help="also export a WORLD-coordinate version (vertices + DJI_OFFSET)")

    # output
    parser.add_argument("--max_tries_per_traj", type=int, default=50,
                        help="max resampling attempts for each successful trajectory")
    parser.add_argument("--min_start_goal_dist", type=float, default=3.0,
                        help="skip if ||start-goal|| is smaller than this (meters)")
    parser.add_argument("--out_dir", type=str, default=None, help="override output directory")

    def apply_env_sampling_defaults(args):
        cfg = ENV_CONFIGS[args.env_id]
        sdef = cfg["start_rect"]
        gdef = cfg["goal_rect"]

        if args.start_node1 is None:
            args.start_node1 = np.array(sdef["node1"], dtype=np.float64)
        if args.start_node2 is None:
            args.start_node2 = np.array(sdef["node2"], dtype=np.float64)
        if args.start_height_l is None:
            args.start_height_l = float(sdef["h_l"])
        if args.start_height_h is None:
            args.start_height_h = float(sdef["h_h"])

        if args.goal_node1 is None:
            args.goal_node1 = np.array(gdef["node1"], dtype=np.float64)
        if args.goal_node2 is None:
            args.goal_node2 = np.array(gdef["node2"], dtype=np.float64)
        if args.goal_height_l is None:
            args.goal_height_l = float(gdef["h_l"])
        if args.goal_height_h is None:
            args.goal_height_h = float(gdef["h_h"])

    args = parser.parse_args()
    apply_env_sampling_defaults(args)

    env_cfg = resolve_env_paths_and_defaults(args.env_id)

    mesh_path = args.mesh_path or str(env_cfg["mesh_path"])
    xml_path = args.xml_path or str(env_cfg["xml_path"])
    metadata_path = args.metadata_path or str(env_cfg["metadata_path"])
    out_dir = Path(args.out_dir) if args.out_dir is not None else env_cfg["out_dir"]

    log(f"[INFO] env_id={args.env_id}")
    log(f"[INFO] data_path={env_cfg['data_path']}")
    log(f"[INFO] mesh_path={mesh_path}")
    log(f"[INFO] xml_path={xml_path}")
    log(f"[INFO] metadata_path={metadata_path}")
    log(f"[INFO] out_dir={out_dir}")
    log(f"[INFO] sample_step={args.sample_step}, yaw_step_deg={args.yaw_step_deg}, seed={args.seed}")

    dji_offset = parse_dji_offset_from_metadata(metadata_path if os.path.isfile(metadata_path) else None)
    intr = parse_intrinsics_from_xml(xml_path if os.path.isfile(xml_path) else None)
    cameras_xml = parse_all_cameras_from_xml(xml_path if os.path.isfile(xml_path) else None)

    rng = np.random.default_rng(int(args.seed))

    # Prepare nodes/heights in REL
    start_n1_xy = args.start_node1.astype(np.float64)
    start_n2_xy = args.start_node2.astype(np.float64)
    goal_n1_xy  = args.goal_node1.astype(np.float64)
    goal_n2_xy  = args.goal_node2.astype(np.float64)

    start_hl = float(args.start_height_l)
    start_hh = float(args.start_height_h)
    goal_hl  = float(args.goal_height_l)
    goal_hh  = float(args.goal_height_h)

    if args.input_world:
        # nodes are WORLD XY -> convert to REL XY
        start_n1_xy = start_n1_xy - dji_offset[:2]
        start_n2_xy = start_n2_xy - dji_offset[:2]
        goal_n1_xy  = goal_n1_xy  - dji_offset[:2]
        goal_n2_xy  = goal_n2_xy  - dji_offset[:2]

        # heights are WORLD Z -> convert to REL Z
        dz = float(dji_offset[2])
        start_hl -= dz
        start_hh -= dz
        goal_hl  -= dz
        goal_hh  -= dz

    base_out_dir = out_dir
    # ==========================
    # Generate merged trajectories
    # ==========================
    target_num = int(args.num_traj)
    max_tries = int(args.max_tries_per_traj)
    min_sg = float(args.min_start_goal_dist)
    max_attempt = target_num * max_tries

    poses_all: List[Dict[str, Any]] = []
    traj_infos: List[Dict[str, Any]] = []

    success = 0
    attempt = 0
    pose_id_global = 0

    # (optional) debug mesh dump folder
    rrt_dump_dir = out_dir / "rrt_mesh"
    if args.dump_rrt_mesh:
        rrt_dump_dir.mkdir(parents=True, exist_ok=True)

    def _cleanup_tmp(p: Optional[Path]):
        if p is None:
            return
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    while success < target_num and attempt < max_attempt:
        seed_i = int(args.seed) + attempt
        rng = np.random.default_rng(seed_i)

        # 1) sample start/goal
        start_rel = sample_point_in_vertical_rect_xy(start_n1_xy, start_n2_xy, start_hl, start_hh, rng)
        goal_rel  = sample_point_in_vertical_rect_xy(goal_n1_xy,  goal_n2_xy,  goal_hl,  goal_hh,  rng)
        init_dir_xy = perpendicular_dir_to_vertical_plane_xy_toward_goal(
                start_n1_xy, start_n2_xy,
                start_rel[:2], goal_rel[:2],
                rng
            )

        sg_dist = float(np.linalg.norm(goal_rel - start_rel))
        if sg_dist < min_sg:
            log(f"[WARN] attempt={attempt} seed={seed_i} skip: start-goal too close (dist={sg_dist:.3f} < {min_sg})")
            attempt += 1
            continue

        log(f"[INFO] success_id={success} attempt={attempt} seed={seed_i}")
        log(f"[INFO] sampled start_rel={start_rel.tolist()}")
        log(f"[INFO] sampled goal_rel ={goal_rel.tolist()}")

        # temp debug mesh paths (only keep on success)
        tmp_rel = (rrt_dump_dir / "_tmp_rrt_rel.ply") if args.dump_rrt_mesh else None
        tmp_world = (rrt_dump_dir / "_tmp_rrt_world.ply") if (args.dump_rrt_mesh and args.dump_rrt_mesh_world) else None

        try:
            # 2) plan
            path_rel = plan_obstacle_avoidance_path(
                start=start_rel,
                goal=goal_rel,
                mesh=mesh_path,
                seed=seed_i,
                resample_step=float(args.sample_step),
                debug_mesh_out_rel=tmp_rel,
                debug_mesh_out_world=tmp_world,
                dji_offset=dji_offset,
            )

            # 3) path -> poses (ids start at 0 here)
            poses_i, ranges_i = path_to_horizontal_poses(
                path_rel,
                yaw_step_deg=float(args.yaw_step_deg),
                init_dir_xy=init_dir_xy,
            )

        except (ValueError, RuntimeError) as e:
            log(f"[WARN] attempt={attempt} seed={seed_i} skip due to error: {repr(e)}")
            # remove temp dump if any
            _cleanup_tmp(tmp_rel)
            _cleanup_tmp(tmp_world)
            attempt += 1
            continue

        # keep debug mesh dump only for successful traj (rename tmp -> final)
        if args.dump_rrt_mesh and tmp_rel is not None and tmp_rel.exists():
            final_rel = rrt_dump_dir / f"rrt_mesh_rel_traj_{success:05d}.ply"
            if final_rel.exists():
                final_rel.unlink()
            tmp_rel.rename(final_rel)
            log(f"[INFO] Kept RRT mesh (REL): {final_rel}")

        if args.dump_rrt_mesh and args.dump_rrt_mesh_world and tmp_world is not None and tmp_world.exists():
            final_world = rrt_dump_dir / f"rrt_mesh_world_traj_{success:05d}.ply"
            if final_world.exists():
                final_world.unlink()
            tmp_world.rename(final_world)
            log(f"[INFO] Kept RRT mesh (WORLD): {final_world}")

        # 4) offset pose ids to global increasing ids
        offset = int(pose_id_global)

        for p in poses_i:
            p["id"] = int(p["id"]) + offset

        def _off(v: int) -> int:
            # ranges may contain -1 (no turn)
            return (int(v) + offset) if int(v) >= 0 else int(v)

        ranges_i = {k: _off(v) for k, v in ranges_i.items()}

        pose_start = int(poses_i[0]["id"])
        pose_end = int(poses_i[-1]["id"])

        poses_all.extend(poses_i)

        traj_infos.append(
            {
                "traj_id": int(success),
                "pose_id_start": pose_start,
                "pose_id_end": pose_end,
                "turn_start": int(ranges_i["turn_start"]),
                "turn_end": int(ranges_i["turn_end"]),
                "fly_start": int(ranges_i["fly_start"]),
                "fly_end": int(ranges_i["fly_end"]),
            }
        )

        pose_id_global = pose_end + 1
        success += 1
        attempt += 1

    if success < target_num:
        log(f"[WARN] Only generated {success}/{target_num} trajectories after {attempt} attempts (max={max_attempt}).")
    else:
        log("[INFO] Done generating trajectories.")

    # ==========================
    # Write merged outputs once
    # ==========================
    if success > 0:
        write_outputs_merged(
            out_dir=out_dir,
            poses_all=poses_all,
            traj_infos=traj_infos,
            dji_offset=dji_offset,
            intr=intr,
            cameras_xml=cameras_xml,
            env_id=args.env_id,
            sample_step=float(args.sample_step),
            yaw_step_deg=float(args.yaw_step_deg),
        )
    else:
        log("[ERROR] No successful trajectories generated; nothing to write.")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData


def ply_bounds(path: Path):
    ply = PlyData.read(path, mmap=True)
    if "vertex" not in ply:
        raise RuntimeError(f"{path} does not contain a vertex element")
    vertices = ply["vertex"].data
    xs = np.asarray(vertices["x"])
    ys = np.asarray(vertices["y"])
    zs = np.asarray(vertices["z"])
    return np.array([xs.min(), ys.min(), zs.min()], dtype=np.float64), np.array([xs.max(), ys.max(), zs.max()], dtype=np.float64)


def obj_bounds(path: Path):
    min_xyz = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    max_xyz = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    vertex_count = 0

    with path.open("r", encoding="utf-8", errors="ignore") as obj_file:
        for raw_line in obj_file:
            if not raw_line.startswith("v "):
                continue
            parts = raw_line.split()
            if len(parts) < 4:
                continue
            xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
            min_xyz = np.minimum(min_xyz, xyz)
            max_xyz = np.maximum(max_xyz, xyz)
            vertex_count += 1

    if vertex_count == 0:
        raise RuntimeError(f"{path} does not contain OBJ vertices")
    return min_xyz, max_xyz


def aabb_intersection(min_a, max_a, min_b, max_b):
    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)
    extent = np.maximum(inter_max - inter_min, 0.0)
    return inter_min, inter_max, extent


def volume(extent):
    return float(np.prod(np.maximum(extent, 0.0)))


def format_vec(vec):
    return "[" + ", ".join(f"{v:.3f}" for v in vec) + "]"


def main():
    parser = argparse.ArgumentParser(description="Print AABB diagnostics for an aligned 3DGS point cloud and mesh")
    parser.add_argument("--point-cloud", type=Path, required=True, help="Aligned 3DGS PLY")
    parser.add_argument("--mesh", type=Path, required=True, help="Aligned mesh OBJ")
    args = parser.parse_args()

    pc_min, pc_max = ply_bounds(args.point_cloud)
    mesh_min, mesh_max = obj_bounds(args.mesh)

    pc_extent = pc_max - pc_min
    mesh_extent = mesh_max - mesh_min
    pc_center = 0.5 * (pc_min + pc_max)
    mesh_center = 0.5 * (mesh_min + mesh_max)
    center_delta = pc_center - mesh_center

    inter_min, inter_max, inter_extent = aabb_intersection(pc_min, pc_max, mesh_min, mesh_max)
    inter_volume = volume(inter_extent)
    pc_volume = max(volume(pc_extent), 1e-9)
    mesh_volume = max(volume(mesh_extent), 1e-9)

    print("[POINT_CLOUD]")
    print(f"min    {format_vec(pc_min)}")
    print(f"max    {format_vec(pc_max)}")
    print(f"extent {format_vec(pc_extent)}")
    print(f"center {format_vec(pc_center)}")

    print("[MESH]")
    print(f"min    {format_vec(mesh_min)}")
    print(f"max    {format_vec(mesh_max)}")
    print(f"extent {format_vec(mesh_extent)}")
    print(f"center {format_vec(mesh_center)}")

    print("[ALIGNMENT]")
    print(f"center_delta_point_cloud_minus_mesh {format_vec(center_delta)}")
    print(f"intersection_min {format_vec(inter_min)}")
    print(f"intersection_max {format_vec(inter_max)}")
    print(f"intersection_extent {format_vec(inter_extent)}")
    print(f"intersection_over_point_cloud_aabb {inter_volume / pc_volume:.6f}")
    print(f"intersection_over_mesh_aabb {inter_volume / mesh_volume:.6f}")

    if inter_volume <= 0.0:
        raise SystemExit("[FAIL] AABBs do not overlap. Check CRS, metadata, and offsets.")

    print("[OK] AABBs overlap")


if __name__ == "__main__":
    main()

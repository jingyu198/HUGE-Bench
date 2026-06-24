import argparse
from pathlib import Path
from typing import Optional

import open3d as o3d


def clean_mesh(mesh):
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    return mesh


def simplify_mesh(input_path: Path, output_path: Path, ratio: float, target_triangles: Optional[int]):
    mesh = o3d.io.read_triangle_mesh(str(input_path))
    if mesh.is_empty():
        raise RuntimeError(f"Failed to read mesh or mesh is empty: {input_path}")

    mesh = clean_mesh(mesh)
    triangle_count = len(mesh.triangles)
    if triangle_count == 0:
        raise RuntimeError("The input mesh contains no triangles")

    if target_triangles is None:
        if not (0.0 < ratio <= 1.0):
            raise ValueError("--ratio must be in (0, 1]")
        target_triangles = max(4, int(triangle_count * ratio))

    target_triangles = max(4, min(int(target_triangles), triangle_count))
    print(f"[INFO] input triangles={triangle_count}")
    print(f"[INFO] target triangles={target_triangles}")

    if target_triangles < triangle_count:
        simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
        simplified = clean_mesh(simplified)
    else:
        simplified = mesh

    simplified.compute_vertex_normals()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_triangle_mesh(str(output_path), simplified, write_triangle_uvs=True)
    if not ok:
        raise RuntimeError(f"Failed to write mesh: {output_path}")

    print(f"[OK] Wrote {output_path}")
    print(f"[OK] output triangles={len(simplified.triangles)}")


def main():
    parser = argparse.ArgumentParser(description="Simplify a mesh for collision and depth queries")
    parser.add_argument("--input", type=Path, required=True, help="Input mesh path")
    parser.add_argument("--output", type=Path, required=True, help="Output mesh path")
    parser.add_argument("--ratio", type=float, default=0.05, help="Triangle retention ratio if --target-triangles is not set")
    parser.add_argument("--target-triangles", type=int, default=None, help="Exact triangle target")
    args = parser.parse_args()

    simplify_mesh(args.input, args.output, args.ratio, args.target_triangles)


if __name__ == "__main__":
    main()

import argparse
import re
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def natural_key(value: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def find_block_plys(input_dir: Path, block_pattern: str, lod: str, filename: str):
    block_dirs = [p for p in input_dir.glob(block_pattern) if p.is_dir()]
    block_dirs.sort(key=lambda p: natural_key(p.name))

    plys = []
    for block_dir in block_dirs:
        ply_path = block_dir / lod / filename
        if ply_path.exists():
            plys.append(ply_path)
        else:
            print(f"[WARN] Missing block PLY, skipping: {ply_path}")
    return plys


def merge_plys(input_paths, output_path: Path, text: bool):
    all_vertices = []
    all_faces = []
    vertex_dtype = None
    face_dtype = None
    vertex_offset = 0
    face_index_property = None

    for ply_path in input_paths:
        print(f"[INFO] Reading {ply_path}")
        ply = PlyData.read(ply_path)

        if "vertex" not in ply:
            raise RuntimeError(f"{ply_path} does not contain a vertex element")

        vertices = ply["vertex"].data
        if vertex_dtype is None:
            vertex_dtype = vertices.dtype
        elif vertices.dtype != vertex_dtype:
            raise RuntimeError(f"Vertex dtype mismatch in {ply_path}")

        all_vertices.append(vertices)

        if "face" in ply:
            faces = ply["face"].data.copy()
            if face_dtype is None:
                face_dtype = faces.dtype
                face_index_property = "vertex_indices" if "vertex_indices" in faces.dtype.names else faces.dtype.names[0]
            elif faces.dtype != face_dtype:
                raise RuntimeError(f"Face dtype mismatch in {ply_path}")

            for idx in range(len(faces)):
                faces[face_index_property][idx] = np.asarray(faces[face_index_property][idx]) + vertex_offset
            all_faces.append(faces)

        vertex_offset += len(vertices)

    if not all_vertices:
        raise RuntimeError("No PLY vertices were found")

    merged_vertices = np.concatenate(all_vertices)
    elements = [PlyElement.describe(merged_vertices, "vertex")]

    if all_faces:
        merged_faces = np.concatenate(all_faces)
        elements.append(PlyElement.describe(merged_faces, "face"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData(elements, text=text).write(output_path)
    print(f"[OK] Wrote {output_path}")
    print(f"[OK] vertices={len(merged_vertices)} faces={sum(len(f) for f in all_faces)}")


def main():
    parser = argparse.ArgumentParser(description="Merge DJI Terra 3DGS PLY blocks into one PLY")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing Block*/LOD*/point_cloud.ply")
    parser.add_argument("--output", type=Path, required=True, help="Merged PLY output path")
    parser.add_argument("--block-pattern", default="Block*", help="Block directory glob pattern")
    parser.add_argument("--lod", default="LOD0", help="LOD directory name inside each block")
    parser.add_argument("--filename", default="point_cloud.ply", help="PLY filename inside each LOD directory")
    parser.add_argument("--ascii", action="store_true", help="Write ASCII PLY instead of binary PLY")
    args = parser.parse_args()

    input_paths = find_block_plys(args.input_dir, args.block_pattern, args.lod, args.filename)
    if not input_paths:
        raise RuntimeError(f"No block PLY files found under {args.input_dir}")

    print("[INFO] Found block PLY files:")
    for path in input_paths:
        print(f"  - {path}")

    merge_plys(input_paths, args.output, text=args.ascii)


if __name__ == "__main__":
    main()

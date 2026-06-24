import argparse
import re
from pathlib import Path


def natural_key(value: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def find_obj_files(input_dir: Path, block_pattern: str):
    block_dirs = [p for p in input_dir.glob(block_pattern) if p.is_dir()]
    block_dirs.sort(key=lambda p: natural_key(p.name))

    obj_files = []
    for block_dir in block_dirs:
        files = sorted(block_dir.glob("*.obj"), key=lambda p: natural_key(p.name))
        obj_files.extend(files)
    return obj_files


def remap_face_token(token: str, vertex_offset: int, local_vertex_count: int):
    vertex_text = token.split("/")[0]
    if not vertex_text:
        raise ValueError(f"Invalid OBJ face token: {token}")

    vertex_index = int(vertex_text)
    if vertex_index < 0:
        global_index = vertex_offset + local_vertex_count + 1 + vertex_index
    else:
        global_index = vertex_offset + vertex_index
    return str(global_index)


def merge_objs(obj_files, output_path: Path):
    vertex_offset = 0
    total_vertices = 0
    total_faces = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_file:
        out_file.write("# Merged geometry-only OBJ generated from DJI Terra blocks\n")

        for obj_path in obj_files:
            print(f"[INFO] Reading {obj_path}")
            local_vertex_count = 0
            local_face_count = 0

            with obj_path.open("r", encoding="utf-8", errors="ignore") as in_file:
                for raw_line in in_file:
                    line = raw_line.strip()
                    if not line:
                        continue

                    if line.startswith("v "):
                        out_file.write(line + "\n")
                        local_vertex_count += 1
                        total_vertices += 1
                    elif line.startswith("f "):
                        parts = line.split()
                        remapped = [
                            remap_face_token(token, vertex_offset, local_vertex_count)
                            for token in parts[1:]
                        ]
                        if len(remapped) >= 3:
                            out_file.write("f " + " ".join(remapped) + "\n")
                            local_face_count += 1
                            total_faces += 1

            print(f"[INFO] vertices={local_vertex_count} faces={local_face_count}")
            vertex_offset += local_vertex_count

    print(f"[OK] Wrote {output_path}")
    print(f"[OK] vertices={total_vertices} faces={total_faces}")


def main():
    parser = argparse.ArgumentParser(description="Merge DJI Terra OBJ blocks into one geometry-only OBJ")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing Block* OBJ folders")
    parser.add_argument("--output", type=Path, required=True, help="Merged OBJ output path")
    parser.add_argument("--block-pattern", default="Block*", help="Block directory glob pattern")
    args = parser.parse_args()

    obj_files = find_obj_files(args.input_dir, args.block_pattern)
    if not obj_files:
        raise RuntimeError(f"No OBJ files found under {args.input_dir}")

    print("[INFO] Found OBJ files:")
    for path in obj_files:
        print(f"  - {path}")

    merge_objs(obj_files, args.output)


if __name__ == "__main__":
    main()

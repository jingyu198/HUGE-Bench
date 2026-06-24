#!/usr/bin/env bash
set -euo pipefail

SCENE_ROOT="${SCENE_ROOT:-/path/to/scene_export}"

python aligned_3dgs_mesh/scripts/merge_3dgs_blocks.py \
  --input-dir "$SCENE_ROOT/3dgs_ply" \
  --output "$SCENE_ROOT/3dgs_ply/merged_3dgs.ply"

python aligned_3dgs_mesh/scripts/merge_terra_mesh_blocks.py \
  --input-dir "$SCENE_ROOT/terra_ply" \
  --output "$SCENE_ROOT/terra_ply/merged_mesh.obj"

python aligned_3dgs_mesh/scripts/convert_enu_ply_to_utm.py \
  --input-ply "$SCENE_ROOT/3dgs_ply/merged_3dgs.ply" \
  --output-ply "$SCENE_ROOT/3dgs_ply/point_cloud_utm50.ply" \
  --source-metadata "$SCENE_ROOT/3dgs_ply/metadata.xml" \
  --target-metadata "$SCENE_ROOT/terra_ply/metadata.xml"

python aligned_3dgs_mesh/scripts/simplify_mesh.py \
  --input "$SCENE_ROOT/terra_ply/merged_mesh.obj" \
  --output "$SCENE_ROOT/terra_ply/simplified_mesh.obj" \
  --ratio 0.05

python aligned_3dgs_mesh/scripts/inspect_alignment.py \
  --point-cloud "$SCENE_ROOT/3dgs_ply/point_cloud_utm50.ply" \
  --mesh "$SCENE_ROOT/terra_ply/simplified_mesh.obj"

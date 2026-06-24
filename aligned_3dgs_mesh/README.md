# Aligned 3DGS-Mesh Construction

This directory contains the utility scripts used to build the aligned 3DGS-Mesh
environment representation used by HUGE-Bench. The recommended fast path is:

1. Collect raw aerial images with a DJI drone.
2. Reconstruct the scene in DJI Terra.
3. Export both the 3DGS point cloud blocks and the textured mesh blocks.
4. Convert both products into the same local metric frame.
5. Simplify the mesh for collision checks and trajectory generation.

The final environment expected by the rest of the benchmark is:

```text
data_3d/<env_id>/
├── 3dgs_ply/
│   └── point_cloud_utm50.ply
└── terra_ply/
    └── simplified_mesh.obj
```

`point_cloud_utm50.ply` is used by the 3DGS renderer. `simplified_mesh.obj` is
used for collision checks, depth rendering, and path sampling.

## Capture Notes

For quick reconstruction, use DJI waypoint or manual flights with high image
overlap. In practice, keep enough oblique views around buildings, trees, and
thin structures; pure nadir captures often produce incomplete side geometry.
Keep the original GPS/RTK metadata, avoid strong motion blur, and avoid dynamic
objects when possible.

The benchmark environments were exported from DJI Terra with a projected metric
coordinate system. For the released examples the target frame is WGS84 / UTM
zone 50N (`EPSG:32650`), but you should choose the UTM zone that matches your
scene. The important requirement is that the 3DGS point cloud and the mesh use
the same final local frame.

## DJI Terra Exports

Export two products from the same reconstruction:

```text
scene_export/
├── 3dgs_ply/
│   ├── metadata.xml
│   ├── Block000/LOD0/point_cloud.ply
│   ├── Block001/LOD0/point_cloud.ply
│   └── ...
└── terra_ply/
    ├── metadata.xml
    ├── BlockRA/*.obj
    ├── BlockRX/*.obj
    └── ...
```

The 3DGS export metadata commonly looks like `SRS=ENU:<lat>,<lon>` with
`SRSOrigin=0,0,<height>`. The mesh export metadata should contain the projected
CRS, such as `EPSG:32650`, and an `SRSOrigin` offset. The conversion script uses
these two metadata files to align the 3DGS point cloud to the mesh-local frame.

## Dependencies

Install the Python dependencies in an environment where you can process large
PLY/OBJ files:

```bash
pip install numpy plyfile pyproj open3d pandas openpyxl
```

`pandas` and `openpyxl` are only needed if you convert DJI Terra coordinate
annotation spreadsheets.

## Build Steps

Set the scene path first:

```bash
export SCENE_ROOT=/path/to/scene_export
```

Merge the DJI Terra 3DGS PLY blocks:

```bash
python aligned_3dgs_mesh/scripts/merge_3dgs_blocks.py \
  --input-dir "$SCENE_ROOT/3dgs_ply" \
  --output "$SCENE_ROOT/3dgs_ply/merged_3dgs.ply"
```

Merge the DJI Terra mesh OBJ blocks into a geometry-only collision mesh:

```bash
python aligned_3dgs_mesh/scripts/merge_terra_mesh_blocks.py \
  --input-dir "$SCENE_ROOT/terra_ply" \
  --output "$SCENE_ROOT/terra_ply/merged_mesh.obj"
```

Convert the merged 3DGS point cloud from Terra ENU coordinates to the mesh-local
UTM frame:

```bash
python aligned_3dgs_mesh/scripts/convert_enu_ply_to_utm.py \
  --input-ply "$SCENE_ROOT/3dgs_ply/merged_3dgs.ply" \
  --output-ply "$SCENE_ROOT/3dgs_ply/point_cloud_utm50.ply" \
  --source-metadata "$SCENE_ROOT/3dgs_ply/metadata.xml" \
  --target-metadata "$SCENE_ROOT/terra_ply/metadata.xml"
```

Simplify the mesh for faster collision queries:

```bash
python aligned_3dgs_mesh/scripts/simplify_mesh.py \
  --input "$SCENE_ROOT/terra_ply/merged_mesh.obj" \
  --output "$SCENE_ROOT/terra_ply/simplified_mesh.obj" \
  --ratio 0.05
```

Check that the transformed 3DGS point cloud and simplified mesh occupy the same
local coordinate frame:

```bash
python aligned_3dgs_mesh/scripts/inspect_alignment.py \
  --point-cloud "$SCENE_ROOT/3dgs_ply/point_cloud_utm50.ply" \
  --mesh "$SCENE_ROOT/terra_ply/simplified_mesh.obj"
```

If you export object, building, road, or landmark coordinates from DJI Terra as
spreadsheets, convert them to the same local frame:

```bash
python aligned_3dgs_mesh/scripts/convert_annotation_xlsx_to_local.py \
  --input-dir "$SCENE_ROOT/building_coords" \
  --output-dir "$SCENE_ROOT/building_coords" \
  --offset-metadata "$SCENE_ROOT/terra_ply/metadata.xml"
```

The output text files use:

```text
#x y z label
12.345 67.890 3.210 building_1
```

These local coordinate annotations can be used by the trajectory-generation
scripts for task landmarks and region-specific sampling.

## Common Pitfalls

- Use the same DJI Terra reconstruction when exporting 3DGS and mesh products.
  Mixing products from different reconstructions can introduce scale or rotation
  drift.
- Do not skip metadata files. They contain the ENU reference GPS coordinate and
  the projected mesh offset needed for alignment.
- Choose the correct target CRS for your location. HUGE-Bench examples use
  `EPSG:32650`; other regions need a different UTM zone.
- Keep both `merged_mesh.obj` and `simplified_mesh.obj` while debugging. The
  simplified mesh is faster, but the merged mesh is useful for checking whether
  simplification removed important collision geometry.
- Large scenes can require substantial RAM. Run the merge and simplification
  steps on a workstation or server with enough memory.

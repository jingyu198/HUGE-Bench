# Scene Annotations

This directory releases the small scene files needed by the trajectory
generation scripts. The large 3DGS point clouds and meshes are distributed in
`HUGE_Environment`; these files provide the metadata, landmark lists, and
contour annotations consumed by the generators.

The layout mirrors the expected `HUGE_DATA_ROOT/data_3d` structure:

```text
scene_annotations/
|-- data_3d/
|   |-- <env_id>/
|   |   |-- BlocksExchangeUndistortAT_WithoutTiePoints.xml
|   |   |-- 3dgs_ply/metadata.xml
|   |   |-- terra_ply/metadata.xml
|   |   |-- location_gen/landmark_merged*.txt
|   |   |-- building_coords/*.txt
|   |   |-- farm_coords/*.txt
|   |   `-- road_coords/*.txt
`-- manifest.json
```

Copy the files into the same root that contains the downloaded environment
assets:

```bash
cp -r trajectory_generation/scene_annotations/data_3d/* "$HUGE_DATA_ROOT/data_3d/"
```

The coordinate filenames and point labels have been normalized to ASCII-safe
names for portability. `manifest.json` records the original internal filename
for each released file. The `BlocksExchangeUndistortAT_WithoutTiePoints.xml`
files keep the camera intrinsics and poses but normalize image paths to
`images/<basename>`.

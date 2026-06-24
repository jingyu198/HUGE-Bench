import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from pyproj import CRS, Transformer


def parse_origin(text: str):
    values = [float(v.strip()) for v in text.split(",")]
    if len(values) != 3:
        raise ValueError(f"Expected three SRSOrigin values, got: {text}")
    return values


def read_model_metadata(path: Path):
    tree = ET.parse(path)
    root = tree.getroot()
    srs = root.findtext("SRS")
    origin_text = root.findtext("SRSOrigin")
    if srs is None:
        raise ValueError(f"{path} does not contain SRS")
    if origin_text is None:
        raise ValueError(f"{path} does not contain SRSOrigin")
    return srs.strip(), parse_origin(origin_text.strip())


def parse_enu_srs(srs: str):
    if not srs.upper().startswith("ENU:"):
        raise ValueError(f"Expected source SRS like ENU:<lat>,<lon>, got: {srs}")
    lat_lon = srs.split(":", 1)[1]
    lat_text, lon_text = [v.strip() for v in lat_lon.split(",", 1)]
    return float(lat_text), float(lon_text)


def enu_to_ecef(east, north, up, lat0_deg: float, lon0_deg: float, h0: float):
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)

    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)

    sin_lat = math.sin(lat0)
    cos_lat = math.cos(lat0)
    sin_lon = math.sin(lon0)
    cos_lon = math.cos(lon0)

    n_phi = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    x0 = (n_phi + h0) * cos_lat * cos_lon
    y0 = (n_phi + h0) * cos_lat * sin_lon
    z0 = (n_phi * (1.0 - e2) + h0) * sin_lat

    rotation = np.array(
        [
            [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
            [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
            [0.0, cos_lat, sin_lat],
        ],
        dtype=np.float64,
    )

    enu = np.stack([east, north, up], axis=0)
    delta_xyz = rotation @ enu
    return x0 + delta_xyz[0], y0 + delta_xyz[1], z0 + delta_xyz[2]


def convert_ply(input_ply: Path, output_ply: Path, lat0: float, lon0: float, h0: float, target_crs, target_offset):
    target_offset = np.asarray(target_offset, dtype=np.float64)

    print(f"[INFO] Reading ENU PLY: {input_ply}")
    plydata = PlyData.read(input_ply)
    if "vertex" not in plydata:
        raise RuntimeError(f"{input_ply} does not contain a vertex element")

    vertex = plydata["vertex"]
    vertex_data = vertex.data
    east = np.asarray(vertex_data["x"], dtype=np.float64)
    north = np.asarray(vertex_data["y"], dtype=np.float64)
    up = np.asarray(vertex_data["z"], dtype=np.float64)
    print(f"[INFO] vertices={len(vertex_data)}")

    print("[INFO] ENU -> ECEF")
    ecef_x, ecef_y, ecef_z = enu_to_ecef(east, north, up, lat0, lon0, h0)

    print(f"[INFO] ECEF -> WGS84 -> {target_crs.to_string()}")
    ecef_crs = CRS.from_epsg(4978)
    geodetic_crs = CRS.from_epsg(4326)
    ecef_to_geodetic = Transformer.from_crs(ecef_crs, geodetic_crs, always_xy=True)
    geodetic_to_target = Transformer.from_crs(geodetic_crs, target_crs, always_xy=True)

    lon, lat, height = ecef_to_geodetic.transform(ecef_x, ecef_y, ecef_z)
    target_x, target_y, target_z = geodetic_to_target.transform(lon, lat, height)

    local_x = target_x - target_offset[0]
    local_y = target_y - target_offset[1]
    local_z = target_z - target_offset[2]

    print(
        "[INFO] local bounds: "
        f"x=[{local_x.min():.3f}, {local_x.max():.3f}] "
        f"y=[{local_y.min():.3f}, {local_y.max():.3f}] "
        f"z=[{local_z.min():.3f}, {local_z.max():.3f}]"
    )

    new_vertex_data = vertex_data.copy()
    new_vertex_data["x"] = local_x.astype(np.float32)
    new_vertex_data["y"] = local_y.astype(np.float32)
    new_vertex_data["z"] = local_z.astype(np.float32)

    new_elements = []
    for element in plydata.elements:
        if element.name == "vertex":
            new_elements.append(PlyElement.describe(new_vertex_data, "vertex"))
        else:
            new_elements.append(element)

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData(new_elements, text=plydata.text).write(output_ply)
    print(f"[OK] Wrote {output_ply}")


def main():
    parser = argparse.ArgumentParser(description="Convert a DJI Terra ENU PLY to the mesh-local projected frame")
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, help="3DGS metadata.xml containing ENU:<lat>,<lon>")
    parser.add_argument("--target-metadata", type=Path, help="Mesh metadata.xml containing EPSG code and SRSOrigin")
    parser.add_argument("--lat0", type=float, help="ENU origin latitude in degrees")
    parser.add_argument("--lon0", type=float, help="ENU origin longitude in degrees")
    parser.add_argument("--h0", type=float, help="ENU origin height in meters")
    parser.add_argument("--target-crs", default=None, help="Target CRS, for example EPSG:32650")
    parser.add_argument("--target-offset", type=float, nargs=3, metavar=("X", "Y", "Z"), help="Target SRSOrigin offset")
    args = parser.parse_args()

    source_srs = None
    source_origin = None
    if args.source_metadata:
        source_srs, source_origin = read_model_metadata(args.source_metadata)

    target_srs = args.target_crs
    target_origin = args.target_offset
    if args.target_metadata:
        metadata_srs, metadata_origin = read_model_metadata(args.target_metadata)
        target_srs = target_srs or metadata_srs
        target_origin = target_origin or metadata_origin

    if args.lat0 is not None and args.lon0 is not None:
        lat0, lon0 = args.lat0, args.lon0
    elif source_srs is not None:
        lat0, lon0 = parse_enu_srs(source_srs)
    else:
        raise ValueError("Provide --source-metadata or both --lat0 and --lon0")

    if args.h0 is not None:
        h0 = args.h0
    elif source_origin is not None:
        h0 = source_origin[2]
    else:
        raise ValueError("Provide --source-metadata or --h0")

    if target_srs is None:
        raise ValueError("Provide --target-metadata or --target-crs")
    if target_origin is None:
        target_origin = [0.0, 0.0, 0.0]

    convert_ply(
        input_ply=args.input_ply,
        output_ply=args.output_ply,
        lat0=lat0,
        lon0=lon0,
        h0=h0,
        target_crs=CRS.from_user_input(target_srs),
        target_offset=target_origin,
    )


if __name__ == "__main__":
    main()

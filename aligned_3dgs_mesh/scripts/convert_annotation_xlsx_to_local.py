import argparse
import glob
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


LABEL_COLUMNS = ["label", "name", "Name", "NAME", "名称"]
X_COLUMNS = ["x", "X", "X/E", "E", "easting", "Easting"]
Y_COLUMNS = ["y", "Y", "Y/N", "N", "northing", "Northing"]
Z_COLUMNS = ["z", "Z", "Z/U", "U", "height", "Height", "altitude", "Altitude"]


def parse_origin(text: str):
    values = [float(v.strip()) for v in text.split(",")]
    if len(values) != 3:
        raise ValueError(f"Expected three SRSOrigin values, got: {text}")
    return values


def read_origin_from_metadata(path: Path):
    tree = ET.parse(path)
    root = tree.getroot()
    origin_text = root.findtext("SRSOrigin")
    if origin_text is None:
        raise ValueError(f"{path} does not contain SRSOrigin")
    return parse_origin(origin_text.strip())


def parse_sheet(value: str):
    try:
        return int(value)
    except ValueError:
        return value


def pick_column(columns, preferred, fallback):
    if preferred:
        if preferred not in columns:
            raise ValueError(f"Column {preferred!r} was requested but not found. Available columns: {columns}")
        return preferred
    for name in fallback:
        if name in columns:
            return name
    raise ValueError(f"Could not find any of these columns: {fallback}. Available columns: {columns}")


def convert_one_xlsx(xlsx_path: Path, output_path: Path, offset, args):
    df = pd.read_excel(xlsx_path, sheet_name=args.sheet, engine="openpyxl")
    df.columns = [str(column).strip() for column in df.columns]

    x_col = pick_column(df.columns, args.x_column, X_COLUMNS)
    y_col = pick_column(df.columns, args.y_column, Y_COLUMNS)
    z_col = pick_column(df.columns, args.z_column, Z_COLUMNS)
    label_col = None
    try:
        label_col = pick_column(df.columns, args.label_column, LABEL_COLUMNS)
    except ValueError:
        if args.label_column:
            raise

    sub = df.copy()
    for col in [x_col, y_col, z_col]:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna(subset=[x_col, y_col, z_col])

    ox, oy, oz = offset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_file:
        out_file.write("#x y z label\n")
        for row_index, row in sub.iterrows():
            label = str(row[label_col]).strip() if label_col else f"{xlsx_path.stem}_{row_index}"
            x = float(row[x_col]) - ox
            y = float(row[y_col]) - oy
            z = float(row[z_col]) - oz
            out_file.write(f"{x:.3f} {y:.3f} {z:.3f} {label}\n")


def main():
    parser = argparse.ArgumentParser(description="Convert DJI Terra coordinate XLSX annotations to local xyz label TXT")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing .xlsx annotation files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for converted .txt files")
    parser.add_argument("--offset-metadata", type=Path, help="Mesh metadata.xml containing SRSOrigin")
    parser.add_argument("--offset", type=float, nargs=3, metavar=("X", "Y", "Z"), help="Manual target SRSOrigin")
    parser.add_argument("--pattern", default="*.xlsx", help="Input spreadsheet glob pattern")
    parser.add_argument("--sheet", type=parse_sheet, default=0, help="Sheet name or index for pandas.read_excel")
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--x-column", default=None)
    parser.add_argument("--y-column", default=None)
    parser.add_argument("--z-column", default=None)
    args = parser.parse_args()

    if args.offset is not None:
        offset = args.offset
    elif args.offset_metadata is not None:
        offset = read_origin_from_metadata(args.offset_metadata)
    else:
        raise ValueError("Provide --offset-metadata or --offset X Y Z")

    xlsx_paths = sorted(Path(p) for p in glob.glob(str(args.input_dir / args.pattern)))
    if not xlsx_paths:
        raise FileNotFoundError(f"No XLSX files found under {args.input_dir} with pattern {args.pattern}")

    success = 0
    failed = 0
    for xlsx_path in xlsx_paths:
        output_path = args.output_dir / f"{xlsx_path.stem}.txt"
        try:
            convert_one_xlsx(xlsx_path, output_path, offset, args)
            success += 1
            print(f"[OK] {xlsx_path.name} -> {output_path}")
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {xlsx_path.name}: {exc}")

    print(f"[DONE] success={success} failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

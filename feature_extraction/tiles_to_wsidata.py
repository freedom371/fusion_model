from __future__ import annotations

import json
import re
import shutil
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TILE_ROOT = PROJECT_ROOT / "data" / "carcinoma" / "WSI"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "carcinoma" / "converted_wsidata"


# Edit the values in this block directly.
CONFIG = {
    "tile_root": DEFAULT_TILE_ROOT,
    "output_root": DEFAULT_OUTPUT_ROOT,
    "case_filters": None,
    "class_filters": None,
    "limit_cases": None,
    "build_bigtiff": True,
    "build_wsidata": True,
    "reader": "tiffslide",
    "wsidata_format": "v01",
    "skip_existing": True,
    "save_thumbnail": True,
    "save_manifest": True,
}


FILENAME_RE = re.compile(
    r"^(?P<slide_id>[^_]+)_(?P<width>\d+)_(?P<height>\d+)_(?P<x0>\d+)_(?P<y0>\d+)_(?P<x1>\d+)_(?P<y1>\d+)\.jpg$",
    re.IGNORECASE,
)


def iter_case_dirs(tile_root: Path):
    immediate_dirs = sorted(p for p in tile_root.iterdir() if p.is_dir())
    if not immediate_dirs:
        return

    sample = immediate_dirs[0]
    is_flat = any(sample.glob("*.jpg")) or any(sample.glob("*.png"))

    if is_flat:
        for case_dir in immediate_dirs:
            yield "_default", case_dir.name, case_dir
    else:
        for class_dir in immediate_dirs:
            for case_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
                yield class_dir.name, case_dir.name, case_dir


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def resolve_wsidata_format(format_name: str):
    if format_name == "v01":
        from spatialdata._io.format import SpatialDataContainerFormatV01

        return SpatialDataContainerFormatV01()
    if format_name == "latest":
        return None
    raise ValueError(f"Unsupported wsidata_format: {format_name}")


def wsidata_complete_marker(zarr_path: Path) -> Path:
    return zarr_path.with_name(f"{zarr_path.name}.complete")


def wsidata_is_complete(zarr_path: Path) -> bool:
    return zarr_path.exists() and wsidata_complete_marker(zarr_path).exists()


def normalize_chunked_raster(raster):
    data = getattr(raster, "data", None)
    chunks = getattr(data, "chunks", None)
    chunksize = getattr(data, "chunksize", None)
    dims = getattr(raster, "dims", None)
    if chunks is None or chunksize is None or dims is None:
        return raster
    if any(isinstance(dim_chunks, tuple) for dim_chunks in chunks):
        chunk_map = {dim: int(size) for dim, size in zip(dims, chunksize)}
        return raster.chunk(chunk_map)
    return raster


def flatten_chunks(chunks) -> tuple[int, ...]:
    flat = []
    for dim_chunks in chunks:
        if isinstance(dim_chunks, tuple):
            flat.append(int(dim_chunks[0]))
        else:
            flat.append(int(dim_chunks))
    return tuple(flat)


@contextmanager
def patched_spatialdata_chunk_writer():
    from spatialdata._io import io_raster

    original = io_raster._write_raster_dataarray

    def wrapped(
        raster_type,
        group,
        element_name,
        raster_data,
        raster_format,
        storage_options=None,
        **metadata,
    ):
        if storage_options is None:
            storage_options = {}
        elif isinstance(storage_options, dict):
            storage_options = dict(storage_options)

        chunks = getattr(raster_data, "chunks", None)
        if chunks is not None and isinstance(storage_options, dict):
            storage_options["chunks"] = flatten_chunks(chunks)

        return original(
            raster_type,
            group,
            element_name,
            raster_data,
            raster_format,
            storage_options=storage_options,
            **metadata,
        )

    io_raster._write_raster_dataarray = wrapped
    try:
        yield
    finally:
        io_raster._write_raster_dataarray = original


def case_allowed(class_name: str, case_name: str, config: dict) -> bool:
    class_filters = config["class_filters"]
    if class_filters is not None and class_name not in set(class_filters):
        return False
    case_filters = config["case_filters"]
    if case_filters is not None:
        allowed = {str(x) for x in case_filters}
        if case_name not in allowed and f"{class_name}/{case_name}" not in allowed:
            return False
    return True


def parse_tile_name(tile_path: Path) -> dict:
    match = FILENAME_RE.match(tile_path.name)
    if not match:
        raise ValueError(f"Unexpected tile filename format: {tile_path.name}")
    data = match.groupdict()
    for key in ("width", "height", "x0", "y0", "x1", "y1"):
        data[key] = int(data[key])
    return data


def inspect_case(case_dir: Path) -> tuple[pd.DataFrame, dict]:
    tile_paths = sorted(case_dir.glob("*.jpg"))
    if not tile_paths:
        raise ValueError(f"No jpg tiles found in {case_dir}")

    records = []
    for tile_path in tile_paths:
        info = parse_tile_name(tile_path)
        info["path"] = str(tile_path)
        records.append(info)

    df = pd.DataFrame(records)
    widths = df["width"].unique()
    heights = df["height"].unique()
    if len(widths) != 1 or len(heights) != 1:
        raise ValueError(f"Inconsistent canvas size in {case_dir}")

    sample_image = Image.open(tile_paths[0])
    tile_width, tile_height = sample_image.size
    channels = len(sample_image.getbands())
    sample_image.close()

    summary = {
        "canvas_width": int(widths[0]),
        "canvas_height": int(heights[0]),
        "tile_count": int(len(df)),
        "tile_width": int(tile_width),
        "tile_height": int(tile_height),
        "channels": int(channels),
    }
    return df, summary


def write_metadata(case_output_dir: Path, class_name: str, case_name: str, df: pd.DataFrame, summary: dict) -> Path:
    case_output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = case_output_dir / "tiles.csv"

    out_df = df.copy()
    out_df.insert(0, "class_name", class_name)
    out_df.insert(1, "case_name", case_name)
    out_df.to_csv(metadata_path, index=False)

    summary_path = case_output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata_path


def stitch_case_to_bigtiff(case_dir: Path, case_output_dir: Path, df: pd.DataFrame, summary: dict) -> Path:
    bigtiff_path = case_output_dir / f"{case_dir.name}.tif"
    if bigtiff_path.exists():
        return bigtiff_path

    shape = (summary["canvas_height"], summary["canvas_width"], summary["channels"])
    canvas = tifffile.memmap(
        bigtiff_path,
        shape=shape,
        dtype=np.uint8,
        photometric="rgb",
        bigtiff=True,
    )
    canvas[:] = 255

    for idx, row in enumerate(df.itertuples(index=False), 1):
        image = np.asarray(Image.open(row.path).convert("RGB"), dtype=np.uint8)
        canvas[row.y0 : row.y1, row.x0 : row.x1, :] = image
        if idx % 500 == 0:
            canvas.flush()
            print(f"    stitched {idx}/{summary['tile_count']} tiles")

    canvas.flush()
    return bigtiff_path


def build_wsidata(
    bigtiff_path: Path,
    case_output_dir: Path,
    reader: str,
    save_thumbnail: bool,
    wsidata_format: str,
) -> Path:
    try:
        import lazyslide as zs
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "lazyslide is required when build_wsidata=True. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        ) from exc

    zarr_path = case_output_dir / f"{bigtiff_path.stem}.zarr"
    marker_path = wsidata_complete_marker(zarr_path)
    if wsidata_is_complete(zarr_path):
        return zarr_path
    if zarr_path.exists():
        shutil.rmtree(zarr_path, ignore_errors=True)
    marker_path.unlink(missing_ok=True)

    try:
        wsi = zs.open_wsi(
            bigtiff_path,
            store=str(zarr_path),
            reader=reader,
            attach_thumbnail=True,
            save_thumbnail=save_thumbnail,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Configured reader '{reader}' is not installed. "
            f"Install it with: python -m pip install {reader}"
        ) from exc

    for image_key, image in list(wsi.images.items()):
        wsi.images[image_key] = normalize_chunked_raster(image)
    for label_key, label in list(wsi.labels.items()):
        wsi.labels[label_key] = normalize_chunked_raster(label)

    with patched_spatialdata_chunk_writer():
        wsi.write(
            zarr_path,
            overwrite=True,
            format=resolve_wsidata_format(wsidata_format),
        )
    marker_path.write_text("ok\n", encoding="utf-8")
    return zarr_path


def main() -> None:
    config = dict(CONFIG)
    config["tile_root"] = resolve_path(config["tile_root"])
    config["output_root"] = resolve_path(config["output_root"])

    tile_root = config["tile_root"]
    output_root = config["output_root"]
    if not tile_root.exists():
        raise FileNotFoundError(f"tile_root does not exist: {tile_root}")
    if not tile_root.is_dir():
        raise NotADirectoryError(f"tile_root is not a directory: {tile_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    processed = 0

    for class_name, case_name, case_dir in iter_case_dirs(tile_root):
        if not case_allowed(class_name, case_name, config):
            continue
        if config["limit_cases"] is not None and processed >= int(config["limit_cases"]):
            break

        print(f"\nProcessing {class_name}/{case_name}")
        case_output_dir = output_root / class_name / case_name
        case_output_dir.mkdir(parents=True, exist_ok=True)

        df, summary = inspect_case(case_dir)
        metadata_path = write_metadata(case_output_dir, class_name, case_name, df, summary)

        bigtiff_path = case_output_dir / f"{case_name}.tif"
        zarr_path = case_output_dir / f"{case_name}.zarr"

        if config["skip_existing"] and bigtiff_path.exists() and (
            not config["build_wsidata"] or wsidata_is_complete(zarr_path)
        ):
            print("  existing outputs found, skipping")
        else:
            if config["build_bigtiff"]:
                bigtiff_path = stitch_case_to_bigtiff(case_dir, case_output_dir, df, summary)
                print(f"  bigtiff: {bigtiff_path}")
            if config["build_wsidata"]:
                if not bigtiff_path.exists():
                    raise FileNotFoundError(f"BigTIFF not found: {bigtiff_path}")
                zarr_path = build_wsidata(
                    bigtiff_path=bigtiff_path,
                    case_output_dir=case_output_dir,
                    reader=config["reader"],
                    save_thumbnail=config["save_thumbnail"],
                    wsidata_format=config["wsidata_format"],
                )
                print(f"  wsidata: {zarr_path}")

        manifest_rows.append(
            {
                "class_name": class_name,
                "case_name": case_name,
                "case_dir": str(case_dir),
                "tile_count": summary["tile_count"],
                "canvas_width": summary["canvas_width"],
                "canvas_height": summary["canvas_height"],
                "tile_width": summary["tile_width"],
                "tile_height": summary["tile_height"],
                "tiles_csv": str(metadata_path),
                "bigtiff_path": str(bigtiff_path) if bigtiff_path.exists() else None,
                "wsidata_path": str(zarr_path) if zarr_path.exists() else None,
            }
        )
        processed += 1

    if config["save_manifest"]:
        manifest = pd.DataFrame(manifest_rows)
        manifest_path = output_root / "manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        print(f"\nManifest written to: {manifest_path}")


if __name__ == "__main__":
    main()

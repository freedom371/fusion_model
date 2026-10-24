from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from demo_lazyslide_uni2 import (
    extract_features_from_tile_paths,
    get_model_name,
    is_cuda_oom_error,
    release_case_resources,
    resolve_device,
    resolve_hf_token,
    resolve_model_path,
    slugify,
)
from tiles_to_wsidata import (
    case_allowed,
    inspect_case,
    iter_case_dirs,
)


PROJECT_ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger("batch_tiles_to_uni2_temp")


CONFIG = {
    "tile_root": Path("/hard0/carcinoma/jiandu/yansebiaozhunhua"),
    "feature_root": PROJECT_ROOT / "data" / "uni2_feature" / "ruojiandu",
    "model": "uni2",
    "model_path": PROJECT_ROOT / "uni2",
    "hf_token": None,
    "device": "auto",
    "batch_size": 8,
    "num_workers": 2,
    "min_batch_size": 1,
    "skip_existing": True,
    "case_filters": None,
    "class_filters": None,
    "limit_cases": None,
    "continue_on_error": True,
    "save_manifest": True,
    "cleanup_cuda_between_cases": True,
    "thumbnail_grid_size": 10,
}


def setup_logging() -> None:
    if logging.getLogger().handlers:
        logging.getLogger().setLevel(logging.INFO)
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remaining:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {remaining:.1f}s"


def resolve_input_path(path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def build_feature_model(model_name: Any, model_path: Path | None, hf_token: str | None):
    if not isinstance(model_name, str):
        return model_name
    from lazyslide.models import MODEL_REGISTRY
    import inspect

    if model_name not in MODEL_REGISTRY:
        return model_name

    model_cls = MODEL_REGISTRY[model_name]
    signature = inspect.signature(model_cls)
    kwargs = {}
    if "model_path" in signature.parameters and model_path is not None:
        kwargs["model_path"] = model_path
    if "token" in signature.parameters and hf_token is not None:
        kwargs["token"] = hf_token
    return model_cls(**kwargs)


def create_thumbnail_from_tiles(tiles_df: pd.DataFrame, output_dir: Path, case_name: str, grid_size: int = 10) -> Path | None:
    """Stitch a subset of tiles into a thumbnail image."""
    try:
        tile_paths = tiles_df["path"].tolist()
        n = min(grid_size * grid_size, len(tile_paths))
        if n == 0:
            return None
        indices = np.linspace(0, len(tile_paths) - 1, n, dtype=int)
        images = []
        for idx in indices:
            img = Image.open(tile_paths[idx]).convert("RGB")
            images.append(img)

        tile_w, tile_h = images[0].size
        grid_cols = min(grid_size, len(images))
        grid_rows = (len(images) + grid_cols - 1) // grid_cols
        canvas = Image.new("RGB", (grid_cols * tile_w, grid_rows * tile_h), (255, 255, 255))
        for i, img in enumerate(images):
            row, col = divmod(i, grid_cols)
            canvas.paste(img, (col * tile_w, row * tile_h))
            img.close()

        thumbnail_path = output_dir / f"{case_name}_thumbnail.png"
        canvas.save(thumbnail_path)
        return thumbnail_path
    except Exception:
        LOGGER.debug("Failed to create thumbnail for %s", case_name, exc_info=True)
        return None


def process_case(
    config: dict[str, Any],
    class_name: str,
    case_name: str,
    case_dir: Path,
    feature_model: Any,
    feature_tag: str,
    feature_table_key: str,
    device: str,
) -> dict[str, Any]:
    case_started = time.perf_counter()
    started_at = now_iso()
    case_label = f"{class_name}/{case_name}"

    case_feature_dir = config["feature_root"] / class_name / case_name
    case_feature_dir.mkdir(parents=True, exist_ok=True)

    tile_feature_path = case_feature_dir / f"{case_name}_{feature_tag}_tile_features.npy"
    slide_feature_path = case_feature_dir / f"{case_name}_{feature_tag}_slide_features.npy"
    tile_meta_path = case_feature_dir / "tile_metadata.csv"
    summary_path = case_feature_dir / "summary.json"

    if config["skip_existing"] and tile_feature_path.exists() and slide_feature_path.exists() and summary_path.exists():
        LOGGER.info("Skipping %s: feature outputs already exist.", case_label)
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        existing["status"] = "skipped"
        return existing

    tiles_df = None

    try:
        LOGGER.info("=== Processing %s ===", case_label)

        LOGGER.info("[1/3] Scanning JPG tiles")
        tiles_df, tile_summary = inspect_case(case_dir)
        tile_paths = tiles_df["path"].tolist()
        tile_count = len(tile_paths)
        LOGGER.info("Found %d tiles (canvas: %dx%d)", tile_count, tile_summary["canvas_width"], tile_summary["canvas_height"])

        LOGGER.info("[2/3] Extracting UNI2 features")
        current_batch_size = int(config["batch_size"])
        while True:
            try:
                tile_features = extract_features_from_tile_paths(
                    tile_paths=tile_paths,
                    feature_model=feature_model,
                    device=device,
                    batch_size=current_batch_size,
                    num_workers=config["num_workers"],
                )
                break
            except RuntimeError as exc:
                if not is_cuda_oom_error(exc) or current_batch_size <= int(config["min_batch_size"]):
                    raise
                next_batch_size = max(int(config["min_batch_size"]), current_batch_size // 2)
                LOGGER.warning(
                    "CUDA OOM at %s with batch_size=%s, retrying with batch_size=%s",
                    case_label,
                    current_batch_size,
                    next_batch_size,
                )
                current_batch_size = next_batch_size
                release_case_resources(device, config["cleanup_cuda_between_cases"])

        LOGGER.info("[3/3] Aggregating and exporting outputs")
        slide_features = tile_features.mean(axis=0, keepdims=True)

        np.save(tile_feature_path, tile_features)
        np.save(slide_feature_path, slide_features)

        meta = tiles_df[["x0", "y0", "x1", "y1"]].copy()
        meta.insert(0, "tile_index", meta.index.to_numpy())
        if "slide_id" in tiles_df.columns:
            meta.insert(0, "slide_id", tiles_df["slide_id"].to_numpy())
        meta.to_csv(tile_meta_path, index=False)

        thumbnail_path = create_thumbnail_from_tiles(
            tiles_df, case_feature_dir, case_name, config.get("thumbnail_grid_size", 10)
        )

        elapsed_seconds = time.perf_counter() - case_started
        summary = {
            "status": "completed",
            "started_at": started_at,
            "completed_at": now_iso(),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "case_label": case_label,
            "class_name": class_name,
            "case_name": case_name,
            "model": get_model_name(config["model"]),
            "device": device,
            "tile_count": int(tile_features.shape[0]),
            "tile_feature_dim": int(tile_features.shape[1]),
            "slide_feature_shape": list(slide_features.shape),
            "batch_size": current_batch_size,
            "tile_source_dir": str(case_dir),
            "tile_features_path": str(tile_feature_path),
            "slide_features_path": str(slide_feature_path),
            "tile_metadata_path": str(tile_meta_path),
            "thumbnail_path": str(thumbnail_path) if thumbnail_path else None,
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("Completed %s in %s", case_label, format_seconds(elapsed_seconds))
        return summary
    except Exception as exc:
        if not config["continue_on_error"]:
            raise
        LOGGER.exception("Failed on %s", case_label)
        return {
            "status": "failed",
            "started_at": started_at,
            "case_label": case_label,
            "class_name": class_name,
            "case_name": case_name,
            "tile_source_dir": str(case_dir),
            "error": str(exc),
        }
    finally:
        release_case_resources(device, config["cleanup_cuda_between_cases"], tiles_df)


def main() -> None:
    setup_logging()
    config = dict(CONFIG)
    config["tile_root"] = resolve_input_path(config["tile_root"])
    config["feature_root"] = resolve_input_path(config["feature_root"])
    config["model_path"] = resolve_model_path(config["model"], config["model_path"])

    if config["tile_root"] is None or not config["tile_root"].exists():
        raise FileNotFoundError(f"tile_root does not exist: {config['tile_root']}")
    if not config["tile_root"].is_dir():
        raise NotADirectoryError(f"tile_root is not a directory: {config['tile_root']}")

    config["feature_root"].mkdir(parents=True, exist_ok=True)

    device = resolve_device(config["device"])
    hf_token = resolve_hf_token(config["hf_token"])
    model_name = get_model_name(config["model"])
    feature_tag = slugify(model_name)
    feature_table_key = f"{feature_tag}_tiles"
    feature_model = build_feature_model(config["model"], config["model_path"], hf_token)

    LOGGER.info("Pipeline started: tile_root=%s", config["tile_root"])
    LOGGER.info("Feature output root: %s", config["feature_root"])
    LOGGER.info(
        "Runtime config: model=%s device=%s batch_size=%s num_workers=%s skip_existing=%s",
        model_name,
        device,
        config["batch_size"],
        config["num_workers"],
        config["skip_existing"],
    )

    manifest_rows = []
    processed = 0
    started = time.perf_counter()

    for class_name, case_name, case_dir in iter_case_dirs(config["tile_root"]):
        if not case_allowed(class_name, case_name, config):
            continue
        if config["limit_cases"] is not None and processed >= int(config["limit_cases"]):
            break

        summary = process_case(
            config=config,
            class_name=class_name,
            case_name=case_name,
            case_dir=case_dir,
            feature_model=feature_model,
            feature_tag=feature_tag,
            feature_table_key=feature_table_key,
            device=device,
        )
        manifest_rows.append(summary)
        processed += 1

    if config["save_manifest"]:
        manifest_path = config["feature_root"] / "manifest.csv"
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
        LOGGER.info("Manifest written to %s", manifest_path)

    elapsed = time.perf_counter() - started
    LOGGER.info("Pipeline finished: processed=%s elapsed=%s", processed, format_seconds(elapsed))
    release_case_resources(device, config["cleanup_cuda_between_cases"], feature_model)


if __name__ == "__main__":
    main()

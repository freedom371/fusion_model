from __future__ import annotations

import gc
import inspect
import json
import logging
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import lazyslide as zs
from lazyslide.models import MODEL_REGISTRY
from wsidata import TileSpec
from wsidata.io import add_features, add_tiles


PROJECT_ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger("demo_lazyslide_uni2")


# Edit the values in this block directly.
# These defaults favor resumable batch processing over maximum single-case speed.
CONFIG = {
    "use_sample": False,
    "wsi_path": PROJECT_ROOT / "data" / "carcinoma" / "converted_wsidata",
    "output_dir": Path("outputs") / "uni2_batch_all",
    "store_path": None,
    "reader": None,
    "model": "uni2",
    "model_path": PROJECT_ROOT / "uni2",
    "hf_token": None,
    "device": "auto",
    "tile_size": 512,
    "stride": None,
    "mpp": 0.5,
    "batch_size": 8,
    "num_workers": 2,
    "limit_tiles": None,
    "case_filters": None,
    "class_filters": None,
    "limit_cases": None,
    "skip_existing": True,
    "save_manifest": True,
    "continue_on_error": True,
    "background_fraction": 0.3,
    "skip_background_filter": False,
    "precomputed_tiles_csv": None,
    "save_zarr": False,
    "list_models": False,
    "reload_model_each_case": False,
    "cleanup_cuda_between_cases": True,
    "log_cuda_memory": True,
    "min_batch_size": 1,
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


def is_cuda_device(device: str) -> bool:
    return device.startswith("cuda") and torch.cuda.is_available()


def log_cuda_memory(device: str, context: str, enabled: bool) -> None:
    if not enabled or not is_cuda_device(device):
        return
    try:
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        LOGGER.info("[%s] CUDA memory allocated=%.2f GiB reserved=%.2f GiB", context, allocated, reserved)
    except Exception:
        LOGGER.debug("Unable to read CUDA memory stats.", exc_info=True)


def release_case_resources(device: str, cleanup_cuda: bool, *objects: Any) -> None:
    for obj in objects:
        if obj is None:
            continue
        close_method = getattr(obj, "close", None)
        if callable(close_method):
            try:
                close_method()
            except Exception:
                LOGGER.debug("Resource close failed for %r", obj, exc_info=True)
    gc.collect()
    if cleanup_cuda and is_cuda_device(device):
        try:
            torch.cuda.synchronize()
        except Exception:
            LOGGER.debug("CUDA synchronize failed during cleanup.", exc_info=True)
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            LOGGER.debug("CUDA IPC collection failed during cleanup.", exc_info=True)


def is_cuda_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def build_case_feature_model(
    config: dict[str, Any],
    hf_token: str | None,
    shared_feature_model: Any | None,
    batch_mode: bool,
) -> Any:
    if shared_feature_model is not None and not config["reload_model_each_case"]:
        return shared_feature_model
    if batch_mode and config["reload_model_each_case"]:
        LOGGER.info("Loading feature model for the current case.")
    return build_feature_model(config["model"], config["model_path"], hf_token)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_hf_token(token: str | None) -> str | None:
    if token:
        return token
    for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(env_name)
        if value:
            return value
    return None


def resolve_model_path(model: Any, model_path: str | Path | None) -> Path | None:
    def normalize_path(path_value: str | Path | None) -> Path | None:
        if path_value is None:
            return None
        path = Path(path_value).expanduser().resolve()
        if path.is_dir():
            for candidate_name in ("pytorch_model.bin", "model.safetensors"):
                candidate = path / candidate_name
                if candidate.exists():
                    return candidate
        return path

    if not isinstance(model, str):
        return normalize_path(model_path)
    if model_path:
        return normalize_path(model_path)

    candidates = [
        f"{model.upper()}_MODEL_PATH",
        "LAZYSLIDE_MODEL_PATH",
        "MODEL_PATH",
    ]
    for env_name in candidates:
        value = os.environ.get(env_name)
        if value:
            return normalize_path(value)
    return None


def slugify(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in name.strip().lower())
    return "_".join(part for part in cleaned.split("_") if part) or "features"


def resolve_input_path(path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def resolve_optional_csv_path(path_like: str | Path | None) -> Path | None:
    path = resolve_input_path(path_like)
    return path


def resolve_wsi_path(wsi_path: Path) -> Path:
    if not wsi_path.exists():
        raise FileNotFoundError(f"WSI not found: {wsi_path}")

    if wsi_path.is_file():
        if wsi_path.suffix == ".zarr":
            for suffix in (".tif", ".tiff"):
                tif_path = wsi_path.with_suffix(suffix)
                if tif_path.exists():
                    return tif_path
        return wsi_path

    preferred = []
    case_name = wsi_path.name
    for suffix in (".tif", ".tiff", ".zarr"):
        preferred_path = wsi_path / f"{case_name}{suffix}"
        if preferred_path.exists():
            preferred.append(preferred_path)
    if preferred:
        return resolve_wsi_path(preferred[0])

    direct_candidates = sorted(
        p for p in wsi_path.iterdir() if p.suffix.lower() in {".tif", ".tiff", ".zarr"}
    )
    if len(direct_candidates) == 1:
        return resolve_wsi_path(direct_candidates[0])

    recursive_candidates = sorted(
        p for p in wsi_path.rglob("*") if p.suffix.lower() in {".tif", ".tiff", ".zarr"}
    )
    if len(recursive_candidates) == 1:
        return resolve_wsi_path(recursive_candidates[0])

    if len(recursive_candidates) > 1:
        examples = "\n".join(f"  - {path}" for path in recursive_candidates[:6])
        raise ValueError(
            f"wsi_path points to a directory with multiple candidate slides: {wsi_path}\n"
            f"Please point CONFIG['wsi_path'] to a specific case directory or .tif file.\n"
            f"Examples:\n{examples}"
        )

    raise FileNotFoundError(f"No .tif/.tiff/.zarr slide found under: {wsi_path}")


def resolve_reader(wsi_path: Path, reader: str | None) -> str | None:
    if reader is not None:
        return reader
    if wsi_path.suffix.lower() in {".tif", ".tiff"}:
        return "tiffslide"
    return None


def find_precomputed_tiles_csv(user_input_path: Path | None, resolved_wsi_path: Path, configured_csv: Path | None) -> Path | None:
    if configured_csv is not None:
        return configured_csv if configured_csv.exists() else None

    candidates = []
    if user_input_path is not None and user_input_path.is_dir():
        candidates.append(user_input_path / "tiles.csv")
    candidates.append(resolved_wsi_path.parent / "tiles.csv")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def infer_stride(values: pd.Series, fallback: int) -> int:
    unique_values = np.sort(values.unique())
    diffs = np.diff(unique_values)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return fallback
    return int(diffs.min())


def load_precomputed_tiles(wsi: Any, tiles_csv_path: Path, key: str = "tiles") -> int:
    # Reuse the original tile grid whenever possible. This is much cheaper than
    # re-running tissue detection and avoids repeated TIFF cropping work.
    tiles_df = pd.read_csv(tiles_csv_path)
    required_columns = {"x0", "y0", "x1", "y1"}
    missing = required_columns - set(tiles_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {tiles_csv_path}: {sorted(missing)}")

    tile_widths = (tiles_df["x1"] - tiles_df["x0"]).unique()
    tile_heights = (tiles_df["y1"] - tiles_df["y0"]).unique()
    if len(tile_widths) != 1 or len(tile_heights) != 1:
        raise ValueError(f"Inconsistent tile sizes in {tiles_csv_path}")

    tile_width = int(tile_widths[0])
    tile_height = int(tile_heights[0])
    stride_width = infer_stride(tiles_df["x0"], tile_width)
    stride_height = infer_stride(tiles_df["y0"], tile_height)

    tile_spec = TileSpec(
        height=tile_height,
        width=tile_width,
        stride_height=stride_height,
        stride_width=stride_width,
        mpp=None,
    )

    xys = tiles_df[["x0", "y0"]].to_numpy(dtype=int)
    if "tissue_id" in tiles_df.columns:
        tissue_ids = tiles_df["tissue_id"].fillna(0).to_numpy(dtype=int)
    else:
        tissue_ids = np.zeros(len(tiles_df), dtype=int)

    add_tiles(wsi, key, xys, tile_spec, tissue_ids)
    return tiles_df


def open_demo_wsi(config: dict[str, Any], store_path: Path) -> tuple[Any, str, Path]:
    if config["use_sample"]:
        wsi = zs.datasets.sample(with_data=False, pbar=False)
        slide_name = "sample"
        return wsi, slide_name, Path("sample")

    wsi_path = config["wsi_path"]
    if wsi_path is None:
        raise ValueError("Please set CONFIG['wsi_path'], or switch CONFIG['use_sample'] to True.")

    resolved_wsi_path = resolve_wsi_path(wsi_path)
    reader = resolve_reader(resolved_wsi_path, config["reader"])

    slide_name = resolved_wsi_path.stem
    wsi = zs.open_wsi(
        resolved_wsi_path,
        store=str(store_path),
        reader=reader,
        attach_thumbnail=True,
        save_thumbnail=True,
    )
    return wsi, slide_name, resolved_wsi_path


def export_tile_metadata(wsi: Any, output_dir: Path) -> Path:
    tiles = wsi.shapes["tiles"].copy()
    bounds = tiles.bounds.rename(
        columns={
            "minx": "x0",
            "miny": "y0",
            "maxx": "x1",
            "maxy": "y1",
        }
    )
    meta = pd.DataFrame(bounds)
    for column in ("tissue_id",):
        if column in tiles.columns:
            meta[column] = tiles[column].to_numpy()
    meta.insert(0, "tile_index", meta.index.to_numpy())

    out_path = output_dir / "tile_metadata.csv"
    meta.to_csv(out_path, index=False)
    return out_path


def save_thumbnail(wsi: Any, output_dir: Path) -> Path | None:
    try:
        thumbnail = wsi.thumbnail
    except Exception:
        return None

    if thumbnail is None:
        return None

    out_path = output_dir / "thumbnail.png"
    try:
        if hasattr(thumbnail, "save"):
            thumbnail.save(out_path)
        else:
            import matplotlib.pyplot as plt

            plt.imsave(out_path, np.asarray(thumbnail))
    except Exception:
        return None
    return out_path


def to_numpy(array_like: Any) -> np.ndarray:
    if isinstance(array_like, np.ndarray):
        return array_like
    if hasattr(array_like, "toarray"):
        return array_like.toarray()
    return np.asarray(array_like)


def get_model_name(model: Any) -> str:
    if isinstance(model, str):
        return model
    return getattr(model, "name", model.__class__.__name__)


def build_feature_model(model_name: Any, model_path: Path | None, hf_token: str | None):
    if not isinstance(model_name, str):
        return model_name
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


def get_active_tile_spec(wsi: Any, tile_key: str = "tiles") -> dict[str, Any] | None:
    spec_data = wsi.attrs.get(getattr(wsi, "TILE_SPEC_KEY", "tile_spec"), {})
    if isinstance(spec_data, dict):
        return spec_data.get(tile_key)
    return None


def default_image_transform():
    from torchvision.transforms.v2 import Compose, Normalize, Resize, ToDtype, ToImage

    return Compose(
        [
            ToImage(),
            ToDtype(dtype=torch.float32, scale=True),
            Resize(size=(224, 224), antialias=False),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


class TilePathDataset(Dataset):
    def __init__(self, tile_paths: list[str], transform):
        self.tile_paths = tile_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.tile_paths)

    def __getitem__(self, index: int):
        with Image.open(self.tile_paths[index]) as pil_image:
            image = pil_image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            return image


def extract_features_from_tile_paths(
    tile_paths: list[str],
    feature_model: Any,
    device: str,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    transform = feature_model.get_transform() if hasattr(feature_model, "get_transform") else default_image_transform()
    dataset = TilePathDataset(tile_paths, transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    outputs = []
    try:
        feature_model.to(device)
    except Exception:
        pass

    with torch.inference_mode():
        for image in loader:
            image = image.to(device)
            if hasattr(feature_model, "encode_image"):
                batch_output = feature_model.encode_image(image)
            else:
                batch_output = feature_model(image)
            if not isinstance(batch_output, np.ndarray):
                batch_output = batch_output.detach().cpu().numpy()
            outputs.append(batch_output)
    return np.concatenate(outputs, axis=0)


def is_case_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "tiles.csv").exists():
        return False
    try:
        resolve_wsi_path(path)
    except Exception:
        return False
    return True


def discover_cases(wsi_path: Path) -> list[dict[str, Any]]:
    if wsi_path.is_file():
        resolved = resolve_wsi_path(wsi_path)
        return [
            {
                "input_path": wsi_path,
                "resolved_wsi_path": resolved,
                "relative_parts": (),
                "case_label": resolved.stem,
            }
        ]

    if is_case_dir(wsi_path):
        resolved = resolve_wsi_path(wsi_path)
        return [
            {
                "input_path": wsi_path,
                "resolved_wsi_path": resolved,
                "relative_parts": (),
                "case_label": wsi_path.name,
            }
        ]

    case_dirs = sorted({path.parent for path in wsi_path.rglob("tiles.csv")})
    cases = []
    for case_dir in case_dirs:
        if not is_case_dir(case_dir):
            continue
        resolved = resolve_wsi_path(case_dir)
        relative_parts = case_dir.relative_to(wsi_path).parts
        case_label = "/".join(relative_parts)
        cases.append(
            {
                "input_path": case_dir,
                "resolved_wsi_path": resolved,
                "relative_parts": relative_parts,
                "case_label": case_label,
            }
        )
    if not cases:
        resolved = resolve_wsi_path(wsi_path)
        return [
            {
                "input_path": wsi_path,
                "resolved_wsi_path": resolved,
                "relative_parts": (),
                "case_label": resolved.stem,
            }
        ]
    return cases


def case_allowed(case_info: dict[str, Any], config: dict[str, Any]) -> bool:
    relative_parts = case_info["relative_parts"]
    class_name = relative_parts[0] if len(relative_parts) >= 2 else None
    case_name = relative_parts[-1] if relative_parts else case_info["resolved_wsi_path"].stem

    class_filters = config["class_filters"]
    if class_filters is not None and class_name is not None:
        if class_name not in {str(value) for value in class_filters}:
            return False

    case_filters = config["case_filters"]
    if case_filters is not None:
        allowed = {str(value) for value in case_filters}
        labels = {case_name, case_info["case_label"]}
        if class_name is not None:
            labels.add(f"{class_name}/{case_name}")
        if labels.isdisjoint(allowed):
            return False
    return True


def get_case_output_dir(base_output_dir: Path, case_info: dict[str, Any], batch_mode: bool) -> Path:
    if batch_mode and case_info["relative_parts"]:
        return base_output_dir.joinpath(*case_info["relative_parts"])
    return base_output_dir


def get_case_store_path(
    configured_store_path: Path | None,
    case_output_dir: Path,
    case_info: dict[str, Any],
    slide_name: str,
    batch_mode: bool,
) -> Path:
    if configured_store_path is None:
        return case_output_dir / f"{slide_name}.zarr"

    if not batch_mode:
        return configured_store_path

    if configured_store_path.suffix.lower() == ".zarr":
        raise ValueError("When batch processing multiple cases, CONFIG['store_path'] must be a directory, not a single .zarr path.")

    if case_info["relative_parts"]:
        return configured_store_path.joinpath(*case_info["relative_parts"], f"{slide_name}.zarr")
    return configured_store_path / f"{slide_name}.zarr"


def process_case(
    config: dict[str, Any],
    case_info: dict[str, Any],
    base_output_dir: Path,
    batch_mode: bool,
    device: str,
    hf_token: str | None,
    feature_model: Any | None,
    feature_tag: str,
    feature_table_key: str,
) -> dict[str, Any]:
    case_started = time.perf_counter()
    case_started_at = now_iso()
    case_output_dir = get_case_output_dir(base_output_dir, case_info, batch_mode)
    case_output_dir.mkdir(parents=True, exist_ok=True)

    slide_name = case_info["resolved_wsi_path"].stem
    summary_path = case_output_dir / "summary.json"
    tile_feature_path = case_output_dir / f"{slide_name}_{feature_tag}_tile_features.npy"
    slide_feature_path = case_output_dir / f"{slide_name}_{feature_tag}_slide_features.npy"
    tile_metadata_path = case_output_dir / "tile_metadata.csv"

    if config["skip_existing"] and summary_path.exists() and tile_feature_path.exists() and slide_feature_path.exists():
        LOGGER.info("Skipping %s: existing outputs found.", case_info["case_label"])
        existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        existing_summary["status"] = "skipped"
        return existing_summary

    store_path = get_case_store_path(
        configured_store_path=config["store_path"],
        case_output_dir=case_output_dir,
        case_info=case_info,
        slide_name=slide_name,
        batch_mode=batch_mode,
    ).resolve()

    LOGGER.info("=== Processing %s ===", case_info["case_label"])
    LOGGER.info("[1/5] Opening WSI on device=%s", device)
    log_cuda_memory(device, f"{case_info['case_label']} before open", config["log_cuda_memory"])

    wsi = None
    tiles_df = None
    case_feature_model = None
    resolved_wsi_path: Path | None = None
    model_name = get_model_name(config["model"])
    current_batch_size = int(config["batch_size"])

    try:
        wsi, _, resolved_wsi_path = open_demo_wsi(
            {**config, "wsi_path": case_info["input_path"]},
            store_path,
        )

        tiles_csv_path = find_precomputed_tiles_csv(
            user_input_path=case_info["input_path"],
            resolved_wsi_path=resolved_wsi_path,
            configured_csv=config["precomputed_tiles_csv"],
        )

        if tiles_csv_path is not None:
            LOGGER.info("[2/5] Loading precomputed tiles from %s", tiles_csv_path)
            tiles_df = load_precomputed_tiles(wsi, tiles_csv_path)
            tile_count = len(tiles_df)
            LOGGER.info("[3/5] Reusing precomputed tiles and skipping tissue detection.")
        else:
            LOGGER.info("[2/5] Detecting tissue regions")
            zs.pp.find_tissues(wsi)

            LOGGER.info("[3/5] Tiling tissue regions")
            zs.pp.tile_tissues(
                wsi,
                tile_px=config["tile_size"],
                stride_px=config["stride"],
                mpp=config["mpp"],
                background_filter=not config["skip_background_filter"],
                background_fraction=config["background_fraction"],
            )
            tile_count = len(wsi.shapes["tiles"])

        if config["limit_tiles"] is not None:
            limit_tiles = int(config["limit_tiles"])
            wsi.shapes["tiles"] = wsi.shapes["tiles"].iloc[:limit_tiles].copy()
            if tiles_df is not None:
                tiles_df = tiles_df.iloc[:limit_tiles].copy()
            tile_count = len(wsi.shapes["tiles"])
            LOGGER.info("Limiting to first %s tiles for this run.", tile_count)

        LOGGER.info("Prepared %s tiles.", tile_count)
        case_feature_model = build_case_feature_model(config, hf_token, feature_model, batch_mode)
        log_cuda_memory(device, f"{case_info['case_label']} before extraction", config["log_cuda_memory"])
        LOGGER.info("[4/5] Extracting features with model=%s", model_name)

        while True:
            try:
                if tiles_df is not None and "path" in tiles_df.columns:
                    features = extract_features_from_tile_paths(
                        tile_paths=tiles_df["path"].tolist(),
                        feature_model=case_feature_model,
                        device=device,
                        batch_size=current_batch_size,
                        num_workers=config["num_workers"],
                    )
                    add_features(wsi, feature_table_key, "tiles", features)
                else:
                    extraction_kwargs = {
                        "model": case_feature_model,
                        "device": device,
                        "batch_size": current_batch_size,
                        "num_workers": config["num_workers"],
                        "key_added": feature_table_key,
                    }
                    zs.tl.feature_extraction(wsi, **extraction_kwargs)
                break
            except RuntimeError as exc:
                if not is_cuda_oom_error(exc) or current_batch_size <= int(config["min_batch_size"]):
                    raise
                next_batch_size = max(int(config["min_batch_size"]), current_batch_size // 2)
                LOGGER.warning(
                    "CUDA OOM while processing %s with batch_size=%s. Retrying with batch_size=%s.",
                    case_info["case_label"],
                    current_batch_size,
                    next_batch_size,
                )
                current_batch_size = next_batch_size
                release_case_resources(device, config["cleanup_cuda_between_cases"])
                if config["reload_model_each_case"]:
                    case_feature_model = build_feature_model(config["model"], config["model_path"], hf_token)

        LOGGER.info("[5/5] Aggregating and exporting outputs")
        zs.tl.feature_aggregation(wsi, feature_key=feature_table_key, encoder="mean")

        feature_table = wsi.tables[feature_table_key]
        tile_features = to_numpy(feature_table.X)
        slide_features = to_numpy(feature_table.uns["agg_ops"]["agg_slide"]["features"])

        np.save(tile_feature_path, tile_features)
        np.save(slide_feature_path, slide_features)

        tile_metadata_path = export_tile_metadata(wsi, case_output_dir)
        thumbnail_path = save_thumbnail(wsi, case_output_dir)
        tile_spec = get_active_tile_spec(wsi, "tiles") or {}
        elapsed_seconds = time.perf_counter() - case_started

        summary = {
            "status": "completed",
            "started_at": case_started_at,
            "completed_at": now_iso(),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "case_label": case_info["case_label"],
            "slide_name": slide_name,
            "model": get_model_name(config["model"]),
            "feature_table_key": feature_table_key,
            "device": device,
            "tile_count": int(tile_features.shape[0]),
            "tile_feature_dim": int(tile_features.shape[1]),
            "slide_feature_shape": list(slide_features.shape),
            "tile_size": int(tile_spec.get("width", config["tile_size"])),
            "stride": int(tile_spec.get("stride_width", config["stride"] if config["stride"] is not None else config["tile_size"])),
            "mpp": tile_spec.get("mpp", config["mpp"]),
            "batch_size": current_batch_size,
            "background_filter": not config["skip_background_filter"],
            "background_fraction": config["background_fraction"],
            "wsi_path": str(resolved_wsi_path),
            "precomputed_tiles_csv": str(tiles_csv_path) if tiles_csv_path else None,
            "tile_features_path": str(tile_feature_path),
            "slide_features_path": str(slide_feature_path),
            "tile_metadata_path": str(tile_metadata_path),
            "thumbnail_path": str(thumbnail_path) if thumbnail_path else None,
            "store_path": str(store_path) if config["save_zarr"] else None,
        }

        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        if config["save_zarr"]:
            wsi.write(store_path, overwrite=True)

        LOGGER.info("Done with %s in %s.", case_info["case_label"], format_seconds(elapsed_seconds))
        LOGGER.info("Summary: %s", summary_path)
        LOGGER.info("Tile features: %s", tile_feature_path)
        LOGGER.info("Slide features: %s", slide_feature_path)
        LOGGER.info("Tile metadata: %s", tile_metadata_path)
        if config["save_zarr"]:
            LOGGER.info("WSIData store: %s", store_path)
        log_cuda_memory(device, f"{case_info['case_label']} after export", config["log_cuda_memory"])
        return summary
    except Exception as exc:
        LOGGER.error("Case %s failed: %s", case_info["case_label"], exc)
        LOGGER.error(traceback.format_exc())
        raise
    finally:
        release_case_resources(
            device,
            config["cleanup_cuda_between_cases"],
            tiles_df,
            wsi,
        )
        if config["reload_model_each_case"] and case_feature_model is not None:
            release_case_resources(device, config["cleanup_cuda_between_cases"], case_feature_model)


def main() -> None:
    setup_logging()
    config = dict(CONFIG)
    config["wsi_path"] = resolve_input_path(config["wsi_path"])
    config["output_dir"] = resolve_input_path(config["output_dir"])
    config["store_path"] = resolve_input_path(config["store_path"])
    config["model_path"] = resolve_model_path(config["model"], config["model_path"])
    config["precomputed_tiles_csv"] = resolve_optional_csv_path(config["precomputed_tiles_csv"])

    if config["list_models"]:
        LOGGER.info("Listing available models.")
        print("\n".join(zs.models.list_models()))
        return

    output_dir = config["output_dir"].resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if config["use_sample"]:
        cases = [{"input_path": None, "resolved_wsi_path": Path("sample"), "relative_parts": (), "case_label": "sample"}]
    else:
        if config["wsi_path"] is None:
            raise ValueError("Please set CONFIG['wsi_path'], or switch CONFIG['use_sample'] to True.")
        cases = [case for case in discover_cases(config["wsi_path"]) if case_allowed(case, config)]
        if config["limit_cases"] is not None:
            cases = cases[: int(config["limit_cases"])]
        if not cases:
            raise ValueError(f"No matching cases found under: {config['wsi_path']}")

    batch_mode = len(cases) > 1
    device = resolve_device(config["device"])
    hf_token = resolve_hf_token(config["hf_token"])
    model_name = get_model_name(config["model"])
    feature_tag = slugify(model_name)
    feature_table_key = f"{feature_tag}_tiles"
    feature_model = None
    if not config["reload_model_each_case"]:
        LOGGER.info("Loading shared feature model once for all cases.")
        feature_model = build_feature_model(config["model"], config["model_path"], hf_token)

    LOGGER.info("Batch started. cases=%s batch_mode=%s device=%s model=%s", len(cases), batch_mode, device, model_name)
    LOGGER.info("Input path: %s", config["wsi_path"])
    LOGGER.info("Output path: %s", output_dir)
    LOGGER.info(
        "Runtime config: batch_size=%s num_workers=%s skip_existing=%s reload_model_each_case=%s",
        config["batch_size"],
        config["num_workers"],
        config["skip_existing"],
        config["reload_model_each_case"],
    )
    log_cuda_memory(device, "batch start", config["log_cuda_memory"])

    manifest_rows = []
    batch_started = time.perf_counter()
    for case_info in cases:
        try:
            summary = process_case(
                config=config,
                case_info=case_info,
                base_output_dir=output_dir,
                batch_mode=batch_mode,
                device=device,
                hf_token=hf_token,
                feature_model=feature_model,
                feature_tag=feature_tag,
                feature_table_key=feature_table_key,
            )
        except Exception as exc:
            if not batch_mode or not config["continue_on_error"]:
                raise
            summary = {
                "status": "failed",
                "started_at": now_iso(),
                "case_label": case_info["case_label"],
                "wsi_path": str(case_info["resolved_wsi_path"]),
                "error": str(exc),
            }
            LOGGER.error("Failed: %s: %s", case_info["case_label"], exc)
        manifest_rows.append(summary)

    if batch_mode and config["save_manifest"]:
        manifest = pd.DataFrame(manifest_rows)
        manifest_path = output_dir / "manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        LOGGER.info("Batch manifest: %s", manifest_path)

    total_elapsed = time.perf_counter() - batch_started
    LOGGER.info("Batch finished in %s.", format_seconds(total_elapsed))
    log_cuda_memory(device, "batch end", config["log_cuda_memory"])
    if feature_model is not None:
        release_case_resources(device, config["cleanup_cuda_between_cases"], feature_model)


if __name__ == "__main__":
    main()

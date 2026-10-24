import csv
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:
    from config import DATA_CT_DIR, LABEL_CSV, ROOT
except ImportError:
    from .config import DATA_CT_DIR, LABEL_CSV, ROOT


VIEW_ORDER = ("max", "above", "below")

conf = {
    # ── 数据路径 ──
    "ct_dir": str(ROOT / "data/images/ct5_cropped"),
    "label_csv": str(ROOT / "data/cvs/label.csv"),
    "output_ct_dir": str(ROOT / "data/images/ct_cropped_augmented_safe"),
    "output_label_csv": str(ROOT / "data/cvs/label_cropped_augmented_safe.csv"),

    # ── 生成控制 ──
    "num_augmentations": 2,   # 每个训练样本生成几份增强副本
    "seed": 20260603,
    "overwrite": True,

    # ── 几何增强（三张 CT 切片同步应用同一组参数） ──
    "geo_hflip_prob": 0,              # 水平翻转概率
    "geo_rotate_prob": 0,             # 旋转概率
    "geo_rotate_degree": 8.0,           # 旋转角度 ±8°
    "geo_translate_prob": 0,          # 平移概率
    "geo_translate_ratio": 0.08,        # 平移比例 ±8%（相对图像宽高）
    "geo_scale_prob": 0,              # 缩放概率
    "geo_scale_range": (0.95, 1.05),    # 缩放因子范围

    # ── 强度增强（三张 CT 切片同步应用同一组噪声/亮度/对比度） ──
    "int_brightness_contrast_prob": 0.5,  # 亮度+对比度概率
    "int_brightness_shift": 0.10,         # 亮度偏移 ±10%（×255 像素值）
    "int_contrast_range": (0.90, 1.10),   # 对比度因子范围
    "int_gamma_prob": 0.5,                # Gamma 校正概率
    "int_gamma_range": (0.90, 1.10),      # Gamma 值范围
    "int_noise_prob": 0.5,                # 高斯噪声概率
    "int_noise_std": 4.0,                 # 噪声标准差（像素值 0-255）
}


def base_id_from_label_id(label_id: str) -> str:
    return label_id.replace(".nii.gz", "")


def ct_path(ct_dir: Path, base_id: str, view: str) -> Path:
    return ct_dir / f"{base_id}_ct_{view}.png"


def load_gray_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def save_gray_image(array: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).save(path)


def pad_to_shape(array: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_shape
    h, w = array.shape[:2]
    pad_top = max((target_h - h) // 2, 0)
    pad_bottom = max(target_h - h - pad_top, 0)
    pad_left = max((target_w - w) // 2, 0)
    pad_right = max(target_w - w - pad_left, 0)
    return np.pad(
        array,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,
    )


def sample_geometry_params(shape: tuple[int, int], rng: np.random.Generator, c: dict) -> dict:
    """从配置中读取概率，生成一组几何增强参数（三视图共享）。"""
    height, width = shape
    do_rotate = rng.random() < c["geo_rotate_prob"]
    do_affine = rng.random() < max(c["geo_translate_prob"], c["geo_scale_prob"])
    return {
        "hflip": rng.random() < c["geo_hflip_prob"],
        "angle": float(rng.uniform(-c["geo_rotate_degree"], c["geo_rotate_degree"])) if do_rotate else 0.0,
        "scale": float(rng.uniform(*c["geo_scale_range"])) if do_affine else 1.0,
        "dx": float(rng.uniform(-c["geo_translate_ratio"], c["geo_translate_ratio"]) * width) if do_affine else 0.0,
        "dy": float(rng.uniform(-c["geo_translate_ratio"], c["geo_translate_ratio"]) * height) if do_affine else 0.0,
    }


def apply_geometry(array: np.ndarray, params: dict) -> np.ndarray:
    image = Image.fromarray(array.astype(np.uint8))
    resample = Image.Resampling.BILINEAR

    if params["hflip"]:
        image = ImageOps.mirror(image)
    if params["angle"] != 0:
        image = image.rotate(params["angle"], resample=resample, fillcolor=0)
    if params["scale"] != 1.0 or params["dx"] != 0.0 or params["dy"] != 0.0:
        width, height = image.size
        cx, cy = width / 2.0, height / 2.0
        scale = params["scale"]
        coeffs = (
            1.0 / scale,
            0.0,
            cx * (1.0 - 1.0 / scale) - params["dx"] / scale,
            0.0,
            1.0 / scale,
            cy * (1.0 - 1.0 / scale) - params["dy"] / scale,
        )
        image = image.transform(image.size, Image.Transform.AFFINE, coeffs, resample=resample, fillcolor=0)

    return np.asarray(image, dtype=np.uint8)


def apply_shared_intensity(images: list[np.ndarray], rng: np.random.Generator, c: dict) -> list[np.ndarray]:
    """从配置中读取概率，对三视图同步应用强度增强（噪声/亮度/对比度/Gamma）。"""
    out = [img.astype(np.float32) for img in images]

    if rng.random() < c["int_brightness_contrast_prob"]:
        brightness = rng.uniform(-c["int_brightness_shift"], c["int_brightness_shift"]) * 255.0
        contrast = rng.uniform(*c["int_contrast_range"])
        out = [(img - 127.5) * contrast + 127.5 + brightness for img in out]

    if rng.random() < c["int_gamma_prob"]:
        gamma = rng.uniform(*c["int_gamma_range"])
        out = [255.0 * np.power(np.clip(img, 0, 255) / 255.0, gamma) for img in out]

    if rng.random() < c["int_noise_prob"]:
        noise = rng.normal(0.0, c["int_noise_std"], size=out[0].shape).astype(np.float32)
        out = [img + noise for img in out]

    return [np.clip(img, 0, 255).astype(np.uint8) for img in out]


def has_all_views(base_id: str, ct_dir: Path) -> bool:
    for view in VIEW_ORDER:
        if not ct_path(ct_dir, base_id, view).exists():
            return False
    return True


def copy_case(base_id: str, src_ct_dir: Path, dst_ct_dir: Path):
    for view in VIEW_ORDER:
        shutil.copy2(ct_path(src_ct_dir, base_id, view), ct_path(dst_ct_dir, base_id, view))


def augment_case(
    base_id: str,
    aug_base_id: str,
    rng: np.random.Generator,
    src_ct_dir: Path,
    dst_ct_dir: Path,
    c: dict,
):
    ct_images = [load_gray_image(ct_path(src_ct_dir, base_id, view)) for view in VIEW_ORDER]
    target_shape = (
        max(image.shape[0] for image in ct_images),
        max(image.shape[1] for image in ct_images),
    )
    ct_images = [pad_to_shape(image, target_shape) for image in ct_images]
    params = sample_geometry_params(ct_images[0].shape, rng, c)
    geom_ct_images = [apply_geometry(image, params) for image in ct_images]
    aug_ct_images = apply_shared_intensity(geom_ct_images, rng, c)

    for view, image in zip(VIEW_ORDER, aug_ct_images):
        save_gray_image(image, ct_path(dst_ct_dir, aug_base_id, view))


def read_label_rows(label_csv: Path) -> tuple[list[str], list[dict]]:
    with label_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def write_label_rows(path: Path, fieldnames: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_augmented_data(c: dict):
    rng = np.random.default_rng(c["seed"])
    random.seed(c["seed"])

    src_ct_dir = Path(c["ct_dir"])
    dst_ct_dir = Path(c["output_ct_dir"])
    output_label_csv = Path(c["output_label_csv"])

    if c["overwrite"]:
        shutil.rmtree(dst_ct_dir, ignore_errors=True)
        if output_label_csv.exists():
            output_label_csv.unlink()

    dst_ct_dir.mkdir(parents=True, exist_ok=True)

    fieldnames, rows = read_label_rows(Path(c["label_csv"]))
    if "ID" not in fieldnames or "group" not in fieldnames:
        raise ValueError("Label CSV must contain ID and group columns.")

    output_rows = [row.copy() for row in rows]

    valid_train_cases = {}
    for row in rows:
        if row["group"] != "train":
            continue
        base_id = base_id_from_label_id(row["ID"])
        if has_all_views(base_id, src_ct_dir):
            valid_train_cases.setdefault(base_id, row)

    print(f"[augment] valid train cases: {len(valid_train_cases)}")
    print(f"[augment] augmentations per train case: {c['num_augmentations']}")

    all_base_ids = sorted({base_id_from_label_id(row["ID"]) for row in rows})
    for idx, base_id in enumerate(all_base_ids, start=1):
        if has_all_views(base_id, src_ct_dir):
            copy_case(base_id, src_ct_dir, dst_ct_dir)
        if idx % 100 == 0:
            print(f"[copy] {idx}/{len(all_base_ids)} cases")

    train_items = sorted(valid_train_cases.items())
    for case_idx, (base_id, row) in enumerate(train_items, start=1):
        for aug_idx in range(1, c["num_augmentations"] + 1):
            aug_base_id = f"{base_id}_aug{aug_idx}"
            augment_case(
                base_id,
                aug_base_id,
                rng,
                src_ct_dir,
                dst_ct_dir,
                c,
            )

            aug_row = row.copy()
            aug_row["ID"] = f"{aug_base_id}.nii.gz"
            aug_row["group"] = "train"
            output_rows.append(aug_row)

        if case_idx % 20 == 0 or case_idx == len(train_items):
            print(f"[augment] {case_idx}/{len(train_items)} train cases")

    write_label_rows(output_label_csv, fieldnames, output_rows)
    print("[augment] done")
    print(f"ct dir       : {dst_ct_dir}")
    print(f"label csv    : {output_label_csv}")
    print(f"label rows   : {len(output_rows)}")

if __name__ == "__main__":
    generate_augmented_data(conf)

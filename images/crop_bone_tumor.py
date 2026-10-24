#!/usr/bin/env python3
"""对骨窗 CT 图像应用 mask 裁剪，复刻 crop_tumor.py 逻辑。"""
import numpy as np
from PIL import Image
from pathlib import Path
from scipy import ndimage

ROOT          = Path(__file__).resolve().parent.parent
SRC_CT        = ROOT / "data/images/ct_bone"
SRC_MASK      = ROOT / "data/images/mask_p5"
OUT_CT        = ROOT / "data/images/ct5_bone_cropped"
EXPAND_FACTOR = 1.25
PADDING       = 5


def expand_mask(mask: np.ndarray, factor: float) -> np.ndarray:
    if factor <= 1.0:
        return mask
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if not rows.any():
        return mask
    rmin, rmax = rows.nonzero()[0][[0, -1]]
    cmin, cmax = cols.nonzero()[0][[0, -1]]
    h = rmax - rmin + 1
    w = cmax - cmin + 1
    radius = int(min(h, w) / 2 * (factor - 1))
    if radius <= 0:
        return mask
    dist = ndimage.distance_transform_edt(~mask)
    return dist <= radius


def extract_ct_by_mask(ct: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0, ct, 0).astype(ct.dtype)


def get_mask_bbox(mask: np.ndarray, padding: int):
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if not rows.any():
        return None
    rmin, rmax = rows.nonzero()[0][[0, -1]]
    cmin, cmax = cols.nonzero()[0][[0, -1]]
    h, w = mask.shape
    rmin = max(0, rmin - padding)
    rmax = min(h - 1, rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(w - 1, cmax + padding)
    return rmin, rmax, cmin, cmax


def crop_by_bbox(img: np.ndarray, bbox: tuple) -> np.ndarray:
    rmin, rmax, cmin, cmax = bbox
    return img[rmin:rmax + 1, cmin:cmax + 1]


def main():
    OUT_CT.mkdir(parents=True, exist_ok=True)
    # 清理旧文件
    for f in OUT_CT.glob("*.png"):
        f.unlink()
    print("已清理旧文件")

    mask_files = sorted(SRC_MASK.glob("*_mask_*.png"))
    print(f"共 {len(mask_files)} 张 mask")
    print(f"CT 输入  (骨窗): {SRC_CT}")
    print(f"mask 输入      : {SRC_MASK}")
    print(f"CT 输出        : {OUT_CT}")

    sizes = []
    skipped_empty = 0
    skipped_missing_ct = 0
    skipped_shape = 0

    for i, mask_f in enumerate(mask_files, start=1):
        ct_name = mask_f.name.replace("_mask_", "_ct_")
        ct_f = SRC_CT / ct_name
        if not ct_f.exists():
            skipped_missing_ct += 1
            continue

        ct_img = np.asarray(Image.open(ct_f))
        mask_img = np.asarray(Image.open(mask_f))
        if ct_img.shape != mask_img.shape:
            skipped_shape += 1
            continue

        if not np.any(mask_img > 0):
            skipped_empty += 1
            continue

        expanded_mask = expand_mask(mask_img, EXPAND_FACTOR)
        ct_extracted = extract_ct_by_mask(ct_img, expanded_mask)
        bbox = get_mask_bbox(expanded_mask, PADDING)
        if bbox is None:
            skipped_empty += 1
            continue
        ct_cropped = crop_by_bbox(ct_extracted, bbox)
        Image.fromarray(ct_cropped).save(OUT_CT / ct_name)
        sizes.append(ct_cropped.shape)

        if i % 500 == 0:
            print(f"[{i}/{len(mask_files)}]")

    if not sizes:
        print(f"没有成功裁剪任何图片 (empty={skipped_empty}, missing_ct={skipped_missing_ct}, shape_mismatch={skipped_shape})")
        return

    sizes = np.array(sizes)
    print(f"完成: {len(sizes)} 张")
    print(f"跳过: empty={skipped_empty}, missing_ct={skipped_missing_ct}, shape_mismatch={skipped_shape}")
    print(f"尺寸范围: h=[{sizes[:,0].min()},{sizes[:,0].max()}], w=[{sizes[:,1].min()},{sizes[:,1].max()}]")
    print(f"尺寸中位数: h={np.median(sizes[:,0]):.0f}, w={np.median(sizes[:,1]):.0f}")


if __name__ == "__main__":
    main()

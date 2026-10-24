#!/usr/bin/env python3
"""
1. masks 找最大层（避开肿瘤边界）
2. 用 mask 定位肿瘤 bbox → 从 images 原图裁剪肿瘤+周边区域
3. masks_p5 外轮廓同区域裁剪
4. 输出裁剪后 CT 和 mask_p5（保留周边组织，非黑底）
"""
import nibabel as nib
import numpy as np
from PIL import Image
from pathlib import Path
from scipy import ndimage

MASKS_DIR    = Path("/hard0/carcinoma/masks")
MASKS_P5_DIR = Path("/hard0/carcinoma/masks_p5")
IMAGES_DIR   = Path("/hard0/carcinoma/images")
ROOT         = Path(__file__).resolve().parent.parent
CT_OUT       = ROOT / "data/images/ct_5"
MASK_OUT     = ROOT / "data/images/mask_p5"
MARGIN       = 40  # 肿瘤周边上下文边距（像素）

# Oral/head-neck tumor CT mainly relies on soft-tissue contrast.
# Fixed soft-tissue window: level 50 HU, width 350 HU -> [-125, 225] HU.
WINDOW_LEVEL = 50.0
WINDOW_WIDTH = 350.0


def process_mask_p5(slice_data):
    tumor = (slice_data == 1.0) | (slice_data == 2.0)
    return ndimage.binary_fill_holes(tumor)


def get_tumor_bbox(mask_slice):
    """返回肿瘤区域的包围盒 (rmin, rmax, cmin, cmax)，mask 是二值图"""
    rows = np.any(mask_slice > 0, axis=1)
    cols = np.any(mask_slice > 0, axis=0)
    if not rows.any():
        return None
    rmin, rmax = rows.nonzero()[0][[0, -1]]
    cmin, cmax = cols.nonzero()[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def crop_with_margin(img, bbox, margin):
    rmin, rmax, cmin, cmax = bbox
    h, w = img.shape
    rmin = max(0, rmin - margin)
    rmax = min(h - 1, rmax + margin)
    cmin = max(0, cmin - margin)
    cmax = min(w - 1, cmax + margin)
    return img[rmin:rmax + 1, cmin:cmax + 1]


def normalize_ct(slice_data):
    """Apply a fixed CT soft-tissue window and map HU values to 0-255."""
    lower = WINDOW_LEVEL - WINDOW_WIDTH / 2.0
    upper = WINDOW_LEVEL + WINDOW_WIDTH / 2.0
    windowed = np.clip(slice_data, lower, upper)
    normalized = (windowed - lower) / (upper - lower) * 255.0
    return normalized.astype(np.uint8)


def extract_one(mask_path: Path):
    sample_id = mask_path.stem.split(".")[0]
    img_path = IMAGES_DIR / f"{sample_id}.nii.gz"
    p5_path  = MASKS_P5_DIR / f"{sample_id}.nii.gz"

    mask_data = nib.load(str(mask_path)).get_fdata()
    ct_data   = nib.load(str(img_path)).get_fdata()
    p5_data   = nib.load(str(p5_path)).get_fdata()

    total_z = mask_data.shape[2]

    # 肿瘤内有内容的 Z 层
    active_zs = [z for z in range(total_z) if np.count_nonzero(mask_data[:, :, z]) > 0]
    if not active_zs:
        return f"SKIP {sample_id}: empty mask"

    # 找最大层（避开边界）
    sorted_by_area = sorted(active_zs, key=lambda z: np.count_nonzero(mask_data[:, :, z]), reverse=True)
    best_z = None
    for z in sorted_by_area:
        if z != active_zs[0] and z != active_zs[-1]:
            best_z = z
            break
    if best_z is None:
        best_z = sorted_by_area[0]

    # 肿瘤内上下层
    idx = active_zs.index(best_z)
    z_above = active_zs[idx - 1] if idx > 0 else best_z
    z_below = active_zs[idx + 1] if idx + 1 < len(active_zs) else best_z

    # 用 mask (二值) 在 max 层确定肿瘤 bbox（三层共用同一 bbox）
    mask_max = (mask_data[:, :, best_z] > 0)
    bbox = get_tumor_bbox(mask_max)
    if bbox is None:
        return f"SKIP {sample_id}: no tumor in max layer"

    for z_slice, tag in [(best_z, "max"), (z_above, "above"), (z_below, "below")]:
        # CT 原图：裁剪 + 归一化（不用 mask 乘，保留周边组织）
        ct_slice = ct_data[:, :, z_slice]
        ct_cropped = crop_with_margin(ct_slice, bbox, MARGIN)
        ct_cropped = ct_cropped.T
        ct_cropped = np.flipud(ct_cropped)
        ct_png = normalize_ct(ct_cropped)

        # masks_p5 外轮廓填充：裁剪同区域
        outer = process_mask_p5(p5_data[:, :, z_slice])
        mask_cropped = crop_with_margin(outer, bbox, MARGIN)
        mask_cropped = mask_cropped.T
        mask_cropped = np.flipud(mask_cropped)
        mask_png = (mask_cropped * 255).astype(np.uint8)

        CT_OUT.mkdir(parents=True, exist_ok=True)
        MASK_OUT.mkdir(parents=True, exist_ok=True)
        Image.fromarray(ct_png).save(str(CT_OUT / f"{sample_id}_ct_{tag}.png"))
        Image.fromarray(mask_png).save(str(MASK_OUT / f"{sample_id}_mask_{tag}.png"))

    return f"OK {sample_id}: best_z={best_z}, above={z_above}, below={z_below}"


def main():
    mask_files = sorted(MASKS_DIR.glob("*.nii.gz"))
    print(f"共 {len(mask_files)} 个样本")

    # 清理旧文件
    for d in [CT_OUT, MASK_OUT]:
        if d.exists():
            for f in d.glob("*.png"):
                f.unlink()
    print("已清理旧文件")

    for i, mf in enumerate(mask_files):
        msg = extract_one(mf)
        if (i + 1) % 100 == 0 or "SKIP" in msg:
            print(f"[{i+1}/{len(mask_files)}] {msg}")

    ct_count = len(list(CT_OUT.glob("*.png")))
    mask_count = len(list(MASK_OUT.glob("*.png")))
    # 统计尺寸
    sizes = []
    for f in CT_OUT.glob("*.png"):
        img = np.array(Image.open(str(f)))
        sizes.append(img.shape)
    sizes = np.array(sizes)
    print(f"完成: ct={ct_count}, mask={mask_count}")
    print(f"尺寸范围: h=[{sizes[:,0].min()},{sizes[:,0].max()}], w=[{sizes[:,1].min()},{sizes[:,1].max()}]")
    print(f"中位数: h={np.median(sizes[:,0]):.0f}, w={np.median(sizes[:,1]):.0f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
从 NIfTI 源文件按骨窗参数提取 CT 切片，复刻 extract_final.py 的裁剪逻辑。
骨窗: W=1500, L=400 → [-350, 1150] HU → 0-255
输出到 data/images/ct_bone/
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
CT_OUT       = ROOT / "data/images/ct_bone"
MARGIN       = 40

# 骨窗参数
WINDOW_LEVEL = 400.0
WINDOW_WIDTH = 1500.0


def process_mask_p5(slice_data):
    tumor = (slice_data == 1.0) | (slice_data == 2.0)
    return ndimage.binary_fill_holes(tumor)


def get_tumor_bbox(mask_slice):
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


def normalize_ct_bone(slice_data):
    """骨窗: W=1500, L=400 → [-350, 1150] HU → 0-255"""
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

    active_zs = [z for z in range(total_z) if np.count_nonzero(mask_data[:, :, z]) > 0]
    if not active_zs:
        return f"SKIP {sample_id}: empty mask"

    sorted_by_area = sorted(active_zs, key=lambda z: np.count_nonzero(mask_data[:, :, z]), reverse=True)
    best_z = None
    for z in sorted_by_area:
        if z != active_zs[0] and z != active_zs[-1]:
            best_z = z
            break
    if best_z is None:
        best_z = sorted_by_area[0]

    idx = active_zs.index(best_z)
    z_above = active_zs[idx - 1] if idx > 0 else best_z
    z_below = active_zs[idx + 1] if idx + 1 < len(active_zs) else best_z

    mask_max = (mask_data[:, :, best_z] > 0)
    bbox = get_tumor_bbox(mask_max)
    if bbox is None:
        return f"SKIP {sample_id}: no tumor in max layer"

    for z_slice, tag in [(best_z, "max"), (z_above, "above"), (z_below, "below")]:
        # CT 原图：裁剪 + 骨窗归一化
        ct_slice = ct_data[:, :, z_slice]
        ct_cropped = crop_with_margin(ct_slice, bbox, MARGIN)
        ct_cropped = ct_cropped.T
        ct_cropped = np.flipud(ct_cropped)
        ct_png = normalize_ct_bone(ct_cropped)

        CT_OUT.mkdir(parents=True, exist_ok=True)
        Image.fromarray(ct_png).save(str(CT_OUT / f"{sample_id}_ct_{tag}.png"))

    return f"OK {sample_id}: best_z={best_z}, above={z_above}, below={z_below}"


def main():
    mask_files = sorted(MASKS_DIR.glob("*.nii.gz"))
    print(f"共 {len(mask_files)} 个样本")
    print(f"骨窗参数: Level={WINDOW_LEVEL}, Width={WINDOW_WIDTH} → [{WINDOW_LEVEL - WINDOW_WIDTH/2:.0f}, {WINDOW_LEVEL + WINDOW_WIDTH/2:.0f}] HU")

    # 清理旧文件
    if CT_OUT.exists():
        for f in CT_OUT.glob("*.png"):
            f.unlink()
    print("已清理旧文件")

    for i, mf in enumerate(mask_files):
        msg = extract_one(mf)
        if (i + 1) % 100 == 0 or "SKIP" in msg:
            print(f"[{i+1}/{len(mask_files)}] {msg}")

    ct_count = len(list(CT_OUT.glob("*.png")))
    sizes = []
    for f in CT_OUT.glob("*.png"):
        img = np.array(Image.open(str(f)))
        sizes.append(img.shape)
    sizes = np.array(sizes)
    print(f"完成: ct_bone={ct_count}")
    print(f"尺寸范围: h=[{sizes[:,0].min()},{sizes[:,0].max()}], w=[{sizes[:,1].min()},{sizes[:,1].max()}]")
    print(f"中位数: h={np.median(sizes[:,0]):.0f}, w={np.median(sizes[:,1]):.0f}")


if __name__ == "__main__":
    main()

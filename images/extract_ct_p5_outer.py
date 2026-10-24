#!/usr/bin/env python3
"""
提取 CT 原图 + masks_p5 外轮廓勾画的最大层及上下层（共三层）。

masks_p5 处理逻辑：
  0 = 背景
  1 = 瘤内部
  2 = 瘤外层/肠壁
  → 合并 1+2 得到完整瘤区域 → 填充空洞 → 外轮廓里面=1, 外面=0

输出（三个目录，每个目录各存 CT 原图 + mask 勾画）：
  contour_p5_outer_max/   最大层
  contour_p5_outer_above/ 上一层
  contour_p5_outer_below/ 下一层
"""
import nibabel as nib
import numpy as np
from PIL import Image
from pathlib import Path
from scipy import ndimage

MASKS_DIR  = Path("/hard0/carcinoma/masks_p5")
IMAGES_DIR = Path("/hard0/carcinoma/images")
OUT_DIR    = Path("/hard0/carcinoma")

# Oral/head-neck tumor CT mainly relies on soft-tissue contrast.
# Fixed soft-tissue window: level 50 HU, width 350 HU -> [-125, 225] HU.
WINDOW_LEVEL = 50.0
WINDOW_WIDTH = 350.0


def apply_ct_window(ct_slice: np.ndarray) -> np.ndarray:
    lower = WINDOW_LEVEL - WINDOW_WIDTH / 2.0
    upper = WINDOW_LEVEL + WINDOW_WIDTH / 2.0
    ct_windowed = np.clip(ct_slice, lower, upper)
    ct_norm = (ct_windowed - lower) / (upper - lower) * 255.0
    return ct_norm.astype(np.uint8)


def extract_one(mask_path: Path):
    sample_id = mask_path.stem.split(".")[0]
    img_path = IMAGES_DIR / f"{sample_id}.nii.gz"

    if not img_path.exists():
        return f"SKIP {sample_id}: CT not found"

    mask_data = nib.load(str(mask_path)).get_fdata()
    ct_data   = nib.load(str(img_path)).get_fdata()

    if mask_data.shape != ct_data.shape:
        return f"SKIP {sample_id}: shape mismatch"

    total_z = mask_data.shape[2]

    # 找最大层：合并 v1+v2，按非零像素数
    best_z, best_nz = max(
        ((z, np.count_nonzero((mask_data[:, :, z] == 1.0) | (mask_data[:, :, z] == 2.0)))
         for z in range(total_z)),
        key=lambda x: x[1],
    )

    if best_nz == 0:
        return f"SKIP {sample_id}: empty mask"

    z_above = max(best_z - 1, 0)
    z_below = min(best_z + 1, total_z - 1)

    for z_slice, tag in [(best_z, "max"), (z_above, "above"), (z_below, "below")]:
        # --- CT 原图 ---
        ct_slice = ct_data[:, :, z_slice].T
        ct_slice = np.flipud(ct_slice)
        ct_norm = apply_ct_window(ct_slice)

        # --- mask 外轮廓 ---
        mask_slice = mask_data[:, :, z_slice]
        # 合并 1+2 为完整瘤区域，填充空洞（外轮廓里面=1）
        tumor = (mask_slice == 1.0) | (mask_slice == 2.0)
        filled = ndimage.binary_fill_holes(tumor)
        mask_png = (filled * 255).astype(np.uint8)
        mask_png = mask_png.T
        mask_png = np.flipud(mask_png)

        out_dir = OUT_DIR / f"contour_p5_outer_{tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(ct_norm).save(str(out_dir / f"{sample_id}_ct.png"))
        Image.fromarray(mask_png).save(str(out_dir / f"{sample_id}_mask.png"))

    return f"OK {sample_id}: max_z={best_z}, nz={best_nz}"


def main():
    mask_files = sorted(MASKS_DIR.glob("*.nii.gz"))
    print(f"共 {len(mask_files)} 个 masks_p5 文件")

    for i, mf in enumerate(mask_files):
        msg = extract_one(mf)
        if (i + 1) % 50 == 0 or "SKIP" in msg:
            print(f"[{i+1}/{len(mask_files)}] {msg}")


if __name__ == "__main__":
    main()

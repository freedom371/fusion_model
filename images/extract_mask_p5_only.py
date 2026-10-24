#!/usr/bin/env python3
"""从 masks_p5 提取处理后的外轮廓 mask 图，层号与 ct 一致（以 masks 最大层为准）。"""
import nibabel as nib
import numpy as np
from PIL import Image
from pathlib import Path
from scipy import ndimage

MASKS_DIR    = Path("/hard0/carcinoma/masks")
MASKS_P5_DIR = Path("/hard0/carcinoma/masks_p5")
OUT_DIR      = Path("/home/guozheng/wsi_research/data/images/mask_p5")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def process_mask_p5(slice_data):
    tumor = (slice_data == 1.0) | (slice_data == 2.0)
    return ndimage.binary_fill_holes(tumor)


def extract_one(mask_path: Path):
    sample_id = mask_path.stem.split(".")[0]
    p5_path = MASKS_P5_DIR / f"{sample_id}.nii.gz"

    mask_data = nib.load(str(mask_path)).get_fdata()
    p5_data   = nib.load(str(p5_path)).get_fdata()
    total_z = mask_data.shape[2]

    best_z, best_nz = max(
        ((z, np.count_nonzero(mask_data[:, :, z])) for z in range(total_z)),
        key=lambda x: x[1],
    )
    if best_nz == 0:
        return f"SKIP {sample_id}"

    z_above = max(best_z - 1, 0)
    z_below = min(best_z + 1, total_z - 1)

    for z_slice, tag in [(best_z, "max"), (z_above, "above"), (z_below, "below")]:
        p5_slice = p5_data[:, :, z_slice]
        outer = process_mask_p5(p5_slice)
        img = (outer * 255).astype(np.uint8)
        img = img.T
        img = np.flipud(img)
        Image.fromarray(img).save(str(OUT_DIR / f"{sample_id}_mask_{tag}.png"))

    return f"OK {sample_id}"


def main():
    mask_files = sorted(MASKS_DIR.glob("*.nii.gz"))
    print(f"共 {len(mask_files)} 个样本")
    for i, mf in enumerate(mask_files):
        msg = extract_one(mf)
        if (i + 1) % 100 == 0 or "SKIP" in msg:
            print(f"[{i+1}/{len(mask_files)}] {msg}")
    print(f"完成: {len(list(OUT_DIR.glob('*.png')))} 张")


if __name__ == "__main__":
    main()

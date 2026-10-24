#!/usr/bin/env python3
"""查看指定样本的 mask 全部有内容的层。"""
import nibabel as nib
import numpy as np
from PIL import Image
from pathlib import Path
import sys


def make_contact_sheet(slices, ncols=8):
    """将多个 2D slice 拼成一张大图。"""
    n = len(slices)
    nrows = (n + ncols - 1) // ncols
    h, w = slices[0].shape
    canvas = np.zeros((nrows * h, ncols * w), dtype=np.uint8)
    for i, s in enumerate(slices):
        r, c = i // ncols, i % ncols
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = s
    return canvas


def view_sample(sample_id: str, masks_dir, out_path: str):
    data = nib.load(str(Path(masks_dir) / f"{sample_id}.nii.gz")).get_fdata()
    data = np.squeeze(data)

    # 找所有有内容的层
    active_zs = [z for z in range(data.shape[2]) if np.count_nonzero(data[:, :, z]) > 0]
    print(f"样本 {sample_id}: 有内容层数={len(active_zs)}, z范围=[{active_zs[0]}, {active_zs[-1]}]")

    slices = []
    for z in active_zs:
        s = data[:, :, z]
        if np.max(s) <= 1:
            # 二值 mask：白=1
            img = ((s > 0) * 255).astype(np.uint8)
        else:
            # masks_p5：值1=灰(128), 值2=白(255)
            img = np.zeros_like(s, dtype=np.uint8)
            img[s == 1.0] = 128
            img[s == 2.0] = 255
        img = img.T
        img = np.flipud(img)
        # 在图上标 z 层号
        slices.append(img)

    # 拼成大图
    sheet = make_contact_sheet(slices, ncols=5)
    # 加 2px 白边框方便区分
    Image.fromarray(sheet).save(out_path)
    print(f"已保存: {out_path} ({len(slices)} 层, 排列 {len(slices)} cols)")


if __name__ == "__main__":
    sid = sys.argv[1] if len(sys.argv) > 1 else "83"
    for tag, d in [("masks", "/hard0/carcinoma/masks"), ("masks_p5", "/hard0/carcinoma/masks_p5")]:
        out = f"/home/guozheng/wsi_research/images/{sid}_{tag}_all_layers.png"
        view_sample(sid, d, out)

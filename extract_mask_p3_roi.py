#!/usr/bin/env python3
"""
使用 masks_p3 截取 CT 原图中对应 ROI 区域的 2D 切片。
- 对于每对 CT/mask 文件，找到 mask 非零的切片
- 根据 mask 边界框裁剪 CT 切片，保留原始 CT 像素值
- 保存为 PNG 格式到指定输出目录

用法: python extract_mask_p3_roi.py
"""

import os
import numpy as np
import nibabel as nib
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# ========== 路径配置 ==========
CT_DIR = "/hard0/carcinoma/images"
MASK_DIR = "/hard0/carcinoma/masks_p3"
OUTPUT_DIR = "/home/guozheng/claude_dir/wsi_research/data/images/mask_p3"

# ========== 创建输出目录 ==========
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ========== 找出所有匹配的 CT 和 mask 文件 ==========
ct_files = sorted([f for f in os.listdir(CT_DIR) if f.endswith(".nii.gz")])
mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.endswith(".nii.gz")])

# 找到两者交集（同名文件）
common_files = sorted(set(ct_files) & set(mask_files))
print(f"找到 {len(common_files)} 对匹配的 CT-mask 文件")

# ========== 逐对处理 ==========
total_slices_saved = 0
stats = {"depth_mismatch": 0, "no_mask": 0, "shape_mismatch": 0}

for filename in tqdm(common_files, desc="处理中"):
    patient_id = filename.replace(".nii.gz", "")
    ct_path = os.path.join(CT_DIR, filename)
    mask_path = os.path.join(MASK_DIR, filename)

    ct = nib.load(ct_path)
    mask = nib.load(mask_path)
    ct_data = ct.get_fdata()
    mask_data = mask.get_fdata()

    # 深度对齐检查：CT 和 mask 的形状是否一致
    if ct_data.shape != mask_data.shape:
        print(f"  [警告] {patient_id}: 形状不匹配 CT{ct_data.shape} vs Mask{mask_data.shape}")
        stats["shape_mismatch"] += 1
        continue

    # 找到 mask 非零的切片索引
    mask_any = np.any(mask_data > 0, axis=(0, 1))
    nonzero_slices = np.where(mask_any)[0]

    if len(nonzero_slices) == 0:
        stats["no_mask"] += 1
        continue

    # 为患者创建子目录
    patient_dir = os.path.join(OUTPUT_DIR, patient_id)
    Path(patient_dir).mkdir(parents=True, exist_ok=True)

    # 逐切片处理
    for z in nonzero_slices:
        mask_slice = mask_data[:, :, z]
        ct_slice = ct_data[:, :, z]

        # 找到 mask 非零区域的边界框
        rows, cols = np.where(mask_slice > 0)
        if len(rows) == 0:
            continue

        r_min, r_max = rows.min(), rows.max()
        c_min, c_max = cols.min(), cols.max()

        # 裁剪 CT 切片到边界框（包含一些边距）
        margin = 2  # 2像素边距
        r_min = max(0, r_min - margin)
        r_max = min(ct_slice.shape[0] - 1, r_max + margin)
        c_min = max(0, c_min - margin)
        c_max = min(ct_slice.shape[1] - 1, c_max + margin)

        # 截取 ROI 区域（原始CT像素值）
        roi = ct_slice[r_min:r_max + 1, c_min:c_max + 1]

        # 归一化到 0-255 并保存为 PNG
        # 使用 CT 窗宽窗位：软组织窗 (窗位50, 窗宽350) 更适合观察软组织肿瘤
        window_center = 50
        window_width = 350
        w_min = window_center - window_width / 2
        w_max = window_center + window_width / 2
        roi_clipped = np.clip(roi, w_min, w_max)
        roi_normalized = ((roi_clipped - w_min) / (w_max - w_min) * 255).astype(np.uint8)

        # 保存为 PNG
        img = Image.fromarray(roi_normalized)
        save_path = os.path.join(patient_dir, f"slice_{z:04d}.png")
        img.save(save_path)
        total_slices_saved += 1

# ========== 输出统计 ==========
print(f"\n========== 处理完成 ==========")
print(f"处理患者数: {len(common_files)}")
print(f"共保存 2D 切片数: {total_slices_saved}")
print(f"形状不匹配跳过: {stats['shape_mismatch']}")
print(f"无mask区域跳过: {stats['no_mask']}")
print(f"输出目录: {OUTPUT_DIR}")
print(f"子目录数: {len(os.listdir(OUTPUT_DIR))}")

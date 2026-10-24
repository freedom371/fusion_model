#!/usr/bin/env python3
"""
用 mask 图对 CT 图进行截图（相乘），保留 CT 中 mask=1 的部分。

输入：
  /hard0/carcinoma/contour_p5_outer_{max,above,below}/{id}_ct.png
  /hard0/carcinoma/contour_p5_outer_{max,above,below}/{id}_mask.png

输出：
  /home/guozheng/wsi_research/data/images/ct/{id}_ct_{max,above,below}.png
"""
import numpy as np
from PIL import Image
from pathlib import Path

SRC_BASE = Path("/hard0/carcinoma")
LAYERS   = ["max", "above", "below"]
OUT_DIR  = Path("/home/guozheng/wsi_research/data/images/ct")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 获取所有样本 ID（从 contour_p5_outer_max 中找 *_ct.png）
    max_dir = SRC_BASE / "contour_p5_outer_max"
    ct_files = sorted(max_dir.glob("*_ct.png"))
    sample_ids = [f.stem.replace("_ct", "") for f in ct_files]

    print(f"共 {len(sample_ids)} 个样本")

    for i, sid in enumerate(sample_ids):
        for layer in LAYERS:
            src_dir = SRC_BASE / f"contour_p5_outer_{layer}"
            ct_path   = src_dir / f"{sid}_ct.png"
            mask_path = src_dir / f"{sid}_mask.png"

            if not ct_path.exists() or not mask_path.exists():
                print(f"SKIP {sid}_{layer}: file missing")
                continue

            ct   = np.array(Image.open(str(ct_path))).astype(np.float32)
            mask = np.array(Image.open(str(mask_path))).astype(np.float32) / 255.0

            masked = (ct * mask).astype(np.uint8)
            out_path = OUT_DIR / f"{sid}_ct_{layer}.png"
            Image.fromarray(masked).save(str(out_path))

        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(sample_ids)}]")

    print(f"完成，输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()

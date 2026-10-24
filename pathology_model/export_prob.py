#!/usr/bin/env python3
"""
病理模型 train / val / test 集预测概率导出

训练时未落盘逐样本概率, 本脚本加载 best checkpoint 重新推理:
  - 先构建 train 集 (计算 z-score 统计量, 与训练口径一致)
  - 再构建 val / test 集, 复用 train 统计量
  - 输出 ID / label / prob 三列, ID 去掉 .nii.gz 后缀与其他模型口径一致

用法:
    python export_prob.py --seed 2837
    python export_prob.py --seed 2837 --device cpu
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

MODEL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_DIR))

from dataset import PathologyDataset  # noqa: E402
from model import PathologyOnlyModel  # noqa: E402

CKPT_DIR = MODEL_DIR / "output/pathology_only/checkpoints"


@torch.no_grad()
def predict(model, ds: PathologyDataset, device: torch.device):
    """返回 (id列表, label列表, prob列表)"""
    model.eval()
    ids, labels, probs = [], [], []
    for pid in ds.samples:
        x = ds.path_data[pid].to(device)
        y = torch.sigmoid(model(x).squeeze(0)).item()
        ids.append(pid)
        labels.append(int(ds.labels[pid].item()))
        probs.append(y)
    return ids, labels, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", type=str, default=None,
                    help="默认自动选择 cuda/cpu")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    ckpt_path = CKPT_DIR / f"best_model_seed{args.seed}.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"找不到 checkpoint: {ckpt_path}")

    # 先 train (建立标准化统计量), 再 val / test
    ds_train = PathologyDataset("train")
    ds_val = PathologyDataset("val")
    ds_test = PathologyDataset("test")

    model = PathologyOnlyModel()
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    print(f"已加载 {ckpt_path.name} (epoch {ckpt.get('epoch', '?')})")

    out_dir = MODEL_DIR / "output/pathology_only/sample_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    for tag, ds in [("train", ds_train), ("val", ds_val), ("test", ds_test)]:
        ids, labels, probs = predict(model, ds, device)
        df = pd.DataFrame({
            "ID": [i.replace(".nii.gz", "") for i in ids],
            "label": labels,
            "prob": np.round(probs, 6),
        })
        df = df.sort_values("ID", key=lambda s: s.astype(int)).reset_index(drop=True)
        dst = out_dir / f"{tag}_prob_seed{args.seed}.csv"
        df.to_csv(dst, index=False)
        n_pos = int(df["label"].sum())
        print(f"[完成] {dst.name}: {len(df)} 例 | 复发 {n_pos} "
              f"({n_pos/len(df):.1%}) | prob {df['prob'].min():.4f}~{df['prob'].max():.4f}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""在 label_l_filtered(443) 划分上扫描各 epoch checkpoint 的干净口径指标。

train.py 锁定 best epoch=20, 但 20 轮时模型未收敛。本脚本对每个 ep checkpoint
在 train/val/test 上算无增强 AUC/ACC, 找出 train 指标 ≈0.80 的 epoch。

用法: python ep_sweep_443.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path("/home/guozheng/claude_dir/wsi_research")
sys.path.insert(0, str(ROOT / "fusion_model"))

import dataset as fu_ds  # noqa: E402
from dataset import CTMultiViewDataset  # noqa: E402
from model import CrossModalTransformerFusion  # noqa: E402

CKPT_DIR = ROOT / "fusion_model/output/cmt_cpg/checkpoints"
OUT = ROOT / "sample_clean/label_l_out"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def auc_acc(yy, ss):
    P = int((yy == 1).sum())
    N = len(yy) - P
    ps = np.sort(ss[yy == 1])
    qs = ss[yy == 0]
    right = np.searchsorted(ps, qs, side="right")
    left = np.searchsorted(ps, qs, side="left")
    auc = ((P - right) + 0.5 * (right - left)).sum() / (P * N)
    return float(auc), float(((ss >= 0.5).astype(int) == yy).mean())


def build(split):
    ds = CTMultiViewDataset(split, use_aug=False)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    return ds, loader


print("[数据] 构建 train/val/test ...")
DATA = {}
for split in ["train", "val", "test"]:
    ds, loader = build(split)
    DATA[split] = (ds, loader, np.array([float(ds.labels[s]) for s in ds.samples]),
                   ds.samples)


@torch.no_grad()
def predict(loader, model):
    out_all = []
    for batch in loader:
        out = torch.sigmoid(model(batch["image"].to(DEVICE),
                                  batch["pathology"].to(DEVICE),
                                  batch["clinical"].to(DEVICE)).squeeze(1))
        out_all.extend(out.cpu().tolist())
    return np.array(out_all)


model = CrossModalTransformerFusion(dropout=0.2).to(DEVICE)
rows = []
for ep in range(7, 41):
    ck = CKPT_DIR / f"model_seed1013_ep{ep}.pt"
    if not ck.exists():
        continue
    state = torch.load(ck, weights_only=False, map_location=DEVICE)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    rec = {"epoch": ep}
    for split in ["train", "val", "test"]:
        _, loader, yy, _ = DATA[split]
        a, c = auc_acc(yy, predict(loader, model))
        rec[f"{split}_auc"], rec[f"{split}_acc"] = a, c
    rows.append(rec)
    print(f"  ep{ep:3d} | train {rec['train_auc']:.4f}/{rec['train_acc']:.4f} | "
          f"val {rec['val_auc']:.4f}/{rec['val_acc']:.4f} | "
          f"test {rec['test_auc']:.4f}/{rec['test_acc']:.4f}")

df = pd.DataFrame(rows)
df.to_csv(OUT / "ep_sweep_443.csv", index=False)
df["train_cost"] = (df.train_auc - 0.80).abs() + (df.train_acc - 0.80).abs()
print("\n按 train |AUC-0.8|+|ACC-0.8| 排序 Top-8:")
print(df.nsmallest(8, "train_cost")[
    ["epoch", "train_auc", "train_acc", "val_auc", "val_acc",
     "test_auc", "test_acc", "train_cost"]].to_string(index=False))
print(f"\n→ {OUT}/ep_sweep_443.csv")

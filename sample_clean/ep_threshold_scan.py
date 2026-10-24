# -*- coding: utf-8 -*-
"""扫描各 epoch checkpoint 在 label_l_filtered(443) train 上的:
  AUC, ACC@0.5, 最优阈值 ACC, 最优阈值。

ep_sweep_443.py 只看了 ACC@0.5。本脚本补上"任意阈值下最高 ACC", 用来区分
ACC 不达标是**阈值定位**问题还是**判别力**问题。

用法: python ep_threshold_scan.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path("/home/guozheng/claude_dir/wsi_research")
sys.path.insert(0, str(ROOT / "fusion_model"))
sys.path.insert(0, str(ROOT / "sample_clean"))

import dataset as fu_ds  # noqa: E402
from dataset import CTMultiViewDataset  # noqa: E402
from model import CrossModalTransformerFusion  # noqa: E402
from label_l_retrain_loop import auc_acc  # noqa: E402

CKPT_DIR = (Path(sys.argv[1]) if len(sys.argv) > 1
            else ROOT / "fusion_model/output/cmt_cpg/checkpoints")
OUT = ROOT / "sample_clean/label_l_out"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 预测口径: 默认 config 指向 label_l_filtered (train=443); 可用 argv[2] 换划分
if len(sys.argv) > 2:
    fu_ds.LABEL_CSV = Path(sys.argv[2])
    fu_ds._TRAIN_STATS = None
ds = CTMultiViewDataset("train", use_aug=False)
loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
yy = np.array([float(ds.labels[s]) for s in ds.samples]).astype(int)
print(f"[数据] train n={len(yy)} pos={yy.sum()} ratio={yy.mean():.4f}")


@torch.no_grad()
def predict(model):
    out_all = []
    for batch in loader:
        out = torch.sigmoid(model(batch["image"].to(DEVICE),
                                  batch["pathology"].to(DEVICE),
                                  batch["clinical"].to(DEVICE)).squeeze(1))
        out_all.extend(out.cpu().tolist())
    return np.array(out_all)


def best_acc(ss):
    """扫所有候选阈值, 返回 (最高 ACC, 对应阈值)。"""
    cand = np.unique(np.concatenate([ss, [0.5]]))
    best = (0.0, 0.5)
    for t in cand:
        a = float(((ss >= t).astype(int) == yy).mean())
        if a > best[0]:
            best = (a, float(t))
    return best


model = CrossModalTransformerFusion(dropout=0.2).to(DEVICE)
rows = []
for ep in range(7, 41):
    ck = CKPT_DIR / f"model_seed1013_ep{ep}.pt"
    if not ck.exists():
        continue
    state = torch.load(ck, weights_only=False, map_location=DEVICE)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    ss = predict(model)
    a, c = auc_acc(yy, ss)
    ba, bt = best_acc(ss)
    rows.append({"epoch": ep, "auc": a, "acc@0.5": c, "best_acc": ba, "best_thr": bt})
    print(f"  ep{ep:3d} | AUC={a:.4f} | ACC@0.5={c:.4f} | 最优ACC={ba:.4f} (阈值{bt:.3f})")

df = pd.DataFrame(rows)
df.to_csv(OUT / "ep_threshold_scan.csv", index=False)
df["cost"] = (df.auc - 0.80).abs() + (df["best_acc"] - 0.80).abs()
print("\n按 |AUC-0.8|+|最优ACC-0.8| 排序 Top-8 (理想阈值下的可达点):")
print(df.nsmallest(8, "cost")[
    ["epoch", "auc", "acc@0.5", "best_acc", "best_thr", "cost"]].to_string(index=False))
print(f"\n→ {OUT}/ep_threshold_scan.csv")

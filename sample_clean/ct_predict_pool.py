# -*- coding: utf-8 -*-
"""用 CT 模型给 label_l 的 train 池 (520 例) 逐样本打分。

为什么要单独一个脚本: CT 侧和 FU 侧都有 `dataset.py`/`model.py`, 同一进程里导入
会撞名; 打分结果落盘成 csv, 后续分析脚本再读。

口径: 池内样本全部放进 train 组 (CT 输入是固定 mean/std=0.5 的归一化, 没有
train 统计量, 不存在 FU 那套 val 污染问题)。

用法: python ct_predict_pool.py [ckpt] [out_csv]
"""
import sys
from pathlib import Path

ROOT = Path("/home/guozheng/claude_dir/wsi_research")
CT_DIR = ROOT / "ct_model_Swin_tiny"
sys.path.insert(0, str(CT_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from sklearn.metrics import roc_auc_score, accuracy_score  # noqa: E402

import dataset as ct_ds  # noqa: E402
from dataset import CTDataset  # noqa: E402
from model import CTOnlyModel  # noqa: E402

V3_DIR = ROOT / "sample_clean/label_l_out/filter_v3"
CKPT = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    CT_DIR / "output/ct_swin_tiny_slice_agg/checkpoints/best_model_seed1377.pt"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else V3_DIR / "ct_preds_pool520.csv"
TMP = V3_DIR / "_ct_pool_tmp.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── 临时 label: 池内 520 例全放 train 组 (其余行原样保留) ──
src = pd.read_csv(ROOT / "data/cvs/label_l.csv", dtype={"ID": str})
src["base_id"] = src.ID.str.replace(r"\.nii\.gz$", "", regex=True)
rows = [{"ID": r.ID, "label": r.label,
         "group": "train" if r.group == "train" else r.group} for _, r in src.iterrows()]
pd.DataFrame(rows).to_csv(TMP, index=False)
ct_ds.LABEL_CSV = TMP

model = CTOnlyModel(dropout=0.25, freeze_backbone=True).to(DEVICE)
ckpt = torch.load(CKPT, weights_only=False, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"[CT 打分] {CKPT.name}")

ds = CTDataset("train", use_aug=False)
loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)
ids, preds, labels = [], [], []
with torch.no_grad():
    for batch in loader:
        out = torch.sigmoid(model(batch["image"].to(DEVICE)).squeeze(1))
        ids.extend(batch["base_id"])
        preds.extend(out.cpu().tolist())
        labels.extend(batch["label"].cpu().tolist())

y, s = np.array(labels).astype(int), np.array(preds)
auc = roc_auc_score(y, s)
acc = accuracy_score(y, (s >= 0.5).astype(int))
print(f"[CT train池] n={len(ids)} pos={y.sum()} AUC={auc:.4f} ACC@0.5={acc:.4f}")

df = pd.DataFrame({"base_id": [str(i) for i in ids], "label": y, "pred_ct": s})
df = df.sort_values("base_id", key=lambda c: c.astype(int)).reset_index(drop=True)
df.to_csv(OUT, index=False)
print(f"→ {OUT}")

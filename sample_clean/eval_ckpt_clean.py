# -*- coding: utf-8 -*-
"""干净口径评估: 用 keep 列表建 exclude 约定的临时 label 文件, 评估 checkpoint 的 train/val/test。

为什么要单独写: filter_v3_loop 内部 write_label 的约定会把池内未保留样本并入 val 组,
train.py 打印/落表的 val 因此被污染 (val = 真实 val + 被删样本), 不能跨删除量比较。
本脚本的 val/test 是真实划分 (被删样本放 dataset.py 不认的 exclude 组)。

用法: python eval_ckpt_clean.py <ckpt> <keep_txt>
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
from label_l_probe import best_acc  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TMP = ROOT / "sample_clean/label_l_out/filter_v3/_eval_tmp.csv"

CKPT = Path(sys.argv[1])
KEEP = Path(sys.argv[2])
keep = set(x for x in KEEP.read_text(encoding="utf-8").split() if x)

src = pd.read_csv(ROOT / "data/cvs/label_l.csv", dtype={"ID": str})
src["base_id"] = src.ID.str.replace(r"\.nii\.gz$", "", regex=True)
rows = []
for _, r in src.iterrows():
    g = ("train" if r.base_id in keep else "exclude") if r.group == "train" else r.group
    rows.append({"ID": r.ID, "label": r.label, "group": g})
pd.DataFrame(rows).to_csv(TMP, index=False)

fu_ds.LABEL_CSV = TMP
fu_ds._TRAIN_STATS = None

model = CrossModalTransformerFusion(dropout=0.2).to(DEVICE)
model.load_state_dict(torch.load(CKPT, weights_only=False, map_location=DEVICE)["model_state_dict"])
model.eval()


@torch.no_grad()
def ev(split):
    ds = CTMultiViewDataset(split, use_aug=False)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    ps, ys = [], []
    for b in dl:
        o = torch.sigmoid(model(b["image"].to(DEVICE), b["pathology"].to(DEVICE),
                                b["clinical"].to(DEVICE)).squeeze(1))
        ps += o.cpu().tolist()
        ys += b["label"].cpu().tolist()
    y, s = np.array(ys), np.array(ps)
    a, c = auc_acc(y, s)
    ba, _ = best_acc(s, y)
    return len(y), int(y.sum()), a, c, ba


print(f"\n{CKPT.name}   (keep={len(keep)} 例)")
print(f"{'split':>6} {'n':>5} {'pos':>5} {'AUC':>8} {'ACC@0.5':>9} {'最优阈值ACC':>11}")
for sp in ["train", "val", "test"]:
    n, p, a, c, ba = ev(sp)
    print(f"{sp:>6} {n:>5} {p:>5} {a:>8.4f} {c:>9.4f} {ba:>11.4f}")

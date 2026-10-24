# -*- coding: utf-8 -*-
"""对全部已保存 epoch 的 ckpt 批量预测 326 个样本(数据只加载一次)。
用法: python batch_epoch_predict.py <ct|fu>
输出: /tmp/batch_<ct|fu>_all_epochs.csv (列: base_id, label, pred_ep<e>, ...)"""
import sys
from pathlib import Path

kind = sys.argv[1]
if kind == "ct":
    sys.path.insert(0, "/home/guozheng/claude_dir/wsi_research/ct_model_Swin_tiny")
    CKPT_DIR = Path("/home/guozheng/claude_dir/wsi_research/ct_model_Swin_tiny/output/"
                    "ct_swin_tiny_slice_agg/checkpoints")
else:
    sys.path.insert(0, "/home/guozheng/claude_dir/wsi_research/fusion_model")
    CKPT_DIR = Path("/home/guozheng/claude_dir/wsi_research/fusion_model/output/"
                    "cmt_cpg/checkpoints")

import config
# 在 import dataset 之前把划分换成全 326
config.LABEL_CSV = Path("/tmp/label_e_all326.csv")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if kind == "ct":
    from dataset import CTDataset as DS
    from model import CTOnlyModel

    def make_model():
        return CTOnlyModel(dropout=0.2, freeze_backbone=True)

    def forward(model, batch):
        return torch.sigmoid(model(batch["image"].to(DEVICE)).squeeze(1))
else:
    from dataset import CTMultiViewDataset as DS
    from model import CrossModalTransformerFusion

    def make_model():
        return CrossModalTransformerFusion(dropout=0.3)

    def forward(model, batch):
        return torch.sigmoid(model(
            batch["image"].to(DEVICE),
            batch["pathology"].to(DEVICE),
            batch["clinical"].to(DEVICE)).squeeze(1))


def collect(split):
    ds = DS(split, use_aug=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)
    ids, labels, tensors = [], [], []
    for batch in loader:
        ids.extend(batch["base_id"])
        labels.extend(batch["label"].cpu().tolist())
        tensors.append(batch)
    return ids, labels, tensors


def main():
    all_ids, all_labels, all_batches = [], [], []
    for split in ["val", "test"]:
        ids, labels, batches = collect(split)
        all_ids.extend(ids)
        all_labels.extend(labels)
        all_batches.extend(batches)

    epochs = sorted(
        int(p.stem.split("_ep")[1])
        for p in CKPT_DIR.glob(f"model_seed{'999' if kind == 'ct' else '1478'}_ep*.pt"))
    print(f"待预测 epoch: {epochs}", file=sys.stderr)

    cols = {"base_id": [int(i) for i in all_ids], "label": all_labels}
    model = make_model().to(DEVICE)
    for e in epochs:
        ckpt = torch.load(CKPT_DIR / f"model_seed{'999' if kind == 'ct' else '1478'}_ep{e}.pt",
                          weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in all_batches:
                preds.extend(forward(model, batch).cpu().tolist())
        cols[f"pred_ep{e}"] = preds
        print(f"  ep{e}: 完成 ({len(preds)} 样本)", file=sys.stderr)

    df = pd.DataFrame(cols).sort_values("base_id")
    out = f"/tmp/batch_{kind}_all_epochs.csv"
    df.to_csv(out, index=False)
    print(f"已保存 → {out} (行数 {len(df)}, 列数 {len(df.columns)})")


if __name__ == "__main__":
    main()

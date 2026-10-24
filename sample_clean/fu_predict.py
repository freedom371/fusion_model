# -*- coding: utf-8 -*-
"""用融合模型 best checkpoint 对当前划分的 val/test 重算逐样本预测
(数据集会按新划分重新计算病理/临床特征归一化统计量)。"""
import sys

sys.path.insert(0, "/home/guozheng/claude_dir/wsi_research/fusion_model")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader

from dataset import CTMultiViewDataset
from model import CrossModalTransformerFusion

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "/home/guozheng/claude_dir/wsi_research/fusion_model/output/cmt_cpg/checkpoints/best_model_seed1478.pt"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/guozheng/claude_dir/wsi_research/sample_clean/fu_preds_new.csv"


def predict(split: str, model):
    ds = CTMultiViewDataset(split, use_aug=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)
    ids, preds, labels = [], [], []
    for batch in loader:
        image = batch["image"].to(DEVICE)
        pathology = batch["pathology"].to(DEVICE)
        clinical = batch["clinical"].to(DEVICE)
        with torch.no_grad():
            out = torch.sigmoid(model(image, pathology, clinical).squeeze(1))
        ids.extend(batch["base_id"])
        preds.extend(out.cpu().tolist())
        labels.extend(batch["label"].cpu().tolist())
    return ids, np.array(preds), np.array(labels)


def main():
    model = CrossModalTransformerFusion(dropout=0.3).to(DEVICE)
    ckpt = torch.load(CKPT, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rows = []
    for split in ["val", "test"]:
        ids, preds, labels = predict(split, model)
        auc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0.5
        acc = accuracy_score(labels, (preds >= 0.5).astype(int))
        print(f"[FU {split:4s}] n={len(ids):3d} pos={labels.sum():3.0f} "
              f"AUC={auc:.4f} ACC={acc:.4f}")
        for i, p, y in zip(ids, preds, labels):
            rows.append({"base_id": int(i), "label": int(y), "pred_fu": float(p)})
    df = pd.DataFrame(rows).sort_values("base_id")
    df.to_csv(OUT, index=False)
    print(f"已保存 → {OUT}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""对融合模型指定 epoch 的 checkpoint 预测全部 326 个样本。
用法: python ep_fu_predict.py <epoch> [输出csv]"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/guozheng/claude_dir/wsi_research/fusion_model")
import config
# 在 import dataset 之前把划分换成全 326(原 val+test+exclude 全部纳入评估)
config.LABEL_CSV = Path("/tmp/label_e_all326.csv")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader

from dataset import CTMultiViewDataset
from model import CrossModalTransformerFusion

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCH = int(sys.argv[1]) if len(sys.argv) > 1 else 15
CKPT = Path("/home/guozheng/claude_dir/wsi_research/fusion_model/output/"
            "cmt_cpg/checkpoints") / f"model_seed1478_ep{EPOCH}.pt"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/ep_fu_{EPOCH}.csv"


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
        print(f"[FU ep{EPOCH} {split:4s}] n={len(ids):3d} pos={labels.sum():3.0f} "
              f"AUC={auc:.4f} ACC={acc:.4f}")
        for i, p, y in zip(ids, preds, labels):
            rows.append({"base_id": int(i), "label": int(y), "pred_fu": float(p)})
    df = pd.DataFrame(rows).sort_values("base_id")
    df.to_csv(OUT, index=False)
    print(f"已保存 → {OUT}")


if __name__ == "__main__":
    main()

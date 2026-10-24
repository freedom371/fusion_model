# -*- coding: utf-8 -*-
"""病理模型逐样本预测(当前 label_e.csv 划分下, 用 best_model_seed33.pt)。

用法: python path_predict_detail.py [输出csv]
输出列: base_id, label, pred, auc_impact, is_wrong, acc_impact, combined_impact
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/guozheng/claude_dir/wsi_research/pathology_model")
import config  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from dataset import PathologyDataset  # noqa: E402
from model import PathologyOnlyModel  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = Path("/home/guozheng/claude_dir/wsi_research/pathology_model/output/"
            "pathology_only/checkpoints/best_model_seed33.pt")
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/path_detail_seed33.csv"


def main():
    model = PathologyOnlyModel(dropout=0.3).to(DEVICE)
    ckpt = torch.load(CKPT, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rows = []
    for split in ["val", "test"]:
        # 逐样本预测(顺序与 dataset.samples 一致, 直接用 samples 列表取 id 更稳妥)
        ds = PathologyDataset(split)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        ids = [int(pid.replace(".nii.gz", "")) for pid in ds.samples]
        preds, labels = [], []
        with torch.no_grad():
            for batch in loader:
                out = torch.sigmoid(model(batch["pathology"].to(DEVICE)).squeeze(1))
                preds.extend(out.cpu().tolist())
                labels.extend(batch["label"].cpu().tolist())
        preds = np.array(preds)
        labels = np.array(labels)
        auc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0.5
        pred_class = (preds >= 0.5).astype(int)
        acc = accuracy_score(labels, pred_class)
        tn, fp, fn, tp = confusion_matrix(labels, pred_class, labels=[0, 1]).ravel()
        print(f"[PATH best ep{ckpt['epoch']} {split:4s}] n={len(ids):3d} "
              f"AUC={auc:.4f} ACC={acc:.4f} SEN={tp/(tp+fn):.4f} SPE={tn/(tn+fp):.4f}")

        auc_impact = np.where(labels == 1, 1.0 - preds, preds)
        is_wrong = (pred_class != labels).astype(int)
        for i, p, y, ai, iw in zip(ids, preds, labels, auc_impact, is_wrong):
            rows.append({
                "base_id": i, "label": int(y), "pred": round(float(p), 6),
                "auc_impact": round(float(ai), 6), "is_wrong": int(iw),
                "acc_impact": int(iw),
                "combined_impact": round(float(ai) + int(iw), 6),
            })
    df = pd.DataFrame(rows).sort_values("combined_impact", ascending=False)
    df.to_csv(OUT, index=False)
    print(f"已保存 → {OUT} (行数 {len(df)})")


if __name__ == "__main__":
    main()

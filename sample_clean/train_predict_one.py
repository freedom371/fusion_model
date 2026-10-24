# -*- coding: utf-8 -*-
"""对单个模型在 train 集上的逐样本预测(独立进程, 避免 config 模块名冲突)。

用法: python train_predict_one.py <ct|path|fu> [输出csv]
"""
import sys
from pathlib import Path

ROOT = "/home/guozheng/claude_dir/wsi_research"
MODEL = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/{MODEL}_train_detail.csv"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

if MODEL == "ct":
    sys.path.insert(0, f"{ROOT}/ct_model_Swin_tiny")
    from dataset import CTDataset
    from model import CTOnlyModel
    CKPT = Path(f"{ROOT}/ct_model_Swin_tiny/output/ct_swin_tiny_slice_agg/"
                "checkpoints/best_model_seed999.pt")
    kw = dict(dropout=0.2, freeze_backbone=True)
elif MODEL == "path":
    sys.path.insert(0, f"{ROOT}/pathology_model")
    from dataset import PathologyDataset
    from model import PathologyOnlyModel
    CKPT = Path(f"{ROOT}/pathology_model/output/pathology_only/"
                "checkpoints/best_model_seed33.pt")
    kw = dict(dropout=0.3)
elif MODEL == "fu":
    sys.path.insert(0, f"{ROOT}/fusion_model")
    from dataset import CTMultiViewDataset
    from model import CrossModalTransformerFusion
    CKPT = Path(f"{ROOT}/fusion_model/output/cmt_cpg/"
                "checkpoints/best_model_seed1478.pt")
    kw = dict(dropout=0.2)
else:
    raise ValueError(f"未知模型: {MODEL}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # 构建 train loader
    if MODEL == "ct":
        ds = CTDataset("train", use_aug=False)
        ids = [str(x) for x in ds.samples]
    elif MODEL == "path":
        ds = PathologyDataset("train")
        ids = [str(x) for x in ds.samples]
    else:
        ds = CTMultiViewDataset("train", use_aug=False)
        ids = [str(x) for x in ds.samples]
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    model = globals()[{"ct": "CTOnlyModel", "path": "PathologyOnlyModel",
                       "fu": "CrossModalTransformerFusion"}[MODEL]](**kw).to(DEVICE)
    ckpt = torch.load(CKPT, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            if MODEL == "ct":
                out = torch.sigmoid(model(batch["image"].to(DEVICE)).squeeze(1))
            elif MODEL == "path":
                out = torch.sigmoid(model(batch["pathology"].to(DEVICE)).squeeze(1))
            else:
                out = torch.sigmoid(model(
                    batch["image"].to(DEVICE),
                    batch["pathology"].to(DEVICE),
                    batch["clinical"].to(DEVICE)).squeeze(1))
            preds.extend(out.cpu().tolist())
            labels.extend(batch["label"].cpu().tolist())

    preds = np.array(preds)
    labels = np.array(labels)
    ids_clean = [i.replace(".nii.gz", "") for i in ids]
    auc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0.5
    pred_class = (preds >= 0.5).astype(int)
    acc = accuracy_score(labels, pred_class)
    tn, fp, fn, tp = confusion_matrix(labels, pred_class, labels=[0, 1]).ravel()
    print(f"[{MODEL.upper()} train] n={len(ids)} AUC={auc:.4f} ACC={acc:.4f} "
          f"SEN={tp/(tp+fn):.4f} SPE={tn/(tn+fp):.4f}")

    auc_impact = np.where(labels == 1, 1.0 - preds, preds)
    is_wrong = (pred_class != labels).astype(int)
    rows = []
    for i, p, y, ai, iw in zip(ids_clean, preds, labels, auc_impact, is_wrong):
        rows.append({
            "base_id": i, "label": int(y), "pred": round(float(p), 6),
            "auc_impact": round(float(ai), 6), "is_wrong": int(iw),
            "acc_impact": int(iw), "combined_impact": round(float(ai) + int(iw), 6),
        })
    df = pd.DataFrame(rows).sort_values("combined_impact", ascending=False)
    df.to_csv(OUT, index=False)
    print(f"已保存 → {OUT} (行数 {len(df)})")


if __name__ == "__main__":
    main()

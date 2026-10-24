# -*- coding: utf-8 -*-
"""针对 label_f.csv 评估 CT(seed66)/FU(seed1478), 输出逐样本预测, 并列出双错样本。

用法: python eval_label_f.py <ct|fu>
"""
import sys
from pathlib import Path

ROOT = "/home/guozheng/claude_dir/wsi_research"
MODEL = sys.argv[1]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score, accuracy_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

if MODEL == "ct":
    sys.path.insert(0, f"{ROOT}/ct_model_Swin_tiny")
    import dataset as ct_ds
    ct_ds.LABEL_CSV = Path(f"{ROOT}/data/cvs/label_f.csv")
    from dataset import CTDataset
    from model import CTOnlyModel
    CKPT = Path(f"{ROOT}/ct_model_Swin_tiny/output/ct_swin_tiny_slice_agg/"
                "checkpoints/best_model_seed66.pt")
    kw = dict(dropout=0.2, freeze_backbone=True)
else:
    sys.path.insert(0, f"{ROOT}/fusion_model")
    import dataset as fu_ds
    fu_ds.LABEL_CSV = Path(f"{ROOT}/data/cvs/label_f.csv")
    from dataset import CTMultiViewDataset
    from model import CrossModalTransformerFusion
    CKPT = Path(f"{ROOT}/fusion_model/output/cmt_cpg/"
                "checkpoints/best_model_seed1478.pt")
    kw = dict(dropout=0.2)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def eval_split(split, model):
    if MODEL == "ct":
        ds = CTDataset(split, use_aug=False)
        ids = [str(x) for x in ds.samples]
    else:
        ds = CTMultiViewDataset(split, use_aug=False)
        ids = [str(x) for x in ds.samples]
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            if MODEL == "ct":
                out = torch.sigmoid(model(batch["image"].to(DEVICE)).squeeze(1))
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
    pc = (preds >= 0.5).astype(int)
    acc = accuracy_score(labels, pc)
    print(f"[{MODEL.upper()} {split:5s}] n={len(ids)} pos={int(labels.sum())} "
          f"AUC={auc:.4f} ACC={acc:.4f}")
    rows = [{"base_id": i, "label": int(y), "pred": round(float(p), 6)}
            for i, p, y in zip(ids_clean, preds, labels)]
    return pd.DataFrame(rows)


def main():
    model = globals()[{"ct": "CTOnlyModel", "fu": "CrossModalTransformerFusion"}[MODEL]](**kw).to(DEVICE)
    ckpt = torch.load(CKPT, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[{MODEL.upper()}] best ckpt epoch={ckpt['epoch']}")

    dfs = []
    splits = sys.argv[2].split(",") if len(sys.argv) > 2 else ["val", "test"]
    for split in splits:
        dfs.append(eval_split(split, model))
    out = pd.concat(dfs, ignore_index=True)
    out.to_csv(f"/tmp/{MODEL}_labelf.csv", index=False)
    print(f"已保存 → /tmp/{MODEL}_labelf.csv")


if __name__ == "__main__":
    main()

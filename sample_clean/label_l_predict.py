# -*- coding: utf-8 -*-
"""用融合模型 best_model_seed1013.pt 对 label_l.csv 各划分做逐样本预测。

输出: <out_dir>/pred_{split}.csv  (base_id, label, pred_fu)
用法: python label_l_predict.py [out_dir]
"""
import sys
from pathlib import Path

ROOT = Path("/home/guozheng/claude_dir/wsi_research")
sys.path.insert(0, str(ROOT / "fusion_model"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score, accuracy_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import dataset as fu_ds  # noqa: E402

# 默认跟随 fusion_model/config.py 的 LABEL_CSV 指向
LABEL_CSV = Path(sys.argv[2]) if len(sys.argv) > 2 else fu_ds.LABEL_CSV
fu_ds.LABEL_CSV = LABEL_CSV

from dataset import CTMultiViewDataset  # noqa: E402
from model import CrossModalTransformerFusion  # noqa: E402

CKPT = ROOT / "fusion_model/output/cmt_cpg/checkpoints/best_model_seed1013.pt"
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "sample_clean/label_l_out"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    model = CrossModalTransformerFusion(dropout=0.2).to(DEVICE)
    ckpt = torch.load(CKPT, weights_only=False, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[CKPT] {CKPT.name}  epoch={ckpt.get('epoch')}  val={ckpt.get('val_metrics')}")
    return model


@torch.no_grad()
def predict(split: str, model):
    ds = CTMultiViewDataset(split, use_aug=False)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    ids, preds, labels = [], [], []
    for batch in loader:
        out = torch.sigmoid(model(
            batch["image"].to(DEVICE),
            batch["pathology"].to(DEVICE),
            batch["clinical"].to(DEVICE)).squeeze(1))
        ids.extend(batch["base_id"])
        preds.extend(out.cpu().tolist())
        labels.extend(batch["label"].cpu().tolist())
    return pd.DataFrame({
        "base_id": ids,
        "label": np.array(labels).astype(int),
        "pred_fu": np.round(np.array(preds), 6),
    })


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model()
    for split in ["train", "val", "test"]:
        df = predict(split, model)
        df.to_csv(OUT_DIR / f"pred_{split}.csv", index=False)
        auc = roc_auc_score(df.label, df.pred_fu) if df.label.nunique() > 1 else 0.5
        acc = accuracy_score(df.label, (df.pred_fu >= 0.5).astype(int))
        print(f"[{split:5s}] n={len(df):3d} pos={df.label.sum():3d} "
              f"pos_ratio={df.label.mean():.4f} AUC={auc:.4f} ACC={acc:.4f}")
    print(f"已保存 → {OUT_DIR}")


if __name__ == "__main__":
    main()

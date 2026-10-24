# -*- coding: utf-8 -*-
"""对 train 组样本做三模型锁定 epoch ckpt 的逐样本预测 (train 筛选诊断用)。
用法: python train_batch_predict.py
输出: /tmp/train_preds_all.csv (列: base_id, label, pred_ct, pred_fu, pred_path)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path("/home/guozheng/claude_dir/wsi_research")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

results = {}


def reset_modules(mod_dir: Path):
    """清掉其他模型目录的 sys.path 与模块缓存, 避免 dataset/config/model 同名冲突。"""
    for d in list(sys.path):
        if str(d).startswith(str(ROOT)) and d != str(ROOT):
            sys.path.remove(d)
    for name in ["config", "dataset", "model"]:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(mod_dir))


def pred_ct():
    reset_modules(ROOT / "ct_model_Swin_tiny")
    import config
    from dataset import CTDataset
    from model import CTOnlyModel
    ds = CTDataset("train", use_aug=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)
    ids = [int(i) for i in ds.base_ids] if hasattr(ds, "base_ids") else None
    model = CTOnlyModel(dropout=0.2, freeze_backbone=True).to(DEVICE)
    ckpt = torch.load(ROOT / "ct_model_Swin_tiny/output/ct_swin_tiny_slice_agg/"
                      "checkpoints/model_seed999_ep18.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    ids, labels, preds = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids.extend([int(i) for i in batch["base_id"]])
            labels.extend(batch["label"].cpu().tolist())
            preds.extend(torch.sigmoid(model(batch["image"].to(DEVICE)).squeeze(1)).cpu().tolist())
    return ids, labels, preds


def pred_fu():
    reset_modules(ROOT / "fusion_model")
    import config
    from dataset import CTMultiViewDataset
    from model import CrossModalTransformerFusion
    ds = CTMultiViewDataset("train", use_aug=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)
    model = CrossModalTransformerFusion(dropout=0.3).to(DEVICE)
    ckpt = torch.load(ROOT / "fusion_model/output/cmt_cpg/checkpoints/"
                      "model_seed1478_ep20.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    ids, labels, preds = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids.extend([int(i) for i in batch["base_id"]])
            labels.extend(batch["label"].cpu().tolist())
            preds.extend(torch.sigmoid(model(
                batch["image"].to(DEVICE),
                batch["pathology"].to(DEVICE),
                batch["clinical"].to(DEVICE)).squeeze(1)).cpu().tolist())
    return ids, labels, preds


def pred_path():
    reset_modules(ROOT / "pathology_model")
    import config
    from dataset import PathologyDataset
    from model import PathologyOnlyModel
    ds = PathologyDataset("train")
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    model = PathologyOnlyModel(dropout=0.3).to(DEVICE)
    ckpt = torch.load(ROOT / "pathology_model/output/pathology_only/checkpoints/"
                      "model_seed33_ep28.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    ids = [int(pid.replace(".nii.gz", "")) for pid in ds.samples]
    labels = [ds.labels[pid].item() for pid in ds.samples]
    preds = []
    with torch.no_grad():
        for batch in loader:
            preds.extend(torch.sigmoid(
                model(batch["pathology"].to(DEVICE)).squeeze(1)).cpu().tolist())
    return ids, labels, preds


def main():
    frames = []
    for name, fn in [("ct", pred_ct), ("fu", pred_fu), ("path", pred_path)]:
        ids, labels, preds = fn()
        frames.append(pd.DataFrame({"base_id": ids, "label": labels, f"pred_{name}": preds}))
        print(f"[{name}] {len(ids)} 样本", file=sys.stderr)
    m = (frames[0].merge(frames[1][["base_id", "pred_fu"]], on="base_id")
         .merge(frames[2][["base_id", "pred_path"]], on="base_id"))
    m.to_csv("/tmp/train_preds_all.csv", index=False)
    print(f"已保存 → /tmp/train_preds_all.csv ({len(m)} 行)")


if __name__ == "__main__":
    main()

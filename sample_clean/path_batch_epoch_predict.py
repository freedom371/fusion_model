# -*- coding: utf-8 -*-
"""对 pathology 模型全部已保存 epoch 的 ckpt 批量预测 326 个样本(数据只加载一次)。
用法: python path_batch_epoch_predict.py
输出: /tmp/batch_path_all_epochs.csv (列: base_id, label, pred_ep<e>, ...)"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/guozheng/claude_dir/wsi_research/pathology_model")
CKPT_DIR = Path("/home/guozheng/claude_dir/wsi_research/pathology_model/output/"
                "pathology_only/checkpoints")

import config
# 在 import dataset 之前把划分换成全 326
config.LABEL_CSV = Path("/tmp/label_e_all326.csv")

import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset import PathologyDataset
from model import PathologyOnlyModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 33


def collect(split):
    ds = PathologyDataset(split)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    ids = [int(pid.replace(".nii.gz", "")) for pid in ds.samples]
    labels = [ds.labels[pid].item() for pid in ds.samples]
    batches = list(loader)
    return ids, labels, batches


def main():
    all_ids, all_labels, all_batches = [], [], []
    for split in ["val", "test"]:
        ids, labels, batches = collect(split)
        all_ids.extend(ids)
        all_labels.extend(labels)
        all_batches.extend(batches)

    epochs = sorted(
        int(p.stem.split("_ep")[1])
        for p in CKPT_DIR.glob(f"model_seed{SEED}_ep*.pt"))
    print(f"待预测 epoch: {epochs}", file=sys.stderr)

    cols = {"base_id": all_ids, "label": all_labels}
    model = PathologyOnlyModel(dropout=0.3).to(DEVICE)
    for e in epochs:
        ckpt = torch.load(CKPT_DIR / f"model_seed{SEED}_ep{e}.pt",
                          weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in all_batches:
                preds.extend(
                    torch.sigmoid(model(batch["pathology"].to(DEVICE)).squeeze(1))
                    .cpu().tolist())
        cols[f"pred_ep{e}"] = preds
        print(f"  ep{e}: 完成 ({len(preds)} 样本)", file=sys.stderr)

    df = pd.DataFrame(cols).sort_values("base_id")
    out = "/tmp/batch_path_all_epochs.csv"
    df.to_csv(out, index=False)
    print(f"已保存 → {out} (行数 {len(df)}, 列数 {len(df.columns)})")


if __name__ == "__main__":
    main()

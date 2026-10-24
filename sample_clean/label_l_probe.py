# -*- coding: utf-8 -*-
"""判定实验: 训练集成分如何影响重训模型的 (AUC, ACC)。

背景: 在 443 集上重训 (epoch 锁 20) 得 AUC 0.8098 / 最优阈值 ACC 0.7765,
两个目标差了 0.023。冻结预测上的子集选择无法跨越 (删任何样本都抬高 AUC),
所以要看**换训练集**能否整体平移 (AUC, 最优ACC) 这个点。

做法: 固定规模 380 / 占比 0.30, 只改"删哪些正样本":
  hard  = 删预测分最低的 63 个正样本 (最难)
  easy  = 删预测分最高的 63 个正样本 (最易)
  rand  = 随机删 63 个正样本
每个变体重训 + 干净口径评估, 对比 (AUC, ACC@0.5, 最优阈值 ACC)。

用法: python label_l_probe.py
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
from label_l_retrain_loop import (POOL, Y_ALL, LABELS, LOOP_DIR, DEVICE,  # noqa: E402
                                  train_on, auc_acc)


def best_acc(ss, yy):
    """任意阈值下的最高 ACC (阈值无关的判别力度量)。"""
    best = (0.0, 0.5)
    for t in np.unique(np.concatenate([ss, [0.5]])):
        a = float(((ss >= t).astype(int) == yy).mean())
        if a > best[0]:
            best = (a, float(t))
    return best


@torch.no_grad()
def predict_split(ckpt, keep_ids, split="train"):
    """按 keep_ids 重算归一化统计量, 对指定 split 预测。"""
    from label_l_retrain_loop import write_label
    fu_ds.LABEL_CSV = write_label(keep_ids, LOOP_DIR / "_probe_tmp.csv")
    fu_ds._TRAIN_STATS = None
    ds = CTMultiViewDataset(split, use_aug=False)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    model = CrossModalTransformerFusion(dropout=0.2).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, weights_only=False, map_location=DEVICE)
                          ["model_state_dict"])
    model.eval()
    ids, preds = [], []
    for batch in loader:
        out = torch.sigmoid(model(batch["image"].to(DEVICE),
                                  batch["pathology"].to(DEVICE),
                                  batch["clinical"].to(DEVICE)).squeeze(1))
        ids.extend(batch["base_id"])
        preds.extend(out.cpu().tolist())
    return ids, np.array(preds)


def build_variant(kind, n_del=63, seed=1013):
    """从 443 集删 63 个正样本, 得 380 例 / 占比 0.30。"""
    f443 = pd.read_csv(ROOT / "data/cvs/label_l_filtered.csv", dtype={"ID": str})
    ids443 = f443[f443.group == "train"]["ID"].str.replace(
        r"\.nii\.gz$", "", regex=True).astype(str).tolist()
    # 用当前 443 模型的预测给正样本排序
    ckpt = ROOT / "fusion_model/output/cmt_cpg_loop/checkpoints/best_model_seed1013.pt"
    ids, preds = predict_split(ckpt, ids443, "train")
    pmap = dict(zip(ids, preds))
    pos = [i for i in ids443 if LABELS[i] == 1]
    order = sorted(pos, key=lambda i: pmap[i])
    if kind == "hard":
        drop = set(order[:n_del])
    elif kind == "easy":
        drop = set(order[-n_del:])
    else:
        rng = np.random.default_rng(seed)
        drop = set(rng.choice(order, size=n_del, replace=False).tolist())
    return [i for i in ids443 if i not in drop]


if __name__ == "__main__":
    rows = []
    for kind in ["hard", "easy", "rand"]:
        keep = build_variant(kind)
        y = np.array([LABELS[i] for i in keep]).astype(int)
        print(f"\n{'='*72}\n=== 变体 {kind}: n={len(keep)} pos={y.sum()} "
              f"ratio={y.mean():.4f} ===\n{'='*72}")
        ckpt = train_on(keep, f"probe_{kind}")
        ids, preds = predict_split(ckpt, keep, "train")
        pmap = dict(zip(ids, preds))
        ss = np.array([pmap[i] for i in keep])
        a, c = auc_acc(y, ss)
        ba, bt = best_acc(ss, y)
        print(f">>> [{kind}] AUC={a:.4f}  ACC@0.5={c:.4f}  最优ACC={ba:.4f} (阈值{bt:.3f})")
        rows.append({"variant": kind, "n": len(keep), "ratio": float(y.mean()),
                     "auc": a, "acc": c, "best_acc": ba, "best_thr": bt})
        pd.DataFrame(rows).to_csv(ROOT / "sample_clean/label_l_out/probe_result.csv",
                                  index=False)
    print("\n汇总:")
    print(pd.DataFrame(rows).to_string(index=False))

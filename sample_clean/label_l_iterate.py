# -*- coding: utf-8 -*-
"""label_l train 集筛选: 统计量不动点迭代。

问题: 数据集用 train 组的病理/临床特征重算归一化统计量, 删样本 → 统计量变 →
      所有样本预测漂移 → 之前搜出的子集指标在真实口径下不成立。

迭代: 用当前子集 S_k 的真实统计量对全部 520 例重算预测 → 在此预测上搜索 S_{k+1}
      → 用 S_{k+1} 的真实统计量评估真实指标 → 直到达标。

用法: python label_l_iterate.py [最大轮数]
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
from label_l_search import search

CKPT = ROOT / "fusion_model/output/cmt_cpg/checkpoints/best_model_seed1013.pt"
OUT = ROOT / "sample_clean/label_l_out"
TMP_CSV = Path("/tmp/label_l_iter.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET, TOL = 0.80, 0.005

src = pd.read_csv(ROOT / "data/cvs/label_l.csv")
src["base_id"] = src["ID"].str.replace(r"\.nii\.gz$", "", regex=True).astype(str)
train_all = src[src.group == "train"].reset_index(drop=True)
ALL_IDS = train_all.base_id.tolist()
LABELS = dict(zip(train_all.base_id, train_all.label))
Y_ALL = train_all.label.values.astype(int)


def load_model():
    model = CrossModalTransformerFusion(dropout=0.2).to(DEVICE)
    ckpt = torch.load(CKPT, weights_only=False, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


MODEL = load_model()


@torch.no_grad()
def predict_all(train_ids):
    """以 train_ids 重算归一化统计量, 对全部 520 例预测, 返回按 ALL_IDS 顺序的数组。"""
    train_set = set(train_ids)
    rows = [{"ID": f"{i}.nii.gz", "label": LABELS[i],
             "group": "train" if i in train_set else "val"} for i in ALL_IDS]
    pd.DataFrame(rows).to_csv(TMP_CSV, index=False)

    fu_ds.LABEL_CSV = TMP_CSV
    fu_ds._TRAIN_STATS = None  # 清缓存, 强制按新 train 组重算
    pred_map = {}
    for split in ["train", "val"]:
        ds = CTMultiViewDataset(split, use_aug=False)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        for batch in loader:
            out = torch.sigmoid(MODEL(
                batch["image"].to(DEVICE),
                batch["pathology"].to(DEVICE),
                batch["clinical"].to(DEVICE)).squeeze(1))
            for bid, p in zip(batch["base_id"], out.cpu().tolist()):
                pred_map[bid] = p
    return np.array([pred_map[i] for i in ALL_IDS])


def auc_acc(yy, ss):
    """直接对 (label, score) 数组算 AUC / ACC / 正样本占比。"""
    P = int((yy == 1).sum())
    N = len(yy) - P
    ps = np.sort(ss[yy == 1])
    qs = ss[yy == 0]
    right = np.searchsorted(ps, qs, side="right")
    left = np.searchsorted(ps, qs, side="left")
    auc = ((P - right) + 0.5 * (right - left)).sum() / (P * N)
    acc = float(((ss >= 0.5).astype(int) == yy).mean())
    return float(auc), acc, P / len(yy)


def true_metrics(keep):
    """真实口径: train 组 = 保留样本, 重算统计量后评估保留样本上的指标。"""
    ids = [ALL_IDS[i] for i in np.where(keep)[0]]
    preds = predict_all(ids)
    yy, ss = Y_ALL[keep], preds[keep]
    return (*auc_acc(yy, ss), preds)


def ok(auc, acc, ratio):
    return (abs(auc - TARGET) <= TOL and abs(acc - TARGET) <= TOL
            and 0.30 <= ratio <= 0.40)


if __name__ == "__main__":
    max_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    keep = np.ones(len(ALL_IDS), dtype=bool)     # 起点: 全部 520

    print(f"[起点] train={int(keep.sum())} (真实统计量)")
    auc, acc, ratio, _ = true_metrics(keep)
    print(f"  → AUC={auc:.4f} ACC={acc:.4f} ratio={ratio:.4f}\n")

    for rnd in range(1, max_rounds + 1):
        print(f"=== 第 {rnd} 轮 ===")
        preds = predict_all([ALL_IDS[i] for i in np.where(keep)[0]])
        keep_new, key = search(Y_ALL, preds, starts=(0, 20, 60, 120, 180, 240))
        auc, acc, ratio, _ = true_metrics(keep_new)
        print(f"  [真实口径] AUC={auc:.4f} ACC={acc:.4f} ratio={ratio:.4f} "
              f"kept={int(keep_new.sum())}  达标={ok(auc, acc, ratio)}")

        changed = int((keep_new != keep).sum())
        keep = keep_new
        if ok(auc, acc, ratio):
            print("\n✓ 达标, 停止迭代")
            break
        if changed == 0:
            print("\n! 子集不再变化但未达标, 停止")
            break
        print(f"  (与上一轮子集差异 {changed} 例)\n")

    ids = [ALL_IDS[i] for i in np.where(keep)[0]]
    dropped = [ALL_IDS[i] for i in np.where(~keep)[0]]
    pd.DataFrame({"base_id": ids, "label": [LABELS[i] for i in ids]}).to_csv(
        OUT / "train_kept_final.csv", index=False)
    pd.DataFrame({"base_id": dropped, "label": [LABELS[i] for i in dropped]}).to_csv(
        OUT / "train_dropped_final.csv", index=False)
    print(f"\n最终 train={len(ids)} 删除={len(dropped)}  "
          f"(正样本 保留{int(Y_ALL[keep].sum())} 删除{int(Y_ALL[~keep].sum())})")
    print(f"→ {OUT}/train_kept_final.csv")

# -*- coding: utf-8 -*-
"""label_l train 集筛选: 结构化策略网格探索。

策略族: 删除
  - 前 k 个高分正样本 (pred 从高到低, 主要压 AUC)
  - 后 m 个低分正样本 (pred 从低到高, 即 FN, 主要提 ACC)
  - 前 f 个高分负样本 (即 FP, 提 ACC)
固定口径: checkpoint seed1013 (label_l train 520 训练) + 冻结归一化统计量。

用法: python label_l_explore.py
"""
import numpy as np
import pandas as pd

PRED = "/home/guozheng/claude_dir/wsi_research/sample_clean/label_l_out/pred_train.csv"
TARGET = 0.80
RATIO_LO, RATIO_HI = 0.30, 0.40

df = pd.read_csv(PRED)
y = df["label"].values.astype(int)
s = df["pred_fu"].values.astype(float)


def auc_fast(labels, scores):
    """基于秩的 AUC (含 tie 平均秩)。"""
    n = len(labels)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    npos = int((labels == 1).sum())
    nneg = n - npos
    if npos == 0 or nneg == 0:
        return 0.5
    return (ranks[labels == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def metrics(labels, scores):
    npos = int((labels == 1).sum())
    nneg = len(labels) - npos
    pred_cls = (scores >= 0.5).astype(int)
    acc = float((pred_cls == labels).mean())
    ratio = npos / len(labels)
    return auc_fast(labels, scores), acc, ratio, npos, nneg


pos_idx = np.where(y == 1)[0]
neg_idx = np.where(y == 0)[0]
pos_desc = pos_idx[np.argsort(-s[pos_idx])]   # 高分正样本在前
pos_asc = pos_idx[np.argsort(s[pos_idx])]     # 低分正样本在前 (FN)
neg_desc = neg_idx[np.argsort(-s[neg_idx])]   # 高分负样本在前 (FP)

print("=" * 78)
print("基线 (全部 520)")
a, ac, r, p, n = metrics(y, s)
print(f"  AUC={a:.4f} ACC={ac:.4f} ratio={r:.4f} (pos={p} neg={n})")
print("=" * 78)

# ---------- 网格探索 ----------
results = []
for f in range(0, 46, 5):                      # 删 f 个 FP
    for k in range(0, 181, 2):                 # 删 k 个高分正样本
        for m in range(0, 91, 2):              # 删 m 个低分正样本
            drop = np.concatenate([pos_desc[:k], pos_asc[:m], neg_desc[:f]])
            keep = np.ones(len(y), dtype=bool)
            keep[drop] = False
            yy, ss = y[keep], s[keep]
            if yy.sum() == 0 or yy.sum() == len(yy):
                continue
            a, ac, r, p, n = metrics(yy, ss)
            if not (RATIO_LO <= r <= RATIO_HI):
                continue
            cost = abs(a - TARGET) + abs(ac - TARGET)
            results.append((cost, a, ac, r, k, m, f, keep.sum()))

results.sort(key=lambda x: x[0])
print(f"满足 ratio∈[{RATIO_LO},{RATIO_HI}] 的组合数: {len(results)}")
print(f"{'cost':>7s} {'AUC':>7s} {'ACC':>7s} {'ratio':>7s} {'k':>4s} {'m':>4s} {'f':>3s} {'n_keep':>7s}")
for row in results[:25]:
    cost, a, ac, r, k, m, f, nk = row
    print(f"{cost:7.4f} {a:7.4f} {ac:7.4f} {r:7.4f} {k:4d} {m:4d} {f:3d} {nk:7d}")

if results:
    np.save("/home/guozheng/claude_dir/wsi_research/sample_clean/label_l_out/_explore_best.npy",
            results[0][7])

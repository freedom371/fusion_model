# -*- coding: utf-8 -*-
"""label_l train 集筛选: 通用局部搜索。

目标: 用固定 checkpoint (best_model_seed1013.pt) 在筛选后的 train 子集上
  - AUC ≈ 0.80, ACC ≈ 0.80  (容差 TOL)
  - label=1 占比 ∈ [0.30, 0.40]
  - 在满足上述前提下, 尽量保留更多样本 (少删)

方法: 贪心删除 → 贪心回加 → 2-swap 局部搜索, 多起点重启。
预测口径: 冻结归一化统计量 (全部 520 例) 下的逐样本预测, 筛选=纯子集选择。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("/home/guozheng/claude_dir/wsi_research/sample_clean/label_l_out")
TARGET = 0.80
TOL = 0.003          # |AUC-0.8| 与 |ACC-0.8| 的容差
AUC_TARGET = TARGET  # 可分别设定 (漂移补偿用)
ACC_TARGET = TARGET
TOL_AUC = TOL
TOL_ACC = TOL
LO, HI = 0.30, 0.40  # 正样本占比区间
NOISE = 0.005        # 打分噪声, 用于跳出平台

df = pd.read_csv(OUT / "pred_train.csv")
y = df["label"].values.astype(int)
s = df["pred_fu"].values.astype(float)
n = len(y)
is_pos = y == 1
correct = ((s >= 0.5).astype(int) == y)
pos_all = np.where(is_pos)[0]
neg_all = np.where(~is_pos)[0]


def set_data(labels, scores):
    """重设当前数据集(供迭代调用)。"""
    global y, s, n, is_pos, correct, pos_all, neg_all, TOL
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores).astype(float)
    n = len(y)
    is_pos = y == 1
    correct = ((s >= 0.5).astype(int) == y)
    pos_all = np.where(is_pos)[0]
    neg_all = np.where(~is_pos)[0]


def metrics(keep):
    """返回 (auc, acc, ratio); keep 为 bool 掩码。"""
    pk = pos_all[keep[pos_all]]
    qk = neg_all[keep[neg_all]]
    P, N = len(pk), len(qk)
    if P == 0 or N == 0:
        return 0.5, 0.0, 0.0
    ps = np.sort(s[pk])
    qs = s[qk]
    right = np.searchsorted(ps, qs, side="right")
    left = np.searchsorted(ps, qs, side="left")
    a = (P - right) + 0.5 * (right - left)
    auc = a.sum() / (P * N)
    nk = int(keep.sum())
    acc = int(correct[keep].sum()) / nk
    return float(auc), float(acc), P / nk


def violation(keep):
    auc, acc, ratio = metrics(keep)
    v = max(0.0, abs(auc - AUC_TARGET) - TOL_AUC) + max(0.0, abs(acc - ACC_TARGET) - TOL_ACC)
    v += max(0.0, LO - ratio) + max(0.0, ratio - HI)
    return v, auc, acc, ratio


def key(keep):
    v, auc, acc, ratio = violation(keep)
    return (round(v, 7), -int(keep.sum())), (auc, acc, ratio)


def descend(keep, max_steps=400, rng=None):
    """贪心删除: 每步尝试删一个样本, 取 key 最小者。"""
    keep = keep.copy()
    cur_key, _ = key(keep)
    for _ in range(max_steps):
        idxs = np.where(keep)[0]
        best_k, best_i = None, None
        for i in idxs:
            trial = keep.copy()
            trial[i] = False
            k, _ = key(trial)
            if best_k is None or k < best_k:
                best_k, best_i = k, i
        if best_k is None or best_k >= cur_key:
            break
        keep[best_i] = False
        cur_key = best_k
    return keep


def ascend(keep):
    """贪心回加: 在 key 不劣化的前提下把删掉的样本加回来。"""
    keep = keep.copy()
    cur_key, _ = key(keep)
    improved = True
    while improved:
        improved = False
        for i in np.where(~keep)[0]:
            trial = keep.copy()
            trial[i] = True
            k, _ = key(trial)
            if k < cur_key:
                keep, cur_key = trial, k
                improved = True
    return keep


def swap2(keep, max_sweeps=6):
    """2-swap: 删一个已保留的 + 加一个已删除的, 保持规模不变地优化。"""
    keep = keep.copy()
    cur_key, _ = key(keep)
    for _ in range(max_sweeps):
        improved = False
        for i in np.where(keep)[0]:
            if not keep[i]:
                continue
            for j in np.where(~keep)[0]:
                trial = keep.copy()
                trial[i], trial[j] = False, True
                k, _ = key(trial)
                if k < cur_key:
                    keep, cur_key = trial, k
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return keep


def solve(seed_drop=0, seed=0):
    rng = np.random.default_rng(seed)
    keep = np.ones(n, dtype=bool)
    if seed_drop > 0:
        drop = rng.choice(n, size=seed_drop, replace=False)
        keep[drop] = False
    keep = descend(keep)
    keep = ascend(keep)
    keep = swap2(keep)
    keep = ascend(keep)
    return keep


def search(labels, scores, tol=TOL, starts=(0, 20, 60, 120, 180, 240), verbose=True,
           auc_target=None, acc_target=None, tol_auc=None, tol_acc=None):
    """在给定逐样本预测上搜索最优保留掩码, 返回 (keep_mask, key)。"""
    global TOL, TOL_AUC, TOL_ACC, AUC_TARGET, ACC_TARGET
    TOL = tol
    TOL_AUC = tol if tol_auc is None else tol_auc
    TOL_ACC = tol if tol_acc is None else tol_acc
    AUC_TARGET = TARGET if auc_target is None else auc_target
    ACC_TARGET = TARGET if acc_target is None else acc_target
    set_data(labels, scores)
    best = None
    for sd in starts:
        keep = solve(seed_drop=sd, seed=sd + 7)
        k, (auc, acc, ratio) = key(keep)
        if verbose:
            print(f"    [起点删{sd:3d}] key={k}  AUC={auc:.4f} ACC={acc:.4f} "
                  f"ratio={ratio:.4f} kept={int(keep.sum())}")
        if best is None or k < best[0]:
            best = (k, keep.copy())
    return best[1], best[0]


if __name__ == "__main__":
    best = None
    for sd in [0, 20, 60, 120, 180, 240]:
        keep = solve(seed_drop=sd, seed=sd + 7)
        k, (auc, acc, ratio) = key(keep)
        print(f"  [起点删{sd:3d}] key={k}  AUC={auc:.4f} ACC={acc:.4f} "
              f"ratio={ratio:.4f} kept={int(keep.sum())}")
        if best is None or k < best[0]:
            best = (k, keep.copy())

    k, keep = best
    auc, acc, ratio = metrics(keep)
    print("\n" + "=" * 70)
    print(f"最优: AUC={auc:.4f} ACC={acc:.4f} ratio={ratio:.4f} "
          f"kept={int(keep.sum())} (删除 {n - int(keep.sum())})")
    print("=" * 70)

    kept = df[keep].copy()
    dropped = df[~keep].copy()
    kept.to_csv(OUT / "train_kept.csv", index=False)
    dropped.to_csv(OUT / "train_dropped.csv", index=False)
    with open(OUT / "search_result.json", "w") as fh:
        json.dump({"auc": auc, "acc": acc, "ratio": ratio,
                   "kept": int(keep.sum()), "dropped": int((~keep).sum()),
                   "dropped_ids": dropped["base_id"].tolist()}, fh, indent=2)
    print(f"保留 {len(kept)} / 删除 {len(dropped)}")
    print("正样本: 保留 %d 删除 %d" % (kept.label.sum(), dropped.label.sum()))
    print(f"→ {OUT}/train_kept.csv, train_dropped.csv")

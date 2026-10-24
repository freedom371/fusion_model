# -*- coding: utf-8 -*-
"""label_l train 集筛选: 漂移补偿精修。

迭代收敛后仍存在系统漂移 (冻结口径 ACC 0.7991 → 真实口径 0.7968, Δ≈-0.0023)。
本脚本以当前最优子集为锚, 对若干"补偿后的 ACC 目标"各搜一次, 用真实口径择优。

用法: python label_l_refine.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/guozheng/claude_dir/wsi_research")
sys.path.insert(0, str(ROOT / "sample_clean"))

from label_l_iterate import (ALL_IDS, LABELS, OUT, Y_ALL, auc_acc,  # noqa: E402
                             predict_all, true_metrics)
from label_l_search import search  # noqa: E402

# 443 例时的 ACC 取值网格: k/443
N_GRID = 443
acc_grid = {round(k / N_GRID, 5): k for k in range(340, 365)}

ANCHOR_CSV = OUT / "train_kept_final.csv"
anchor_ids = pd.read_csv(ANCHOR_CSV, dtype={"base_id": str}).base_id.tolist()
keep_anchor = np.array([i in set(anchor_ids) for i in ALL_IDS])

print(f"[锚] train={int(keep_anchor.sum())} (用其统计量重算 520 例预测)")
preds_anchor = predict_all([ALL_IDS[i] for i in np.where(keep_anchor)[0]])

# 目标: 真实口径 ACC 尽量贴近 0.80; 冻结口径相对真实的漂移约 -0.0023
# tol_auc=0.0025 为实测最优权衡: 保持最大规模 443 例, AUC/ACC 均紧贴 0.80
candidates = [0.7991]
TOL_AUC = 0.0025
results = []
for acc_t in candidates:
    if acc_t not in acc_grid.values() and round(acc_t, 5) not in acc_grid:
        pass
    print(f"\n--- 冻结 ACC 目标 {acc_t:.4f} (tol_auc={TOL_AUC}, tol_acc=0.0011) ---")
    keep, key = search(Y_ALL, preds_anchor, starts=(0, 20, 60, 120, 180, 240),
                       auc_target=0.80, acc_target=acc_t, tol_auc=TOL_AUC, tol_acc=0.0011,
                       verbose=False)
    auc_t, acc_t_true, ratio_t, _ = true_metrics(keep)
    kept_auc, kept_acc, kept_ratio = auc_acc(Y_ALL[keep], preds_anchor[keep])
    print(f"  冻结: AUC={kept_auc:.4f} ACC={kept_acc:.4f} ratio={kept_ratio:.4f} "
          f"kept={int(keep.sum())}")
    print(f"  真实: AUC={auc_t:.4f} ACC={acc_t_true:.4f} ratio={ratio_t:.4f} "
          f"kept={int(keep.sum())}")
    results.append((abs(auc_t - 0.80) + abs(acc_t_true - 0.80), acc_t, keep.copy(),
                    auc_t, acc_t_true, ratio_t))

results.sort(key=lambda x: x[0])
cost, acc_t, keep, auc_t, acc_t_true, ratio_t = results[0]
print("\n" + "=" * 70)
print(f"最优: 真实 AUC={auc_t:.4f} ACC={acc_t_true:.4f} ratio={ratio_t:.4f} "
      f"kept={int(keep.sum())}  (冻结目标 ACC={acc_t})")
print("=" * 70)

ids = [ALL_IDS[i] for i in np.where(keep)[0]]
dropped = [ALL_IDS[i] for i in np.where(~keep)[0]]
pd.DataFrame({"base_id": ids, "label": [LABELS[i] for i in ids]}).to_csv(
    OUT / "train_kept_final.csv", index=False)
pd.DataFrame({"base_id": dropped, "label": [LABELS[i] for i in dropped]}).to_csv(
    OUT / "train_dropped_final.csv", index=False)
with open(OUT / "final_result.json", "w") as fh:
    json.dump({"true_auc": auc_t, "true_acc": acc_t_true, "true_ratio": ratio_t,
               "kept": int(keep.sum()), "dropped": len(dropped),
               "dropped_ids": dropped}, fh, indent=2)
print(f"→ {OUT}/train_kept_final.csv ({len(ids)} 例)")

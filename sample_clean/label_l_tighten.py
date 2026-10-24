# -*- coding: utf-8 -*-
"""收紧 AUC 容差测试: 能否在保持 443 例的前提下让 AUC 更贴近 0.80。
本脚本只做对比, 不覆盖 train_kept_final.csv。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/guozheng/claude_dir/wsi_research")
sys.path.insert(0, str(ROOT / "sample_clean"))

from label_l_iterate import ALL_IDS, OUT, Y_ALL, auc_acc, predict_all, true_metrics  # noqa: E402
from label_l_search import search  # noqa: E402

anchor_ids = pd.read_csv(OUT / "train_kept_final.csv", dtype={"base_id": str}).base_id.tolist()
keep_anchor = np.array([i in set(anchor_ids) for i in ALL_IDS])
print(f"[锚] train={int(keep_anchor.sum())}")
preds = predict_all([ALL_IDS[i] for i in np.where(keep_anchor)[0]])

for tol_auc in [0.0015, 0.0025, 0.0035, 0.005]:
    keep, _ = search(Y_ALL, preds, starts=(0, 20, 60, 120, 180, 240),
                     auc_target=0.80, acc_target=0.7991,
                     tol_auc=tol_auc, tol_acc=0.0011, verbose=False)
    fa, fac, fr = auc_acc(Y_ALL[keep], preds[keep])
    ta, tac, tr, _ = true_metrics(keep)
    print(f"  tol_auc={tol_auc:.4f} | 冻结 AUC={fa:.4f} ACC={fac:.4f} | "
          f"真实 AUC={ta:.4f} ACC={tac:.4f} ratio={tr:.4f} kept={int(keep.sum())}")

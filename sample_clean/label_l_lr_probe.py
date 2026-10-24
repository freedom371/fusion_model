# -*- coding: utf-8 -*-
"""诊断实验: 学习率 (收敛程度) 对重训模型校准的影响。

现象: 所有变体的"最优阈值"都恒等于训练集占比 (0.32~0.45), 远低于 0.5,
导致 ACC@0.5 比"最优阈值 ACC"低 0.07~0.10。怀疑是 lr=2e-5 步数太少,
模型只学到基率、正样本概率被压到 0.5 以下。

本脚本在同一个子集上只改 lr 重训, 看最优阈值是否向 0.5 靠拢。
**纯诊断, 不改 train.py 的默认配置。**

用法: python label_l_lr_probe.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/guozheng/claude_dir/wsi_research")
sys.path.insert(0, str(ROOT / "fusion_model"))
sys.path.insert(0, str(ROOT / "sample_clean"))

from label_l_probe import best_acc, predict_split  # noqa: E402
from label_l_retrain_loop import (LABELS, LOOP_DIR, OUT, auc_acc,  # noqa: E402
                                  write_label)
import dataset as fu_ds  # noqa: E402

OUT_CSV = OUT / "lr_probe_result.csv"


def train_with_lr(keep_ids, lr, tag):
    """与 train_on 相同, 但覆盖 conf 的 lr。"""
    label_file = write_label(keep_ids, LOOP_DIR / f"label_{tag}.csv")
    fu_ds.LABEL_CSV = label_file
    fu_ds._TRAIN_STATS = None
    import train as train_mod
    train_mod.conf["exp_name"] = "cmt_cpg_loop"
    train_mod.conf["lr"] = lr
    train_mod._train_impl(train_mod.conf)
    return ROOT / "fusion_model/output/cmt_cpg_loop/checkpoints/best_model_seed1013.pt"


if __name__ == "__main__":
    ids = [l.strip() for l in (LOOP_DIR / "p2_rand_ids.txt").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    y = np.array([LABELS[i] for i in ids]).astype(int)
    print(f"[子集] rand-380: n={len(ids)} pos={y.sum()} ratio={y.mean():.4f}")

    rows = []
    for lr in [2e-5, 5e-5, 1e-4, 2e-4]:
        tag = f"lr{lr}"
        print(f"\n{'='*72}\n=== lr={lr} ===\n{'='*72}")
        ckpt = train_with_lr(ids, lr, tag)
        p_ids, p_pred = predict_split(ckpt, ids, "train")
        p = dict(zip(p_ids, p_pred))
        ss = np.array([p[i] for i in ids])
        a, c = auc_acc(y, ss)
        ba, bt = best_acc(ss, y)
        print(f">>> [lr={lr}] AUC={a:.4f}  ACC@0.5={c:.4f}  "
              f"最优ACC={ba:.4f} (阈值{bt:.3f})")
        rows.append({"lr": lr, "auc": a, "acc": c, "best_acc": ba, "best_thr": bt})
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    print("\n汇总:")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n→ {OUT_CSV}")

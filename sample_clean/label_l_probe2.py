# -*- coding: utf-8 -*-
"""判定实验 v2: 训练集成分 → 重训模型 (AUC, 最优阈值 ACC) 的映射。

修正 v1 的缺陷: v1 里"难/易"排序用的是上一个变体的模型预测, 参考系串了。
本版统一用 443 集的模型 (retrain_loop/model_round1.pt) 当唯一参考预测。

变体 (全部从 443 集出发, 占比恒在 [0.30, 0.40]):
  1. rand  : 随机删 63 个正样本            → n=380, ratio=0.300
  2. easy  : 删分数最高的 63 个正样本      → n=380, ratio=0.300
  3. hard  : 删分数最低的 63 个正样本      → n=380, ratio=0.300
  4. easy+neg: 在 easy 基础上再删 95 个最难负样本 → n=285, ratio=0.400
     (删负样本会抬高占比, 0.40 是上限, 故最多删 95 个)

用法: python label_l_probe2.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/home/guozheng/claude_dir/wsi_research")
sys.path.insert(0, str(ROOT / "fusion_model"))
sys.path.insert(0, str(ROOT / "sample_clean"))

from label_l_probe import predict_split, best_acc  # noqa: E402
from label_l_retrain_loop import LABELS, LOOP_DIR, auc_acc, train_on  # noqa: E402

REF_CKPT = LOOP_DIR / "model_round1.pt"   # 443 集 (ratio 0.3995) 的模型
F443 = ROOT / "data/cvs/label_l_filtered.csv"
OUT_CSV = ROOT / "sample_clean/label_l_out/probe2_result.csv"


def load443():
    df = pd.read_csv(F443, dtype={"ID": str})
    return df[df.group == "train"]["ID"].str.replace(
        r"\.nii\.gz$", "", regex=True).astype(str).tolist()


def make_variant(kind, ids443, pmap, n_del=63, n_neg=95):
    pos = sorted([i for i in ids443 if LABELS[i] == 1], key=lambda i: pmap[i])
    neg = sorted([i for i in ids443 if LABELS[i] == 0], key=lambda i: pmap[i])
    if kind == "rand":
        rng = np.random.default_rng(1013)
        drop = set(rng.choice(pos, size=n_del, replace=False).tolist())
    elif kind == "easy":
        drop = set(pos[-n_del:])
    elif kind == "hard":
        drop = set(pos[:n_del])
    elif kind == "easy+neg":
        drop = set(pos[-n_del:]) | set(neg[:n_neg])   # 再删最"容易"的负样本
    return [i for i in ids443 if i not in drop]


if __name__ == "__main__":
    ids443 = load443()
    ids_ref, preds_ref = predict_split(REF_CKPT, ids443, "train")
    pmap = dict(zip(ids_ref, preds_ref))
    print(f"[参考] {REF_CKPT.name} 在 443 集上; train n={len(pmap)}")

    rows = []
    for kind in ["rand", "easy", "hard", "easy+neg"]:
        keep = make_variant(kind, ids443, pmap)
        y = np.array([LABELS[i] for i in keep]).astype(int)
        print(f"\n{'='*72}\n=== 变体 {kind}: n={len(keep)} pos={y.sum()} "
              f"ratio={y.mean():.4f} ===\n{'='*72}")
        ckpt = train_on(keep, f"p2_{kind}")
        ids, preds = predict_split(ckpt, keep, "train")
        p = dict(zip(ids, preds))
        ss = np.array([p[i] for i in keep])
        a, c = auc_acc(y, ss)
        ba, bt = best_acc(ss, y)
        print(f">>> [{kind}] n={len(keep)} ratio={y.mean():.4f} | AUC={a:.4f}  "
              f"ACC@0.5={c:.4f}  最优ACC={ba:.4f} (阈值{bt:.3f})")
        rows.append({"variant": kind, "n": len(keep), "ratio": float(y.mean()),
                     "auc": a, "acc": c, "best_acc": ba, "best_thr": bt})
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        (LOOP_DIR / f"p2_{kind}_ids.txt").write_text("\n".join(keep), encoding="utf-8")

    print("\n汇总:")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n→ {OUT_CSV}")

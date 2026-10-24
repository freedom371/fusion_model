# -*- coding: utf-8 -*-
"""挑出对 ACC/AUC 伤害最大的 K 个样本 (优先正样本), 供人工复核。

伤害度定义 (参考模型 = 在完整 520 池上重训的模型, 干净口径打分):
  正样本 i:  d_i = 分数高于它的负样本比例   (1.0 = 被全部负样本压过)
  负样本 i:  d_i = 分数低于它的正样本比例
d_i 即"该样本把 AUC 往下拽的比例", 两类同尺度; 被阈值判错的样本必然 d_i 很大。
K=80 时被删的一定全是正样本 —— 混入负样本会让占比越过 0.40 上限。

用法:
  python pick_harmful.py [ckpt] [K=80]
输出:
  harmful{K}_review.csv   被删的 K 例明细 (按伤害度降序)
  harmful_all520.csv      全池伤害度明细 (便于复核是否有漏网)
  keep_harm{K}.txt        保留的 520-K 例 (喂给 filter_v3_loop.py --keep)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/guozheng/claude_dir/wsi_research")
sys.path.insert(0, str(ROOT / "fusion_model"))
sys.path.insert(0, str(ROOT / "sample_clean"))

from label_l_retrain_loop import POOL, LABELS, predict_pool, auc_acc  # noqa: E402
from label_l_probe import best_acc  # noqa: E402

V3_DIR = ROOT / "sample_clean/label_l_out/filter_v3"

# 参考 checkpoint: 完整 520 池上重训的模型 (filter_v3_loop.py --k 0 的产物)
if len(sys.argv) > 1:
    CKPT = Path(sys.argv[1])
else:
    cands = sorted(V3_DIR.glob("best_e0_h0_n0_wd*.pt"))
    if not cands:
        raise SystemExit("未找到参考 checkpoint, 先跑 filter_v3_loop.py --k 0")
    CKPT = cands[-1]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 80

y = np.array([LABELS[i] for i in POOL]).astype(int)
print(f"[参考模型] {CKPT.name} 给池内 {len(POOL)} 例打分 (干净口径) ...")
s = predict_pool(CKPT, POOL)

P, N = int(y.sum()), int((y == 0).sum())
sp, sn = np.sort(s[y == 1]), np.sort(s[y == 0])
dmg = np.empty(len(POOL))
dmg[y == 1] = (N - np.searchsorted(sn, s[y == 1], "right")) / N   # 高于它的负样本比例
dmg[y == 0] = np.searchsorted(sp, s[y == 0], "left") / P          # 低于它的正样本比例

acc, thr = best_acc(s, y)
df = pd.DataFrame({"base_id": POOL, "label": y, "score": s, "damage": dmg})
df["viol_pairs"] = np.where(y == 1, dmg * N, dmg * P).round().astype(int)  # 逆序对数
df["wrong_at_thr"] = ((df.label == 1) & (df.score < thr)) | ((df.label == 0) & (df.score >= thr))
df.to_csv(V3_DIR / "harmful_all520.csv", index=False)

pos = df[df.label == 1].sort_values("damage", ascending=False).reset_index(drop=True)
pick = pos.head(K).base_id.tolist()
sel = df[df.base_id.isin(pick)].sort_values("damage", ascending=False)
sel.to_csv(V3_DIR / f"harmful{K}_review.csv", index=False)

keep = [i for i in POOL if i not in set(pick)]
(V3_DIR / f"keep_harm{K}.txt").write_text("\n".join(keep), encoding="utf-8")

# ── 汇总 ──
yr = np.array([LABELS[i] for i in keep]).astype(int)
ref_auc, _ = auc_acc(y, s)
print(f"\n参考模型在全 {len(POOL)} 池上: AUC={ref_auc:.4f}  最优阈值 ACC={acc:.4f} (阈值 {thr:.4f})")
print(f"\n被删 {K} 例: 全为正样本, 占全部正样本 {K}/{P} = {K/P:.1%}")
print(f"  分数: 均值 {sel.score.mean():.4f} 范围 [{sel.score.min():.4f}, {sel.score.max():.4f}]"
      f"   (全体正样本均值 {df[df.label==1].score.mean():.4f})")
print(f"  伤害度: 均值 {sel.damage.mean():.4f} 范围 [{sel.damage.min():.4f}, {sel.damage.max():.4f}]")
print(f"  其中被参考模型判错的: {int(sel.wrong_at_thr.sum())}/{K}")
print(f"  被删样本共涉及逆序对 {int(sel.viol_pairs.sum())} 对"
      f" (池内总逆序对 {int(df.viol_pairs.sum()/2)} 对)")
print(f"\n删除后 train: n={len(keep)} pos={int(yr.sum())} 占比={yr.mean():.4f}")
print(f"→ {V3_DIR/f'harmful{K}_review.csv'}")
print(f"→ {V3_DIR/f'keep_harm{K}.txt'}")
print("\n被删样本 (前 15, 按伤害度降序):")
print(sel.head(15).to_string(index=False))

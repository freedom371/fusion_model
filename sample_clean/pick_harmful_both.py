# -*- coding: utf-8 -*-
"""挑出对 FU 和 CT 两个模型都最有害的 K 个正样本 (供人工复核)。

伤害度 d_i (与 pick_harmful.py 同定义, 两个模型各算一份, 同为 [0,1] 尺度):
  正样本 i: d_i = 分数高于它的负样本比例
  负样本 i: d_i = 分数低于它的正样本比例
d_i 即"该样本把该模型 AUC 往下拽的比例"。

综合伤害 (--mode):
  sum (默认) = dmg_fu + dmg_ct, 稳健, 强调总副作用
  min        = min(dmg_fu, dmg_ct), 强调"两边都有害"
只从正样本里挑 —— 混入负样本会把 train 占比推过 0.40 上限。

用法: python pick_harmful_both.py [K=80] [--mode sum|min]
输出:
  harmful{K}_both_review.csv          被删明细 (按综合伤害降序)
  harmful_all520_both.csv             全池两模型伤害度
  keep_harm{K}_both.txt               保留的 520-K 例
  data/cvs/label_l_harm{K}_both.csv   exclude 约定 label 文件 (train=520-K)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("/home/guozheng/claude_dir/wsi_research")
sys.path.insert(0, str(ROOT / "fusion_model"))
sys.path.insert(0, str(ROOT / "sample_clean"))

from label_l_retrain_loop import POOL, LABELS, auc_acc  # noqa: E402
from label_l_probe import best_acc  # noqa: E402

V3_DIR = ROOT / "sample_clean/label_l_out/filter_v3"
K = int(sys.argv[1]) if len(sys.argv) > 1 else 80
MODE = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "sum"


def damage(s, y):
    """与 pick_harmful.py 完全相同的伤害度定义。"""
    P, N = int(y.sum()), int((y == 0).sum())
    sp, sn = np.sort(s[y == 1]), np.sort(s[y == 0])
    d = np.empty(len(s))
    d[y == 1] = (N - np.searchsorted(sn, s[y == 1], "right")) / N
    d[y == 0] = np.searchsorted(sp, s[y == 0], "left") / P
    return d, P, N


# ── 读两个模型的打分 ──
fu = pd.read_csv(V3_DIR / "harmful_all520.csv", dtype={"base_id": str})
ct = pd.read_csv(V3_DIR / "ct_preds_pool520.csv", dtype={"base_id": str})
ct.columns = ["base_id", "label_ct", "pred_ct"]
df = fu.merge(ct, on="base_id", how="inner", validate="one_to_one")
assert len(df) == len(POOL), f"对齐失败: {len(df)} vs {len(POOL)}"
assert (df.label == df.label_ct).all(), "两表 label 不一致"
assert (df.label == [LABELS[i] for i in df.base_id]).all(), "与前缀池 label 不一致"
y = df.label.values.astype(int)

dmg_fu = df.damage.values
dmg_ct, P, N = damage(df.pred_ct.values, y)
df["dmg_fu"], df["dmg_ct"] = dmg_fu, dmg_ct
df["dmg_comb"] = dmg_fu + dmg_ct if MODE == "sum" else np.minimum(dmg_fu, dmg_ct)
df["viol_fu"] = np.where(y == 1, dmg_fu * N, dmg_fu * P).round().astype(int)
df["viol_ct"] = np.where(y == 1, dmg_ct * N, dmg_ct * P).round().astype(int)
df.to_csv(V3_DIR / "harmful_all520_both.csv", index=False)

# ── 诊断: 两模型伤害度的关系 ──
acc_fu, thr_fu = best_acc(df.score.values, y)
acc_ct, thr_ct = best_acc(df.pred_ct.values, y)
print(f"全池 {len(df)} 例 (pos {P} / neg {N})")
a_fu, _ = auc_acc(y, df.score.values)
a_ct, _ = auc_acc(y, df.pred_ct.values)
print(f"  FU: AUC={a_fu:.4f} 最优阈值ACC={acc_fu:.4f} (thr {thr_fu:.4f}, 判错 {int(((df.score>=thr_fu).astype(int)!=y).sum())})")
print(f"  CT: AUC={a_ct:.4f} 最优阈值ACC={acc_ct:.4f} (thr {thr_ct:.4f}, 判错 {int(((df.pred_ct>=thr_ct).astype(int)!=y).sum())})")
rho_all = spearmanr(dmg_fu, dmg_ct).statistic
pm = df[df.label == 1]
rho_pos = spearmanr(pm.dmg_fu, pm.dmg_ct).statistic
print(f"  伤害度 Spearman: 全池 {rho_all:.3f} / 正样本 {rho_pos:.3f}")

# ── 挑选: 正样本里综合伤害最高的 K 个 ──
pos = df[df.label == 1].sort_values("dmg_comb", ascending=False).reset_index(drop=True)
pick = pos.head(K).base_id.tolist()
sel = df[df.base_id.isin(pick)].sort_values("dmg_comb", ascending=False)
sel.to_csv(V3_DIR / f"harmful{K}_both_review.csv", index=False)

keep = [i for i in POOL if i not in set(pick)]
(V3_DIR / f"keep_harm{K}_both.txt").write_text("\n".join(keep), encoding="utf-8")

# ── exclude 约定 label 文件: train=keep, 被删样本进 exclude, val/test 原样 ──
src = pd.read_csv(ROOT / "data/cvs/label_l.csv", dtype={"ID": str})
src["base_id"] = src.ID.str.replace(r"\.nii\.gz$", "", regex=True)
drop = set(pick)
lab_rows = [{"ID": r.ID, "label": r.label,
             "group": ("exclude" if r.base_id in drop else "train") if r.group == "train" else r.group}
            for _, r in src.iterrows()]
lab_path = ROOT / f"data/cvs/label_l_harm{K}_both.csv"
pd.DataFrame(lab_rows).to_csv(lab_path, index=False)

# ── 汇总 ──
top_fu = set(df[df.label == 1].nlargest(K, "dmg_fu").base_id)
top_ct = set(df[df.label == 1].nlargest(K, "dmg_ct").base_id)
yr = np.array([LABELS[i] for i in keep]).astype(int)
print(f"\n被删 {K} 例 (mode={MODE}): 全为正样本, 占全部正样本 {K}/{P} = {K/P:.1%}")
print(f"  dmg_fu: 均值 {sel.dmg_fu.mean():.4f}  最小 {sel.dmg_fu.min():.4f}   (全体正样本均值 {pm.dmg_fu.mean():.4f})")
print(f"  dmg_ct: 均值 {sel.dmg_ct.mean():.4f}  最小 {sel.dmg_ct.min():.4f}   (全体正样本均值 {pm.dmg_ct.mean():.4f})")
print(f"  被 FU 模型判错 {int((sel.score < thr_fu).sum())}/{K}"
      f", 被 CT 模型判错 {int((sel.pred_ct < thr_ct).sum())}/{K}")
print(f"  与 FU 单独 top{K} 重合 {len(set(pick) & top_fu)}/{K}"
      f", 与 CT 单独 top{K} 重合 {len(set(pick) & top_ct)}/{K}")
print(f"  逆序对: FU {int(sel.viol_fu.sum())}/{int(df.viol_fu.sum()/2)}"
      f", CT {int(sel.viol_ct.sum())}/{int(df.viol_ct.sum()/2)}")
print(f"\n删除后 train: n={len(keep)} pos={int(yr.sum())} 占比={yr.mean():.4f}")
print(f"→ {V3_DIR/f'harmful{K}_both_review.csv'}")
print(f"→ {V3_DIR/f'keep_harm{K}_both.txt'}")
print(f"→ {lab_path}")
print(f"\n被删样本 (前 15, 按综合伤害降序):")
print(sel[["base_id", "label", "score", "pred_ct", "dmg_fu", "dmg_ct", "dmg_comb"]].head(15).to_string(index=False))

# -*- coding: utf-8 -*-
"""用两个模型 train 集逐样本分析, 挑出对 CT 和 FU 都最有害的 K 个正样本 (供人工复核)。

数据来源 = train_impact.py 的产出 (即两个 train.py 新加的 train 集逐样本分析):
  fusion_model/output/cmt_cpg/sample_analysis/train_auc_impact_seed1013.csv
  ct_model_Swin_tiny/output/ct_swin_tiny_slice_agg/sample_analysis/train_auc_impact_seed1377.csv
影响度 = combined_impact = auc_impact(与标签反的 margin, 0~1) + is_wrong(0/1), 范围 0~2。

综合影响 (--mode):
  min (默认) = min(fu, ct), 强调"两边都被它拖累"
  sum        = fu + ct, 强调总拖累
只从正样本里挑 —— 删负样本会把 train 占比推过 0.40 上限。

用法: python pick_from_train_impact.py [K=80] [--mode min|sum]
输出:
  impact{K}_review.csv          被删明细 (按综合影响降序)
  impact_all520.csv             全池两模型影响度
  keep_impact{K}.txt            保留的 520-K 例
  data/cvs/label_l_impact{K}.csv   exclude 约定 label 文件 (train=520-K)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/guozheng/claude_dir/wsi_research")
V3_DIR = ROOT / "sample_clean/label_l_out/filter_v3"
FU_CSV = ROOT / "fusion_model/output/cmt_cpg/sample_analysis/train_auc_impact_seed1013.csv"
CT_CSV = (ROOT / "ct_model_Swin_tiny/output/ct_swin_tiny_slice_agg/sample_analysis"
          / "train_auc_impact_seed1377.csv")

K = int(sys.argv[1]) if len(sys.argv) > 1 else 80
MODE = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "min"

fu = pd.read_csv(FU_CSV, dtype={"base_id": str})
ct = pd.read_csv(CT_CSV, dtype={"base_id": str})
fu = fu[["base_id", "label", "pred", "combined_impact"]].rename(
    columns={"pred": "pred_fu", "combined_impact": "imp_fu"})
ct = ct[["base_id", "label", "pred", "combined_impact"]].rename(
    columns={"pred": "pred_ct", "combined_impact": "imp_ct"})
d = fu.merge(ct, on="base_id", validate="one_to_one")
assert (d.label_x == d.label_y).all()
d = d.rename(columns={"label_x": "label"}).drop(columns="label_y")
d["imp_comb"] = d[["imp_fu", "imp_ct"]].min(axis=1) if MODE == "min" else d.imp_fu + d.imp_ct

y = d.label.values.astype(int)
P, N = int(y.sum()), int((y == 0).sum())
print(f"全池 {len(d)} 例 (pos {P} / neg {N}) | mode={MODE}")
print(f"  FU: ACC={ (d.label == (d.pred_fu >= 0.5).astype(int)).mean():.4f}"
      f"  错分 {int((d.label != (d.pred_fu >= 0.5).astype(int)).sum())} 例")
print(f"  CT: ACC={ (d.label == (d.pred_ct >= 0.5).astype(int)).mean():.4f}"
      f"  错分 {int((d.label != (d.pred_ct >= 0.5).astype(int)).sum())} 例")
pm = d[d.label == 1]
print(f"  正样本 imp_fu 中位 {pm.imp_fu.median():.3f} / imp_ct 中位 {pm.imp_ct.median():.3f}"
      f" | 相关系数 {pm.imp_fu.corr(pm.imp_ct):.3f}")

pos = pm.sort_values("imp_comb", ascending=False).reset_index(drop=True)
pick = pos.head(K).base_id.tolist()
sel = d[d.base_id.isin(pick)].sort_values("imp_comb", ascending=False)
sel.to_csv(V3_DIR / f"impact{K}_review.csv", index=False)
d.to_csv(V3_DIR / "impact_all520.csv", index=False)

keep = [i for i in pd.read_csv(ROOT / "data/cvs/label_l.csv", dtype={"ID": str})
        .assign(base_id=lambda t: t.ID.str.replace(r"\.nii\.gz$", "", regex=True))
        .query("group == 'train'").base_id if i not in set(pick)]
(V3_DIR / f"keep_impact{K}.txt").write_text("\n".join(keep), encoding="utf-8")

src = pd.read_csv(ROOT / "data/cvs/label_l.csv", dtype={"ID": str})
src["base_id"] = src.ID.str.replace(r"\.nii\.gz$", "", regex=True)
drop = set(pick)
lab_rows = [{"ID": r.ID, "label": r.label,
             "group": ("exclude" if r.base_id in drop else "train") if r.group == "train" else r.group}
            for _, r in src.iterrows()]
lab_path = ROOT / f"data/cvs/label_l_impact{K}.csv"
pd.DataFrame(lab_rows).to_csv(lab_path, index=False)

yr = np.array([d.set_index("base_id").label[i] for i in keep]).astype(int)
top_fu = set(pm.nlargest(K, "imp_fu").base_id)
top_ct = set(pm.nlargest(K, "imp_ct").base_id)
print(f"\n被删 {K} 例: 全为正样本, 占全部正样本 {K}/{P} = {K/P:.1%}")
print(f"  imp_fu: 均值 {sel.imp_fu.mean():.3f} 最小 {sel.imp_fu.min():.3f}   (正样本均值 {pm.imp_fu.mean():.3f})")
print(f"  imp_ct: 均值 {sel.imp_ct.mean():.3f} 最小 {sel.imp_ct.min():.3f}   (正样本均值 {pm.imp_ct.mean():.3f})")
print(f"  两侧都 >1 (都判错或有明显反向 margin): {int(((sel.imp_fu>1)&(sel.imp_ct>1)).sum())}/{K}")
print(f"  与 FU 单独 top{K} 重合 {len(set(pick)&top_fu)}/{K}, 与 CT 单独 top{K} 重合 {len(set(pick)&top_ct)}/{K}")
print(f"\n删除后 train: n={len(keep)} pos={int(yr.sum())} 占比={yr.mean():.4f}")
print(f"→ {V3_DIR/f'impact{K}_review.csv'}")
print(f"→ {V3_DIR/f'keep_impact{K}.txt'}")
print(f"→ {lab_path}")
print(f"\n被删样本 (前 15):")
print(sel[["base_id", "pred_fu", "pred_ct", "imp_fu", "imp_ct", "imp_comb"]].head(15).to_string(index=False))

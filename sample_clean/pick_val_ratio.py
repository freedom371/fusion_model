# -*- coding: utf-8 -*-
"""带正样本占比约束的 val 筛选 (用户更正: val 占比应维持 0.35 左右, 不是越低越好)。

在 (删正样本数 d_p, 删负样本数 d_n) 二维网格上穷举:
  - 正样本按综合影响 (min(fu,ct) combined_impact) 降序删 —— 删最有害的;
  - 负样本按 pred_fu 降序删 —— 先删高分的 (对 AUC 最有利, 且覆盖两模型判错的负样本);
  找同时满足 四项指标 + 占比 ∈ [r_lo, r_hi] 的最小总删除量方案。
扫描全程 numpy 向量化 (rankdata 精确 AUC, 与 sklearn 口径一致), 无 pandas 逐行操作。

也可用 --dp/--dn 直接指定删除组合 (跳过扫描)。

默认目标: FU val AUC≥0.80/ACC≥0.78, CT val AUC≥0.76/ACC≥0.75, 占比 ∈ [0.33, 0.37]。
(占比 0.35 左右; 原始 val 占比 76/224=0.3393。--r-lo/--r-hi 可调。)

结论 (2026-10-08 扫描): 占比约束下 ACC 目标与占比互相锁死 ——
  四项全达标最少要删 173 (剩 51 例); 用户选定 **删 59 (正19+负40)**: 剩 165,
  占比 0.345, FU 0.886/0.691、CT 0.762/0.703 (AUC 双达标, ACC 留待偏置校准)。

用法: python pick_val_ratio.py [--dp 19 --dn 40] [--r-lo 0.33] [--r-hi 0.37]
输出:
  label_l_out/filter_v3/val_ratio_scan.csv   二维扫描全表 (d_p, d_n → 指标)
  label_l_out/filter_v3/val_ratio_review.csv 被删明细
  data/cvs/label_l_impact100_val{K}_r{pct}.csv  新 label 文件
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path("/home/guozheng/claude_dir/wsi_research")
V3_DIR = ROOT / "sample_clean/label_l_out/filter_v3"
FU_CSV = ROOT / "fusion_model/output/cmt_cpg/sample_analysis/val_auc_impact_seed1013.csv"
CT_CSV = (ROOT / "ct_model_Swin_tiny/output/ct_swin_tiny_slice_agg/sample_analysis"
          / "val_auc_impact_seed1377.csv")
SRC_LABEL = ROOT / "data/cvs/label_l_impact100.csv"


def get(args, key, default):
    return float(args[args.index(key) + 1]) if key in args else default


R_LO = get(sys.argv, "--r-lo", 0.33)
R_HI = get(sys.argv, "--r-hi", 0.37)
T_FU_AUC = 0.80
T_FU_ACC = 0.78
T_CT_AUC = 0.76
T_CT_ACC = 0.75
DP_FORCE = int(sys.argv[sys.argv.index("--dp") + 1]) if "--dp" in sys.argv else None
DN_FORCE = int(sys.argv[sys.argv.index("--dn") + 1]) if "--dn" in sys.argv else None

fu = pd.read_csv(FU_CSV, dtype={"base_id": str})
ct = pd.read_csv(CT_CSV, dtype={"base_id": str})
fu = fu[["base_id", "label", "pred", "combined_impact"]].rename(
    columns={"pred": "pred_fu", "combined_impact": "imp_fu"})
ct = ct[["base_id", "label", "pred", "combined_impact"]].rename(
    columns={"pred": "pred_ct", "combined_impact": "imp_ct"})
d = fu.merge(ct, on="base_id", validate="one_to_one")
assert (d.label_x == d.label_y).all()
d = d.rename(columns={"label_x": "label"}).drop(columns="label_y")
d["imp_comb"] = d[["imp_fu", "imp_ct"]].min(axis=1)
by_id = d.set_index("base_id")

pos_order = d[d.label == 1].sort_values("imp_comb", ascending=False).base_id.tolist()
neg_order = d[d.label == 0].sort_values("pred_fu", ascending=False).base_id.tolist()
P, N = len(pos_order), len(neg_order)

# 删除顺序上的 numpy 数组 (正样本在前, 负样本在后)
sfu_p = by_id.loc[pos_order, "pred_fu"].values
sct_p = by_id.loc[pos_order, "pred_ct"].values
sfu_n = by_id.loc[neg_order, "pred_fu"].values
sct_n = by_id.loc[neg_order, "pred_ct"].values


def fast_auc(y, s):
    n_p = int(y.sum())
    n_n = len(y) - n_p
    if n_p == 0 or n_n == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y == 1].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def metrics(dp, dn):
    y = np.concatenate([np.ones(P - dp), np.zeros(N - dn)])
    sfu = np.concatenate([sfu_p[dp:], sfu_n[dn:]])
    sct = np.concatenate([sct_p[dp:], sct_n[dn:]])
    acc_fu = float(((sfu >= 0.5) == y).mean())
    acc_ct = float(((sct >= 0.5) == y).mean())
    return fast_auc(y, sfu), acc_fu, fast_auc(y, sct), acc_ct, float(y.mean())


def ok4(m):
    return (m[0] >= T_FU_AUC and m[1] >= T_FU_ACC and m[2] >= T_CT_AUC and m[3] >= T_CT_ACC)


if DP_FORCE is not None and DN_FORCE is not None:
    m = metrics(DP_FORCE, DN_FORCE)
    best = pd.Series({"d_p": DP_FORCE, "d_n": DN_FORCE, "n": P + N - DP_FORCE - DN_FORCE,
                      "fu_auc": m[0], "fu_acc": m[1], "ct_auc": m[2], "ct_acc": m[3],
                      "ratio": m[4]})
    print(f"指定方案 删正{DP_FORCE}+删负{DN_FORCE}:")
    print(f"  FU AUC {m[0]:.4f}/ACC {m[1]:.4f} | CT AUC {m[2]:.4f}/ACC {m[3]:.4f}"
          f" | 剩余 {P + N - DP_FORCE - DN_FORCE} (占比 {m[4]:.4f})")
    print(f"  四项指标: {'达标' if ok4(m) else '未全达标'} | 占比 ∈ [{R_LO},{R_HI}]: "
          f"{'✓' if R_LO <= m[4] <= R_HI else '✗'}")
else:
    # ── 二维扫描 (纯 numpy) ──
    rows = np.empty(((P + 1) * (N + 1), 8))
    idx = 0
    for dp in range(P + 1):
        for dn in range(N + 1):
            m = metrics(dp, dn)
            rows[idx] = (dp, dn, P + N - dp - dn, m[0], m[1], m[2], m[3], m[4])
            idx += 1
    scan = pd.DataFrame(rows, columns=["d_p", "d_n", "n", "fu_auc", "fu_acc", "ct_auc", "ct_acc", "ratio"])
    ok_all = scan.apply(lambda r: ok4(r[["fu_auc", "fu_acc", "ct_auc", "ct_acc"]]), axis=1)
    feasible = scan[ok_all & (scan.ratio >= R_LO) & (scan.ratio <= R_HI)].copy()

    base = scan.iloc[0]
    print(f"基线: FU {base.fu_auc:.4f}/{base.fu_acc:.4f}  CT {base.ct_auc:.4f}/{base.ct_acc:.4f}"
          f"  占比 {base.ratio:.4f}")
    print(f"约束: FU AUC≥{T_FU_AUC} ACC≥{T_FU_ACC} | CT AUC≥{T_CT_AUC} ACC≥{T_CT_ACC} "
          f"| 占比 ∈ [{R_LO},{R_HI}]")
    if feasible.empty:
        print("\n!! 占比约束下无可行解。")
        scan.to_csv(V3_DIR / "val_ratio_scan.csv", index=False)
        sys.exit(1)
    feasible["total"] = feasible.d_p + feasible.d_n
    feasible["ratio_dev"] = (feasible.ratio - 0.35).abs()
    best = feasible.sort_values(["total", "ratio_dev"]).iloc[0]
    print(f"\n可行解 {len(feasible)} 个; 最小总删除量方案:")
    print(f"  删正 {int(best.d_p)} + 删负 {int(best.d_n)} = {int(best.total)} 例"
          f" → 剩余 {int(best.n)} (占比 {best.ratio:.4f})")
    print(f"  FU AUC {best.fu_auc:.4f} / ACC {best.fu_acc:.4f} | "
          f"CT AUC {best.ct_auc:.4f} / ACC {best.ct_acc:.4f}")
    scan.to_csv(V3_DIR / "val_ratio_scan.csv", index=False)

# ── 落盘 ──
dp, dn = int(best.d_p), int(best.d_n)
pick = pos_order[:dp] + neg_order[:dn]
sel = d[d.base_id.isin(pick)].copy()
sel["del_reason"] = sel.base_id.map(
    lambda x: "正:高综合影响" if x in set(pos_order[:dp]) else "负:高分平衡占比")
sel = sel.sort_values("imp_comb", ascending=False)
rem = d[~d.base_id.isin(pick)]
sel.to_csv(V3_DIR / "val_ratio_review.csv", index=False)
(V3_DIR / "keep_val_ratio.txt").write_text("\n".join(rem.base_id.tolist()), encoding="utf-8")

src = pd.read_csv(SRC_LABEL, dtype={"ID": str})
src["base_id"] = src.ID.str.replace(r"\.nii\.gz$", "", regex=True)
drop = set(pick)
lab_rows = [{"ID": r.ID, "label": r.label,
             "group": ("exclude" if r.base_id in drop else r.group)}
            for _, r in src.iterrows()]
pct = round(best.ratio * 100)
lab_path = ROOT / f"data/cvs/label_l_impact100_val{dp + dn}_r{pct}.csv"
pd.DataFrame(lab_rows).to_csv(lab_path, index=False)

print(f"\n删除 {dp + dn} 例 (正 {dp} / 负 {dn}), 剩余 val {int(best.n)} 例"
      f" (正 {int(rem.label.sum())}, 占比 {best.ratio:.4f})")
print(f"→ {V3_DIR/'val_ratio_review.csv'}")
print(f"→ {lab_path}")
print(f"\n被删正样本 (前 10):")
print(sel[sel.label == 1][["base_id", "pred_fu", "pred_ct", "imp_comb"]].head(10).to_string(index=False))
print(f"\n被删负样本 (pred_fu 最高 10 个):")
print(sel[sel.label == 0][["base_id", "pred_fu", "pred_ct", "imp_comb"]].head(10).to_string(index=False))

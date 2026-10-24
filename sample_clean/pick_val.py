# -*- coding: utf-8 -*-
"""用 FU/CT 两个模型的 **val 集**逐样本分析, 按"对两边都最有害"贪心删除,
直到四项 val 指标全部达标, 产出新的 exclude 约定 label 文件。

数据来源 = 两个 train.py 训练结束时自动保存的 val 逐样本分析 (均为 best 权重、无增强):
  fusion_model/output/cmt_cpg/sample_analysis/val_auc_impact_seed1013.csv   (23:47 运行, ep39)
  ct_model_Swin_tiny/output/ct_swin_tiny_slice_agg/sample_analysis/val_auc_impact_seed1377.csv (23:06 运行, ep34)
影响度 = combined_impact = auc_impact(正=1-pred, 负=pred) + is_wrong(0/1), 范围 0~2;
综合影响 = min(fu, ct), 强调"两边都被它拖累" (与 train 筛选 pick_from_train_impact.py 同口径)。

默认目标 (--target-* 可覆盖): FU val AUC≥0.80 / ACC≥0.78, CT val AUC≥0.76 / ACC≥0.75。
删除顺序: 默认按综合影响降序全样本贪心; --pos-first 时正样本先删 (负样本仅当
纯正样本删完仍不达标才补)。指标在删除后对剩余集逐步重算。

新 label 文件从 label_l_impact100.csv 派生 (train 420 / exclude 100 不动),
被删 val 样本放进 exclude 组。绝不改写原 label_l.csv 与 label_l_impact100.csv。

用法: python pick_val.py [--pos-first]
  [--target-fu-auc 0.80] [--target-fu-acc 0.78] [--target-ct-auc 0.76] [--target-ct-acc 0.75]
输出:
  label_l_out/filter_v3/val_impact_all224.csv    全 val 两模型影响度
  label_l_out/filter_v3/val{K}_review.csv        被删明细 (按综合影响降序)
  label_l_out/filter_v3/keep_val{K}.txt          保留的 val 样本
  data/cvs/label_l_impact100_val{K}.csv          exclude 约定 label 文件 (val=224-K)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

ROOT = Path("/home/guozheng/claude_dir/wsi_research")
V3_DIR = ROOT / "sample_clean/label_l_out/filter_v3"
FU_CSV = ROOT / "fusion_model/output/cmt_cpg/sample_analysis/val_auc_impact_seed1013.csv"
CT_CSV = (ROOT / "ct_model_Swin_tiny/output/ct_swin_tiny_slice_agg/sample_analysis"
          / "val_auc_impact_seed1377.csv")
SRC_LABEL = ROOT / "data/cvs/label_l_impact100.csv"


def get(args, key, default):
    return float(args[args.index(key) + 1]) if key in args else default


T_FU_AUC = get(sys.argv, "--target-fu-auc", 0.80)
T_FU_ACC = get(sys.argv, "--target-fu-acc", 0.78)
T_CT_AUC = get(sys.argv, "--target-ct-auc", 0.76)
T_CT_ACC = get(sys.argv, "--target-ct-acc", 0.75)
POS_FIRST = "--pos-first" in sys.argv

fu = pd.read_csv(FU_CSV, dtype={"base_id": str})
ct = pd.read_csv(CT_CSV, dtype={"base_id": str})
fu = fu[["base_id", "label", "pred", "combined_impact"]].rename(
    columns={"pred": "pred_fu", "combined_impact": "imp_fu"})
ct = ct[["base_id", "label", "pred", "combined_impact"]].rename(
    columns={"pred": "pred_ct", "combined_impact": "imp_ct"})
d = fu.merge(ct, on="base_id", validate="one_to_one")
assert (d.label_x == d.label_y).all(), "两模型 val CSV 的 label 不一致"
d = d.rename(columns={"label_x": "label"}).drop(columns="label_y")
d["imp_comb"] = d[["imp_fu", "imp_ct"]].min(axis=1)
assert len(d) == 224, f"val 应为 224 例, 实为 {len(d)}"


def metrics(df):
    y = df.label.values.astype(int)
    s_fu, s_ct = df.pred_fu.values, df.pred_ct.values
    try:
        auc_fu = roc_auc_score(y, s_fu)
    except ValueError:
        auc_fu = np.nan
    try:
        auc_ct = roc_auc_score(y, s_ct)
    except ValueError:
        auc_ct = np.nan
    return (auc_fu, accuracy_score(y, (s_fu >= 0.5).astype(int)),
            auc_ct, accuracy_score(y, (s_ct >= 0.5).astype(int)))


def meets(m):
    return m[0] >= T_FU_AUC and m[1] >= T_FU_ACC and m[2] >= T_CT_AUC and m[3] >= T_CT_ACC


def greedy(order_df):
    rem = d.drop(columns="order_tmp", errors="ignore")
    deleted = []
    first_k = None
    for _, row in order_df.iterrows():
        deleted.append(row.base_id)
        rem = rem[rem.base_id != row.base_id]
        m = metrics(rem)
        if first_k is None and meets(m):
            first_k = len(deleted)
            yield len(deleted), m, "达到目标"
            break
        yield len(deleted), m, ""
    # 若未达标, 输出最后 10 步与上限
    if first_k is None:
        yield -1, metrics(rem), "未达标"


# ── 删除顺序 ──
d["order_tmp"] = 0
if POS_FIRST:
    order = pd.concat([
        d[d.label == 1].sort_values("imp_comb", ascending=False),
        d[d.label == 0].sort_values("imp_comb", ascending=False)])
else:
    order = d.sort_values("imp_comb", ascending=False)

print(f"val 基线: FU AUC={metrics(d)[0]:.4f} ACC={metrics(d)[1]:.4f} | "
      f"CT AUC={metrics(d)[2]:.4f} ACC={metrics(d)[3]:.4f}")
print(f"目标: FU AUC≥{T_FU_AUC} ACC≥{T_FU_ACC} | CT AUC≥{T_CT_AUC} ACC≥{T_CT_ACC}"
      f" | 顺序: {'正样本优先' if POS_FIRST else '全样本按综合影响'}")
print(f"\n{'K':>3} {'FU_AUC':>7} {'FU_ACC':>7} {'CT_AUC':>7} {'CT_ACC':>7}  备注")

result_k, result_m = None, None
last_m = metrics(d)
for k, m, tag in greedy(order):
    if k % 5 == 0 or tag:
        note = tag if tag else ("✓" if any(
            (last_m[i] < th and m[i] >= th) for i, th in
            enumerate([T_FU_AUC, T_FU_ACC, T_CT_AUC, T_CT_ACC])) else "")
        print(f"{k:>3} {m[0]:>7.4f} {m[1]:>7.4f} {m[2]:>7.4f} {m[3]:>7.4f}  {note}")
    last_m = m
    if tag == "达到目标":
        result_k, result_m = k, m
        break

if result_k is None:
    print(f"\n!! 全部删完也不达标, 最后剩余指标: "
          f"FU AUC={last_m[0]:.4f} ACC={last_m[1]:.4f} CT AUC={last_m[2]:.4f} ACC={last_m[3]:.4f}")
    sys.exit(1)

K = result_k
pick = order.head(K).base_id.tolist()
sel = d[d.base_id.isin(pick)].sort_values("imp_comb", ascending=False)
rem = d[~d.base_id.isin(pick)]

# ── 落盘 ──
d.drop(columns="order_tmp").to_csv(V3_DIR / "val_impact_all224.csv", index=False)
sel.to_csv(V3_DIR / f"val{K}_review.csv", index=False)
keep_ids = rem.base_id.tolist()
(V3_DIR / f"keep_val{K}.txt").write_text("\n".join(keep_ids), encoding="utf-8")

src = pd.read_csv(SRC_LABEL, dtype={"ID": str})
src["base_id"] = src.ID.str.replace(r"\.nii\.gz$", "", regex=True)
drop = set(pick)
assert src[src.group == "val"].base_id.isin(d.base_id).all(), "label 文件 val 组与 CSV 不一致"
lab_rows = [{"ID": r.ID, "label": r.label,
             "group": ("exclude" if r.base_id in drop else r.group)}
            for _, r in src.iterrows()]
lab_path = ROOT / f"data/cvs/label_l_impact100_val{K}.csv"
pd.DataFrame(lab_rows).to_csv(lab_path, index=False)

yr = rem.label.values.astype(int)
n_del_pos = int(sel.label.sum())
print(f"\n删除 {K}/{len(d)} 例 (val 224 → {len(rem)}): 正 {n_del_pos} / 负 {K - n_del_pos}"
      f" | 剩余正样本占比 {yr.mean():.4f}")
print(f"删除后 val: FU AUC={result_m[0]:.4f} ACC={result_m[1]:.4f}"
      f" | CT AUC={result_m[2]:.4f} ACC={result_m[3]:.4f}")
print(f"  (另: 单模型口径下两模型各自还错的样本数: "
      f"FU {int((rem.label != (rem.pred_fu >= 0.5).astype(int)).sum())}, "
      f"CT {int((rem.label != (rem.pred_ct >= 0.5).astype(int)).sum())})")
print(f"\n→ {V3_DIR/'val_impact_all224.csv'}")
print(f"→ {V3_DIR/f'val{K}_review.csv'}")
print(f"→ {V3_DIR/f'keep_val{K}.txt'}")
print(f"→ {lab_path}")
print(f"\n被删样本 (前 15):")
print(sel[["base_id", "label", "pred_fu", "pred_ct", "imp_fu", "imp_ct", "imp_comb"]]
      .head(15).to_string(index=False))

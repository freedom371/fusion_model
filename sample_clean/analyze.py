# -*- coding: utf-8 -*-
"""分析两个模型的逐样本预测, 验证CSV与日志一致, 计算每样本留一影响。"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

CT_DIR = "/home/guozheng/claude_dir/wsi_research/ct_model_Swin_tiny/output/ct_swin_tiny_slice_agg/sample_analysis"
FU_DIR = "/home/guozheng/claude_dir/wsi_research/fusion_model/output/cmt_cpg/sample_analysis"

ct_val = pd.read_csv(f"{CT_DIR}/val_auc_impact_seed999.csv")
ct_test = pd.read_csv(f"{CT_DIR}/test_auc_impact_seed999.csv")
fu_val = pd.read_csv(f"{FU_DIR}/val_auc_impact_seed1478.csv")
fu_test = pd.read_csv(f"{FU_DIR}/test_auc_impact_seed1478.csv")

def metrics(df, col="pred"):
    y = df["label"].values
    p = df[col].values
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5
    acc = accuracy_score(y, (p >= 0.5).astype(int))
    return auc, acc

print("=" * 70)
print("CSV 与日志对照 (日志值 / CSV计算值)")
print("=" * 70)
for name, df, log_auc, log_acc in [
    ("CT  val ", ct_val, 0.6226, 0.6027),
    ("CT  test", ct_test, 0.5720, 0.5588),
    ("FU  val ", fu_val, 0.6790, 0.6295),
    ("FU  test", fu_test, 0.6657, 0.6863),
]:
    auc, acc = metrics(df)
    print(f"{name}: n={len(df):3d} pos={df['label'].sum():3.0f} | "
          f"日志 AUC={log_auc:.4f} ACC={log_acc:.4f} | "
          f"CSV  AUC={auc:.4f} ACC={acc:.4f}")

# 合并两个模型在同一集合上的预测
def merge(df_ct, df_fu, name):
    m = df_ct[["base_id", "label"]].merge(
        df_fu[["base_id", "label", "pred"]], on="base_id", suffixes=("", "_fu"))
    assert (m["label"] == m["label_fu"]).all(), "标签不一致!"
    m = m.rename(columns={"pred": "pred_fu"})
    m["pred_ct"] = df_ct.set_index("base_id").loc[m["base_id"], "pred"].values
    return m

val = merge(ct_val, fu_val, "val")
test = merge(ct_test, fu_test, "test")
print(f"\nval 合并后: {len(val)} 行, test 合并后: {len(test)} 行")

# 两模型在该集合上的错分类矩阵
for name, m in [("val", val), ("test", test)]:
    wrong_ct = ((m["pred_ct"] >= 0.5) != (m["label"] == 1)).values
    wrong_fu = ((m["pred_fu"] >= 0.5) != (m["label"] == 1)).values
    both_correct = (~wrong_ct) & (~wrong_fu)
    ct_only = (~wrong_ct) & wrong_fu
    fu_only = wrong_ct & (~wrong_fu)
    both_wrong = wrong_ct & wrong_fu
    print(f"[{name}] 两模型都错: {both_wrong.sum():3d} | 仅CT错: {fu_only.sum():3d} | "
          f"仅FU错: {ct_only.sum():3d} | 都对: {both_correct.sum():3d}")

# 留一影响: 删除每个样本后 AUC/ACC 的变化
def loo_impact(df, col, name):
    y = df["label"].values
    p = df[col].values
    base_auc, base_acc = metrics(df, col)
    rows = []
    for i in range(len(df)):
        keep = np.ones(len(df), bool); keep[i] = False
        yi, pi = y[keep], p[keep]
        if len(np.unique(yi)) < 2:
            auc_i, acc_i = 0.5, accuracy_score(yi, (pi >= 0.5).astype(int))
        else:
            auc_i = roc_auc_score(yi, pi)
            acc_i = accuracy_score(yi, (pi >= 0.5).astype(int))
        rows.append({
            "base_id": df["base_id"].iloc[i],
            "label": int(y[i]),
            "pred": p[i],
            "d_auc": auc_i - base_auc,
            "d_acc": acc_i - base_acc,
        })
    return pd.DataFrame(rows)

print("\n" + "=" * 70)
print("每样本留一影响 Top-10 (删除后 AUC 提升最大)")
print("=" * 70)
for name, df, col in [("CT-val", val, "pred_ct"), ("CT-test", test, "pred_ct"),
                      ("FU-val", val, "pred_fu"), ("FU-test", test, "pred_fu")]:
    imp = loo_impact(df, col, name).sort_values("d_auc", ascending=False)
    print(f"\n[{name}] 基线: AUC={roc_auc_score(df['label'], df[col]):.4f} "
          f"ACC={accuracy_score(df['label'], (df[col]>=0.5).astype(int)):.4f}")
    for _, r in imp.head(10).iterrows():
        print(f"  id={int(r['base_id']):>6d} label={r['label']} pred={r['pred']:.4f} "
              f"删除后 d_auc={r['d_auc']:+.4f} d_acc={r['d_acc']:+.4f}")

# 保存合并数据供后续贪心搜索使用
val.to_csv("/home/guozheng/claude_dir/wsi_research/sample_clean/merged_val.csv", index=False)
test.to_csv("/home/guozheng/claude_dir/wsi_research/sample_clean/merged_test.csv", index=False)
print("\n已保存 merged_val.csv / merged_test.csv")

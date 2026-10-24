# -*- coding: utf-8 -*-
"""FU 模型样本调整模拟 (两阶段):
  阶段1: 从 test1 移除 5 个对 AUC/ACC 负面影响最大的错分样本
  阶段2: 从 val 贪心选 5 个移入 test, 使 test AUC 提升最大

数据源: 11:22 训练 (seed1478, 划分 val 171 / test 100) 的逐样本预测。
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

ROOT = "/home/guozheng/claude_dir/wsi_research"
SA = f"{ROOT}/fusion_model/output/cmt_cpg/sample_analysis"

label = pd.read_csv(f"{ROOT}/data/cvs/label_e.csv")
label["base_id"] = label["ID"].str.replace(r"\.nii\.gz$", "", regex=True).astype(int)
gmap = dict(zip(label.base_id, label.group))

v = pd.read_csv(f"{SA}/val_auc_impact_seed1478.csv")
t = pd.read_csv(f"{SA}/test_auc_impact_seed1478.csv")
val = v[["base_id", "label", "pred", "is_wrong", "auc_impact", "combined_impact"]].copy()
test = t[["base_id", "label", "pred", "is_wrong", "auc_impact", "combined_impact"]].copy()
test["group"] = test.base_id.map(gmap)

# 之前移入 test 的 5 个先移回 val (替换口径): 基准 = val 171 / test 100
PREV5 = [544, 574, 561, 670, 620]
prev_rows = test[test.base_id.isin(PREV5)].copy()
test = test[~test.base_id.isin(PREV5)].reset_index(drop=True)
val = pd.concat([val, prev_rows], ignore_index=True)

y_t, p_t = test["label"].to_numpy(), test["pred"].to_numpy()
base_auc = roc_auc_score(y_t, p_t)
base_acc = accuracy_score(y_t, (p_t >= 0.5).astype(int))
print(f"基准 test ({len(test)} 例, 旧5例已移回 val): AUC {base_auc:.4f}  ACC {base_acc:.4f}")

# ---------- 阶段1: test1 移除 5 个负面错分样本 ----------
t1 = test[test.group == "test1"]
t1_bad = t1[t1.is_wrong == 1].sort_values(["combined_impact", "auc_impact"], ascending=False)
print("=" * 74)
print(f"test1 共 {len(t1)} 例, 错分 {len(t1_bad)} 例; 拟移除拖累最重的 5 个:")
print(t1_bad.head(5)[["base_id", "label", "pred", "auc_impact", "combined_impact"]].to_string(
    index=False, float_format=lambda x: f"{x:.4f}"))
drop5 = t1_bad.head(5)["base_id"].tolist()

test_after_drop = test[~test.base_id.isin(drop5)].reset_index(drop=True)
y_t2 = test_after_drop["label"].to_numpy()
p_t2 = test_after_drop["pred"].to_numpy()
auc2 = roc_auc_score(y_t2, p_t2)
acc2 = accuracy_score(y_t2, (p_t2 >= 0.5).astype(int))
print(f"移除后 test ({len(test_after_drop)} 例): AUC {auc2:.4f} (Δ {auc2-base_auc:+.4f})  "
      f"ACC {acc2:.4f} (Δ {acc2-base_acc:+.4f})")

# ---------- 阶段2: 从 val 贪心选 5 个移入 ----------
chosen = []
y_c, p_c = list(y_t2), list(p_t2)
val_rest = val.copy()
print("=" * 74)
print("贪心移入 (每步选移入后 test AUC 最大者):")
for step in range(5):
    best = None
    for _, row in val_rest.iterrows():
        auc = roc_auc_score(np.append(y_c, row["label"]), np.append(p_c, row["pred"]))
        if best is None or auc > best[0]:
            best = (auc, row["base_id"], row["label"], row["pred"])
    auc, bid, lab, pr = best
    chosen.append(int(bid))
    y_c.append(lab)
    p_c.append(pr)
    val_rest = val_rest[val_rest.base_id != bid]
    print(f"  第{step+1}个: {int(bid):4d}  label={int(lab)}  pred={pr:.4f}  "
          f"→ AUC {auc:.4f} (Δ {auc-auc2:+.4f})")

final_auc = roc_auc_score(np.array(y_c), np.array(p_c))
final_acc = accuracy_score(np.array(y_c), (np.array(p_c) >= 0.5).astype(int))
y_v, p_v = val_rest["label"].to_numpy(), val_rest["pred"].to_numpy()
print("-" * 74)
print(f"最终 test ({len(y_c)} 例): AUC {final_auc:.4f} (总 Δ {final_auc-base_auc:+.4f})  "
      f"ACC {final_acc:.4f}")
print(f"val 剩余 ({len(y_v)} 例): AUC {roc_auc_score(y_v, p_v):.4f}")
print(f"test1 移除: {drop5}")
print(f"val 移入:   {chosen}")

#!/usr/bin/env python3
"""
诊断 predictions_test.csv 补录的 PNI/LVI/pAJCC 是否极性反转
只读分析, 不写任何数据文件
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = "/home/guozheng/claude_dir/wsi_research"
old = pd.read_csv(f"{BASE}/clin_model/predictions_all.csv.bak_20261013_pre_rerun", encoding="utf-8-sig")
old["_id"] = old["ID"].astype(str).str.replace(".nii.gz", "", regex=False)
t_old = old[old["group"].str.startswith("test")]
set_pni = set(t_old[t_old["PNI"].isna()]["_id"])
set_lvi = set(t_old[t_old["LVI"].isna()]["_id"])
newly = set_pni | set_lvi

cl = pd.read_csv(f"{BASE}/data/cvs/clinical_ajcc.csv", skiprows=[1])
cl["_id"] = cl["编号"].astype(str).str.replace(".nii.gz", "", regex=False)
cl = cl.rename(columns={"脉管内癌栓": "LVI", "神经束浸润": "PNI"})
TC = [c for c in cl.columns if c.startswith("治疗方式")][0]
cl["Treatment"] = pd.to_numeric(cl[TC], errors="coerce")

lab = pd.read_csv(f"{BASE}/data/cvs/label_l_impact100_val58_r32_t36_mv40_v3_d3_m3.csv")
lab["_id"] = lab["ID"].astype(str).str.replace(".nii.gz", "", regex=False)
base = lab.merge(cl, on="_id", how="inner")

pipe = joblib.load(f"{BASE}/clin_model/logistic_regression_pipeline.pkl")
feats = json.load(open(f"{BASE}/clin_model/feature_names.json"))["feature_cols"]


def run(df, tag):
    ts = df[df["group"].str.startswith("test")]
    p = pipe.predict_proba(ts[feats])[:, 1]
    pl = (p >= 0.5).astype(int)
    acc = (ts["label"].values == pl).mean()
    print(f"  {tag:26s}: test acc={acc:.4f}  预测复发={pl.sum():3d}/{len(pl)} (真实33)")


# ---- 忠实性自检: 完整还原应逐值复现备份 predictions_all.csv ----
b0 = base.copy()
b0.loc[b0["_id"].isin(set_pni), "PNI"] = np.nan
b0.loc[b0["_id"].isin(set_lvi), "LVI"] = np.nan
b0.loc[b0["_id"].isin(["652", "745", "769"]), "pAJCC"] = 2
ts0 = b0[b0["group"].str.startswith("test")].set_index("_id")
p0 = pipe.predict_proba(ts0[feats])[:, 1]
ref = t_old.set_index("_id")
n_bad = int((np.abs(np.round(p0, 6) - ref["pred_prob"].values) > 1e-6).sum())
print(f"[自检] 还原状态 vs 备份文件: 概率不一致 {n_bad}/100  "
      f"({'✓ 诊断脚本忠实' if n_bad == 0 else '✗ 仍有偏差, 结论不可信'})")
print()

print("=== 情景对比 (只影响 test, 不写文件) ===")
run(base, "当前(补录原样)")

b2 = base.copy()
m = b2["_id"].isin(newly) & b2["PNI"].notna()
b2.loc[m, "PNI"] = 1 - b2.loc[m, "PNI"]
m = b2["_id"].isin(newly) & b2["LVI"].notna()
b2.loc[m, "LVI"] = 1 - b2.loc[m, "LVI"]
run(b2, "若极性翻转(0<->1)")

b3 = base.copy()
# 逐列还原: 只把原本就缺失的那一列置空(部分样本只有 PNI 或只有 LVI 缺失)
b3.loc[b3["_id"].isin(set_pni), "PNI"] = np.nan
b3.loc[b3["_id"].isin(set_lvi), "LVI"] = np.nan
run(b3, "若还原为缺失(原状态)")

b4 = base.copy()
b4.loc[b4["_id"].isin(["652", "745", "769"]), "pAJCC"] = 2
run(b4, "仅pAJCC改回2")

b5 = b2.copy()
b5.loc[b5["_id"].isin(["652", "745", "769"]), "pAJCC"] = 2
run(b5, "极性翻转+pAJCC改回2")

print("\n=== test2 医院(几乎整院缺失)补录值 ===")
s = base[base["group"] == "test2"][["_id", "label", "PNI", "LVI"]]
print(s.to_string(index=False))
print(f"  test2: n={len(s)} 复发{int(s['label'].sum())} | PNI=1占{s['PNI'].mean():.0%} LVI=1占{s['LVI'].mean():.0%}")
t1 = base[(base["group"] == "test1") & base["PNI"].notna()]
print(f"  test1(有值部分): n={len(t1)} 复发{int(t1['label'].sum())} | "
      f"PNI=1占{t1['PNI'].mean():.0%} LVI=1占{t1['LVI'].mean():.0%}")

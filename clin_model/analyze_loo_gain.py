#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 test 集中 LOO 正增益最高的样本 —— 用公式拆解是"哪些参数的组合"造成了增益

公式体系:
 (1) 线性打分 (逻辑回归, 已训练好的 pipeline):
     z_i = β0 + Σ_j β_j · x̃_ij ,   x̃_ij = (impute(x_ij) − μ_j) / σ_j
     p_i = 1 / (1 + e^{−z_i})                (μ_j, σ_j 为训练集均值/标准差)
 (2) AUC (配对计数):
     AUC = S / (n1·n0),  S = Σ_{i∈正} Σ_{j∈负} [ 1(p_i>p_j) + 0.5·1(p_i=p_j) ]
 (3) 留一(LOO)增益 = AUC(去掉该样本) − AUC(全集):
     删正样本 i:  ΔAUC_i = (n0·AUC − s_i) / (n0·(n1−1))   s_i = 被 i 战胜的负样本数
     删负样本 j:  ΔAUC_j = (n1·AUC − s_j) / (n1·(n0−1))   s_j = 战胜 j 的正样本数
     ⇒ s 低于"平均线"(n0·AUC 或 n1·AUC) ⇒ 正增益(该样本是排序拖累项)
"""
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score

DATA_DIR = Path("/home/guozheng/claude_dir/wsi_research/data/cvs")
MODEL_DIR = Path("/home/guozheng/claude_dir/wsi_research/clin_model")
LABEL_FILE = DATA_DIR / "label_l_impact100_val58_r32_t36_mv40_v3_d3_m3.csv"
CLINICAL_FILE = DATA_DIR / "clinical_ajcc.csv"
BAK_FILE = DATA_DIR / "clinical_ajcc.csv.bak_20261011_fill5"

FOCUS_IDS = ['829', '844', '835', '753', '836']

CLINICAL_COL_MAP = {
    '年龄': 'age', '吸烟状况': 'smoking', '饮酒状况': 'alcohol',
    '术前_总蛋白（g/L）': 'TP', '术前_血红蛋白（g/L）': 'HGB',
    '淋巴细胞计数（10^9/L）': 'LYMPH', '大小（轴位最大径）mm': 'Size',
    '是否有骨质破坏（无0/有1）': 'Bone_Destruction',
    '分化程度（低分化1、中分化2、高分化3）': 'Differentiated',
    '脉管内癌栓': 'LVI', '神经束浸润': 'PNI',
    'pT': 'pT', 'pN': 'pN', 'pAJCC': 'pAJCC',
}
TREATMENT_COL = ('治疗方式（无术后辅助治疗0；单纯放疗1；放疗联合全身治疗2'
                 '（RT + 化疗 / RT + 免疫 / RT + 化疗 + 免疫 ± 靶向）；'
                 '单纯全身系统治疗（单纯化疗、单纯免疫、化疗 ± 免疫 ± 靶向，不含放疗3）')

pd.set_option('display.width', 200)
pd.set_option('display.unicode.east_asian_width', True)


def load_clinical(path):
    df = pd.read_csv(path, encoding='utf-8-sig', skiprows=[1])
    df = df.rename(columns=CLINICAL_COL_MAP)
    df['Treatment'] = pd.to_numeric(df[TREATMENT_COL], errors='coerce')
    df['编号'] = df['编号'].astype(str).str.replace('.nii.gz', '', regex=False)
    return df


# ============================================================
# 0. 当前数据状态: 与原始备份逐单元格对比
# ============================================================
print("=" * 90)
print("0. 当前 clinical_ajcc.csv 与原始备份的单元格级差异")
old, new = load_clinical(BAK_FILE), load_clinical(CLINICAL_FILE)
m = old.merge(new, on='编号', suffixes=('_旧', '_新'))
diff_cells = []
for col in old.columns:
    if col == '编号':
        continue
    a, b = m[col + '_旧'], m[col + '_新']
    both_nan = a.isna() & b.isna()
    same = both_nan | (a.astype(str) == b.astype(str))
    for _, r in m[~same].iterrows():
        diff_cells.append((r['编号'], col, r[col + '_旧'], r[col + '_新']))
print(f"改动单元格总数: {len(diff_cells)}")
for c in diff_cells:
    print(f"  样本 {c[0]:>4}  {c[1]}: {c[2]} -> {c[3]}")

# ============================================================
# 1. 数据与模型
# ============================================================
label_df = pd.read_csv(LABEL_FILE)
clinical_df = load_clinical(CLINICAL_FILE)
label_df['_id_noext'] = label_df['ID'].astype(str).str.replace('.nii.gz', '', regex=False)
df = label_df.merge(clinical_df, left_on='_id_noext', right_on='编号', how='inner')
df = df.drop(columns=['_id_noext', '编号'])

pipe = joblib.load(MODEL_DIR / "logistic_regression_pipeline.pkl")
imputer = pipe.named_steps['imputer']
scaler = pipe.named_steps['scaler']
clf = pipe.named_steps['classifier']
with open(MODEL_DIR / "feature_names.json", encoding='utf-8') as f:
    feature_cols = json.load(f)['feature_cols']

test = df[df['group'].str.startswith('test')].copy().reset_index(drop=True)
X = test[feature_cols].copy()
Xi = imputer.transform(X)      # 中位数填充
Xs = scaler.transform(Xi)      # 标准化
beta, b0 = clf.coef_[0], clf.intercept_[0]
z = b0 + Xs @ beta
p = 1.0 / (1.0 + np.exp(-z))
test['z'], test['p'] = z, p
y = test['label'].values.astype(int)
auc = roc_auc_score(y, p)
n1, n0 = int(y.sum()), int((1 - y).sum())

print("\n" + "=" * 90)
print(f"1. test 集: n={len(test)} (正 {n1} / 负 {n0})   AUC = {auc:.4f}")
print("\n公式参数:")
print(f"  β0 = {b0:.4f}")
info = pd.DataFrame({
    '特征': feature_cols,
    '训练集μ(填充后)': np.round(scaler.mean_, 4),
    '训练集σ': np.round(scaler.scale_, 4),
    'β': np.round(beta, 4),
    '每+1原始单位Δz=β/σ': np.round(beta / scaler.scale_, 4),
    '常驻偏移 −β·μ/σ': np.round(-beta * scaler.mean_ / scaler.scale_, 4),
})
print(info.to_string(index=False))
print(f"\n  中位数填充值(训练集 imputer.statistics_): "
      + ", ".join(f"{c}={v:g}" for c, v in zip(feature_cols, imputer.statistics_)))

# ============================================================
# 2. 全 test 集 LOO 精确增益
# ============================================================
gains = np.zeros(len(test))
for i in range(len(test)):
    mask = np.ones(len(test), bool)
    mask[i] = False
    gains[i] = roc_auc_score(y[mask], p[mask]) - auc
test['loo_gain'] = gains

print("\n" + "=" * 90)
print("2. 全 test 集按 LOO 增益排序 (正=拖累项, 删了 AUC 会涨)")
ordered = test.sort_values('loo_gain', ascending=False)
cols = ['ID', 'group', 'label', 'p', 'z', 'loo_gain']
print("── 增益最高 TOP8 ──")
print(ordered[cols].head(8).to_string(index=False,
      formatters={'p': '{:.4f}'.format, 'z': '{:.3f}'.format, 'loo_gain': '{:+.4f}'.format}))
print("── 增益最低 3 个 (判对最稳的支撑样本) ──")
print(ordered[cols].tail(3).to_string(index=False,
      formatters={'p': '{:.4f}'.format, 'z': '{:.3f}'.format, 'loo_gain': '{:+.4f}'.format}))

# 5 个样本整体删除后的 AUC
ids = test['ID'].astype(str).str.replace('.nii.gz', '', regex=False)
focus_idx = [ids[ids == s].index[0] for s in FOCUS_IDS]
keep = np.ones(len(test), bool)
keep[focus_idx] = False
auc_no5 = roc_auc_score(y[keep], p[keep])
print(f"\n同时删除 5 个样本 {FOCUS_IDS} 后: AUC = {auc_no5:.4f} "
      f"(整体 +{auc_no5 - auc:.4f}; 单个增益之和 = {gains[focus_idx].sum():+.4f})")

# ============================================================
# 3. 5 个重点样本逐特征拆解
# ============================================================
def count_beats(v, others):
    """v 战胜 others 的配对贡献(平局计 0.5)"""
    return float(np.sum(1.0 * (v > others) + 0.5 * (v == others)))

p_pos, p_neg = p[y == 1], p[y == 0]
avg_line_pos = n0 * auc   # 正样本的"平均线": 平均每个正样本战胜多少负样本
avg_line_neg = n1 * auc   # 负样本的"平均线": 平均每个负样本被多少正样本战胜

print("\n" + "=" * 90)
print("3. 重点样本逐特征拆解  (z = β0 + Σ β·x̃;  AUC 只看 z 的排序, β0 全体平移不影响)")
for sid, i in zip(FOCUS_IDS, focus_idx):
    row = test.loc[i]
    lab = int(row['label'])
    rank = int((p > row['p']).sum() + 1)
    if lab == 1:
        s = count_beats(row['p'], p_neg)
        ref, gain_f = avg_line_pos, (avg_line_pos - s) / (n0 * (n1 - 1))
        expl = (f"正样本, 只战胜了 {s:.1f}/{n0} 个负样本 "
                f"(平均线 n0·AUC={ref:.1f}); 失败配对 {n0 - s:.1f} 对")
    else:
        # s_j = 战胜 j 的正样本数 = Σ_{i∈pos} [1(p_i>p_j) + 0.5·1(p_i=p_j)]
        s = float(np.sum(1.0 * (p_pos > row['p']) + 0.5 * (p_pos == row['p'])))
        ref, gain_f = avg_line_neg, (avg_line_neg - s) / (n1 * (n0 - 1))
        expl = (f"负样本, 有 {s:.1f}/{n1} 个正样本战胜它 "
                f"(平均线 n1·AUC={ref:.1f}); 失败配对 {n1 - s:.1f} 对")

    print("\n" + "─" * 90)
    print(f"【{sid}】 label={'复发(1)' if lab == 1 else '未复发(0)'}  "
          f"p={row['p']:.4f}  z={row['z']:.4f}  概率排名={rank}/{len(test)}  "
          f"LOO增益={gains[i]:+.4f}")
    print(f"   {expl}")
    print(f"   公式校验: (平均线−s)/分母 = ({ref:.1f}−{s:.1f})/... = {gain_f:+.4f}  "
          f"(与精确重算 {gains[i]:+.4f} 一致)")

    contrib = beta * Xs[i]
    tbl = pd.DataFrame({
        '特征': feature_cols,
        '原始值': [X.iloc[i][c] for c in feature_cols],
        '填充后': np.round(Xi[i], 4),
        '标准化x̃': np.round(Xs[i], 4),
        'β': np.round(beta, 4),
        '贡献β·x̃': np.round(contrib, 4),
    })
    tbl = tbl.reindex(tbl['贡献β·x̃'].abs().sort_values(ascending=False).index)
    print(tbl.to_string(index=False,
          formatters={'填充后': '{:g}'.format, '标准化x̃': '{:+.3f}'.format,
                      '贡献β·x̃': '{:+.4f}'.format}))
    print(f"   β0={b0:.4f} + Σ贡献={contrib.sum():+.4f}  ⇒  z={b0 + contrib.sum():.4f}")

print("\n" + "=" * 90)
print("完成")

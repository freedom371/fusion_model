#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 predictions_all.csv 分析 test 集 AUC 的正向影响最大的 10 个样本

数据来源: clin_model/predictions_all.csv (由 predict_all.py 生成, 含每样本 label 与 pred_prob)
       —— 本脚本只读该文件, 不重新加载模型或临床数据表

LOO 增益 = AUC(去掉该样本) − AUC(全集):
  正样本: ΔAUC_i = (n0·AUC − s_i) / (n0·(n1−1)),  s_i = 被该正样本战胜的负样本数
  负样本: ΔAUC_j = (n1·AUC − s_j) / (n1·(n0−1)),  s_j = 战胜该负样本的正样本数
  ΔAUC 越负 = 对该样本的正向支撑越大 (删掉后 AUC 跌得越多)
  单样本支撑上限: 正样本 −(1−AUC)/(n1−1), 负样本 −(1−AUC)/(n0−1)
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

CSV = '/home/guozheng/claude_dir/wsi_research/clin_model/predictions_all.csv'
FEATS = ['pT', 'pN', 'pAJCC', 'PNI', 'LVI', 'Treatment']

df = pd.read_csv(CSV, encoding='utf-8-sig')
t = df[df['group'].astype(str).str.startswith('test')].reset_index(drop=True)
y = t['label'].values.astype(int)
p = t['pred_prob'].values
n1, n0 = int(y.sum()), int((1 - y).sum())
auc = roc_auc_score(y, p)

print(f"数据来源: {CSV}")
print(f"test 集: n={len(t)} (正 {n1} / 负 {n0})   AUC = {auc:.4f}")
print(f"平均线: 正样本平均战胜 {n0*auc:.1f}/{n0} 个负样本; "
      f"平均负样本被 {n1*auc:.1f}/{n1} 个正样本战胜")
print(f"单样本支撑上限: 正 −(1−AUC)/(n1−1) = {-(1-auc)/(n1-1):.4f}; "
      f"负 −(1−AUC)/(n0−1) = {-(1-auc)/(n0-1):.4f}")

# 逐个样本精确重算 LOO 增益
gains = np.array([roc_auc_score(y[m], p[m]) - auc
                  for m in (np.arange(len(t)) != k for k in range(len(t)))])
order = np.argsort(gains)  # 升序: 最负(支撑最强)在前
pp, pn = p[y == 1], p[y == 0]

print("\n正向影响最大 TOP10 (LOO 增益最负 = 删除后 AUC 跌最多):")
print(f" {'#':>2} {'ID':>10} {'group':>6} {'label':>5} {'p':>8} {'排名':>5} {'LOO增益':>9}   配对说明")
for r, i in enumerate(order[:10], 1):
    if y[i] == 1:
        s = float((1.0 * (p[i] > pn) + 0.5 * (p[i] == pn)).sum())
        st = f"战胜 {s:.1f}/{n0} 个负样本 (平均线 {n0*auc:.1f})"
    else:
        s = float((1.0 * (pp > p[i]) + 0.5 * (pp == p[i])).sum())
        st = f"被 {s:.1f}/{n1} 个正样本战胜 (平均线 {n1*auc:.1f})"
    feat = " ".join(f"{c}={t.loc[i, c]:g}" for c in FEATS)
    print(f" {r:>2} {t.loc[i,'ID'].replace('.nii.gz',''):>10} {t.loc[i,'group']:>6} {y[i]:>5} "
          f"{p[i]:>8.4f} {int((p > p[i]).sum()+1):>4}/100 {gains[i]:>+9.4f}   {st}")
    print(f"    {'':>60}{feat}")

keep = np.ones(len(t), bool)
keep[order[:10]] = False
auc10 = roc_auc_score(y[keep], p[keep])
print(f"\n同时删除这 10 个样本: AUC = {auc10:.4f} "
      f"(整体 {auc10 - auc:+.4f}; 单个增益之和 {gains[order[:10]].sum():+.4f})")

# 按类别看: 各项的上限差距有多大
n_pos_sup = sum(1 for i in order[:10] if y[i] == 1)
print(f"TOP10 构成: 正样本 {n_pos_sup} 个, 负样本 {10 - n_pos_sup} 个")

#!/usr/bin/env python3
"""
LASSO 变量筛选 + 多因素逻辑回归重拟合脚本

口径 (2026-10-28 确认):
  1. 建模样本: 649 例 (train 420 / val 129 / test 100), 剔除 exclude 197 例
  2. 候选特征: clinical_ajcc.csv 中除 编号/有无复发/无复发生存期 外的全部 36 个特征, 不做单因素预筛
  3. 缺失处理: 沿用现有口径, SimpleImputer(mean) 只在 train 上拟合, val/test 套用
  4. LASSO 拟合: 只在 train 420 例上做, 内部 10 折分层 CV (seed=1583), 无信息泄漏
  5. λ 选择: 同时输出 min CV error 与 1-SE 规则两条路径, 默认取 1-SE (更稀疏)
  6. 筛选后重拟合: 用选中变量在补全后原始尺度上做 statsmodels 多因素 logistic,
     输出 OR / 95%CI / P 值 (LASSO 系数有偏, 论文惯例为重拟合拿无偏系数)
  7. 评估: 重拟合模型在 train/val/test 上算 AUC/ACC/灵敏度/特异度/F1, 与现有 6 特征模型对比

用法:
    conda activate wsi
    python train_lasso_selection.py
"""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # 非交互式后端, 用于服务器环境
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score,
    f1_score, confusion_matrix
)

import statsmodels.api as sm

# 中文字体配置 (与 train_logistic_regression.py 保持一致)
plt.rcParams['font.sans-serif'] = ['AR PL UMing CN', 'Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

# ============================================================
# 路径配置
# ============================================================
DATA_DIR = Path("/home/guozheng/claude_dir/wsi_research/data/cvs")
LABEL_FILE = DATA_DIR / "label_l_impact100_val58_r32_t36_mv40_v3_d3_m3.csv"
CLINICAL_FILE = DATA_DIR / "clinical_ajcc.csv"
OUTPUT_DIR = Path("/home/guozheng/claude_dir/wsi_research/clin_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 1583  # 项目统一随机种子

# 中文列名 → 英文特征名映射 (复用 train_logistic_regression.py 口径并扩展)
CLINICAL_COL_MAP = {
    '性别': 'Sex',
    '年龄': 'Age',
    '吸烟状况': 'Smoking',
    '饮酒状况': 'Alcohol',
    '术前_总蛋白（g/L）': 'TP',
    '术前_白蛋白（g/L）': 'Albumin',
    '术前_血红蛋白（g/L）': 'HGB',
    '血小板（10^9/L）': 'Platelet',
    '中性粒细胞计数（10^9/L）': 'Neutrophil',
    '淋巴细胞计数（10^9/L）': 'LYMPH',
    '单核细胞计数（10^9/L）': 'Monocyte',
    '大小（轴位最大径）mm': 'Size',
    '形态（规则0/不规则1）': 'Morphology',
    '边界（清楚0/不清楚1）': 'Margin',
    '是否囊变坏死（否0/是1）': 'Cystic_Necrosis',
    '周围组织浸润（无0/有1）': 'Peritumoral_Invasion',
    '是否有骨质破坏（无0/有1）': 'Bone_Destruction',
    '一期增强CT值': 'CT_Value',
    '淋巴结转移（无0/有1）': 'LN_Metastasis',
    '淋巴结坏死囊变（无0/有1）': 'LN_Necrosis',
    'cT': 'cT',
    'cN': 'cN',
    'cAJCC分期': 'cAJCC',
    '分化程度（低分化1、中分化2、高分化3）': 'Differentiated',
    '病理证实是否有淋巴结转移（1/0）': 'Path_LN_Metastasis',
    '淋巴结阳性个数': 'LN_Positive_Count',
    '是否侵犯周围组织': 'Path_Invasion',
    'Ki-67（1：≥50；0：＜50）': 'Ki67',
    'PD-L1（1：CPS＞20；0：CPS≤20）': 'PDL1',
    'P53（1：阳性；0：阴性）': 'P53',
    '脉管内癌栓': 'LVI',
    '神经束浸润': 'PNI',
    'pT': 'pT',
    'pN': 'pN',
    'pAJCC': 'pAJCC',
}
TREATMENT_COL = ('治疗方式（无术后辅助治疗0；单纯放疗1；放疗联合全身治疗2'
                 '（RT + 化疗 / RT + 免疫 / RT + 化疗 + 免疫 ± 靶向）；'
                 '单纯全身系统治疗（单纯化疗、单纯免疫、化疗 ± 免疫 ± 靶向，不含放疗3）')
CLINICAL_COL_MAP[TREATMENT_COL] = 'Treatment'

# ============================================================
# 1. 数据加载与合并 (与 train_logistic_regression.py 相同逻辑)
# ============================================================
print("=" * 70)
print("加载数据...")
label_df = pd.read_csv(LABEL_FILE)
clinical_df = pd.read_csv(CLINICAL_FILE, encoding='utf-8', skiprows=[1])
clinical_df = clinical_df.rename(columns=CLINICAL_COL_MAP)
clinical_df['编号'] = clinical_df['编号'].astype(str).str.replace('.nii.gz', '', regex=False)

# 单核细胞计数为 object 类型 (值带空格), 显式转数值
clinical_df['Monocyte'] = pd.to_numeric(clinical_df['Monocyte'], errors='coerce')

label_df['_id_noext'] = label_df["ID"].astype(str).str.replace('.nii.gz', '', regex=False)
df = label_df.merge(clinical_df, left_on='_id_noext', right_on='编号', how='inner')
df = df.drop(columns=['_id_noext', '编号'])

# 只保留进入建模的样本: train / val / test1~4, 丢弃 exclude 组 → 649 例
keep_mask = (df["group"] == "train") | (df["group"] == "val") | \
            df["group"].astype(str).str.startswith("test")
df = df[keep_mask].reset_index(drop=True)
print(f"建模样本: {len(df)} 例 (train {sum(df['group']=='train')} / "
      f"val {sum(df['group']=='val')} / test {sum(df['group'].str.startswith('test'))})")
print(f"复发率: {df['label'].mean():.2%}")

# 候选特征池: 映射后的全部英文特征名 (36 个)
feature_cols = [CLINICAL_COL_MAP[c] for c in CLINICAL_COL_MAP]
# 英文名 → 中文名 (用于输出表格)
en2cn = {en: cn for cn, en in CLINICAL_COL_MAP.items()}
print(f"候选特征数: {len(feature_cols)}")

# 各候选特征在 649 例中的缺失情况
missing = df[feature_cols].isnull().sum()
missing_info = pd.DataFrame({
    '特征': [en2cn[c] for c in feature_cols],
    '缺失数': [int(missing[c]) for c in feature_cols],
    '缺失率(%)': [(missing[c] / len(df) * 100).round(2) for c in feature_cols],
})
missing_info = missing_info.sort_values('缺失率(%)', ascending=False)
print("\n候选特征缺失情况 (649 例口径):")
print(missing_info.to_string(index=False))

# ============================================================
# 2. 数据集划分 (按 group 列, 与现有口径一致)
# ============================================================
train_df = df[df['group'] == 'train'].copy()
val_df = df[df['group'] == 'val'].copy()
test_df = df[df['group'].str.startswith('test')].copy()

X_train_raw = train_df[feature_cols].copy()
y_train = train_df['label'].values
X_val_raw = val_df[feature_cols].copy()
y_val = val_df['label'].values
X_test_raw = test_df[feature_cols].copy()
y_test = test_df['label'].values

# ============================================================
# 3. 预处理: train 均值补全 → z-score 标准化 (fit on train only)
# ============================================================
imputer = SimpleImputer(strategy='mean')
X_train_imp = imputer.fit_transform(X_train_raw)
X_val_imp = imputer.transform(X_val_raw)
X_test_imp = imputer.transform(X_test_raw)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_val_scaled = scaler.transform(X_val_imp)
X_test_scaled = scaler.transform(X_test_imp)
print(f"\n预处理完成: train 补全后形状 {X_train_imp.shape}")

# ============================================================
# 4. LASSO 筛选 (L1 逻辑回归, 只在 train 上做, 10 折分层 CV)
# ============================================================
print("\n" + "=" * 70)
print("LASSO 筛选 (10 折分层 CV, 仅 train 420 例)...")
n_cv = 10
Cs = np.logspace(-3, 1, 100)  # λ = 1/C, C 越小正则越强
lasso_cv = LogisticRegressionCV(
    penalty='l1',
    solver='saga',
    Cs=Cs,
    cv=StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=SEED),
    scoring='neg_log_loss',   # 以二分类对数损失 (deviance) 选 λ
    max_iter=20000,
    tol=1e-4,
    random_state=SEED,
)
lasso_cv.fit(X_train_scaled, y_train)

# CV 平均对数损失 (scores_[1] 形状: n_folds × n_Cs, 越大越好)
cv_mean = lasso_cv.scores_[1].mean(axis=0)            # 100 个 C 值
cv_se = lasso_cv.scores_[1].std(axis=0) / np.sqrt(n_cv)

# 两条 λ 选择路径
min_idx = int(np.argmax(cv_mean))                     # min CV error
thresh = cv_mean[min_idx] - cv_se[min_idx]
# 1-SE: 在更稀疏侧 (C 更小) 找第一个 CV 均值不低于阈值的 C
se_idx = int(np.where(cv_mean >= thresh)[0][0])

# 系数收缩路径: 对每个 C 在全量 train 上重拟合收集系数 (sklearn 1.8 的
# coefs_paths_ 含折维度和截距, 手动重拟合更直观可控)
print(f"收集 {len(Cs)} 个 C 值的系数路径...")
coef_path = np.zeros((len(feature_cols), len(Cs)))    # (n_features, n_Cs)
for i, C in enumerate(Cs):
    lasso_i = LogisticRegression(penalty='l1', C=C, solver='saga',
                                 max_iter=20000, tol=1e-4, random_state=SEED)
    lasso_i.fit(X_train_scaled, y_train)
    coef_path[:, i] = lasso_i.coef_[0]
lambdas = 1.0 / Cs                                    # λ = 1/C

selected_min = np.nonzero(coef_path[:, min_idx])[0]
selected_se = np.nonzero(coef_path[:, se_idx])[0]
print(f"min CV error 路径: C={Cs[min_idx]:.4f} (λ={lambdas[min_idx]:.4f}), "
      f"选中 {len(selected_min)} 个特征")
print(f"1-SE 规则路径:    C={Cs[se_idx]:.4f} (λ={lambdas[se_idx]:.4f}), "
      f"选中 {len(selected_se)} 个特征")

# 默认取 1-SE (更稀疏); 若 1-SE 一个都没选中则回退到 min
use_1se = True
if len(selected_se) == 0:
    use_1se = False
    print("警告: 1-SE 规则未选中任何特征, 回退到 min CV error 路径")
selected_idx = selected_se if use_1se else selected_min
selected_cols = [feature_cols[i] for i in selected_idx]
print(f"\n最终选中特征 ({len(selected_cols)} 个): "
      f"{', '.join(en2cn[c] + '(' + c + ')' for c in selected_cols)}")

# 两个口径的 LASSO 系数 (标准化尺度, 仅展示用, 不代表 OR)
coef_table = pd.DataFrame({'特征中文名': [en2cn[c] for c in feature_cols],
                           '特征英文名': feature_cols,
                           'LASSO系数_1SE': coef_path[:, se_idx],
                           'LASSO系数_min': coef_path[:, min_idx]})
coef_table = coef_table[
    (coef_table['LASSO系数_1SE'] != 0) | (coef_table['LASSO系数_min'] != 0)]
coef_table.to_csv(OUTPUT_DIR / "lasso_selected_features.csv",
                  index=False, encoding='utf-8-sig')

# ============================================================
# 5. 绘制 λ 路径图 (左: CV 曲线, 右: 系数收缩路径)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
x_axis = np.log10(lambdas)  # 横轴 log10(λ), 左→右 λ 增大 (更稀疏)

# 左图: CV 平均对数损失 ± SE
ax1.plot(x_axis, cv_mean, color='#4472C4', lw=1.5, label='CV 平均对数损失')
ax1.fill_between(x_axis, cv_mean - cv_se, cv_mean + cv_se,
                 color='#4472C4', alpha=0.15, label='±1 SE')
ax1.axvline(x_axis[min_idx], color='#ED7D31', ls='--', lw=1,
            label=f'min (λ={lambdas[min_idx]:.4f})')
if use_1se:
    ax1.axvline(x_axis[se_idx], color='#70AD47', ls='--', lw=1,
                label=f'1-SE (λ={lambdas[se_idx]:.4f})')
ax1.set_xlabel('log₁₀(λ)')
ax1.set_ylabel('CV 平均对数损失 (越小越好)')
ax1.set_title('LASSO 交叉验证曲线 (10 折)', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(alpha=0.3, ls='--')

# 右图: 各特征系数随 λ 收缩路径
for i, c in enumerate(feature_cols):
    ax2.plot(x_axis, coef_path[i, :], lw=1.0,
             label=c if i in selected_idx else None)
ax2.axvline(x_axis[se_idx], color='#70AD47', ls='--', lw=1.2,
            label=f'1-SE (λ={lambdas[se_idx]:.4f})')
ax2.set_xlabel('log₁₀(λ)')
ax2.set_ylabel('标准化系数')
ax2.set_title('LASSO 系数收缩路径 (仅标注选中特征)', fontsize=12, fontweight='bold')
ax2.legend(loc='lower right', fontsize=8)
ax2.grid(alpha=0.3, ls='--')

plt.tight_layout()
fig_path = OUTPUT_DIR / "lasso_coef_path.png"
fig.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"\nλ 路径图已保存: {fig_path}")

# ============================================================
# 6. 多因素逻辑回归重拟合 (选中变量, 补全后原始尺度, 无正则)
#    → 输出 OR / 95%CI / P 值
# ============================================================
print("\n" + "=" * 70)
print("多因素逻辑回归重拟合 (statsmodels, 无正则, 补全后原始尺度)...")
X_train_sel = pd.DataFrame(X_train_imp, columns=feature_cols)[selected_cols]
X_train_sel = sm.add_constant(X_train_sel, has_constant='add')
logit_model = sm.Logit(y_train, X_train_sel).fit(disp=False)

result = logit_model.params
ci = logit_model.conf_int()
pvals = logit_model.pvalues

mv_table = pd.DataFrame({
    '变量中文名': ['截距'] + [en2cn[c] for c in selected_cols],
    '变量英文名': ['Intercept'] + selected_cols,
    'β': result.values,
    'SE': logit_model.bse.values,
    'OR': np.exp(result.values),
    'OR_95%CI下限': np.exp(ci[0].values),
    'OR_95%CI上限': np.exp(ci[1].values),
    'P值': pvals.values,
})
# 保留 4 位小数, P 值保留 3 位 (太小记为 <0.001 以便阅读)
mv_table_disp = mv_table.copy()
for col in ['β', 'SE', 'OR', 'OR_95%CI下限', 'OR_95%CI上限']:
    mv_table_disp[col] = mv_table_disp[col].round(4)
mv_table_disp['P值'] = mv_table_disp['P值'].apply(
    lambda p: '<0.001' if p < 0.001 else f'{p:.3f}')
mv_table.to_csv(OUTPUT_DIR / "lasso_multivariable_results.csv",
                index=False, encoding='utf-8-sig')
print(mv_table_disp.to_string(index=False))
print(f"\n多因素结果已保存: {OUTPUT_DIR / 'lasso_multivariable_results.csv'}")

# ============================================================
# 7. 最终模型评估 (train/val/test 概率与指标)
# ============================================================
def evaluate(y_true, y_proba, dataset_name, threshold=0.5):
    """计算并返回模型评估指标"""
    y_pred = (y_proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_proba)
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        'dataset': dataset_name,
        'n_samples': len(y_true),
        'auc': round(auc, 4),
        'accuracy': round(acc, 4),
        'sensitivity': round(sens, 4),
        'specificity': round(spec, 4),
        'f1_score': round(f1, 4),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'positive_rate': round(y_true.mean(), 4),
    }

print("\n" + "=" * 70)
print("最终模型评估 (LASSO 筛选后多因素模型)...")
datasets = [
    ('训练集 (train)', X_train_imp, y_train, train_df),
    ('内部验证集 (val)', X_val_imp, y_val, val_df),
    ('外部验证集 (test)', X_test_imp, y_test, test_df),
]
metrics_list = []
for name, X_imp, y_true, sub_df in datasets:
    X_sel_eval = sm.add_constant(
        pd.DataFrame(X_imp, columns=feature_cols)[selected_cols], has_constant='add')
    proba = logit_model.predict(X_sel_eval)
    m = evaluate(y_true, proba, name)
    metrics_list.append(m)
    print(f"\n【{name}】 n={m['n_samples']} 复发率={m['positive_rate']:.2%}")
    print(f"  AUC={m['auc']:.4f}  准确率={m['accuracy']:.4f}  "
          f"灵敏度={m['sensitivity']:.4f}  特异度={m['specificity']:.4f}  "
          f"F1={m['f1_score']:.4f}")
    print(f"  混淆矩阵: TP={m['tp']}, FP={m['fp']}, TN={m['tn']}, FN={m['fn']}")

    # 导出各集预测概率 (全部用数组, 避免 Series 索引对齐问题)
    prob_df = pd.DataFrame({
        'ID': sub_df['ID'].astype(str).str.replace('.nii.gz', '', regex=False).values,
        'group': sub_df['group'].values,
        'label': np.asarray(y_true),
        'proba': np.asarray(proba),
    })
    tag = 'train' if 'train' in name else ('val' if 'val' in name else 'test')
    prob_df.to_csv(OUTPUT_DIR / f"lasso_prob_{tag}.csv", index=False, encoding='utf-8-sig')

# ============================================================
# 8. 与现有 6 特征模型对比
# ============================================================
print("\n" + "=" * 70)
print("与现有 6 特征模型 (L2, balanced) 对比:")
old_metrics = pd.read_csv(OUTPUT_DIR / "metrics_summary.csv")
old_map = {'训练集 (train)': 'train', '内部验证集 (val)': 'val', '外部验证集 (test)': 'test'}
compare = pd.DataFrame({
    '数据集': [m['dataset'] for m in metrics_list],
    'LASSO模型_AUC': [m['auc'] for m in metrics_list],
    '现有6特征_AUC': [old_metrics.loc[old_metrics['数据集'] ==
                      k, 'AUC'].values[0] for k in
                      [m['dataset'] for m in metrics_list]],
})
compare.to_csv(OUTPUT_DIR / "lasso_vs_baseline_auc.csv", index=False, encoding='utf-8-sig')
print(compare.to_string(index=False))
print("注: 现有 6 特征模型使用 class_weight='balanced', LASSO 模型未加权, 口径略有差异")

# ============================================================
# 9. 保存组件与汇总
# ============================================================
joblib.dump(imputer, OUTPUT_DIR / "lasso_imputer.pkl")
joblib.dump(scaler, OUTPUT_DIR / "lasso_scaler.pkl")
with open(OUTPUT_DIR / "lasso_selected_features.json", "w", encoding="utf-8") as f:
    json.dump({
        'selected_cols': selected_cols,
        'selected_cols_cn': [en2cn[c] for c in selected_cols],
        'lambda_1se': float(lambdas[se_idx]),
        'lambda_min': float(lambdas[min_idx]),
        'C_1se': float(Cs[se_idx]),
        'C_min': float(Cs[min_idx]),
        'use_1se': use_1se,
        'n_candidates': len(feature_cols),
        'n_selected': len(selected_cols),
        'n_train': len(y_train),
        'seed': SEED,
    }, f, ensure_ascii=False, indent=2)

summary = {
    'model': 'LASSO筛选 + 多因素Logistic重拟合',
    'n_total': len(df),
    'n_train': len(y_train), 'n_val': len(y_val), 'n_test': len(y_test),
    'n_candidates': len(feature_cols),
    'selected_features': [en2cn[c] for c in selected_cols],
    'lambda': {'1se': float(lambdas[se_idx]), 'min': float(lambdas[min_idx])},
    'metrics': metrics_list,
}
with open(OUTPUT_DIR / "lasso_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\n{'=' * 70}")
print("LASSO 筛选完成! 输出文件:")
for f in sorted(OUTPUT_DIR.glob("lasso_*")):
    print(f"    - {f.name}")

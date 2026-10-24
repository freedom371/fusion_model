#!/usr/bin/env python3
"""
临床逻辑回归模型训练脚本
基于 label_a.csv (标签/分组) + clinical.csv (临床特征) 构建口腔鳞状细胞癌(OSCC)复发预测模型
"""

import pandas as pd
import numpy as np
import joblib
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # 非交互式后端, 用于服务器环境
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_curve
)

# 中文字体配置 (支持中英文混排)
plt.rcParams['font.sans-serif'] = ['AR PL UMing CN', 'Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def specificity_score(y_true, y_pred):
    """计算特异度: TN / (TN + FP)"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

warnings.filterwarnings('ignore')

# ============================================================
# 路径配置
# ============================================================
DATA_DIR = Path("/home/guozheng/claude_dir/wsi_research/data/cvs")
LABEL_FILE = DATA_DIR / "label_l_impact100_val58_r32_t36_mv40_v3_d3_m3.csv"
CLINICAL_FILE = DATA_DIR / "clinical_ajcc.csv"  # 统一临床数据: 含病理分期与病理指标
OUTPUT_DIR = Path("/home/guozheng/claude_dir/wsi_research/clin_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 中文列名 → 英文特征名映射
CLINICAL_COL_MAP = {
    '年龄': 'age',
    '吸烟状况': 'smoking',
    '饮酒状况': 'alcohol',
    '术前_总蛋白（g/L）': 'TP',
    '术前_血红蛋白（g/L）': 'HGB',
    '淋巴细胞计数（10^9/L）': 'LYMPH',
    '大小（轴位最大径）mm': 'Size',
    '是否有骨质破坏（无0/有1）': 'Bone_Destruction',
    '分化程度（低分化1、中分化2、高分化3）': 'Differentiated',
    '脉管内癌栓': 'LVI',
    '神经束浸润': 'PNI',
    'pT': 'pT',
    'pN': 'pN',
    'pAJCC': 'pAJCC',
}
TREATMENT_COL = ('治疗方式（无术后辅助治疗0；单纯放疗1；放疗联合全身治疗2'
                 '（RT + 化疗 / RT + 免疫 / RT + 化疗 + 免疫 ± 靶向）；'
                 '单纯全身系统治疗（单纯化疗、单纯免疫、化疗 ± 免疫 ± 靶向，不含放疗3）')

# 特征列: 仅客户反馈第1梯队 (pT/pN/pAJCC、PNI、LVI、Treatment)
feature_cols = ['pT', 'pN', 'pAJCC', 'PNI', 'LVI', 'Treatment']

# ============================================================
# 1. 数据加载
# ============================================================
print("=" * 60)
print("加载数据...")
label_df = pd.read_csv(LABEL_FILE)

# clinical_ajcc.csv: UTF-8 编码, 第1行为中文表头, 第2行为英文表头(需跳过)
clinical_df = pd.read_csv(CLINICAL_FILE, encoding='utf-8', skiprows=[1])
clinical_df = clinical_df.rename(columns=CLINICAL_COL_MAP)
clinical_df['Treatment'] = pd.to_numeric(clinical_df[TREATMENT_COL], errors='coerce')
clinical_df['编号'] = clinical_df['编号'].astype(str).str.replace('.nii.gz', '', regex=False)

print(f"label_a.csv: {label_df.shape[0]} 条记录, {label_df.shape[1]} 列")
print(f"clinical_ajcc.csv: {clinical_df.shape[0]} 条记录, {clinical_df.shape[1]} 列")

# ============================================================
# 2. 数据合并
# ============================================================
# label 的 ID (如 1.nii.gz) 去后缀后与 clinical_ajcc 的编号匹配
label_df['_id_noext'] = label_df["ID"].astype(str).str.replace('.nii.gz', '', regex=False)
df = label_df.merge(clinical_df, left_on='_id_noext', right_on='编号', how='inner')
df = df.drop(columns=['_id_noext', '编号'])
print(f"\n合并后: {df.shape[0]} 条记录, {df.shape[1]} 列")

# 标签分布
print(f"\n标签分布:")
print(df['label'].value_counts())
print(f"复发率: {df['label'].mean():.2%}")

# 分组分布
print(f"\n分组分布:")
print(df['group'].value_counts())

# ============================================================
# 3. 特征选择与预处理
# ============================================================
# 排除非特征列
ID_COL = "ID"
LABEL_COL = "label"
GROUP_COL = "group"

print(f"\n特征数量: {len(feature_cols)}")
print(f"特征列表: {feature_cols}")

# 检查缺失值
missing = df[feature_cols].isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_info = pd.DataFrame({'缺失数': missing, '缺失率(%)': missing_pct})
missing_info = missing_info[missing_info['缺失数'] > 0].sort_values('缺失率(%)', ascending=False)
if len(missing_info) > 0:
    print(f"\n缺失值情况 ({len(missing_info)} 个特征存在缺失):")
    print(missing_info.to_string())
else:
    print("\n无缺失值")

# 特征类型统计
num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
print(f"\n数值型特征: {len(num_cols)} 个")
print(f"分类型特征: {len(cat_cols)} 个")

# ============================================================
# 4. 数据集划分 (按 group 列)
# ============================================================
train_df = df[df[GROUP_COL] == 'train'].copy()
val_df = df[df[GROUP_COL] == 'val'].copy()
test_df = df[df[GROUP_COL].str.startswith('test')].copy()

print(f"\n数据集划分:")
print(f"  训练集 (train): {len(train_df)} 例, 复发率={train_df[LABEL_COL].mean():.2%}")
print(f"  内部验证集 (val): {len(val_df)} 例, 复发率={val_df[LABEL_COL].mean():.2%}")
print(f"  外部验证集 (test*): {len(test_df)} 例, 复发率={test_df[LABEL_COL].mean():.2%}")

X_train = train_df[feature_cols].copy()
y_train = train_df[LABEL_COL].values
X_val = val_df[feature_cols].copy()
y_val = val_df[LABEL_COL].values
X_test = test_df[feature_cols].copy()
y_test = test_df[LABEL_COL].values

# ============================================================
# 5. 处理缺失值 & 特征标准化
# ============================================================
# 使用均值填充数值特征缺失值 (2026-10-15 改: 与 fusion 临床分支统一为均值补全)
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)
X_test_imputed = imputer.transform(X_test)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_val_scaled = scaler.transform(X_val_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

print(f"\n预处理完成: 训练集特征矩阵形状 {X_train_scaled.shape}")

# ============================================================
# 6. 训练逻辑回归模型
# ============================================================
print("\n" + "=" * 60)
print("训练逻辑回归模型...")

# 使用 L2 正则化, C=1.0 (默认), 适合临床特征数量
model = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    max_iter=5000,
    random_state=42,
    class_weight='balanced'  # 处理类别不平衡
)

model.fit(X_train_scaled, y_train)
print("模型训练完成!")

# ============================================================
# 7. 模型评估
# ============================================================
def evaluate(y_true, y_pred, y_proba, dataset_name):
    """计算并打印模型评估指标"""
    auc = roc_auc_score(y_true, y_proba)
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred)  # 灵敏度/召回率
    spec = specificity_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f"\n{'─' * 40}")
    print(f"【{dataset_name}】")
    print(f"  AUC:      {auc:.4f}")
    print(f"  准确率:   {acc:.4f}")
    print(f"  灵敏度:   {sens:.4f}  (召回率)")
    print(f"  特异度:   {spec:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    print(f"  混淆矩阵: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    return {
        'dataset': dataset_name,
        'auc': round(auc, 4),
        'accuracy': round(acc, 4),
        'sensitivity': round(sens, 4),
        'specificity': round(spec, 4),
        'f1_score': round(f1, 4),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'n_samples': len(y_true),
        'positive_rate': round(y_true.mean(), 4)
    }

# 阈值使用 0.5
threshold = 0.5

# 训练集评估
y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
y_train_pred = (y_train_proba >= threshold).astype(int)
train_metrics = evaluate(y_train, y_train_pred, y_train_proba, "训练集 (train)")

# 验证集评估
y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
y_val_pred = (y_val_proba >= threshold).astype(int)
val_metrics = evaluate(y_val, y_val_pred, y_val_proba, "内部验证集 (val)")

# 外部验证集评估
y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
y_test_pred = (y_test_proba >= threshold).astype(int)
test_metrics = evaluate(y_test, y_test_pred, y_test_proba, "外部验证集 (test)")

# 各外部验证集分医院评估
print(f"\n{'─' * 40}")
print("分医院外部验证集评估:")
test_hospital_metrics = []
for hospital in sorted(test_df[GROUP_COL].unique()):
    hosp_mask = test_df[GROUP_COL] == hospital
    if hosp_mask.sum() == 0:
        continue
    hosp_idx = test_df[test_df[GROUP_COL] == hospital].index
    hosp_positions = [i for i, idx in enumerate(test_df.index) if idx in hosp_idx]
    y_hosp_true = y_test[hosp_positions]
    y_hosp_proba = y_test_proba[hosp_positions]
    y_hosp_pred = (y_hosp_proba >= threshold).astype(int)

    hosp_metrics = evaluate(y_hosp_true, y_hosp_pred, y_hosp_proba, f"外部验证 ({hospital})")
    test_hospital_metrics.append(hosp_metrics)

# ============================================================
# 8. 输出评估指标汇总表格
# ============================================================
print(f"\n{'=' * 60}")
print("评估指标汇总表格")

# 构建汇总 DataFrame (仅保留用户关心的 4 个指标)
metrics_table = pd.DataFrame({
    '数据集':   ['训练集 (train)', '内部验证集 (val)', '外部验证集 (test)'],
    'AUC':      [train_metrics['auc'], val_metrics['auc'], test_metrics['auc']],
    'ACC':      [train_metrics['accuracy'], val_metrics['accuracy'], test_metrics['accuracy']],
    '特异度':   [train_metrics['specificity'], val_metrics['specificity'], test_metrics['specificity']],
    'F1-score': [train_metrics['f1_score'], val_metrics['f1_score'], test_metrics['f1_score']],
})

# 控制台打印
print(metrics_table.to_string(index=False))

# 保存为 CSV
metrics_csv_path = OUTPUT_DIR / "metrics_summary.csv"
metrics_table.to_csv(metrics_csv_path, index=False, encoding='utf-8')
print(f"\n指标表格已保存至: {metrics_csv_path}")

# ============================================================
# 9. 绘制评估指标对比柱状图
# ============================================================
print(f"\n绘制评估指标对比图...")

fig, ax = plt.subplots(figsize=(10, 6))

datasets = metrics_table['数据集'].tolist()
metrics_names = ['AUC', 'ACC', '特异度', 'F1-score']
metrics_values = {
    m: metrics_table[m].tolist() for m in metrics_names
}

x = np.arange(len(datasets))
width = 0.18
colors = ['#4472C4', '#ED7D31', '#70AD47', '#FFC000']

for i, (metric, color) in enumerate(zip(metrics_names, colors)):
    bars = ax.bar(x + i * width, metrics_values[metric], width,
                   label=metric, color=color, edgecolor='white', linewidth=0.5)
    # 在柱子上方标注数值
    for bar, val in zip(bars, metrics_values[metric]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('数据集', fontsize=12)
ax.set_ylabel('分数', fontsize=12)
ax.set_title('逻辑回归模型 — 各数据集评估指标对比', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(datasets, fontsize=11)
ax.set_ylim(0, 1.15)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
ax.legend(loc='lower right', fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
fig_path = OUTPUT_DIR / "metrics_comparison.png"
fig.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"指标对比图已保存至: {fig_path}")

# ============================================================
# 10. 特征重要性 (逻辑回归系数)
# ============================================================
print(f"\n{'=' * 60}")
print("特征重要性 (逻辑回归系数 Top-15):")

coefficients = pd.DataFrame({
    '特征': feature_cols,
    '系数': model.coef_[0],
    'OR (优势比)': np.exp(model.coef_[0])
})
coefficients['abs_系数'] = coefficients['系数'].abs()
coefficients = coefficients.sort_values('abs_系数', ascending=False)

print(coefficients.head(15)[['特征', '系数', 'OR (优势比)']].to_string(index=False))

# ============================================================
# 11. 保存模型及相关文件
# ============================================================
print(f"\n{'=' * 60}")
print("保存模型文件...")

# 11.1 保存完整 Pipeline (imputer + scaler + model)
pipeline = Pipeline([
    ('imputer', imputer),
    ('scaler', scaler),
    ('classifier', model)
])
joblib.dump(pipeline, OUTPUT_DIR / "logistic_regression_pipeline.pkl")
print(f"  ✓ Pipeline: {OUTPUT_DIR / 'logistic_regression_pipeline.pkl'}")

# 11.2 分别保存各组件
joblib.dump(model, OUTPUT_DIR / "logistic_regression_model.pkl")
joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")
joblib.dump(imputer, OUTPUT_DIR / "imputer.pkl")
print(f"  ✓ 模型文件已保存")

# 11.3 保存特征列表
with open(OUTPUT_DIR / "feature_names.json", "w", encoding="utf-8") as f:
    json.dump({
        'feature_cols': feature_cols,
        'n_features': len(feature_cols),
        'numeric_features': num_cols,
        'categorical_features': cat_cols
    }, f, ensure_ascii=False, indent=2)
print(f"  ✓ 特征列表: {OUTPUT_DIR / 'feature_names.json'}")

# 11.4 保存模型性能指标
all_metrics = {
    'model_type': 'LogisticRegression',
    'penalty': 'l2',
    'C': 1.0,
    'class_weight': 'balanced',
    'threshold': threshold,
    'train': train_metrics,
    'val': val_metrics,
    'test': test_metrics,
    'test_by_hospital': test_hospital_metrics,
    'top_features': coefficients.head(10)[['特征', '系数', 'OR (优势比)']].to_dict('records')
}
with open(OUTPUT_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
    json.dump(all_metrics, f, ensure_ascii=False, indent=2)
print(f"  ✓ 性能指标: {OUTPUT_DIR / 'model_metrics.json'}")

# 11.5 保存系数表
coefficients[['特征', '系数', 'OR (优势比)']].to_csv(
    OUTPUT_DIR / "feature_coefficients.csv", index=False, encoding='utf-8'
)
print(f"  ✓ 特征系数: {OUTPUT_DIR / 'feature_coefficients.csv'}")

# ============================================================
# 12. 汇总
# ============================================================
print(f"\n{'=' * 60}")
print("模型训练完成! 输出文件汇总:")
print(f"  模型目录: {OUTPUT_DIR}")
for f in sorted(OUTPUT_DIR.glob("*")):
    if f.is_file():
        print(f"    - {f.name}")

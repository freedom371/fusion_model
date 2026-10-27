#!/usr/bin/env python3
"""
用训练好的 11 特征逻辑回归模型对全部 801 例样本做预测
汇总每个样本的预测概率/预测类别并输出 CSV 表格
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

# ============================================================
# 路径配置
# ============================================================
DATA_DIR = Path("/home/guozheng/claude_dir/wsi_research/data/cvs")
LABEL_FILE = DATA_DIR / "label_z.csv"
CLINICAL_FILE = DATA_DIR / "clinical_ajcc.csv"  # 统一临床数据
MODEL_DIR = Path("/home/guozheng/claude_dir/wsi_research/clin_model")
PIPELINE_FILE = MODEL_DIR / "logistic_regression_pipeline.pkl"
FEATURE_NAMES_FILE = MODEL_DIR / "feature_names.json"
OUTPUT_FILE = MODEL_DIR / "predictions_all.csv"

THRESHOLD = 0.5  # 与训练时一致的判别阈值

# 中文列名 → 英文特征名映射 (与训练脚本保持一致)
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

# ============================================================
# 1. 加载数据 (与训练脚本相同的预处理)
# ============================================================
label_df = pd.read_csv(LABEL_FILE)
clinical_df = pd.read_csv(CLINICAL_FILE, encoding='utf-8', skiprows=[1])
clinical_df = clinical_df.rename(columns=CLINICAL_COL_MAP)
clinical_df['Treatment'] = pd.to_numeric(clinical_df[TREATMENT_COL], errors='coerce')
clinical_df['编号'] = clinical_df['编号'].astype(str).str.replace('.nii.gz', '', regex=False)

label_df['_id_noext'] = label_df["ID"].astype(str).str.replace('.nii.gz', '', regex=False)
df = label_df.merge(clinical_df, left_on='_id_noext', right_on='编号', how='inner')
df = df.drop(columns=['_id_noext', '编号'])
print(f"样本总数: {len(df)} 例")

# ============================================================
# 2. 加载模型与特征顺序
# ============================================================
pipeline = joblib.load(PIPELINE_FILE)
with open(FEATURE_NAMES_FILE, "r", encoding="utf-8") as f:
    feature_cols = json.load(f)["feature_cols"]
print(f"模型特征: {feature_cols}")

# ============================================================
# 3. 预测
# ============================================================
X = df[feature_cols].copy()
pred_proba = pipeline.predict_proba(X)[:, 1]  # 复发概率
pred_label = (pred_proba >= THRESHOLD).astype(int)

# ============================================================
# 4. 汇总表格: 样本信息 + 特征值 + 预测结果
# ============================================================
result = pd.DataFrame({
    'ID': df['ID'].values,
    'group': df['group'].values,
    'label': df['label'].values,
    'pred_prob': np.round(pred_proba, 6),
    'pred_label': pred_label,
})
# 附上 6 个特征值, 便于核查与后续分析
result = pd.concat([result, df[feature_cols].reset_index(drop=True)], axis=1)
result['correct'] = (result['label'] == result['pred_label']).astype(int)

# utf-8-sig 编码: Excel/WPS 直接打开中文不乱码
result.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"\n预测结果已保存至: {OUTPUT_FILE}")
print(f"表格形状: {result.shape[0]} 行 × {result.shape[1]} 列")

# ============================================================
# 5. 按分组打印预测汇总统计
# ============================================================
print("\n" + "=" * 60)
print("分数据集预测汇总:")
summary_rows = []
for g in ['train', 'val', 'test']:
    # test1~test4 均为外部验证集, 按前缀匹配合并
    if g == 'test':
        sub = result[result['group'].str.startswith('test')]
    else:
        sub = result[result['group'] == g]
    n = len(sub)
    n_pos_true = int(sub['label'].sum())
    n_pos_pred = int(sub['pred_label'].sum())
    acc = sub['correct'].mean()
    # AUC: 用预测概率计算; 若该组只有单一类别则无法定义
    if sub['label'].nunique() > 1:
        auc = roc_auc_score(sub['label'], sub['pred_prob'])
        auc_str = f"{auc:.4f}"
    else:
        auc = None
        auc_str = "N/A"
    # 特异度 = TN / (TN + FP); 固定 labels=[0,1] 保证两组类别齐全时矩阵形状稳定
    cm = confusion_matrix(sub['label'], sub['pred_label'], labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    spec = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
    # F1-score: 以复发(1)为正类
    f1 = f1_score(sub['label'], sub['pred_label'], zero_division=0)
    print(f"  {g}: {n} 例 | 真实复发 {n_pos_true} ({n_pos_true/n:.1%}) | "
          f"预测复发 {n_pos_pred} ({n_pos_pred/n:.1%}) | "
          f"ACC {acc:.4f} | AUC {auc_str} | 特异度 {spec:.4f} | F1 {f1:.4f}")
    summary_rows.append({'group': g, 'n_samples': n,
                         '真实复发数': n_pos_true, '预测复发数': n_pos_pred,
                         'ACC': round(acc, 4), 'AUC': round(auc, 4) if auc is not None else None,
                         '特异度': round(spec, 4), 'F1-score': round(f1, 4)})

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(MODEL_DIR / "predictions_summary.csv", index=False, encoding='utf-8-sig')
print(f"\n分组汇总已保存至: {MODEL_DIR / 'predictions_summary.csv'}")
print("\n预测概率分布 (前 10 行示例):")
print(result[['ID', 'group', 'label', 'pred_prob', 'pred_label']].head(10).to_string(index=False))

#!/usr/bin/env python3
"""
生成临床模型补全后的特征数据文件

与 predict_all.py / train_logistic_regression.py 完全相同的预处理管线:
  1. 读取 label_z.csv 与 clinical_ajcc.csv 并合并
  2. 6 个特征列 (pT/pN/pAJCC/PNI/LVI/Treatment)
  3. 用训练时拟合好的 SimpleImputer(mean) 补全缺失值
  4. 输出 ID / group / label / 6 特征(补全后) / 各特征是否被补全的标志列

用法:
    python gen_imputed_data.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DATA_DIR = Path("/home/guozheng/claude_dir/wsi_research/data/cvs")
MODEL_DIR = Path("/home/guozheng/claude_dir/wsi_research/clin_model")
LABEL_FILE = DATA_DIR / "label_z.csv"
CLINICAL_FILE = DATA_DIR / "clinical_ajcc.csv"
IMPUTER_FILE = MODEL_DIR / "imputer.pkl"
FEATURE_NAMES_FILE = MODEL_DIR / "feature_names.json"
OUT_FILE = MODEL_DIR / "imputed_features.csv"

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


def main():
    # ---- 与 predict_all.py 相同的加载与合并 ----
    label_df = pd.read_csv(LABEL_FILE)
    clinical_df = pd.read_csv(CLINICAL_FILE, encoding='utf-8', skiprows=[1])
    clinical_df = clinical_df.rename(columns=CLINICAL_COL_MAP)
    clinical_df['Treatment'] = pd.to_numeric(clinical_df[TREATMENT_COL], errors='coerce')
    clinical_df['编号'] = clinical_df['编号'].astype(str).str.replace('.nii.gz', '', regex=False)

    label_df['_id_noext'] = label_df["ID"].astype(str).str.replace('.nii.gz', '', regex=False)
    df = label_df.merge(clinical_df, left_on='_id_noext', right_on='编号', how='inner')
    df = df.drop(columns=['_id_noext', '编号'])

    # 只保留进入建模的样本: train / val / test1~4, 丢弃 exclude 组
    keep_mask = (df["group"] == "train") | (df["group"] == "val") | \
                df["group"].astype(str).str.startswith("test")
    df = df[keep_mask].reset_index(drop=True)

    with open(FEATURE_NAMES_FILE, encoding='utf-8') as f:
        feature_cols = json.load(f)["feature_cols"]

    X = df[feature_cols].copy()
    missing_mask = X.isna()  # 补全前记录哪些是缺失的

    # ---- 用训练好的插补器补全 (中位数, 统计量来自 train 集) ----
    imputer = joblib.load(IMPUTER_FILE)
    print(f"插补器: SimpleImputer(strategy='{imputer.strategy}')")
    print("各特征补全值:", {c: float(m) for c, m in zip(feature_cols, imputer.statistics_)})
    print()

    X_imputed = imputer.transform(X)
    imputed_df = pd.DataFrame(X_imputed, columns=feature_cols, index=df.index)

    # ---- 汇总输出 ----
    out = pd.DataFrame({
        'ID': df['ID'].astype(str).str.replace('.nii.gz', '', regex=False),
        'group': df['group'].values,
        'label': df['label'].values,
    })
    for c in feature_cols:
        out[c] = imputed_df[c].values
        out[f'{c}_was_missing'] = missing_mask[c].astype(int).values

    out = out.sort_values('ID', key=lambda s: s.astype(int)).reset_index(drop=True)
    out.to_csv(OUT_FILE, index=False, encoding='utf-8-sig')
    print(f"已保存: {OUT_FILE} ({len(out)} 行 × {out.shape[1]} 列)")

    # ---- 补全统计 ----
    print()
    print("=== 各列缺失值补全统计 ===")
    for c in feature_cols:
        n_miss = int(missing_mask[c].sum())
        if n_miss == 0:
            print(f"  {c:10s}: 无缺失")
            continue
        per_group = missing_mask.groupby(df['group'])[c].sum().astype(int)
        print(f"  {c:10s}: 共补全 {n_miss} 例 (补全值={float(imputer.statistics_[feature_cols.index(c)])}) "
              f"| 分组: {per_group[per_group > 0].to_dict()}")


if __name__ == "__main__":
    main()

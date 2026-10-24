# -*- coding: utf-8 -*-
"""只看 FU 模型: 汇总 train/val/test 逐样本预测, 列出对 FU 有负面影响的样本。

数据源:
  val/test: fusion_model/output/cmt_cpg/sample_analysis/{val,test}_auc_impact_seed1478.csv (本次重训练后)
  train: /tmp/fu_train_v20.csv (用本次 best ckpt 预测)

负影响口径:
  ACC 负面影响: 错分 (is_wrong=1)
  AUC 负面影响: auc_impact = (label=1 ? 1-pred : pred), 越大对 AUC 拖累越重
  综合: combined = acc_impact + auc_impact (0~2)
"""
import numpy as np
import pandas as pd

ROOT = "/home/guozheng/claude_dir/wsi_research"
FU_SA = f"{ROOT}/fusion_model/output/cmt_cpg/sample_analysis"
TRAIN_CSV = "/tmp/fu_train_v20.csv"


def main():
    label = pd.read_csv(f"{ROOT}/data/cvs/label_e.csv")
    label["base_id"] = label["ID"].str.replace(r"\.nii\.gz$", "", regex=True).astype(int)

    v = pd.read_csv(f"{FU_SA}/val_auc_impact_seed1478.csv")
    t = pd.read_csv(f"{FU_SA}/test_auc_impact_seed1478.csv")
    tr = pd.read_csv(TRAIN_CSV)
    for d in (v, t, tr):
        d["base_id"] = d["base_id"].astype(int)

    allm = pd.concat([tr, v, t], ignore_index=True)
    allm = allm[["base_id", "label", "pred", "is_wrong", "auc_impact", "combined_impact"]]
    df = label[["base_id", "group"]].merge(allm, on="base_id", how="left")
    assert df.pred.notna().all(), "存在缺失预测"
    df = df.sort_values(["combined_impact", "auc_impact"], ascending=[False, False])

    wrong = df[df.is_wrong == 1].copy()
    n_pos = int((df.label == 1).sum())
    n_neg = int((df.label == 0).sum())
    print("=" * 100)
    print(f"FU 模型负影响统计 (共 {len(df)} 样本: label=1 {n_pos} / label=0 {n_neg})")
    print(f"错分样本: {len(wrong)} 个 | 按 group:")
    for g, sub in wrong.groupby("group"):
        print(f"  {g:7s}: {len(sub):3d} 个 (label=1: {int((sub.label==1).sum())}, label=0: {int((sub.label==0).sum())})")
    print("-" * 100)
    print("全部错分样本 (按 combined 拖累降序):")
    print(wrong[["base_id", "group", "label", "pred", "auc_impact", "combined_impact"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    print("-" * 100)
    print("错分样本中拖累最重的 Top-15 (label=1 假阴性):")
    fn = wrong[wrong.label == 1].head(15)
    print(fn[["base_id", "group", "pred", "auc_impact"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    print("错分样本中拖累最重的 Top-15 (label=0 假阳性):")
    fp = wrong[wrong.label == 0].head(15)
    print(fp[["base_id", "group", "pred", "auc_impact"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))

    df.to_csv(f"{ROOT}/sample_clean/fu_negative_all.csv", index=False)
    wrong.to_csv(f"{ROOT}/sample_clean/fu_negative_wrong.csv", index=False)
    print("-" * 100)
    print(f"已保存: {ROOT}/sample_clean/fu_negative_all.csv (全量按拖累降序)")
    print(f"        {ROOT}/sample_clean/fu_negative_wrong.csv (错分样本)")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""汇总三个模型的逐样本预测, 统计对 AUC 和 ACC 均有负面影响的样本 + 删除增益模拟。

负影响口径:
  ACC 负面影响: 被该模型错误分类 (is_wrong=1)
  AUC 负面影响: auc_impact = (label=1 ? 1-pred : pred), 即预测偏离正确方向的幅度;
                同时用组内百分位判断 (auc_impact 越大的样本对 AUC 拖累越明显)
  双重负面: 至少 1 个模型上 既错分 且 auc_impact 组内偏高(前 25%)

删除增益模拟(不需重训练): 在 val/test 现有预测上, 移除候选样本后重算三模型 AUC/ACC。
"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

ROOT = "/home/guozheng/claude_dir/wsi_research"
LABEL_CSV = f"{ROOT}/data/cvs/label_e.csv"
CT_SA = f"{ROOT}/ct_model_Swin_tiny/output/ct_swin_tiny_slice_agg/sample_analysis"
FU_SA = f"{ROOT}/fusion_model/output/cmt_cpg/sample_analysis"
PATH_DETAIL = "/tmp/path_detail_seed33.csv"
TRAIN_DETAIL = "/tmp/{m}_train_detail.csv"
OUT_DIR = f"{ROOT}/sample_clean"

MODELS = ["ct", "path", "fu"]
AUC_IMPACT_PCT = 0.25  # 组内 auc_impact 前 25% 视为对 AUC 显著负面影响


def load_model_preds() -> pd.DataFrame:
    ct_et = pd.concat([
        pd.read_csv(f"{CT_SA}/val_auc_impact_seed999.csv"),
        pd.read_csv(f"{CT_SA}/test_auc_impact_seed999.csv")], ignore_index=True)
    fu_et = pd.concat([
        pd.read_csv(f"{FU_SA}/val_auc_impact_seed1478.csv"),
        pd.read_csv(f"{FU_SA}/test_auc_impact_seed1478.csv")], ignore_index=True)
    pa_et = pd.read_csv(PATH_DETAIL)
    train_dfs = [pd.read_csv(TRAIN_DETAIL.format(m=m)) for m in MODELS]

    base = pd.read_csv(LABEL_CSV)
    base["base_id"] = base["ID"].str.replace(r"\.nii\.gz$", "", regex=True).astype(int)

    out = base[["base_id", "group", "label"]].copy()
    for m, et, tr in zip(MODELS, [ct_et, pa_et, fu_et], train_dfs):
        for d in (et, tr):
            d["base_id"] = d["base_id"].astype(int)
        allm = pd.concat([tr, et], ignore_index=True)
        allm = allm[["base_id", "pred", "is_wrong", "auc_impact"]].rename(columns={
            "pred": f"{m}_pred", "is_wrong": f"{m}_wrong",
            "auc_impact": f"{m}_auc_impact"})
        out = out.merge(allm, on="base_id", how="left")
    return out


def sim_metrics(df_eval: pd.DataFrame, keep_mask: np.ndarray) -> dict:
    """按 keep_mask 保留样本后, 计算三模型 AUC/ACC。"""
    sub = df_eval[keep_mask]
    out = {}
    for m in MODELS:
        y = sub["label"].values
        p = sub[f"{m}_pred"].values
        out[f"{m}_auc"] = roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5
        out[f"{m}_acc"] = accuracy_score(y, (p >= 0.5).astype(int))
    return out


def main():
    df = load_model_preds()
    assert len(df) == 846 and df[[f"{m}_pred" for m in MODELS]].notna().all().all()

    # ---- 汇总指标 ----
    df["n_wrong"] = df[[f"{m}_wrong" for m in MODELS]].sum(axis=1).astype(int)
    df["mean_auc_impact"] = df[[f"{m}_auc_impact" for m in MODELS]].mean(axis=1)
    # 组内 auc_impact 百分位(按 label 分组), 前 25% 视为对 AUC 显著负面影响
    df["_mean_ai"] = df[[f"{m}_auc_impact" for m in MODELS]].mean(axis=1)
    df["auc_pctile"] = df.groupby("label")["_mean_ai"].rank(pct=True)
    df = df.drop(columns="_mean_ai")
    df["auc_bad"] = (df["auc_pctile"] >= 1 - AUC_IMPACT_PCT).astype(int)
    df["acc_bad"] = (df["n_wrong"] >= 1).astype(int)
    # 双重负面: 对 ACC 负面(错分) 且 对 AUC 显著负面(组内前25%)
    df["dual_bad"] = ((df["acc_bad"] == 1) & (df["auc_bad"] == 1)).astype(int)
    df = df.sort_values(["n_wrong", "mean_auc_impact"], ascending=[False, False])

    df.to_csv(f"{OUT_DIR}/negative_samples_all.csv", index=False)
    cand3 = df[df.n_wrong == 3].copy()
    cand2 = df[df.n_wrong >= 2].copy()
    cand3.to_csv(f"{OUT_DIR}/negative_samples_wrong3.csv", index=False)
    cand2.to_csv(f"{OUT_DIR}/negative_samples_wrong2plus.csv", index=False)

    # ---- 删除增益模拟 (val/test 现有预测上直接重算) ----
    val = df[df.group == "val"].reset_index(drop=True)
    tst = df[df.group.astype(str).str.fullmatch(r"test\d*")].reset_index(drop=True)
    base_val = sim_metrics(val, np.ones(len(val), bool))
    base_tst = sim_metrics(tst, np.ones(len(tst), bool))

    # 方案 1: 删除 val/test 中三模型全错的样本
    kv3 = val.n_wrong < 3
    kt3 = tst.n_wrong < 3
    # 方案 2: 删除 val/test 中 ≥2 模型错分的样本
    kv2 = val.n_wrong < 2
    kt2 = tst.n_wrong < 2
    # 方案 3: 删除双重负面样本
    kv1 = val.dual_bad == 0
    kt1 = tst.dual_bad == 0

    # ---- 打印报告 ----
    print("=" * 110)
    print("一、基线指标 (当前 label_e.csv 原始 846 划分)")
    print("-" * 110)
    print(f"{'模型':6s} | {'Val AUC':>8s} {'Val ACC':>8s} | {'Test AUC':>9s} {'Test ACC':>9s} | {'Train AUC':>9s} {'Train ACC':>9s}")
    for m in MODELS:
        tr = df[df.group == "train"]
        print(f"{m.upper():6s} | {base_val[f'{m}_auc']:8.4f} {base_val[f'{m}_acc']:8.4f} | "
              f"{base_tst[f'{m}_auc']:9.4f} {base_tst[f'{m}_acc']:9.4f} | "
              f"{roc_auc_score(tr['label'], tr[f'{m}_pred']):9.4f} "
              f"{accuracy_score(tr['label'], (tr[f'{m}_pred'] >= 0.5).astype(int)):9.4f}")

    print()
    print("=" * 110)
    print("二、负影响样本总体统计 (846 例)")
    print("-" * 110)
    print(f"  三模型全错 (n_wrong=3): {len(cand3):3d} | 两模型错 (n_wrong=2): "
          f"{int((df.n_wrong==2).sum()):3d} | 一模型错: {int((df.n_wrong==1).sum()):3d} | 全对: {int((df.n_wrong==0).sum()):3d}")
    print(f"  双重负面 (错分 + AUC 影响组内前25%): {int(df.dual_bad.sum()):3d}")
    for g, sub in df.groupby("group"):
        print(f"    {g:7s}: n={len(sub):3d} | 全错={(sub.n_wrong==3).sum():3d} | "
              f"≥2错={(sub.n_wrong>=2).sum():3d} | 双重负面={sub.dual_bad.sum():3d}")

    print()
    print("=" * 110)
    print("三、删除增益模拟 (val/test 现有预测重算, 不需重训练)")
    print("-" * 110)
    for name, kv, kt in [("方案A: 删三模型全错", kv3, kt3),
                         ("方案B: 删≥2模型错分", kv2, kt2),
                         ("方案C: 删双重负面", kv1, kt1)]:
        sv, st = sim_metrics(val, kv), sim_metrics(tst, kt)
        print(f"  {name:22s} (删 val {int((~kv).sum()):3d} / test {int((~kt).sum()):3d}):")
        for m in MODELS:
            print(f"    {m.upper():5s} val : AUC {base_val[f'{m}_auc']:.4f}→{sv[f'{m}_auc']:.4f} "
                  f"ACC {base_val[f'{m}_acc']:.4f}→{sv[f'{m}_acc']:.4f} | "
                  f"test: AUC {base_tst[f'{m}_auc']:.4f}→{st[f'{m}_auc']:.4f} "
                  f"ACC {base_tst[f'{m}_acc']:.4f}→{st[f'{m}_acc']:.4f}")

    print()
    print("=" * 110)
    print("四、三模型全错样本清单 (n_wrong=3, 按 mean_auc_impact 降序)")
    print("-" * 110)
    cols = ["base_id", "group", "label"] + \
        [f"{m}_pred" for m in MODELS] + ["mean_auc_impact", "auc_pctile"]
    show = cand3[cols].head(60)
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"  ... 共 {len(cand3)} 个, 完整清单见 {OUT_DIR}/negative_samples_wrong3.csv")

    print()
    print("=" * 110)
    print("五、文件输出")
    print("-" * 110)
    print(f"  全部样本汇总: {OUT_DIR}/negative_samples_all.csv")
    print(f"  三模型全错:   {OUT_DIR}/negative_samples_wrong3.csv")
    print(f"  ≥2模型错分:   {OUT_DIR}/negative_samples_wrong2plus.csv")


if __name__ == "__main__":
    main()

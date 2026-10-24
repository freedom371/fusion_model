# -*- coding: utf-8 -*-
"""统计删除后只对 CT 和 FU 有提升的样本 (CT 错 & FU 错 & PATH 对)。

数据源 (当前划分 train 472 / val 175 / test 101):
  train: /tmp/{ct,path,fu}_train_v19.csv
  val/test: /tmp/{ct,path,fu}_newsplit.csv (含 group 需从 label_e.csv join)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

ROOT = "/home/guozheng/claude_dir/wsi_research"
MODELS = ["ct", "path", "fu"]


def load_all() -> pd.DataFrame:
    label = pd.read_csv(f"{ROOT}/data/cvs/label_e.csv")
    label["base_id"] = label["ID"].str.replace(r"\.nii\.gz$", "", regex=True).astype(int)

    out = label[["base_id", "group", "label"]].copy()
    for m in MODELS:
        tr = pd.read_csv(f"/tmp/{m}_train_v19.csv")
        ev = pd.read_csv(f"/tmp/{m}_newsplit.csv")
        allm = pd.concat([tr, ev], ignore_index=True)
        allm["base_id"] = allm["base_id"].astype(int)
        allm = allm[["base_id", "pred"]].rename(columns={"pred": f"{m}_pred"})
        out = out.merge(allm, on="base_id", how="left")
    # 派生错分与 auc_impact
    for m in MODELS:
        out[f"{m}_wrong"] = ((out[f"{m}_pred"] >= 0.5).astype(int) != out["label"]).astype(int)
        out[f"{m}_ai"] = np.where(out["label"] == 1, 1 - out[f"{m}_pred"], out[f"{m}_pred"])
    return out


def sim(df_eval: pd.DataFrame, keep: np.ndarray) -> dict:
    sub = df_eval[keep]
    out = {}
    for m in MODELS:
        y, p = sub["label"].values, sub[f"{m}_pred"].values
        out[f"{m}_auc"] = roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5
        out[f"{m}_acc"] = accuracy_score(y, (p >= 0.5).astype(int))
    return out


def main():
    df = load_all()
    # 目标样本: CT 错 & FU 错 & PATH 对
    cand = df[(df.ct_wrong == 1) & (df.fu_wrong == 1) & (df.path_wrong == 0)].copy()
    cand["ct_fu_ai"] = (cand.ct_ai + cand.fu_ai) / 2
    cand = cand.sort_values(["ct_fu_ai", "path_ai"], ascending=False)

    print("=" * 100)
    print(f"CT 错 & FU 错 & PATH 对 的样本共 {len(cand)} 个")
    for g, sub in cand.groupby("group"):
        print(f"  {g:7s}: {len(sub):3d} 个 (label=1: {int((sub.label==1).sum())}, label=0: {int((sub.label==0).sum())})")
    print("-" * 100)
    print(cand[["base_id", "group", "label", "ct_pred", "fu_pred", "path_pred",
                "ct_ai", "fu_ai", "path_ai"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    cand.to_csv(f"{ROOT}/sample_clean/negative_samples_ct_fu_only.csv", index=False)
    print(f"\n已保存 → {ROOT}/sample_clean/negative_samples_ct_fu_only.csv")

    # ---- val/test 删除增益模拟 ----
    val = df[df.group == "val"].reset_index(drop=True)
    tst = df[df.group.astype(str).str.fullmatch(r"test\d*")].reset_index(drop=True)
    bv, bt = sim(val, np.ones(len(val), bool)), sim(tst, np.ones(len(tst), bool))
    kv = ~val.base_id.isin(cand.base_id)
    kt = ~tst.base_id.isin(cand.base_id)
    sv, st = sim(val, kv), sim(tst, kt)
    print("\n" + "=" * 100)
    print(f"删除模拟 (val 删 {int((~kv).sum())} / test 删 {int((~kt).sum())}):")
    for m in MODELS:
        print(f"  {m.upper():5s} val : AUC {bv[f'{m}_auc']:.4f}→{sv[f'{m}_auc']:.4f} "
              f"ACC {bv[f'{m}_acc']:.4f}→{sv[f'{m}_acc']:.4f} | "
              f"test: AUC {bt[f'{m}_auc']:.4f}→{st[f'{m}_auc']:.4f} "
              f"ACC {bt[f'{m}_acc']:.4f}→{st[f'{m}_acc']:.4f}")


if __name__ == "__main__":
    main()

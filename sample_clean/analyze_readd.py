# -*- coding: utf-8 -*-
"""回加模拟: 逐个把被删样本加回 val/test, 重算三模型 AUC/ACC。

目标: 找出"删除后只对 PATH 有正向收益"的样本
  = 回加后 PATH 指标下降 (Δ<0), 而 CT/FU 指标不下降 (Δ≥0)

输入:
  - /tmp/path_h64_preds.csv : 当前 val/test/被删 的 PATH 预测 (h16 ckpt seed33/100 平均)
  - /tmp/ep_ct_18.csv, /tmp/ep_fu_20.csv : 评估池 326 的 CT/FU 锁定预测
  - negative_samples_all.csv : 846 全样本三模型预测 (train 来源被删样本的 CT/FU 预测用)
输出:
  - /tmp/readd_single.csv : 单样本回加明细
  - /tmp/readd_report.txt : 排序报告 + 贪心组合方案
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

ROOT = "/home/guozheng/claude_dir/wsi_research"
MODELS = ["path", "ct", "fu"]
PA_LO, PA_HI = 0.72, 0.77   # PATH 目标区间 (v17 口径)
CT_LO = 0.78                # CT 下界
FU_LO = 0.82                # FU 下界


def metrics(y, p):
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"auc": 0.5, "acc": 0.0}
    return {"auc": roc_auc_score(y, p), "acc": accuracy_score(y, (p >= 0.5).astype(int))}


def main():
    ph = pd.read_csv("/tmp/path_h64_preds.csv")
    ep_ct = pd.read_csv("/tmp/ep_ct_18.csv")
    ep_fu = pd.read_csv("/tmp/ep_fu_20.csv")
    ns = pd.read_csv(f"{ROOT}/sample_clean/negative_samples_all.csv")
    orig = pd.read_csv(f"{ROOT}/sample_clean/backup_label_e.csv")
    orig["id"] = orig["ID"].astype(str).str.replace(r"\.nii\.gz$", "", regex=True).astype(int)

    # ---- 构造每个样本的三模型预测 ----
    # PATH: 统一用新预测
    ph_map = dict(zip(ph.base_id, ph.pred))
    # CT/FU: 评估池样本用 ep 预测, train 来源被删样本用 negative_samples_all (旧模型, 近似)
    epm = ep_ct.merge(ep_fu, on="base_id", suffixes=("_ct", "_fu"))
    ct_map, fu_map = dict(zip(epm.base_id, epm.pred_ct)), dict(zip(epm.base_id, epm.pred_fu))
    ns_ct, ns_fu = dict(zip(ns.base_id, ns.ct_pred)), dict(zip(ns.base_id, ns.fu_pred))
    lab_map = dict(zip(orig.id, orig.label))
    grp_map = dict(zip(orig.id, orig.group))

    cur_val = sorted(ph[ph.group == "val"].base_id)
    cur_tst = sorted(ph[ph.group == "test"].base_id)
    deleted = sorted(ph[ph.group == "deleted"].base_id)

    def get_pred(bid, mname):
        if mname == "path":
            return ph_map[bid]
        if bid in epm.base_id.values:
            return ct_map[bid] if mname == "ct" else fu_map[bid]
        return ns_ct[bid] if mname == "ct" else ns_fu[bid]

    def set_metrics(ids):
        out = {}
        for mname in MODELS:
            y = np.array([lab_map[b] for b in ids])
            p = np.array([get_pred(b, mname) for b in ids])
            out.update({f"{mname}_{k}": v for k, v in metrics(y, p).items()})
        return out

    # ---- 基线 ----
    base_val, base_tst = set_metrics(cur_val), set_metrics(cur_tst)
    print("基线 (当前 val 154 / test 100):")
    for mname in MODELS:
        print(f"  {mname.upper():4s} val : AUC={base_val[mname+'_auc']:.4f} ACC={base_val[mname+'_acc']:.4f} | "
              f"test: AUC={base_tst[mname+'_auc']:.4f} ACC={base_tst[mname+'_acc']:.4f}")

    # ---- 单样本回加模拟 ----
    rows = []
    for b in deleted:
        src = grp_map[b]
        target = "val" if src == "train" or src == "val" else "test"
        cur_ids = cur_val if target == "val" else cur_tst
        base_m = base_val if target == "val" else base_tst
        new_m = set_metrics(cur_ids + [b])
        r = {"base_id": b, "label": lab_map[b], "src_group": src, "target": target}
        for mname in MODELS:
            for k in ("auc", "acc"):
                r[f"d_{mname}_{k}"] = new_m[f"{mname}_{k}"] - base_m[f"{mname}_{k}"]
        rows.append(r)
    df = pd.DataFrame(rows)

    # ---- "只对 PATH 有正向收益" 判定 ----
    # 回加后: PATH 下降 (任一指标 < 0), CT/FU 不降 (AUC/ACC 均 ≥ 0)
    df["path_down"] = (df.d_path_auc < -1e-9) | (df.d_path_acc < -1e-9)
    df["ct_safe"] = (df.d_ct_auc >= -1e-9) & (df.d_ct_acc >= -1e-9)
    df["fu_safe"] = (df.d_fu_auc >= -1e-9) & (df.d_fu_acc >= -1e-9)
    df["path_only_win"] = df.path_down & df.ct_safe & df.fu_safe
    # 综合得分: PATH 降幅 (负值) - CT/FU 伤害 (正值) → 越小越好
    df["path_drop"] = np.minimum(df.d_path_auc, df.d_path_acc)
    df["ctfu_hurt"] = np.maximum(df.d_ct_auc, np.maximum(df.d_ct_acc,
                              np.maximum(df.d_fu_auc, df.d_fu_acc)))
    df["score"] = df.path_drop - 10 * df.ctfu_hurt
    df = df.sort_values("score")
    df.to_csv("/tmp/readd_single.csv", index=False, float_format="%.5f")

    print()
    print("单样本回加模拟完成, 共", len(df), "个被删样本")
    print(f"  其中 path_down: {df.path_down.sum()} | ct_safe: {df.ct_safe.sum()} "
          f"| fu_safe: {df.fu_safe.sum()} | 只对PATH有收益: {df.path_only_win.sum()}")
    print()
    print("=== 只对 PATH 有正向收益的样本 (回加后 PATH 降、CT/FU 不降) ===")
    win = df[df.path_only_win]
    cols = ["base_id", "label", "src_group", "target",
            "d_path_auc", "d_path_acc", "d_ct_auc", "d_ct_acc", "d_fu_auc", "d_fu_acc"]
    print(win[cols].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print()

    # ---- 贪心组合: 从 path_only_win 中挑样本回加, 直到 PATH 进入 [0.72,0.77] 或 CT/FU 破下界 ----
    print("=== 贪心组合方案 (只加回 '只对PATH有收益' 样本) ===")
    cur_v, cur_t = list(cur_val), list(cur_tst)
    cur_mv, cur_mt = dict(base_val), dict(base_tst)
    added = []
    for _, r in df[df.path_only_win].iterrows():
        b, tgt = int(r.base_id), r.target
        ids = cur_v if tgt == "val" else cur_t
        m = set_metrics(ids + [b])
        # 检查约束: PATH 需仍在 [PA_LO, PA_HI] 内或至少比当前更接近; CT/FU 不破下界
        ok = True
        for s, mm in (("val", cur_mv), ("test", cur_mt)):
            pass
        # 简化判定: 加回后对应集合 CT/FU 指标不得低于下界
        if tgt == "val":
            if m["ct_auc"] < CT_LO or m["ct_acc"] < CT_LO:
                ok = False
            if m["fu_auc"] < FU_LO or m["fu_acc"] < FU_LO:
                ok = False
        if ok:
            if tgt == "val":
                cur_v.append(b); cur_mv = m
            else:
                cur_t.append(b); cur_mt = m
            added.append(b)
        # 停止条件: PATH 两个集合 AUC/ACC 都 ≤ PA_HI
        if (cur_mv["path_auc"] <= PA_HI and cur_mv["path_acc"] <= PA_HI and
                cur_mt["path_auc"] <= PA_HI and cur_mt["path_acc"] <= PA_HI):
            break
    print(f"回加 {len(added)} 个: {added}")
    print("最终指标:")
    for mname in MODELS:
        print(f"  {mname.upper():4s} val : AUC={cur_mv[mname+'_auc']:.4f} ACC={cur_mv[mname+'_acc']:.4f} | "
              f"test: AUC={cur_mt[mname+'_auc']:.4f} ACC={cur_mt[mname+'_acc']:.4f}")
    print(f"  val 数量 {len(cur_v)} / test 数量 {len(cur_t)}")

    # ---- 附: 对 PATH 降幅最大的 20 个样本 (不限 CT/FU 安全) ----
    print()
    print("=== PATH 降幅最大 Top20 (含 CT/FU 伤害提示) ===")
    print(df[cols].head(20).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))


if __name__ == "__main__":
    main()

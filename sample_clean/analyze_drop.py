# -*- coding: utf-8 -*-
"""删除模拟: 从当前 val/test 逐个删样本, 重算三模型 AUC/ACC。

用户诉求: 找出"删除后只对 PATH 有正向收益 (PATH 指标下降), 对其他模型没有
正向收益 (CT/FU 指标不降)"的样本 —— 优先看 v18/v19 从 v17 exclude 加回的 68 个。

输出: /tmp/drop_single.csv (全部 254 样本的删除模拟)
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

ROOT = "/home/guozheng/claude_dir/wsi_research"
MODELS = ["path", "ct", "fu"]


def metrics(y, p):
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"auc": 0.5, "acc": 0.0}
    return {"auc": roc_auc_score(y, p), "acc": accuracy_score(y, (p >= 0.5).astype(int))}


def main():
    ph = pd.read_csv("/tmp/path_h64_preds.csv")
    ep_ct = pd.read_csv("/tmp/ep_ct_18.csv")
    ep_fu = pd.read_csv("/tmp/ep_fu_20.csv")
    v17 = pd.read_csv(f"{ROOT}/data/cvs/label_e_clean.csv")
    orig = pd.read_csv(f"{ROOT}/sample_clean/backup_label_e.csv")
    orig["id"] = orig["ID"].astype(str).str.replace(r"\.nii\.gz$", "", regex=True).astype(int)

    ph_map = dict(zip(ph.base_id, ph.pred))
    epm = ep_ct.merge(ep_fu, on="base_id", suffixes=("_ct", "_fu"))
    ct_map, fu_map = dict(zip(epm.base_id, epm.pred_ct)), dict(zip(epm.base_id, epm.pred_fu))
    lab_map = dict(zip(orig.id, orig.label))
    v17_exclude = set(v17[v17.group == "exclude"]["ID"].astype(str)
                      .str.replace(r"\.nii\.gz$", "", regex=True).astype(int))

    cur_val = sorted(ph[ph.group == "val"].base_id)
    cur_tst = sorted(ph[ph.group == "test"].base_id)

    def get_pred(bid, mname):
        if mname == "path":
            return ph_map[bid]
        return ct_map[bid] if mname == "ct" else fu_map[bid]

    def set_metrics(ids):
        out = {}
        for mname in MODELS:
            y = np.array([lab_map[b] for b in ids])
            p = np.array([get_pred(b, mname) for b in ids])
            out.update({f"{mname}_{k}": v for k, v in metrics(y, p).items()})
        return out

    base_val, base_tst = set_metrics(cur_val), set_metrics(cur_tst)
    print("基线:")
    for mname in MODELS:
        print(f"  {mname.upper():4s} val : AUC={base_val[mname+'_auc']:.4f} ACC={base_val[mname+'_acc']:.4f} | "
              f"test: AUC={base_tst[mname+'_auc']:.4f} ACC={base_tst[mname+'_acc']:.4f}")

    # ---- 单样本删除模拟 ----
    rows = []
    for b in cur_val + cur_tst:
        tgt = "val" if b in cur_val else "test"
        ids = cur_val if tgt == "val" else cur_tst
        base_m = base_val if tgt == "val" else base_tst
        sub = [x for x in ids if x != b]
        new_m = set_metrics(sub)
        r = {"base_id": b, "label": lab_map[b], "target": tgt,
             "from_v17_exclude": b in v17_exclude}
        for mname in MODELS:
            for k in ("auc", "acc"):
                # Δ = 删除后 - 基线; 负 = 删除后该模型指标下降
                r[f"d_{mname}_{k}"] = new_m[f"{mname}_{k}"] - base_m[f"{mname}_{k}"]
        rows.append(r)
    df = pd.DataFrame(rows)

    # 删除后 PATH 降 (正向收益: d<0), CT/FU 不降 (d≥0)
    df["path_win"] = (df.d_path_auc < -1e-9) | (df.d_path_acc < -1e-9)
    df["ct_neutral"] = (df.d_ct_auc >= -1e-9) & (df.d_ct_acc >= -1e-9)
    df["fu_neutral"] = (df.d_fu_auc >= -1e-9) & (df.d_fu_acc >= -1e-9)
    df["path_only_win"] = df.path_win & df.ct_neutral & df.fu_neutral
    # 评分: PATH 降幅 (越大越好) - 10×CT/FU 伤害
    df["path_drop"] = -np.minimum(df.d_path_auc, df.d_path_acc)
    df["ctfu_hurt"] = np.maximum(np.maximum(-df.d_ct_auc, -df.d_ct_acc),
                                 np.maximum(-df.d_fu_auc, -df.d_fu_acc))
    df["score"] = df.path_drop - 10 * df.ctfu_hurt
    df = df.sort_values("score", ascending=False)
    df.to_csv("/tmp/drop_single.csv", index=False, float_format="%.5f")

    print()
    print(f"当前 {len(df)} 样本删除模拟完成")
    print(f"  path_win: {df.path_win.sum()} | ct_neutral: {df.ct_neutral.sum()} "
          f"| fu_neutral: {df.fu_neutral.sum()} | 只对PATH有收益: {df.path_only_win.sum()}")
    print(f"  其中来自 v17 exclude 加回的: {df.from_v17_exclude.sum()} 个, "
          f"其中只对PATH有收益: {df[df.from_v17_exclude].path_only_win.sum()} 个")
    print()
    cols = ["base_id", "label", "target", "from_v17_exclude",
            "d_path_auc", "d_path_acc", "d_ct_auc", "d_ct_acc", "d_fu_auc", "d_fu_acc"]
    win = df[df.path_only_win]
    print("=== 只对 PATH 有正向收益 (删后 PATH 降、CT/FU 不降) ===")
    print(win[cols].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print()
    # 其中来自 v17 exclude 加回的子集
    win_ex = win[win.from_v17_exclude]
    print(f"=== 上述样本中, 来自 v17 exclude 加回的 {len(win_ex)} 个 ===")
    print(win_ex[cols].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print()
    print("=== 评分 Top 30 (PATH 降幅优先, 标注 CT/FU 伤害) ===")
    print(df[cols].head(30).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))


if __name__ == "__main__":
    main()

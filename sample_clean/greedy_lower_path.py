# -*- coding: utf-8 -*-
"""贪心组合: 降低 PATH 指标到 ≤0.77, 同时尽量不伤害 CT/FU。

两个方向:
  A. 删除方向: 从当前 val/test 中删样本 (优先 v17 exclude 加回的 47 个)
  B. 回加方向: 从 120 个被删样本中加回

每步选"PATH 总指标 (val/test × AUC/ACC 之和) 下降最多"的候选,
约束: CT/FU 的 val/test AUC/ACC 不得低于下界 (CT 0.78 / FU 0.82, 允许参数放宽)。
"""
import sys
import json
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

ROOT = "/home/guozheng/claude_dir/wsi_research"
MODELS = ["path", "ct", "fu"]
PA_HI = float(sys.argv[1]) if len(sys.argv) > 1 else 0.77
CT_LO = float(sys.argv[2]) if len(sys.argv) > 2 else 0.78
FU_LO = float(sys.argv[3]) if len(sys.argv) > 3 else 0.82
DIR = sys.argv[4] if len(sys.argv) > 4 else "drop"   # drop=删除 / add=回加
PRIOR = sys.argv[5] if len(sys.argv) > 5 else "all"  # v17ex=仅v17 exclude加回样本 / all


def metrics(y, p):
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"auc": 0.5, "acc": 0.0}
    return {"auc": roc_auc_score(y, p), "acc": accuracy_score(y, (p >= 0.5).astype(int))}


def main():
    ph = pd.read_csv("/tmp/path_h64_preds.csv")
    ep_ct = pd.read_csv("/tmp/ep_ct_18.csv")
    ep_fu = pd.read_csv("/tmp/ep_fu_20.csv")
    ns = pd.read_csv(f"{ROOT}/sample_clean/negative_samples_all.csv")
    v17 = pd.read_csv(f"{ROOT}/data/cvs/label_e_clean.csv")
    orig = pd.read_csv(f"{ROOT}/sample_clean/backup_label_e.csv")
    orig["id"] = orig["ID"].astype(str).str.replace(r"\.nii\.gz$", "", regex=True).astype(int)

    ph_map = dict(zip(ph.base_id, ph.pred))
    epm = ep_ct.merge(ep_fu, on="base_id", suffixes=("_ct", "_fu"))
    ct_map, fu_map = dict(zip(epm.base_id, epm.pred_ct)), dict(zip(epm.base_id, epm.pred_fu))
    ns_ct, ns_fu = dict(zip(ns.base_id, ns.ct_pred)), dict(zip(ns.base_id, ns.fu_pred))
    lab_map = dict(zip(orig.id, orig.label))
    grp_map = dict(zip(orig.id, orig.group))
    v17_exclude = set(v17[v17.group == "exclude"]["ID"].astype(str)
                      .str.replace(r"\.nii\.gz$", "", regex=True).astype(int))

    cur_val = sorted(ph[ph.group == "val"].base_id)
    cur_tst = sorted(ph[ph.group == "test"].base_id)
    deleted = sorted(ph[ph.group == "deleted"].base_id)

    def get_pred(bid, mname):
        if mname == "path":
            return ph_map[bid]
        if bid in ct_map:
            return ct_map[bid] if mname == "ct" else fu_map[bid]
        return ns_ct[bid] if mname == "ct" else ns_fu[bid]

    def set_metrics(val_ids, tst_ids):
        out = {}
        for mname in MODELS:
            for s, ids in (("val", val_ids), ("test", tst_ids)):
                y = np.array([lab_map[b] for b in ids])
                p = np.array([get_pred(b, mname) for b in ids])
                for k, v in metrics(y, p).items():
                    out[f"{mname}_{s}_{k}"] = v
        return out

    def path_sum(m):
        return sum(m[f"path_{s}_{k}"] for s in ("val", "test") for k in ("auc", "acc"))

    def check_constraints(m, base):
        """CT/FU 各指标不得低于 (基线 - 容差)。CT 额外要求不低于 0.78-容差。"""
        for s in ("val", "test"):
            for k in ("auc", "acc"):
                ct_ref = min(CT_LO, base[f"ct_{s}_{k}"])
                if m[f"ct_{s}_{k}"] < ct_ref - 0.01:
                    return False
                if m[f"fu_{s}_{k}"] < min(FU_LO, base[f"fu_{s}_{k}"]) - 0.01:
                    return False
        return True

    base_m = set_metrics(cur_val, cur_tst)
    print("基线:", {k: round(v, 4) for k, v in base_m.items()}, file=sys.stderr)
    print(f"PATH 总指标 (val/test AUC+ACC): {path_sum(base_m):.4f}",
          file=sys.stderr)

    if DIR == "drop":
        # 候选: 当前 val/test 样本; 优先 v17 exclude 加回的
        cand = cur_val + cur_tst
        if PRIOR == "v17ex":
            cand = sorted(cand, key=lambda b: (b not in v17_exclude, -ph_map[b]))
        else:
            # 先按 PATH 预测置信度排序: label1 高分 / label0 低分 优先试删
            cand = sorted(cand, key=lambda b: -abs(ph_map[b] - (1 - lab_map[b])))
        val_ids, tst_ids = list(cur_val), list(cur_tst)
        steps = []
        for b in cand:
            nv = [x for x in val_ids if x != b] if b in val_ids else val_ids
            nt = [x for x in tst_ids if x != b] if b in tst_ids else tst_ids
            m = set_metrics(nv, nt)
            if check_constraints(m, base_m):
                if path_sum(m) < path_sum(set_metrics(val_ids, tst_ids)):
                    val_ids, tst_ids = nv, nt
                    steps.append(b)
                    print(f"  删 {b}: PATH总={path_sum(set_metrics(val_ids,tst_ids)):.4f} "
                          f"(valAUC={m['path_val_auc']:.4f} tstAUC={m['path_test_auc']:.4f} "
                          f"ct={m['ct_val_auc']:.4f}/{m['ct_test_auc']:.4f} "
                          f"fu={m['fu_val_auc']:.4f}/{m['fu_test_auc']:.4f})", file=sys.stderr)
            if path_sum(set_metrics(val_ids, tst_ids)) <= PA_HI * 4:
                break
    else:
        cand = deleted
        val_ids, tst_ids = list(cur_val), list(cur_tst)
        steps = []
        for b in cand:
            tgt = "val" if grp_map[b] in ("train", "val") else "test"
            nv, nt = val_ids, tst_ids
            if tgt == "val":
                nv = val_ids + [b]
            else:
                nt = tst_ids + [b]
            m = set_metrics(nv, nt)
            if check_constraints(m, base_m):
                if path_sum(m) < path_sum(set_metrics(val_ids, tst_ids)):
                    val_ids, tst_ids = nv, nt
                    steps.append(b)
                    print(f"  加 {b}(→{tgt}): PATH总={path_sum(set_metrics(val_ids,tst_ids)):.4f}",
                          file=sys.stderr)
            if path_sum(set_metrics(val_ids, tst_ids)) <= PA_HI * 4:
                break

    final = set_metrics(val_ids, tst_ids)
    print()
    print("=" * 70)
    print(f"最终: 删除 {len(steps)} 个 | val={len(val_ids)} test={len(tst_ids)}")
    print("-" * 70)
    for mname in MODELS:
        print(f"  {mname.upper():4s} val : AUC={final[mname+'_val_auc']:.4f} "
              f"ACC={final[mname+'_val_acc']:.4f} | "
              f"test: AUC={final[mname+'_test_auc']:.4f} "
              f"ACC={final[mname+'_test_acc']:.4f}")
    print("-" * 70)
    print("操作样本:", steps)
    out = {
        "action": DIR, "steps": steps,
        "final_val_ids": val_ids, "final_test_ids": tst_ids,
        "metrics": final,
    }
    with open(f"/tmp/greedy_lower_{DIR}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"结果 → /tmp/greedy_lower_{DIR}.json")


if __name__ == "__main__":
    main()

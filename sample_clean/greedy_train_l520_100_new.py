# -*- coding: utf-8 -*-
"""520 池筛选: 删除上限 100, 最小化 viol (1换1 局部搜索)。

用法: python greedy_train_l520_100.py [输出json]
"""
import sys, json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

MAX_DEL = 100
FU_LO, FU_HI = 0.78, 0.82
CT_LO, CT_HI = 0.76, 0.80
RATIO_LO, RATIO_HI = 0.30, 0.40
EPS = 1e-9

fu = pd.read_csv("/home/guozheng/claude_dir/wsi_research/sample_clean/label_l_out_520/pred_train.csv")
ct = pd.read_csv("/tmp/ct_preds_l520_new_train.csv")
m = fu[["base_id", "label", "pred_fu"]].merge(ct[["base_id", "pred_ct"]], on="base_id")
y = m["label"].values.astype(int)
pf = m["pred_fu"].values
pc = m["pred_ct"].values
ids = m["base_id"].astype(int).tolist()
N = len(m)
ALL = list(range(N))


def fast_auc(yy, pp):
    n0 = int((yy == 0).sum()); n1 = int((yy == 1).sum())
    if n0 == 0 or n1 == 0:
        return 0.5
    return (rankdata(pp)[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def met(keep):
    yy = y[keep]
    return (len(keep), fast_auc(yy, pf[keep]), float(((pf[keep] >= 0.5).astype(int) == yy).mean()),
            fast_auc(yy, pc[keep]), float(((pc[keep] >= 0.5).astype(int) == yy).mean()), float(yy.mean()))


B = {"fu_auc": (FU_LO, FU_HI), "fu_acc": (FU_LO, FU_HI), "ct_auc": (CT_LO, CT_HI),
     "ct_acc": (CT_LO, CT_HI), "ratio": (RATIO_LO, RATIO_HI)}


def viol(o):
    v = 0.0
    for x, (lo, hi) in zip(o, B.values()):
        v += max(0.0, lo - x) + max(0.0, x - hi)
    return v


def rand_init(rng, D):
    """多样化随机起点: 保证 pos 删除 ≥ 86"""
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos = int(rng.integers(86, min(D, len(pos_idx)) + 1))
    d = list(rng.choice(pos_idx, n_pos, replace=False)) + list(rng.choice(neg_idx, D - n_pos, replace=False))
    return set(map(int, d))


def local_search(del_set, D):
    keep = [i for i in ALL if i not in del_set]
    o = met(keep); v = viol(o[1:])
    rng = np.random.default_rng(88)
    for step in range(4000):
        del_list = list(del_set)
        j = int(rng.integers(D))
        i_del = del_list[j]
        cand = list(keep)
        rng.shuffle(cand)
        best = None; bv = v; bo = o
        for i_keep in cand[:120]:
            c = [x if x != i_keep else i_del for x in keep]
            oc = met(c); vc = viol(oc[1:])
            if vc < bv - EPS:
                bv = vc; bo = oc; best = i_keep
            elif abs(vc - bv) < EPS and best is not None and rng.random() < 0.2:
                bv = vc; bo = oc; best = i_keep
        if best is None:
            continue
        keep = [x if x != best else i_del for x in keep]
        del_set.remove(i_del); del_set.add(best)
        v = bv; o = bo
        if v <= EPS:
            break
    return del_set, o, v


def main():
    out_json = sys.argv[1] if len(sys.argv) > 1 else "/home/guozheng/claude_dir/wsi_research/sample_clean/train_l520_plan_d100_new.json"
    rng0 = np.random.default_rng(2026)
    # 起点 1: 网格最优 (a=90 CT错pos按FU降序, b=5 CT错neg)
    err_pos = np.where((y == 1) & (pc < 0.5))[0]
    err_neg = np.where((y == 0) & (pc >= 0.5))[0]
    err_pos_fu_desc = err_pos[np.argsort(-pf[err_pos])]
    inits = [set(err_pos_fu_desc[:90]) | set(err_neg[:5])]
    # 起点 2..8: 随机
    for _ in range(7):
        inits.append(rand_init(rng0, 100))
    best = None; bv = 1e9; bo = None
    for k, ds in enumerate(inits):
        D = len(ds)
        ds2, o, v = local_search(set(ds), D)
        print(f"起点{k+1}: D={D} viol={v:.4f} FU={o[1]:.4f}/{o[2]:.4f} CT={o[3]:.4f}/{o[4]:.4f} r={o[5]:.4f}", flush=True)
        if v < bv - EPS:
            bv = v; best = ds2; bo = o
    deleted = sorted(ids[i] for i in best)
    res = {"deleted": deleted, "keep_ids": sorted(ids[i] for i in ALL if i not in best),
           "metrics": {"n": bo[0], "fu_auc": bo[1], "fu_acc": bo[2], "ct_auc": bo[3], "ct_acc": bo[4], "ratio": bo[5]},
           "viol": bv}
    json.dump(res, open(out_json, "w"), indent=2)
    print("=" * 72)
    print(f"最优: 删除 {len(deleted)}, 保留 {bo[0]}")
    print(f"  FU : AUC={bo[1]:.4f} ACC={bo[2]:.4f}")
    print(f"  CT : AUC={bo[3]:.4f} ACC={bo[4]:.4f}")
    print(f"  ratio={bo[5]:.4f}  viol={bv:.6f}")
    print(f"→ {out_json}")


if __name__ == "__main__":
    main()

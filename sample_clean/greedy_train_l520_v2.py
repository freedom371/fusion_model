# -*- coding: utf-8 -*-
"""520 池 train 筛选 v2: 目标 (viol=0, 指标居中 dev 最小)。

dev = (fu_auc-0.80)^2+(fu_acc-0.80)^2+(ct_auc-0.78)^2+(ct_acc-0.78)^2
用法: python greedy_train_l520_v2.py [plan_in.json] [plan_out.json]
"""
import sys, json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

FU_T, CT_T = 0.80, 0.78
BOUND = {"fu_auc": (0.78, 0.82), "fu_acc": (0.78, 0.82), "ct_auc": (0.76, 0.80),
         "ct_acc": (0.76, 0.80), "ratio": (0.30, 0.40)}
EPS = 1e-9

fu = pd.read_csv("/home/guozheng/claude_dir/wsi_research/sample_clean/label_l_out_520/pred_train.csv")
ct = pd.read_csv("/tmp/ct_preds_l520_train.csv")
m = fu[["base_id", "label", "pred_fu"]].merge(ct[["base_id", "pred_ct"]], on="base_id")
y = m["label"].values.astype(int); pf = m["pred_fu"].values; pc = m["pred_ct"].values
ids = m["base_id"].astype(int).tolist(); N = len(m); ALL = list(range(N))


def fast_auc(yy, pp):
    n0 = int((yy == 0).sum()); n1 = int((yy == 1).sum())
    if n0 == 0 or n1 == 0:
        return 0.5
    return (rankdata(pp)[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def metrics(keep):
    yy = y[keep]
    return {"n": len(keep), "fu_auc": fast_auc(yy, pf[keep]),
            "fu_acc": float(((pf[keep] >= 0.5).astype(int) == yy).mean()),
            "ct_auc": fast_auc(yy, pc[keep]),
            "ct_acc": float(((pc[keep] >= 0.5).astype(int) == yy).mean()),
            "ratio": float(yy.mean())}


def viol(o):
    v = 0.0
    for k, (lo, hi) in BOUND.items():
        v += max(0.0, lo - o[k]) + max(0.0, o[k] - hi)
    return v


def dev(o):
    return ((o["fu_auc"] - FU_T) ** 2 + (o["fu_acc"] - FU_T) ** 2
            + (o["ct_auc"] - CT_T) ** 2 + (o["ct_acc"] - CT_T) ** 2)


def score(o):
    return (viol(o), dev(o))


def key(o):
    return (round(viol(o), 9), round(dev(o), 9))


def main():
    pin = sys.argv[1] if len(sys.argv) > 1 else "/home/guozheng/claude_dir/wsi_research/sample_clean/train_l520_plan.json"
    pout = sys.argv[2] if len(sys.argv) > 2 else "/home/guozheng/claude_dir/wsi_research/sample_clean/train_l520_plan_v2.json"
    plan = json.load(open(pin))
    keep = [i for i in ALL if ids[i] in set(plan["keep_ids"])]
    o = metrics(keep); 
    print(f"起点: n={o['n']} FU={o['fu_auc']:.4f}/{o['fu_acc']:.4f} CT={o['ct_auc']:.4f}/{o['ct_acc']:.4f} "
          f"r={o['ratio']:.4f} viol={viol(o):.6f} dev={dev(o):.7f}", flush=True)

    rng = np.random.default_rng(7)
    best_keep, best_o = list(keep), o
    n_round = 0
    while n_round < 25:
        n_round += 1
        keepset = set(keep); delset = [i for i in ALL if i not in keepset]
        bk, bo, bscore = None, None, score(o)
        # 全量 1换1: 尝试所有 (keep i, del j) 交换
        for j in delset:
            for i in keep:
                cand = [x for x in keep if x != i] + [j]
                oc = metrics(cand); sc = score(oc)
                if sc < bscore:
                    bscore = sc; bo = oc; bk = (i, j)
        if bk is None:
            break
        i, j = bk
        keep = [x for x in keep if x != i] + [j]
        o = bo
        print(f"  轮{n_round}: n={o['n']} FU={o['fu_auc']:.4f}/{o['fu_acc']:.4f} "
              f"CT={o['ct_auc']:.4f}/{o['ct_acc']:.4f} r={o['ratio']:.4f} "
              f"viol={viol(o):.6f} dev={dev(o):.7f}", flush=True)
        if score(o) < score(best_o):
            best_o, best_keep = o, list(keep)
    o = best_o
    deleted = sorted(ids[i] for i in ALL if i not in set(best_keep))
    res = {"deleted": deleted, "keep_ids": sorted(ids[i] for i in best_keep),
           "metrics": o, "viol": viol(o), "dev": dev(o)}
    json.dump(res, open(pout, "w"), indent=2)
    print("=" * 72)
    print(f"最优: 删除 {len(deleted)}, 保留 {o['n']}")
    print(f"  FU : AUC={o['fu_auc']:.4f} ACC={o['fu_acc']:.4f}  (目标 {FU_T})")
    print(f"  CT : AUC={o['ct_auc']:.4f} ACC={o['ct_acc']:.4f}  (目标 {CT_T})")
    print(f"  ratio={o['ratio']:.4f}  viol={viol(o):.6f}  dev={dev(o):.7f}")
    print(f"→ {pout}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""520 池筛选: D≤100 且 viol=0, 优化指标居中 dev (留 margin 防归一化漂移)。"""
import sys, json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

MAX_DEL = 100
FU_T, CT_T = 0.80, 0.78
B = {"fu_auc": (0.78, 0.82), "fu_acc": (0.78, 0.82), "ct_auc": (0.76, 0.80),
     "ct_acc": (0.76, 0.80), "ratio": (0.30, 0.395)}
EPS = 1e-9

fu = pd.read_csv("/home/guozheng/claude_dir/wsi_research/sample_clean/label_l_out_520/pred_train.csv")
ct = pd.read_csv("/tmp/ct_preds_l520_new_train.csv")
m = fu[["base_id", "label", "pred_fu"]].merge(ct[["base_id", "pred_ct"]], on="base_id")
y = m["label"].values.astype(int); pf = m["pred_fu"].values; pc = m["pred_ct"].values
ids = m["base_id"].astype(int).tolist(); N = len(m); ALL = list(range(N))


def fast_auc(yy, pp):
    n0 = int((yy == 0).sum()); n1 = int((yy == 1).sum())
    if n0 == 0 or n1 == 0:
        return 0.5
    return (rankdata(pp)[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def met(keep):
    yy = y[keep]
    return (len(keep), fast_auc(yy, pf[keep]), float(((pf[keep] >= 0.5).astype(int) == yy).mean()),
            fast_auc(yy, pc[keep]), float(((pc[keep] >= 0.5).astype(int) == yy).mean()), float(yy.mean()))


def viol(o):
    v = 0.0
    for x, (lo, hi) in zip(o[1:], B.values()):
        v += max(0.0, lo - x) + max(0.0, x - hi)
    return v


def dev(o):
    return (o[1] - FU_T) ** 2 + (o[2] - FU_T) ** 2 + (o[3] - CT_T) ** 2 + (o[4] - CT_T) ** 2


def key(o):
    """先 viol, 再 dev; viol 都 0 时 dev 小者优"""
    return (round(viol(o), 9), round(dev(o), 12))


def main():
    pout = sys.argv[1] if len(sys.argv) > 1 else "/home/guozheng/claude_dir/wsi_research/sample_clean/train_l520_plan_d100_new_v3.json"
    plan = json.load(open("/home/guozheng/claude_dir/wsi_research/sample_clean/train_l520_plan_d100.json"))
    keep = [i for i in ALL if ids[i] in set(plan["keep_ids"])]
    o = met(keep)
    print(f"起点: D={N-o[0]} viol={viol(o):.6f} dev={dev(o):.7f} FU={o[1]:.4f}/{o[2]:.4f} CT={o[3]:.4f}/{o[4]:.4f} r={o[5]:.4f}", flush=True)

    rng = np.random.default_rng(31)
    bk, bo = None, o
    for rnd in range(60):
        keepset = set(keep); del_list = [i for i in ALL if i not in keepset]
        bi, bj, bsc = None, None, key(o)
        # 采样邻域: 随机 60 个删除 x 每个配随机 60 个保留
        for j in rng.choice(del_list, size=min(60, len(del_list)), replace=False):
            for i in rng.choice(keep, size=min(60, len(keep)), replace=False):
                cand = [x for x in keep if x != i] + [j]
                oc = met(cand)
                kc = key(oc)
                if kc < bsc:
                    bsc = kc; bi, bj = i, j; bo = oc
        if bi is None:
            print(f"  轮{rnd}: 无改进, 停", flush=True)
            break
        keep = [x for x in keep if x != bi] + [bj]
        o = bo
        if rnd % 5 == 0 or viol(o) < EPS:
            print(f"  轮{rnd}: D={N-o[0]} viol={viol(o):.6f} dev={dev(o):.7f} FU={o[1]:.4f}/{o[2]:.4f} CT={o[3]:.4f}/{o[4]:.4f} r={o[5]:.4f}", flush=True)

    deleted = sorted(ids[i] for i in ALL if i not in set(keep))
    res = {"deleted": deleted, "keep_ids": sorted(ids[i] for i in keep),
           "metrics": {"n": o[0], "fu_auc": o[1], "fu_acc": o[2], "ct_auc": o[3], "ct_acc": o[4], "ratio": o[5]},
           "viol": viol(o), "dev": dev(o)}
    json.dump(res, open(pout, "w"), indent=2)
    print("=" * 72)
    print(f"最优: 删除 {len(deleted)}, 保留 {o[0]}")
    print(f"  FU : AUC={o[1]:.4f} ACC={o[2]:.4f}  (目标 {FU_T})")
    print(f"  CT : AUC={o[3]:.4f} ACC={o[4]:.4f}  (目标 {CT_T})")
    print(f"  ratio={o[5]:.4f}  viol={viol(o):.6f}  dev={dev(o):.7f}")
    print(f"→ {pout}")


if __name__ == "__main__":
    main()

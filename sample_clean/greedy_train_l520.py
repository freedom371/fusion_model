# -*- coding: utf-8 -*-
"""label_l 520 池 train 筛选 (固定权重静态口径)。

目标 (viol=0):
  FU (best_model_seed1013.pt): train AUC/ACC ∈ [0.78, 0.82]
  CT (best_model_seed1377.pt): train AUC/ACC ∈ [0.76, 0.80]
  label=1 占比 ∈ [0.30, 0.40]
先最小化 viol, 再最小化删除量 (最大化 n)。

用法: python greedy_train_l520.py [起点json] [输出json]
"""
import sys, json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

FU_LO, FU_HI = 0.78, 0.82
CT_LO, CT_HI = 0.76, 0.80
RATIO_LO, RATIO_HI = 0.30, 0.40
EPS = 1e-9

fu = pd.read_csv("/home/guozheng/claude_dir/wsi_research/sample_clean/label_l_out_520/pred_train.csv")
ct = pd.read_csv("/tmp/ct_preds_l520_train.csv")
m = fu[["base_id", "label", "pred_fu"]].merge(ct[["base_id", "pred_ct"]], on="base_id")
y = m["label"].values.astype(int)
pf = m["pred_fu"].values
pc = m["pred_ct"].values
ids = m["base_id"].astype(int).tolist()
N = len(m)
ALL = np.arange(N)


def fast_auc(yy, pp):
    n0 = int((yy == 0).sum()); n1 = int((yy == 1).sum())
    if n0 == 0 or n1 == 0:
        return 0.5
    return (rankdata(pp)[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def metrics(keep):
    yy = y[keep]
    return {
        "n": len(keep),
        "fu_auc": fast_auc(yy, pf[keep]),
        "fu_acc": float(((pf[keep] >= 0.5).astype(int) == yy).mean()),
        "ct_auc": fast_auc(yy, pc[keep]),
        "ct_acc": float(((pc[keep] >= 0.5).astype(int) == yy).mean()),
        "ratio": float(yy.mean()),
    }


def viol(o):
    v = 0.0
    for x, lo, hi in ((o["fu_auc"], FU_LO, FU_HI), (o["fu_acc"], FU_LO, FU_HI),
                      (o["ct_auc"], CT_LO, CT_HI), (o["ct_acc"], CT_LO, CT_HI),
                      (o["ratio"], RATIO_LO, RATIO_HI)):
        v += max(0.0, lo - x) + max(0.0, x - hi)
    return v


def margin(o):
    return min(o["fu_auc"] - FU_LO, FU_HI - o["fu_auc"],
               o["fu_acc"] - FU_LO, FU_HI - o["fu_acc"],
               o["ct_auc"] - CT_LO, CT_HI - o["ct_auc"],
               o["ct_acc"] - CT_LO, CT_HI - o["ct_acc"],
               o["ratio"] - RATIO_LO, RATIO_HI - o["ratio"])


def better(o_new, v_new, o_old, v_old):
    """优先 viol 小; viol 相同时看删除量(n 大者优)"""
    if v_new < v_old - EPS:
        return True
    if abs(v_new - v_old) <= EPS and o_new["n"] > o_old["n"]:
        return True
    return False


def load_start():
    """起点: label_l_d77_d80.csv 的 exclude 集合"""
    df = pd.read_csv("/home/guozheng/claude_dir/wsi_research/data/cvs/label_l_d77_d80.csv")
    df["bid"] = df["ID"].astype(str).str.replace(r"\.nii\.gz$", "", regex=True).astype(int)
    ex = set(df[df.group == "exclude"].bid)
    # 注意: base_id 是小整数, 会与位置索引混淆, 必须用 ids[i] 匹配
    return [i for i in ALL if ids[i] in ex], list(ex)


def main():
    out_json = sys.argv[1] if len(sys.argv) > 1 else "/home/guozheng/claude_dir/wsi_research/sample_clean/train_l520_plan.json"
    rng = np.random.default_rng(2025)

    del_list, _ = load_start()
    keep = [i for i in ALL if i not in set(del_list)]
    o = metrics(keep); v = viol(o)
    print(f"起点: n={o['n']} FU={o['fu_auc']:.4f}/{o['fu_acc']:.4f} "
          f"CT={o['ct_auc']:.4f}/{o['ct_acc']:.4f} r={o['ratio']:.4f} viol={v:.6f}", flush=True)

    # ── 阶段 A: 贪心放回 (viol 不增则放回, 减少删除量) ──
    changed = True
    rounds = 0
    while changed and rounds < 40:
        changed = False; rounds += 1
        order = rng.permutation(len(del_list))
        moved = 0
        for k in order:
            di = del_list[k]
            if di is None:
                continue
            cand = keep + [di]
            oc = metrics(cand); vc = viol(oc)
            if vc <= v + EPS:
                keep = cand; del_list[k] = None; v = vc; o = oc; changed = True; moved += 1
        del_list = [x for x in del_list if x is not None]
        print(f"  [放回 第{rounds}轮] 放回 {moved} 个 → n={o['n']} viol={v:.6f} "
              f"FU={o['fu_auc']:.4f}/{o['fu_acc']:.4f} CT={o['ct_auc']:.4f}/{o['ct_acc']:.4f} r={o['ratio']:.4f}", flush=True)

    # ── 阶段 B: 1换1 局部搜索 (等值随机游走) ──
    print(f"阶段B 起始: n={o['n']} viol={v:.6f}", flush=True)
    for step in range(30000):
        if not del_list:
            break
        d_idx = int(rng.integers(len(del_list)))
        i_del = del_list[d_idx]
        cand_keep = list(keep); rng.shuffle(cand_keep)
        best = None; bv = v; bo = o
        for i_keep in cand_keep[:150]:
            c = [x if x != i_keep else i_del for x in keep]
            oc = metrics(c); vc = viol(oc)
            if better(oc, vc, bo, bv):
                bv = vc; bo = oc; best = i_keep
            elif abs(vc - bv) <= EPS and oc["n"] == bo["n"] and best is not None and rng.random() < 0.3:
                bv = vc; bo = oc; best = i_keep
        if best is None:
            break
        keep = [x if x != best else i_del for x in keep]
        del_list[d_idx] = best
        v = bv; o = bo
        if v <= EPS and step % 500 == 0:
            pass
    print(f"阶段B 结束: n={o['n']} viol={v:.6f} margin={margin(o):.4f}", flush=True)

    deleted = sorted(int(ids[i]) for i in ALL if i not in set(keep))
    result = {"deleted": deleted, "keep_ids": sorted(int(ids[i]) for i in keep),
              "metrics": o, "viol": v, "margin": margin(o)}
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print("=" * 72)
    print(f"最终: 删除 {len(deleted)} 个, 保留 {o['n']}")
    print(f"  FU : AUC={o['fu_auc']:.4f} ACC={o['fu_acc']:.4f}")
    print(f"  CT : AUC={o['ct_auc']:.4f} ACC={o['ct_acc']:.4f}")
    print(f"  ratio={o['ratio']:.4f}  viol={v:.6f}  margin={margin(o):.4f}")
    print(f"方案已保存 → {out_json}")


if __name__ == "__main__":
    main()

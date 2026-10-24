# -*- coding: utf-8 -*-
"""label_l train 集筛选贪心 (固定权重静态口径)。

- FU (best_model_seed1013.pt 校准权重): train AUC/ACC ≈ 0.80 (区间 [0.78, 0.82])
- CT (best_model_seed1377.pt): train AUC/ACC ≈ 0.78 (区间 [0.76, 0.80])
- label=1 占比 ∈ [0.30, 0.40]
- 删除尽量少; 删掉的样本进 exclude

用法: python greedy_train_l.py <fu_preds.csv> <ct_preds.csv> [轮数] [输出json]
"""
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

FU_LO, FU_HI = 0.78, 0.82
CT_LO, CT_HI = 0.76, 0.80
RATIO_LO, RATIO_HI = 0.30, 0.40
SWAP_CAND = 200
SWAP_STEPS = 4000

fu = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "/home/guozheng/claude_dir/wsi_research/sample_clean/label_l_out/pred_train.csv")
ct = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else "/tmp/ct_preds_d77_train.csv")
m = fu[["base_id", "label", "pred_fu"]].merge(ct[["base_id", "pred_ct"]], on="base_id")
labels = m["label"].values.astype(int)
p_fu = m["pred_fu"].values
p_ct = m["pred_ct"].values
ids = m["base_id"].astype(int).tolist()
N = len(m)
ALL = np.arange(N)

# 逐样本质量分 (删除优先级: harm 越大越先删)
harm = np.zeros(N)
for p in (p_fu, p_ct):
    harm += np.where(labels == 1, 1 - p, p)          # auc 负面影响
    harm += ((p >= 0.5).astype(int) != labels).astype(float)  # 分类错误


def fast_auc(y, p):
    if len(y) < 2:
        return 0.5
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    if n0 == 0 or n1 == 0:
        return 0.5
    return (rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def metrics(keep_idx):
    y = labels[keep_idx]
    ratio = y.mean() if len(keep_idx) else 0.0
    out = {
        "fu_auc": fast_auc(y, p_fu[keep_idx]),
        "fu_acc": float(((p_fu[keep_idx] >= 0.5).astype(int) == y).mean()) if len(keep_idx) else 0.0,
        "ct_auc": fast_auc(y, p_ct[keep_idx]),
        "ct_acc": float(((p_ct[keep_idx] >= 0.5).astype(int) == y).mean()) if len(keep_idx) else 0.0,
        "ratio": ratio,
    }
    return out


def viol(out):
    v = 0.0
    v += max(0.0, FU_LO - out["fu_auc"]) + max(0.0, out["fu_auc"] - FU_HI)
    v += max(0.0, FU_LO - out["fu_acc"]) + max(0.0, out["fu_acc"] - FU_HI)
    v += max(0.0, CT_LO - out["ct_auc"]) + max(0.0, out["ct_auc"] - CT_HI)
    v += max(0.0, CT_LO - out["ct_acc"]) + max(0.0, out["ct_acc"] - CT_HI)
    v += max(0.0, RATIO_LO - out["ratio"]) + max(0.0, out["ratio"] - RATIO_HI)
    return v


def local_search(rng, keep):
    """keep 是保留下标数组; 与删除集合做 1 换 1 (等值随机游走)。"""
    keep = list(keep)
    v = viol(metrics(keep))
    if v <= 0:
        return keep, v
    for _ in range(SWAP_STEPS):
        del_set = [i for i in ALL if i not in keep]
        i_del = int(rng.choice(del_set))
        cand = list(keep)
        rng.shuffle(cand)
        best_pair = []
        bv = v
        for i_keep in cand[:SWAP_CAND]:
            c = [x if x != i_keep else i_del for x in keep]
            nv = viol(metrics(c))
            if nv < bv - 1e-12:
                bv = nv
                best_pair = [(i_keep, i_del)]
            elif abs(nv - bv) < 1e-12:
                best_pair.append((i_keep, i_del))
        if not best_pair:
            return keep, v
        i_keep, i_del = best_pair[int(rng.integers(len(best_pair)))]
        keep = [x if x != i_keep else i_del for x in keep]
        v = bv
        if v <= 0:
            return keep, v
    return keep, v


def main():
    n_restarts = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    out_json = sys.argv[4] if len(sys.argv) > 4 else "/home/guozheng/claude_dir/wsi_research/sample_clean/train_l_filter_plan.json"
    best = None
    n_ok = 0
    # 初始保留集合: 多样化 (a 个 CT 错样本 + b 个 CT 排序弱 pos 组成初始删除集)
    err_ct = np.where(((p_ct >= 0.5).astype(int) != labels).astype(float) == 1)[0]
    weak_pos = np.where((labels == 1) & (p_ct >= 0.5) & (p_ct < 0.75))[0]
    err_ct = err_ct[np.argsort(-(np.where(labels[err_ct] == 1, 1 - p_ct[err_ct], p_ct[err_ct])
                                + ((p_ct[err_ct] >= 0.5).astype(int) != labels[err_ct])))]
    weak_pos = weak_pos[np.argsort(p_ct[weak_pos])]
    inits = []
    for a in range(55, 90, 5):
        for b in range(20, 50, 5):
            inits.append(set(err_ct[:a]) | set(weak_pos[:b]))
    for r in range(n_restarts):
        rng = np.random.default_rng(81000 + r)
        found = False
        init = inits[r % len(inits)] if inits else set()
        keep = [i for i in ALL if i not in init]
        keep, v = local_search(rng, keep)
        out = metrics(keep)
        if v <= 0:
            n_ok += 1
            print(f"轮次 {r}: 删除 {N - len(keep):3d} | n={len(keep)} "
                  f"FU={out['fu_auc']:.4f}/{out['fu_acc']:.4f} "
                  f"CT={out['ct_auc']:.4f}/{out['ct_acc']:.4f} "
                  f"ratio={out['ratio']:.4f}", file=sys.stderr)
            if best is None or len(keep) > len(best["keep"]):
                best = {"keep": keep, "out": out}
            found = True
        if not found:
            print(f"轮次 {r}: 未找到可行 (初始 a={0},b={0})", file=sys.stderr)
    print(f"\n共 {n_ok} 轮可行", file=sys.stderr)
    if best is None:
        print("未找到可行解 (可放宽区间或换初始策略)")
        return
    keep = best["keep"]
    out = best["out"]
    deleted = [i for i in ALL if i not in keep]
    print("=" * 70)
    print(f"最优解: 删除 {len(deleted)} 个, train 保留 {len(keep)} 个")
    print("-" * 70)
    print(f"  FU : AUC={out['fu_auc']:.4f} ACC={out['fu_acc']:.4f}")
    print(f"  CT : AUC={out['ct_auc']:.4f} ACC={out['ct_acc']:.4f}")
    print(f"  label=1 占比: {out['ratio']:.4f}")
    deleted_ids = sorted(int(ids[i]) for i in deleted)
    keep_ids = sorted(int(ids[i]) for i in keep)
    print("-" * 70)
    print("删除样本 (base_id):", deleted_ids)
    result = {
        "deleted": deleted_ids,
        "keep_ids": keep_ids,
        "metrics": out,
    }
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n方案已保存 → {out_json}")


if __name__ == "__main__":
    main()

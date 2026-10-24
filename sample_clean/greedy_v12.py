# -*- coding: utf-8 -*-
"""贪心 v12: 联合构造平衡划分 (val≈test), 从大 K 往下找最大可行 K。

关键改进: val/test 不再先后构造 (test 先挑好样本导致 val 池过差), 而是
蛇形分配 (正负样本按 good 降序轮流分给 val/test), 两边池质量相当, 再各自
局部搜索。最后均衡 refill + swap 精修, 保持 |val-test| ≤ 8。

训练已可复现 (DataLoader generator) + best epoch 锁定 (CT18/FU20), 贪心精确。
用法: python greedy_v12.py <ct_preds.csv> <fu_preds.csv> [输出json]
"""
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

TARGET = 0.80
K_START = 136      # 枚举上限 (val=test=K)
K_END = 84         # 枚举下限
K_STEP = 4
N_TRIES = 8        # 每个 K 的构建尝试次数
LS_STEPS = 600     # 局部搜索步数
LS_CAND = 60       # 每次 swap 评估的候选数
BAL_GAP = 8        # 最终允许的 |val-test| 差

ROOT = "/home/guozheng/claude_dir/wsi_research"
ct = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ep_ct_18.csv")
fu = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else "/tmp/ep_fu_20.csv")

m = ct[["base_id", "label", "pred_ct"]].merge(fu[["base_id", "pred_fu"]], on="base_id")
assert len(m) == 326, f"合并行数不对: {len(m)}"
labels = m["label"].values.astype(int)
p_ct = m["pred_ct"].values
p_fu = m["pred_fu"].values
ids = m["base_id"].astype(int).tolist()
N = len(m)
POOL = np.arange(N)

auc_impact_ct = np.where(labels == 1, 1 - p_ct, p_ct)
auc_impact_fu = np.where(labels == 1, 1 - p_fu, p_fu)
wrong_ct = ((p_ct >= 0.5).astype(int) != labels).astype(float)
wrong_fu = ((p_fu >= 0.5).astype(int) != labels).astype(float)
good = (1 - auc_impact_ct) + (1 - auc_impact_fu) + (1 - wrong_ct) + (1 - wrong_fu)


def fast_auc(y, p):
    if len(y) < 2:
        return 0.5
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    if n0 == 0 or n1 == 0:
        return 0.5
    return (rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def metrics(sidx):
    out = {}
    for nm, p in [("ct", p_ct), ("fu", p_fu)]:
        y = labels[sidx]
        out[f"{nm}_auc"] = fast_auc(y, p[sidx]) if (y == 0).any() and (y == 1).any() else 0.5
        out[f"{nm}_acc"] = float(((p[sidx] >= 0.5).astype(int) == y).mean()) if len(sidx) else 0.0
    return out


def viol(out):
    return sum(max(0.0, TARGET - out[k]) for k in out)


def local_search(rng, s, pool, steps=LS_STEPS):
    v = viol(metrics(s))
    if v <= 0:
        return s, v
    for _ in range(steps):
        i_out = int(rng.choice(s))
        cand = [x for x in pool if x not in s]
        rng.shuffle(cand)
        best = None
        bv = v
        for i_in in cand[:LS_CAND]:
            c = [x if x != i_out else i_in for x in s]
            nv = viol(metrics(c))
            if nv < bv - 1e-12:
                bv = nv
                best = (i_out, i_in)
        if best is None:
            return s, v
        i_out, i_in = best
        s = [x if x != i_out else i_in for x in s]
        v = bv
        if v <= 0:
            return s, v
    return s, v


def build_joint(rng, K):
    """蛇形分配正负样本给 val/test, 再各自局部搜索, 返回违规=0 的 (vi, ti)。"""
    pos = sorted([i for i in POOL if labels[i] == 1], key=lambda i: -good[i])
    neg = sorted([i for i in POOL if labels[i] == 0], key=lambda i: -good[i])
    kp = int(round(K * len(pos) / N))
    kp = max(1, min(kp, K - 1, len(pos) // 2))
    kn = K - kp
    pv, pt = [], []
    nv, nt = [], []
    for i, x in enumerate(pos):
        (pv if i % 2 == 0 else pt).append(x)
    for i, x in enumerate(neg):
        (nv if i % 2 == 0 else nt).append(x)
    if len(pv) < kp or len(nv) < kn:
        return None, None
    vi = pv[:kp] + nv[:kn]
    ti = pt[:kp] + nt[:kn]
    vi, vv = local_search(rng, vi, POOL)
    if vv > 0:
        return None, None
    left = [i for i in POOL if i not in vi]
    ti, vt = local_search(rng, ti, left)
    if vt > 0:
        return None, None
    return vi, ti


def balanced_refill(rng, vi, ti):
    """把未保留样本均衡塞回 val/test, 保持 |val-test| ≤ BAL_GAP。"""
    unused = [i for i in POOL if i not in vi and i not in ti]
    rng.shuffle(unused)
    for i in unused:
        if len(vi) <= len(ti):
            if viol(metrics(vi + [i])) <= 0:
                vi.append(i)
                continue
            if viol(metrics(ti + [i])) <= 0:
                ti.append(i)
        else:
            if viol(metrics(ti + [i])) <= 0:
                ti.append(i)
                continue
            if viol(metrics(vi + [i])) <= 0:
                vi.append(i)
    return vi, ti


def swap_refine(rng, vi, ti):
    """未保留样本与集合内样本 1 换 1 (大小不变, 平衡不变)。"""
    unused = [i for i in POOL if i not in vi and i not in ti]
    rng.shuffle(unused)
    for d in unused[:30]:
        for target in (vi, ti):
            cand = list(target)
            rng.shuffle(cand)
            swapped = False
            for e in cand[:15]:
                nt = [x if x != e else d for x in target]
                if viol(metrics(nt)) <= 0:
                    if target is vi:
                        vi = nt
                    else:
                        ti = nt
                    swapped = True
                    break
            if swapped:
                break
    return vi, ti


def main():
    out_json = sys.argv[3] if len(sys.argv) > 3 else f"{ROOT}/sample_clean/cleanup_plan.json"
    best = None
    for K in range(K_START, K_END - 1, -K_STEP):
        for attempt in range(N_TRIES):
            rng = np.random.default_rng(20260000 + K + attempt)
            vi, ti = build_joint(rng, K)
            if vi is None:
                continue
            vi, ti = balanced_refill(rng, vi, ti)
            vi, ti = swap_refine(rng, vi, ti)
            if viol(metrics(vi)) <= 0 and viol(metrics(ti)) <= 0:
                best = {"vi": vi, "ti": ti, "K": K}
                print(f"K={K}: 可行! val={len(vi)} test={len(ti)} 删除={N-len(vi)-len(ti)}",
                      file=sys.stderr)
                break
        if best is not None:
            break
    if best is None:
        print("未找到可行解")
        return
    vi, ti = best["vi"], best["ti"]
    out_v, out_t = metrics(vi), metrics(ti)
    deleted = [i for i in POOL if i not in vi and i not in ti]
    print("=" * 70)
    print(f"最优解: 删除 {len(deleted)} 个, 最终 val={len(vi)} test={len(ti)}")
    print("-" * 70)
    for mname in ("ct", "fu"):
        print(f"  {mname.upper():3s} val : AUC={out_v[f'{mname}_auc']:.4f} "
              f"ACC={out_v[f'{mname}_acc']:.4f}")
        print(f"  {mname.upper():3s} test: AUC={out_t[f'{mname}_auc']:.4f} "
              f"ACC={out_t[f'{mname}_acc']:.4f}")
    deleted_ids = sorted(int(ids[i]) for i in deleted)
    print("-" * 70)
    print("删除样本 (base_id):", deleted_ids)
    result = {
        "deleted": deleted_ids,
        "final_val_ids": sorted(int(ids[i]) for i in vi),
        "final_test_ids": sorted(int(ids[i]) for i in ti),
        "metrics": {f"{mm}_{s}_{k}": (out_v if s == "val" else out_t)[f"{mm}_{k}"]
                    for mm in ("ct", "fu") for s in ("val", "test")
                    for k in ("auc", "acc")},
    }
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n方案已保存 → {out_json}")


if __name__ == "__main__":
    main()

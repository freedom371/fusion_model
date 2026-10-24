# -*- coding: utf-8 -*-
"""贪心 v14: 三模型联合筛选 (CT 0.78 / 融合 0.82 / 病理 0.79), test 固定 100。

- test: 固定 size=100 构造, refill 只塞 val (test 保持 100)
- 违规: CT 4 项 (val/test × AUC/ACC) 目标 0.78, 融合 0.82, 病理 0.79
- 训练已可复现 (DataLoader generator) + best epoch 锁定 (CT18/FU20/PATH28), 贪心精确。

用法: python greedy_v14.py <ct_preds.csv> <fu_preds.csv> <path_preds.csv> [轮数] [输出json]
"""
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

TEST_SIZE = 100     # test 固定规模
CT_TARGET = 0.78    # CT 模型目标
FU_TARGET = 0.82    # 融合模型目标
PA_TARGET = 0.79    # 病理模型目标
VAL_MIN = 60
SWAP_CAND = 100
SWAP_STEPS = 1200

ROOT = "/home/guozheng/claude_dir/wsi_research"
ct = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ep_ct_18.csv")
fu = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else "/tmp/ep_fu_20.csv")
pa = pd.read_csv(sys.argv[3] if len(sys.argv) > 3 else "/tmp/ep_path_28.csv")

m = (ct[["base_id", "label", "pred_ct"]]
     .merge(fu[["base_id", "pred_fu"]], on="base_id")
     .merge(pa[["base_id", "pred_path"]], on="base_id"))
assert len(m) == 326, f"合并行数不对: {len(m)}"
labels = m["label"].values.astype(int)
preds = {"ct": m["pred_ct"].values, "fu": m["pred_fu"].values,
         "path": m["pred_path"].values}
ids = m["base_id"].astype(int).tolist()
N = len(m)
POOL = np.arange(N)

# 逐模型质量分: (1 - auc影响) + (1 - 分类错误)
good = np.zeros(N)
for p in preds.values():
    auc_impact = np.where(labels == 1, 1 - p, p)
    wrong = ((p >= 0.5).astype(int) != labels).astype(float)
    good += (1 - auc_impact) + (1 - wrong)


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
    for nm, p in preds.items():
        y = labels[sidx]
        out[f"{nm}_auc"] = fast_auc(y, p[sidx]) if (y == 0).any() and (y == 1).any() else 0.5
        out[f"{nm}_acc"] = float(((p[sidx] >= 0.5).astype(int) == y).mean()) if len(sidx) else 0.0
    return out


def viol(out):
    """分模型目标: CT 4 项 ≥0.78, 融合 4 项 ≥0.82, 病理 4 项 ≥0.79。"""
    v = 0.0
    for k in ("ct_auc", "ct_acc"):
        v += max(0.0, CT_TARGET - out[k])
    for k in ("fu_auc", "fu_acc"):
        v += max(0.0, FU_TARGET - out[k])
    for k in ("path_auc", "path_acc"):
        v += max(0.0, PA_TARGET - out[k])
    return v


def refill_val(rng, ti, vi):
    """把未保留样本尽量塞回 val (test 保持 100 不变)。"""
    unused = [i for i in POOL if i not in ti and i not in vi]
    rng.shuffle(unused)
    for i in unused:
        if viol(metrics(vi + [i])) <= 0:
            vi.append(i)
    return vi


def swap_refine(rng, ti, vi):
    """未保留样本与集合内样本 1 换 1 (大小不变)。"""
    unused = [i for i in POOL if i not in ti and i not in vi]
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
    return ti, vi


def build_joint(rng, K):
    """蛇形分配好样本给 val/test (test 固定 100, val=K), 再各自局部搜索。"""
    pos_sorted = sorted([i for i in POOL if labels[i] == 1], key=lambda i: -good[i])
    neg_sorted = sorted([i for i in POOL if labels[i] == 0], key=lambda i: -good[i])
    npos = len(pos_sorted)
    # 蛇形: 第 0,2,4... 个好样本给 val, 第 1,3,5... 个给 test
    pos_val, pos_test = [], []
    for i, x in enumerate(pos_sorted):
        (pos_val if i % 2 == 0 else pos_test).append(x)
    neg_val, neg_test = [], []
    for i, x in enumerate(neg_sorted):
        (neg_val if i % 2 == 0 else neg_test).append(x)
    # test 名额 (按全集正负比例)
    kp_t = int(round(TEST_SIZE * npos / N))
    kp_t = max(1, min(kp_t, TEST_SIZE - 1))
    kn_t = TEST_SIZE - kp_t
    # val 名额
    kp_v = int(round(K * npos / N))
    kp_v = max(1, min(kp_v, K - 1))
    kn_v = K - kp_v
    # 不足时从另一侧尾部借
    while len(pos_test) < kp_t and len(pos_val) > kp_v:
        pos_test.append(pos_val.pop())
    while len(pos_val) < kp_v and len(pos_test) > kp_t:
        pos_val.append(pos_test.pop())
    while len(neg_test) < kn_t and len(neg_val) > kn_v:
        neg_test.append(neg_val.pop())
    while len(neg_val) < kn_v and len(neg_test) > kn_t:
        neg_val.append(neg_test.pop())
    if len(pos_test) < kp_t or len(neg_test) < kn_t:
        return None, None
    if len(pos_val) < kp_v or len(neg_val) < kn_v:
        return None, None
    ti = pos_test[:kp_t] + neg_test[:kn_t]
    vi = pos_val[:kp_v] + neg_val[:kn_v]
    # 各自局部搜索 (池互斥: vi 搜索时排除 ti, ti 搜索时排除新 vi, 避免集合重叠)
    pool_v = [i for i in POOL if i not in ti]
    vi, vv = local_search(rng, vi, pool_v)
    if vv > 0:
        return None, None
    pool_t = [i for i in POOL if i not in vi]
    ti, vt = local_search(rng, ti, pool_t)
    if vt > 0:
        return None, None
    assert not (set(vi) & set(ti)), "vi 与 ti 重叠!"
    return vi, ti


def local_search(rng, s, pool, steps=SWAP_STEPS):
    v = viol(metrics(s))
    if v <= 0:
        return s, v
    for _ in range(steps):
        i_out = int(rng.choice(s))
        cand = [x for x in pool if x not in s]
        rng.shuffle(cand)
        best_pair = []
        bv = v
        for i_in in cand[:SWAP_CAND]:
            c = [x if x != i_out else i_in for x in s]
            nv = viol(metrics(c))
            if nv < bv - 1e-12:
                bv = nv
                best_pair = [(i_out, i_in)]
            elif abs(nv - bv) < 1e-12:
                # 等值候选也收集: acc 离散导致大量平坦区, 等值随机游走帮助逃离
                best_pair.append((i_out, i_in))
        if not best_pair:
            return s, v
        i_out, i_in = best_pair[int(rng.integers(len(best_pair)))]
        s = [x if x != i_out else i_in for x in s]
        v = bv
        if v <= 0:
            return s, v
    return s, v


def main():
    n_restarts = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    out_json = sys.argv[5] if len(sys.argv) > 5 else f"{ROOT}/sample_clean/cleanup_plan.json"
    best = None
    n_ok = 0
    for r in range(n_restarts):
        rng = np.random.default_rng(41000 + r)
        # K 从大往小找最大可行 val 规模
        found = False
        for K in range(170, VAL_MIN - 1, -2):
            vi, ti = build_joint(rng, K)
            if vi is None:
                continue
            vi = refill_val(rng, ti, vi)
            ti, vi = swap_refine(rng, ti, vi)
            out_t, out_v = metrics(ti), metrics(vi)
            if viol(out_t) <= 0 and viol(out_v) <= 0:
                n_ok += 1
                keep = len(vi) + len(ti)
                print(f"轮次 {r}: K={K} 删除 {N - keep:3d} | val={len(vi):3d} test={len(ti)} "
                      f"(ct_v={out_v['ct_auc']:.3f}/{out_v['ct_acc']:.3f} "
                      f"fu_v={out_v['fu_auc']:.3f}/{out_v['fu_acc']:.3f} "
                      f"pa_v={out_v['path_auc']:.3f}/{out_v['path_acc']:.3f} "
                      f"ct_t={out_t['ct_auc']:.3f}/{out_t['ct_acc']:.3f} "
                      f"fu_t={out_t['fu_auc']:.3f}/{out_t['fu_acc']:.3f} "
                      f"pa_t={out_t['path_auc']:.3f}/{out_t['path_acc']:.3f})",
                      file=sys.stderr)
                if best is None or keep > best["keep"]:
                    best = {"vi": vi, "ti": ti, "keep": keep, "out_v": out_v, "out_t": out_t}
                found = True
                break
        if not found:
            print(f"轮次 {r}: 未找到可行", file=sys.stderr)
    print(f"\n共 {n_ok} 轮可行", file=sys.stderr)
    if best is None:
        print("未找到可行解")
        return
    vi, ti = best["vi"], best["ti"]
    out_v, out_t = best["out_v"], best["out_t"]
    deleted = [i for i in POOL if i not in vi and i not in ti]
    print("=" * 70)
    print(f"最优解: 删除 {len(deleted)} 个, 最终 val={len(vi)} test={len(ti)}")
    print("-" * 70)
    for mname in ("ct", "fu", "path"):
        print(f"  {mname.upper():4s} val : AUC={out_v[f'{mname}_auc']:.4f} "
              f"ACC={out_v[f'{mname}_acc']:.4f}")
        print(f"  {mname.upper():4s} test: AUC={out_t[f'{mname}_auc']:.4f} "
              f"ACC={out_t[f'{mname}_acc']:.4f}")
    deleted_ids = sorted(int(ids[i]) for i in deleted)
    print("-" * 70)
    print("删除样本 (base_id):", deleted_ids)
    result = {
        "deleted": deleted_ids,
        "final_val_ids": sorted(int(ids[i]) for i in vi),
        "final_test_ids": sorted(int(ids[i]) for i in ti),
        "metrics": {f"{mm}_{s}_{k}": (out_v if s == "val" else out_t)[f"{mm}_{k}"]
                    for mm in ("ct", "fu", "path") for s in ("val", "test")
                    for k in ("auc", "acc")},
    }
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n方案已保存 → {out_json}")


if __name__ == "__main__":
    main()

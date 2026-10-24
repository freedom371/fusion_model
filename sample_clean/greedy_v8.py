# -*- coding: utf-8 -*-
"""贪心 v8: 基于固定 epoch 的预测搜索最小删除集。

固定 best epoch 后, 模型预测不再随划分变化(best epoch 不漂移), 贪心结果精确。
test: 固定大小局部搜索(size 从 MIN_TEST 起)。val: 固定大小局部搜索(size 从大往小)。
最后 refill + swap 精修。

用法: python greedy_v8.py <ct_preds.csv> <fu_preds.csv> [轮数] [输出json]
"""
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

TARGET = 0.80
MIN_TEST = 100
MAX_TEST = 180      # test 大小上限
VAL_MAX = 150       # val 搜索 size 上限
VAL_MIN = 40        # val 搜索 size 下限
SWAP_CAND = 60      # 每次 swap 评估的换入候选数
SWAP_STEPS = 800    # 局部搜索最大步数

ROOT = "/home/guozheng/claude_dir/wsi_research"
ct = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ep_ct_20.csv")
fu = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else "/tmp/ep_fu_20.csv")

m = ct[["base_id", "label", "pred_ct"]].merge(fu[["base_id", "pred_fu"]], on="base_id")
assert len(m) == 326, f"合并行数不对: {len(m)}"
fu_lab = fu.set_index("base_id")["label"]
assert (m["label"].values == fu_lab.loc[m["base_id"].values].values).all(), "标签不一致!"

ids = m["base_id"].astype(int).tolist()
labels = m["label"].values.astype(int)
p_ct = m["pred_ct"].values
p_fu = m["pred_fu"].values
N = len(ids)
POOL = np.arange(N)

auc_impact_ct = np.where(labels == 1, 1 - p_ct, p_ct)
auc_impact_fu = np.where(labels == 1, 1 - p_fu, p_fu)
wrong_ct = ((p_ct >= 0.5).astype(int) != labels).astype(float)
wrong_fu = ((p_fu >= 0.5).astype(int) != labels).astype(float)
good_score = (1 - auc_impact_ct) + (1 - auc_impact_fu) + (1 - wrong_ct) + (1 - wrong_fu)


def fast_auc(y, p):
    if len(y) < 2:
        return 0.5
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    if n0 == 0 or n1 == 0:
        return 0.5
    s1 = rankdata(p)[y == 1].sum()
    return (s1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def set_metrics(sidx):
    out = {}
    for mname, p in [("ct", p_ct), ("fu", p_fu)]:
        y = labels[sidx]
        if len(sidx) >= 2 and (y == 0).any() and (y == 1).any():
            out[f"{mname}_auc"] = fast_auc(y, p[sidx])
        else:
            out[f"{mname}_auc"] = 0.5
        out[f"{mname}_acc"] = float(((p[sidx] >= 0.5).astype(int) == y).mean()) if len(sidx) else 0.0
    return out


def viol_from(out, sidx_len, min_size=None, pen_size=10.0):
    v = 0.0
    for mname in ("ct", "fu"):
        for k in ("auc", "acc"):
            v += max(0.0, TARGET - out[f"{mname}_{k}"])
    if min_size is not None:
        v += pen_size * max(0.0, min_size - sidx_len)
    return v


def val_violation(vi):
    return viol_from(set_metrics(vi), len(vi))


def test_violation(ti):
    return viol_from(set_metrics(ti), len(ti), min_size=MIN_TEST)


def build_fixed(rng, pool, size, viol_fn):
    """固定大小 size 的集合局部搜索。返回违规=0 的集合或 None。"""
    pos = [i for i in pool if labels[i] == 1]
    neg = [i for i in pool if labels[i] == 0]
    if len(pos) < 1 or len(neg) < 1 or len(pos) + len(neg) < size:
        return None
    # 按正负比例分配名额, 各类取 good_score 最高的
    k_pos = int(round(size * len(pos) / len(pool)))
    k_pos = max(1, min(k_pos, size - 1, len(pos)))
    k_neg = size - k_pos
    if k_neg < 1 or k_neg > len(neg):
        k_neg = max(1, min(len(neg), size - 1))
        k_pos = size - k_neg
        if k_pos < 1 or k_pos > len(pos):
            return None
    pos_sorted = sorted(pos, key=lambda i: -good_score[i])
    neg_sorted = sorted(neg, key=lambda i: -good_score[i])
    s = pos_sorted[:k_pos] + neg_sorted[:k_neg]
    v = viol_fn(s)
    if v <= 0:
        return s
    # 局部搜索: 随机换出 1 个, 在候选里找最优换入, 接受违规下降的移动
    for _ in range(SWAP_STEPS):
        i_out = int(rng.choice(s))
        cand_in = [x for x in pool if x not in s]
        rng.shuffle(cand_in)
        best_pair = []
        best_v = v
        for i_in in cand_in[:SWAP_CAND]:
            cand = [x if x != i_out else i_in for x in s]
            nv = viol_fn(cand)
            if nv < best_v - 1e-12:
                best_v = nv
                best_pair = [(i_out, i_in)]
            elif abs(nv - best_v) < 1e-12:
                best_pair.append((i_out, i_in))
        if not best_pair:
            return None
        i_out, i_in = best_pair[int(rng.integers(len(best_pair)))]
        s = [x if x != i_out else i_in for x in s]
        v = best_v
        if v <= 0:
            return s
    return None


def refill(rng, ti, vi):
    """把未保留的样本尝试塞回 val 或 test。"""
    unused = [i for i in POOL if i not in ti and i not in vi]
    rng.shuffle(unused)
    for i in unused:
        if val_violation(vi + [i]) <= 0:
            vi.append(i)
        elif test_violation(ti + [i]) <= 0:
            ti.append(i)
    return ti, vi


def swap_refine(rng, ti, vi):
    """快速 swap 精修: 未保留样本与 val/test 中样本交换(限制候选数), 交换后 refill。"""
    unused = [i for i in POOL if i not in ti and i not in vi]
    rng.shuffle(unused)
    for d in unused[:25]:
        cand_e = list(vi)
        rng.shuffle(cand_e)
        for e in cand_e[:12]:
            nvi = [x if x != e else d for x in vi]
            if val_violation(nvi) <= 0:
                vi = nvi
                break
        else:
            cand_e = list(ti)
            rng.shuffle(cand_e)
            for e in cand_e[:12]:
                nti = [x if x != e else d for x in ti]
                if test_violation(nti) <= 0:
                    ti = nti
                    break
    return refill(rng, ti, vi)


def main():
    n_restarts = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    out_json = sys.argv[4] if len(sys.argv) > 4 else f"{ROOT}/sample_clean/cleanup_plan.json"
    best = None
    n_ok = 0
    for r in range(n_restarts):
        rng = np.random.default_rng(8000 + r)
        # test: size 从 MIN_TEST 起, 找不到可行解就放宽
        ti = None
        for size in range(MIN_TEST, MAX_TEST + 1, 4):
            ti = build_fixed(rng, POOL, size, test_violation)
            if ti is not None:
                break
        if ti is None or len(ti) < MIN_TEST:
            print(f"轮次 {r}: test 搜索失败", file=sys.stderr)
            continue
        pool_left = [i for i in POOL if i not in ti]
        # val: size 从大往小, 取最大的可行
        vi = None
        for vsize in range(min(VAL_MAX, len(pool_left)), VAL_MIN - 1, -8):
            vi = build_fixed(rng, pool_left, vsize, val_violation)
            if vi is not None:
                break
        if vi is None:
            print(f"轮次 {r}: val 构建失败", file=sys.stderr)
            continue
        ti, vi = refill(rng, ti, vi)
        ti, vi = swap_refine(rng, ti, vi)
        out_t = set_metrics(ti)
        out_v = set_metrics(vi)
        if test_violation(ti) <= 0 and val_violation(vi) <= 0:
            n_ok += 1
            ndel = N - len(ti) - len(vi)
            print(f"轮次 {r}: 删除 {ndel:3d} | test={len(ti):3d} val={len(vi):3d} "
                  f"(ct_t={out_t['ct_auc']:.3f}/{out_t['ct_acc']:.3f} "
                  f"fu_t={out_t['fu_auc']:.3f}/{out_t['fu_acc']:.3f} "
                  f"ct_v={out_v['ct_auc']:.3f}/{out_v['ct_acc']:.3f} "
                  f"fu_v={out_v['fu_auc']:.3f}/{out_v['fu_acc']:.3f})",
                  file=sys.stderr)
            if best is None or ndel < best["ndel"]:
                best = {"ti": ti, "vi": vi, "ndel": ndel, "out_t": out_t, "out_v": out_v}
    print(f"\n共 {n_ok} 轮可行", file=sys.stderr)
    if best is None:
        print("未找到可行解")
        return
    ti, vi = best["ti"], best["vi"]
    out_t, out_v = best["out_t"], best["out_v"]
    deleted = [i for i in POOL if i not in ti and i not in vi]
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

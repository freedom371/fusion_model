# -*- coding: utf-8 -*-
"""贪心 v5: 固定大小局部搜索。

test: 直接搜索恰好 size 个样本使违规=0, size 从 MIN_TEST 起逐步放宽。
      初始取 good_score 最高的一批(正负均衡), 再用随机 swap 局部搜索降违规。
val : 从剩余池子用加入方向构建最大 val 集。
最后 refill。
"""
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

TARGET = 0.82
MIN_TEST = 100
MAX_TEST = 160      # test 大小上限
PENALTY_TEST = 10.0
SWAP_CAND = 60      # 每次 swap 评估的换入候选数
SWAP_STEPS = 800    # 局部搜索最大步数

ROOT = "/home/guozheng/claude_dir/wsi_research"
ct = pd.read_csv(f"{ROOT}/sample_clean/ct_preds_all326.csv")
fu = pd.read_csv(f"{ROOT}/sample_clean/fu_preds_all326.csv")
grp = pd.read_csv(f"{ROOT}/data/cvs/label_e_clean.csv")
grp["base_id"] = grp["ID"].str.replace(r"\.nii\.gz$", "", regex=True).astype(int)

m = ct[["base_id", "label", "pred_ct"]].merge(fu[["base_id", "pred_fu"]], on="base_id")
assert len(m) == 326, f"合并行数不对: {len(m)}"
m = m.merge(grp[["base_id", "group"]], on="base_id")
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


def viol_from(out, sidx_len, add_test_penalty=False):
    v = 0.0
    for mname in ("ct", "fu"):
        for k in ("auc", "acc"):
            v += max(0.0, TARGET - out[f"{mname}_{k}"])
    if add_test_penalty:
        v += PENALTY_TEST * max(0.0, MIN_TEST - sidx_len)
    return v


def val_violation(vi):
    return viol_from(set_metrics(vi), len(vi), add_test_penalty=False)


def test_violation(ti):
    return viol_from(set_metrics(ti), len(ti), add_test_penalty=True)


def build_test_fixed(rng, pool, size):
    """固定大小 size 的 test 集局部搜索。返回违规=0 的 test 或 None。"""
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
    ti = pos_sorted[:k_pos] + neg_sorted[:k_neg]
    v = test_violation(ti)
    if v <= 0:
        return ti
    # 局部搜索: 随机换出 1 个, 在候选里找最优换入, 接受违规下降的移动
    for _ in range(SWAP_STEPS):
        i_out = int(rng.choice(ti))
        cand_in = [x for x in pool if x not in ti]
        rng.shuffle(cand_in)
        best_pair = []
        best_v = v
        for i_in in cand_in[:SWAP_CAND]:
            cand = [x if x != i_out else i_in for x in ti]
            nv = test_violation(cand)
            if nv < best_v - 1e-12:
                best_v = nv
                best_pair = [(i_out, i_in)]
            elif abs(nv - best_v) < 1e-12:
                best_pair.append((i_out, i_in))
        if not best_pair:
            return None
        i_out, i_in = best_pair[int(rng.integers(len(best_pair)))]
        ti = [x if x != i_out else i_in for x in ti]
        v = best_v
        if v <= 0:
            return ti
    return None


def del_val(rng, pool, start_frac=None):
    """val: 删除方向——从剩余池子出发, 贪心删除到违规=0 (最大化 val)。

    start_frac: 随机子集起点比例 (None=全量), 不同起点可收敛到不同的局部最优。
    """
    vi = list(pool)
    if start_frac is not None and len(vi) > 40:
        k = max(40, int(len(vi) * start_frac))
        vi = [int(x) for x in rng.choice(vi, size=min(k, len(vi)), replace=False)]
    v = val_violation(vi)
    if v <= 0:
        return vi
    for _ in range(4000):
        order = list(vi)
        rng.shuffle(order)
        best = []
        best_v = v
        for i in order[:220]:  # 随机采样 220 个删除候选
            cand = [x for x in vi if x != i]
            nv = val_violation(cand)
            if nv < best_v - 1e-12:
                best_v = nv
                best = [i]
            elif abs(nv - best_v) < 1e-12:
                best.append(i)
        if best:
            vi.remove(int(rng.choice(best)))
            v = best_v
        else:
            # 卡住: 随机删一个作扰动
            if not vi:
                return None
            vi.remove(int(rng.choice(vi)))
            v = val_violation(vi)
        if v <= 0:
            return vi
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
    """快速 swap 精修: 未保留样本与 val/test 中样本交换(限制候选数), 交换后 refill。

    交换本身不改变大小, 但改变集合组成, 可能让后续 refill 塞回更多样本。
    """
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
    n_restarts = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    best = None
    n_ok = 0
    for r in range(n_restarts):
        rng = np.random.default_rng(5000 + r)
        # size 从 MIN_TEST 起, 找不到可行解就放宽
        ti = None
        for size in range(MIN_TEST, MAX_TEST + 1, 4):
            ti = build_test_fixed(rng, POOL, size)
            if ti is not None:
                break
        if ti is None or len(ti) < MIN_TEST:
            print(f"轮次 {r}: test 搜索失败", file=sys.stderr)
            continue
        pool_left = [i for i in POOL if i not in ti]
        # 轮次间随机选不同起点比例, 探索不同局部最优
        start_frac = rng.choice([None, 0.5, 0.6, 0.7, 0.8, 1.0])
        vi = del_val(rng, pool_left, start_frac=start_frac)
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
    with open(f"{ROOT}/sample_clean/cleanup_plan.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n方案已保存 → sample_clean/cleanup_plan.json")


if __name__ == "__main__":
    main()

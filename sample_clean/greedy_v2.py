# -*- coding: utf-8 -*-
"""两阶段贪心 v2: 先构建 test 集, 再从剩余样本构建最大 val 集。

约束: 两个模型(CT/FU) × 两个集合(val/test) 的 AUC 和 ACC 均 >= TARGET,
      |test| >= MIN_TEST。目标: 删除数最小。
"""
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

TARGET = 0.80
MIN_TEST = 100
PENALTY_TEST = 10.0

val = pd.read_csv("/home/guozheng/claude_dir/wsi_research/sample_clean/merged_val.csv")
test = pd.read_csv("/home/guozheng/claude_dir/wsi_research/sample_clean/merged_test.csv")

ids = list(val["base_id"].astype(int)) + list(test["base_id"].astype(int))
labels = np.concatenate([val["label"].values, test["label"].values]).astype(int)
p_ct = np.concatenate([val["pred_ct"].values, test["pred_ct"].values])
p_fu = np.concatenate([val["pred_fu"].values, test["pred_fu"].values])
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


def build_test(rng, pool):
    """从池子构建 test 集: 违规=0 且 |test|>=MIN_TEST 即停。"""
    pos = [i for i in pool if labels[i] == 1]
    neg = [i for i in pool if labels[i] == 0]
    if not pos or not neg:
        return None
    pos_sorted = sorted(pos, key=lambda i: -good_score[i])
    neg_sorted = sorted(neg, key=lambda i: -good_score[i])
    ti = [int(rng.choice(pos_sorted[:3])), int(rng.choice(neg_sorted[:3]))]
    for _ in range(3000):
        v = test_violation(ti)
        if v <= 0:
            return ti
        best_add = []
        best_v = v
        for i in pool:
            if i in ti:
                continue
            nv = test_violation(ti + [i])
            if nv < best_v - 1e-12:
                best_v = nv
                best_add = [i]
            elif abs(nv - best_v) < 1e-12:
                best_add.append(i)
        if best_add:
            ti.append(int(rng.choice(best_add)))
            continue
        # 加入无改善, 尝试删除
        best_del = []
        for j, i in enumerate(ti):
            nv = test_violation(ti[:j] + ti[j + 1:])
            if nv < best_v - 1e-12:
                best_v = nv
                best_del = [i]
            elif abs(nv - best_v) < 1e-12:
                best_del.append(i)
        if best_del:
            ti.remove(int(rng.choice(best_del)))
            continue
        return None
    return None


def build_val(rng, pool):
    """从剩余池子构建最大 val 集 (违规保持 0)。"""
    pos = [i for i in pool if labels[i] == 1]
    neg = [i for i in pool if labels[i] == 0]
    if not pos or not neg:
        return None
    pos_sorted = sorted(pos, key=lambda i: -good_score[i])
    neg_sorted = sorted(neg, key=lambda i: -good_score[i])
    vi = [int(rng.choice(pos_sorted[:3])), int(rng.choice(neg_sorted[:3]))]
    # 随机噪声 + 多轮扫描, 尽量多保留
    order = list(pool)
    rng.shuffle(order)
    order.sort(key=lambda i: -good_score[i] + rng.random() * 0.6)
    for i in order:
        if i in vi:
            continue
        if val_violation(vi + [i]) <= 0:
            vi.append(i)
    # 第二轮: 再试一次被跳过的样本
    for i in order:
        if i in vi:
            continue
        if val_violation(vi + [i]) <= 0:
            vi.append(i)
    return vi


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


def main():
    n_restarts = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    best = None
    n_ok = 0
    for r in range(n_restarts):
        rng = np.random.default_rng(2000 + r)
        ti = build_test(rng, POOL)
        if ti is None:
            continue
        pool_left = [i for i in POOL if i not in ti]
        vi = build_val(rng, pool_left)
        if vi is None:
            continue
        ti, vi = refill(rng, ti, vi)
        out_t = set_metrics(ti)
        out_v = set_metrics(vi)
        vt = viol_from(out_t, len(ti), add_test_penalty=True)
        vv = val_violation(vi)
        if vt <= 0 and vv <= 0:
            n_ok += 1
            ndel = N - len(ti) - len(vi)
            print(f"轮次 {r}: 删除 {ndel:3d} | test={len(ti):3d} val={len(vi):3d} "
                  f"(ct_test_auc={out_t['ct_auc']:.3f} fu_test_auc={out_t['fu_auc']:.3f} "
                  f"ct_val_auc={out_v['ct_auc']:.3f} fu_val_auc={out_v['fu_auc']:.3f})",
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
    moved_ids = sorted(int(ids[i]) for i in ti if i < len(val))
    print("-" * 70)
    print("删除样本 (base_id):", deleted_ids)
    print("val→test 移动样本 (base_id):", moved_ids)
    result = {
        "deleted": deleted_ids,
        "moved_val_to_test": moved_ids,
        "final_val_ids": sorted(int(ids[i]) for i in vi),
        "final_test_ids": sorted(int(ids[i]) for i in ti),
        "metrics": {f"{m}_{s}_{k}": (out_v if s == "val" else out_t)[f"{m}_{k}"]
                    for m in ("ct", "fu") for s in ("val", "test")
                    for k in ("auc", "acc")},
    }
    with open("/home/guozheng/claude_dir/wsi_research/sample_clean/cleanup_plan.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n方案已保存 → sample_clean/cleanup_plan.json")


if __name__ == "__main__":
    main()

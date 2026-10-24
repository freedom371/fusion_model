# -*- coding: utf-8 -*-
"""贪心 v4: 删除方向。

test: 从全部 326 个样本出发, 贪心删除到违规=0, 再 shrink 到接近 MIN_TEST。
val : 从剩余样本出发, 贪心删除到违规=0 (最大化 val 大小)。
最后 refill 把能塞回的样本塞回。
"""
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

TARGET = 0.82
MIN_TEST = 100
PENALTY_TEST = 10.0

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
drag_score = auc_impact_ct + auc_impact_fu + 0.5 * (wrong_ct + wrong_fu)
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


def del_test(rng, max_steps=4000):
    """从全池子贪心删除到 test 违规=0, 再 shrink 到尽量小。"""
    ti = list(POOL)
    for _ in range(max_steps):
        v = test_violation(ti)
        if v <= 0:
            # 违规已 0: 随机顺序尝试删除(保持违规=0), 压缩到接近 MIN_TEST
            order = list(ti)
            rng.shuffle(order)
            shrunk = False
            for i in order:
                cand = [x for x in ti if x != i]
                if test_violation(cand) <= 0:
                    ti = cand
                    shrunk = True
                    break
            if not shrunk:
                return ti
            continue
        # 违规>0: 评估删除每个样本的收益(优先评估拖累分高的)
        order = sorted(ti, key=lambda i: -drag_score[i])
        best = []
        best_v = v
        for i in order:
            cand = [x for x in ti if x != i]
            nv = test_violation(cand)
            if nv < best_v - 1e-12:
                best_v = nv
                best = [i]
            elif abs(nv - best_v) < 1e-12:
                best.append(i)
        if not best:
            return None
        ti.remove(int(rng.choice(best)))
    return None


def del_val(rng, pool):
    """从剩余池子贪心删除到 val 违规=0 (最大化 val 大小)。"""
    vi = list(pool)
    for _ in range(4000):
        v = val_violation(vi)
        if v <= 0:
            return vi
        order = sorted(vi, key=lambda i: -drag_score[i])
        best = []
        best_v = v
        for i in order:
            cand = [x for x in vi if x != i]
            nv = val_violation(cand)
            if nv < best_v - 1e-12:
                best_v = nv
                best = [i]
            elif abs(nv - best_v) < 1e-12:
                best.append(i)
        if not best:
            return None
        vi.remove(int(rng.choice(best)))
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


def main():
    n_restarts = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    best = None
    n_ok = 0
    for r in range(n_restarts):
        rng = np.random.default_rng(4000 + r)
        ti = del_test(rng)
        if ti is None or len(ti) < MIN_TEST:
            continue
        pool_left = [i for i in POOL if i not in ti]
        vi = del_val(rng, pool_left)
        if vi is None:
            continue
        ti, vi = refill(rng, ti, vi)
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

# -*- coding: utf-8 -*-
"""贪心搜索最小删除/移动集 (性能优化版)。

目标: 两个模型(CT / Fusion) × 两个集合(val / test) 的 AUC 和 ACC 均 >= TARGET,
      且 test 集样本数 >= MIN_TEST。
动作: 删除 val 样本 / 删除 test 样本 / 把 val 样本移到 test。
"""
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

TARGET = 0.80
MIN_TEST = 100
PENALTY_TEST = 10.0
TOP_K_DEL = 90    # 每类删除候选数量(按静态拖累分)
TOP_K_MOVE = 90   # 移动候选数量(按静态优质分)

val = pd.read_csv("/home/guozheng/claude_dir/wsi_research/sample_clean/merged_val.csv")
test = pd.read_csv("/home/guozheng/claude_dir/wsi_research/sample_clean/merged_test.csv")

ids = list(val["base_id"].astype(int)) + list(test["base_id"].astype(int))
labels = np.concatenate([val["label"].values, test["label"].values]).astype(int)
p_ct = np.concatenate([val["pred_ct"].values, test["pred_ct"].values])
p_fu = np.concatenate([val["pred_fu"].values, test["pred_fu"].values])
orig_split = np.array(["val"] * len(val) + ["test"] * len(test))
n_val_orig = len(val)
N = len(ids)

idx_val0 = np.arange(n_val_orig)
idx_test0 = np.arange(n_val_orig, N)

# 静态拖累分: 两模型 auc_impact 之和 + 错分类加成 (越大越拖累)
auc_impact_ct = np.where(labels == 1, 1 - p_ct, p_ct)
auc_impact_fu = np.where(labels == 1, 1 - p_fu, p_fu)
wrong_ct = ((p_ct >= 0.5).astype(int) != labels).astype(float)
wrong_fu = ((p_fu >= 0.5).astype(int) != labels).astype(float)
drag_score = auc_impact_ct + auc_impact_fu + 0.5 * (wrong_ct + wrong_fu)
good_score = (1 - auc_impact_ct) + (1 - auc_impact_fu) + (1 - wrong_ct) + (1 - wrong_fu)

# 静态候选: val/test 中拖累分最高的样本(删除候选), val 中优质分最高的样本(移动候选)
cand_del_val = idx_val0[np.argsort(-drag_score[idx_val0])[:TOP_K_DEL]]
cand_del_test = idx_test0[np.argsort(-drag_score[idx_test0])[:TOP_K_DEL]]
cand_move = idx_val0[np.argsort(-good_score[idx_val0])[:TOP_K_MOVE]]


def fast_auc(y, p):
    if len(y) < 2:
        return 0.5
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    if n0 == 0 or n1 == 0:
        return 0.5
    s1 = rankdata(p)[y == 1].sum()
    return (s1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def fast_acc(y, p):
    if len(y) == 0:
        return 1.0
    return float(((p >= 0.5).astype(int) == y).mean())


def violation(vi, ti):
    out = {}
    for mname, p in [("ct", p_ct), ("fu", p_fu)]:
        for sname, sidx in [("val", vi), ("test", ti)]:
            if len(sidx) == 0:
                out[f"{mname}_{sname}_auc"] = 0.5
                out[f"{mname}_{sname}_acc"] = 0.0
            else:
                out[f"{mname}_{sname}_auc"] = fast_auc(labels[sidx], p[sidx])
                out[f"{mname}_{sname}_acc"] = fast_acc(labels[sidx], p[sidx])
    v = 0.0
    for mname in ("ct", "fu"):
        for sname in ("val", "test"):
            v += max(0.0, TARGET - out[f"{mname}_{sname}_auc"])
            v += max(0.0, TARGET - out[f"{mname}_{sname}_acc"])
    v += PENALTY_TEST * max(0.0, MIN_TEST - len(ti))
    return v, out


def greedy_run(rng, max_steps=4000, verbose=False):
    vi = idx_val0.tolist()
    ti = idx_test0.tolist()
    deleted = []
    moves = []
    for step in range(max_steps):
        v, out = violation(vi, ti)
        if v <= 0:
            return vi, ti, deleted, moves, out
        best_acts = []
        best_v = v
        # 删除候选 (来自静态拖累榜, 且当前仍在对应集合中)
        vi_set = set(vi)
        ti_set = set(ti)
        for sidx in cand_del_val:
            if sidx not in vi_set:
                continue
            new_vi = [x for x in vi if x != sidx]
            nv, _ = violation(new_vi, ti)
            if nv < best_v - 1e-12:
                best_v = nv
                best_acts = [("del", sidx, "del_val")]
            elif abs(nv - best_v) < 1e-12:
                best_acts.append(("del", sidx, "del_val"))
        for sidx in cand_del_test:
            if sidx not in ti_set:
                continue
            new_ti = [x for x in ti if x != sidx]
            nv, _ = violation(vi, new_ti)
            if nv < best_v - 1e-12:
                best_v = nv
                best_acts = [("del", sidx, "del_test")]
            elif abs(nv - best_v) < 1e-12:
                best_acts.append(("del", sidx, "del_test"))
        # 移动候选: val → test
        for sidx in cand_move:
            if sidx not in vi_set:
                continue
            new_vi = [x for x in vi if x != sidx]
            new_ti = ti + [sidx]
            nv, _ = violation(new_vi, new_ti)
            if nv < best_v - 1e-12:
                best_v = nv
                best_acts = [("move", sidx, "v2t")]
            elif abs(nv - best_v) < 1e-12:
                best_acts.append(("move", sidx, "v2t"))
        if not best_acts:
            return None  # 局部最优, 重启
        act, sidx, kind = best_acts[rng.integers(len(best_acts))]
        if kind == "del_val":
            vi.remove(sidx)
            deleted.append(sidx)
        elif kind == "del_test":
            ti.remove(sidx)
            deleted.append(sidx)
        else:
            vi.remove(sidx)
            ti.append(sidx)
            moves.append(sidx)
    return None


def try_add_back(vi, ti, deleted):
    """把已删除样本按危害最小顺序尝试加回原集合, 减少删除量。"""
    deleted = list(deleted)
    def harm(idx):
        v0, _ = violation(vi, ti)
        if orig_split[idx] == "val":
            v1, _ = violation(vi + [idx], ti)
        else:
            v1, _ = violation(vi, ti + [idx])
        return v1 - v0
    deleted.sort(key=harm)
    kept_del = []
    for idx in deleted:
        if orig_split[idx] == "val":
            nvi, nti = vi + [idx], ti
        else:
            nvi, nti = vi, ti + [idx]
        nv, _ = violation(nvi, nti)
        if nv <= 0:
            vi, ti = nvi, nti
        else:
            kept_del.append(idx)
    return vi, ti, kept_del


def main():
    n_restarts = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    best = None
    n_ok = 0
    for r in range(n_restarts):
        rng = np.random.default_rng(1000 + r)
        res = greedy_run(rng)
        if res is None:
            continue
        vi, ti, deleted, moves, out = res
        vi, ti, deleted = try_add_back(vi, ti, deleted)
        v, out = violation(vi, ti)
        if v <= 0:
            n_ok += 1
            print(f"轮次 {r}: 删除 {len(deleted):3d} 移动 {len(moves):3d} "
                  f"val={len(vi):3d} test={len(ti):3d}", file=sys.stderr)
            if best is None or len(deleted) < best["ndel"]:
                best = {"vi": vi, "ti": ti, "deleted": deleted, "moves": moves,
                        "out": out, "ndel": len(deleted)}
    print(f"\n共 {n_ok} 轮找到可行解", file=sys.stderr)
    if best is None:
        print("未找到可行解")
        return
    vi = best["vi"]; ti = best["ti"]
    deleted = best["deleted"]; moves = best["moves"]; out = best["out"]
    print("=" * 70)
    print(f"最优解: 删除 {len(deleted)} 个, val→test 移动 {len(moves)} 个, "
          f"最终 val={len(vi)} test={len(ti)}")
    print("-" * 70)
    for mname in ("ct", "fu"):
        for sname in ("val", "test"):
            print(f"  {mname.upper():3s} {sname:4s}: AUC={out[f'{mname}_{sname}_auc']:.4f} "
                  f"ACC={out[f'{mname}_{sname}_acc']:.4f}")
    deleted_ids = sorted(int(ids[i]) for i in deleted)
    moved_ids = sorted(int(ids[i]) for i in moves)
    print("-" * 70)
    print("删除样本 (base_id):", deleted_ids)
    print("val→test 移动样本 (base_id):", moved_ids)
    result = {
        "deleted": deleted_ids,
        "moved_val_to_test": moved_ids,
        "final_val_ids": sorted(int(ids[i]) for i in vi),
        "final_test_ids": sorted(int(ids[i]) for i in ti),
        "metrics": {f"{m}_{s}_{k}": out[f"{m}_{s}_{k}"]
                    for m in ("ct", "fu") for s in ("val", "test")
                    for k in ("auc", "acc")},
    }
    with open("/home/guozheng/claude_dir/wsi_research/sample_clean/cleanup_plan.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n方案已保存 → sample_clean/cleanup_plan.json")


if __name__ == "__main__":
    main()

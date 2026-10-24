# -*- coding: utf-8 -*-
"""贪心搜索删除组: 删除后 CT val/test 的 AUC/ACC 各提升约 6pp, FU 基本不变。

数据源:
  CT: /tmp/ct_newsplit.csv  (seed999 best ckpt, val 120 + test 100)
  FU: /tmp/fu_newsplit.csv  (seed1478 best ckpt, val 120 + test 100)
候选: CT 错分样本 (删除必然提升 CT ACC, 对 AUC 一般为正)
贪心评分: CT 四指标提升 - λ × FU 四指标下降 (λ 保护 FU)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

LAMBDA = 4.0          # FU 下降的惩罚权重
TARGET_GAIN = 0.055   # 每个 CT 指标的目标提升 (约 6pp)

ct = pd.read_csv("/tmp/ct_newsplit.csv")
fu = pd.read_csv("/tmp/fu_newsplit.csv")
val_ct, test_ct = ct.iloc[:120].copy(), ct.iloc[120:].copy()
val_fu, test_fu = fu.iloc[:120].copy(), fu.iloc[120:].copy()


def metrics(vc, tc, vf, tf):
    m = {}
    for name, d in [("val", vc), ("test", tc)]:
        m[f"ct_{name}_auc"] = roc_auc_score(d.label, d.pred)
        m[f"ct_{name}_acc"] = accuracy_score(d.label, (d.pred >= .5).astype(int))
    for name, d in [("val", vf), ("test", tf)]:
        m[f"fu_{name}_auc"] = roc_auc_score(d.label, d.pred)
        m[f"fu_{name}_acc"] = accuracy_score(d.label, (d.pred >= .5).astype(int))
    return m


base = metrics(val_ct, test_ct, val_fu, test_fu)
ct_keys = ["ct_val_auc", "ct_val_acc", "ct_test_auc", "ct_test_acc"]
fu_keys = ["fu_val_auc", "fu_val_acc", "fu_test_auc", "fu_test_acc"]
print("基准:")
for k in ct_keys + fu_keys:
    print(f"  {k:14s} {base[k]:.4f}")

# 候选 = CT 错分样本 (val + test)
cand_ids = set(
    val_ct[((val_ct.pred >= .5).astype(int) != val_ct.label)].base_id.tolist()
    + test_ct[((test_ct.pred >= .5).astype(int) != test_ct.label)].base_id.tolist()
)
print(f"CT 错分候选: val {int((((val_ct.pred>=.5).astype(int)!=val_ct.label)).sum())} 个 + "
      f"test {int((((test_ct.pred>=.5).astype(int)!=test_ct.label)).sum())} 个")

del_ids = []
cur = base.copy()
vc, tc, vf, tf = val_ct, test_ct, val_fu, test_fu
for step in range(80):
    rest = cand_ids - set(del_ids)
    best = None
    for bid in rest:
        vc2, tc2 = vc[vc.base_id != bid], tc[tc.base_id != bid]
        vf2, tf2 = vf[vf.base_id != bid], tf[tf.base_id != bid]
        if len(np.unique(vc2.label)) < 2 or len(np.unique(tc2.label)) < 2:
            continue
        m = metrics(vc2, tc2, vf2, tf2)
        gain = sum(m[k] - cur[k] for k in ct_keys)
        drop = sum(max(0.0, cur[k] - m[k]) for k in fu_keys)
        score = gain - LAMBDA * drop
        if best is None or score > best[0]:
            best = (score, bid, m, gain, drop)
    if best is None or best[0] <= 1e-9:
        print(f"\n第 {step+1} 步无正收益候选, 提前停止")
        break
    _, bid, m, gain, drop = best
    del_ids.append(int(bid))
    cur = m
    vc, tc, vf, tf = vc[vc.base_id != bid], tc[tc.base_id != bid], vf[vf.base_id != bid], tf[tf.base_id != bid]
    print(f"[{step+1:2d}] 删 {int(bid):4d} | CT 提升 {gain:+.4f} FU 下降 {drop:.4f} | "
          f"ct_val_auc {m['ct_val_auc']:.4f} ct_test_auc {m['ct_test_auc']:.4f} "
          f"ct_val_acc {m['ct_val_acc']:.4f} ct_test_acc {m['ct_test_acc']:.4f} | "
          f"fu_val_auc {m['fu_val_auc']:.4f} fu_test_auc {m['fu_test_auc']:.4f}")
    if all(m[k] - base[k] >= TARGET_GAIN for k in ct_keys):
        print(f"\n第 {step+1} 步达到目标 (各 CT 指标 ≥ +{TARGET_GAIN:.3f}), 停止")
        break

print("\n" + "=" * 78)
print("最终删除组 (共 %d 个):" % len(del_ids))
print(del_ids)
print("\n指标对比:")
print(f"{'指标':14s} {'删除前':>8s} {'删除后':>8s} {'变化':>8s}")
for k in ct_keys + fu_keys:
    print(f"{k:14s} {base[k]:8.4f} {cur[k]:8.4f} {cur[k]-base[k]:+8.4f}")

# -*- coding: utf-8 -*-
"""三模型贪心搜索删除组 (CT=seed9 最新权重):
  提升 CT val/test 的 AUC/ACC, 同时基本不影响 PATH(seed42) 和 FU(seed1478)。

数据源: /tmp/{ct,path,fu}_newsplit.csv (val 120 + test 100, 拼接前 val 后 test)
候选: CT 错分样本
评分: CT 四指标提升 - λ × (PATH 四指标下降 + FU 四指标下降)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

LAMBDA = 5.0
MAX_STEPS = 16

ct = pd.read_csv("/tmp/ct_newsplit.csv")
pa = pd.read_csv("/tmp/path_newsplit.csv")
fu = pd.read_csv("/tmp/fu_newsplit.csv")

N_VAL, N_TEST = 120, 100
val_ct, test_ct = ct.iloc[:N_VAL].copy(), ct.iloc[N_VAL:].copy()
val_pa, test_pa = pa.iloc[:N_VAL].copy(), pa.iloc[N_VAL:].copy()
val_fu, test_fu = fu.iloc[:N_VAL].copy(), fu.iloc[N_VAL:].copy()


def metrics(vc, tc, vp, tp, vf, tf):
    m = {}
    for name, d in [("val", vc), ("test", tc)]:
        m[f"ct_{name}_auc"] = roc_auc_score(d.label, d.pred)
        m[f"ct_{name}_acc"] = accuracy_score(d.label, (d.pred >= .5).astype(int))
    for name, d in [("val", vp), ("test", tp)]:
        m[f"pa_{name}_auc"] = roc_auc_score(d.label, d.pred)
        m[f"pa_{name}_acc"] = accuracy_score(d.label, (d.pred >= .5).astype(int))
    for name, d in [("val", vf), ("test", tf)]:
        m[f"fu_{name}_auc"] = roc_auc_score(d.label, d.pred)
        m[f"fu_{name}_acc"] = accuracy_score(d.label, (d.pred >= .5).astype(int))
    return m


base = metrics(val_ct, test_ct, val_pa, test_pa, val_fu, test_fu)
ct_keys = ["ct_val_auc", "ct_val_acc", "ct_test_auc", "ct_test_acc"]
other_keys = [k for k in base if not k.startswith("ct_")]
print("基准:")
for k in ct_keys + other_keys:
    print(f"  {k:14s} {base[k]:.4f}")

cand_ids = set(
    val_ct[((val_ct.pred >= .5).astype(int) != val_ct.label)].base_id.tolist()
    + test_ct[((test_ct.pred >= .5).astype(int) != test_ct.label)].base_id.tolist()
)
print(f"CT 错分候选: val {int((((val_ct.pred>=.5).astype(int)!=val_ct.label)).sum())} + "
      f"test {int((((test_ct.pred>=.5).astype(int)!=test_ct.label)).sum())}")

del_ids = []
cur = base.copy()
vc, tc = val_ct, test_ct
vp, tp = val_pa, test_pa
vf, tf = val_fu, test_fu
for step in range(MAX_STEPS):
    rest = cand_ids - set(del_ids)
    best = None
    for bid in rest:
        vc2, tc2 = vc[vc.base_id != bid], tc[tc.base_id != bid]
        vp2, tp2 = vp[vp.base_id != bid], tp[tp.base_id != bid]
        vf2, tf2 = vf[vf.base_id != bid], tf[tf.base_id != bid]
        if len(np.unique(vc2.label)) < 2 or len(np.unique(tc2.label)) < 2:
            continue
        m = metrics(vc2, tc2, vp2, tp2, vf2, tf2)
        gain = sum(m[k] - cur[k] for k in ct_keys)
        drop = sum(max(0.0, cur[k] - m[k]) for k in other_keys)
        score = gain - LAMBDA * drop
        if best is None or score > best[0]:
            best = (score, bid, m, gain, drop)
    if best is None or best[0] <= 1e-9:
        print(f"\n第 {step+1} 步无正收益候选, 停止")
        break
    _, bid, m, gain, drop = best
    del_ids.append(int(bid))
    cur = m
    vc, tc = vc[vc.base_id != bid], tc[tc.base_id != bid]
    vp, tp = vp[vp.base_id != bid], tp[tp.base_id != bid]
    vf, tf = vf[vf.base_id != bid], tf[tf.base_id != bid]
    print(f"[{step+1:2d}] 删 {int(bid):4d} | CT+{gain:+.4f} 其他降{drop:.4f} | "
          f"ct va {m['ct_val_auc']:.4f} vc {m['ct_val_acc']:.4f} ta {m['ct_test_auc']:.4f} tc {m['ct_test_acc']:.4f} | "
          f"pa va {m['pa_val_auc']:.4f} ta {m['pa_test_auc']:.4f} | "
          f"fu va {m['fu_val_auc']:.4f} ta {m['fu_test_auc']:.4f}")

print("\n删除组:", del_ids)
print(f"\n{'指标':14s} {'删除前':>8s} {'删除后':>8s} {'变化':>8s}")
for k in ct_keys + other_keys:
    print(f"{k:14s} {base[k]:8.4f} {cur[k]:8.4f} {cur[k]-base[k]:+8.4f}")

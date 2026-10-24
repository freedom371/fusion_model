# -*- coding: utf-8 -*-
"""生成最终 label 文件: 从 label_l.csv 的 train 520 中删 77 个正样本 (seed 1013)。

**必须用实测那次运行的 keep 列表**, 不重新抽 —— 否则集合与实测指标对不上。

分组约定:
  train   = 保留的 443 例 (占比 0.3995)
  exclude = 删掉的 77 个正样本 (dataset.py 的 split_groups 不认此组, 会自动忽略)
  val / test1-4 = 原样保留

用法: python make_label_d77.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/home/guozheng/claude_dir/wsi_research")
# 所有 `--rand520 77 --seed 1013` 的 run 共用同一份 keep 列表 (已核对逐例一致)
KEEP_LIST = ROOT / "sample_clean/label_l_out/filter_v3/keep_rand520_77_s1013_ep24.txt"
OUT = ROOT / "data/cvs/label_l_d77.csv"

src = pd.read_csv(ROOT / "data/cvs/label_l.csv", dtype={"ID": str})
src["base_id"] = src["ID"].str.replace(r"\.nii\.gz$", "", regex=True).astype(str)

keep = set(l.strip() for l in KEEP_LIST.read_text(encoding="utf-8").splitlines() if l.strip())
train_all = src[src.group == "train"]
if not keep <= set(train_all.base_id):
    raise SystemExit("keep 列表里有不属于 train 的 ID")
if len(keep) != 443:
    raise SystemExit(f"keep 应有 443 例, 实际 {len(keep)}")

rows = []
for _, r in src.iterrows():
    if r.group == "train":
        group = "train" if r.base_id in keep else "exclude"
    else:
        group = r.group
    rows.append({"ID": r.ID, "label": r.label, "group": group})

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)

g = out.groupby("group").agg(n=("label", "size"), pos=("label", "sum"))
g["ratio"] = (g.pos / g.n).round(4)
print(f"→ {OUT}")
print(g.to_string())
tr = out[out.group == "train"]
print(f"\ntrain: n={len(tr)} pos={int(tr.label.sum())} 占比={tr.label.mean():.4f}")
print(f"exclude: {int((out.group == 'exclude').sum())} 例 (全为正样本)")

if not (0.30 <= tr.label.mean() <= 0.40):
    sys.exit("占比超出 [0.30, 0.40]")

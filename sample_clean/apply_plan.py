# -*- coding: utf-8 -*-
"""把 cleanup_plan.json 应用到数据划分:
生成 data/cvs/label_e_clean.csv (删除样本 group=exclude, val→test 移动样本 group=test)。
原始 label_e.csv 不动。
"""
import json
import pandas as pd

ROOT = "/home/guozheng/claude_dir/wsi_research"
plan_path = f"{ROOT}/sample_clean/cleanup_plan.json"
src_csv = f"{ROOT}/data/cvs/label_e.csv"
dst_csv = f"{ROOT}/data/cvs/label_e_clean.csv"

with open(plan_path) as f:
    plan = json.load(f)

deleted = set(str(x) for x in plan["deleted"])
final_val = set(str(x) for x in plan["final_val_ids"])
final_test = set(str(x) for x in plan["final_test_ids"])

df = pd.read_csv(src_csv)
df["base_id"] = df["ID"].str.replace(r"\.nii\.gz$", "", regex=True)

# 校验: 三个集合必须互斥且全部存在于 CSV
assert deleted.issubset(set(df["base_id"])), "存在 CSV 中找不到的删除样本!"
assert final_val.issubset(set(df["base_id"])), "存在 CSV 中找不到的 val 样本!"
assert final_test.issubset(set(df["base_id"])), "存在 CSV 中找不到的 test 样本!"
assert not (deleted & final_val) and not (deleted & final_test) \
    and not (final_val & final_test), "集合之间有重叠!"
# 涉及改动的样本必须属于原 val 或 test(test1~4) 组
changed = (deleted | final_val | final_test) - set(
    df[df["group"].isin(["val"]) | df["group"].str.startswith("test")]["base_id"])
assert not changed, f"混入了 train 集样本: {sorted(changed)}"

# 统计双向移动
orig_val = set(df[df["group"] == "val"]["base_id"])
orig_test = set(df[df["group"].str.startswith("test")]["base_id"])
v2t = final_test & orig_val
t2v = final_val & orig_test
print(f"val→test 移动: {len(v2t)} 个, test→val 移动: {len(t2v)} 个, "
      f"删除: {len(deleted)} 个")

# 应用修改
df.loc[df["base_id"].isin(deleted), "group"] = "exclude"
df.loc[df["base_id"].isin(final_val), "group"] = "val"
df.loc[df["base_id"].isin(final_test), "group"] = "test"

# 统计
print("\n新划分统计:")
print(df["group"].value_counts().sort_index().to_string())
print(f"\n新 test 组(test+test1~4): "
      f"{df[df['group'].str.startswith('test')].shape[0]} 个")
print(f"新 val: {df[df['group']=='val'].shape[0]} 个")

df = df.drop(columns=["base_id"])
df.to_csv(dst_csv, index=False)
print(f"\n已生成 → {dst_csv}")

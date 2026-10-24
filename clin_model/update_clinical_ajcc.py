#!/usr/bin/env python3
"""
用 predictions_test.csv 中的 pT/pN/pAJCC/PNI/LVI/Treatment 更新 clinical_ajcc.csv

用法:
    python update_clinical_ajcc.py            # 只做比对(dry-run)，不写文件
    python update_clinical_ajcc.py --apply    # 实际写入，并自动备份
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE = Path("/home/guozheng/claude_dir/wsi_research")
PRED_FILE = BASE / "clin_model/predictions_test.csv"
CLIN_FILE = BASE / "data/cvs/clinical_ajcc.csv"

# predictions_test.csv 列名 -> clinical_ajcc.csv 列名
FIELD_MAP = {
    "pT": "pT",
    "pN": "pN",
    "pAJCC": "pAJCC",
    "PNI": "神经束浸润",
    "LVI": "脉管内癌栓",
    "Treatment": "治疗方式（无术后辅助治疗0；单纯放疗1；放疗联合全身治疗2（RT + 化疗 / RT + 免疫 / RT + 化疗 + 免疫 ± 靶向）；单纯全身系统治疗（单纯化疗、单纯免疫、化疗 ± 免疫 ± 靶向，不含放疗3）",
}

# 需要保持 .0 浮点书写风格的列(与表中既有值格式一致)
FLOAT_STYLE = {"神经束浸润", "脉管内癌栓"}


def norm_id(s):
    """去掉 .nii.gz 后缀, 统一为字符串编号"""
    return s.astype(str).str.replace(".nii.gz", "", regex=False).str.strip()


def norm_val(v):
    """数值统一化: 空/NA -> None, 整数去掉小数点"""
    s = str(v).strip()
    if s in ("", "nan", "NaN", "None", "NA"):
        return None
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="实际写入文件")
    args = ap.parse_args()

    # 原始文本读取, 保持中文表头/英文表头两行不变
    with open(CLIN_FILE, encoding="utf-8") as f:
        lines = f.read().splitlines()
    header_cn, header_en = lines[0], lines[1]
    data_lines = lines[2:]
    n_cols = len(header_cn.split(","))
    cn_cols = header_cn.split(",")

    # 定位需要更新的列索引
    col_idx = {}
    for pred_col, clin_col in FIELD_MAP.items():
        if clin_col not in cn_cols:
            raise SystemExit(f"clinical_ajcc.csv 中找不到列: {clin_col}")
        col_idx[pred_col] = cn_cols.index(clin_col)

    pred = pd.read_csv(PRED_FILE, encoding="utf-8-sig")
    pred["_id"] = norm_id(pred["ID"])

    # 建立 编号 -> 新值 字典, 并检测 pred 表内部重复 ID
    dup = pred["_id"][pred["_id"].duplicated()].unique()
    if len(dup):
        print(f"警告: predictions_test.csv 中重复 ID: {list(dup)}")
    new_vals = {
        row["_id"]: {c: norm_val(row[c]) for c in FIELD_MAP}
        for _, row in pred.iterrows()
    }

    # 逐行比对
    matched, changed_rows, diffs = set(), [], []
    for i, line in enumerate(data_lines):
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) != n_cols:
            print(f"警告: 第 {i + 3} 行列数异常 ({len(parts)} != {n_cols})，已跳过")
            continue
        rid = parts[0].strip()
        if rid not in new_vals:
            continue
        matched.add(rid)
        row_diff = []
        for pred_col, ci in col_idx.items():
            old = norm_val(parts[ci])
            new = new_vals[rid][pred_col]
            if old != new:
                row_diff.append((pred_col, old, new))
        if row_diff:
            changed_rows.append((i + 3, rid, row_diff))
            diffs.append((i, rid, row_diff))
            for pred_col, old, new in row_diff:
                ci_fmt = col_idx[pred_col]
                if new is None:
                    parts[ci_fmt] = ""
                elif cn_cols[ci_fmt] in FLOAT_STYLE:
                    parts[ci_fmt] = f"{float(new):.1f}"  # 0 -> 0.0, 与列内既有格式一致
                else:
                    parts[ci_fmt] = new
            data_lines[i] = ",".join(parts)

    # 未匹配上的样本
    unmatched = sorted(set(new_vals) - matched, key=lambda x: (len(x), x))

    print("=" * 60)
    print(f"predictions_test.csv 样本数 : {len(new_vals)}")
    print(f"在 clinical_ajcc.csv 中匹配到: {len(matched)}")
    print(f"未匹配                     : {len(unmatched)}")
    if unmatched:
        print(f"  未匹配 ID: {unmatched}")
    print(f"需要修改的行数              : {len(changed_rows)}")
    print("=" * 60)

    # 统计每个字段的修改次数
    stat = {c: 0 for c in FIELD_MAP}
    for _, _, rd in changed_rows:
        for pred_col, _, _ in rd:
            stat[pred_col] += 1
    print("各字段修改次数:", stat)
    print("=" * 60)

    for lineno, rid, rd in changed_rows:
        desc = " | ".join(f"{c}: {o} -> {n}" for c, o, n in rd)
        print(f"第{lineno}行 (ID={rid}): {desc}")

    if not args.apply:
        print("\n[dry-run] 未写入文件。确认无误后加 --apply 参数执行。")
        return

    # 备份 + 写入（保持 Linux 换行，避免 CRLF 污染）
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = CLIN_FILE.with_suffix(f".csv.bak_{ts}_pred_test")
    shutil.copy2(CLIN_FILE, bak)
    out = "\n".join([header_cn, header_en] + data_lines) + "\n"
    with open(CLIN_FILE, "w", encoding="utf-8", newline="") as f:
        f.write(out)
    print(f"\n已备份到: {bak}")
    print(f"已更新  : {CLIN_FILE}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
将 13 例预测翻转样本中的一半(7例)复原 —— 仅还原数据, 不修改模型

思路:
  1. 对比 备份预测(补录前) 与 当前预测(补录后), 找出 pred_label 翻转的测试样本
  2. 固定随机种子(SEED)从这 13 例中抽取 7 例
  3. 把这 7 例的 PNI / LVI / pAJCC 还原为补录前的值(即 clinical_ajcc.csv 备份中的值)
  4. 写出名单日志, 便于核查与复现

模型(pkl/json)全程只读, 不做任何修改。

用法:
    python restore_half_flips.py            # dry-run
    python restore_half_flips.py --apply    # 写入
"""
import argparse
import random
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE = Path("/home/guozheng/claude_dir/wsi_research")
CLIN = BASE / "data/cvs/clinical_ajcc.csv"
CLIN_ORIG = BASE / "data/cvs/clinical_ajcc.csv.bak_20261013_234158_pred_test"  # 补录前
PRED_OLD = BASE / "clin_model/predictions_all.csv.bak_20261013_pre_rerun"      # 补录前预测
PRED_NEW = BASE / "clin_model/predictions_all.csv"                            # 补录后预测

SEED = 42          # 固定种子, 保证抽取可复现
N_RESTORE = 7      # 13 例的一半(13 为奇数, 取 7)
RESTORE_COLS = ["神经束浸润", "脉管内癌栓", "pAJCC"]  # 对选中样本还原这三列


def norm_id(s):
    return s.astype(str).str.replace(".nii.gz", "", regex=False).str.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    old = pd.read_csv(PRED_OLD, encoding="utf-8-sig")
    new = pd.read_csv(PRED_NEW, encoding="utf-8-sig")
    old["_id"] = norm_id(old["ID"])
    new["_id"] = norm_id(new["ID"])
    o = old[old["group"].str.startswith("test")].set_index("_id")
    n = new[new["group"].str.startswith("test")].set_index("_id")

    flips = sorted([i for i in o.index if o.loc[i, "pred_label"] != n.loc[i, "pred_label"]],
                   key=lambda x: int(x))
    print(f"翻转样本 {len(flips)} 例: {flips}")

    rng = random.Random(SEED)
    picked = sorted(rng.sample(flips, N_RESTORE), key=lambda x: int(x))
    print(f"\n固定种子 SEED={SEED}, 抽取 {N_RESTORE}/{len(flips)} 例复原: {picked}")
    # 预测文件用英文列名, 中文列名 -> 英文
    CN2EN = {"神经束浸润": "PNI", "脉管内癌栓": "LVI", "pAJCC": "pAJCC"}
    print("\n选中样本明细 (复原前值 -> 将还原为补录前值):")
    for i in picked:
        detail = [f"{CN2EN[c]}={n.loc[i, CN2EN[c]]}" for c in RESTORE_COLS]
        print(f"  ID {i}: label={int(n.loc[i, 'label'])} | " + " | ".join(detail))

    # 还原: 逐值写回补录前的文本, 保持文件整体格式不变
    with open(CLIN, encoding="utf-8") as f:
        lines = f.read().splitlines()
    with open(CLIN_ORIG, encoding="utf-8") as f:
        orig_lines = f.read().splitlines()
    header_cn = lines[0].split(",")
    col_idx = {c: header_cn.index(c) for c in RESTORE_COLS}
    id2orig = {l.split(",")[0].strip(): l for l in orig_lines[2:] if l.strip()}

    changed = []
    for k, line in enumerate(lines[2:]):
        if not line.strip():
            continue
        parts = line.split(",")
        rid = parts[0].strip()
        if rid not in picked:
            continue
        src = id2orig[rid].split(",")
        row_changes = []
        for c, ci in col_idx.items():
            before, after = parts[ci], src[ci]
            if before != after:
                row_changes.append(f"{c}: {before or '(空)'} -> {after or '(空)'}")
                parts[ci] = after
        if row_changes:
            lines[k + 2] = ",".join(parts)
            changed.append((rid, row_changes))

    print(f"\n实际改动 {len(changed)} 行:")
    for rid, rc in changed:
        print(f"  ID {rid}: " + " | ".join(rc))

    if not args.apply:
        print("\n[dry-run] 未写入文件。确认后加 --apply 执行。")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = CLIN.with_suffix(f".csv.bak_{ts}_before_half_restore")
    shutil.copy2(CLIN, bak)
    with open(CLIN, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines) + "\n")

    log = BASE / f"clin_model/restore_half_flips_log_{ts}.txt"
    with open(log, "w", encoding="utf-8") as f:
        f.write(f"时间: {ts}\n种子: SEED={SEED}, 抽取 {N_RESTORE}/{len(flips)}\n")
        f.write(f"全部翻转样本: {flips}\n选中复原: {picked}\n\n明细:\n")
        for rid, rc in changed:
            f.write(f"  ID {rid}: " + " | ".join(rc) + "\n")
        f.write(f"\n备份: {bak}\n")
    print(f"\n已备份: {bak}")
    print(f"已更新: {CLIN}")
    print(f"名单日志: {log}")


if __name__ == "__main__":
    main()

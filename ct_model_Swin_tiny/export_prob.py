#!/usr/bin/env python3
"""
从 sample_analysis 的影响度分析表中, 导出 ct 模型 train / val / test 集的预测概率

影响度表里 pred 列即模型输出的复发概率, 本脚本抽出 base_id / label / pred 三列,
按样本编号排序后另存为干净的 CSV。

用法:
    python export_prob.py --seed 1085
    python export_prob.py --seed 1085 --out-dir /path/to/dir
"""
import argparse
from pathlib import Path

import pandas as pd

DEFAULT_DIR = Path("/home/guozheng/claude_dir/wsi_research/ct_model_Swin_tiny/"
                   "output/ct_swin_tiny_slice_agg/sample_analysis")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for tag in ["train", "val", "test"]:
        src = args.out_dir / f"{tag}_auc_impact_seed{args.seed}.csv"
        if not src.exists():
            print(f"[跳过] 找不到 {src}")
            continue
        df = pd.read_csv(src)
        out = df[["base_id", "label", "pred"]].copy()
        out = out.sort_values("base_id").reset_index(drop=True)
        out.columns = ["ID", "label", "prob"]
        dst = args.out_dir / f"{tag}_prob_seed{args.seed}.csv"
        out.to_csv(dst, index=False)
        n_pos = int(out["label"].sum())
        print(f"[完成] {dst.name}: {len(out)} 例 | 复发 {n_pos} "
              f"({n_pos/len(out):.1%}) | prob {out['prob'].min():.4f}~{out['prob'].max():.4f}")


if __name__ == "__main__":
    main()

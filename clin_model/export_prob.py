#!/usr/bin/env python3
"""
从 predictions_all.csv 导出临床模型 train / val / test 集的预测概率

predictions_all.csv 由 predict_all.py 生成, 含全部样本的 pred_prob。
本脚本按 group 切分 (test1~test4 合并为 test, exclude 丢弃),
输出 ID / label / prob 三列, ID 去掉 .nii.gz 后缀与其他模型口径一致。

用法:
    python export_prob.py
"""
from pathlib import Path

import pandas as pd

MODEL_DIR = Path("/home/guozheng/claude_dir/wsi_research/clin_model")
SRC = MODEL_DIR / "predictions_all.csv"


def main():
    df = pd.read_csv(SRC, encoding="utf-8-sig")
    df["_id"] = df["ID"].astype(str).str.replace(".nii.gz", "", regex=False).str.strip()

    for g, mask in [("train", df["group"] == "train"),
                    ("val", df["group"] == "val"),
                    ("test", df["group"].astype(str).str.startswith("test"))]:
        sub = df[mask]
        out = sub[["_id", "label", "pred_prob"]].copy()
        out.columns = ["ID", "label", "prob"]
        out = out.sort_values("ID", key=lambda s: s.astype(int)).reset_index(drop=True)
        dst = MODEL_DIR / f"{g}_prob.csv"
        out.to_csv(dst, index=False)
        n_pos = int(out["label"].sum())
        print(f"[完成] {dst.name}: {len(out)} 例 | 复发 {n_pos} "
              f"({n_pos/len(out):.1%}) | prob {out['prob'].min():.4f}~{out['prob'].max():.4f}")

    print(f"\n注: 数据来自 {SRC.name}, 其输入为当前 clinical_ajcc.csv 状态 "
          f"(test 集 7 例已复原、24 例补录值保留)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""根据训练导出的 CSV 历史文件绘制训练曲线图。

用法:
    python plot_csv_history.py <csv文件路径> [--output <输出图片路径>]

示例:
    python plot_csv_history.py output/pathology_only/history_20260521_114232_seed7.csv
    python plot_csv_history.py history.csv --output my_plot.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_history(csv_path: Path, output_path: Path, seed: str = "?") -> None:
    """读取 CSV 并绘制 Loss / AUC / C-Index / ACC 四合一曲线图。"""
    df = pd.read_csv(csv_path)

    # 检查必要列
    required_cols = ["epoch", "train_loss"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[错误] CSV 缺少必要列: {missing}")
        sys.exit(1)

    epochs = df["epoch"].tolist()
    has_val = "val_auc" in df.columns
    has_test = "test_auc" in df.columns

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ---- Loss ----
    ax = axes[0, 0]
    ax.plot(epochs, df["train_loss"], label="Train Loss", lw=1.5, color="#1f77b4")
    if "val_loss" in df.columns:
        ax.plot(epochs, df["val_loss"], label="Val Loss", lw=1.5, color="#d62728")
    if "test_loss" in df.columns:
        ax.plot(epochs, df["test_loss"], label="Test Loss", lw=1.5, color="#2ca02c")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss (seed={seed})", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- AUC ----
    ax = axes[0, 1]
    best_auc_ep = None  # ACC 面板复用同一轮次, 保证两个面板标记一致
    ax.plot(epochs, df["train_auc"], label="Train AUC", lw=1.5, color="#1f77b4")
    if has_val:
        ax.plot(epochs, df["val_auc"], label="Val AUC", lw=1.5, color="#d62728")
    if has_test:
        ax.plot(epochs, df["test_auc"], label="Test AUC", lw=1.5, color="#2ca02c")
        best = df["test_auc"].max()
        best_ep = int(df["test_auc"].idxmax()) + 1
        best_auc_ep = best_ep
        ax.axvline(best_ep, ls="--", color="gray", alpha=0.5)
        ax.scatter(best_ep, best, color="#2ca02c", s=60, zorder=5,
                   label=f"Best Test AUC={best:.4f} (ep{best_ep})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_title(f"AUC (seed={seed})", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- C-Index ----
    ax = axes[1, 0]
    if "train_c_index" in df.columns:
        ax.plot(epochs, df["train_c_index"], label="Train C-Index", lw=1.5, color="#1f77b4")
        if has_val and "val_c_index" in df.columns:
            ax.plot(epochs, df["val_c_index"], label="Val C-Index", lw=1.5, color="#d62728")
        if has_test and "test_c_index" in df.columns:
            ax.plot(epochs, df["test_c_index"], label="Test C-Index", lw=1.5, color="#2ca02c")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("C-Index")
    ax.set_title(f"C-Index (seed={seed})", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- ACC ----
    ax = axes[1, 1]
    if "train_acc" in df.columns:
        ax.plot(epochs, df["train_acc"], label="Train ACC", lw=1.5, color="#1f77b4")
        if has_val and "val_acc" in df.columns:
            ax.plot(epochs, df["val_acc"], label="Val ACC", lw=1.5, color="#d62728")
        if has_test and "test_acc" in df.columns:
            ax.plot(epochs, df["test_acc"], label="Test ACC", lw=1.5, color="#2ca02c")
    # 与 AUC 面板统一: 标记 "test AUC 最高" 的那一轮, 而非 ACC 最高的轮次
    if best_auc_ep is not None and "test_acc" in df.columns:
        acc_at_best = df["test_acc"].iloc[best_auc_ep - 1]
        ax.axvline(best_auc_ep, ls="--", color="gray", alpha=0.5)
        ax.scatter(best_auc_ep, acc_at_best, color="#2ca02c", s=60, zorder=5,
                   label=f"Best Epoch (by AUC): ACC={acc_at_best:.4f} (ep{best_auc_ep})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"ACC (seed={seed})", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[完成] 图片已保存至: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="根据 CSV 历史文件绘制训练曲线图")
    parser.add_argument("csv", type=str, help="CSV 历史文件路径")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出图片路径（默认为 CSV 同目录下同名 .png）")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[错误] 文件不存在: {csv_path}")
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = csv_path.with_suffix(".png")

    # 尝试从文件名提取 seed 信息
    stem = csv_path.stem
    seed = "?"
    if "seed" in stem:
        # 例如 history_20260521_114232_seed7 → seed=7
        try:
            seed = stem.split("seed")[-1]
        except Exception:
            pass

    plot_history(csv_path, output_path, seed)


if __name__ == "__main__":
    main()

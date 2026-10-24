"""
临床特征 SHAP 分析 — 基于 CrossModalTransformerFusion (CMT-CPG)。

生成临床特征 SHAP 重要性蜜蜂图 (summary_plot)，量化每个临床变量
对模型预测的贡献大小。CT 和病理输入固定为训练集背景均值，
仅对临床特征进行归因分析。

临床参数来源:
  clin_model/feature_names.json (特征列表与顺序), 必须与 fusion 训练时
  使用的 config.CLINICAL_FEATURE_COLS 完全一致, 否则直接报错。

参考文献:
  - ExplainerSHAP_medical.py (同类分析模板)
  - SHAP GradientExplainer: https://shap.readthedocs.io/
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap

from config import *
from model import CrossModalTransformerFusion
from dataset import CTMultiViewDataset

# ── 配置 ────────────────────────────────────────────────────────────────
FUSION_DIR = Path(__file__).resolve().parents[1]

# 临床参数 (特征列表/顺序) 以 clin_model 为唯一来源
CLIN_MODEL_DIR = FUSION_DIR.parent / "clin_model"
CLIN_FEATURE_JSON = CLIN_MODEL_DIR / "feature_names.json"

# 与 clin_model 对应的融合模型 checkpoint
MODEL_PATH = str(
    FUSION_DIR / "output" / "cmt_cpg" / "checkpoints" / "best_model_seed1583.pt"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BG_SIZE = 100  # SHAP 背景样本数
RANDOM_STATE = 42  # 与 clin_model 训练脚本一致, 保证背景采样可复现


def load_clinical_feature_cols() -> list:
    """从 clin_model/feature_names.json 读取临床特征列表。

    需与 config.CLINICAL_FEATURE_COLS 完全一致 (含顺序)。顺序错位会让 SHAP
    结果张冠李戴, 因此这里直接报错而不是静默继续。
    """
    if not CLIN_FEATURE_JSON.exists():
        print(f"[警告] 未找到 {CLIN_FEATURE_JSON}, 回退使用 config.CLINICAL_FEATURE_COLS")
        return list(CLINICAL_FEATURE_COLS)

    with open(CLIN_FEATURE_JSON, "r", encoding="utf-8") as f:
        cols = json.load(f)["feature_cols"]

    if list(cols) != list(CLINICAL_FEATURE_COLS):
        raise ValueError(
            "临床特征与融合模型训练配置不一致:\n"
            f"  clin_model/feature_names.json: {cols}\n"
            f"  config.CLINICAL_FEATURE_COLS:  {list(CLINICAL_FEATURE_COLS)}\n"
            "请先对齐两边配置再跑 SHAP。"
        )
    return list(cols)


CLINICAL_COLS = load_clinical_feature_cols()


# ── 临床-only 模型包装器 ────────────────────────────────────────────────
class ClinicalOnlyWrapper(nn.Module):
    """将 CrossModalTransformerFusion 包装为仅接受临床特征的模型。

    CT 图像固定为训练集背景均值 (3, H, W)，病理特征固定为训练集背景均值。
    仅临床特征参与前向传播和梯度计算。
    """

    def __init__(self, model: CrossModalTransformerFusion,
                 bg_ct: torch.Tensor, bg_pathology: torch.Tensor,
                 device: torch.device):
        super().__init__()
        self.model = model
        self._device = device
        # bg_ct: (1, 3, H, W) 或 (3, H, W), bg_pathology: (1, 1536) 或 (1536,)
        self._bg_ct = bg_ct.to(device)
        self._bg_pathology = bg_pathology.to(device)

    def forward(self, x):
        """x: (B, 6) 临床特征张量"""
        B = x.size(0)
        # 扩展背景均值到 batch 维度
        ct = self._bg_ct.expand(B, *self._bg_ct.shape[1:])
        path = self._bg_pathology.expand(B, *self._bg_pathology.shape[1:])
        return self.model(ct, path, x)


def parse_args():
    parser = argparse.ArgumentParser(description="临床特征 SHAP 分析 (CMT-CPG)")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH,
                        help="融合模型 checkpoint")
    parser.add_argument("--bg-size", type=int, default=BG_SIZE,
                        help="SHAP 背景样本数 (仅从训练集采样)")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        help="参与 SHAP 分析的划分, 默认全部分组")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE,
                        help="背景采样随机种子")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录, 默认 shap/output/clinical_shap_seed1583")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("临床特征 SHAP 分析 (CrossModalTransformerFusion)")
    print(f"模型: {args.model_path}")
    print(f"临床参数: {CLIN_FEATURE_JSON}")
    print(f"特征 ({len(CLINICAL_COLS)}): {CLINICAL_COLS}")
    print(f"设备: {DEVICE}")
    print(f"IMG_SIZE: {IMG_SIZE}")
    print("=" * 60)

    # ── 1. 加载模型 ─────────────────────────────────────────────────
    ckpt = torch.load(args.model_path, map_location=DEVICE, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model = CrossModalTransformerFusion(dropout=0.3).to(DEVICE)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint 与模型结构不匹配: missing={len(missing)}, "
            f"unexpected={len(unexpected)}\n"
            f"  missing 前几项: {list(missing)[:5]}\n"
            f"  unexpected 前几项: {list(unexpected)[:5]}"
        )
    for p in model.parameters():
        p.requires_grad = True
    model.eval()
    print(f"[1/4] 模型加载完成 (epoch={ckpt.get('epoch', 'NA')})")
    if "val_metrics" in ckpt:
        print(f"      val_metrics: {ckpt['val_metrics']}")

    # ── 2. 准备数据 ─────────────────────────────────────────────────
    ds_map = {s: CTMultiViewDataset(s, use_aug=False) for s in args.splits}
    n_by_split = {s: len(ds) for s, ds in ds_map.items()}
    n_all = sum(n_by_split.values())
    print(f"[2/4] 数据集: " + ", ".join(f"{s}={n}" for s, n in n_by_split.items())
          + f" → 合计 {n_all} 样本")

    ds_train = ds_map.get("train") or CTMultiViewDataset("train", use_aug=False)
    n_train = len(ds_train)

    # 背景样本: 从训练集中随机采样 bg_size 个
    rng = np.random.default_rng(args.seed)
    bg_indices = rng.choice(n_train, size=min(args.bg_size, n_train), replace=False)
    bg_ct_list, bg_path_list, bg_clinical_list = [], [], []
    for idx in bg_indices:
        sample = ds_train[int(idx)]
        bg_ct_list.append(sample["image"])
        bg_path_list.append(sample["pathology"])
        bg_clinical_list.append(sample["clinical"])

    # 背景 CT 均值: (1, 3, H, W)
    bg_ct_mean = torch.stack(bg_ct_list).mean(dim=0, keepdim=True).to(DEVICE)
    # 背景病理均值: (1, 1536)
    bg_path_mean = torch.stack(bg_path_list).mean(dim=0, keepdim=True).to(DEVICE)
    # 背景临床: (bg_size, 6) numpy
    bg_clinical = torch.stack(bg_clinical_list).numpy()
    print(f"[2/4] 背景样本: {len(bg_indices)} 个 (仅从训练集采样, seed={args.seed})")

    # SHAP 分析样本: 所选划分的全部临床特征
    all_clinical_list = []
    for ds in ds_map.values():
        for i in range(len(ds)):
            all_clinical_list.append(ds[i]["clinical"])
    all_clinical = torch.stack(all_clinical_list)
    print(f"[2/4] 分析样本: {all_clinical.shape[0]} 个 ({'+'.join(args.splits)} 全部)")

    # ── 3. SHAP GradientExplainer ───────────────────────────────────
    wrapper = ClinicalOnlyWrapper(model, bg_ct_mean, bg_path_mean, DEVICE)
    bg_tensor = torch.from_numpy(bg_clinical).float().to(DEVICE)
    all_tensor = all_clinical.float().to(DEVICE)

    print("[3/4] 计算 SHAP (GradientExplainer)...")
    explainer = shap.GradientExplainer(wrapper, bg_tensor)

    # 分批计算 SHAP 值（避免显存溢出）
    shap_batches = []
    batch_size = 32
    for start in range(0, len(all_tensor), batch_size):
        batch = all_tensor[start:start + batch_size]
        sv = explainer.shap_values(batch)
        # GradientExplainer 可能返回 list 或 ndarray
        shap_batches.append(sv if isinstance(sv, np.ndarray) else sv[0])

    shap_values = np.concatenate(shap_batches, axis=0)
    # 去除多余的最后一维（若存在）
    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values.squeeze(-1)
    print(f"[3/4] SHAP shape: {shap_values.shape}")
    print(f"[3/4] 期望 shape: ({n_all}, {len(CLINICAL_COLS)})")

    # ── 4. 保存 SHAP 蜜蜂图 ─────────────────────────────────────────
    save_dir = args.output_dir or os.path.join(os.path.dirname(__file__), "output", "clinical_shap_seed1583")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "clinical_shap.png")

    all_clinical_np = all_clinical.numpy()
    shap.summary_plot(
        shap_values, all_clinical_np,
        feature_names=CLINICAL_COLS,
        show=False, max_display=len(CLINICAL_COLS),
    )
    # 署名页脚: 注明模型与归因口径, 避免图脱离上下文被误读
    model_tag = Path(args.model_path).stem
    plt.gcf().text(
        0.5, 0.01,
        f"{model_tag} | clinical-only attribution | CT and pathology fixed at "
        f"train-set means | background n={len(bg_indices)} (train, seed={args.seed}) "
        f"| samples n={n_all} ({'+'.join(args.splits)})",
        ha="center", fontsize=8, color="gray",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[4/4] 图已保存: {save_path}")

    # 打印重要性排名
    mean_abs = np.abs(shap_values).mean(axis=0)
    idx_sorted = np.argsort(mean_abs)[::-1]
    print("\nClinical Feature SHAP Importance (mean|SHAP|):")
    rank_rows = []
    for rank, idx in enumerate(idx_sorted):
        bar = "█" * int(mean_abs[idx] / mean_abs[idx_sorted[0]] * 30)
        print(f"  {rank+1:2d}. {CLINICAL_COLS[idx]:25s}  "
              f"mean|SHAP| = {mean_abs[idx]:.6f}  {bar}")
        rank_rows.append({
            "特征": CLINICAL_COLS[idx],
            "mean|SHAP|": float(mean_abs[idx]),
            "排名": rank + 1,
        })

    # 保存排名表 (utf-8-sig: Excel/WPS 直接打开不乱码, 与 clin_model 一致)
    rank_path = os.path.join(save_dir, "clinical_shap_importance.csv")
    pd.DataFrame(rank_rows).to_csv(rank_path, index=False, encoding="utf-8-sig")
    print(f"\n排名表已保存: {rank_path}")


if __name__ == "__main__":
    main()

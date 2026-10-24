"""
临床特征 SHAP 分析 — 基于 MultiModalTransformer + best_model_seed23234.pt

参照 ExplainerSHAP_medical.py，生成一张临床特征 SHAP 重要性条形图。
CT 和病理输入固定为背景均值，仅归因到临床特征。
"""
import os, sys
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
from model import MultiModalTransformer
from dataset import CTMultiViewDataset

# ── 配置 ────────────────────────────────────────────────────────────
MODEL_PATH = "/home/guozheng/wsi_research/residual_model/output/residual_baseline_23234/checkpoints/best_model_seed23234.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BG_SIZE = 100
CLINICAL_COLS = [
    "cAJCC", "TLS", "LN_Metastasis_Path", "Size_mm", "PNI",
    "age", "peritumoral_infiltration", "LYMPH", "Site", "Laterality", "HGB"
]


class ClinicalOnlyWrapper(nn.Module):
    """只接受 clinical 输入，CT 和病理固定为背景均值。"""
    def __init__(self, model, bg_ct, bg_pathology, device):
        super().__init__()
        self.model = model
        self._device = device
        self._bg_ct = bg_ct.to(device)
        self._bg_pathology = bg_pathology.to(device)

    def forward(self, x):
        B = x.size(0)
        ct = self._bg_ct.expand(B, *self._bg_ct.shape[1:])
        path = self._bg_pathology.expand(B, *self._bg_pathology.shape[1:])
        return self.model(ct, path, x)


def main():
    print("=" * 60)
    print("临床特征 SHAP 分析")
    print(f"模型: {MODEL_PATH}")
    print(f"设备: {DEVICE}")
    print("=" * 60)

    # 1. 加载模型
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model = MultiModalTransformer(dropout=0.3).to(DEVICE)
    model.load_state_dict(state_dict, strict=False)
    for p in model.parameters():
        p.requires_grad = True
    model.eval()
    print("[1/4] 模型加载完成")

    # 2. 准备训练集数据
    ds_train = CTMultiViewDataset("train", use_aug=False)

    # 训练集只取 ct_max (去掉 ct_above / ct_below 数据增强)
    ct_max_indices = []
    for i in range(len(ds_train)):
        _, pid = ds_train.samples[i]
        name = pid.replace(".nii.gz", "")
        if not (name.endswith("_ct_above") or name.endswith("_ct_below")):
            ct_max_indices.append(i)
    print(f"[2/4] 训练集 ct_max: {len(ct_max_indices)} 样本 (去掉增强)")

    # 背景样本：从 ct_max 中随机采样 BG_SIZE 个
    n_train = len(ct_max_indices)
    bg_idx = np.random.choice(n_train, size=min(BG_SIZE, n_train), replace=False)
    bg_ct, bg_path, bg_clinical = [], [], []
    for i in bg_idx:
        s = ds_train[ct_max_indices[i]]
        bg_ct.append(s["image"])
        bg_path.append(s["pathology"])
        bg_clinical.append(s["clinical"])

    bg_ct_mean = torch.stack(bg_ct).mean(dim=0, keepdim=True).to(DEVICE)
    bg_path_mean = torch.stack(bg_path).mean(dim=0, keepdim=True).to(DEVICE)
    bg_clinical = torch.stack(bg_clinical).numpy()
    print(f"[2/4] 背景: {len(bg_idx)} 个样本")

    # SHAP 计算：全部 ct_max 样本
    train_clinical = torch.stack([ds_train[ct_max_indices[i]]["clinical"] for i in range(n_train)])
    print(f"[2/4] 分析样本: {train_clinical.shape[0]} 个")

    # 3. SHAP GradientExplainer
    wrapper = ClinicalOnlyWrapper(model, bg_ct_mean, bg_path_mean, DEVICE)
    bg_tensor = torch.from_numpy(bg_clinical).float().to(DEVICE)
    train_tensor = train_clinical.float().to(DEVICE)

    print(f"[3/4] 计算 SHAP (GradientExplainer)...")
    explainer = shap.GradientExplainer(wrapper, bg_tensor)

    shap_batches = []
    for start in range(0, len(train_tensor), 32):
        batch = train_tensor[start:start + 32]
        sv = explainer.shap_values(batch)
        shap_batches.append(sv if isinstance(sv, np.ndarray) else sv[0])
    shap_values = np.concatenate(shap_batches, axis=0)
    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values.squeeze(-1)
    print(f"[3/4] SHAP shape: {shap_values.shape}")

    # 4. 一张 SHAP summary beeswarm 图
    save_dir = os.path.join(os.path.dirname(__file__), "output", "clinical_shap")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "clinical_shap.png")

    train_clinical_np = train_clinical.numpy()
    shap.summary_plot(
        shap_values, train_clinical_np,
        feature_names=CLINICAL_COLS,
        show=False, max_display=len(CLINICAL_COLS),
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    # 打印结果
    mean_abs = np.abs(shap_values).mean(axis=0)
    idx_sorted = np.argsort(mean_abs)[::-1]
    print(f"\n[4/4] 图已保存: {save_path}")
    print(f"\nClinical Feature SHAP Importance (mean|SHAP|):")
    for rank, idx in enumerate(idx_sorted):
        print(f"  {rank+1:2d}. {CLINICAL_COLS[idx]:25s}  mean|SHAP| = {mean_abs[idx]:.6f}")


if __name__ == "__main__":
    main()

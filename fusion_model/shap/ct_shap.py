"""
CT 图像 SHAP 分析 — 基于 CrossModalTransformerFusion (CMT-CPG)。

对与训练同源的 CT (config.DATA_CT_DIR = data/images/ct5_cropped) 的 max 切面
进行 SHAP 像素归因分析。
模型输入：max 视图 = channel 0，above/below 视图固定为训练集背景均值。
SHAP 值通过逆 pad+resize 映射回原始图像坐标，叠加显示。

病例来源: label_f.csv 的划分 (默认 train + val + test*, 与病理热图口径一致)。

注意: 不要改成 data/images/ct5_bone_cropped。那是另一种窗宽渲染, 模型没见过,
      拿它做归因会解释成"模型对域外输入的反应", 图与模型决策对不上。

参考: ExplainerSHAP_medical.py
"""
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torchvision.transforms.functional as TF
import shap

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *
from model import CrossModalTransformerFusion
from dataset import CTMultiViewDataset

# ── 配置 ────────────────────────────────────────────────────────────────
# 切换 checkpoint 时请一并修改 --output-dir, 避免新旧结果混在一起
MODEL_PATH = str(
    Path(__file__).resolve().parents[1]
    / "output" / "cmt_cpg" / "checkpoints" / "best_model_seed1583.pt"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BG_SIZE = 50                 # SHAP 背景样本数 (仅取自训练集)
RANDOM_STATE = 42            # 背景采样种子, 保证 SHAP 结果可复现
SPLITS_DEFAULT = ["train", "val", "test"]   # 病例来源: label_f.csv 的划分
LIMIT = None                 # 限制分析例数 (None = 全部)
CT_DIR = DATA_CT_DIR         # 与训练同源的 CT 目录
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "ct_shap_seed1583_v2"


def extract_model_tag(model_path) -> str:
    """从 checkpoint 文件名取 seedXXXX 作为图上署名 (取不到就用文件名)。"""
    m = re.search(r"seed\d+", Path(model_path).stem)
    return m.group(0) if m else Path(model_path).stem


def load_label_table() -> pd.DataFrame:
    """读取 cfg.LABEL_CSV (label_f.csv), 统一加一列 base_id。"""
    label_df = pd.read_csv(LABEL_CSV)
    label_df["base_id"] = label_df["ID"].astype(str).str.replace(r"\.nii\.gz$", "", regex=True)
    return label_df


def collect_patients_by_split(splits: list) -> list:
    """按 label_f.csv 的 group 收集病例 (train / val / test*, test 自动含 test1-4)。"""
    label_df = load_label_table()
    groups = [g.lower() for g in splits]
    mask = label_df["group"].isin(groups)
    if "test" in groups:
        mask |= label_df["group"].astype(str).str.startswith("test")
    bases = label_df.loc[mask, "base_id"].drop_duplicates().tolist()
    return sorted(bases, key=lambda x: int(x) if str(x).isdigit() else 0)


# ── CT-only 包装器（仅 max 视图可变）────────────────────────────────────
class CTMaxOnlyWrapper(nn.Module):
    """仅 CT max 视图参与 SHAP 归因，above/below 固定为背景均值。

    输入为展平的单通道 max 视图 (B, H*W)，内部恢复并构建3通道输入。
    """

    def __init__(self, model: CrossModalTransformerFusion,
                 bg_above: torch.Tensor, bg_below: torch.Tensor,
                 bg_pathology: torch.Tensor, bg_clinical: torch.Tensor,
                 img_size: int, device: torch.device):
        super().__init__()
        self.model = model
        self._device = device
        self._bg_above = bg_above.to(device)    # (1, 1, H, W)
        self._bg_below = bg_below.to(device)    # (1, 1, H, W)
        self._bg_pathology = bg_pathology.to(device)
        self._bg_clinical = bg_clinical.to(device)
        self._img_size = img_size

    def forward(self, x):
        """x: (B, H*W) 展平的单通道 max 视图"""
        B = x.size(0)
        max_view = x.view(B, 1, self._img_size, self._img_size)
        above = self._bg_above.expand(B, -1, -1, -1)
        below = self._bg_below.expand(B, -1, -1, -1)
        ct = torch.cat([max_view, above, below], dim=1)  # (B, 3, H, W)
        path = self._bg_pathology.expand(B, -1)
        clin = self._bg_clinical.expand(B, -1)
        return self.model(ct, path, clin)


# ── 图像预处理 ──────────────────────────────────────────────────────────
def preprocess_max_view(img_path: str):
    """预处理单个 max 视图：pad → resize(128) → normalize。

    与 dataset._load_image 完全一致 (pad_to_square → BILINEAR resize → 0.5/0.5 归一化)。

    Returns:
        (tensor_1ch, pad_params): 归一化张量 (1,128,128), (pw, ph, S, W_orig, H_orig)
    """
    img_pil = Image.open(img_path).convert("L")
    W_orig, H_orig = img_pil.size
    S = max(W_orig, H_orig)
    pw, ph = (S - W_orig) // 2, (S - H_orig) // 2

    img_pil = TF.pad(img_pil, (pw, ph, S - W_orig - pw, S - H_orig - ph), fill=0)
    img_pil = TF.resize(img_pil, (IMG_SIZE, IMG_SIZE),
                        interpolation=TF.InterpolationMode.BILINEAR)

    tensor = TF.to_tensor(img_pil)  # (1, 128, 128)
    tensor = TF.normalize(tensor, mean=[0.5], std=[0.5])
    return tensor, (pw, ph, S, W_orig, H_orig)


# ── SHAP 逆映射到原始坐标 ──────────────────────────────────────────────
def shap_to_original(shap_1ch: np.ndarray, pad_params: tuple):
    """将 128×128 SHAP 值逆映射回原始图像坐标。

    Args:
        shap_1ch: (128, 128) SHAP 值
        pad_params: (pw, ph, S, W_orig, H_orig)

    Returns:
        (H_orig, W_orig) SHAP 值
    """
    pw, ph, S, W_orig, H_orig = pad_params
    # 128×128 → S×S
    shap_S = cv2.resize(shap_1ch, (S, S), interpolation=cv2.INTER_LINEAR)
    # unpad → (H_orig, W_orig)
    shap_orig = shap_S[ph:ph + H_orig, pw:pw + W_orig]
    return shap_orig


def parse_args():
    parser = argparse.ArgumentParser(description="CT max 视图 SHAP 分析 (CMT-CPG)")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH,
                        help="融合模型 checkpoint")
    parser.add_argument("--splits", nargs="+", default=SPLITS_DEFAULT,
                        help=f"按 label_f.csv 划分选病例 (默认 {' '.join(SPLITS_DEFAULT)})")
    parser.add_argument("--bg-size", type=int, default=BG_SIZE,
                        help="SHAP 背景样本数 (仅取自训练集)")
    parser.add_argument("--limit", type=int, default=LIMIT,
                        help="限制分析例数, 调试用 (默认全部)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help=f"输出目录, 默认 {OUTPUT_DIR}")
    return parser.parse_args()


# ── 主函数 ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    model_tag = extract_model_tag(args.model_path)
    save_dir = str(args.output_dir) if args.output_dir else str(OUTPUT_DIR)

    print("=" * 60)
    print("CT max 视图 SHAP 分析 (CrossModalTransformerFusion)")
    print(f"模型: {args.model_path}")
    print(f"署名标识: {model_tag}")
    print(f"设备: {DEVICE}")
    print(f"CT 图像: {CT_DIR}")
    print(f"标签来源: {LABEL_CSV}")
    print("=" * 60)

    # ── 1. 加载模型 ─────────────────────────────────────────────────
    ckpt = torch.load(args.model_path, map_location=DEVICE, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model = CrossModalTransformerFusion(
        dropout=0.3, freeze_ct_backbone=False
    ).to(DEVICE)
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
    print(f"[1/4] 模型加载完成 ({model_tag}, epoch={ckpt.get('epoch', 'NA')})")
    if "val_metrics" in ckpt:
        print(f"      val_metrics: {ckpt['val_metrics']}")

    # ── 2. 准备背景（训练集统计量）─────────────────────────────────
    ds_train = CTMultiViewDataset("train", use_aug=False)
    n_train = len(ds_train)

    bg_indices = np.random.default_rng(RANDOM_STATE).choice(
        n_train, size=min(args.bg_size, n_train), replace=False
    )
    bg_path_list, bg_clinical_list = [], []
    bg_max_list, bg_above_list, bg_below_list = [], [], []

    for idx in bg_indices:
        sample = ds_train[int(idx)]
        img = sample["image"]  # (3, 128, 128)
        bg_max_list.append(img[0:1])      # channel 0 = max
        bg_above_list.append(img[1:2])    # channel 1 = above
        bg_below_list.append(img[2:3])    # channel 2 = below
        bg_path_list.append(sample["pathology"])
        bg_clinical_list.append(sample["clinical"])

    # 注意: 背景直接用手样本 (不是均值), 见下方 bg_ct_flat
    bg_above_mean = torch.stack(bg_above_list).mean(dim=0, keepdim=True).to(DEVICE)
    bg_below_mean = torch.stack(bg_below_list).mean(dim=0, keepdim=True).to(DEVICE)
    bg_path_mean = torch.stack(bg_path_list).mean(dim=0, keepdim=True).to(DEVICE)
    bg_clinical_mean = torch.stack(bg_clinical_list).mean(dim=0, keepdim=True).to(DEVICE)
    print(f"[2/4] 背景: {len(bg_indices)} 个训练样本 (seed={RANDOM_STATE})")

    # ── 3. 按 label_f.csv 划分挑选病例（仅 max 视图）────────────────
    patients = collect_patients_by_split(args.splits)
    if args.limit:
        patients = patients[:args.limit]

    missing_files = [pid for pid in patients if not (CT_DIR / f"{pid}_ct_max.png").exists()]
    patients = [pid for pid in patients if pid not in set(missing_files)]
    if not patients:
        raise FileNotFoundError(f"{CT_DIR} 中没有可用病例的 *_ct_max.png")
    print(f"[3/4] 病例: {len(patients)} 例 "
          f"(来自 {LABEL_CSV.name} 的划分 {' '.join(args.splits)})")
    if missing_files:
        preview = " ".join(missing_files[:10]) + (" ..." if len(missing_files) > 10 else "")
        print(f"      [警告] {len(missing_files)} 例在 {CT_DIR.name} 中缺 _ct_max.png: {preview}")

    test_tensors = []      # 展平的 (H*W, ) tensors
    test_pad_params = []   # pad 参数
    test_display_imgs = [] # 原始 CT 图 (numpy)
    test_ids = []

    for pid in patients:
        path = CT_DIR / f"{pid}_ct_max.png"
        test_ids.append(pid)

        tensor, pad_params = preprocess_max_view(str(path))
        test_tensors.append(tensor.flatten())         # (128*128,)
        test_pad_params.append(pad_params)

        # 加载原始 CT 图用于显示 (与模型输入同源同一张图)
        orig_img = np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0
        test_display_imgs.append(orig_img)

    test_ct = torch.stack(test_tensors).to(DEVICE)  # (N, 128*128)

    # ── 4. SHAP 计算 + 可视化 ───────────────────────────────────────
    wrapper = CTMaxOnlyWrapper(
        model, bg_above_mean, bg_below_mean,
        bg_path_mean, bg_clinical_mean, IMG_SIZE, DEVICE
    )

    # 背景
    bg_ct_flat = torch.stack([s.flatten() for s in bg_max_list]).to(DEVICE)

    print("[4/4] 计算 SHAP (GradientExplainer)...")
    explainer = shap.GradientExplainer(wrapper, bg_ct_flat)

    shap_batches = []
    for start in range(0, len(test_ct), 4):
        batch = test_ct[start:start + 4]
        sv = explainer.shap_values(batch)
        shap_batches.append(sv if isinstance(sv, np.ndarray) else sv[0])

    shap_values = np.concatenate(shap_batches, axis=0)  # (N, 16384, 1)
    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values.squeeze(-1)            # (N, 16384)
    print(f"  SHAP shape: {shap_values.shape}")

    # ── 5. 可视化：SHAP 逆映射到原始 CT 图坐标 ──────────────────────
    os.makedirs(save_dir, exist_ok=True)

    for i in range(len(test_ct)):
        pid = test_ids[i]
        sv_1ch = shap_values[i].reshape(IMG_SIZE, IMG_SIZE)
        display_img = test_display_imgs[i]
        pad_params = test_pad_params[i]

        # SHAP 逆映射到原始 CT 图坐标
        sv_orig = shap_to_original(sv_1ch, pad_params)
        H_o, W_o = display_img.shape

        # 确保尺寸对齐
        if sv_orig.shape != (H_o, W_o):
            sv_orig = cv2.resize(sv_orig, (W_o, H_o), interpolation=cv2.INTER_LINEAR)

        # ── 身体轮廓掩膜: 原图二值化(>0), 乘掉黑色背景区域的 SHAP ──
        body_mask = (display_img > 0).astype(np.float32)
        sv_orig = sv_orig * body_mask

        vmax = max(abs(sv_orig.min()), abs(sv_orig.max()))
        if vmax == 0:
            vmax = 1e-6

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 左: 原始 CT (max 视图, 模型输入)
        axes[0].imshow(display_img, cmap="gray")
        axes[0].set_title(f"CT max view\nPatient {pid}", fontsize=11)
        axes[0].axis("off")

        # 中: SHAP 叠加原图
        axes[1].imshow(display_img, cmap="gray")
        im = axes[1].imshow(sv_orig, cmap="seismic", vmin=-vmax, vmax=vmax,
                            alpha=0.5)
        axes[1].set_title(f"SHAP Overlay\n(red=+, blue=–)", fontsize=11)
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        # 右: 纯 SHAP 热力图
        im2 = axes[2].imshow(sv_orig, cmap="seismic", vmin=-vmax, vmax=vmax)
        axes[2].set_title("SHAP Values", fontsize=11)
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        # 页脚注明模型标识与归因范围, 避免图脱离上下文被误读
        fig.text(
            0.5, 0.02,
            f"CT max-view attribution only | above/below views, pathology and "
            f"clinical fixed at train-set means | background-masked (img>0) | {model_tag}",
            ha="center", fontsize=9, color="gray",
        )

        save_path = os.path.join(save_dir, f"{pid}_ct_shap.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

        if (i + 1) % 20 == 0 or (i + 1) == len(test_ct):
            print(f"  已保存 {i + 1}/{len(test_ct)}")

    print(f"\n全部完成! 结果保存在: {os.path.abspath(save_dir)}")


if __name__ == "__main__":
    main()

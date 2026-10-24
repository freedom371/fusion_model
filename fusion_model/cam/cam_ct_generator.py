"""
CT 图像 GradCAM 热力图生成器 — 适配 CrossModalTransformerFusion (Swin-Tiny backbone)。

为融合模型的 CT 分支生成 GradCAM 可视化热力图：
  - Backbone: Swin-Tiny (timm)
  - 目标层: layers[3].blocks[-1].norm2 (最后一个 Swin block 的 LayerNorm)
  - reshape_transform: 将 (B, num_patches, C) 序列重塑为 (B, C, H, W)
  - 单张 CT 图像复制到 3 通道以匹配模型的三视图输入
"""
import torch
import torch.nn as nn
import numpy as np
import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from PIL import Image
from scipy import ndimage

import torchvision.transforms.functional as TF

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import CrossModalTransformerFusion
from config import *

# ── 原始 CT 图像目录（未裁剪，用于可视化叠底）────────────────────
DATA_CT_ORIGINAL_DIR = ROOT / "data/images/ct_5"

# ── 裁剪参数（与 crop_tumor.py 保持一致）──────────────────────────
_CROP_EXPAND_FACTOR = 1.35
_CROP_PADDING = 5


def _expand_mask(mask: np.ndarray, factor: float) -> np.ndarray:
    """用距离变换向外膨胀 mask，保持轮廓形状。与 crop_tumor.py 一致。"""
    if factor <= 1.0:
        return mask
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if not rows.any():
        return mask
    rmin, rmax = rows.nonzero()[0][[0, -1]]
    cmin, cmax = cols.nonzero()[0][[0, -1]]
    h, w = rmax - rmin + 1, cmax - cmin + 1
    radius = int(min(h, w) / 2 * (factor - 1))
    if radius <= 0:
        return mask
    dist = ndimage.distance_transform_edt(~mask)
    return dist <= radius


def _get_mask_bbox(mask: np.ndarray, padding: int):
    """获取膨胀后 mask 的 bbox（含额外边距），与 crop_tumor.py 一致。"""
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if not rows.any():
        return None
    rmin, rmax = rows.nonzero()[0][[0, -1]]
    cmin, cmax = cols.nonzero()[0][[0, -1]]
    h, w = mask.shape
    rmin = max(0, rmin - padding)
    rmax = min(h - 1, rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(w - 1, cmax + padding)
    return rmin, rmax, cmin, cmax
from dataset import (get_train_stats, pid_to_img_path,
                     pid_to_mask_path, _get_base_pid)

# ── Swin Transformer reshape_transform ─────────────────────────────────
# 在 IMG_SIZE=128 下，layers[3] 的特征图大小为 128/32 = 4×4
_SWIN_GRID_SIZE = IMG_SIZE // 32  # 4


def swin_reshape_transform(tensor: torch.Tensor, grid: int = _SWIN_GRID_SIZE):
    """将 Swin Transformer 的序列输出 (B, L, C) 重塑为 (B, C, H, W)。

    Args:
        tensor: (B, num_patches, embed_dim) 特征序列
        grid: 空间网格大小 (默认 4×4)

    Returns:
        (B, C, grid, grid) 空间特征图
    """
    # tensor shape: (B, L, C) where L = grid * grid
    result = tensor.reshape(tensor.size(0), grid, grid, tensor.size(2))
    result = result.permute(0, 3, 1, 2)  # → (B, C, grid, grid)
    return result


# ── CT-only 模型包装器 ─────────────────────────────────────────────────
class CTOnlyWrapper(nn.Module):
    """将 CrossModalTransformerFusion 包装为仅接受 CT 图像的模型。

    病理特征和临床特征固定为指定患者的实际数据，CT 单通道图像
    自动复制为 3 通道以匹配模型的三视图输入格式。
    """

    def __init__(self, model: CrossModalTransformerFusion,
                 pathology: torch.Tensor, clinical: torch.Tensor,
                 device: torch.device):
        super().__init__()
        self.model = model
        self._device = device
        self._pathology = pathology.to(device)
        self._clinical = clinical.to(device)

    def forward(self, x):
        batch_size = x.size(0)
        # 单通道 → 3 通道 (匹配模型的三视图输入)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        path = self._pathology.expand(batch_size, -1)
        clin = self._clinical.expand(batch_size, -1)
        return self.model(x, path, clin)


# ── GradCAM 生成器 ─────────────────────────────────────────────────────
class ResidualCAMGenerator:
    """为 CrossModalTransformerFusion 的 CT 分支生成 GradCAM 热力图。

    使用 Swin-Tiny 最后一个 Transformer block 的 LayerNorm 作为目标层，
    通过 reshape_transform 将 patch token 序列还原为空间热力图。
    """

    def __init__(self, model_path: str, device: torch.device = None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_path = model_path
        self.model = self._load_model()
        self._unfreeze_model()
        self._init_normalization_stats()

    def _load_model(self) -> CrossModalTransformerFusion:
        print(f"[模型] 加载: {self.model_path}")
        ckpt = torch.load(self.model_path, map_location=self.device,
                          weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        model = CrossModalTransformerFusion(
            dropout=0.3, freeze_ct_backbone=False
        ).to(self.device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[模型] 加载成功 — {total_params:.1f}M 参数, 设备: {self.device}")
        return model

    def _unfreeze_model(self):
        """解冻所有参数以允许梯度计算（GradCAM 需要）。"""
        for param in self.model.parameters():
            param.requires_grad = True

    def _init_normalization_stats(self):
        """从训练集计算病理/临床特征的归一化统计量，确保与训练时一致。"""
        print("[统计量] 计算特征归一化统计量 (使用训练集)...")
        path_mean, path_std, clinical_mean, clinical_std, col_means = get_train_stats()
        self.path_mean = torch.from_numpy(path_mean).to(self.device)
        self.path_std = torch.from_numpy(path_std).to(self.device)
        self.clinical_mean = torch.from_numpy(clinical_mean).to(self.device)
        self.clinical_std = torch.from_numpy(clinical_std).to(self.device)
        # train 集各列均值，用于填充 NaN（与 CTMultiViewDataset 行为一致）
        self.clinical_col_means = np.asarray(col_means, dtype=np.float32)
        # 临床表只解析一次，避免每张图重复读 CSV
        self.clinical_df = load_clinical_df()
        print(f"  病理特征: mean={tuple(self.path_mean.shape)}, std={tuple(self.path_std.shape)}")
        print(f"  临床特征: mean={tuple(self.clinical_mean.shape)}, std={tuple(self.clinical_std.shape)}")

    def _find_target_layer(self):
        """定位 Swin-Tiny backbone 最后一个 Transformer block 的 LayerNorm。

        Swin-Tiny 结构: layers[0..3], 每层含多个 SwinTransformerBlock。
        layers[3].blocks[-1].norm2 是最后一层第二个 LayerNorm。
        """
        backbone = self.model.ct_encoder.backbone
        target = backbone.layers[3].blocks[-1].norm2
        print(f"[目标层] layers[3].blocks[-1].norm2 ({type(target).__name__})")
        return [target]

    def _load_patient_features(self, base_id: str):
        """加载指定患者的病理特征和临床特征（已归一化）。

        Args:
            base_id: 患者 base ID (不含 .nii.gz 后缀)

        Returns:
            (pathology_tensor, clinical_tensor): 已归一化的特征张量
        """
        # ── 病理特征 ──
        feat_path = DATA_PATH_DIR / base_id / f"{base_id}_uni2_slide_features.npy"
        if not feat_path.exists():
            raise FileNotFoundError(f"病理特征文件不存在: {feat_path}")
        pathology = torch.from_numpy(
            np.load(feat_path).squeeze(0).astype(np.float32)
        )
        pathology = (pathology - self.path_mean.cpu()) / (self.path_std.cpu() + 1e-8)

        # ── 临床特征（与 CTMultiViewDataset 保持一致：CLINICAL_FEATURE_COLS，6 列）──
        # load_clinical_df() 内部已 skiprows=[1] 跳过 CSV 第 2 行英文表头，
        # 并把索引规范化为不含 .nii.gz 后缀的 base_id
        if base_id not in self.clinical_df.index:
            raise ValueError(f"无法找到患者 {base_id} 的临床数据")

        clinical_raw = (
            self.clinical_df.loc[base_id, CLINICAL_FEATURE_COLS]
            .values.astype(np.float32)
        )
        # NaN 填充为 train 集各列均值（与 dataset 行为一致）
        nan_mask = np.isnan(clinical_raw)
        if nan_mask.any():
            clinical_raw = np.where(nan_mask, self.clinical_col_means, clinical_raw)
        clinical = torch.from_numpy(clinical_raw)
        clinical = (clinical - self.clinical_mean.cpu()) / (self.clinical_std.cpu() + 1e-8)

        return pathology, clinical

    def generate_gradcam(self, image_tensor: torch.Tensor, base_id: str,
                         target_class: int = 1) -> np.ndarray:
        """为一张 CT 图像生成 GradCAM 热力图。

        Args:
            image_tensor: CT 图像张量 (1, H, W), 已应用 mask 并归一化
            base_id: 患者 base ID
            target_class: 目标类别 (1=复发/阳性, 0=无复发/阴性)

        Returns:
            np.ndarray: 灰度 GradCAM 热力图 (H, W)，范围 [0, 1]
        """
        pathology, clinical = self._load_patient_features(base_id)
        wrapper = CTOnlyWrapper(self.model, pathology, clinical, self.device)
        target_layers = self._find_target_layer()

        cam = GradCAM(
            model=wrapper,
            target_layers=target_layers,
            reshape_transform=swin_reshape_transform,
        )

        # 模型输出为 (B, 1) 单 logit
        # 索引 [0] 对应阳性类（高 logit = 复发）
        targets = [ClassifierOutputTarget(0)]

        grayscale_cam = cam(
            input_tensor=image_tensor.unsqueeze(0).to(self.device),
            targets=targets,
        )
        # target_class=1: 阳性（保持原值，logit 越高激活越强）
        # target_class=0: 阴性（取反，低 logit 区域激活更强）
        if target_class == 0:
            grayscale_cam = -grayscale_cam
        return grayscale_cam[0]

    def process_single_image(self, img_path: str, mask_path: str, base_id: str,
                             target_class: int = 1) -> np.ndarray:
        """处理单张 CT 图像，返回在原始 ct_5 图上叠加 CAM 的可视化图。

        模型输入使用 ct5_cropped（裁剪图），可视化叠底使用 ct_5（原始大图）。
        CAM 热力图通过裁剪 bbox + pad/resize 的逆映射回原始坐标。

        Args:
            img_path: ct5_cropped 裁剪图路径
            mask_path: mask_p5 原始 mask 路径
            base_id: 患者 base ID
            target_class: 目标类别

        Returns:
            np.ndarray: 叠加后的可视化图 (H, W, 3)，范围 [0, 1]
        """
        # ── 1. 加载原始 ct_5 大图和 mask ──
        orig_ct_path = DATA_CT_ORIGINAL_DIR / f"{base_id}_ct_max.png"
        if not orig_ct_path.exists():
            raise FileNotFoundError(f"原始 CT 图不存在: {orig_ct_path}")
        orig_img_pil = Image.open(orig_ct_path).convert("L")
        orig_img_np = np.array(orig_img_pil).astype(np.float32) / 255.0
        H_orig, W_orig = orig_img_np.shape

        mask_full_pil = Image.open(mask_path).convert("L")
        mask_full_np = np.array(mask_full_pil).astype(np.uint8)

        # ── 2. 计算裁剪 bbox（与 crop_tumor.py 一致）──
        if np.any(mask_full_np > 0):
            expanded = _expand_mask(mask_full_np > 0, _CROP_EXPAND_FACTOR)
            bbox = _get_mask_bbox(expanded, _CROP_PADDING)
            if bbox is None:
                bbox = (0, H_orig - 1, 0, W_orig - 1)
        else:
            bbox = (0, H_orig - 1, 0, W_orig - 1)
        rmin, rmax, cmin, cmax = bbox

        # ── 3. CAM 生成（与训练一致：ct5_cropped → pad → resize → 模型）──
        img_cropped_pil = Image.open(img_path)
        W_c, H_c = img_cropped_pil.size

        # 记录 pad 参数，用于 CAM 逆映射
        S = max(W_c, H_c)
        pw, ph = (S - W_c) // 2, (S - H_c) // 2

        img_pil = TF.pad(img_cropped_pil, (pw, ph, S - W_c - pw, S - H_c - ph), fill=0)
        img_pil = TF.resize(img_pil, (IMG_SIZE, IMG_SIZE),
                            interpolation=TF.InterpolationMode.BILINEAR)

        img_tensor = TF.to_tensor(img_pil)  # (1, H, W)
        img_tensor = TF.normalize(img_tensor, mean=[0.5], std=[0.5])

        grayscale_cam = self.generate_gradcam(img_tensor, base_id, target_class)
        grayscale_cam = np.clip(grayscale_cam, 0, 1)  # (128, 128)

        # ── 4. CAM 逆映射到原始 ct_5 坐标 ──
        # 128×128 → S×S → unpad → (H_c, W_c) → place at bbox
        cam_resized = cv2.resize(grayscale_cam, (S, S))
        cam_cropped = cam_resized[ph:ph + H_c, pw:pw + W_c]

        cam_full = np.zeros((H_orig, W_orig), dtype=np.float32)
        # 确保 bbox 尺寸与 cam_cropped 对齐
        bbox_h, bbox_w = rmax - rmin + 1, cmax - cmin + 1
        if bbox_h != H_c or bbox_w != W_c:
            cam_cropped = cv2.resize(cam_cropped, (bbox_w, bbox_h))
        cam_full[rmin:rmax + 1, cmin:cmax + 1] = cam_cropped

        # ── 5. CAM 上色后 mask 非肿瘤区域 ──
        # 注意：必须先上色再 mask，否则 JET colormap 会把 0 映射为深蓝色
        heatmap_full = cv2.applyColorMap(
            np.uint8(255 * cam_full), cv2.COLORMAP_JET
        )
        heatmap_full = cv2.cvtColor(heatmap_full, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # mask 乘黑非肿瘤区域（此时 heatmap 已上色，0→真黑色）
        mask_3ch = np.stack([mask_full_np.astype(np.float32) / 255.0] * 3, axis=-1)
        heatmap_full = heatmap_full * mask_3ch

        # ── 6. 可视化：原始 ct_5 + 50% CAM 叠加 + 绿色肿瘤轮廓 ──
        img_3ch = np.stack([orig_img_np] * 3, axis=-1)

        cam_img = img_3ch + 0.5 * heatmap_full
        cam_img = np.clip(cam_img, 0, 1)

        # 绿色轮廓线（基于原始 mask）
        contours, _ = cv2.findContours(mask_full_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cam_img_uint8 = (cam_img * 255).astype(np.uint8)
        cv2.drawContours(cam_img_uint8, contours, -1, (0, 255, 0), 1)
        cam_img = cam_img_uint8.astype(np.float32) / 255.0

        return cam_img

    def process_patient(self, base_id: str, output_dir: str,
                        target_class: int = 1):
        """处理单个患者的所有 CT 切面并保存 CAM 图。

        Args:
            base_id: 患者 base ID
            output_dir: 输出目录
            target_class: 目标类别
        """
        os.makedirs(output_dir, exist_ok=True)

        # CT 三切面视图
        views = [
            ("ct_max", f"{base_id}.nii.gz"),
        ]

        for view_name, pid in views:
            img_path = pid_to_img_path(pid, DATA_CT_DIR)
            mask_path = pid_to_mask_path(pid, DATA_MASK_DIR)

            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                print(f"  跳过 {view_name}: 文件不存在 "
                      f"({img_path} 或 {mask_path})")
                continue

            print(f"  处理 {view_name}: {img_path}")
            try:
                cam_img = self.process_single_image(
                    str(img_path), str(mask_path), base_id, target_class
                )
                save_path = os.path.join(
                    output_dir, f"{base_id}_{view_name}_gradcam.png"
                )
                self._save_result(cam_img, save_path, f"{base_id}_{view_name}")
            except Exception as e:
                print(f"  处理 {view_name} 时出错: {e}")
                import traceback
                traceback.print_exc()

    def _save_result(self, cam_img: np.ndarray, save_path: str, title: str):
        """保存 GradCAM 可视化结果为 PNG 文件。"""
        plt.figure(figsize=(5, 5))
        plt.imshow(cam_img)
        plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(f"Grad-CAM ({title})", fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  已保存: {os.path.abspath(save_path)}")

    def process_dataset(self, split: str = "test",
                        output_root: str = None,
                        target_class: int = 1,
                        max_patients: int = None):
        """批量处理某个数据集划分的所有患者。

        Args:
            split: 数据集划分 ("train", "val", "test", "all")
            output_root: 输出根目录
            target_class: 目标类别 (1=复发, 0=无复发)
            max_patients: 最多处理患者数 (None = 全部)
        """
        if output_root is None:
            output_root = os.path.join(os.path.dirname(__file__), "output")

        label_df = pd.read_csv(LABEL_CSV)

        if split == "train":
            ids = label_df[label_df["group"] == "train"]["ID"].tolist()
        elif split == "val":
            ids = label_df[label_df["group"] == "val"]["ID"].tolist()
        elif split == "test":
            ids = label_df[
                label_df["group"].isin(["test1", "test2", "test3", "test4"])
            ]["ID"].tolist()
        elif split == "all":
            # all = train + val + test*（排除 exclude 组, 口径与 ct_shap.py 一致）
            groups = label_df["group"].astype(str)
            ids = label_df[
                groups.isin(["train", "val", "test"]) | groups.str.startswith("test")
            ]["ID"].tolist()
        else:
            raise ValueError(f"未知 split: {split}")

        # 解析 base_id 列表（去重）
        base_ids = list(dict.fromkeys(_get_base_pid(pid) for pid in ids))
        if max_patients:
            base_ids = base_ids[:max_patients]

        print(f"处理 {split} 集: {len(base_ids)} 个患者")

        for i, base_id in enumerate(base_ids):
            print(f"\n[{i+1}/{len(base_ids)}] 患者 {base_id}")
            patient_output_dir = os.path.join(output_root, base_id)
            self.process_patient(base_id, patient_output_dir, target_class)

        print(f"\n全部处理完成，结果保存在: {os.path.abspath(output_root)}")


def main():
    """主函数：配置参数并运行 CAM 生成。"""
    config = {
        "model_path": str(
            Path(__file__).resolve().parents[1]
            / "output" / "cmt_cpg" / "checkpoints" / "best_model_seed1583.pt"
        ),
        # 输出按 seed 分开存放, 避免覆盖旧 seed 的结果
        "output_root": str(
            Path(__file__).resolve().parent / "output_seed1583"
        ),
        "split": "all",          # all = train + val + test (label_l 共 649 例, 已排除 exclude 组)
        "target_class": 1,       # 1=复发, 0=无复发
        "max_patients": None,    # None = 全部, 设为数字可限制测试数量
    }

    print("=" * 60)
    print("CT GradCAM 生成器 (CrossModalTransformerFusion + Swin-Tiny)")
    print(f"模型路径: {config['model_path']}")
    print(f"模型文件存在: {os.path.exists(config['model_path'])}")
    print(f"IMG_SIZE: {IMG_SIZE}")
    print(f"输出目录: {config['output_root']}")
    print("=" * 60)

    generator = ResidualCAMGenerator(config["model_path"])
    print("模型加载成功\n")

    generator.process_dataset(
        split=config["split"],
        output_root=config["output_root"],
        target_class=config["target_class"],
        max_patients=config["max_patients"],
    )


if __name__ == "__main__":
    main()

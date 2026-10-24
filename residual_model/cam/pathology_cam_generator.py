"""
针对病理 WSI 的 UNI2 模型生成 GradCAM 热力图。

UNI2 = ViT-G + 寄存器 token（no_embed_class=True，9 个前缀 token：
1 CLS + 8 register），通过 reshape_transform 将 256 个 patch tokens
转为 16×16 空间热力图，再上采样回原图尺寸。
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys
import gc
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import timm
import torchvision.transforms.functional as TF

from pytorch_grad_cam import GradCAM

# ── 路径 ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNI2_WEIGHTS_PATH = PROJECT_ROOT / "feature_extraction" / "uni2" / "pytorch_model.bin"
TILE_ROOT = Path("/hard0/carcinoma/jiandu/yansebiaozhunhua")
LABEL_CSV = PROJECT_ROOT / "data" / "cvs" / "label.csv"

# UNI2 模型参数（与 lazyslide 一致）
UNI2_NUM_PREFIX = 9       # 1 CLS + 8 register tokens
UNI2_NUM_PATCHES = 256    # (224/14)^2 = 16x16
UNI2_GRID_SIZE = 16
UNI2_EMBED_DIM = 1536
UNI2_IMG_SIZE = 224
UNI2_MEAN = [0.485, 0.456, 0.406]
UNI2_STD  = [0.229, 0.224, 0.225]


# ── ViT reshape（去除前缀 token）────────────────────────────────────
def vit_reshape_transform(tensor: torch.Tensor,
                           num_prefix: int = UNI2_NUM_PREFIX,
                           grid: int = UNI2_GRID_SIZE):
    """去除 CLS + register tokens，将 patch tokens 重塑为 (B, C, H, W)。"""
    # tensor shape: (B, num_prefix + num_patches, embed_dim)
    result = tensor[:, num_prefix:, :].reshape(tensor.size(0), grid, grid, tensor.size(2))
    result = result.permute(0, 3, 1, 2)  # → (B, C, H, W)
    return result


# ── L2 norm target ──────────────────────────────────────────────────
class L2NormTarget:
    """最大化 UNI2 特征向量的 L2 范数。
    pytorch_grad_cam 内部用 zip(targets, outputs) 遍历 batch，
    传入的 model_output 是单个样本的特征向量 (D,)，而非 (B, D)。
    """
    def __call__(self, model_output):
        # model_output 可能是 (D,) 或 (B, D)
        if model_output.ndim == 1:
            return model_output.norm(p=2)
        return model_output.norm(p=2, dim=1)


# ── Tile Dataset ────────────────────────────────────────────────────
class TileDataset(Dataset):
    """加载 WSI tile 图像并做 UNI2 预处理。"""
    def __init__(self, tile_paths: list[str]):
        self.tile_paths = tile_paths

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx):
        path = self.tile_paths[idx]
        img = Image.open(path).convert("RGB")
        img = TF.resize(img, (UNI2_IMG_SIZE, UNI2_IMG_SIZE),
                        interpolation=TF.InterpolationMode.BILINEAR)
        tensor = TF.to_tensor(img)
        tensor = TF.normalize(tensor, mean=UNI2_MEAN, std=UNI2_STD)
        return tensor, path


# ── Pathology CAM Generator ─────────────────────────────────────────
class PathologyCAMGenerator:
    """加载 UNI2 模型，为 WSI tile 生成 GradCAM 热力图。"""

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_uni2_model()

    def _load_uni2_model(self):
        print(f"[UNI2] 加载模型: {UNI2_WEIGHTS_PATH}")
        model = timm.create_model(
            "vit_giant_patch14_224",
            pretrained=False,
            img_size=224,
            patch_size=14,
            depth=24,
            num_heads=24,
            init_values=1e-5,
            embed_dim=UNI2_EMBED_DIM,
            mlp_ratio=2.66667 * 2,
            num_classes=0,
            no_embed_class=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            reg_tokens=8,
            global_pool="token",
        )
        state_dict = torch.load(UNI2_WEIGHTS_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()

        for param in model.parameters():
            param.requires_grad = True

        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[UNI2] 加载成功 — {n_params:.1f}M 参数, 设备: {self.device}")
        return model

    def _get_target_layer(self):
        return [self.model.blocks[-1].norm1]

    def _preprocess_image(self, img_path: str) -> torch.Tensor:
        img = Image.open(img_path).convert("RGB")
        img = TF.resize(img, (UNI2_IMG_SIZE, UNI2_IMG_SIZE),
                        interpolation=TF.InterpolationMode.BILINEAR)
        tensor = TF.to_tensor(img)
        tensor = TF.normalize(tensor, mean=UNI2_MEAN, std=UNI2_STD)
        return tensor

    def generate_tile_cam(self, tile_path: str) -> np.ndarray:
        """为单张 tile 生成 CAM (224×224, 范围 [0,1])。"""
        tensor = self._preprocess_image(tile_path)
        cam = GradCAM(
            model=self.model,
            target_layers=self._get_target_layer(),
            reshape_transform=vit_reshape_transform,
        )
        grayscale_cam = cam(
            input_tensor=tensor.unsqueeze(0).to(self.device),
            targets=[L2NormTarget()],
        )
        return np.clip(grayscale_cam[0], 0, 1)

    def generate_tile_cam_batch(self, tile_paths: list[str],
                                batch_size: int = 8) -> dict[str, np.ndarray]:
        """批量生成 tile CAM。"""
        results = {}
        dataset = TileDataset(tile_paths)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        cam = GradCAM(
            model=self.model,
            target_layers=self._get_target_layer(),
            reshape_transform=vit_reshape_transform,
        )

        for batch_tensors, batch_paths in loader:
            batch_tensors = batch_tensors.to(self.device)
            grayscale_cams = cam(
                input_tensor=batch_tensors,
                targets=[L2NormTarget()],
            )
            for i, path in enumerate(batch_paths):
                results[path] = np.clip(grayscale_cams[i], 0, 1)

        return results

    def overlay_cam_on_tile(self, tile_path: str, cam_map: np.ndarray,
                           alpha: float = 0.5) -> np.ndarray:
        """将 CAM 叠加到 tile 原图上。"""
        original = Image.open(tile_path).convert("RGB")
        original = original.resize((UNI2_IMG_SIZE, UNI2_IMG_SIZE), Image.BILINEAR)
        img_np = np.array(original).astype(np.float32) / 255.0

        heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        overlay = alpha * heatmap + (1 - alpha) * img_np
        return np.clip(overlay, 0, 1)

    def _save_cam_image(self, cam_img: np.ndarray, save_path: str, title: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(4, 4))
        plt.imshow(cam_img)
        plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(title, fontsize=8)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def process_patient_tiles(self, base_id: str, output_dir: str,
                              max_tiles: int = 20, batch_size: int = 8):
        """处理单个患者的部分 tile，保存 CAM 图和概览图。"""
        tile_dir = TILE_ROOT / base_id
        if not tile_dir.exists():
            print(f"  tile 目录不存在: {tile_dir}，跳过")
            return

        all_tiles = sorted(p for p in tile_dir.iterdir() if p.suffix == ".jpg")
        if not all_tiles:
            print(f"  无 tile 文件: {tile_dir}，跳过")
            return

        if len(all_tiles) > max_tiles:
            indices = np.linspace(0, len(all_tiles) - 1, max_tiles, dtype=int)
            tiles = [all_tiles[i] for i in indices]
        else:
            tiles = all_tiles

        print(f"  处理 {len(tiles)}/{len(all_tiles)} tiles (患者 {base_id})")

        patient_out = os.path.join(output_dir, base_id)
        os.makedirs(patient_out, exist_ok=True)

        tile_paths = [str(t) for t in tiles]
        cam_results = self.generate_tile_cam_batch(tile_paths, batch_size)

        for tile_path_str, cam_map in cam_results.items():
            tile_name = os.path.basename(tile_path_str).replace(".jpg", "")
            cam_overlay = self.overlay_cam_on_tile(tile_path_str, cam_map)
            save_path = os.path.join(patient_out, f"{tile_name}_cam.png")
            self._save_cam_image(cam_overlay, save_path, tile_name[:30])

        self._create_overview(cam_results, patient_out, base_id)

        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _create_overview(self, cam_results: dict[str, np.ndarray],
                         patient_out: str, base_id: str):
        """将 tile CAM 拼合成概览图。"""
        tile_list = list(cam_results.items())
        n = len(tile_list)
        if n == 0:
            return

        cols = min(5, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = np.atleast_2d(axes)

        for idx, (tile_path, cam_map) in enumerate(tile_list):
            r, c = idx // cols, idx % cols
            overlay = self.overlay_cam_on_tile(tile_path, cam_map)
            axes[r, c].imshow(overlay)
            tile_name = os.path.basename(tile_path).replace(".jpg", "")[:20]
            axes[r, c].set_title(tile_name, fontsize=6)
            axes[r, c].axis("off")

        for idx in range(n, rows * cols):
            r, c = idx // cols, idx % cols
            axes[r, c].axis("off")

        plt.suptitle(f"Patient {base_id} — Tile CAM Overview", fontsize=10, fontweight="bold")
        plt.tight_layout()
        overview_path = os.path.join(patient_out, f"{base_id}_overview.png")
        plt.savefig(overview_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  概览图已保存: {overview_path}")

    def process_dataset(self, split: str = "test", output_root: str = None,
                        max_tiles: int = 20, batch_size: int = 8,
                        max_patients: int = None):
        """批量处理某个数据集划分的所有患者。"""
        import pandas as pd
        from dataset import _get_base_pid

        if output_root is None:
            output_root = os.path.join(os.path.dirname(__file__), "pathology_cam_output")

        label_df = pd.read_csv(LABEL_CSV)

        if split == "train":
            ids = label_df[label_df["group"] == "train"]["ID"].tolist()
        elif split == "val":
            ids = label_df[label_df["group"] == "val"]["ID"].tolist()
        elif split == "test":
            ids = label_df[label_df["group"].isin(["test1", "test2", "test3", "test4"])]["ID"].tolist()
        else:
            raise ValueError(f"未知 split: {split}")

        base_ids = list(dict.fromkeys(_get_base_pid(pid) for pid in ids))
        if max_patients:
            base_ids = base_ids[:max_patients]

        print(f"处理 {split} 集: {len(base_ids)} 个患者")
        print(f"每患者最多 {max_tiles} tiles, batch_size={batch_size}")

        for i, base_id in enumerate(base_ids):
            print(f"\n[{i+1}/{len(base_ids)}] 患者 {base_id}")
            try:
                self.process_patient_tiles(base_id, output_root, max_tiles, batch_size)
            except Exception as e:
                print(f"  处理患者 {base_id} 时出错: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n全部完成，结果保存在: {os.path.abspath(output_root)}")


def main():
    config = {
        "split": "test",
        "output_root": str(PROJECT_ROOT / "residual_model" / "cam" / "pathology_cam_output"),
        "max_tiles": 20,
        "batch_size": 8,
        "max_patients": None,   # None = 全部患者
    }

    print("=" * 60)
    print("病理特征 CAM 生成器 (UNI2 ViT + GradCAM)")
    print(f"UNI2 模型: {UNI2_WEIGHTS_PATH}")
    print(f"Tile 数据: {TILE_ROOT}")
    print(f"配置: {config}")
    print("=" * 60)

    generator = PathologyCAMGenerator()
    generator.process_dataset(
        split=config["split"],
        output_root=config["output_root"],
        max_tiles=config["max_tiles"],
        batch_size=config["batch_size"],
        max_patients=config["max_patients"],
    )


if __name__ == "__main__":
    main()

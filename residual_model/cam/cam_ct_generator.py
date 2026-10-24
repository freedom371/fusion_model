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
from PIL import Image

import torchvision.transforms.functional as TF

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import MultiModalTransformer
from config import *
from dataset import CTMultiViewDataset, pid_to_img_path, pid_to_mask_path, _get_base_pid


class CTOnlyWrapper(nn.Module):
    """
    只对 CT 图像做 GradCAM，病理和临床特征固定为患者实际数据。
    """
    def __init__(self, model: MultiModalTransformer, pathology: torch.Tensor, clinical: torch.Tensor,
                 device: torch.device):
        super().__init__()
        self.model = model
        self._device = device
        self._pathology = pathology.to(device)
        self._clinical = clinical.to(device)

    def forward(self, x):
        batch_size = x.size(0)
        # 与训练时一致的预处理: 单通道 → 3通道
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        path = self._pathology.expand(batch_size, -1)
        clin = self._clinical.expand(batch_size, -1)
        return self.model(x, path, clin)


class ResidualCAMGenerator:
    """
    为 MultiModalTransformer 生成 CT 图像的 GradCAM 热力图。
    """
    def __init__(self, model_path: str, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = self._load_model()
        self._unfreeze_model()
        self._init_normalization_stats()

    def _load_model(self) -> MultiModalTransformer:
        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        model = MultiModalTransformer(dropout=0.3).to(self.device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print(f"模型已加载: {self.model_path}")
        return model

    def _unfreeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def _init_normalization_stats(self):
        """从训练集计算病理/临床特征的归一化统计量，确保与训练时一致。"""
        print("计算特征归一化统计量 (使用训练集)...")
        ds_train = CTMultiViewDataset("train", use_aug=False)
        self.path_mean = torch.from_numpy(ds_train.path_mean).to(self.device)
        self.path_std = torch.from_numpy(ds_train.path_std).to(self.device)
        self.clinical_mean = torch.from_numpy(ds_train.clinical_mean).to(self.device)
        self.clinical_std = torch.from_numpy(ds_train.clinical_std).to(self.device)
        print(f"  病理特征归一化: mean={self.path_mean.shape}, std={self.path_std.shape}")
        print(f"  临床特征归一化: mean={self.clinical_mean.shape}, std={self.clinical_std.shape}")

    def _find_target_layer(self):
        """找到 DenseNet121 backbone 的最后一个 norm 层作为 GradCAM 目标层。"""
        backbone = self.model.ct_encoder.backbone
        return [backbone.features.norm5]

    def _load_patient_features(self, base_id: str):
        """加载指定患者 (base_id) 的病理特征和临床特征 (已归一化)。"""
        # ── 病理特征 ──
        feat_path = DATA_PATH_DIR / base_id / f"{base_id}_uni2_slide_features.npy"
        if not feat_path.exists():
            raise FileNotFoundError(f"病理特征文件不存在: {feat_path}")
        pathology = torch.from_numpy(
            np.load(feat_path).squeeze(0).astype(np.float32)
        )
        pathology = (pathology - self.path_mean.cpu()) / (self.path_std.cpu() + 1e-8)

        # ── 临床特征 ──
        clinical_cols = ["cAJCC", "TLS", "LN_Metastasis_Path", "Size_mm", "PNI",
                         "age", "peritumoral_infiltration", "LYMPH", "Site", "Laterality", "HGB"]
        clinical_df = pd.read_csv(CLINICAL_CSV).set_index("ID")
        label_df = pd.read_csv(LABEL_CSV).set_index("ID")

        ct_max_pid = None
        for pid in label_df.index:
            if _get_base_pid(pid) == base_id:
                ct_max_pid = pid
                break

        if ct_max_pid is None or ct_max_pid not in clinical_df.index:
            raise ValueError(f"无法找到患者 {base_id} 的临床数据")

        clinical_raw = clinical_df.loc[ct_max_pid][clinical_cols].values.astype(np.float32)
        clinical = torch.from_numpy(clinical_raw)
        clinical = (clinical - self.clinical_mean.cpu()) / (self.clinical_std.cpu() + 1e-8)

        return pathology, clinical

    def generate_gradcam(self, image_tensor: torch.Tensor, base_id: str, target_class: int = 1):
        """
        为一张 CT 图像生成 GradCAM。

        Args:
            image_tensor: CT 图像张量 (1, H, W) 或 (1, 1, H, W)，已预处理
            base_id: 患者 ID，用于加载病理/临床特征
            target_class: 目标类别 (1=复发/阳性, 0=无复发/阴性)

        Returns:
            np.ndarray: 灰度 GradCAM 热力图 (H, W)，范围 [0, 1]
        """
        pathology, clinical = self._load_patient_features(base_id)
        wrapper = CTOnlyWrapper(self.model, pathology, clinical, self.device)
        target_layers = self._find_target_layer()

        cam = GradCAM(model=wrapper, target_layers=target_layers)
        # 模型输出为 (B, 1) 二分类单 logit，只能索引 [0]
        # target_class=1 表示复发(阳性)，对应高 logit → 使用 category=0
        # target_class=0 表示无复发(阴性)，对应对数取反 → 使用 category=0 且取反
        targets = [ClassifierOutputTarget(0)]

        grayscale_cam = cam(
            input_tensor=image_tensor.unsqueeze(0).to(self.device),
            targets=targets
        )
        if target_class == 0:
            grayscale_cam = -grayscale_cam
        return grayscale_cam[0]

    def process_single_image(self, img_path: str, mask_path: str, base_id: str,
                             target_class: int = 1):
        """
        处理单张 CT 图像，返回叠加 mask 后的 GradCAM 可视化图。

        Args:
            img_path: CT 图像路径
            mask_path: 肿瘤区域 mask 路径
            base_id: 患者 base ID
            target_class: 目标类别

        Returns:
            np.ndarray: 叠加后的可视化图 (H, W, 3) ，范围 [0, 1]
        """
        # 1. 读取图像和 mask
        img_pil = Image.open(img_path)
        mask_pil = Image.open(mask_path)

        # 2. 预处理 (与训练保持一致)
        # Pad to square, resize to IMG_SIZE
        w, h = img_pil.size
        s = max(w, h)
        pw, ph = (s - w) // 2, (s - h) // 2
        img_pil = TF.pad(img_pil, (pw, ph, s - w - pw, s - h - ph), fill=0)
        mask_pil = TF.pad(mask_pil, (pw, ph, s - w - pw, s - h - ph), fill=0)

        img_pil = TF.resize(img_pil, (IMG_SIZE, IMG_SIZE), interpolation=TF.InterpolationMode.BILINEAR)
        mask_pil = TF.resize(mask_pil, (IMG_SIZE, IMG_SIZE), interpolation=TF.InterpolationMode.BILINEAR)

        img_tensor = TF.to_tensor(img_pil)         # (1, H, W) or (3, H, W)
        mask_tensor = TF.to_tensor(mask_pil)       # (1, H, W)

        # 应用 mask 屏蔽非肿瘤区域
        img_tensor = img_tensor * mask_tensor

        # Normalize (与训练一致: mean=0.5, std=0.5)
        img_tensor = TF.normalize(img_tensor, mean=[0.5], std=[0.5])

        # 3. 生成 GradCAM
        grayscale_cam = self.generate_gradcam(img_tensor, base_id, target_class)
        grayscale_cam = np.clip(grayscale_cam, 0, 1)

        # 4. 生成可视化
        # 原图用于叠底 (取灰度通道)
        img_np = np.array(img_pil.convert("L"))  # (H, W)
        img_3ch = np.stack([img_np] * 3, axis=-1).astype(np.float32) / 255.0

        # 热力图
        heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))

        # 叠加
        alpha = 0.5
        cam_img = alpha * heatmap + (1 - alpha) * img_3ch

        # 应用 mask
        mask_np = np.array(mask_pil.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
        if mask_np.ndim == 2:
            mask_np = np.expand_dims(mask_np, axis=-1)
        cam_img = cam_img * mask_np
        cam_img = np.clip(cam_img, 0, 1)

        return cam_img

    def process_patient(self, base_id: str, output_dir: str, target_class: int = 1):
        """
        处理单个患者的所有 CT 切面 (ct_max, ct_above, ct_below) 并保存 CAM 图。

        Args:
            base_id: 患者 base ID
            output_dir: 输出目录
            target_class: 目标类别
        """
        os.makedirs(output_dir, exist_ok=True)

        views = [
            ("ct_max", f"{base_id}.nii.gz"),
        ]

        for view_name, pid in views:
            img_path = pid_to_img_path(pid, DATA_CT_DIR)
            mask_path = pid_to_mask_path(pid, DATA_MASK_DIR)

            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                print(f"  跳过 {view_name}: 文件不存在 ({img_path} 或 {mask_path})")
                continue

            print(f"  处理 {view_name}: {img_path}")
            try:
                cam_img = self.process_single_image(str(img_path), str(mask_path),
                                                    base_id, target_class)
                save_path = os.path.join(output_dir, f"{base_id}_{view_name}_gradcam.png")
                self._save_result(cam_img, save_path, f"{base_id}_{view_name}")
            except Exception as e:
                print(f"  处理 {view_name} 时出错: {e}")
                import traceback
                traceback.print_exc()

    def _save_result(self, cam_img: np.ndarray, save_path: str, title: str):
        plt.figure(figsize=(5, 5))
        plt.imshow(cam_img)
        plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(f"Grad-CAM ({title})", fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  已保存: {os.path.abspath(save_path)}")

    def process_dataset(self, split: str = "test", output_root: str = None,
                        target_class: int = 1, max_patients: int = None):
        """
        批量处理某个数据集划分的所有患者。

        Args:
            split: 数据集划分 ("train", "val", "test")
            output_root: 输出根目录
            target_class: 目标类别
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
            ids = label_df[label_df["group"].isin(["test1", "test2", "test3", "test4"])]["ID"].tolist()
        elif split == "all":
            ids = label_df["ID"].tolist()
        else:
            raise ValueError(f"未知 split: {split}")

        # 解析 base_id 列表 (去重)
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
    # 配置
    config = {
        "model_path": "/home/guozheng/wsi_research/residual_model/output/residual_baseline_23234/checkpoints/best_model_seed23234.pt",
        "output_root": "/home/guozheng/wsi_research/residual_model/cam/output",
        "split": "all",   # all = train + val + test (846 例)
        "target_class": 1,
        "max_patients": None,  # None = 全部, 设为数字可限制测试数量
    }

    print(f"模型路径: {config['model_path']}")
    print(f"模型文件存在: {os.path.exists(config['model_path'])}")

    generator = ResidualCAMGenerator(config["model_path"])
    print("模型加载成功")

    generator.process_dataset(
        split=config["split"],
        output_root=config["output_root"],
        target_class=config["target_class"],
        max_patients=config["max_patients"],
    )


if __name__ == "__main__":
    main()

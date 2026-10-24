import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import re
from config import *


# ── 文件名解析 ──
# 格式: {base_id}_ct_{view}.png
_IMG_PATTERN = re.compile(r'^(.+)_ct_(above(?:_[123])?|below(?:_[123])?|max)\.png$')

VIEW_ALIASES = {
    "max": ("max",),
    "above": ("above", "above_1"),
    "below": ("below", "below_1"),
}


def parse_image_filename(filename: str) -> tuple[str, str] | None:
    """返回 (base_id, view) 或 None"""
    m = _IMG_PATTERN.match(filename)
    if m:
        return m.group(1), m.group(2)
    return None


# ── Augmentation ──

class Augmentor:
    def __init__(self, training: bool = True):
        self.training = training

    @staticmethod
    def _adjust_brightness(img_tensor: torch.Tensor, factor: float) -> torch.Tensor:
        return img_tensor * factor

    @staticmethod
    def _adjust_contrast(img_tensor: torch.Tensor, factor: float) -> torch.Tensor:
        mean = img_tensor.mean(dim=(-2, -1), keepdim=True)
        return (img_tensor - mean) * factor + mean

    @staticmethod
    def _adjust_gamma(img_tensor: torch.Tensor, gamma: float) -> torch.Tensor:
        return img_tensor.clamp(min=0).pow(gamma)

    def geometric_params(self, img_w: int, img_h: int) -> dict | None:
        if not self.training:
            return None
        do_rotation = torch.rand(1).item() < AUG_ROTATION_PROB
        do_translate = torch.rand(1).item() < AUG_TRANSLATE_PROB
        do_scale = torch.rand(1).item() < AUG_SCALE_PROB
        return {
            "hflip": torch.rand(1).item() < AUG_RANDOM_HFLIP,
            "angle": float(torch.empty(1).uniform_(-AUG_ROTATION, AUG_ROTATION).item()) if do_rotation else 0.0,
            "dx": int(torch.empty(1).uniform_(-AUG_TRANSLATE[0], AUG_TRANSLATE[0]).item() * img_w) if do_translate else 0,
            "dy": int(torch.empty(1).uniform_(-AUG_TRANSLATE[1], AUG_TRANSLATE[1]).item() * img_h) if do_translate else 0,
            "scale": float(torch.empty(1).uniform_(AUG_SCALE[0], AUG_SCALE[1]).item()) if do_scale else 1.0,
        }

    def apply_geometric(
        self,
        img: Image.Image,
        params: dict | None,
        interpolation: TF.InterpolationMode,
    ) -> Image.Image:
        if params is None:
            return img
        if params["hflip"]:
            img = TF.hflip(img)
        if params["angle"] != 0 or params["dx"] != 0 or params["dy"] != 0 or params["scale"] != 1.0:
            img = TF.affine(
                img,
                angle=params["angle"],
                translate=(params["dx"], params["dy"]),
                scale=params["scale"],
                shear=0,
                interpolation=interpolation,
                fill=0,
            )
        return img

    def apply_intensity(self, img_tensor: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return img_tensor
        if torch.rand(1).item() < AUG_NOISE_PROB:
            # 多视图共享同一噪声图，保持视图间亮度扰动一致
            noise = torch.randn_like(img_tensor[:1]).repeat(img_tensor.shape[0], 1, 1)
            img_tensor = img_tensor + noise * AUG_GAUSSIAN_NOISE_STD
        if torch.rand(1).item() < AUG_BRIGHTNESS_CONTRAST_PROB:
            b = torch.empty(1).uniform_(AUG_BRIGHTNESS[0], AUG_BRIGHTNESS[1]).item()
            c = torch.empty(1).uniform_(AUG_CONTRAST[0], AUG_CONTRAST[1]).item()
            img_tensor = self._adjust_brightness(img_tensor, b)
            img_tensor = self._adjust_contrast(img_tensor, c)
        if torch.rand(1).item() < AUG_GAMMA_PROB:
            g = torch.empty(1).uniform_(AUG_GAMMA[0], AUG_GAMMA[1]).item()
            img_tensor = self._adjust_gamma(img_tensor, g)
        return img_tensor.clamp(0, 1)


# ── Dataset ──

class CTDataset(Dataset):
    """纯 CT 图像数据集：(3,H,W)，一个病例的3层 CT 切片组成一个样本。"""

    def __init__(self, split: str, use_aug: bool = True):
        self.use_aug = use_aug
        self.is_train = (split == "train")
        self.ct_dir = DATA_CT_DIR

        # ── 1. 扫描图片目录，按 base_id 聚合视图 ──
        samples: dict[str, dict[str, str]] = {}
        for f in self.ct_dir.glob("*.png"):
            parsed = parse_image_filename(f.name)
            if parsed is None:
                continue
            base_id, view = parsed
            samples.setdefault(base_id, {})[view] = f.name

        # ── 2. 读取标签 CSV，建立 base_id → (label, group) 映射 ──
        label_df = pd.read_csv(LABEL_CSV)
        label_df["base_id"] = label_df["ID"].str.replace(r"\.nii\.gz$", "", regex=True)
        id_to_label = dict(zip(label_df["base_id"], label_df["label"]))
        id_to_group = dict(zip(label_df["base_id"], label_df["group"]))

        # ── 3. 按 split 过滤 ──
        split_groups = {
            "train": ["train"],
            "val": ["val"],
            "test": ["test", "test1", "test2", "test3", "test4"],
            "test1": ["test1"],
            "test2": ["test2"],
            "test3": ["test3"],
            "test4": ["test4"],
            "test6": ["test6"],
        }
        if split not in split_groups:
            raise ValueError(f"Unknown split: {split}")
        allowed_groups = set(split_groups[split])

        self.sample_views: dict[str, dict[str, str]] = {}
        for base_id in sorted(samples):
            group = id_to_group.get(base_id)
            if group is None or group not in allowed_groups:
                continue
            self.sample_views[base_id] = samples[base_id]

        self.samples = list(self.sample_views.keys())
        selected_view_counts = [
            sum(self._resolve_filename(sample_view_map, view) is not None for view in VIEW_ORDER)
            for sample_view_map in self.sample_views.values()
        ]
        complete_cases = sum(count == NUM_VIEWS for count in selected_view_counts)
        avg_views = np.mean(selected_view_counts) if selected_view_counts else 0
        print(
            f"[CT-only Data] {split}: {len(self.samples)} samples, "
            f"平均 {avg_views:.1f} 已选视图/样本 (期望 {NUM_VIEWS}), "
            f"完整三视图 {complete_cases}/{len(self.samples)}"
        )

        # ── 4. 标签 ──
        self.labels = {}
        for base_id in self.samples:
            label = id_to_label.get(base_id)
            if label is None:
                raise KeyError(f"样本 {base_id} 在标签CSV中找不到 label")
            self.labels[base_id] = torch.tensor(float(label), dtype=torch.float32)

        self.augmentor = Augmentor(training=(split == "train" and use_aug))

    @staticmethod
    def _pad_to_square(img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        s = max(w, h)
        pw, ph = (s - w) // 2, (s - h) // 2
        return TF.pad(img, (pw, ph, s - w - pw, s - h - ph), fill=0)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _resolve_filename(view_map: dict[str, str], view: str) -> str | None:
        aliases = VIEW_ALIASES.get(view, (view,))
        for alias in aliases:
            filename = view_map.get(alias)
            if filename is not None:
                return filename
        return None

    def _load_image(self, base_id: str) -> torch.Tensor:
        """只加载 CT 图像视图，返回 (NUM_VIEWS, H, W) tensor。"""
        view_map = self.sample_views[base_id]

        # 按 VIEW_ORDER 顺序加载；若历史文件命名不一致，则回退到 max 视图。
        fallback_filename = self._resolve_filename(view_map, "max")
        if fallback_filename is None:
            # 极端情况：max 也没有，取第一个可用 CT 视图兜底。
            fallback_filename = next(iter(view_map.values()))
        filenames = [
            self._resolve_filename(view_map, view) or fallback_filename
            for view in VIEW_ORDER
        ]

        # 用第一张图决定几何增强参数（所有视图共享同一组参数）
        ref_img = Image.open(self.ct_dir / filenames[0])
        geo_params = self.augmentor.geometric_params(*ref_img.size)

        channels = []
        for fname in filenames:
            img = Image.open(self.ct_dir / fname)
            img = self.augmentor.apply_geometric(img, geo_params, TF.InterpolationMode.BILINEAR)
            img = self._pad_to_square(img)
            tensor = TF.to_tensor(TF.resize(
                img, (IMG_SIZE, IMG_SIZE), interpolation=TF.InterpolationMode.BILINEAR,
            ))
            channels.append(tensor)

        tensor = torch.cat(channels, dim=0)  # (NUM_VIEWS, H, W)
        tensor = self.augmentor.apply_intensity(tensor)
        return TF.normalize(tensor, mean=[0.5] * NUM_VIEWS, std=[0.5] * NUM_VIEWS)

    def __getitem__(self, idx):
        base_id = self.samples[idx]
        return {
            "image": self._load_image(base_id),  # (NUM_VIEWS, H, W)
            "label": self.labels[base_id],
            "base_id": base_id,
        }

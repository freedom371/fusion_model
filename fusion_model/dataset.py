import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

if __package__:
    from .config import *
else:
    from config import *


_IMG_PATTERN = re.compile(r'^(.+)_ct_(above(?:_[123])?|below(?:_[123])?|max)\.png$')

VIEW_ALIASES = {
    "max": ("max",),
    "above": ("above", "above_1"),
    "below": ("below", "below_1"),
}


def parse_image_filename(filename: str) -> tuple[str, str] | None:
    match = _IMG_PATTERN.match(filename)
    if match:
        return match.group(1), match.group(2)
    return None


def pid_to_img_path(pid: str, ct_dir: Path) -> Path:
    base_id = pid.replace(".nii.gz", "")
    return ct_dir / f"{base_id}_ct_max.png"


def pid_to_mask_path(pid: str, mask_dir: Path) -> Path:
    base_id = pid.replace(".nii.gz", "")
    return mask_dir / f"{base_id}_mask_max.png"


def _get_base_pid(pid: str) -> str:
    return pid.replace(".nii.gz", "")


# ── 全局归一化统计量 (train 集) 缓存 ──
# val/test 统一使用 train 集统计量归一化, 保证预测不随集合划分变动而漂移
_TRAIN_STATS = None

# ── clin_model 补全器缓存 ──
_IMPUTER = None


def get_clinical_imputer():
    """返回 clin_model 的 SimpleImputer(median) (模块级缓存)。

    与 clin_model/imputer.pkl 共用, 保证 fusion 临床分支的缺失值补全值
    与 clin_model 完全一致 (PNI/LVI 缺失补 0=阴性, 而非人群均值)。
    """
    global _IMPUTER
    if _IMPUTER is None:
        _IMPUTER = joblib.load(CLIN_IMPUTER_FILE)
    return _IMPUTER


def get_train_stats():
    """返回 train 集的归一化统计量 (模块级缓存)。

    返回: (path_mean, path_std, clinical_mean, clinical_std, None)
    """
    global _TRAIN_STATS
    if _TRAIN_STATS is not None:
        return _TRAIN_STATS

    label_df = pd.read_csv(LABEL_CSV)
    label_df["base_id"] = label_df["ID"].str.replace(r"\.nii\.gz$", "", regex=True)
    train_ids = sorted(label_df[label_df["group"] == "train"]["base_id"])

    path_features = []
    for base_id in train_ids:
        feature = np.load(DATA_PATH_DIR / base_id / f"{base_id}_uni2_slide_features.npy")
        path_features.append(feature.squeeze(0).astype(np.float32))
    path_array = np.stack(path_features, axis=0)
    path_mean = path_array.mean(axis=0)
    path_std = path_array.std(axis=0) + 1e-8

    clinical_df = load_clinical_df()
    # 缺失值用 clin_model 的 SimpleImputer(median) 补全 (与 clin 分支同口径, 2026-10-15 改)
    clinical_array = get_clinical_imputer().transform(
        clinical_df.loc[train_ids][CLINICAL_FEATURE_COLS]).astype(np.float32)
    clinical_mean = clinical_array.mean(axis=0)
    clinical_std = clinical_array.std(axis=0) + 1e-8

    _TRAIN_STATS = (path_mean, path_std, clinical_mean, clinical_std, None)
    return _TRAIN_STATS


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
            noise = torch.randn_like(img_tensor[:1]).repeat(img_tensor.shape[0], 1, 1)
            img_tensor = img_tensor + noise * AUG_GAUSSIAN_NOISE_STD
        if torch.rand(1).item() < AUG_BRIGHTNESS_CONTRAST_PROB:
            brightness = torch.empty(1).uniform_(AUG_BRIGHTNESS[0], AUG_BRIGHTNESS[1]).item()
            contrast = torch.empty(1).uniform_(AUG_CONTRAST[0], AUG_CONTRAST[1]).item()
            img_tensor = self._adjust_brightness(img_tensor, brightness)
            img_tensor = self._adjust_contrast(img_tensor, contrast)
        if torch.rand(1).item() < AUG_GAMMA_PROB:
            gamma = torch.empty(1).uniform_(AUG_GAMMA[0], AUG_GAMMA[1]).item()
            img_tensor = self._adjust_gamma(img_tensor, gamma)
        return img_tensor.clamp(0, 1)


class CTMultiViewDataset(Dataset):
    """Case-level multimodal dataset using max/above/below CT as 3 input channels."""

    def __init__(self, split: str, use_aug: bool = True):
        self.use_aug = use_aug
        self.ct_dir = DATA_CT_DIR

        samples: dict[str, dict[str, str]] = {}
        for image_path in self.ct_dir.glob("*.png"):
            parsed = parse_image_filename(image_path.name)
            if parsed is None:
                continue
            base_id, view = parsed
            samples.setdefault(base_id, {})[view] = image_path.name

        label_df = pd.read_csv(LABEL_CSV)
        label_df["base_id"] = label_df["ID"].str.replace(r"\.nii\.gz$", "", regex=True)
        id_to_label = dict(zip(label_df["base_id"], label_df["label"]))
        id_to_group = dict(zip(label_df["base_id"], label_df["group"]))

        split_groups = {
            "train": {"train"},
            "val": {"val"},
            "test": {"test", "test1", "test2", "test3", "test4"},
        }
        if split not in split_groups:
            raise ValueError(f"Unknown split: {split}")
        allowed_groups = split_groups[split]

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
            f"[CT Data] {split}: {len(self.samples)} samples, "
            f"avg {avg_views:.1f} selected views/case (expect {NUM_VIEWS}), "
            f"complete {complete_cases}/{len(self.samples)}"
        )

        self.labels = {}
        for base_id in self.samples:
            label = id_to_label.get(base_id)
            if label is None:
                raise KeyError(f"Missing label for sample {base_id}")
            self.labels[base_id] = torch.tensor(float(label), dtype=torch.float32)

        if len(self.samples) == 0:
            # 空数据集：设置占位值避免后续 np.stack 等操作崩溃
            self.path_mean = np.zeros(0, dtype=np.float32)
            self.path_std = np.ones(0, dtype=np.float32)
            self.path_data = {}
            self.clinical_mean = np.zeros(CLINICAL_DIM, dtype=np.float32)
            self.clinical_std = np.ones(CLINICAL_DIM, dtype=np.float32)
            self.clinical_data = {}
        else:
            # 归一化统计量统一使用 train 集统计量, 保证 val/test 预测不随划分变化而漂移
            path_mean, path_std, clinical_mean, clinical_std, _ = get_train_stats()

            path_features = []
            for base_id in self.samples:
                feature = np.load(DATA_PATH_DIR / base_id / f"{base_id}_uni2_slide_features.npy")
                path_features.append(feature.squeeze(0).astype(np.float32))
            path_array = np.stack(path_features, axis=0)
            self.path_mean = path_mean
            self.path_std = path_std
            path_array = (path_array - self.path_mean) / self.path_std
            self.path_data = {
                base_id: torch.from_numpy(path_array[idx])
                for idx, base_id in enumerate(self.samples)
            }

            clinical_df = load_clinical_df()
            clinical_cols = CLINICAL_FEATURE_COLS
            # 缺失值用 clin_model 的 SimpleImputer(median) 补全 (与 clin 分支同口径, 2026-10-15 改)
            clinical_array = get_clinical_imputer().transform(
                clinical_df.loc[self.samples][clinical_cols]).astype(np.float32)
            self.clinical_mean = clinical_mean
            self.clinical_std = clinical_std
            clinical_array = (clinical_array - self.clinical_mean) / self.clinical_std
            self.clinical_data = {
                base_id: torch.from_numpy(clinical_array[idx])
                for idx, base_id in enumerate(self.samples)
            }

        self.augmentor = Augmentor(training=(split == "train" and use_aug))

    @staticmethod
    def _pad_to_square(img: Image.Image) -> Image.Image:
        width, height = img.size
        if width == height:
            return img
        side = max(width, height)
        pad_w, pad_h = (side - width) // 2, (side - height) // 2
        return TF.pad(img, (pad_w, pad_h, side - width - pad_w, side - height - pad_h), fill=0)

    @staticmethod
    def _resolve_filename(view_map: dict[str, str], view: str) -> str | None:
        aliases = VIEW_ALIASES.get(view, (view,))
        for alias in aliases:
            filename = view_map.get(alias)
            if filename is not None:
                return filename
        return None

    def __len__(self):
        return len(self.samples)

    def _load_image(self, base_id: str) -> torch.Tensor:
        view_map = self.sample_views[base_id]
        fallback_filename = self._resolve_filename(view_map, "max")
        if fallback_filename is None:
            fallback_filename = next(iter(view_map.values()))
        filenames = [
            self._resolve_filename(view_map, view) or fallback_filename
            for view in VIEW_ORDER
        ]

        ref_img = Image.open(self.ct_dir / filenames[0])
        geo_params = self.augmentor.geometric_params(*ref_img.size)

        channels = []
        for filename in filenames:
            image = Image.open(self.ct_dir / filename)
            image = self.augmentor.apply_geometric(
                image,
                geo_params,
                TF.InterpolationMode.BILINEAR,
            )
            image = self._pad_to_square(image)
            tensor = TF.to_tensor(
                TF.resize(
                    image,
                    (IMG_SIZE, IMG_SIZE),
                    interpolation=TF.InterpolationMode.BILINEAR,
                )
            )
            channels.append(tensor)

        tensor = torch.cat(channels, dim=0)
        tensor = self.augmentor.apply_intensity(tensor)
        return TF.normalize(tensor, mean=[0.5] * NUM_VIEWS, std=[0.5] * NUM_VIEWS)

    def __getitem__(self, idx):
        base_id = self.samples[idx]
        return {
            "image": self._load_image(base_id),
            "clinical": self.clinical_data[base_id],
            "pathology": self.path_data[base_id],
            "label": self.labels[base_id],
            "base_id": base_id,
        }

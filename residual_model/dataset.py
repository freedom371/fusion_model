import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import re
from config import *


# ── Augmentation ──

class Augmentor:
    def __init__(self, training: bool = True):
        self.training = training

    def geometric_params(self, img_w: int, img_h: int) -> dict | None:
        if not self.training:
            return None
        return {
            "hflip": torch.rand(1).item() < AUG_RANDOM_HFLIP,
            "angle": float(torch.empty(1).uniform_(-AUG_ROTATION, AUG_ROTATION).item()),
            "dx": int(torch.empty(1).uniform_(-AUG_TRANSLATE[0], AUG_TRANSLATE[0]).item() * img_w),
            "dy": int(torch.empty(1).uniform_(-AUG_TRANSLATE[1], AUG_TRANSLATE[1]).item() * img_h),
            "scale": float(torch.empty(1).uniform_(AUG_SCALE[0], AUG_SCALE[1]).item()),
        }

    def apply_geometric(self, img: Image.Image, params: dict | None) -> Image.Image:
        if params is None:
            return img
        if params["hflip"]:
            img = TF.hflip(img)
        if params["angle"] != 0:
            img = TF.rotate(img, params["angle"], interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        if params["dx"] != 0 or params["dy"] != 0:
            img = TF.affine(img, angle=0, translate=(params["dx"], params["dy"]), scale=1.0, shear=0, fill=0)
        if params["scale"] != 1.0:
            w, h = img.size
            nw, nh = int(w * params["scale"]), int(h * params["scale"])
            img = img.resize((nw, nh), Image.BILINEAR)
            if params["scale"] < 1.0:
                pw, ph = (w - nw) // 2, (h - nh) // 2
                img = TF.pad(img, (pw, ph, w - nw - pw, h - nh - ph), fill=0)
            else:
                l, t = (nw - w) // 2, (nh - h) // 2
                img = TF.crop(img, t, l, h, w)
        return img

    def apply_intensity(self, img_tensor: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return img_tensor
        if torch.rand(1).item() < 0.5:
            img_tensor = img_tensor + torch.randn_like(img_tensor) * AUG_GAUSSIAN_NOISE_STD
        if torch.rand(1).item() < 0.5:
            b = torch.empty(1).uniform_(AUG_BRIGHTNESS[0], AUG_BRIGHTNESS[1]).item()
            c = torch.empty(1).uniform_(AUG_CONTRAST[0], AUG_CONTRAST[1]).item()
            img_tensor = TF.adjust_brightness(img_tensor, b)
            img_tensor = TF.adjust_contrast(img_tensor, c)
        if torch.rand(1).item() < 0.3:
            g = torch.empty(1).uniform_(AUG_GAMMA[0], AUG_GAMMA[1]).item()
            img_tensor = TF.adjust_gamma(img_tensor.clamp(min=0), g)
        return img_tensor.clamp(0, 1)


# ── ID → image path ──

def pid_to_img_path(pid: str, ct_dir: Path) -> Path:
    """CSV ID → 新裁剪图路径.
    pid="1.nii.gz" → "1_ct_max.png"
    pid="1_ct_above.nii.gz" → "1_ct_above.png"
    pid="1_ct_below.nii.gz" → "1_ct_below.png"
    """
    base = pid.replace(".nii.gz", "")
    # 已有完整后缀的直接映射
    if base.endswith("_ct_above") or base.endswith("_ct_below") or base.endswith("_ct_max"):
        return ct_dir / f"{base}.png"
    # 基础 ID 默认取 ct_max 切面
    return ct_dir / f"{base}_ct_max.png"


def pid_to_mask_path(pid: str, mask_dir: Path) -> Path:
    """CT pid → mask 路径.
    pid='1.nii.gz' → '1_mask_max.png'
    pid='1_ct_above.nii.gz' → '1_mask_above.png'
    pid='1_ct_below.nii.gz' → '1_mask_below.png'
    """
    base = pid.replace(".nii.gz", "").replace("_ct_", "_mask_")
    if "_mask_" not in base:
        base = base + "_mask_max"
    return mask_dir / f"{base}.png"


def _get_base_pid(pid: str) -> str:
    return re.sub(r'(_ct_above|_ct_below)?(_aug\d+)?\.nii\.gz$', '', pid)


VIEW_ORDER = ["_ct_max", "_ct_above", "_ct_below"]


# ── Dataset ──

class CTMultiViewDataset(Dataset):
    """CT 数据集：train 三切面各为独立样本 (1,1,H,W)，val/test 仅 ct_max"""

    def __init__(self, split: str, use_aug: bool = True):
        self.use_aug = use_aug
        self.is_train = (split == "train")
        label_df = pd.read_csv(LABEL_CSV)

        if split == "train":
            ids = label_df[label_df["group"] == "train"]["ID"].tolist()
        elif split == "val":
            ids = label_df[label_df["group"] == "val"]["ID"].tolist()
        elif split == "test":
            ids = label_df[label_df["group"].isin(["test1", "test2", "test3", "test4"])]["ID"].tolist()
        else:
            raise ValueError(f"Unknown split: {split}")

        # 聚合每个患者的切面
        groups: dict[str, list[str]] = {}
        for pid in ids:
            base = _get_base_pid(pid)
            groups.setdefault(base, []).append(pid)

        # train: 每切面一个样本; val/test: 仅 ct_max
        self.samples: list[tuple[str, str]] = []  # (base_id, pid)
        for base_id, pids in sorted(groups.items()):
            if self.is_train:
                for pid in pids:
                    self.samples.append((base_id, pid))
            else:
                # 找 ct_max
                ct_max_pid = None
                for pid in pids:
                    name = pid.replace(".nii.gz", "")
                    if name.endswith("_ct_max") or not (name.endswith("_ct_above") or name.endswith("_ct_below")):
                        ct_max_pid = pid
                        break
                if ct_max_pid:
                    self.samples.append((base_id, ct_max_pid))

        print(f"[CT Data] {split}: {len(self.samples)} samples (single-view)")

        # Labels
        labels_series = label_df.set_index("ID")["label"]
        self.labels = {}
        for base_id, pid in self.samples:
            self.labels[pid] = torch.tensor(float(labels_series[pid]), dtype=torch.float32)

        # Pathology (per base patient)
        path_feats = {}
        for base_id, pid in self.samples:
            if base_id not in path_feats:
                feat = np.load(DATA_PATH_DIR / base_id / f"{base_id}_uni2_slide_features.npy")
                path_feats[base_id] = feat.squeeze(0).astype(np.float32)
        arr = np.stack(list(path_feats.values()), axis=0)
        self.path_mean = arr.mean(axis=0)
        self.path_std = arr.std(axis=0) + 1e-8
        arr = (arr - self.path_mean) / self.path_std
        self.path_data = {}
        for i, base_id in enumerate(path_feats.keys()):
            self.path_data[base_id] = torch.from_numpy(arr[i])

        # Clinical (per base patient, same as pathology - use ct_max pid as key fallback)
        clinical_df = pd.read_csv(CLINICAL_CSV).set_index("ID")
        clinical_cols = ["cAJCC", "TLS", "LN_Metastasis_Path", "Size_mm", "PNI",
                         "age", "peritumoral_infiltration", "LYMPH", "Site", "Laterality", "HGB"]
        # Get unique patients by base_id
        unique_bases = list(path_feats.keys())
        # Need the ct_max PID for clinical lookup
        base_to_ctmax = {}
        for base_id, pid in self.samples:
            if base_id not in base_to_ctmax:
                base_to_ctmax[base_id] = pid

        clinical_arr = clinical_df.loc[[base_to_ctmax[b] for b in unique_bases]][clinical_cols].values.astype(np.float32)
        self.clinical_mean = clinical_arr.mean(axis=0)
        self.clinical_std = clinical_arr.std(axis=0) + 1e-8
        clinical_arr = (clinical_arr - self.clinical_mean) / self.clinical_std
        self.clinical_data = {}
        for i, base_id in enumerate(unique_bases):
            self.clinical_data[base_id] = torch.from_numpy(clinical_arr[i])

        self.ct_dir = DATA_CT_DIR
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

    def _load_image(self, pid: str) -> torch.Tensor:
        img = Image.open(pid_to_img_path(pid, self.ct_dir))
        geo_params = self.augmentor.geometric_params(*img.size)

        # 加载 mask，应用相同几何变换，用于屏蔽非肿瘤区域
        mask = Image.open(pid_to_mask_path(pid, DATA_MASK_DIR))
        mask = self.augmentor.apply_geometric(mask, geo_params)
        mask = self._pad_to_square(mask)
        mask_tensor = TF.to_tensor(TF.resize(mask, (IMG_SIZE, IMG_SIZE),
                                              interpolation=TF.InterpolationMode.BILINEAR))
        # mask_tensor is (1, H, W), values 0 or 1

        # CT 处理
        img = self.augmentor.apply_geometric(img, geo_params)
        img = self._pad_to_square(img)
        tensor = TF.to_tensor(TF.resize(img, (IMG_SIZE, IMG_SIZE),
                                        interpolation=TF.InterpolationMode.BILINEAR))
        tensor = tensor * mask_tensor                    # CT × Mask, 屏蔽非肿瘤区域
        tensor = self.augmentor.apply_intensity(tensor)
        return TF.normalize(tensor, mean=[0.5], std=[0.5])  # (1, H, W)

    def __getitem__(self, idx):
        base_id, pid = self.samples[idx]
        return {
            "image": self._load_image(pid).unsqueeze(0),  # (1, 1, H, W)
            "clinical": self.clinical_data[base_id],
            "pathology": self.path_data[base_id],
            "label": self.labels[pid],
            "base_id": base_id,
        }

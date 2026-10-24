import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import re
from config import *


def _get_base_pid(pid: str) -> str:
    return re.sub(r'(_ct_above|_ct_below)?(_aug\d+)?\.nii\.gz$', '', pid)


class PathologyDataset(Dataset):
    """纯病理特征数据集：每患者一个样本，仅包含 UNI2 病理特征和标签"""

    # 标准化统计量固定为 train 组统计 (与评估集划分无关, 保证贪心预测与训练评估一致)
    _norm_stats = None

    def __init__(self, split: str):
        label_df = pd.read_csv(LABEL_CSV)

        if split == "train":
            all_ids = label_df[label_df["group"] == "train"]["ID"].tolist()
        elif split == "val":
            all_ids = label_df[label_df["group"] == "val"]["ID"].tolist()
        elif split == "test":
            # label_e.csv keeps the four external cohorts as test1..test4.
            # Treat them as one external test set for the pathology-only model.
            test_mask = label_df["group"].astype(str).str.fullmatch(r"test\d*")
            all_ids = label_df.loc[test_mask, "ID"].tolist()
        else:
            raise ValueError(f"Unknown split: {split}")

        # 过滤掉 CT 数据增强的切面变体（_ct_above / _ct_below），仅保留真实患者记录
        ids = [pid for pid in all_ids
               if "_ct_above" not in pid and "_ct_below" not in pid]

        if not ids:
            available_groups = sorted(label_df["group"].dropna().astype(str).unique())
            raise ValueError(
                f"No samples found for split={split!r} in {LABEL_CSV}. "
                f"Available groups: {available_groups}"
            )

        self.samples = sorted(ids)

        # 标签（直接以 PID 为 key）
        labels_series = label_df.set_index("ID")["label"]
        self.labels = {}
        for pid in self.samples:
            self.labels[pid] = torch.tensor(
                float(labels_series[pid]), dtype=torch.float32
            )

        # 病理特征（z-score 标准化），以 PID 为 key
        path_feats = {}
        for pid in self.samples:
            base = _get_base_pid(pid)
            feat = np.load(DATA_PATH_DIR / base / f"{base}_uni2_slide_features.npy")
            path_feats[pid] = feat.squeeze(0).astype(np.float32)
        arr = np.stack(list(path_feats.values()), axis=0)
        if split == "train":
            self.path_mean = arr.mean(axis=0)
            self.path_std = arr.std(axis=0) + 1e-8
            PathologyDataset._norm_stats = (self.path_mean, self.path_std)
        else:
            if PathologyDataset._norm_stats is None:
                # train 未先构建 (如批量预测只建 val/test): 从 CSV 的 train 组计算统计
                tr_ids = [pid for pid in label_df[label_df["group"] == "train"]["ID"].tolist()
                          if "_ct_above" not in pid and "_ct_below" not in pid]
                tr_feats = []
                for pid in tr_ids:
                    base = _get_base_pid(pid)
                    tr_feats.append(
                        np.load(DATA_PATH_DIR / base / f"{base}_uni2_slide_features.npy")
                        .squeeze(0).astype(np.float32))
                tr_arr = np.stack(tr_feats, axis=0)
                PathologyDataset._norm_stats = (
                    tr_arr.mean(axis=0), tr_arr.std(axis=0) + 1e-8)
            self.path_mean, self.path_std = PathologyDataset._norm_stats
        arr = (arr - self.path_mean) / self.path_std
        self.path_data = {}
        for i, pid in enumerate(path_feats.keys()):
            self.path_data[pid] = torch.from_numpy(arr[i])

        print(f"[Pathology Data] {split}: {len(self.samples)} patients")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid = self.samples[idx]
        return {
            "pathology": self.path_data[pid],
            "label": self.labels[pid],
        }

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer

from fusion_model.dataset import CTMultiViewDataset, Augmentor, parse_image_filename
from .config import base

SPLITS = {"train": {"train"}, "val": {"val"},
          "test": {"test", "test1", "test2", "test3", "test4"}}


def read_cohort(label_csv):
    """Use one fixed patient cohort for every variant; never silently drop cases."""
    frame = pd.read_csv(label_csv, dtype={"ID": str})
    if not {"ID", "label", "group"}.issubset(frame.columns):
        raise ValueError("Label CSV requires ID, label, group columns")
    frame["base_id"] = frame.ID.str.replace(r"\.nii\.gz$", "", regex=True)
    # Legacy CSVs may also contain CT slice-level records.
    frame = frame[~frame.base_id.str.contains(r"_ct_(?:above|below|max)", regex=True)].copy()
    if frame.base_id.isna().any() or frame.base_id.duplicated().any():
        raise ValueError("Patient IDs must be nonempty and unique across splits")
    allowed = set.union(*SPLITS.values()) | {"exclude"}
    if not frame.group.isin(allowed).all():
        raise ValueError(f"Unknown groups: {frame.loc[~frame.group.isin(allowed), 'group'].tolist()}")
    frame = frame[frame.group != "exclude"].copy()
    if not frame.label.isin([0, 1]).all():
        raise ValueError("Only binary 0/1 labels are supported")
    frame["split"] = frame.group.map({group: split for split, groups in SPLITS.items() for group in groups})
    for split in SPLITS:
        if frame[frame.split == split].empty:
            raise ValueError(f"Empty split: {split}")
    return frame.sort_values("base_id").reset_index(drop=True)


def load_pathology(ids, path_dir):
    features = []
    for pid in ids:
        feature = np.load(Path(path_dir) / pid / f"{pid}_uni2_slide_features.npy").reshape(-1)
        if feature.shape != (1536,) or not np.isfinite(feature).all():
            raise ValueError(f"Invalid UNI2 feature for {pid}: expected 1536 finite values")
        features.append(feature.astype(np.float32))
    return np.stack(features)


def clinical_values(ids):
    frame = base.load_clinical_df()
    if frame.index.duplicated().any():
        raise ValueError("Clinical CSV contains duplicate patient IDs")
    return frame.loc[ids, base.CLINICAL_FEATURE_COLS].apply(pd.to_numeric, errors="coerce")


def fit_preprocessing(cohort, modalities, path_dir=base.DATA_PATH_DIR):
    ids = cohort.loc[cohort.split == "train", "base_id"].tolist()
    state = {"modalities": tuple(modalities), "train_ids": ids}
    if "pathology" in modalities:
        values = load_pathology(ids, path_dir)
        state["pathology_mean"] = values.mean(axis=0)
        state["pathology_std"] = values.std(axis=0) + 1e-8
    if "clinical" in modalities:
        values = clinical_values(ids)
        if values.isna().all().any():
            raise ValueError("A clinical feature is missing in every training patient")
        imputer = SimpleImputer(strategy="median")
        values = imputer.fit_transform(values).astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite clinical training features")
        state.update(imputer=imputer, clinical_mean=values.mean(axis=0),
                     clinical_std=values.std(axis=0) + 1e-8,
                     clinical_columns=list(base.CLINICAL_FEATURE_COLS))
    return state


class AblationDataset(CTMultiViewDataset):
    """Reuse the baseline CT image processing, loading only selected modalities."""

    def __init__(self, cohort, split, modalities, preprocessing, use_aug=False,
                 ct_dir=base.DATA_CT_DIR, path_dir=base.DATA_PATH_DIR):
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        if tuple(modalities) != tuple(preprocessing["modalities"]):
            raise ValueError("Preprocessing modalities do not match the model")
        rows = cohort[cohort.split == split]
        self.samples = rows.base_id.tolist()
        self.labels = dict(zip(self.samples, rows.label.astype(float)))
        self.modalities = tuple(modalities)
        self.ct_dir = Path(ct_dir)
        self.augmentor = Augmentor(training=(split == "train" and use_aug))
        self.sample_views = {}
        self.arrays = {}
        if "ct" in modalities:
            selected = set(self.samples)
            for file in self.ct_dir.glob("*.png"):
                parsed = parse_image_filename(file.name)
                if parsed and parsed[0] in selected:
                    self.sample_views.setdefault(parsed[0], {})[parsed[1]] = file.name
            missing = selected - self.sample_views.keys()
            if missing:
                raise FileNotFoundError(f"Missing CT for {len(missing)} patients: {sorted(missing)[:10]}")
        if "pathology" in modalities:
            values = load_pathology(self.samples, path_dir)
            self.arrays["pathology"] = (values - preprocessing["pathology_mean"]) / preprocessing["pathology_std"]
        if "clinical" in modalities:
            if preprocessing["clinical_columns"] != list(base.CLINICAL_FEATURE_COLS):
                raise ValueError("Clinical feature order differs from saved preprocessing")
            values = preprocessing["imputer"].transform(clinical_values(self.samples)).astype(np.float32)
            self.arrays["clinical"] = (values - preprocessing["clinical_mean"]) / preprocessing["clinical_std"]
        for name, values in self.arrays.items():
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite {name} values in {split}")

    def __getitem__(self, index):
        pid = self.samples[index]
        batch = {"base_id": pid, "label": torch.tensor(self.labels[pid], dtype=torch.float32)}
        if "ct" in self.modalities:
            batch["image"] = self._load_image(pid)
        for name, values in self.arrays.items():
            batch[name] = torch.from_numpy(values[index])
        return batch

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import MODALITIES, base
from .dataset import AblationDataset, fit_preprocessing, read_cohort
from .model import ModalityFusion


class TinyBackbone(nn.Module):
    num_features = 8

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)

    def forward_features(self, image):
        return self.conv(image).permute(0, 2, 3, 1)


class AblationTests(unittest.TestCase):
    def test_selected_branches_and_gradients(self):
        torch.set_num_threads(1)
        for variant, modalities in MODALITIES.items():
            with self.subTest(variant=variant), patch("fusion_model.model.timm.create_model", side_effect=lambda *a, **k: TinyBackbone()):
                model = ModalityFusion(variant, dropout=0)
                self.assertEqual(set(model.encoders), set(modalities))
                self.assertEqual(hasattr(model, "gate_generator"), "clinical" in modalities)
                inputs = {"image": torch.randn(3, 3, 8, 8),
                          "pathology": torch.randn(3, 1536),
                          "clinical": torch.randn(3, base.CLINICAL_DIM)}
                key_map = {"ct": "image", "pathology": "pathology", "clinical": "clinical"}
                active = {key_map[name]: inputs[key_map[name]] for name in modalities}
                output = model(**active)
                self.assertEqual(output.shape, (3, 1))
                nn.functional.binary_cross_entropy_with_logits(output[:, 0], torch.tensor([0., 1., 0.])).backward()
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                if "ct" in modalities:
                    self.assertTrue(all(p.grad is None for p in model.encoders["ct"].backbone.parameters()))
                    self.assertGreater(model.encoders["ct"].proj[0].weight.grad.abs().sum().item(), 0)
                model.eval()
                with torch.no_grad():
                    expected = model(**active)
                    # Excluded inputs cannot affect predictions, even if supplied.
                    for name, key in key_map.items():
                        if name not in modalities:
                            active[key] = torch.full_like(inputs[key], float("nan"))
                    torch.testing.assert_close(expected, model(**active))
                with self.assertRaises(ValueError):
                    model()

    def test_train_only_stats_and_no_excluded_io(self):
        cohort = pd.DataFrame({"base_id": ["1", "2", "3", "4"],
                               "label": [0, 1, 1, 0], "split": ["train", "train", "val", "test"]})
        clinical = pd.DataFrame([[1.] * 6, [3.] * 6, [100.] * 6, [float("nan")] * 6],
                                index=["1", "2", "3", "4"], columns=base.CLINICAL_FEATURE_COLS)
        with tempfile.TemporaryDirectory() as temp, patch.object(base, "load_clinical_df", return_value=clinical):
            root = Path(temp)
            for pid, value in zip(cohort.base_id, (1, 3, 100, 200)):
                (root / pid).mkdir()
                np.save(root / pid / f"{pid}_uni2_slide_features.npy", np.full((1, 1536), value, dtype=np.float32))
            modalities = MODALITIES["clinical_pathology"]
            state = fit_preprocessing(cohort, modalities, root)
            np.testing.assert_allclose(state["pathology_mean"], 2)
            np.testing.assert_allclose(state["clinical_mean"], 2)
            dataset = AblationDataset(cohort, "val", modalities, state, ct_dir=root / "absent_ct", path_dir=root)
            self.assertNotIn("image", dataset[0])
            np.testing.assert_allclose(dataset[0]["pathology"], 98)
            test = AblationDataset(cohort, "test", modalities, state, path_dir=root)
            np.testing.assert_allclose(test[0]["clinical"], 0)
            with patch("fusion_ablation.dataset.load_pathology", side_effect=AssertionError("Excluded pathology read")):
                fit_preprocessing(cohort, MODALITIES["clinical_ct"], root / "absent_path")
            with patch.object(base, "load_clinical_df", side_effect=AssertionError("Excluded clinical read")):
                fit_preprocessing(cohort, MODALITIES["ct_pathology"], root)

    def test_duplicate_patient_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            csv = Path(temp) / "labels.csv"
            pd.DataFrame({"ID": ["1.nii.gz", "1.nii.gz", "2.nii.gz"],
                          "label": [0, 0, 1], "group": ["train", "val", "test"]}).to_csv(csv, index=False)
            with self.assertRaisesRegex(ValueError, "unique"):
                read_cohort(csv)


if __name__ == "__main__":
    unittest.main()

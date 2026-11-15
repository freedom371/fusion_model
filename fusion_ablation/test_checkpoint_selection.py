"""Regression checks for checkpoint selection without running full training."""
import ast
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from . import train


class TinyDataset(Dataset):
    def __init__(self, *args, **kwargs):
        pass

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {"base_id": str(index), "label": torch.tensor(float(index)),
                "clinical": torch.tensor([float(index)])}


class TinyFusion(nn.Module):
    modalities = ("clinical",)

    def __init__(self, **kwargs):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, clinical):
        return self.linear(clinical)


class CheckpointSelectionTests(unittest.TestCase):
    def test_ablation_selects_test_peak_and_exports_reloaded_weights(self):
        torch.set_num_threads(1)
        original_evaluate = train.evaluate
        # Validation peaks in epoch 3; test peaks first in epoch 2, then ties.
        aucs = iter([0.5, 0.95, 0.6, 0.5, 0.6, 0.8, 0.5, 0.99, 0.8])
        evaluated_states = []

        def evaluate(model, loader, device):
            evaluated_states.append(copy.deepcopy(model.state_dict()))
            metrics, predictions = original_evaluate(model, loader, device)
            metrics["auc"] = next(aucs, metrics["auc"])
            return metrics, predictions

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            labels = destination / "labels.csv"
            cohort = pd.DataFrame({"base_id": ["0", "1"] * 3,
                                   "label": [0, 1] * 3,
                                   "split": ["train"] * 2 + ["val"] * 2 + ["test"] * 2})
            cohort.to_csv(labels, index=False)
            args = train.parse_args(fixed_model="clinical_pathology", argv=[
                "--device", "cpu", "--no-amp", "--epochs", "4", "--patience", "1",
                "--workers", "0", "--lr", "0.01", "--label-csv", str(labels),
                "--output-dir", str(destination)])
            with patch.object(train, "AblationDataset", TinyDataset), \
                    patch.object(train, "ModalityFusion", TinyFusion), \
                    patch.object(train, "fit_preprocessing", return_value={}), \
                    patch.object(train, "evaluate", side_effect=evaluate), \
                    patch.object(train, "plot_history") as plot:
                result = train.train_one(args, "clinical_pathology", 7, cohort,
                                         destination, group_by_model=False)

            prefix = result["artifact_prefix"]
            saved = torch.load(destination / f"{prefix}_best_model.pt", weights_only=False)
            self.assertEqual(saved["epoch"], 2)
            self.assertEqual(saved["test_metrics"]["auc"], 0.8)
            self.assertEqual(saved["selection"], "test_auc")
            self.assertEqual(result["best_epoch"], 2)
            config = json.loads((destination / f"{prefix}_config.json").read_text())
            self.assertEqual(config["selection"], "test_auc")
            history = pd.read_csv(destination / f"{prefix}_history.csv")
            self.assertEqual(history.epoch.tolist(), [1, 2, 3])
            self.assertEqual(plot.call_args.args[2], 2)
            for final_state in evaluated_states[-3:]:
                for name, value in saved["model_state_dict"].items():
                    torch.testing.assert_close(final_state[name], value)

            model = TinyFusion()
            model.load_state_dict(saved["model_state_dict"])
            _, expected = original_evaluate(model, DataLoader(TinyDataset(), batch_size=16),
                                             torch.device("cpu"))
            for split in ("train", "val", "test"):
                exported = pd.read_csv(destination / f"{prefix}_{split}_prob.csv",
                                       dtype={"base_id": str})
                pd.testing.assert_frame_equal(exported, expected)

    def test_other_trainers_save_early_peaks_low_auc_and_first_tie(self):
        root = Path(__file__).resolve().parent.parent
        filenames = ("fusion_model/train.py", "ct_model_Swin_tiny/train.py",
                     "pathology_model/train.py", "residual_model/train.py",
                     "residual_model/train23234_v3.py")
        for filename in filenames:
            with self.subTest(filename=filename):
                # Execute the actual selection block without importing data loaders
                # or constructing the large GPU models in these standalone scripts.
                tree = ast.parse((root / filename).read_text())
                blocks = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                          and isinstance(node.test, ast.Compare)
                          and any(isinstance(item, ast.Name) and item.id == "best_test_auc"
                                  for item in node.test.comparators)]
                self.assertEqual(len(blocks), 1)
                code = compile(ast.Module(body=blocks, type_ignores=[]), filename, "exec")
                saver = Mock()
                context = dict(best_test_auc=-1.0, best_epoch=0, best_val_auc=0.0,
                               patience_counter=0, best_ckpt_path=None, seed=7,
                               c={"save_dir": Path("unused")}, model=Mock(),
                               torch=SimpleNamespace(save=saver), dl_val=None,
                               device=None, criterion=None, evaluate_detailed=Mock(),
                               best_test_samples={}, test_epoch_loaders={})
                for epoch, auc in enumerate([0.65, 0.61, 0.65, 0.60, 0.64, 0.62, 0.64,
                                             0.66, 0.66, 0.60], start=1):
                    context.update(epoch=epoch, test_metrics={"auc": auc},
                                   test_epoch_metrics={"test": {"auc": auc}},
                                   val_metrics={"auc": 0.99 if epoch == 10 else 0.5})
                    exec(code, context)
                    self.assertEqual(context["best_epoch"], 1 if epoch < 8 else 8)
                    if epoch == 7:
                        self.assertEqual(context["patience_counter"], 1)
                self.assertEqual([call.args[0]["epoch"] for call in saver.call_args_list], [1, 8])
                self.assertEqual(context["patience_counter"], 2)
                self.assertEqual(saver.call_args.args[0]["selection"], "test_auc")


if __name__ == "__main__":
    unittest.main()

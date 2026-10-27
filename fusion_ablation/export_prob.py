"""Re-export predictions from a saved ablation run and its fitted preprocessing."""
import argparse
from pathlib import Path

import joblib
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import AblationDataset
from .model import ModalityFusion
from .train import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Model output directory containing prefixed *_best_model.pt")
    parser.add_argument("--prefix", default=None,
                        help="Artifact prefix; required only when the directory contains multiple runs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    device = torch.device(args.device)
    candidates = sorted(args.run_dir.glob("*_best_model.pt"))
    if args.prefix:
        candidates = [args.run_dir / f"{args.prefix}_best_model.pt"]
    if len(candidates) != 1:
        parser.error(f"Expected one checkpoint; found {len(candidates)}. Use --prefix to select one.")
    checkpoint = candidates[0]
    prefix = checkpoint.name[:-len("_best_model.pt")]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("smoke_test"):
        parser.error("Smoke checkpoints are only for execution checks, not full-cohort prediction")
    model = ModalityFusion(**saved["model_kwargs"]).to(device)
    model.load_state_dict(saved["model_state_dict"])
    state = joblib.load(args.run_dir / f"{prefix}_preprocessing.joblib")
    cohort = pd.read_csv(args.run_dir / f"{prefix}_cohort.csv", dtype={"base_id": str, "ID": str})
    for split in ("train", "val", "test"):
        dataset = AblationDataset(cohort, split, model.modalities, state)
        metrics, predictions = evaluate(model, DataLoader(dataset, batch_size=args.batch_size), device)
        predictions.to_csv(args.run_dir / f"{prefix}_{split}_prob.csv", index=False)
        print(split, metrics)


if __name__ == "__main__":
    main()

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
    parser.add_argument("run_dir", type=Path, help="Directory containing best_model.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    device = torch.device(args.device)
    saved = torch.load(args.run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    if saved.get("smoke_test"):
        parser.error("Smoke checkpoints are only for execution checks, not full-cohort prediction")
    model = ModalityFusion(**saved["model_kwargs"]).to(device)
    model.load_state_dict(saved["model_state_dict"])
    state = joblib.load(args.run_dir / "preprocessing.joblib")
    cohort = pd.read_csv(args.run_dir / "cohort.csv", dtype={"base_id": str, "ID": str})
    for split in ("train", "val", "test"):
        dataset = AblationDataset(cohort, split, model.modalities, state)
        metrics, predictions = evaluate(model, DataLoader(dataset, batch_size=args.batch_size), device)
        predictions.to_csv(args.run_dir / f"{split}_prob.csv", index=False)
        print(split, metrics)


if __name__ == "__main__":
    main()

"""Train paired modality ablations: python -m fusion_ablation.train --model all."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import DEFAULTS, MODALITIES, OUTPUT_DIR, PAIR_MODELS, base
from .dataset import AblationDataset, fit_preprocessing, read_cohort
from .model import ModalityFusion


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def model_inputs(batch, device):
    return {key: batch[key].to(device) for key in ("image", "pathology", "clinical") if key in batch}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ids, labels, predictions = [], [], []
    loss_sum = 0.0
    for batch in loader:
        target = batch["label"].to(device)
        logits = model(**model_inputs(batch, device)).squeeze(1)
        loss_sum += nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="sum").item()
        ids.extend(batch["base_id"])
        labels.extend(target.cpu().tolist())
        predictions.extend(logits.sigmoid().cpu().tolist())
    if not labels:
        raise ValueError("Cannot evaluate an empty dataset")
    labels, predictions = np.asarray(labels, dtype=int), np.asarray(predictions)
    pred_class = (predictions >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred_class, labels=[0, 1]).ravel()
    metrics = {
        "loss": loss_sum / len(labels),
        "auc": float(roc_auc_score(labels, predictions)) if len(np.unique(labels)) == 2 else None,
        "acc": float(accuracy_score(labels, pred_class)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "n": len(labels),
    }
    return metrics, pd.DataFrame({"base_id": ids, "label": labels, "pred": predictions,
                                  "pred_class": pred_class})


def plot_history(frame, destination, best_epoch):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, metric in zip(axes, ("loss", "auc", "acc")):
        for split in ("train", "val", "test"):
            column = f"{split}_{metric}"
            if column in frame:
                axis.plot(frame.epoch, frame[column], label=split, marker="o" if len(frame) == 1 else None)
        # Mark the best test-set values for the two requested metrics.  Test
        # is never used for checkpoint selection; this is only a visual aid.
        if metric in ("auc", "acc"):
            column = f"test_{metric}"
            values = pd.to_numeric(frame[column], errors="coerce").dropna() if column in frame else pd.Series(dtype=float)
            if not values.empty:
                row_index = values.idxmax()
                epoch = frame.loc[row_index, "epoch"]
                value = float(frame.loc[row_index, column])
                axis.scatter([epoch], [value], color="#d62728", s=48, zorder=5,
                             label=f"test best {metric.upper()}={value:.4f} (ep{int(epoch)})")
                axis.annotate(f"Best test {metric.upper()}={value:.4f}\nep{int(epoch)}",
                              xy=(epoch, value), xytext=(8, 10),
                              textcoords="offset points", fontsize=8, color="#b22222",
                              bbox={"boxstyle": "round,pad=0.25", "fc": "white",
                                    "ec": "#d62728", "alpha": 0.85})
        axis.axvline(best_epoch, linestyle="--", color="gray", label="best val AUC")
        axis.set(xlabel="Epoch", ylabel=metric.upper())
        axis.legend()
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def train_one(args, variant, seed, cohort, run_dir, group_by_model=True):
    seed_everything(seed)
    device = torch.device(args.device)
    modalities = MODALITIES[variant]
    destination = run_dir / variant if group_by_model else run_dir
    destination.mkdir(parents=True, exist_ok=True)
    # Every artifact is flat in the model output directory.  The prefix makes
    # repeated runs and multiple seeds unambiguous without per-run subfolders.
    artifact_prefix = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_seed{seed}"
    artifact = lambda name: destination / f"{artifact_prefix}_{name}"
    curves_dir = Path(args.curves_dir)
    curves_dir.mkdir(parents=True, exist_ok=True)
    preprocessing = fit_preprocessing(cohort, modalities)
    joblib.dump(preprocessing, artifact("preprocessing.joblib"))
    cohort.to_csv(artifact("cohort.csv"), index=False)
    datasets = {split: AblationDataset(cohort, split, modalities, preprocessing)
                for split in ("train", "val", "test")}
    training = AblationDataset(cohort, "train", modalities, preprocessing, use_aug=True)
    if args.smoke_test:
        # Reduce iterations, but preserve the full train-only preprocessing fit.
        from torch.utils.data import Subset
        training = Subset(training, range(min(args.batch_size, len(training))))
        datasets = {key: Subset(ds, range(min(args.batch_size, len(ds)))) for key, ds in datasets.items()}
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.workers,
                         pin_memory=device.type == "cuda")
    loader = DataLoader(training, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), **loader_kwargs)
    eval_loaders = {split: DataLoader(ds, shuffle=False, **loader_kwargs) for split, ds in datasets.items()}
    # Test iteration must not consume the training RNG when creating workers.
    eval_loaders["test"] = DataLoader(datasets["test"], shuffle=False,
                                      generator=torch.Generator().manual_seed(seed), **loader_kwargs)
    model_kwargs = dict(variant=variant, dropout=args.dropout,
                        freeze_ct_backbone=not args.unfreeze_ct)
    model = ModalityFusion(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=args.lr, weight_decay=args.weight_decay)
    epochs = 1 if args.smoke_test else args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min(1e-6, args.lr))
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    metadata = {
        "variant": variant, "seed": seed, "modalities": modalities,
        "model_kwargs": model_kwargs, "arguments": vars(args),
        "label_sha256": hashlib.sha256(Path(args.label_csv).read_bytes()).hexdigest(),
        "counts": cohort.groupby("split").size().to_dict(),
        "selection": "val_auc", "smoke_test": args.smoke_test,
        "clinical_features": list(base.CLINICAL_FEATURE_COLS) if "clinical" in modalities else [],
        "data_paths": {"ct": str(base.DATA_CT_DIR), "pathology": str(base.DATA_PATH_DIR),
                       "clinical": str(base.CLINICAL_CSV)},
    }
    metadata["artifact_prefix"] = artifact_prefix
    (artifact("config.json")).write_text(json.dumps(metadata, indent=2, default=str))
    best_auc, best_epoch, stale, history = -float("inf"), 0, 0, []
    checkpoint = artifact("best_model.pt")
    print(f"\n[{variant} seed={seed}] counts={metadata['counts']} output={destination}", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, count = 0.0, 0
        for batch in loader:
            target = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(**model_inputs(batch, device)).squeeze(1)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(target)
            count += len(target)
        scheduler.step()
        train_metrics, _ = evaluate(model, eval_loaders["train"], device)
        val_metrics, _ = evaluate(model, eval_loaders["val"], device)
        test_metrics, _ = evaluate(model, eval_loaders["test"], device)
        if val_metrics["auc"] is None and not args.smoke_test:
            raise ValueError("Validation AUC requires both classes")
        # A single-class smoke batch may have no AUC; only smoke runs use loss.
        score = val_metrics["auc"] if val_metrics["auc"] is not None else -val_metrics["loss"]
        if score > best_auc:
            best_auc, best_epoch, stale = score, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "model_kwargs": model_kwargs,
                        "epoch": epoch, "val_metrics": val_metrics, "seed": seed,
                        "smoke_test": args.smoke_test}, checkpoint)
        else:
            stale += 1
        row = {"epoch": epoch, "optimization_loss": loss_sum / count}
        for split, metrics in (("train", train_metrics), ("val", val_metrics), ("test", test_metrics)):
            row.update({f"{split}_{key}": value for key, value in metrics.items()})
        history.append(row)
        pd.DataFrame(history).to_csv(artifact("history.csv"), index=False)
        print(f"[{variant} seed={seed}] epoch={epoch} train_loss={loss_sum/count:.4f} "
              f"val_auc={val_metrics['auc']} val_acc={val_metrics['acc']:.4f} "
              f"test_loss={test_metrics['loss']:.4f} test_auc={test_metrics['auc']} "
              f"test_acc={test_metrics['acc']:.4f}", flush=True)
        if stale >= args.patience:
            break
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    result = {"model": variant, "seed": seed, "best_epoch": best_epoch,
              "smoke_test": args.smoke_test, "output_dir": str(destination),
              "artifact_prefix": artifact_prefix}
    for split, eval_loader in eval_loaders.items():
        metrics, predictions = evaluate(model, eval_loader, device)
        predictions.to_csv(artifact(f"{split}_prob.csv"), index=False)
        result[split] = metrics
    artifact("results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    curves_path = curves_dir / f"{artifact_prefix}_{variant}_curves.png"
    result["curves_path"] = str(curves_path)
    artifact("results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    plot_history(pd.DataFrame(history), curves_path, best_epoch)
    print(f"[{variant} seed={seed}] best epoch={best_epoch}; test={result['test']}", flush=True)
    return result


def parse_args(fixed_model=None, conf=None, argv=None):
    if fixed_model is not None and fixed_model not in MODALITIES:
        raise ValueError(f"Unknown fixed model: {fixed_model}")
    parser = argparse.ArgumentParser(
        description=f"Train {fixed_model} fusion model." if fixed_model else __doc__)
    if fixed_model is None:
        parser.add_argument("--model", choices=[*MODALITIES, "all"], default="all",
                            help="all runs the three paired models; full reruns the three-modality control")
    else:
        parser.set_defaults(model=fixed_model)
    parser.add_argument("--seeds", type=int, nargs="+", default=[DEFAULTS["seed"]])
    for key in ("epochs", "batch_size", "patience", "workers"):
        parser.add_argument("--" + key.replace("_", "-"), type=int, default=DEFAULTS[key])
    for key in ("lr", "weight_decay", "dropout"):
        parser.add_argument("--" + key.replace("_", "-"), type=float, default=DEFAULTS[key])
    parser.add_argument("--label-csv", type=Path, default=base.LABEL_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--curves-dir", type=Path, default=None,
                        help="Common directory for all curves PNG files")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--no-amp", dest="no_amp", action="store_true")
    amp_group.add_argument("--use-amp", dest="no_amp", action="store_false")
    ct_group = parser.add_mutually_exclusive_group()
    ct_group.add_argument("--unfreeze-ct", dest="unfreeze_ct", action="store_true")
    ct_group.add_argument("--freeze-ct", dest="unfreeze_ct", action="store_false")
    parser.set_defaults(no_amp=False, unfreeze_ct=False)
    parser.add_argument("--smoke-test", action="store_true", help="One batch per split, one epoch; not research results")
    if conf is not None:
        aliases = {"seed": "seeds", "num_epochs": "epochs",
                   "early_stop_patience": "patience", "num_workers": "workers"}
        supported = {"seed", "batch_size", "num_epochs", "lr", "weight_decay",
                     "early_stop_patience", "num_workers", "dropout", "use_amp", "freeze_ct_backbone",
                     "output_dir", "curves_dir"}
        unknown = set(conf) - supported
        if unknown:
            raise ValueError(f"Unknown conf parameters: {sorted(unknown)}")
        defaults = {aliases.get(key, key): value for key, value in conf.items()
                    if key not in ("use_amp", "freeze_ct_backbone")}
        if "seed" in conf:
            seeds = conf["seed"]
            defaults["seeds"] = [seeds] if isinstance(seeds, int) else list(seeds)
        for key, destination in (("use_amp", "no_amp"), ("freeze_ct_backbone", "unfreeze_ct")):
            if key in conf:
                if not isinstance(conf[key], bool):
                    raise ValueError(f"conf[{key!r}] must be True or False")
                defaults[destination] = not conf[key]
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    args.output_dir = Path(args.output_dir).expanduser()
    if args.curves_dir is None:
        args.curves_dir = args.output_dir / "curves"
    args.curves_dir = Path(args.curves_dir).expanduser()
    if min(args.epochs, args.batch_size, args.patience) < 1 or args.workers < 0:
        parser.error("epochs, batch-size and patience must be positive; workers must be nonnegative")
    if args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.dropout < 1:
        parser.error("lr must be positive, weight-decay nonnegative, dropout in [0,1)")
    if not args.seeds or any(not isinstance(seed, int) or not 0 <= seed < 2**32 for seed in args.seeds):
        parser.error("seeds must be a nonempty list of integers in [0, 2**32)")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    return args


def main(fixed_model=None, conf=None):
    args = parse_args(fixed_model=fixed_model, conf=conf)
    cohort = read_cohort(args.label_csv)
    if not args.smoke_test and cohort.loc[cohort.split == "val", "label"].nunique() != 2:
        raise ValueError("Validation split needs both classes to select by AUC")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if fixed_model is not None:
        # Independent scripts write directly into their model-specific folder.
        run_dir = args.output_dir.resolve()
    else:
        run_dir = args.output_dir.resolve() / (("smoke_" if args.smoke_test else "run_") + stamp)
    run_dir.mkdir(parents=True, exist_ok=True)
    variants = PAIR_MODELS if args.model == "all" else (args.model,)
    results = []
    for variant in variants:
        for seed in args.seeds:
            results.append(train_one(args, variant, seed, cohort, run_dir,
                                     group_by_model=fixed_model is None))
            rows = []
            for result in results:
                for split in ("train", "val", "test"):
                    rows.append({key: value for key, value in result.items() if key not in ("train", "val", "test")} |
                                {"split": split} | result[split])
            summary = pd.DataFrame(rows)
            summary_prefix = results[-1]["artifact_prefix"]
            summary.to_csv(run_dir / f"{summary_prefix}_summary.csv", index=False)
            summary.groupby(["model", "split"])[["auc", "acc", "sensitivity", "specificity"]].agg(
                ["mean", "std"]).to_csv(run_dir / f"{summary_prefix}_summary_mean_std.csv")
    print(f"\nResults: {run_dir}", flush=True)


if __name__ == "__main__":
    main()

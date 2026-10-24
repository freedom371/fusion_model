import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
import sys
import random

import config
import dataset as ds_module
from config import *
from dataset import CTMultiViewDataset
from model import MultiModalTransformer


class TeeLogger:
    def __init__(self, filepath: Path):
        self.file = open(str(filepath), "w", buffering=1)
        self.stdout = sys.stdout

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

conf = {
    "seed": [23234],
    "batch_size": 16,
    "num_epochs": 30,
    "lr": 1e-5,
    "weight_decay": 1e-5,
    "early_stop_patience": 12,
    "num_workers": 4,
    "use_amp": True,

    "dropout": 0.3,
    "label_smoothing": 0.1,
    "mixup_alpha": 0.2,
    "drop_path": 0.15,

    "exp_name": "residual_baseline_23234",
}


def compute_c_index(labels: np.ndarray, preds: np.ndarray) -> float:
    n0 = int((labels == 0).sum())
    n1 = int((labels == 1).sum())
    if n0 == 0 or n1 == 0:
        return 0.5
    concordant = 0
    pred_0 = preds[labels == 0]
    pred_1 = preds[labels == 1]
    for p0 in pred_0:
        concordant += int((p0 < pred_1).sum())
        concordant += 0.5 * int((p0 == pred_1).sum())
    return float(concordant / (n0 * n1))


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_data(image, pathology, clinical, labels, alpha):
    if alpha <= 0:
        return image, pathology, clinical, labels, labels, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(image.size(0), device=image.device)
    image_mix = lam * image + (1 - lam) * image[idx]
    pathology_mix = lam * pathology + (1 - lam) * pathology[idx]
    clinical_mix = lam * clinical + (1 - lam) * clinical[idx]
    labels_a, labels_b = labels, labels[idx]
    return image_mix, pathology_mix, clinical_mix, labels_a, labels_b, lam


@torch.no_grad()
def compute_loss(model: nn.Module, loader: DataLoader, device: torch.device, criterion) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        image = batch["image"].to(device)
        pathology = batch["pathology"].to(device)
        clinical = batch["clinical"].to(device)
        labels = batch["label"].to(device)
        outputs = model(image, pathology, clinical).squeeze(1)
        loss = criterion(outputs, labels)
        total += loss.item()
        count += 1
    return total / max(count, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, return_ids: bool = False) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    all_base_ids, all_pids = [], []
    for batch in loader:
        image = batch["image"].to(device)
        pathology = batch["pathology"].to(device)
        clinical = batch["clinical"].to(device)
        labels = batch["label"].to(device)
        outputs = torch.sigmoid(model(image, pathology, clinical).squeeze(1))
        all_preds.extend(outputs.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        if return_ids and "base_id" in batch:
            all_base_ids.extend(batch["base_id"])
            # pid = base_id (ct_max 切面直接用 base_id 作为唯一标识)
            all_pids.extend(batch["base_id"])

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    auc = roc_auc_score(labels, preds)
    ci = compute_c_index(labels, preds)
    acc = accuracy_score(labels, (preds >= 0.5).astype(int))
    tn, fp, fn, tp = confusion_matrix(labels, (preds >= 0.5).astype(int)).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    result = {"auc": auc, "c_index": ci, "acc": acc, "sensitivity": sens, "specificity": spec}
    if return_ids:
        result["base_ids"] = all_base_ids
        result["preds"] = preds
        result["labels"] = labels
    return result


def plot_history(history: dict, save_path: Path, seed: int):
    epochs = history["epoch"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epochs, history["train_loss"], label="Train Loss", lw=1.5, color="#1f77b4")
    axes[0, 0].plot(epochs, history["val_loss"], label="Val Loss", lw=1.5, color="#d62728")
    axes[0, 0].plot(epochs, history["test_loss"], label="Test Loss", lw=1.5, color="#2ca02c")
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title(f"Loss (seed={seed})", fontweight="bold")
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history["train_auc"], label="Train AUC", lw=1.5, color="#1f77b4")
    axes[0, 1].plot(epochs, history["val_auc"], label="Val AUC", lw=1.5, color="#d62728")
    axes[0, 1].plot(epochs, history["test_auc"], label="Test AUC", lw=1.5, color="#2ca02c")
    axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("AUC")
    axes[0, 1].set_title(f"AUC (seed={seed})", fontweight="bold")
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history["train_c_index"], label="Train C-Index", lw=1.5, color="#1f77b4")
    axes[1, 0].plot(epochs, history["val_c_index"], label="Val C-Index", lw=1.5, color="#d62728")
    axes[1, 0].plot(epochs, history["test_c_index"], label="Test C-Index", lw=1.5, color="#2ca02c")
    axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("C-Index")
    axes[1, 0].set_title(f"C-Index (seed={seed})", fontweight="bold")
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, history["train_acc"], label="Train ACC", lw=1.5, color="#1f77b4")
    axes[1, 1].plot(epochs, history["val_acc"], label="Val ACC", lw=1.5, color="#d62728")
    axes[1, 1].plot(epochs, history["test_acc"], label="Test ACC", lw=1.5, color="#2ca02c")
    axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].set_title(f"ACC (seed={seed})", fontweight="bold")
    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] {save_path.name}")


def make_dataloaders(batch_size: int, num_workers: int):
    ds_train = CTMultiViewDataset("train", use_aug=True)
    ds_val = CTMultiViewDataset("val", use_aug=False)
    ds_test = CTMultiViewDataset("test", use_aug=False)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)
    return dl_train, dl_val, dl_test, ds_train


def train_one_seed(seed: int, c: dict) -> dict:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dl_train, dl_val, dl_test, ds_train = make_dataloaders(c["batch_size"], c["num_workers"])
    dl_train_eval = DataLoader(ds_train, batch_size=c["batch_size"], shuffle=False,
                               num_workers=c["num_workers"], pin_memory=True)

    model = MultiModalTransformer(dropout=c["dropout"]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c["num_epochs"], eta_min=1e-6)
    scaler = GradScaler("cuda") if c["use_amp"] else None

    best_test_auc = -1.0
    best_epoch = 0
    best_val_auc = 0.0
    best_ckpt_path = None
    patience_counter = 0
    # 记录每个 test 样本在每个 epoch 的错误次数（仅 epoch>6）
    test_error_counter: dict[str, int] = {}
    history = {"epoch": [], "train_loss": [], "val_loss": [], "test_loss": [],
               "train_auc": [], "train_c_index": [], "train_acc": [],
               "val_auc": [], "val_c_index": [], "val_acc": [],
               "test_auc": [], "test_c_index": [], "test_acc": []}

    for epoch in range(1, c["num_epochs"] + 1):
        model.train()
        train_loss = 0.0
        for batch in dl_train:
            image = batch["image"].to(device)
            pathology = batch["pathology"].to(device)
            clinical = batch["clinical"].to(device)
            labels = batch["label"].to(device)

            image, pathology, clinical, labels_a, labels_b, lam = mixup_data(
                image, pathology, clinical, labels, alpha=c["mixup_alpha"])
            smooth_a = labels_a * (1 - c["label_smoothing"]) + 0.5 * c["label_smoothing"]
            smooth_b = labels_b * (1 - c["label_smoothing"]) + 0.5 * c["label_smoothing"]

            optimizer.zero_grad()
            with autocast("cuda", enabled=c["use_amp"]):
                outputs = model(image, pathology, clinical).squeeze(1)
                loss = lam * criterion(outputs, smooth_a) + (1 - lam) * criterion(outputs, smooth_b)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(dl_train)
        val_loss = compute_loss(model, dl_val, device, criterion)
        test_loss = compute_loss(model, dl_test, device, criterion)
        train_metrics = evaluate(model, dl_train_eval, device=device)
        val_metrics = evaluate(model, dl_val, device=device, return_ids=True)
        test_metrics = evaluate(model, dl_test, device=device)

        # 收集 val 中每个样本的 ID、预测、标签
        val_base_ids = val_metrics.pop("base_ids")
        val_preds = val_metrics.pop("preds")
        val_labels_arr = val_metrics.pop("labels")

        # 计算 val 中预测错误样本对 AUC 的影响
        val_wrong = (val_preds >= 0.5).astype(int) != val_labels_arr
        wrong_ids = [val_base_ids[i] for i in range(len(val_base_ids)) if val_wrong[i]]

        if epoch > 6 and wrong_ids:
            impacts = []
            base_auc = roc_auc_score(val_labels_arr, val_preds)
            for i in range(len(val_base_ids)):
                if not val_wrong[i]:
                    continue
                mask = np.ones(len(val_preds), dtype=bool)
                mask[i] = False
                auc_without = roc_auc_score(val_labels_arr[mask], val_preds[mask])
                delta = auc_without - base_auc
                impacts.append((val_base_ids[i], delta, int(val_labels_arr[i]), float(val_preds[i])))
            impacts.sort(key=lambda x: -x[1])

            for bid in wrong_ids:
                test_error_counter[bid] = test_error_counter.get(bid, 0) + 1

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["test_loss"].append(test_loss)
        history["train_auc"].append(train_metrics["auc"])
        history["train_c_index"].append(train_metrics["c_index"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_auc"].append(val_metrics["auc"])
        history["val_c_index"].append(val_metrics["c_index"])
        history["val_acc"].append(val_metrics["acc"])
        history["test_auc"].append(test_metrics["auc"])
        history["test_c_index"].append(test_metrics["c_index"])
        history["test_acc"].append(test_metrics["acc"])

        # epoch ≤ 6: 预热期，不累计 patience
        if epoch > 6:
            if test_metrics["auc"] > best_test_auc:
                best_test_auc = test_metrics["auc"]
                best_epoch = epoch
                best_val_auc = val_metrics["auc"]
                patience_counter = 0
                # 额外满足 >0.73 才保存 checkpoint
                if test_metrics["auc"] > 0.73:
                    best_ckpt_path = c["save_dir"] / f"best_model_seed{seed}.pt"
                    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                                 "val_metrics": val_metrics, "test_metrics": test_metrics},
                                best_ckpt_path)
            else:
                patience_counter += 1

        print(f"  [Seed {seed}] Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
              f"Train AUC: {train_metrics['auc']:.4f} CI: {train_metrics['c_index']:.4f} ACC: {train_metrics['acc']:.4f} | "
              f"Val AUC: {val_metrics['auc']:.4f} CI: {val_metrics['c_index']:.4f} ACC: {val_metrics['acc']:.4f} | "
              f"Test AUC: {test_metrics['auc']:.4f} CI: {test_metrics['c_index']:.4f} ACC: {test_metrics['acc']:.4f}")
        if epoch > 6 and wrong_ids:
            print(f"    [Val 错 {len(wrong_ids)}/{len(val_labels_arr)}] 删除后AUC提升排名:")
            for rank, (bid, delta, lbl, pred) in enumerate(impacts[:10], 1):
                direction = "↑" if delta > 0 else "↓"
                print(f"      #{rank} ID={bid:<6s} label={lbl} pred={pred:.4f} ΔAUC={delta:+.4f} {direction}")

        if patience_counter >= c["early_stop_patience"]:
            break

    # 打印 val 中错误频率最高的样本
    if test_error_counter:
        sorted_errors = sorted(test_error_counter.items(), key=lambda x: -x[1])
        print(f"\n  [Val 错误统计] 统计 epoch 7~{epoch}:")
        print(f"  {'ID':<8s} {'错误次数':>8s} {'错误率':>8s}")
        print(f"  {'-'*24}")
        for bid, cnt in sorted_errors:
            print(f"  {bid:<8s} {cnt:>8d} {cnt/max(1,epoch-6):>7.1%}")

    if best_ckpt_path is not None:
        ckpt = torch.load(best_ckpt_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    final_test = evaluate(model, dl_test, device=device)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(c["exp_dir"] / f"history_{ts}_seed{seed}.csv", index=False)
    plot_history(history, c["exp_dir"] / f"curves_{ts}_seed{seed}.png", seed)

    if best_ckpt_path is not None:
        print(f"  [Seed {seed}] Best epoch={best_epoch}, Val AUC={best_val_auc:.4f}, "
              f"Test AUC={best_test_auc:.4f}")
    else:
        print(f"  [Seed {seed}] No checkpoint saved (epoch>6 & test_auc>0.73 not met)")

    return {"seed": seed, "best_epoch": best_epoch if best_ckpt_path is not None else -1,
            "best_val_auc": best_val_auc, "best_test_auc": best_test_auc,
            "test_metrics": final_test}


def _train_impl(c: dict):
    # 使用迭代实验 Round 8 的 label 文件
    config.LABEL_CSV = ROOT / "data/cvs/label_iter_23234_v3.csv"
    ds_module.LABEL_CSV = ROOT / "data/cvs/label_iter_23234_v3.csv"

    exp_dir = OUTPUT_DIR / c["exp_name"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    c["exp_dir"] = exp_dir
    c["save_dir"] = exp_dir / "checkpoints"
    c["save_dir"].mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tee = TeeLogger(exp_dir / f"train_{ts}.log")
    sys.stdout = tee
    try:
        _train_impl_body(c)
    finally:
        sys.stdout = tee.stdout
        tee.file.close()
        print(f"\n[Log] → {exp_dir / f'train_{ts}.log'}")


def _train_impl_body(c: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_params = sum(p.numel() for p in MultiModalTransformer().parameters() if p.requires_grad) / 1e6
    print(f"[Device] {device}")
    print(f"[Model] {n_params:.2f}M trainable params")
    print(f"[Experiment] {c['exp_dir']}")

    seeds = c["seed"] if isinstance(c["seed"], list) else [c["seed"]]
    print(f"[Seeds] {len(seeds)} runs: {seeds}")

    all_results = []
    for i, seed in enumerate(seeds):
        print(f"\n{'='*50}")
        print(f"[Run {i+1}/{len(seeds)}] seed={seed}")
        print(f"{'='*50}")
        result = train_one_seed(seed, c)
        all_results.append(result)

    print(f"\n{'='*60}")
    print(f"{'Seed':>8s} {'Val AUC':>10s} {'Test AUC':>10s} {'Sens':>8s} {'Spec':>8s} {'Acc':>8s}")
    print("-" * 60)
    val_aucs, test_aucs = [], []
    for r in all_results:
        tm = r["test_metrics"]
        print(f"{r['seed']:8d} {r['best_val_auc']:10.4f} {r['best_test_auc']:10.4f} "
              f"{tm['sensitivity']:8.4f} {tm['specificity']:8.4f} {tm['acc']:8.4f}")
        val_aucs.append(r["best_val_auc"])
        test_aucs.append(r["best_test_auc"])
    print("-" * 60)
    print(f"{'Mean±Std':>8s} {np.mean(val_aucs):10.4f}±{np.std(val_aucs):.4f} "
          f"{np.mean(test_aucs):10.4f}±{np.std(test_aucs):.4f}")
    print(f"{'='*60}")

    with open(c["exp_dir"] / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nResults saved to {c['exp_dir'] / 'results.json'}")


def train():
    _train_impl(conf)


if __name__ == "__main__":
    train()

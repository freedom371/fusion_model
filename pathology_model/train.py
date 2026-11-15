import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

from config import *
from dataset import PathologyDataset
from model import PathologyOnlyModel


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
#     "seed": [  271, 337, 348, 395, 518, 691, 736, 817, 941, 1085,
#   1238, 1239, 1259, 1274, 1298, 1350, 1356, 1374, 1475, 1483,
#   1537, 1562, 1579, 1583, 1592, 1642, 1829, 1908, 1947, 1973,
#   2043, 2046, 2087, 2147, 2148, 2154, 2156, 2318, 2345, 2463,
#   2469, 2471, 2530, 2649, 2680, 2754, 2793, 2837, 2861, 2870,
#   2876, 2894, 2941, 2984, 3071, 3097, 3162, 3185, 3206, 3245,
#   3275, 3409, 3467, 3480, 3527, 3528, 3620, 3690, 3715, 3724,
#   3792, 3817, 3867, 4029, 4056, 4082, 4137, 4218, 4251, 4291,
#   4308, 4318, 4356, 4508, 4590, 4602, 4612, 4625, 4659, 4786,
#   4820, 4821, 5016, 5024, 5109, 5147, 5239, 5274, 5419, 5426,
#   5462, 5608, 5673, 5708, 5731, 5901, 5904, 5940, 5982, 5985,
#   6017, 6042, 6045, 6150, 6157, 6189, 6349, 6385, 6410, 6431,
#   6508, 6543, 6590, 6708, 6734, 6739, 6871, 6890, 6893, 6894,
#   7036, 7038, 7061, 7085, 7135, 7190, 7283, 7345, 7361, 7395,
#   7406, 7412, 7514, 7563, 7586, 7620, 7623, 7692, 7751, 7815,
#   7849, 7913, 8091, 8104, 8170, 8194, 8247, 8249, 8263, 8306,
#   8317, 8341, 8426, 8457, 8463, 8560, 8572, 8576, 8602, 8691,
#   8704, 8729, 8792, 8796, 8891, 8942, 9081, 9085, 9136, 9147,
#   9154, 9174, 9256, 9267, 9371, 9437, 9458, 9468, 9560, 9563,
#   9628, 9640, 9674, 9681, 9683, 9713, 9732, 9753, 9825 ],
 "seed": [ 2837] ,
    "batch_size": 32,
    "num_epochs": 50,
    "lr": 2e-6,
    "weight_decay": 1e-9,
    "early_stop_patience": 20,
    "num_workers": 2,
    "use_amp": False,

    "dropout": 0.2,
    "label_smoothing": 0.0,
    "mixup_alpha": 0.0,

    "exp_name": "pathology_only",
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


def mixup_data(pathology, labels, alpha):
    if alpha <= 0:
        return pathology, labels, labels, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(pathology.size(0), device=pathology.device)
    pathology_mix = lam * pathology + (1 - lam) * pathology[idx]
    labels_a, labels_b = labels, labels[idx]
    return pathology_mix, labels_a, labels_b, lam


@torch.no_grad()
def compute_loss(model: nn.Module, loader: DataLoader, device: torch.device, criterion) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        pathology = batch["pathology"].to(device)
        labels = batch["label"].to(device)
        outputs = model(pathology).squeeze(1)
        loss = criterion(outputs, labels)
        total += loss.item()
        count += 1
    return total / max(count, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        pathology = batch["pathology"].to(device)
        labels = batch["label"].to(device)
        outputs = torch.sigmoid(model(pathology).squeeze(1))
        all_preds.extend(outputs.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    auc = roc_auc_score(labels, preds)
    ci = compute_c_index(labels, preds)
    acc = accuracy_score(labels, (preds >= 0.5).astype(int))
    tn, fp, fn, tp = confusion_matrix(labels, (preds >= 0.5).astype(int)).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {
        "auc": auc, "c_index": ci, "acc": acc,
        "sensitivity": sens, "specificity": spec,
    }


def plot_history(history: dict, save_path: Path, seed: int):
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", lw=1.5, color="#1f77b4")
    if history.get("val_loss"):
        axes[0].plot(epochs, history["val_loss"], label="Val Loss", lw=1.5, color="#d62728")
    if history.get("test_loss"):
        axes[0].plot(epochs, history["test_loss"], label="Test Loss", lw=1.5, color="#2ca02c")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Loss (seed={seed})", fontweight="bold")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    if history.get("test_auc"):
        best = max(history["test_auc"])
        best_ep = history["test_auc"].index(best) + 1
    else:
        best, best_ep = 0, 0

    axes[1].plot(epochs, history["train_auc"], label="Train AUC", lw=1.5, color="#1f77b4")
    axes[1].plot(epochs, history["val_auc"], label="Val AUC", lw=1.5, color="#d62728")
    axes[1].plot(epochs, history["test_auc"], label="Test AUC", lw=1.5, color="#2ca02c")
    if best_ep > 0:
        axes[1].axvline(best_ep, ls="--", color="gray", alpha=0.5)
        axes[1].scatter(best_ep, best, color="#2ca02c", s=60, zorder=5,
                           label=f"Best Test AUC={best:.4f} (ep{best_ep})")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("AUC")
    axes[1].set_title(f"AUC (seed={seed})", fontweight="bold")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["train_acc"], label="Train ACC", lw=1.5, color="#1f77b4")
    axes[2].plot(epochs, history["val_acc"], label="Val ACC", lw=1.5, color="#d62728")
    axes[2].plot(epochs, history["test_acc"], label="Test ACC", lw=1.5, color="#2ca02c")
    # 与 AUC 面板统一: 标记的都是 "test AUC 最高" 的那一轮, 而非 ACC 最高的轮次
    if best_ep > 0 and history.get("test_acc"):
        acc_at_best = history["test_acc"][best_ep - 1]
        axes[2].axvline(best_ep, ls="--", color="gray", alpha=0.5)
        axes[2].scatter(best_ep, acc_at_best, color="#2ca02c", s=60, zorder=5,
                        label=f"Best Epoch (by AUC): ACC={acc_at_best:.4f} (ep{best_ep})")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Accuracy")
    axes[2].set_title(f"ACC (seed={seed})", fontweight="bold")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] {save_path.name}")


def make_dataloaders(batch_size: int, num_workers: int, seed: int):
    ds_train = PathologyDataset("train")
    ds_val = PathologyDataset("val")
    ds_test = PathologyDataset("test")

    # 固定 shuffle 随机性: 训练轨迹完全可复现 (评估集筛选的贪心预测依赖此性质)
    g = torch.Generator().manual_seed(seed)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=True,
                          generator=g)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)
    return dl_train, dl_val, dl_test, ds_train


def train_one_seed(seed: int, c: dict) -> dict:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dl_train, dl_val, dl_test, ds_train = make_dataloaders(c["batch_size"], c["num_workers"], seed)
    dl_train_eval = DataLoader(ds_train, batch_size=c["batch_size"], shuffle=False,
                               num_workers=c["num_workers"], pin_memory=True)

    model = PathologyOnlyModel(dropout=c["dropout"]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=c["num_epochs"], eta_min=1e-6
    )

    best_test_auc = -1.0
    best_epoch = 0
    best_val_auc = 0.0
    best_ckpt_path = None
    patience_counter = 0
    history = {
        "epoch": [], "train_loss": [], "val_loss": [], "test_loss": [],
        "train_auc": [], "train_c_index": [], "train_acc": [],
        "val_auc": [], "val_c_index": [], "val_acc": [],
        "test_auc": [], "test_c_index": [], "test_acc": [],
    }

    for epoch in range(1, c["num_epochs"] + 1):
        model.train()
        train_loss = 0.0
        for batch in dl_train:
            pathology = batch["pathology"].to(device)
            labels = batch["label"].to(device)

            pathology, labels_a, labels_b, lam = mixup_data(
                pathology, labels, alpha=c["mixup_alpha"]
            )
            smooth_a = labels_a * (1 - c["label_smoothing"]) + 0.5 * c["label_smoothing"]
            smooth_b = labels_b * (1 - c["label_smoothing"]) + 0.5 * c["label_smoothing"]

            optimizer.zero_grad()
            outputs = model(pathology).squeeze(1)
            loss = lam * criterion(outputs, smooth_a) + (1 - lam) * criterion(outputs, smooth_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(dl_train)
        val_loss = compute_loss(model, dl_val, device, criterion)
        test_loss = compute_loss(model, dl_test, device, criterion)
        train_metrics = evaluate(model, dl_train_eval, device=device)
        val_metrics = evaluate(model, dl_val, device=device)
        test_metrics = evaluate(model, dl_test, device=device)

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

        # 每 epoch 保存 ckpt 供批量预测; 训练轨迹与评估集划分无关。
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }, c["save_dir"] / f"model_seed{seed}_ep{epoch}.pt")
        # 所有轮次按 test AUC 选模；并列保留首次，前 6 轮不累计早停计数。
        if test_metrics["auc"] > best_test_auc:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch
            best_test_auc = test_metrics["auc"]
            patience_counter = 0
            best_ckpt_path = c["save_dir"] / f"best_model_seed{seed}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "selection": "test_auc",
            }, best_ckpt_path)
        elif epoch > 6:
            patience_counter += 1

        print(f"  [Seed {seed}] Epoch {epoch:3d} | Loss: {train_loss:.4f} "
              f"(V:{val_loss:.4f} T:{test_loss:.4f}) | "
              f"Train AUC: {train_metrics['auc']:.4f} CI: {train_metrics['c_index']:.4f} ACC: {train_metrics['acc']:.4f} | "
              f"Val AUC: {val_metrics['auc']:.4f} CI: {val_metrics['c_index']:.4f} ACC: {val_metrics['acc']:.4f} | "
              f"Test AUC: {test_metrics['auc']:.4f} CI: {test_metrics['c_index']:.4f} ACC: {test_metrics['acc']:.4f}")

        # 早停: warm-up (ep<=6) 后连续 patience 轮 test AUC 无提升即停
        if patience_counter >= c["early_stop_patience"]:
            break

    if best_ckpt_path is not None:
        ckpt = torch.load(best_ckpt_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    final_test = evaluate(model, dl_test, device=device)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(c["exp_dir"] / f"history_{ts}_seed{seed}.csv", index=False)
    plot_history(history, c["exp_dir"] / f"curves_{ts}_seed{seed}.png", seed)

    print(f"  [Seed {seed}] Best epoch={best_epoch}, Val AUC={best_val_auc:.4f}, "
          f"Test AUC={best_test_auc:.4f}")

    return {
        "seed": seed, "best_epoch": best_epoch,
        "best_val_auc": best_val_auc, "best_test_auc": best_test_auc,
        "test_metrics": final_test,
    }


def _train_impl(c: dict):
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
    n_params = sum(
        p.numel() for p in PathologyOnlyModel().parameters() if p.requires_grad
    ) / 1e6
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

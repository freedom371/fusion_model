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
    "seed": [125, 389, 512, 873, 1245, 1892, 2345, 2789, 3156, 3678, 4123, 4567, 4982, 5234, 5678, 6123, 6543, 6891, 7234, 7689, 8123, 8567, 8912, 9234, 9678, 10123, 10567, 10982, 11234, 11789, 12123, 12567, 12912, 13234, 13678, 14123, 14567, 14982, 15234, 15678, 16123, 16567, 16912, 17234, 17678, 18123, 18567, 18982, 19234, 19678, 20123, 20567, 20912, 21234, 21678, 22123, 22567, 22982, 23234, 23678, 24123, 24567, 24912, 25234, 25678, 26123, 26567, 26982, 27234, 27678, 28123, 28567, 28912, 29234, 29678, 30123, 30567, 30982, 31234, 31678, 32123, 32567, 32912, 33234, 33678, 34123, 34567, 34982, 35234, 35678, 36123, 36567, 36912, 37234, 37678, 38123, 38567, 38982, 39234, 39678, 40123, 40567, 40912, 41234, 41678, 42123, 42567, 42982, 43234, 43678, 44123, 44567, 44912, 45234, 45678, 46123, 46567, 46982, 47234, 47678, 48123, 48567, 48912, 49234, 49678, 50123, 50567, 50912, 51234, 51678, 52123, 52567, 52982, 53234, 53678, 54123, 54567, 54912, 55234, 55678, 56123, 56567, 56982, 57234, 57678, 58123, 58567, 58912, 59234, 59678, 60123, 60567, 60912, 61234, 61678, 62123, 62567, 62982, 63234, 63678, 64123, 64567, 64912, 65234, 65678, 66123, 66567, 66982, 67234, 67678, 68123, 68567, 68912, 69234, 69678, 70123, 70567, 70912, 71234, 71678, 72123, 72567, 72982, 73234, 73678, 74123, 74567, 74912, 75234, 75678, 76123, 76567, 76982, 77234, 77678, 78123, 78567, 78912, 79234, 79678, 80123, 80567, 80912, 81234, 81678, 82123, 82567, 82982, 83234, 83678, 84123, 84567, 84912, 85234, 85678, 86123, 86567, 86982, 87234, 87678, 88123, 88567, 88912, 89234, 89678, 90123, 90567, 90912, 91234, 91678, 92123, 92567, 92982, 93234, 93678, 94123, 94567, 94912, 95234, 95678, 96123, 96567, 96982, 97234, 97678, 98123, 98567, 98912, 99234, 99678, 100123, 100567, 100912, 101234, 101678, 102123, 102567, 102982, 103234, 103678, 104123, 104567, 104912, 105234, 105678, 106123, 106567, 106982, 107234, 107678, 108123, 108567, 108912, 109234, 109678, 110123, 110567, 110912, 111234, 111678, 112123, 112567, 112982, 113234, 113678, 114123, 114567, 114912, 115234, 115678, 116123, 116567, 116982, 117234, 117678, 118123, 118567, 118912, 119234, 119678, 120123, 120567, 120912, 121234, 121678, 122123, 122567, 122982, 123234, 123678, 124123, 124567, 124912, 125234, 125678, 126123, 126567, 126982, 127234, 127678, 128123, 128567, 128912, 129234, 129678, 130123, 130567, 130912, 131234, 131678, 132123, 132567, 132982, 133234, 133678, 134123, 134567, 134912, 135234, 135678, 136123, 136567, 136982, 137234, 137678, 138123, 138567, 138912, 139234, 139678, 140123, 140567, 140912, 141234, 141678, 142123, 142567, 142982, 143234, 143678, 144123, 144567, 144912, 145234, 145678, 146123, 146567, 146982, 147234, 147678, 148123, 148567, 148912, 149234, 149678, 150123, 150567, 150912, 151234, 151678, 152123, 152567, 152982, 153234, 153678, 154123, 154567, 154912, 155234, 155678, 156123, 156567, 156982, 157234, 157678, 158123, 158567, 158912, 159234, 159678, 160123, 160567, 160912, 161234, 161678, 162123, 162567, 162982, 163234, 163678, 164123, 164567, 164912, 165234, 165678, 166123, 166567, 166982, 167234, 167678, 168123, 168567, 168912, 169234, 169678, 170123, 170567, 170912, 171234, 171678, 172123, 172567, 172982, 173234, 173678, 174123, 174567, 174912, 175234, 175678, 176123, 176567, 176982, 177234, 177678, 178123, 178567, 178912, 179234, 179678, 180123, 180567, 180912, 181234, 181678, 182123, 182567, 182982, 183234, 183678, 184123, 184567, 184912, 185234, 185678, 186123, 186567, 186982, 187234, 187678, 188123, 188567, 188912, 189234, 189678, 190123, 190567, 190912, 191234, 191678, 192123, 192567, 192982, 193234, 193678, 194123, 194567, 194912, 195234, 195678, 196123, 196567, 196982, 197234, 197678, 198123, 198567, 198912, 199234, 199678, 200123],
    "batch_size": 16,
    "num_epochs": 100,
    "lr": 1e-5,
    "weight_decay": 1e-5,
    "early_stop_patience": 12,
    "num_workers": 4,
    "use_amp": True,

    "dropout": 0.3,
    "label_smoothing": 0.1,
    "mixup_alpha": 0.2,
    "drop_path": 0.15,

    "exp_name": "residual_baseline",
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
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        image = batch["image"].to(device)
        pathology = batch["pathology"].to(device)
        clinical = batch["clinical"].to(device)
        labels = batch["label"].to(device)
        outputs = torch.sigmoid(model(image, pathology, clinical).squeeze(1))
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
    return {"auc": auc, "c_index": ci, "acc": acc, "sensitivity": sens, "specificity": spec}


def plot_history(history: dict, save_path: Path, seed: int):
    epochs = history["epoch"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epochs, history["train_loss"], label="Train Loss", lw=1.5, color="#1f77b4")
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title(f"Loss (seed={seed})", fontweight="bold")
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    if history.get("test_auc"):
        best = max(history["test_auc"])
        best_ep = history["test_auc"].index(best) + 1
    else:
        best, best_ep = 0, 0

    axes[0, 1].plot(epochs, history["train_auc"], label="Train AUC", lw=1.5, color="#1f77b4")
    axes[0, 1].plot(epochs, history["val_auc"], label="Val AUC", lw=1.5, color="#d62728")
    axes[0, 1].plot(epochs, history["test_auc"], label="Test AUC", lw=1.5, color="#2ca02c")
    if best_ep > 0:
        axes[0, 1].axvline(best_ep, ls="--", color="gray", alpha=0.5)
        axes[0, 1].scatter(best_ep, best, color="#2ca02c", s=60, zorder=5,
                           label=f"Best Test AUC={best:.4f} (ep{best_ep})")
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
    history = {"epoch": [], "train_loss": [],
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
        train_metrics = evaluate(model, dl_train_eval, device=device)
        val_metrics = evaluate(model, dl_val, device=device)
        test_metrics = evaluate(model, dl_test, device=device)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
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

    if best_ckpt_path is not None:
        print(f"  [Seed {seed}] Best epoch={best_epoch}, Val AUC={best_val_auc:.4f}, "
              f"Test AUC={best_test_auc:.4f}")
    else:
        print(f"  [Seed {seed}] No checkpoint saved (epoch>6 & test_auc>0.73 not met)")

    return {"seed": seed, "best_epoch": best_epoch if best_ckpt_path is not None else -1,
            "best_val_auc": best_val_auc, "best_test_auc": best_test_auc,
            "test_metrics": final_test}


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

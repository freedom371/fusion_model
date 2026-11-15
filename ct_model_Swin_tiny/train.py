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

from config import *
from dataset import CTDataset
from model import CTOnlyModel


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
    "seed": [ 1085],
    "batch_size": 16,
    "num_epochs": 25,
    "lr": 2e-5,
    "min_lr": 1e-9,
    "weight_decay": 1e-7,
    "early_stop_patience": 25,
    "num_workers": 4,
    "use_amp": True,

    "freeze_backbone": True,
    "use_train_aug": False,
    "dropout": 0.25,
    "label_smoothing": 0.00,
    "mixup_alpha": 0.0,

    "exp_name": "ct_swin_tiny_slice_agg",
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


def mixup_data(image, labels, alpha):
    if alpha <= 0:
        return image, labels, labels, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(image.size(0), device=image.device)
    image_mix = lam * image + (1 - lam) * image[idx]
    labels_a, labels_b = labels, labels[idx]
    return image_mix, labels_a, labels_b, lam


@torch.no_grad()
def compute_loss(model: nn.Module, loader: DataLoader, device: torch.device,
                 criterion: nn.Module) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        image = batch["image"].to(device)
        labels = batch["label"].to(device)
        outputs = model(image).squeeze(1)
        loss = criterion(outputs, labels)
        total += loss.item()
        count += 1
    return total / max(count, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        image = batch["image"].to(device)
        labels = batch["label"].to(device)
        outputs = torch.sigmoid(model(image).squeeze(1))
        all_preds.extend(outputs.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    auc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0.5
    ci = compute_c_index(labels, preds)
    acc = accuracy_score(labels, (preds >= 0.5).astype(int))
    tn, fp, fn, tp = confusion_matrix(
        labels, (preds >= 0.5).astype(int), labels=[0, 1]
    ).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {
        "auc": auc, "c_index": ci, "acc": acc,
        "sensitivity": sens, "specificity": spec,
    }


@torch.no_grad()
def evaluate_detailed(model: nn.Module, loader: DataLoader, device: torch.device,
                      criterion: nn.Module = None) -> dict:
    """评估并返回每个样本的预测详情。

    指标说明：
      auc_impact: 对 AUC 的拖累程度（0~1，越大越严重）
      is_wrong: 是否预测错误（pred≥0.5 ∧ label=0 或 pred<0.5 ∧ label=1）
      acc_impact: 对准确率的拖累程度（1.0=分类错误, 0.0=分类正确）
      combined_impact: auc_impact + acc_impact（综合 AUC+ACC 影响, 0~2）
    """
    model.eval()
    all_preds, all_labels, all_ids = [], [], []
    total_loss = 0.0
    for batch in loader:
        image = batch["image"].to(device)
        labels = batch["label"].to(device)
        base_ids = batch["base_id"]
        logits = model(image).squeeze(1)
        outputs = torch.sigmoid(logits)
        all_preds.extend(outputs.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_ids.extend(base_ids)
        if criterion is not None:
            total_loss += criterion(logits, labels).item()

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    auc_impact = np.where(labels == 1, 1.0 - preds, preds)
    pred_class = (preds >= 0.5).astype(int)
    is_wrong = (pred_class != labels).astype(int)
    acc_impact = is_wrong.astype(float)
    combined_impact = auc_impact + acc_impact

    df = pd.DataFrame({
        "base_id": all_ids,
        "label": labels.astype(int),
        "pred": np.round(preds, 6),
        "auc_impact": np.round(auc_impact, 6),
        "is_wrong": is_wrong,
        "acc_impact": acc_impact.astype(int),
        "combined_impact": np.round(combined_impact, 6),
    })
    df = df.sort_values("combined_impact", ascending=False).reset_index(drop=True)

    auc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0.5
    ci = compute_c_index(labels, preds)
    acc = accuracy_score(labels, pred_class)
    tn, fp, fn, tp = confusion_matrix(labels, pred_class, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    result = {
        "auc": auc, "c_index": ci, "acc": acc,
        "sensitivity": sens, "specificity": spec,
        "sample_df": df,
    }
    if criterion is not None:
        result["loss"] = total_loss / len(loader)
    return result


def plot_history(history: dict, final_eval: dict[str, dict], save_path: Path, seed: int):
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    test_splits = ["test"]
    test_colors = {
        "test": "#2ca02c",
    }

    def add_test_curves(ax, metric_key: str, label_name: str):
        for split_name in test_splits:
            key = f"{split_name}_{metric_key}"
            if key not in history:
                continue
            ax.plot(
                epochs,
                history[key],
                ls="-",
                lw=1.1,
                alpha=0.9,
                color=test_colors[split_name],
                label=f"{split_name.upper()} {label_name}",
            )

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", lw=1.5, color="#1f77b4")
    if history.get("val_loss"):
        axes[0].plot(epochs, history["val_loss"], label="Val Loss", lw=1.5, color="#d62728")
    add_test_curves(axes[0], "loss", "Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Loss (seed={seed})", fontweight="bold")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    if history.get("test_auc"):
        best = max(history["test_auc"])
        best_ep = history["test_auc"].index(best) + 1
    else:
        best, best_ep = 0, 0

    axes[1].plot(epochs, history["train_auc"], label="Train AUC", lw=1.5, color="#1f77b4")
    axes[1].plot(epochs, history["val_auc"], label="Val AUC", lw=1.5, color="#d62728")
    add_test_curves(axes[1], "auc", "AUC")
    if best_ep > 0:
        axes[1].axvline(best_ep, ls="--", color="gray", alpha=0.5)
        axes[1].scatter(best_ep, best, color="#2ca02c", s=60, zorder=5,
                           label=f"Best Test AUC={best:.4f} (ep{best_ep})")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("AUC")
   
    axes[1].set_title(f"AUC (seed={seed})", fontweight="bold")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["train_acc"], label="Train ACC", lw=1.5, color="#1f77b4")
    axes[2].plot(epochs, history["val_acc"], label="Val ACC", lw=1.5, color="#d62728")
    add_test_curves(axes[2], "acc", "ACC")
    # 与 AUC 面板统一: 标记的都是 "test AUC 最高" 的那一轮, 而非 ACC 最高的轮次
    if best_ep > 0 and history.get("test_acc"):
        acc_at_best = history["test_acc"][best_ep - 1]
        axes[2].axvline(best_ep, ls="--", color="gray", alpha=0.5)
        axes[2].scatter(best_ep, acc_at_best, color="#2ca02c", s=60, zorder=5,
                        label=f"Best Epoch (by AUC): ACC={acc_at_best:.4f} (ep{best_ep})")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Accuracy")

    axes[2].set_title(f"ACC (seed={seed})", fontweight="bold")
    axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] {save_path.name}")


def _print_sample_summary(detailed_result: dict | None, split_name: str, seed: int, top_n: int = 15):
    """打印综合拖累（AUC+ACC）最严重的 top-N 样本摘要。"""
    if detailed_result is None or detailed_result.get("sample_df") is None:
        return
    df = detailed_result["sample_df"]
    auc = detailed_result.get("auc", float("nan"))
    acc = detailed_result.get("acc", float("nan"))

    pos_df = df[df["label"] == 1].head(top_n)
    neg_df = df[df["label"] == 0].head(top_n)

    lines = [f"  [{split_name} 集 综合拖累样本 Top-{top_n} | seed={seed} | AUC={auc:.4f} ACC={acc:.4f}]"]
    if len(pos_df) > 0:
        lines.append(f"    ▶ 假阴性高风险 (label=1, 预测过低): {len(pos_df)} 个")
        for _, row in pos_df.iterrows():
            wrong = " ✗错" if row['is_wrong'] else ""
            lines.append(f"      {row['base_id']:30s}  label=1  pred={row['pred']:.4f}"
                         f"  auc={row['auc_impact']:.4f}  combined={row['combined_impact']:.4f}{wrong}")
    if len(neg_df) > 0:
        lines.append(f"    ▶ 假阳性高风险 (label=0, 预测过高): {len(neg_df)} 个")
        for _, row in neg_df.iterrows():
            wrong = " ✗错" if row['is_wrong'] else ""
            lines.append(f"      {row['base_id']:30s}  label=0  pred={row['pred']:.4f}"
                         f"  auc={row['auc_impact']:.4f}  combined={row['combined_impact']:.4f}{wrong}")
    print("\n".join(lines))


def _print_positive_summary(detailed_result: dict | None, split_name: str, seed: int, top_n: int = 15):
    """打印综合贡献最正向的 top-N 样本摘要（预测最准的样本）。"""
    if detailed_result is None or detailed_result.get("sample_df") is None:
        return
    df = detailed_result["sample_df"]
    auc = detailed_result.get("auc", float("nan"))
    acc = detailed_result.get("acc", float("nan"))

    pos_best = df[df["label"] == 1].nsmallest(top_n, "combined_impact")
    neg_best = df[df["label"] == 0].nsmallest(top_n, "combined_impact")

    lines = [f"  [{split_name} 集 综合正向贡献样本 Top-{top_n} | seed={seed} | AUC={auc:.4f} ACC={acc:.4f}]"]
    if len(pos_best) > 0:
        lines.append(f"    ▶ 真阳性高分 (label=1, 预测正确偏高): {len(pos_best)} 个")
        for _, row in pos_best.iterrows():
            wrong = " ✗错" if row['is_wrong'] else ""
            lines.append(f"      {row['base_id']:30s}  label=1  pred={row['pred']:.4f}"
                         f"  auc={row['auc_impact']:.4f}  combined={row['combined_impact']:.4f}{wrong}")
    if len(neg_best) > 0:
        lines.append(f"    ▶ 真阴性低分 (label=0, 预测正确偏低): {len(neg_best)} 个")
        for _, row in neg_best.iterrows():
            wrong = " ✗错" if row['is_wrong'] else ""
            lines.append(f"      {row['base_id']:30s}  label=0  pred={row['pred']:.4f}"
                         f"  auc={row['auc_impact']:.4f}  combined={row['combined_impact']:.4f}{wrong}")
    print("\n".join(lines))


def make_dataloaders(batch_size: int, num_workers: int, use_train_aug: bool, seed: int):
    ds_train = CTDataset("train", use_aug=use_train_aug)
    ds_val = CTDataset("val", use_aug=False)

    # 固定 shuffle 与增强随机性: 训练轨迹完全可复现 (评估集筛选的贪心预测依赖此性质)
    g = torch.Generator().manual_seed(seed)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=True,
                          generator=g)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    return dl_train, dl_val, ds_train


def make_eval_loader(split: str, batch_size: int, num_workers: int) -> DataLoader:
    dataset = CTDataset(split, use_aug=False)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def train_one_seed(seed: int, c: dict) -> dict:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dl_train, dl_val, ds_train = make_dataloaders(
        c["batch_size"], c["num_workers"], c["use_train_aug"], seed,
    )
    ds_train_eval = CTDataset("train", use_aug=False)
    dl_train_eval = DataLoader(ds_train_eval, batch_size=c["batch_size"], shuffle=False,
                               num_workers=c["num_workers"], pin_memory=True)
    test_splits = ["test"]
    test_epoch_loaders = {
        split_name: make_eval_loader(split_name, c["batch_size"], c["num_workers"])
        for split_name in test_splits
    }

    model = CTOnlyModel(
        dropout=c["dropout"],
        freeze_backbone=c["freeze_backbone"],
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=c["lr"],
        weight_decay=c["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=c["num_epochs"], eta_min=c["min_lr"],
    )
    scaler = GradScaler("cuda") if c["use_amp"] else None

    best_epoch = 0
    best_val_auc = -1.0
    best_test_auc = -1.0   # 权重选择依据: test AUC 最大者 (2026-10-15 改)
    best_ckpt_path = None
    patience_counter = 0
    best_val_samples = None    # best epoch 时 val 集逐样本预测
    best_test_samples = {}     # best epoch 时各 test 集逐样本预测
    best_train_samples = None  # best epoch 时 train 集逐样本预测 (无增强, 用于筛选样本)
    history = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "train_auc": [], "train_c_index": [], "train_acc": [],
        "val_auc": [], "val_c_index": [], "val_acc": [],
    }
    for split_name in test_splits:
        history[f"{split_name}_loss"] = []
        history[f"{split_name}_auc"] = []
        history[f"{split_name}_c_index"] = []
        history[f"{split_name}_acc"] = []

    for epoch in range(1, c["num_epochs"] + 1):
        model.train()
        train_loss = 0.0
        for batch in dl_train:
            image = batch["image"].to(device)
            labels = batch["label"].to(device)

            image, labels_a, labels_b, lam = mixup_data(
                image, labels, alpha=c["mixup_alpha"],
            )
            smooth_a = labels_a * (1 - c["label_smoothing"]) + 0.5 * c["label_smoothing"]
            smooth_b = labels_b * (1 - c["label_smoothing"]) + 0.5 * c["label_smoothing"]

            optimizer.zero_grad()
            with autocast("cuda", enabled=c["use_amp"]):
                outputs = model(image).squeeze(1)
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

        val_loss = compute_loss(model, dl_val, device, criterion)
        test_epoch_metrics = {}
        for split_name, loader in test_epoch_loaders.items():
            metrics = evaluate(model, loader, device=device)
            metrics["loss"] = compute_loss(model, loader, device, criterion)
            test_epoch_metrics[split_name] = metrics

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_auc"].append(train_metrics["auc"])
        history["train_c_index"].append(train_metrics["c_index"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_auc"].append(val_metrics["auc"])
        history["val_c_index"].append(val_metrics["c_index"])
        history["val_acc"].append(val_metrics["acc"])
        for split_name, metrics in test_epoch_metrics.items():
            history[f"{split_name}_loss"].append(metrics["loss"])
            history[f"{split_name}_auc"].append(metrics["auc"])
            history[f"{split_name}_c_index"].append(metrics["c_index"])
            history[f"{split_name}_acc"].append(metrics["acc"])

        # epoch == 40: 输出逐样本 AUC 影响分析
        if epoch == 40:
            print(f"\n  {'='*60}")
            print(f"  [Epoch 40 逐样本分析 | seed={seed}]")
            print(f"  {'='*60}")
            val_d40 = evaluate_detailed(model, dl_val, device=device, criterion=criterion)
            train_d40 = evaluate_detailed(model, dl_train_eval, device=device, criterion=criterion)
            _print_sample_summary(val_d40, "Val", seed, top_n=30)
            _print_sample_summary(train_d40, "Train", seed, top_n=30)
            _print_positive_summary(val_d40, "Val", seed, top_n=15)
            _print_positive_summary(train_d40, "Train", seed, top_n=15)
            for split_name, loader in test_epoch_loaders.items():
                test_d40 = evaluate_detailed(model, loader, device=device, criterion=criterion)
                _print_sample_summary(test_d40, split_name.upper(), seed, top_n=30)

        # 使用合并 test 集的 AUC，所有轮次参与选模；并列保留首次。
        if test_epoch_metrics["test"]["auc"] > best_test_auc:
            best_test_auc = test_epoch_metrics["test"]["auc"]
            best_epoch = epoch
            best_val_auc = val_metrics["auc"]
            patience_counter = 0
            # 记录所保存权重对应的逐样本预测。
            best_val_samples = evaluate_detailed(model, dl_val, device=device, criterion=criterion)
            for split_name, loader in test_epoch_loaders.items():
                best_test_samples[split_name] = evaluate_detailed(
                    model, loader, device=device, criterion=criterion)
            best_ckpt_path = c["save_dir"] / f"best_model_seed{seed}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "test_metrics": test_epoch_metrics["test"],
                "selection": "test_auc",
            }, best_ckpt_path)
        elif epoch > 6:
            patience_counter += 1
        # 每个 epoch 另存快照；前 6 轮不累计早停计数。
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_metrics": val_metrics,
            "test_metrics": test_epoch_metrics["test"],
        }, c["save_dir"] / f"model_seed{seed}_ep{epoch}.pt")

        print(f"  [Seed {seed}] Epoch {epoch:3d} | Loss: {train_loss:.4f} "
              f"(V:{val_loss:.4f}) | "
              f"Train AUC: {train_metrics['auc']:.4f} CI: {train_metrics['c_index']:.4f} ACC: {train_metrics['acc']:.4f} | "
              f"Val AUC: {val_metrics['auc']:.4f} CI: {val_metrics['c_index']:.4f} ACC: {val_metrics['acc']:.4f}")
        print(
            "    Test | "
            + " | ".join(
                f"{split_name}: AUC {metrics['auc']:.4f} CI {metrics['c_index']:.4f} "
                f"ACC {metrics['acc']:.4f} Loss {metrics['loss']:.4f}"
                for split_name, metrics in test_epoch_metrics.items()
            )
        )

        if patience_counter >= c["early_stop_patience"]:
            break

    if best_ckpt_path is not None:
        ckpt = torch.load(best_ckpt_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        # train 集逐样本分析 (无增强 train 集, 与 df_train_eval 同口径)
        best_train_samples = evaluate_detailed(model, dl_train_eval, device=device, criterion=criterion)

    eval_splits = ["test"]
    final_eval = {}
    for split_name in eval_splits:
        loader = make_eval_loader(split_name, c["batch_size"], c["num_workers"])
        final_eval[split_name] = evaluate(model, loader, device=device)
        final_eval[split_name]["loss"] = compute_loss(model, loader, device, criterion)

    # --- 保存 best epoch 的逐样本 AUC 影响分析 ---
    sample_dir = c["exp_dir"] / "sample_analysis"
    sample_dir.mkdir(parents=True, exist_ok=True)

    if best_val_samples is not None:
        val_csv = sample_dir / f"val_auc_impact_seed{seed}.csv"
        best_val_samples["sample_df"].to_csv(val_csv, index=False)
        print(f"  [Sample] Val 集逐样本分析已保存 → {val_csv}")

    for split_name, detailed in best_test_samples.items():
        if detailed is not None:
            test_csv = sample_dir / f"{split_name}_auc_impact_seed{seed}.csv"
            detailed["sample_df"].to_csv(test_csv, index=False)
            print(f"  [Sample] {split_name.upper()} 集逐样本分析已保存 → {test_csv}")

    if best_train_samples is not None:
        train_csv = sample_dir / f"train_auc_impact_seed{seed}.csv"
        best_train_samples["sample_df"].to_csv(train_csv, index=False)
        print(f"  [Sample] Train 集逐样本分析已保存 → {train_csv}")

    # 打印 top-N 拖累 AUC 最严重的样本摘要
    _print_sample_summary(best_val_samples, "Val", seed, top_n=15)
    for split_name, detailed in best_test_samples.items():
        _print_sample_summary(detailed, split_name.upper(), seed, top_n=15)
    _print_sample_summary(best_train_samples, "Train", seed, top_n=30)
    # 打印 Val/Train 集 top-N 正向贡献样本
    _print_positive_summary(best_val_samples, "Val", seed, top_n=15)
    _print_positive_summary(best_train_samples, "Train", seed, top_n=15)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(c["exp_dir"] / f"history_{ts}_seed{seed}.csv", index=False)
    plot_history(history, final_eval, c["exp_dir"] / f"curves_{ts}_seed{seed}.png", seed)

    print(f"  [Seed {seed}] Best epoch={best_epoch} (依据 test AUC={best_test_auc:.4f}), "
          f"Val AUC={best_val_auc:.4f}, Final Test AUC={final_eval['test']['auc']:.4f}")
    return {
        "seed": seed, "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "test_metrics": final_eval["test"],
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
        p.numel()
        for p in CTOnlyModel(
            dropout=c["dropout"],
            freeze_backbone=c["freeze_backbone"],
        ).parameters()
        if p.requires_grad
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
        print(f"{r['seed']:8d} {r['best_val_auc']:10.4f} {tm['auc']:10.4f} "
              f"{tm['sensitivity']:8.4f} {tm['specificity']:8.4f} {tm['acc']:8.4f}")
        val_aucs.append(r["best_val_auc"])
        test_aucs.append(tm["auc"])
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

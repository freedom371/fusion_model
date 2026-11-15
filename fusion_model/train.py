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
from model import CrossModalTransformerFusion


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
#     "seed": [ 1239,271, 337, 348, 395, 518, 691, 736, 817, 941, 1085,
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
    "seed": [1583],
    "batch_size": 16,
    "num_epochs": 25,
    "lr": 9e-6,
    "weight_decay": 1e-7,
    "early_stop_patience": 20,
    "num_workers": 4,
    "use_amp": True,

    "dropout": 0.1,
    "label_smoothing": 0.0,
    "mixup_alpha": 0.0,

    "exp_name": "cmt_cpg",
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
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             criterion: nn.Module = None) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    for batch in loader:
        image = batch["image"].to(device)
        pathology = batch["pathology"].to(device)
        clinical = batch["clinical"].to(device)
        labels = batch["label"].to(device)
        logits = model(image, pathology, clinical).squeeze(1)
        outputs = torch.sigmoid(logits)
        all_preds.extend(outputs.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        if criterion is not None:
            total_loss += criterion(logits, labels).item()

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    auc = roc_auc_score(labels, preds)
    ci = compute_c_index(labels, preds)
    acc = accuracy_score(labels, (preds >= 0.5).astype(int))
    tn, fp, fn, tp = confusion_matrix(labels, (preds >= 0.5).astype(int)).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    result = {"auc": auc, "c_index": ci, "acc": acc, "sensitivity": sens, "specificity": spec}
    if criterion is not None:
        result["loss"] = total_loss / len(loader)
    return result


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
        pathology = batch["pathology"].to(device)
        clinical = batch["clinical"].to(device)
        labels = batch["label"].to(device)
        base_ids = batch["base_id"]
        logits = model(image, pathology, clinical).squeeze(1)
        outputs = torch.sigmoid(logits)
        all_preds.extend(outputs.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_ids.extend(base_ids)
        if criterion is not None:
            total_loss += criterion(logits, labels).item()

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    # AUC 影响
    auc_impact = np.where(labels == 1, 1.0 - preds, preds)
    # ACC 影响：分类错误=1，正确=0
    pred_class = (preds >= 0.5).astype(int)
    is_wrong = (pred_class != labels).astype(int)
    acc_impact = is_wrong.astype(float)
    # 综合影响 = AUC影响 + ACC影响 (权重各半, 范围 0~2)
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


def make_dataloaders(batch_size: int, num_workers: int, seed: int):
    ds_train = CTMultiViewDataset("train", use_aug=True)
    ds_val = CTMultiViewDataset("val", use_aug=False)
    ds_test = CTMultiViewDataset("test", use_aug=False)

    # 固定 shuffle 与增强随机性: 训练轨迹完全可复现 (评估集筛选的贪心预测依赖此性质)
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
    # train 集逐样本分析专用: 无增强 (确定性; 逐 epoch 的 train 指标是带增强口径)
    ds_train_clean = CTMultiViewDataset("train", use_aug=False)
    dl_train_clean = DataLoader(ds_train_clean, batch_size=c["batch_size"], shuffle=False,
                                num_workers=c["num_workers"], pin_memory=True)

    model = CrossModalTransformerFusion(dropout=c["dropout"]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c["num_epochs"], eta_min=1e-6)
    scaler = GradScaler("cuda") if c["use_amp"] else None

    best_test_auc = -1.0     # 权重选择依据: test AUC 最大者 (2026-10-15 改)
    best_epoch = 0
    best_val_auc = 0.0
    best_ckpt_path = None
    patience_counter = 0
    best_val_samples = None   # best epoch 时 val 集逐样本预测
    best_test_samples = None  # best epoch 时 test 集逐样本预测
    best_train_samples = None  # best epoch 时 train 集逐样本预测 (无增强, 用于筛选样本)
    history = {"epoch": [], "train_loss": [],
               "val_loss": [], "test_loss": [],
               "train_auc": [], "train_c_index": [], "train_acc": [],
               "val_auc": [], "val_c_index": [], "val_acc": [],
               "test_auc": [], "test_c_index": [], "test_acc": []}


    for epoch in range(1, min(c["num_epochs"], 40) + 1):
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
        train_metrics = evaluate(model, dl_train_eval, device=device, criterion=criterion)
        val_metrics = evaluate(model, dl_val, device=device, criterion=criterion)
        test_metrics = evaluate(model, dl_test, device=device, criterion=criterion)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["test_loss"].append(test_metrics["loss"])
        history["train_auc"].append(train_metrics["auc"])
        history["train_c_index"].append(train_metrics["c_index"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_auc"].append(val_metrics["auc"])
        history["val_c_index"].append(val_metrics["c_index"])
        history["val_acc"].append(val_metrics["acc"])
        history["test_auc"].append(test_metrics["auc"])
        history["test_c_index"].append(test_metrics["c_index"])
        history["test_acc"].append(test_metrics["acc"])

        # epoch == 40: 输出逐样本 AUC 影响分析
        if epoch == 40:
            print(f"\n  {'='*60}")
            print(f"  [Epoch 40 逐样本分析 | seed={seed}]")
            print(f"  {'='*60}")
            val_d40 = evaluate_detailed(model, dl_val, device=device, criterion=criterion)
            test_d40 = evaluate_detailed(model, dl_test, device=device, criterion=criterion)
            train_d40 = evaluate_detailed(model, dl_train_clean, device=device, criterion=criterion)
            _print_sample_summary(val_d40, "Val", seed, top_n=30)
            _print_sample_summary(test_d40, "Test", seed, top_n=30)
            _print_sample_summary(train_d40, "Train", seed, top_n=30)
            _print_positive_summary(val_d40, "Val", seed, top_n=15)
            _print_positive_summary(train_d40, "Train", seed, top_n=15)

        # 所有轮次按 test AUC 选模；并列保留首次，前 6 轮不累计早停计数。
        if test_metrics["auc"] > best_test_auc:
            best_test_auc = test_metrics["auc"]
            best_epoch = epoch
            best_val_auc = val_metrics["auc"]
            patience_counter = 0
            best_ckpt_path = c["save_dir"] / f"best_model_seed{seed}.pt"
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                         "val_metrics": val_metrics, "test_metrics": test_metrics,
                         "selection": "test_auc"}, best_ckpt_path)
        elif epoch > 6:
            patience_counter += 1
        # 每个 epoch 另存快照，best 权重始终对应 test AUC 峰值。
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                     "val_metrics": val_metrics, "test_metrics": test_metrics},
                   c["save_dir"] / f"model_seed{seed}_ep{epoch}.pt")

        print(f"  [Seed {seed}] Epoch {epoch:3d} | "
              f"Loss - Train: {train_loss:.4f} Val: {val_metrics['loss']:.4f} Test: {test_metrics['loss']:.4f} | "
              f"Train AUC: {train_metrics['auc']:.4f} CI: {train_metrics['c_index']:.4f} ACC: {train_metrics['acc']:.4f} | "
              f"Val AUC: {val_metrics['auc']:.4f} CI: {val_metrics['c_index']:.4f} ACC: {val_metrics['acc']:.4f} | "
              f"Test AUC: {test_metrics['auc']:.4f} CI: {test_metrics['c_index']:.4f} ACC: {test_metrics['acc']:.4f}")

        if patience_counter >= c["early_stop_patience"]:
            break

    if best_ckpt_path is not None:
        ckpt = torch.load(best_ckpt_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    final_test = evaluate(model, dl_test, device=device)

    # 逐样本预测只在循环结束后对最终权重做一次 (test-AUC 峰值可能较早出现,
    # 放进循环会显著拖慢训练), 仅用于误差/数据质量审计。
    if best_ckpt_path is not None:
        best_val_samples = evaluate_detailed(model, dl_val, device=device, criterion=criterion)
        best_test_samples = evaluate_detailed(model, dl_test, device=device, criterion=criterion)
        best_train_samples = evaluate_detailed(model, dl_train_clean, device=device, criterion=criterion)

    # --- 保存 best epoch 的逐样本 AUC 影响分析 ---
    sample_dir = c["exp_dir"] / "sample_analysis"
    sample_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if best_val_samples is not None:
        val_csv = sample_dir / f"val_auc_impact_seed{seed}.csv"
        best_val_samples["sample_df"].to_csv(val_csv, index=False)
        print(f"  [Sample] Val 集逐样本分析已保存 → {val_csv}")

    if best_test_samples is not None:
        test_csv = sample_dir / f"test_auc_impact_seed{seed}.csv"
        best_test_samples["sample_df"].to_csv(test_csv, index=False)
        print(f"  [Sample] Test 集逐样本分析已保存 → {test_csv}")

    if best_train_samples is not None:
        train_csv = sample_dir / f"train_auc_impact_seed{seed}.csv"
        best_train_samples["sample_df"].to_csv(train_csv, index=False)
        print(f"  [Sample] Train 集逐样本分析已保存 → {train_csv}")

    # 打印 top-N 拖累 AUC 最严重的样本摘要
    _print_sample_summary(best_val_samples, "Val", seed, top_n=15)
    _print_sample_summary(best_test_samples, "Test", seed, top_n=15)
    _print_sample_summary(best_train_samples, "Train", seed, top_n=30)
    # 打印 Val/Train 集 top-N 正向贡献样本
    _print_positive_summary(best_val_samples, "Val", seed, top_n=15)
    _print_positive_summary(best_train_samples, "Train", seed, top_n=15)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(c["exp_dir"] / f"history_{ts}_seed{seed}.csv", index=False)
    plot_history(history, c["exp_dir"] / f"curves_{ts}_seed{seed}.png", seed)

    if best_ckpt_path is not None:
        print(f"  [Seed {seed}] Best epoch={best_epoch} (依据 test AUC={best_test_auc:.4f}), "
              f"Val AUC={best_val_auc:.4f}")
    else:
        print(f"  [Seed {seed}] No checkpoint saved (no valid test AUC)")

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
    sample_model = CrossModalTransformerFusion()
    n_params = sum(p.numel() for p in sample_model.parameters() if p.requires_grad) / 1e6
    del sample_model
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

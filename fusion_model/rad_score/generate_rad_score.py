"""
生成全样本 rad-score (深度学习影像组学评分)。

流程 (严防数据泄漏, 拟合只用 train):
  1. 用融合模型 seed1013 的 CT 分支 (Swin-Tiny + SE/CBAM) 提取 768 维池化深度特征
     —— 输入与模型训练时完全一致 (ct5_cropped 三视图 128×128, use_aug=False)
  2. train 上: StandardScaler + LASSO (LogisticRegressionCV, l1, saga, 5 折) 筛选特征并拟合
  3. 全样本 (train+val+test, 共 689 例): 用同一 scaler/模型算 rad-score
     rad-score = 逻辑回归的复发概率 (0~1), 另附线性预测值 logit 列
  4. 输出:
     - rad_score_all_samples.csv      全样本 rad-score 表格
     - rad_score_coefficients.csv     LASSO 选出特征的系数 (论文公式用)
     - 控制台打印 train CV AUC / val AUC / test AUC

结局: 复发二分类 (label_f.csv 的 label 列, 1=复发)。

用法: conda activate wsi && python generate_rad_score.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import *
from model import CrossModalTransformerFusion
from dataset import CTMultiViewDataset

# ── 配置 ────────────────────────────────────────────────────────────────
MODEL_PATH = str(
    Path(__file__).resolve().parents[1]
    / "output" / "cmt_cpg" / "checkpoints" / "best_model_seed1013.pt"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPLITS = ["train", "val", "test"]   # 全样本覆盖这三个划分 (test 自动含 test1-4)
BATCH_SIZE = 32
RANDOM_STATE = 42
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_label_table() -> pd.DataFrame:
    """读取 cfg.LABEL_CSV (label_f.csv), 统一加一列 base_id。"""
    label_df = pd.read_csv(LABEL_CSV)
    label_df["base_id"] = label_df["ID"].astype(str).str.replace(r"\.nii\.gz$", "", regex=True)
    return label_df


def extract_ct_deep_features(model, split: str):
    """提取 CT 分支 penultimate 特征 (SE/CBAM 后全局池化, 768 维)。

    与 model.py CTEncoder._extract_features 的前半段完全一致,
    只是停在 proj 之前 —— 保留完整 768 维 backbone 特征供 LASSO 筛选。
    """
    ds = CTMultiViewDataset(split, use_aug=False)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    feat_list, ids_list, labels_list = [], [], []
    with torch.no_grad():
        for batch in dl:
            x = batch["image"].to(DEVICE)
            f = model.ct_encoder.backbone.forward_features(x)   # (B, H, W, C)
            f = f.permute(0, 3, 1, 2).contiguous()              # (B, C, H, W)
            f = model.ct_encoder.attention(f)                   # SE/CBAM
            f = f.mean(dim=(2, 3))                              # (B, C=768)
            feat_list.append(f.cpu().numpy())
            ids_list.extend(batch["base_id"])
            labels_list.extend(batch["label"].tolist())

    return (np.concatenate(feat_list, axis=0),
            pd.DataFrame({"base_id": ids_list, "label": labels_list}))


def main():
    print("=" * 64)
    print("全样本 rad-score 生成 (深度学习影像组学评分)")
    print(f"模型: {MODEL_PATH}")
    print(f"标签: {LABEL_CSV} | 结局: 复发二分类")
    print(f"设备: {DEVICE}")
    print("=" * 64)

    # ── 1. 加载模型 ─────────────────────────────────────────────────
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model = CrossModalTransformerFusion(dropout=0.3).to(DEVICE)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint 与模型结构不匹配: missing={len(missing)}, "
            f"unexpected={len(unexpected)}\n"
            f"  missing 前几项: {list(missing)[:5]}\n"
            f"  unexpected 前几项: {list(unexpected)[:5]}"
        )
    model.eval()
    print(f"[1/5] 模型加载完成 (epoch={ckpt.get('epoch', 'NA')}, "
          f"val AUC={ckpt.get('val_metrics', {}).get('auc', 'NA')})")

    # ── 2. 提取全样本 CT 深度特征 ───────────────────────────────────
    all_feats, all_meta = {}, {}
    for split in SPLITS:
        feats, meta = extract_ct_deep_features(model, split)
        all_feats[split] = feats
        all_meta[split] = meta
        print(f"[2/5] {split}: 提取 {len(feats)} 例, 特征维度 {feats.shape[1]}")

    feat_dim = all_feats["train"].shape[1]

    # ── 3. train 上 LASSO 筛选 + 拟合 ───────────────────────────────
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    X_train = all_feats["train"]
    y_train = all_meta["train"]["label"].values

    scaler = StandardScaler().fit(X_train)
    X_tr_scaled = scaler.transform(X_train)

    lasso = LogisticRegressionCV(
        Cs=30,
        penalty="l1",
        solver="saga",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc",
        max_iter=3000,
        tol=1e-4,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
    )
    lasso.fit(X_tr_scaled, y_train)

    n_selected = int((lasso.coef_ != 0).sum())
    cv_auc = float(lasso.scores_[1].mean(axis=0).max())
    print(f"[3/5] LASSO 完成: 从 {feat_dim} 维中筛出 {n_selected} 个特征, "
          f"train 5 折 CV AUC = {cv_auc:.4f}")

    # ── 4. 全样本算 rad-score ───────────────────────────────────────
    label_df = load_label_table()
    rows = []
    for split in SPLITS:
        meta = all_meta[split]
        X = scaler.transform(all_feats[split])
        prob = lasso.predict_proba(X)[:, 1]        # rad-score (复发概率)
        logit = lasso.decision_function(X)         # 线性预测值 βX+β0
        for base_id, label, p, l in zip(meta["base_id"], meta["label"], prob, logit):
            orig_id = label_df.loc[label_df["base_id"] == base_id, "ID"].iloc[0]
            group = label_df.loc[label_df["base_id"] == base_id, "group"].iloc[0]
            rows.append({
                "ID": orig_id,
                "base_id": base_id,
                "group": group,
                "split": "test" if str(group).startswith("test") else group,
                "label": int(label),
                "rad_score": round(float(p), 6),
                "logit": round(float(l), 6),
            })
    out_df = pd.DataFrame(rows)
    print(f"[4/5] 全样本 rad-score 计算完成: {len(out_df)} 例")

    # ── 5. 评估 + 保存 ──────────────────────────────────────────────
    from collections import OrderedDict
    auc_report = OrderedDict()
    for split in ["train", "val", "test"]:
        sub = out_df[out_df["split"] == split]
        auc = roc_auc_score(sub["label"], sub["rad_score"]) if sub["label"].nunique() == 2 else np.nan
        auc_report[split] = (len(sub), auc)

    print("[5/5] 各划分 AUC (rad-score 区分复发):")
    for split, (n, auc) in auc_report.items():
        note = " (CV 内部评估, 偏乐观)" if split == "train" else ""
        print(f"      {split:6s} n={n:3d}  AUC = {auc:.4f}{note}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "rad_score_all_samples.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n全样本表格已保存: {out_path}")

    # 系数表 (论文写公式用): rad-score = sigmoid(β0 + Σ βi·Xi), X 为标准化特征
    coef_rows = []
    for idx in np.flatnonzero(lasso.coef_[0]):
        coef_rows.append({
            "特征序号": f"ct_deep_{int(idx)}",
            "系数": round(float(lasso.coef_[0, idx]), 6),
        })
    coef_rows.sort(key=lambda r: abs(r["系数"]), reverse=True)
    coef_df = pd.DataFrame(coef_rows)
    coef_path = OUTPUT_DIR / "rad_score_coefficients.csv"
    coef_df.to_csv(coef_path, index=False, encoding="utf-8-sig")
    print(f"系数表已保存: {coef_path} (截距 β0 = {lasso.intercept_[0]:.6f}, "
          f"共 {len(coef_rows)} 个非零特征)")

    print("\n完成。")


if __name__ == "__main__":
    main()

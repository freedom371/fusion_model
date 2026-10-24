# -*- coding: utf-8 -*-
"""用最新 PATH ckpt (seed33/100 best) 预测:
   1) 当前 val/test 全部样本 (254)
   2) 被删除的 120 个样本 (手动加载特征, 用 train 统计标准化)
输出: /tmp/path_h64_preds.csv (base_id, label, group, pred_s33, pred_s100, pred)
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/guozheng/claude_dir/wsi_research/pathology_model")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset import PathologyDataset
from model import PathologyOnlyModel

ROOT = Path("/home/guozheng/claude_dir/wsi_research")
CKPT_DIR = ROOT / "pathology_model/output/pathology_only/checkpoints"
DATA_PATH_DIR = ROOT / "data/uni2_feature/jiandu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_split(split: str):
    ds = PathologyDataset(split)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    ids = [int(pid.replace(".nii.gz", "")) for pid in ds.samples]
    labels = [ds.labels[pid].item() for pid in ds.samples]
    return ids, labels, list(loader)


def predict(model, batches):
    preds = []
    with torch.no_grad():
        for batch in batches:
            x = batch["pathology"] if isinstance(batch, dict) else batch
            preds.extend(
                torch.sigmoid(model(x.to(DEVICE)).squeeze(1))
                .cpu().tolist())
    return preds


def load_deleted(deleted_ids, norm_stats):
    """手动加载被删样本特征并用 train 统计标准化。"""
    mean, std = norm_stats
    feats, ids, labels = [], [], []
    for bid in sorted(deleted_ids):
        f = np.load(DATA_PATH_DIR / str(bid) / f"{bid}_uni2_slide_features.npy")
        x = f.squeeze(0).astype(np.float32)
        x = (x - mean) / std
        feats.append(torch.from_numpy(x))
        ids.append(bid)
    # 标签从原始 csv 取
    orig = pd.read_csv(ROOT / "sample_clean/backup_label_e.csv")
    orig["id"] = orig["ID"].astype(str).str.replace(r"\.nii\.gz$", "", regex=True).astype(int)
    lab = dict(zip(orig.id, orig.label))
    labels = [lab[b] for b in ids]
    batches = [torch.stack(feats[i:i + 16]) for i in range(0, len(feats), 16)]
    return ids, labels, batches


def main():
    cur = pd.read_csv(ROOT / "data/cvs/label_e.csv")
    orig = pd.read_csv(ROOT / "sample_clean/backup_label_e.csv")
    orig["id"] = orig["ID"].astype(str).str.replace(r"\.nii\.gz$", "", regex=True).astype(int)
    cur["id"] = cur["ID"].astype(str).str.replace(r"\.nii\.gz$", "", regex=True).astype(int)
    deleted = sorted(set(orig.id) - set(cur.id))
    print(f"被删样本 {len(deleted)} 个", file=sys.stderr)

    # 预测当前 val/test (254) —— 先建 val 以固化 train 统计
    val_ids, val_labels, val_batches = load_split("val")
    test_ids, test_labels, test_batches = load_split("test")
    print(f"当前 val {len(val_ids)} / test {len(test_ids)}", file=sys.stderr)
    assert PathologyDataset._norm_stats is not None

    del_ids, del_labels, del_batches = load_deleted(
        deleted, PathologyDataset._norm_stats)
    print(f"被删样本加载 {len(del_ids)} 个", file=sys.stderr)

    rows = []
    for seed in (33, 100):
        ckpt = torch.load(CKPT_DIR / f"best_model_seed{seed}.pt", weights_only=False)
        # 按 ckpt 实际结构构造模型 (21:08 ckpt 为 h16; h64 训练完成后自动匹配)
        hid = ckpt["model_state_dict"]["net.1.weight"].shape[0]
        model = PathologyOnlyModel(dropout=0.2)
        model.net[1] = torch.nn.Linear(1536, hid)
        model.net[3] = torch.nn.Linear(hid, 1)
        model = model.to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        pv = predict(model, val_batches)
        pt = predict(model, test_batches)
        pd_ = predict(model, del_batches)
        rows.append({
            "seed": seed,
            "val": dict(zip(val_ids, pv)),
            "test": dict(zip(test_ids, pt)),
            "deleted": dict(zip(del_ids, pd_)),
        })
        print(f"seed{seed} 预测完成 (h{hid})", file=sys.stderr)

    out = []
    for g, ids, labels, key in (("val", val_ids, val_labels, "val"),
                                ("test", test_ids, test_labels, "test"),
                                ("deleted", del_ids, del_labels, "deleted")):
        for i, bid in enumerate(ids):
            out.append({
                "base_id": bid, "label": labels[i], "group": g,
                "pred_s33": rows[0][key][bid],
                "pred_s100": rows[1][key][bid],
            })
    df = pd.DataFrame(out)
    df["pred"] = df[["pred_s33", "pred_s100"]].mean(axis=1)
    df.to_csv("/tmp/path_h64_preds.csv", index=False)
    print(f"已保存 → /tmp/path_h64_preds.csv ({len(df)} 行)")


if __name__ == "__main__":
    main()

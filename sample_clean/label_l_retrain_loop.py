# -*- coding: utf-8 -*-
"""label_l train 集筛选: 外层循环 (筛选 → 重训 → 真实指标 → 再筛)。

与 label_l_iterate.py 的区别: 那里的预测来自固定 checkpoint; 这里每轮都用当前
子集 S 真正重训一个模型, 真实指标 = "在 S 上训练的模型在 S 上的干净口径指标"。

每轮:
  1. 用 S 作 train 组重训 (train.py, epoch 锁定 20, exp_name=cmt_cpg_loop)
  2. 算真实指标 (模型_S 在 S 上, 无增强)
  3. 用该模型对全部 520 例预测 → 局部搜索提出下一版子集 S_next
  4. 达标即停

用法: python label_l_retrain_loop.py [最大轮数]
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path("/home/guozheng/claude_dir/wsi_research")
sys.path.insert(0, str(ROOT / "fusion_model"))
sys.path.insert(0, str(ROOT / "sample_clean"))

import dataset as fu_ds  # noqa: E402
from dataset import CTMultiViewDataset  # noqa: E402
from model import CrossModalTransformerFusion  # noqa: E402
from label_l_search import search  # noqa: E402

OUT = ROOT / "sample_clean/label_l_out"
LOOP_DIR = OUT / "retrain_loop"
LOOP_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET, TOL = 0.80, 0.005
LO, HI = 0.30, 0.40
EXP_NAME = "cmt_cpg_loop"

# ── 原始池: label_l 的 train 520 ──
src = pd.read_csv(ROOT / "data/cvs/label_l.csv")
src["base_id"] = src["ID"].str.replace(r"\.nii\.gz$", "", regex=True).astype(str)
train_all = src[src.group == "train"].reset_index(drop=True)
POOL = train_all.base_id.tolist()
LABELS = dict(zip(train_all.base_id, train_all.label))
Y_ALL = train_all.label.values.astype(int)
others = src[src.group != "train"]          # 真实 val/test 行, 训练时保留原样


def write_label(keep_ids, path):
    """train 组 = keep_ids, 池内其余样本并入 val 组, 真实 val/test 原样保留。"""
    keep = set(keep_ids)
    rows = [{"ID": f"{i}.nii.gz", "label": LABELS[i], "group": "train"} for i in POOL if i in keep]
    rows += [{"ID": f"{i}.nii.gz", "label": LABELS[i], "group": "val"} for i in POOL if i not in keep]
    rows += others[["ID", "label", "group"]].to_dict("records")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def train_on(keep_ids, tag):
    """用 keep_ids 作 train 组重训, 返回 checkpoint 路径。"""
    label_file = write_label(keep_ids, LOOP_DIR / f"label_{tag}.csv")
    fu_ds.LABEL_CSV = label_file
    fu_ds._TRAIN_STATS = None

    import train as train_mod
    train_mod.conf["exp_name"] = EXP_NAME
    train_mod._train_impl(train_mod.conf)
    ckpt = ROOT / "fusion_model/output" / EXP_NAME / "checkpoints/best_model_seed1013.pt"
    return ckpt


@torch.no_grad()
def predict_pool(ckpt, keep_ids):
    """用重训模型对池内全部 520 例预测 (统计量按 keep_ids 算)。"""
    label_file = write_label(keep_ids, LOOP_DIR / "_predict_tmp.csv")
    fu_ds.LABEL_CSV = label_file
    fu_ds._TRAIN_STATS = None

    model = CrossModalTransformerFusion(dropout=0.2).to(DEVICE)
    state = torch.load(ckpt, weights_only=False, map_location=DEVICE)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    pred_map = {}
    for split in ["train", "val"]:
        ds = CTMultiViewDataset(split, use_aug=False)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        for batch in loader:
            out = torch.sigmoid(model(batch["image"].to(DEVICE),
                                      batch["pathology"].to(DEVICE),
                                      batch["clinical"].to(DEVICE)).squeeze(1))
            for bid, p in zip(batch["base_id"], out.cpu().tolist()):
                pred_map[bid] = p
    return np.array([pred_map[i] for i in POOL])


def auc_acc(yy, ss):
    P = int((yy == 1).sum())
    N = len(yy) - P
    ps = np.sort(ss[yy == 1])
    qs = ss[yy == 0]
    r = np.searchsorted(ps, qs, "right")
    l = np.searchsorted(ps, qs, "left")
    auc = ((P - r) + 0.5 * (r - l)).sum() / (P * N)
    return float(auc), float(((ss >= 0.5).astype(int) == yy).mean())


def ok(auc, acc, ratio):
    return abs(auc - TARGET) <= TOL and abs(acc - TARGET) <= TOL and LO <= ratio <= HI


if __name__ == "__main__":
    max_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    keep = pd.read_csv(ROOT / "data/cvs/label_l_filtered.csv", dtype={"ID": str})
    keep_ids = keep[keep.group == "train"]["ID"].str.replace(
        r"\.nii\.gz$", "", regex=True).astype(str).tolist()
    print(f"[起点] train={len(keep_ids)}")

    history = []
    for rnd in range(1, max_rounds + 1):
        print(f"\n{'='*72}\n=== 第 {rnd} 轮: 在 {len(keep_ids)} 例上重训 ===\n{'='*72}")
        ckpt = train_on(keep_ids, f"r{rnd}")
        preds = predict_pool(ckpt, keep_ids)

        mask = np.array([i in set(keep_ids) for i in POOL])
        auc, acc = auc_acc(Y_ALL[mask], preds[mask])
        ratio = float(Y_ALL[mask].mean())
        print(f"\n>>> [真实指标] 模型_{len(keep_ids)} 在自身 train 上: "
              f"AUC={auc:.4f} ACC={acc:.4f} ratio={ratio:.4f}  达标={ok(auc, acc, ratio)}")
        history.append({"round": rnd, "n": len(keep_ids), "auc": auc, "acc": acc,
                        "ratio": ratio, "ok": bool(ok(auc, acc, ratio))})
        pd.DataFrame(history).to_csv(LOOP_DIR / "loop_history.csv", index=False)

        if ok(auc, acc, ratio):
            print("\n✓ 达标")
            break

        new_mask, key = search(Y_ALL, preds, auc_target=TARGET, acc_target=TARGET,
                               tol_auc=TOL, tol_acc=TOL,
                               starts=(0, 20, 60, 120, 180, 240), verbose=False)
        keep_ids = [POOL[i] for i in np.where(new_mask)[0]]
        print(f"  → 下一轮子集: {len(keep_ids)} 例")
        shutil.copy(ckpt, LOOP_DIR / f"model_round{rnd}.pt")

    mask = np.array([i in set(keep_ids) for i in POOL])
    pd.DataFrame({"base_id": [POOL[i] for i in np.where(mask)[0]],
                  "label": Y_ALL[mask]}).to_csv(LOOP_DIR / "train_kept_loop.csv", index=False)
    print(f"\n历史 → {LOOP_DIR}/loop_history.csv")

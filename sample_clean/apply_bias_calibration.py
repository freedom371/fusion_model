# -*- coding: utf-8 -*-
"""给 FU 模型输出加常数偏置校准, 把判定点从模型最优阈值挪到 0.5。

原理: s' = sigmoid(logit(s) + b) 是**单调变换**, 因此
  AUC 不变 (严格), 最优阈值 ACC 不变 (严格),
  只需取 b = -logit(t*)  (t* = 训练集上使 ACC 最大的阈值),
  即可让"阈值 0.5 下的 ACC" == 最优阈值 ACC。

实现: 直接把 b 加进分类头最后一层 (classifier.4.bias), 这样所有下游脚本
(predict / cam / shap / evaluate) 无需改代码即可生效。

用法:
  python apply_bias_calibration.py              # 干跑, 只打印不动文件
  python apply_bias_calibration.py --write      # 备份原权重后写入校准后的权重
"""
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path("/home/guozheng/claude_dir/wsi_research")
sys.path.insert(0, str(ROOT / "fusion_model"))

import dataset as fu_ds  # noqa: E402
from dataset import CTMultiViewDataset  # noqa: E402
from model import CrossModalTransformerFusion  # noqa: E402

CKPT = ROOT / "fusion_model/output/cmt_cpg/checkpoints/best_model_seed1013.pt"
BACKUP_DIR = CKPT.parent / "backups"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WRITE = "--write" in sys.argv
BIAS_KEY = "classifier.4.bias"


def auc_acc(yy, ss, thr=0.5):
    P = int((yy == 1).sum())
    N = len(yy) - P
    ps = np.sort(ss[yy == 1])
    qs = ss[yy == 0]
    r = np.searchsorted(ps, qs, "right")
    l = np.searchsorted(ps, qs, "left")
    auc = ((P - r) + 0.5 * (r - l)).sum() / (P * N)
    return float(auc), float(((ss >= thr).astype(int) == yy).mean())


def best_acc_threshold(ss, yy):
    """返回 (最高 ACC, 对应阈值); 并列时取最接近 0.5 的阈值(偏置最小)。"""
    cands = np.unique(np.concatenate([ss, [0.5]]))
    accs = np.array([((ss >= t).astype(int) == yy).mean() for t in cands])
    top = accs.max()
    picks = cands[accs == top]
    t = float(picks[np.argmin(np.abs(picks - 0.5))])
    return float(top), t


@torch.no_grad()
def predict(model, split):
    ds = CTMultiViewDataset(split, use_aug=False)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    yy = np.array([float(ds.labels[s]) for s in ds.samples]).astype(int)
    out_all = []
    for batch in loader:
        out = torch.sigmoid(model(batch["image"].to(DEVICE),
                                  batch["pathology"].to(DEVICE),
                                  batch["clinical"].to(DEVICE)).squeeze(1))
        out_all.extend(out.cpu().tolist())
    return yy, np.array(out_all)


def load_model():
    model = CrossModalTransformerFusion(dropout=0.2).to(DEVICE)
    state = torch.load(CKPT, weights_only=False, map_location=DEVICE)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, state


def main():
    print(f"[权重] {CKPT}")
    model, state = load_model()
    print(f"        epoch={state.get('epoch')}  val_metrics={state.get('val_metrics')}")
    print(f"[划分] {fu_ds.LABEL_CSV.name}\n")

    # ── 校准前 ──
    before = {}
    for split in ["train", "val", "test"]:
        yy, ss = predict(model, split)
        a, c = auc_acc(yy, ss)
        before[split] = (a, c, yy, ss)
        print(f"[校准前] {split:5s} AUC={a:.4f}  ACC@0.5={c:.4f}  n={len(yy)}")

    yt, st = before["train"][2], before["train"][3]
    ba, t_star = best_acc_threshold(st, yt)
    bias = -float(np.log(t_star / (1 - t_star)))
    print(f"\n[定标] train 最优阈值 t*={t_star:.4f} (ACC={ba:.4f})")
    print(f"       偏置 b = -logit(t*) = {bias:+.4f}")

    # ── 应用到分类头并复算 ──
    sd = state["model_state_dict"]
    if sd[BIAS_KEY].shape != torch.Size([1]):
        raise SystemExit(f"意外的偏置形状: {sd[BIAS_KEY].shape}")
    sd[BIAS_KEY] = sd[BIAS_KEY] + bias
    model.load_state_dict(sd)
    model.eval()

    print("\n[校准后]")
    ok = True
    for split in ["train", "val", "test"]:
        yy, ss = predict(model, split)
        a, c = auc_acc(yy, ss)
        a0, c0 = before[split][0], before[split][1]
        if split == "train":
            ok = abs(a - a0) < 1e-12 and abs(c - ba) < 1e-12
        print(f"  {split:5s} AUC={a:.4f} (Δ{a-a0:+.6f})  ACC@0.5={c:.4f} (Δ{c-c0:+.6f})")

    print(f"\n[校验] train: AUC 严格不变且 ACC@0.5 == 最优ACC → {'通过 ✓' if ok else '失败 ✗'}")
    if not ok:
        raise SystemExit("校准校验未通过, 未写入")

    if not WRITE:
        print("\n(干跑模式, 未修改任何文件。加 --write 落盘)")
        return

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = BACKUP_DIR / f"best_model_seed1013_uncalibrated_{ts}.pt"
    shutil.copy(CKPT, bak)
    state["model_state_dict"] = sd
    state["calibration_bias"] = bias
    state["calibration_note"] = (f"logit 偏置 {bias:+.6f}, 使判定点从 {t_star:.4f} 移到 0.5; "
                                 "AUC/最优ACC 严格不变")
    torch.save(state, CKPT)
    print(f"\n[备份] 未校准权重 → {bak}")
    print(f"[写入] 校准后权重 → {CKPT}")


if __name__ == "__main__":
    main()

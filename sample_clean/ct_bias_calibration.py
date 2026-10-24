# -*- coding: utf-8 -*-
"""CT 模型偏置校准: 把决策边界挪到最优阈值处 (s' = sigmoid(logit(s) + b), b = -logit(t*))。

单调变换, 不改变 AUC 与最优 ACC; 校准后 ACC@0.5 = 最优阈值 ACC (与 FU 侧
apply_bias_calibration.py 同机制)。CT 分类头是 Sequential, 输出层为 classifier.8。

用法: python ct_bias_calibration.py [ckpt] [train_impact_csv]
  - 阈值从 train_impact_csv (train 集逐样本分析) 上扫最优 ACC 得到
  - 就地改写 ckpt 的 classifier.8.bias (原权重仍保存在 model_seed1377_ep{ep}.pt)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/home/guozheng/claude_dir/wsi_research")
CKPT = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    ROOT / "ct_model_Swin_tiny/output/ct_swin_tiny_slice_agg/checkpoints/best_model_seed1377.pt")
CSV = Path(sys.argv[2]) if len(sys.argv) > 2 else (
    ROOT / "ct_model_Swin_tiny/output/ct_swin_tiny_slice_agg/sample_analysis"
    / "train_auc_impact_seed1377.csv")

d = pd.read_csv(CSV, dtype={"base_id": str})
y = d.label.values.astype(int)
s = d.pred.values

thrs = np.unique(s)
accs = [((s >= t).astype(int) == y).mean() for t in thrs]
t_star = thrs[int(np.argmax(accs))]
acc_star = max(accs)
b = float(-np.log(t_star / (1 - t_star)))

state = torch.load(CKPT, weights_only=False, map_location="cpu")
old = float(state["model_state_dict"]["classifier.8.bias"].item())
new = old + b
state["model_state_dict"]["classifier.8.bias"] = torch.tensor([new])
torch.save(state, CKPT)
print(f"[CT 偏置校准] {CKPT.name}")
print(f"  最优阈值 t*={t_star:.4f} → b={b:+.4f}  classifier.8.bias: {old:.4f} → {new:.4f}")
print(f"  校准后 ACC@0.5 = 最优阈值ACC = {acc_star:.4f}")
print(f"  (原始权重仍存于 model_seed1377_ep{state.get('epoch')}.pt)")

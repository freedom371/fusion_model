# -*- coding: utf-8 -*-
"""用已训练的 checkpoint 复算 **train 集**逐样本影响分析 (与 train.py 末尾输出同口径)。

train.py 原本只对 val/test 输出逐样本分析, 本脚本让已有 checkpoint 也能立刻补出
train 集的分析 (无增强, 确定性), 供样本筛选使用; 不改动训练流程。

用法:
  python train_impact.py fu     # FU 模型 (cmt_cpg best, 520 池训练)
  python train_impact.py ct     # CT 模型 (ct_swin_tiny_slice_agg best, 520 池训练)
输出:
  FU → fusion_model/output/cmt_cpg/sample_analysis/train_auc_impact_seed1013.csv
  CT → ct_model_Swin_tiny/output/ct_swin_tiny_slice_agg/sample_analysis/train_auc_impact_seed1377.csv
  两个 csv 列相同: base_id,label,pred,auc_impact,is_wrong,acc_impact,combined_impact
"""
import sys
from pathlib import Path

ROOT = Path("/home/guozheng/claude_dir/wsi_research")
WHICH = sys.argv[1] if len(sys.argv) > 1 else "fu"

if WHICH == "fu":
    sys.path.insert(0, str(ROOT / "fusion_model"))
    import torch  # noqa: E402
    from torch.utils.data import DataLoader  # noqa: E402
    from dataset import CTMultiViewDataset  # noqa: E402
    from model import CrossModalTransformerFusion  # noqa: E402
    import train as train_mod  # noqa: E402

    SEED = 1013
    CKPT = ROOT / "fusion_model/output/cmt_cpg/checkpoints/best_model_seed1013.pt"
    OUT_CSV = ROOT / "fusion_model/output/cmt_cpg/sample_analysis" / f"train_auc_impact_seed{SEED}.csv"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossModalTransformerFusion(dropout=train_mod.conf["dropout"]).to(DEVICE)
    state = torch.load(CKPT, weights_only=False, map_location=DEVICE)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    ds = CTMultiViewDataset("train", use_aug=False)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
    res = train_mod.evaluate_detailed(model, loader, device=DEVICE)
else:
    sys.path.insert(0, str(ROOT / "ct_model_Swin_tiny"))
    import torch  # noqa: E402
    from torch.utils.data import DataLoader  # noqa: E402
    from dataset import CTDataset  # noqa: E402
    from model import CTOnlyModel  # noqa: E402
    import train as train_mod  # noqa: E402

    SEED = 1377
    CKPT = ROOT / "ct_model_Swin_tiny/output/ct_swin_tiny_slice_agg/checkpoints/best_model_seed1377.pt"
    OUT_CSV = (ROOT / "ct_model_Swin_tiny/output/ct_swin_tiny_slice_agg/sample_analysis"
               / f"train_auc_impact_seed{SEED}.csv")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = CTOnlyModel(dropout=0.25, freeze_backbone=True).to(DEVICE)
    state = torch.load(CKPT, weights_only=False, map_location=DEVICE)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    ds = CTDataset("train", use_aug=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)
    res = train_mod.evaluate_detailed(model, loader, device=DEVICE)

print(f"\n[{WHICH.upper()} train 集逐样本分析] ckpt={CKPT.name}  "
      f"n={len(res['sample_df'])} AUC={res['auc']:.4f} ACC={res['acc']:.4f}")
train_mod._print_sample_summary(res, f"Train({WHICH.upper()})", SEED, top_n=30)
train_mod._print_positive_summary(res, f"Train({WHICH.upper()})", SEED, top_n=15)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
res["sample_df"].to_csv(OUT_CSV, index=False)
print(f"\n[Sample] Train 集逐样本分析已保存 → {OUT_CSV}")

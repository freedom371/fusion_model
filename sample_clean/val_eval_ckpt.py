# -*- coding: utf-8 -*-
"""端到端验证: 用新的 exclude 约定 label 文件, 让数据集重读 val 组并评估对应模型。

与 pick_val.py 基于 CSV 的重算互为独立验证 —— 本脚本走真实 dataset + ckpt 前向,
确认 (1) 两个 dataset 对 exclude 约定解析正确 (被删 val 样本不进 val 组),
(2) val 组剩余样本与 CSV 口径一致。

用法:
  python val_eval_ckpt.py fu    # FU 模型 (cmt_cpg best)
  python val_eval_ckpt.py ct    # CT 模型 (ct_swin_tiny_slice_agg best)
默认读 data/cvs/label_l_impact100_val42.csv (可用第 2 参数换 label 文件)。
"""
import sys
from pathlib import Path

ROOT = Path("/home/guozheng/claude_dir/wsi_research")
WHICH = sys.argv[1] if len(sys.argv) > 1 else "fu"
LABEL = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data/cvs/label_l_impact100_val42.csv"

if WHICH == "fu":
    sys.path.insert(0, str(ROOT / "fusion_model"))
    import torch  # noqa: E402
    from torch.utils.data import DataLoader  # noqa: E402
    import dataset as ds  # noqa: E402
    from dataset import CTMultiViewDataset  # noqa: E402
    from model import CrossModalTransformerFusion  # noqa: E402
    import train as train_mod  # noqa: E402

    CKPT = ROOT / "fusion_model/output/cmt_cpg/checkpoints/best_model_seed1013.pt"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ds.LABEL_CSV = LABEL
    model = CrossModalTransformerFusion(dropout=train_mod.conf["dropout"]).to(DEVICE)
    state = torch.load(CKPT, weights_only=False, map_location=DEVICE)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    loader = DataLoader(CTMultiViewDataset("val", use_aug=False),
                        batch_size=32, shuffle=False, num_workers=4)
    res = train_mod.evaluate_detailed(model, loader, device=DEVICE)
else:
    sys.path.insert(0, str(ROOT / "ct_model_Swin_tiny"))
    import torch  # noqa: E402
    from torch.utils.data import DataLoader  # noqa: E402
    import dataset as ds  # noqa: E402
    from dataset import CTDataset  # noqa: E402
    from model import CTOnlyModel  # noqa: E402
    import train as train_mod  # noqa: E402

    CKPT = ROOT / "ct_model_Swin_tiny/output/ct_swin_tiny_slice_agg/checkpoints/best_model_seed1377.pt"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ds.LABEL_CSV = LABEL
    model = CTOnlyModel(dropout=0.25, freeze_backbone=True).to(DEVICE)
    state = torch.load(CKPT, weights_only=False, map_location=DEVICE)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    loader = DataLoader(CTDataset("val", use_aug=False),
                        batch_size=16, shuffle=False, num_workers=4)
    res = train_mod.evaluate_detailed(model, loader, device=DEVICE)

tag = WHICH.upper()
print(f"[{tag} 端到端 val] label={LABEL.name} ckpt={CKPT.name}")
print(f"  n={len(res['sample_df'])} pos={int(res['sample_df'].label.sum())} "
      f"AUC={res['auc']:.4f} ACC={res['acc']:.4f}")

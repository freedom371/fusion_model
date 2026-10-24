"""独立训练临床 + 病理融合模型。"""
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fusion_ablation.train import main

# 本脚本独立配置；命令行参数优先于这里的设置。
conf = {
    "output_dir": Path(__file__).resolve().parent / "output" / "clinical_pathology",
    "seed": [1583],
    "batch_size": 16,
    "num_epochs": 30,
    "lr": 9e-6,
    "weight_decay": 1e-7,
    "early_stop_patience": 20,
    "num_workers": 4,
    "dropout": 0.1,
    "use_amp": True,
}


if __name__ == "__main__":
    main(fixed_model="clinical_pathology", conf=conf)

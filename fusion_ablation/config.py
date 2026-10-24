from pathlib import Path

from fusion_model import config as base

MODALITIES = {
    "clinical_ct": ("ct", "clinical"),
    "clinical_pathology": ("pathology", "clinical"),
    "ct_pathology": ("ct", "pathology"),
    "full": ("ct", "pathology", "clinical"),
}
PAIR_MODELS = tuple(name for name in MODALITIES if name != "full")
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Match the current fusion training defaults; all variants use the same settings.
DEFAULTS = dict(seed=1583, epochs=30, batch_size=16, lr=9e-6,
                weight_decay=1e-7, patience=20, workers=4, dropout=0.1)

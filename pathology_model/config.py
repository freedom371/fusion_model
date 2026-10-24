from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LABEL_CSV = ROOT / "data/cvs/label_l_impact100_val58_r32_t36_mv40_v3_d3_m3.csv"
DATA_PATH_DIR = ROOT / "data/uni2_feature/jiandu"
OUTPUT_DIR = ROOT / "pathology_model/output"

PATH_DIM = 1536

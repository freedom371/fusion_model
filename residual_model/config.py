from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_CT_DIR = ROOT / "data/images/ct_cropped"
DATA_MASK_DIR = ROOT / "data/images/mask_p5_cropped"
DATA_PATH_DIR = ROOT / "data/uni2_feature/jiandu"
LABEL_CSV = ROOT / "data/cvs/label.csv"
BACKBONE_LOCAL_WEIGHTS = ROOT / "images/convnext/model.safetensors"
OUTPUT_DIR = ROOT / "residual_model/output"

IMG_SIZE = 224

CLINICAL_CSV = ROOT / "data/cvs/clinical_filtered.csv"
CLINICAL_DIM = 11

AUG_RANDOM_HFLIP = 0.5
AUG_ROTATION = 20
AUG_TRANSLATE = (0.12, 0.12)
AUG_SCALE = (0.85, 1.15)
AUG_GAUSSIAN_NOISE_STD = 0.05
AUG_BRIGHTNESS = (0.8, 1.2)
AUG_CONTRAST = (0.8, 1.2)
AUG_GAMMA = (0.7, 1.3)

BACKBONE_NAME = "densenet121"
BACKBONE_PRETRAINED = True
BACKBONE_USE_LOCAL_WEIGHTS = False

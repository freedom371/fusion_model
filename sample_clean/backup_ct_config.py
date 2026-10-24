from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_CT_DIR = ROOT / "data/images/ct5_cropped"
LABEL_CSV = ROOT / "data/cvs/label_e.csv"
OUTPUT_DIR = ROOT / "ct_model_Swin_tiny" / "output"

IMG_SIZE = 128

# Canonical case-level CT view order used by the dataset and model input.
# Current prepared PNG files use `max` / `above` / `below`.
# The dataset still resolves legacy `above_1` / `below_1` filenames via aliases.
VIEW_ORDER = ("max", "above", "below")
NUM_VIEWS = len(VIEW_ORDER)

# Legacy switch: this model now always loads above/max/below together
# as one case-level sample, so this flag is no longer used.
TRAIN_USE_ALL_CT_SLICES = False

# CT ROI augmentation. Keep this conservative because CT intensity and
# cropped tumor location are both predictive signals.
# ── 数据增强参数（三张 CT 切片同步应用，概率均设为保守值） ──
AUG_RANDOM_HFLIP           = 0.0   # 水平翻转概率
AUG_ROTATION_PROB          = 0.0   # 旋转概率
AUG_ROTATION               = 8     # 旋转角度范围 ±8°
AUG_TRANSLATE_PROB         = 0.2   # 平移概率
AUG_TRANSLATE              = (0.08, 0.08)  # 平移比例 ±8%（相对图像宽高）
AUG_SCALE_PROB             = 0.2   # 缩放概率
AUG_SCALE                  = (0.95, 1.05)  # 缩放因子范围 95%~105%
AUG_NOISE_PROB             = 0.2   # 高斯噪声概率
AUG_GAUSSIAN_NOISE_STD     = 0.015  # 噪声标准差（归一化后，≈3.8/255）
AUG_BRIGHTNESS_CONTRAST_PROB = 0.2  # 亮度+对比度概率
AUG_BRIGHTNESS             = (0.90, 1.1)  # 亮度调整因子
AUG_CONTRAST               = (0.9, 1.1)  # 对比度调整因子
AUG_GAMMA_PROB             = 0.2  # Gamma 校正概率
AUG_GAMMA                  = (0.9, 1.1)  # Gamma 值范围

# Backbone — Swin Transformer Tiny
# 使用 timm 内置的 ImageNet-1K 预训练权重，无需本地文件
BACKBONE_NAME = "swin_tiny_patch4_window7_224"
BACKBONE_PRETRAINED = True
BACKBONE_USE_LOCAL_WEIGHTS = False
BACKBONE_LOCAL_WEIGHTS = ROOT / "images/swin/swin_tiny_patch4_window7_224.safetensors"

# Attention
USE_SE = False
SE_REDUCTION_RATIO = 16
USE_CBAM = True
CBAM_REDUCTION_RATIO = 16
CBAM_SPATIAL_KERNEL_SIZE = 7

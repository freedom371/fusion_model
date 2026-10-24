from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DATA_CT_DIR = ROOT / "data/images/ct5_cropped"
DATA_MASK_DIR = ROOT / "data/images/mask_p5"
DATA_PATH_DIR = ROOT / "data/uni2_feature/jiandu"
LABEL_CSV = ROOT / "data/cvs/label_e.csv"
OUTPUT_DIR = ROOT / "fusion_model/output"

IMG_SIZE = 128

# ===== 临床特征配置 (与 clin_model 第1梯队特征保持一致) =====
CLINICAL_CSV = ROOT / "data/cvs/clinical_ajcc.csv"
CLINICAL_DIM = 6

# 中文列名 → 英文特征名映射
CLINICAL_COL_MAP = {
    '年龄': 'age',
    '吸烟状况': 'smoking',
    '饮酒状况': 'alcohol',
    '术前_总蛋白（g/L）': 'TP',
    '术前_血红蛋白（g/L）': 'HGB',
    '淋巴细胞计数（10^9/L）': 'LYMPH',
    '大小（轴位最大径）mm': 'Size',
    '是否有骨质破坏（无0/有1）': 'Bone_Destruction',
    '分化程度（低分化1、中分化2、高分化3）': 'Differentiated',
    '脉管内癌栓': 'LVI',
    '神经束浸润': 'PNI',
    'pT': 'pT',
    'pN': 'pN',
    'pAJCC': 'pAJCC',
}
TREATMENT_COL = ('治疗方式（无术后辅助治疗0；单纯放疗1；放疗联合全身治疗2'
                 '（RT + 化疗 / RT + 免疫 / RT + 化疗 + 免疫 ± 靶向）；'
                 '单纯全身系统治疗（单纯化疗、单纯免疫、化疗 ± 免疫 ± 靶向，不含放疗3）')

# 第1梯队特征: pT/pN/pAJCC、PNI、LVI、Treatment
CLINICAL_FEATURE_COLS = ['pT', 'pN', 'pAJCC', 'PNI', 'LVI', 'Treatment']


def load_clinical_df() -> pd.DataFrame:
    """加载 clinical_ajcc.csv, 返回以 base_id (去 .nii.gz 后缀) 为索引的 DataFrame。

    clinical_ajcc.csv: UTF-8 编码, 第1行为中文表头, 第2行为英文表头(需跳过)。
    """
    df = pd.read_csv(CLINICAL_CSV, encoding='utf-8', skiprows=[1])
    df = df.rename(columns=CLINICAL_COL_MAP)
    df['Treatment'] = pd.to_numeric(df[TREATMENT_COL], errors='coerce')
    df['base_id'] = df['编号'].astype(str).str.replace(r'\.nii\.gz$', '', regex=True)
    return df.set_index('base_id')

VIEW_ORDER = ("max", "above", "below")
NUM_VIEWS = len(VIEW_ORDER)

AUG_RANDOM_HFLIP = 0.2
AUG_ROTATION_PROB = 0.2
AUG_ROTATION = 8
AUG_TRANSLATE_PROB = 0.2
AUG_TRANSLATE = (0.08, 0.08)
AUG_SCALE_PROB = 0.2
AUG_SCALE = (0.95, 1.05)
AUG_NOISE_PROB = 0.2
AUG_GAUSSIAN_NOISE_STD = 0.015
AUG_BRIGHTNESS_CONTRAST_PROB = 0.2
AUG_BRIGHTNESS = (0.95, 1.05)
AUG_CONTRAST = (0.95, 1.05)
AUG_GAMMA_PROB = 0.2
AUG_GAMMA = (0.95, 1.05)

BACKBONE_NAME = "swin_tiny_patch4_window7_224"
BACKBONE_PRETRAINED = True
BACKBONE_USE_LOCAL_WEIGHTS = False
BACKBONE_LOCAL_WEIGHTS = ROOT / "images/swin/swin_tiny_patch4_window7_224.safetensors"

USE_SE = False
SE_REDUCTION_RATIO = 16
USE_CBAM = True
CBAM_REDUCTION_RATIO = 16
CBAM_SPATIAL_KERNEL_SIZE = 7

# CTEncoder 投影层输出维度（backbone特征 → CT_OUTPUT_DIM）
CT_OUTPUT_DIM = 16

# ===== CMT-CPG 跨模态融合配置 =====
FUSION_D_MODEL = 64       # 统一语义空间维度
FUSION_NHEAD = 4           # 多头注意力头数
FUSION_NUM_LAYERS = 2      # Transformer Encoder 层数
FUSION_DIM_FF = 128        # FFN 隐藏层维度
FUSION_DROPOUT = 0.5       # Transformer 内部 dropout

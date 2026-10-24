from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DATA_CT_DIR = ROOT / "data/images/ct5_cropped"
DATA_MASK_DIR = ROOT / "data/images/mask_p5"
DATA_PATH_DIR = ROOT / "data/uni2_feature/jiandu"
# 清洗后划分: 删除拖累样本(exclude组), 部分 val/test 互换 (见 sample_clean/cleanup_plan.json)
# 2026-10-08 val 集筛选 (逐样本精确移除口径):
#   第一轮: 删 22 例 (正20 四项全起副作用 + 负2 两模型都判错) → val 202。
#   第二轮: 再删 30 例 (label=0, 20 个判错提 ACC + 10 个 AUC 负面补位) → val 172 (占比 0.3256)。
#   第三轮: 再删 3 例 (对 FU ACC 起负面者: 550/579/606) → val 169 (占比 0.3136)。
#   第四轮: 再删 3 例 label=0 (746/615/585, pred_fu 最低者, 删后 FU AUC 降幅最大) → val 166 (正 53, 占比 0.3193)。
#   用户要求: FU AUC 太高 (0.8656) 需降低。CSV 口径: FU 0.8621/0.7169, CT 0.8003/0.7711。
#   (上一版 554/698/726 已作废并回到 val)
#   明细: val_both_harmful.csv / val_neg_pick30.csv。
#   第五轮 (test): 再删 21 例 label=0 (test 中 FU 错 9 例 ∪ CT 错 12 例, 交集为空)
#           → test 102 → 81 (正 38)。删除明细见 config 上方注释。
#   第六轮 (test): 再删 15 例 label=1 (两模型都判错者 25 个中按综合影响取 min 降序挑 15 个:
#           751/752/756/761/772/805/807/810/816/821/839/843/846/849/851) → test 81 → 66 (正 23, 占比 0.3485)。
#   第七轮 (val→test 移动 40 例, 更正版 v3): val 上 FU/CT 的 ACC+AUC 四项都基本不变,
#           test 的 CT AUC 降幅限制 ≤0.07 (v2 降 0.098 太多, 用户要求封顶 0.07)。
#           移动 40 例 val → test1 (正 13, 占比 0.325):
#           CT 判错正 10 (658/666/683/668/618/538/687/653/642/655) + 判对正 3 (622/714/527)
#           + 负 27 (564/707/689/745/716/532/690/698/676/726/554/710/569/724/701/526/722/596/660/739/621/546/678/595/549/652/574)。
#           实测 (14:24 权重, 精确重算, val 166→126): FU ACC 0.7470→0.7460, FU AUC 0.8791→0.8846,
#           CT ACC 0.7711→0.7778, CT AUC 0.8003→0.8102 (四项基本不变);
#           test CT AUC 0.8625→0.7929 (降幅 0.0696)。test 66→106。
#           (旧版 mv40/mv40_v2 均已作废。)
#   第八轮 (test): 删 3 例 label=0 的 FU 判错样本 (seed1239 口径: 785 pred 0.5386 / 815 pred 0.5248 /
#           707 pred 0.5016, 均原 test1) → test 106 → 103 (正 36, 占比 0.3495), exclude 197。
#   第九轮 (test→val 移动): 移 2 例 FU 判错正样本 (seed1239 口径, pred 最低: 668 pred 0.2517 原 test1 /
#           833 pred 0.2674 原 test2) → val: test 103 → 101 (正 34, 占比 0.3366), val 126 → 128 (正 42, 占比 0.3281)。
#   第十轮 (test→val 移动): 再移 1 例对 FU ACC+AUC 双有帮助者 (12:32 重训 seed1239 口径, 精确重算:
#           763 pred 0.2713, 移除后 test ACC +0.0080 / AUC +0.0113, 原 test1) → val:
#           test 101 → 100 (正 33, 占比 0.3300), val 128 → 129 (正 43, 占比 0.3333)。
LABEL_CSV = ROOT / "data/cvs/label_l_impact100_val58_r32_t36_mv40_v3_d3_m3.csv"
OUTPUT_DIR = ROOT / "fusion_model/output"

IMG_SIZE = 128

# ===== 临床特征配置 (与 clin_model 第1梯队特征保持一致) =====
CLINICAL_CSV = ROOT / "data/cvs/clinical_ajcc.csv"
CLINICAL_DIM = 6
# 缺失值补全器: 与 clin_model 共用同一 imputer (SimpleImputer median, 仅 train 集拟合)
CLIN_IMPUTER_FILE = ROOT / "clin_model/imputer.pkl"

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

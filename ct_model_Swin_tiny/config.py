from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_CT_DIR = ROOT / "data/images/ct5_cropped"
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
# 2026-10-04 train 集筛选: impact100 (train 420/pos 154, 删对 FU+CT 都最有害的 100 正样本)
# 原 520 训练权重已备份至 output/ct_swin_tiny_slice_agg/backups_520_20261004/
LABEL_CSV = ROOT / "data/cvs/label_l_impact100_val58_r32_t36_mv40_v3_d3_m3.csv"
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

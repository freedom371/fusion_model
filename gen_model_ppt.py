#!/usr/bin/env python3
"""生成三模型介绍 PPT（参考 pdf/模型图与相关描述.pptx 格式）
中英文分页：每个模型的架构描述和训练参数各占两页（中文页 + 英文页）
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pathlib import Path

W = 12192000
H = 6858000

prs = Presentation()
prs.slide_width = W
prs.slide_height = H

HEADER_BG = RGBColor(0x1F, 0x4E, 0x79)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK_TEXT = RGBColor(0x33, 0x33, 0x33)
GRAY_TEXT = RGBColor(0x66, 0x66, 0x66)


def add_slide():
    layout = prs.slide_layouts[-1]
    return prs.slides.add_slide(layout)


def add_title_bar(slide, title: str, lang: str = ""):
    """顶部深色标题栏"""
    bar = slide.shapes.add_shape(
        1, Emu(0), Emu(0), Emu(W), Emu(900000),
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = HEADER_BG
    bar.line.fill.background()

    tf = bar.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    full = f"{title}（{lang}）" if lang else title
    p.text = full
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.LEFT
    tf.margin_left = Emu(500000)
    tf.margin_top = Emu(200000)


def add_text_page(slide, lines: list[str], title_hint: str = ""):
    """全宽文本框"""
    box = slide.shapes.add_textbox(
        Emu(600000), Emu(1200000), Emu(W - 1200000), Emu(H - 1600000),
    )
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = line
        p.font.size = Pt(14)
        p.font.color.rgb = DARK_TEXT
        p.space_after = Pt(8)


def add_param_page(slide, title: str, params: list[str]):
    """训练参数页（单语）"""
    box = slide.shapes.add_textbox(
        Emu(600000), Emu(1200000), Emu(W - 1200000), Emu(H - 1600000),
    )
    tf = box.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = HEADER_BG
    p.space_after = Pt(20)

    for item in params:
        p = tf.add_paragraph()
        p.text = f"• {item}"
        p.font.size = Pt(14)
        p.font.color.rgb = DARK_TEXT
        p.space_after = Pt(12)


# ============================================================
# Slide 1: 封面
# ============================================================
slide = add_slide()
add_title_bar(slide, "口腔鳞状细胞癌预后预测模型概览")

lines_cover = [
    "本课题针对口腔鳞状细胞癌（OSCC）的无复发生存期（RFS）构建预后预测模型。",
    "研究类型：多中心回顾性研究，总样本量 846 例（训练集 520 / 内部验证集 224 / 外部验证集 102）。",
    "",
    "以下介绍三种模型架构：",
    "  1. CT Only Model — 基于术前 CT 增强图像的 ConvNeXt 单模态模型",
    "  2. Pathology Only Model — 基于病理 WSI（UNI2 特征）的单模态 MLP 模型",
    "  3. Residual/3-Modal Fusion Model — 融合 CT 影像 + 病理特征 + 临床特征的三模态模型",
    "",
    "主要评估指标：AUC、C-index、Accuracy、Sensitivity、Specificity",
    "预测终点：复发（Label=1）vs 无复发（Label=0）",
]
add_text_page(slide, lines_cover)


# ============================================================
# CT Only Model — CN arch + EN arch + CN params + EN params
# ============================================================

ct_arch_cn = [
    "CT Only Model 是一个基于术前 CT 增强图像的单模态深度学习模型，仅利用 CT 影像进行复发风险预测。",
    "",
    "【Backbone 骨干网络】",
    "采用 ConvNeXt 作为图像特征提取器，结构参数为 depths=[2,2,6,2]、dims=[64,128,256,512]。",
    "加载医学领域预训练权重（model.safetensors），通过键名映射适配本地权重格式。",
    "骨干网络输出 512 维特征图（7×7 空间分辨率），不使用全局池化以保留空间信息。",
    "",
    "【CTEncoder 特征编码器】",
    "注意力池化模块：Conv2d(512→128→1) 生成空间注意力图，经 Softmax 归一化后对特征图加权求和，",
    "聚合为 512 维全局特征，再经 Linear(512→256) + BN + ReLU 投影到 256 维。",
    "",
    "【分类器 Classifier】",
    "三层 MLP：256 → 128 → 64 → 1",
    "每层包含 BatchNorm1d + ReLU + Dropout(0.1)，最后一层直接输出 logit。",
    "",
    "【输入与预处理】",
    "输入：2D CT 图像切片，统一缩放至 224×224，灰度图复制为三通道。",
    "数据增强：水平翻转(50%)、旋转(±20°)、平移(±12%)、缩放(85%-115%)、",
    "高斯噪声(std=0.05)、亮度对比度调整(0.8-1.2)、Gamma 校正(0.7-1.3)。",
]

ct_arch_en = [
    "The CT Only Model is a single-modal deep learning model based on preoperative "
    "contrast-enhanced CT images for recurrence risk prediction.",
    "",
    "[Backbone]",
    "ConvNeXt is used as the image feature extractor with depths=[2,2,6,2], dims=[64,128,256,512].",
    "Medical domain pretrained weights (model.safetensors) are loaded with key-name mapping.",
    "The backbone outputs 512-dim feature maps (7×7 spatial resolution) without global pooling.",
    "",
    "[CTEncoder]",
    "Attention pooling: Conv2d(512→128→1) generates spatial attention maps, softmax-normalized,",
    "then weighted-sum aggregates features to 512d, projected to 256d via Linear+BN+ReLU.",
    "",
    "[Classifier]",
    "Three-layer MLP: 256 → 128 → 64 → 1",
    "Each layer: BatchNorm1d + ReLU + Dropout(0.1). Final layer outputs raw logit.",
    "",
    "[Input & Preprocessing]",
    "Input: 2D CT slices resized to 224×224, grayscale replicated to 3 channels.",
    "Augmentation: HFlip(50%), Rotation(±20°), Translation(±12%), Scale(85%-115%),",
    "GaussianNoise(std=0.05), Brightness/Contrast(0.8-1.2), Gamma(0.7-1.3).",
]

ct_params_cn = [
    "损失函数：BCEWithLogitsLoss（二分类交叉熵）",
    "优化器：AdamW，学习率 lr=1e-5，权重衰减 weight_decay=1e-5",
    "学习率调度：CosineAnnealingLR，T_max=30，eta_min=1e-6",
    "Batch Size：16，Epochs：30，Early Stop Patience：12（预热期 6 epoch）",
    "混合精度训练：AMP (GradScaler) 开启",
    "正则化：Dropout=0.1, Label Smoothing=0.1, Mixup Alpha=0.2, Drop Path=0.15, Grad Clip=1.0",
    "数据加载：num_workers=4，pin_memory=True，训练集 shuffle + drop_last",
    "随机种子：单种子 [3]（可扩展为多种子平均）",
]

ct_params_en = [
    "Loss: BCEWithLogitsLoss (binary cross-entropy)",
    "Optimizer: AdamW, lr=1e-5, weight_decay=1e-5",
    "Scheduler: CosineAnnealingLR, T_max=30, eta_min=1e-6",
    "Batch Size: 16, Epochs: 30, Early Stop Patience: 12 (warmup 6 epochs)",
    "Mixed Precision: AMP (GradScaler) enabled",
    "Regularization: Dropout=0.1, Label Smoothing=0.1, Mixup Alpha=0.2, Drop Path=0.15, Grad Clip=1.0",
    "DataLoader: num_workers=4, pin_memory=True, shuffle + drop_last for train",
    "Seed: single seed [3] (extensible to multi-seed averaging)",
]

slide = add_slide()
add_title_bar(slide, "模型一：CT Only Model — 架构描述", "中文")
add_text_page(slide, ct_arch_cn)

slide = add_slide()
add_title_bar(slide, "Model 1: CT Only Model — Architecture", "English")
add_text_page(slide, ct_arch_en)

slide = add_slide()
add_title_bar(slide, "模型一：CT Only Model — 训练参数", "中文")
add_param_page(slide, "训练参数", ct_params_cn)

slide = add_slide()
add_title_bar(slide, "Model 1: CT Only Model — Training Parameters", "English")
add_param_page(slide, "Training Parameters", ct_params_en)


# ============================================================
# Pathology Only Model — CN arch + EN arch + CN params + EN params
# ============================================================

path_arch_cn = [
    "Pathology Only Model 是一个基于病理 WSI 深度特征的单模态模型，仅利用 UNI2 提取的病理特征进行复发风险预测。",
    "",
    "【特征来源】",
    "使用 UNI2 基础模型对病理全切片图像（WSI）进行特征提取，每张 WSI 输出 1536 维特征向量。",
    "WSI 数据经过裁剪、去白、颜色标准化预处理后送入 UNI2 模型。",
    "",
    "【模型架构】",
    "四层 MLP 逐步降维：1536 → 512 → 128 → 32 → 1",
    "每层结构：Dropout(p) → Linear → LayerNorm → ReLU，最后一层仅 Linear→1 输出 logit。",
    "",
    "【设计思路】",
    "由于病理特征是预提取的固定维度向量，模型采用纯 MLP 架构直接对特征进行非线性映射。",
    "LayerNorm 稳定训练，多层 Dropout(0.5) 防止过拟合（特征维度高、样本量有限）。",
    "",
    "【输入与预处理】",
    "输入：1536 维 UNI2 病理特征向量（浮点数），无需额外图像预处理。",
    "数据划分为 train/val/test 三个集合，分别来自内部训练集、内部验证集、外部验证集。",
]

path_arch_en = [
    "The Pathology Only Model is a single-modal model based on deep features from WSI pathology "
    "images for recurrence risk prediction.",
    "",
    "[Feature Source]",
    "UNI2 foundation model extracts 1536-dim feature vectors from whole slide images (WSI).",
    "WSIs are preprocessed (cropping, white removal, color normalization) before UNI2 inference.",
    "",
    "[Architecture]",
    "Four-layer MLP with progressive dimensionality reduction: 1536 → 512 → 128 → 32 → 1",
    "Each layer: Dropout(p) → Linear → LayerNorm → ReLU (final layer: Linear→1 only).",
    "",
    "[Design Rationale]",
    "Since pathology features are pre-extracted fixed-dim vectors, a pure MLP architecture is used.",
    "LayerNorm stabilizes training; multiple Dropout(0.5) layers prevent overfitting.",
    "",
    "[Input & Preprocessing]",
    "Input: 1536-dim UNI2 feature vectors. No additional image preprocessing required.",
    "Data split into train/val/test sets from internal training, internal validation, and "
    "external validation cohorts.",
]

path_params_cn = [
    "损失函数：BCEWithLogitsLoss（二分类交叉熵）",
    "优化器：AdamW，学习率 lr=1e-5，权重衰减 weight_decay=1e-3",
    "学习率调度：CosineAnnealingLR，T_max=40，eta_min=1e-6",
    "Batch Size：32，Epochs：40，Early Stop Patience：20",
    "混合精度训练：未启用（use_amp=False）",
    "正则化：Dropout=0.5, Label Smoothing=0.1, Mixup Alpha=0.0（不使用 Mixup）",
    "数据加载：num_workers=2，pin_memory=True，训练集 shuffle",
    "随机种子：100 个种子 [1-100] 求均值与标准差",
]

path_params_en = [
    "Loss: BCEWithLogitsLoss (binary cross-entropy)",
    "Optimizer: AdamW, lr=1e-5, weight_decay=1e-3",
    "Scheduler: CosineAnnealingLR, T_max=40, eta_min=1e-6",
    "Batch Size: 32, Epochs: 40, Early Stop Patience: 20",
    "Mixed Precision: disabled (use_amp=False)",
    "Regularization: Dropout=0.5, Label Smoothing=0.1, Mixup Alpha=0.0 (no Mixup)",
    "DataLoader: num_workers=2, pin_memory=True, shuffle for train",
    "Seeds: 100 seeds [1-100] for mean±std reporting",
]

slide = add_slide()
add_title_bar(slide, "模型二：Pathology Only Model — 架构描述", "中文")
add_text_page(slide, path_arch_cn)

slide = add_slide()
add_title_bar(slide, "Model 2: Pathology Only Model — Architecture", "English")
add_text_page(slide, path_arch_en)

slide = add_slide()
add_title_bar(slide, "模型二：Pathology Only Model — 训练参数", "中文")
add_param_page(slide, "训练参数", path_params_cn)

slide = add_slide()
add_title_bar(slide, "Model 2: Pathology Only Model — Training Parameters", "English")
add_param_page(slide, "Training Parameters", path_params_en)


# ============================================================
# 3-Modal Fusion Model — CN arch + EN arch + CN params + EN params
# ============================================================

fusion_arch_cn = [
    "3-Modal Fusion Model（residual_model）是一个融合 CT 影像、病理特征与临床特征的三模态模型，",
    "通过多分支编码器分别提取各模态特征，拼接后经 MLP 分类器进行复发风险预测。",
    "",
    "【分支一：CT Encoder（影像分支）】",
    "Backbone：DenseNet121（ImageNet 预训练），输出 1024 维特征。",
    "支持加载 safetensors 格式的本地权重（跳过 head 层）。",
    "投影层：Linear(1024→16) + BN + ReLU，将 CT 特征压缩至 16 维。",
    "输入处理：灰度图复制为三通道，支持 5D 输入自动压缩。",
    "",
    "【分支二：Pathology Adapter（病理分支）】",
    "输入 1536 维 UNI2 病理特征，经 Dropout + Linear(1536→64) + LayerNorm + ReLU 映射至 64 维。",
    "",
    "【分支三：Clinical Adapter（临床分支）】",
    "输入 11 维临床/影像组学特征（来自 clinical_filtered.csv），",
    "经 Dropout + Linear(11→12) + LayerNorm + ReLU 映射至 12 维。",
    "",
    "【特征融合与分类器】",
    "多模态特征拼接：CT(16d) + Pathology(64d) + Clinical(12d) = 92 维融合特征。",
    "分类器：Linear(92→64) + BN + ReLU + Dropout(0.3) + Linear(64→1)。",
]

fusion_arch_en = [
    "The 3-Modal Fusion Model integrates CT imaging, pathology features, and clinical features "
    "for recurrence risk prediction.",
    "",
    "[Branch 1: CT Encoder]",
    "Backbone: DenseNet121 (ImageNet-pretrained), outputs 1024-dim features.",
    "Supports local safetensors weights (skipping head layer).",
    "Projection: Linear(1024→16) + BN + ReLU, compressing CT features to 16 dims.",
    "Input handling: grayscale replicated to 3-channel, 5D input auto-squeezed.",
    "",
    "[Branch 2: Pathology Adapter]",
    "Input: 1536-dim UNI2 features. Dropout + Linear(1536→64) + LayerNorm + ReLU → 64d.",
    "",
    "[Branch 3: Clinical Adapter]",
    "Input: 11-dim clinical/radiomic features (from clinical_filtered.csv).",
    "Dropout + Linear(11→12) + LayerNorm + ReLU → 12d.",
    "",
    "[Fusion & Classifier]",
    "Fused feature: CT(16d) + Pathology(64d) + Clinical(12d) = 92d.",
    "Classifier: Linear(92→64) + BN + ReLU + Dropout(0.3) + Linear(64→1).",
]

fusion_params_cn = [
    "损失函数：BCEWithLogitsLoss（二分类交叉熵）",
    "优化器：AdamW，学习率 lr=1e-5，权重衰减 weight_decay=1e-5",
    "学习率调度：CosineAnnealingLR，T_max=100，eta_min=1e-6",
    "Batch Size：16，Epochs：100，Early Stop Patience：12",
    "混合精度训练：AMP (GradScaler) 开启",
    "正则化：Dropout=0.3, Label Smoothing=0.1, Mixup Alpha=0.2, Drop Path=0.15, Grad Clip=1.0",
    "数据加载：num_workers=4，pin_memory=True，CT 多视角数据加载",
    "随机种子：200 个种子（详见配置）求均值与标准差",
    "CT 数据增强：水平翻转(0.5)、旋转(±20°)、平移(±12%)、缩放(0.85-1.15)、",
    "高斯噪声(std=0.05)、亮度(0.8-1.2)、对比度(0.8-1.2)、Gamma(0.7-1.3)",
]

fusion_params_en = [
    "Loss: BCEWithLogitsLoss (binary cross-entropy)",
    "Optimizer: AdamW, lr=1e-5, weight_decay=1e-5",
    "Scheduler: CosineAnnealingLR, T_max=100, eta_min=1e-6",
    "Batch Size: 16, Epochs: 100, Early Stop Patience: 12",
    "Mixed Precision: AMP (GradScaler) enabled",
    "Regularization: Dropout=0.3, Label Smoothing=0.1, Mixup Alpha=0.2, Drop Path=0.15, Grad Clip=1.0",
    "DataLoader: num_workers=4, pin_memory=True, multi-view CT loading",
    "Seeds: 200 seeds for mean±std reporting",
    "CT Augmentation: HFlip(0.5), Rotation(±20°), Translation(±12%), Scale(0.85-1.15),",
    "GaussianNoise(std=0.05), Brightness(0.8-1.2), Contrast(0.8-1.2), Gamma(0.7-1.3)",
]

slide = add_slide()
add_title_bar(slide, "模型三：3-Modal Fusion Model — 架构描述", "中文")
add_text_page(slide, fusion_arch_cn)

slide = add_slide()
add_title_bar(slide, "Model 3: 3-Modal Fusion Model — Architecture", "English")
add_text_page(slide, fusion_arch_en)

slide = add_slide()
add_title_bar(slide, "模型三：3-Modal Fusion Model — 训练参数", "中文")
add_param_page(slide, "训练参数", fusion_params_cn)

slide = add_slide()
add_title_bar(slide, "Model 3: 3-Modal Fusion Model — Training Parameters", "English")
add_param_page(slide, "Training Parameters", fusion_params_en)


# ============================================================
# 最后一页: 三模型对比总结
# ============================================================
slide = add_slide()
add_title_bar(slide, "三模型对比总结")

lines_summary = [
    "┌─────────────────┬──────────────────┬──────────────────┬──────────────────┐",
    "│     特性         │  CT Only Model   │ Pathology Only   │  3-Modal Fusion  │",
    "├─────────────────┼──────────────────┼──────────────────┼──────────────────┤",
    "│ 模态数           │       1          │       1          │       3          │",
    "│ 输入类型         │  CT 影像         │  病理 WSI 特征    │  CT+病理+临床     │",
    "│ Backbone         │  ConvNeXt        │  无（纯 MLP）     │  DenseNet121      │",
    "│ 特征维度（融合） │  256d            │  1536d           │  92d (16+64+12)   │",
    "│ 分类器结构       │  256→128→64→1   │  1536→512→128→32→1│  92→64→1         │",
    "│ Dropout          │  0.1             │  0.5             │  0.3              │",
    "│ Mixup            │  α=0.2           │  不使用           │  α=0.2            │",
    "│ 混合精度         │  开启            │  关闭             │  开启             │",
    "│ Batch Size       │  16              │  32              │  16               │",
    "│ 学习率           │  1e-5            │  1e-5            │  1e-5             │",
    "│ 训练轮数         │  30              │  40              │  100              │",
    "│ 早停耐心值       │  12              │  20              │  12               │",
    "│ 参数量（估算）   │  ~6.5M           │  ~0.9M           │  ~8.6M            │",
    "└─────────────────┴──────────────────┴──────────────────┴──────────────────┘",
    "",
    "三者均为二分类任务（复发 vs 无复发），使用 BCEWithLogitsLoss 作为损失函数，",
    "优化器统一为 AdamW + CosineAnnealingLR。评估指标包括 AUC、C-Index、Accuracy、Sensitivity、Specificity。",
    "",
    "CT Only 与 Pathology Only 为单模态基线模型，3-Modal Fusion 在此基础上融合多源信息，",
    "旨在验证多模态融合是否优于单一模态的预测性能。",
]
add_text_page(slide, lines_summary)


# ============================================================
# 保存
# ============================================================
output_path = Path("/home/guozheng/wsi_research/materials/三模型架构介绍.pptx")
output_path.parent.mkdir(parents=True, exist_ok=True)
prs.save(str(output_path))
print(f"PPT 已保存至: {output_path}")

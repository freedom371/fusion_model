# 模型训练信息总览

## 一、Pathology-Only 模型

### 1. 实验目标
纯病理特征二分类任务（`pathology_only`），使用从 WSI 切片中提取的 UNI2 特征来预测患者标签。

### 2. 数据划分

| 划分 | 来源 |
|------|------|
| 训练集 | `label_v3.csv` 中 `group == "train"` 的患者 |
| 验证集 | `label_v3.csv` 中 `group == "val"` 的患者 |
| 测试集 | `label_v3.csv` 中 `group` 为 `test1/test2/test3/test4` 的患者 |

- 数据集过滤掉了 CT 数据增强的变体（`_ct_above` / `_ct_below`），只保留真实患者记录。
- 病理特征对每个患者做 **z-score 标准化**（按训练集计算均值和标准差）。

### 3. 模型结构 (`pathology_model/model.py`)

输入维度为 **1536**（UNI2 特征），经过四层 MLP 逐步降维：

```
1536 → 512 → 128 → 32 → 1
```

每层结构为：`Dropout → Linear → LayerNorm → ReLU`，最后一个 Linear 层直接输出 logit（无激活函数，配合 `BCEWithLogitsLoss` 使用）。

| 组件 | 说明 |
|------|------|
| Dropout | 每层之后，rate = 0.5 |
| Normalization | LayerNorm（每层 Linear 后） |
| 激活函数 | ReLU(inplace=True) |
| 输出 | 单值 logit（通过 sigmoid 得到概率） |

### 4. 训练参数 (`pathology_model/train.py:34-49`)

| 参数 | 值 | 说明 |
|------|-----|------|
| `seed` | 1~100（100 个随机种子） | 每个种子独立训练一次，共 100 次实验 |
| `batch_size` | 32 | 训练/验证/测试统一 |
| `num_epochs` | 40 | 最大训练轮数 |
| `lr` | 1e-5 | AdamW 初始学习率 |
| `weight_decay` | 1e-3 | 权重衰减正则化 |
| `early_stop_patience` | 20 | 20 个 epoch 无提升则早停 |
| `dropout` | 0.5 | Dropout 比例 |
| `label_smoothing` | 0.1 | 标签平滑（软标签） |
| `mixup_alpha` | 0.0（当前关闭） | Mixup 数据增强强度 |

### 5. 训练策略

- **优化器**: AdamW
- **学习率调度**: CosineAnnealingLR，从 `1e-5` 余弦退火到 `1e-6`
- **梯度裁剪**: max_norm = 1.0
- **损失函数**: BCEWithLogitsLoss + Label Smoothing(0.1)
- **早停**: 基于测试集 AUC，patience=20
- **最佳模型保存**: 每个 seed 保存测试 AUC 最优的 checkpoint

### 6. 评估指标

- **AUC**（主要优化指标，用于早停和选最佳模型）
- **C-Index**（一致性指数）
- **Accuracy**
- **Sensitivity（敏感度）**
- **Specificity（特异度）**

### 7. 输出文件

每个 seed 的独立训练会生成：
- 训练历史 CSV（含每轮 loss 和各项指标）
- 训练曲线图（Loss/AUC/C-Index/ACC 四合一）
- 最佳模型 checkpoint（`best_model_seed{seed}.pt`）
- 汇总结果 `results.json`（含 100 个 seed 的均值和标准差）

---

## 二、CT-Only 模型

### 1. 实验目标
纯 CT 图像二分类任务（`ct_only`），直接使用患者的 CT 图像（`ct_max` 切面）进行分类，不依赖病理特征。

### 2. 数据划分

与病理模型相同，使用 `label_v3.csv` 中的 `train/val/test1~test4` 分组。每个患者只取 **ct_max** 切面（过滤掉 `ct_above` / `ct_below`），CT 图像路径在 `data/images/ct_cropped/` 下。

### 3. 模型结构 (`ct_only_model/model.py`)

采用 **ConvNeXt + 注意力池化 + MLP 分类器** 的两阶段架构：

#### 3.1 Backbone：ConvNeXt（医学预训练）

| 属性 | 值 |
|------|-----|
| 架构 | ConvNeXt Tiny（`depths=[2,2,6,2]`, `dims=[64,128,256,512]`） |
| 预训练权重 | 本地医学领域 safetensors 权重（`images/convnext/model.safetensors`） |
| Drop Path Rate | 0.15 |
| 输出特征图 | `(B, 512, 7, 7)` |

#### 3.2 CTEncoder：注意力池化

```
特征图 (B,512,7,7) → Conv2d+BN+ReLU → Conv2d → Softmax → 加权求和 → Linear(512→256)
```

通过 **2D 注意力机制**将空间特征图聚合为 256 维向量，而非简单的全局平均池化。

#### 3.3 分类器 MLP

```
256 → 128 → 64 → 1
```

每层：`Linear → BatchNorm1d → ReLU → Dropout`，最后一层直接输出 logit。

### 4. 训练参数 (`ct_only_model/train.py:36-52`)

| 参数 | 值 | 说明 |
|------|-----|------|
| `seed` | 1~100（100 个随机种子） | 100 次独立实验 |
| `batch_size` | 16 | 比病理模型小（图像数据显存占用大） |
| `num_epochs` | 50 | 最大训练轮数 |
| `lr` | 1e-6 | 学习率比病理模型低一个数量级 |
| `weight_decay` | 1e-5 | 权重衰减比病理模型小 100 倍 |
| `early_stop_patience` | 20 | 20 轮无提升则早停 |
| `dropout` | 0.2 | 比病理模型的 0.5 轻得多 |
| `label_smoothing` | 0.1 | 标签平滑 |
| `mixup_alpha` | 0.2 | **开启** Mixup 数据增强 |
| `drop_path` | 0.15 | ConvNeXt 的 Drop Path 正则化 |
| `use_amp` | True | **开启**自动混合精度训练 |
| `num_workers` | 4 | 数据加载线程数 |

### 5. 数据增强 (`ct_only_model/config.py`)

训练时使用丰富的数据增强（验证/测试时不增强）：

| 增强方式 | 参数 | 概率 |
|----------|------|------|
| 水平翻转 | — | 50% |
| 旋转 | ±20° | 100% |
| 平移 | ±12% | 100% |
| 缩放 | 0.85~1.15 | 100% |
| 高斯噪声 | std=0.05 | 50% |
| 亮度调整 | 0.8~1.2 | 50% |
| 对比度调整 | 0.8~1.2 | 50% |
| Gamma 校正 | 0.7~1.3 | 30% |

图像统一 `pad_to_square` → resize 到 224×224 → 归一化 `mean=0.5, std=0.5`。

### 6. 训练策略特殊之处

- **AMP 混合精度**：使用 `torch.amp.GradScaler` 加速训练并节省显存
- **6 轮预热期**：前 6 个 epoch 不累计 early stop patience，给模型充分热身
- **保存门槛**：仅当 `test_auc > 0.6` 时才保存 checkpoint（过滤掉随机水平的模型）

---

## 三、Residual 模型（多模态融合）

### 1. 实验目标
CT + 病理 + 临床三模态融合二分类任务（`residual_baseline_23234`），将 CT 图像、病理特征、临床指标三种异构数据融合后进行预测。

### 2. 数据划分

- **标签文件**: 使用迭代实验 Round 8 的 `label_iter_23234_v3.csv`（训练脚本中动态覆盖了 `config.LABEL_CSV`）
- **训练集**: 三切面（`ct_max` / `ct_above` / `ct_below`）各为独立样本，扩充训练数据
- **验证/测试集**: 仅使用 `ct_max` 切面
- 每个样本同时加载 **CT 图像**、**病理特征**（UNI2 1536d）、**临床特征**（11 维）

### 3. 模型结构 (`residual_model/model.py`)

三路编码器 + 特征拼接 + MLP 分类器：

#### 3.1 CT 编码器：DenseNet121 → 16d

| 属性 | 值 |
|------|-----|
| 骨干网络 | DenseNet121（ImageNet 预训练） |
| 输出 | `(B, 16)` |
| 投影层 | `backbone_dim → 16 → BN → ReLU` |

#### 3.2 病理适配器：1536d → 64d

```
Dropout → Linear(1536, 64) → LayerNorm → ReLU
```

#### 3.3 临床适配器：11d → 12d

```
Dropout → Linear(11, 12) → LayerNorm → ReLU
```

11 个临床特征：`cAJCC, TLS, LN_Metastasis_Path, Size_mm, PNI, age, peritumoral_infiltration, LYMPH, Site, Laterality, HGB`

#### 3.4 分类器

```
concat(16+64+12=92d) → Linear(92, 64) → BN → ReLU → Dropout → Linear(64, 1)
```

### 4. 训练参数 (`residual_model/train23234_v3.py:37-53`)

| 参数 | 值 | 说明 |
|------|-----|------|
| `seed` | `[23234]` | 单次实验（非 100 次） |
| `batch_size` | 16 | — |
| `num_epochs` | 30 | 比单模态模型少 |
| `lr` | 1e-5 | — |
| `weight_decay` | 1e-5 | — |
| `early_stop_patience` | 12 | 比前两个模型的 20 更短 |
| `dropout` | 0.3 | 介于 CT-only(0.2) 和 Pathology-only(0.5) 之间 |
| `label_smoothing` | 0.1 | — |
| `mixup_alpha` | 0.2 | 对三模态同时 mixup |
| `drop_path` | 0.15 | — |
| `use_amp` | True | 混合精度训练 |

### 5. 数据增强

与 CT-Only 模型相同的增强策略（几何变换 + 强度扰动），**额外增加了 CT 图像 × mask 的肿瘤区域屏蔽**（`tensor = tensor * mask_tensor`），使模型只关注肿瘤区域。

### 6. 训练策略特殊之处

#### 6.1 错误分析与诊断（核心亮点）

训练脚本在每个 epoch（>6 轮预热后）进行**验证集错误样本诊断**：

1. 计算每个验证集错误样本对 AUC 的影响（删除该样本后 AUC 的变化量 `ΔAUC`）
2. 按 `ΔAUC` 降序排列，打印 Top 10
3. 统计每个样本在预热期之后的**累计错误次数和错误率**

#### 6.2 保存门槛

仅当 `test_auc > 0.73` 时才保存 checkpoint（比 CT-Only 的 0.6 更高）。

---

## 四、三个模型对比总览

| 维度 | Pathology-Only | CT-Only | Residual (多模态) |
|------|:---:|:---:|:---:|
| 模态 | 病理特征 | CT 图像 | CT + 病理 + 临床 |
| 骨干网络 | 无（纯 MLP） | ConvNeXt Tiny | DenseNet121 |
| 输入维度 | 1536d | 224×224 图像 | 图像 + 1536d + 11d |
| batch_size | 32 | 16 | 16 |
| num_epochs | 40 | 50 | 30 |
| lr | 1e-5 | 1e-6 | 1e-5 |
| weight_decay | 1e-3 | 1e-5 | 1e-5 |
| early_stop | 20 | 20 | 12 |
| dropout | 0.5 | 0.2 | 0.3 |
| mixup_alpha | 0.0（关闭） | 0.2 | 0.2 |
| use_amp | 关闭 | 开启 | 开启 |
| 训练集策略 | 每患者 1 样本 | ct_max 单切面 | 三切面各为独立样本 |
| CT×Mask | 无 | 无 | 有 |
| 保存门槛 | 无 | test_auc > 0.6 | test_auc > 0.73 |
| 错误诊断 | 无 | 无 | 验证集错误分析 |
| seeds 数量 | 100 | 100 | 1 |
| 标签文件 | label_v3.csv | label_v3.csv | label_iter_23234_v3.csv |
| 预热期 | 无 | 6 epoch | 6 epoch |
| 数据标准化 | z-score（病理特征） | mean/std=0.5（图像） | z-score（病理+临床）+ 0.5（图像） |

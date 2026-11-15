# CT Swin-Tiny Model

这个目录用于纯 CT 二分类复发预测。

当前实际实现不是旧版的 `ConvNeXt-Tiny + slice aggregation`，而是：

`3-slice CT as 3 channels -> Swin-Tiny -> optional CBAM/SE -> MLP classifier`

## 输入

- 数据目录：`data/images/ct5_cropped`
- 标签文件：`data/cvs/label_a.csv`
- 每个病例固定使用 3 张 CT slice：
  - `max`
  - `above`
  - `below`

兼容说明：

- 数据加载器仍兼容历史命名 `above_1` / `below_1`
- 一个病例聚合为一个样本，返回形状 `(3, 128, 128)`

## 模型结构

```text
输入: (B, 3, 128, 128)
  └─ 3 张灰度 CT slice 按通道堆叠: max / above / below

特征编码
  └─ Swin Transformer Tiny
     pretrained from timm
     forward_features 输出 NHWC 特征图

特征增强
  ├─ NHWC -> NCHW
  ├─ optional SE
  └─ optional CBAM

池化
  └─ global average pooling -> 768-d feature

分类头
  └─ 768 -> 64 -> 8 -> 1
     LayerNorm + GELU + Dropout
```

## 当前配置

- backbone：`swin_tiny_patch4_window7_224`
- `IMG_SIZE = 128`
- 默认使用 `timm` 预训练权重
- 当前配置：`USE_CBAM = True`，`USE_SE = False`
- 训练脚本按合并 test 集的 `test AUC` 选择最佳模型，所有轮次参与选模，并列保留首次；前 6 轮只豁免早停计数。

## 说明

- 当前不读取临床、病理、WSI 或 mask。
- 模型把 3 张切片直接当作 3 通道图像输入 backbone，而不是逐 slice 编码后再做聚合。
- 目录内 README 现已按代码同步。

## 运行

```bash
conda activate wsi
cd ct_model_Swin_tiny
python train.py
```

# Fusion Model

这个目录用于多模态二分类复发预测，融合 3 个模态：

- CT 三切片图像
- 病理 UNI2 slide feature
- 临床表型特征

当前实现名称可概括为：

`CMT-CPG: Cross-Modal Transformer with Clinical-Prior Gating`

## 输入

- CT：`data/images/ct5_cropped`
- 病理特征：`data/uni2_feature/jiandu`
- 临床特征：`data/cvs/clinical_filtered.csv`
- 标签：`data/cvs/label_a.csv`

每个病例对应：

- CT 图像 `(3, 128, 128)`，由 `max / above / below` 三张切片组成
- 病理特征 `(1536,)`
- 临床特征 `(11,)`

## 模型结构

```text
CT branch
  3-slice CT -> Swin-Tiny -> optional CBAM/SE -> global average pooling -> 768 -> 16

Pathology branch
  UNI2 feature 1536 -> 64

Clinical branch
  11 clinical variables -> 12

Fusion
  CT / pathology / clinical -> project to d_model=64
  stack as 3 modality tokens
  add learnable modality embedding
  2-layer Transformer encoder for cross-modal interaction

Clinical-prior gating
  use clinical token as query
  MLP -> 3 gate logits
  temperature = exp(log_temperature)
  gate = sigmoid(logits / temperature)

Aggregation
  learnable modality weights
  softmax-weighted sum across 3 gated tokens

Classifier
  64 -> 32 -> 1
```

## 当前配置

- CT backbone：`swin_tiny_patch4_window7_224`
- `CT_OUTPUT_DIM = 16`
- `FUSION_D_MODEL = 64`
- `FUSION_NHEAD = 4`
- `FUSION_NUM_LAYERS = 2`
- 当前配置：`USE_CBAM = True`，`USE_SE = False`

## 说明

- 这个目录的核心模型是 `CrossModalTransformerFusion`。
- 当前训练脚本仍然使用 `test AUC` 参与最佳模型选择；这属于训练协议问题，不影响这里描述的网络结构。
- `DATA_MASK_DIR` 目前在数据集实现中没有真正参与模型输入。

## 运行

```bash
source /home/guozheng/miniconda3/etc/profile.d/conda.sh
conda activate wsi
cd /home/guozheng/claude_dir/wsi_research/fusion_model
python train.py
```

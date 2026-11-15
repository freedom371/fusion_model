# Pathology Model

这个目录用于纯病理特征的二分类复发预测。

当前实现是一个基于 UNI2 slide feature 的 MLP 分类器：

`1536 -> 512 -> 128 -> 32 -> 1`

## 输入

- 病理特征目录：`data/uni2_feature/jiandu`
- 标签文件：`data/cvs/label_a.csv`

每个患者保留 1 个病理样本：

- 自动过滤 `label_a.csv` 中的 `_ct_above` / `_ct_below` 变体
- 只保留真实患者级记录
- 输入特征形状 `(1536,)`

## 模型结构

```text
输入
  UNI2 slide feature: 1536-d

MLP classifier
  Dropout
  1536 -> 512 -> LayerNorm -> ReLU
  Dropout
  512 -> 128 -> LayerNorm -> ReLU
  Dropout
  128 -> 32 -> LayerNorm -> ReLU
  Dropout
  32 -> 1
```

## 说明

- 当前数据集在每个 split 内部独立计算病理特征 z-score。
- 当前训练脚本按 `test AUC` 保存最佳权重，所有轮次参与选模，并列保留首次；前 6 轮只豁免早停计数。

## 运行

```bash
conda activate wsi
cd pathology_model
python train.py
```

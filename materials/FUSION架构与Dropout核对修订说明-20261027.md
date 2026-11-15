# FUSION 架构与 Dropout 核对修订说明

核对日期：2026-10-27。

源文件：[三模型架构描述-20261027.pptx](../ppt/三模型架构描述-20261027.pptx)。

修订副本：[三模型架构描述-20261027_FUSION核对修订版.pptx](../ppt/三模型架构描述-20261027_FUSION核对修订版.pptx)。

页面预览：[FUSION修订页面预览-20261027.pdf](../ppt/FUSION修订页面预览-20261027.pdf)。

## 1. 最终临床原始变量为 6 个

严格按 `fusion_model/config.py` 的顺序：

| 输入序号 | 变量 | 含义 |
|---|---|---|
| 1 | pT | 病理 T 分期 |
| 2 | pN | 病理 N 分期 |
| 3 | pAJCC | 病理 AJCC 分期 |
| 4 | PNI | 神经束浸润 |
| 5 | LVI | 脉管内癌栓 |
| 6 | Treatment | 治疗方式 |

临床输入为 `(B,6)`。数据加载路径未进行 one-hot 编码；适配器输出仍为 6 维。其后投影到 64 维是学习得到的语义表示，不代表增加原始变量。

第 26 页图中的 8 项（Age、LYMPH、Site、Laterality、HGB、smoking、alcohol、TP），第 30 页末尾“8 个”，第 31 页“11 个”及末尾“8 个”，以及第 32 页“12 维临床”，均与当前实际输入不符，属于旧版残留描述，不能解释为当前编码后维度。修订版已统一为上述 6 项。

依据：[配置](../fusion_model/config.py)、[临床数据加载](../fusion_model/dataset.py)、[ClinicalAdapter](../fusion_model/model.py)。

## 2. 最终 CT 投影路径为 768 → 64 → 8 → 64

```text
三张 CT 按通道输入 (B,3,128,128)
→ Swin-Tiny：768 通道特征图
→ CBAM
→ 全局平均池化：(B,768)
→ Linear(768,64) → LayerNorm(64) → GELU → Dropout(0.1)
→ Linear(64,8) → LayerNorm(8) → GELU
→ CTEncoder 输出：(B,8)
→ ct_proj：Linear(8,64) → LayerNorm(64)
→ 融合 CT token：(B,64)
```

第 30 页原本的 `768→64→8` 是当前 CTEncoder 的正确路径；第 31 页的 `768→16 + BatchNorm + ReLU` 不是当前执行路径，第 32 页的 16 维 CT token 也不是当前实现。

虽然 `config.py` 还存在 `CT_OUTPUT_DIM=16`，当前执行的 `model.py` 显式构造 `Linear(64,8)`，随后设置 `self.feature_dim=8`；融合层读取这个实际 `feature_dim`，因此最终是 `Linear(8,64)`。16 是遗留配置，不控制当前 CT 投影输出。

另两条路径为：

```text
病理：1536 → Dropout(0.1) → Linear(1536,64) → LayerNorm → ReLU
      → Linear(64,64) → LayerNorm → (B,64)

临床：6 → Dropout(0.1) → Linear(6,6) → LayerNorm → ReLU
      → Linear(6,64) → LayerNorm → (B,64)
```

三种 token 堆叠为 `(B,3,64)`，加可学习模态嵌入后进入 Transformer。

依据：[CTEncoder 及统一投影层](../fusion_model/model.py)。

## 3. 各模块 Dropout 的实际概率与位置

训练入口实际创建：

```python
# fusion_model/train.py
conf["dropout"] = 0.1
model = CrossModalTransformerFusion(dropout=c["dropout"]).to(device)
```

`CrossModalTransformerFusion` 将同一个实参继续传给 CTEncoder、PathologyAdapter、ClinicalAdapter、TransformerEncoderLayer 和分类器，因此各处显式 Dropout 都是 **p=0.1**。

| 模块 | 实际位置 | 训练概率 |
|---|---|---:|
| CTEncoder 投影头 | `Linear(768,64) → LayerNorm → GELU` 之后，`Linear(64,8)` 之前 | 0.1 |
| 病理适配器 | `Linear(1536,64)` 之前 | 0.1 |
| 临床适配器 | `Linear(6,6)` 之前 | 0.1 |
| Transformer 每层自注意力 | MultiheadAttention 的注意力权重 Dropout | 0.1 |
| Transformer 每层 FFN | `Linear(64,128) → GELU` 后，`Linear(128,64)` 前 | 0.1 |
| Transformer 每层注意力残差分支 | 自注意力输出后、与残差相加前 | 0.1 |
| Transformer 每层 FFN 残差分支 | FFN 输出后、与残差相加前 | 0.1 |
| 最终分类器 | `Linear(64,32) → LayerNorm → GELU` 后，`Linear(32,1)` 前 | 0.1 |

Transformer 共 2 层，上述 Transformer 设置在两层均适用。三个模态的统一投影层、CBAM、门控 MLP、自适应聚合层没有额外的显式 Dropout。模型进入 `eval()` 后，Dropout 关闭。

图中 `Transformer dropout=0.5` 对应的值与构造默认值接近，但**训练实参 0.1 会覆盖默认值**；分类器的 `Dropout(0.3)` 与当前训练入口不符。二者在修订版中统一更正为 0.1。

`config.py` 的 `FUSION_DROPOUT=0.5` 是模型省略 dropout 实参时的默认值。`train.py` 中另有无参模型用于参数数量统计，但真正执行训练的实例位于 `train_one_seed`，显式传入 0.1；不能根据统计实例或类默认值推断实际训练概率。

依据：[模型定义](../fusion_model/model.py)、[训练配置及实例化](../fusion_model/train.py)。

## 4. 同步修正的页面

| 页面 | 修订内容 |
|---|---|
| 26 | 将整页旧图片重绘为可编辑的 PowerPoint 图形和文字；统一 6 项临床输入、8 维 CT 表示及 Dropout=0.1 |
| 27–28 | 同步调整英文功能说明，准确区分 CTEncoder 内部投影与融合前的独立 CT token 投影，并改善排版 |
| 30 | 统一中文输入名单、完整 CT 路径和 Dropout 位置；明确旧数量不属于编码后维度 |
| 31 | 与中文页一致；删除旧版 `768→16`、BatchNorm/ReLU 和 11/8 项临床输入描述 |
| 32 | CT 16→8，临床 12→6；统一投影、注意力及分类器描述 |
| 33–34 | 中英文页明确各处 Dropout 位置及 p=0.1，并同步当前训练参数 |

第 33–34 页还保留了旧训练参数，本次按当前 `train.py` 统一为：

```text
AdamW：lr=9e-6，weight_decay=1e-7
CosineAnnealingLR：T_max=30，eta_min=1e-6
batch_size=16，num_epochs=30，early_stop_patience=20
AMP=True，num_workers=4，seed=[1583]
dropout=0.1，label_smoothing=0，mixup_alpha=0
```

## 5. 冻结路径的额外核对

当前 `CTEncoder.forward` 在 `freeze_backbone=True` 时，使用 `torch.no_grad()` 包住 `_extract_features`；后者包括 backbone、CBAM、全局池化和内部 `768→64→8` 投影。因此，按当前执行代码，该路径中的 CBAM 和内部投影不会获得训练梯度；独立的 `ct_proj: 8→64` 位于此范围之外，可以训练。

`no_grad()` 不等于 `eval()`：外层模型训练时 CT 内部投影头的 Dropout 仍会按 0.1 运行，只是不对该路径反向求导。修订版据此去掉了“CT 内部投影正在训练”的不准确描述。这是现有实现的记录，没有修改冻结逻辑或模型代码。

## 6. 验证

- 以训练配置 `dropout=0.1` 实例化相同架构，逐项读取了显式 Dropout 及 MultiheadAttention 的概率，均符合上述表格。架构检查使用 `pretrained=False` 避免下载，未改动项目训练配置或权重。
- 核对参数形状：CT 投影权重为 `(64,768)`、`(8,64)`；融合 CT 投影为 `(64,8)`；临床适配器为 `(6,6)`；融合临床投影为 `(64,6)`。
- 修订副本保持 48 页，仅第 26、27、28、30、31、32、33、34 页内容发生变化。
- 已用 LibreOffice 渲染并查看修订页；第 26、30–34 页新增文字均可在 PDF 中完整提取，形状未超出画布边界。
- 原始 PPT、模型代码、训练配置和权重文件未修改。

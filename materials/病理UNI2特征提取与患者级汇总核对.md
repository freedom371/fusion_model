# 病理 UNI2 特征提取与患者级汇总核对

核对日期：2026-10-27。

核对范围：`pathology_model` 当前代码、`feature_extraction` 提取脚本、本地 UNI2 权重与 `data/uni2_feature/jiandu` 现存结果。

## 可直接用于方法说明的回答

接收的病理数据为已经切块、去白及颜色标准化的 JPG patch。现存提取摘要记录的来源为 `/hard0/carcinoma/jiandu/yansebiaozhunhua/<患者ID>`。本项目提取流程直接读取病例目录内全部符合命名格式的 JPG，不再进行肿瘤区域筛选、patch 随机抽样或二次染色标准化。图像转为 RGB，缩放至 224×224，转为 `[0,1]` 浮点数，并使用 ImageNet RGB 均值 `(0.485, 0.456, 0.406)` 与标准差 `(0.229, 0.224, 0.225)` 进行通道标准化，再输入预训练的 **UNI2-h（MahmoodLab/UNI2-h）** 提取每个 patch 的 1536 维特征。UNI2 仅用于推理，在本项目中未微调。

对每位患者目录中的全部 patch 特征逐维计算等权算术平均，形成一个 `(1,1536)` 的患者级特征文件。下游病理模型读取该文件，去掉长度为 1 的首维后得到 `(1536,)` 向量，再使用训练集患者的逐维均值和标准差进行 z-score 标准化；验证集和测试集使用相同的训练集统计量。当前仅训练下游 MLP 分类器，不更新 UNI2 权重。

上述“已切块、去白及颜色标准化”依据项目说明及输入目录记录；本仓库未提供交付前去白阈值、染色标准化算法、扫描倍率或物理分辨率的完整处理记录，不能据此补写具体参数。

## ① 接收后是否又做了筛选、抽样、缩放或数值归一化

需要区分图像提取阶段和患者级特征加载阶段。

### 图像提取阶段

现存 `summary.json` 的字段与 `batch_tiles_to_uni2_temp.py` 的输出一致，记录的是直接从 JPG patch 提取特征的路径。不能把通用 WSI demo 中的组织检测、重新切块或背景过滤参数直接当作这批结果的实际处理。

| 项目 | 当前可确认的处理 |
|---|---|
| patch 筛选 | 扫描病例目录下全部 `*.jpg`，解析文件名与坐标，检查画布尺寸一致性；未见按肿瘤概率、清晰度、组织含量或预后标签再次筛选 patch |
| patch 抽样 | 提取时使用扫描得到的全部 patch，未进行固定数量抽样、随机抽样或 top-k 选择 |
| 组织检测、重新切块 | 直接读取已有 patch，此提取路径未调用 WSI 组织检测或重新切块 |
| 颜色处理 | 转换为 RGB；未见再次调用染色标准化算法 |
| 空间缩放 | 通过 UNI2 的 `get_transform()` 将图像缩放至 224×224；当前安装的 LazySlide 实现使用双三次插值、`antialias=True`，随后 `CenterCrop(224)`，因前一步已为 224×224，不会额外缩小视野 |
| 像素值处理 | 转为 float32，将 8 位像素从 `[0,255]` 缩放为 `[0,1]`，再进行 ImageNet RGB 通道标准化 |
| 随机图像增强 | 此特征提取路径中未见随机旋转、翻转、颜色扰动等增强 |

RGB 通道标准化公式为：

```text
x_norm[c] = (x[c] / 255 - mean[c]) / std[c]
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)
```

这一步属于模型输入的数值标准化，不等同于病理染色标准化，也不等同于 CT 的 `Normalize(mean=0.5, std=0.5)`。

抽查患者 1、100、846：输入 JPG 均为 512×512；源目录 JPG 数量分别为 3248、6486、825，与各自保存的 patch 特征行数一致。512×512 是抽查结果，不应仅凭该抽查推断所有图像的原始尺寸或扫描倍率。

缩略图生成代码会最多选取 100 张 patch 拼图，仅用于展示，不影响特征提取与聚合使用的 patch 数量。

代码依据：[批量 JPG 特征提取](../feature_extraction/batch_tiles_to_uni2_temp.py)、[目录扫描与坐标检查](../feature_extraction/tiles_to_wsidata.py)、[图像加载及推理](../feature_extraction/demo_lazyslide_uni2.py)。

### 患者级特征加载阶段

病理模型读取：

```text
data/uni2_feature/jiandu/<患者ID>/<患者ID>_uni2_slide_features.npy
```

加载后执行以下处理：

1. 使用 `squeeze(0)` 将 `(1,1536)` 转为 `(1536,)`，并转换为 float32。
2. 根据标签 CSV 的 `group` 选择训练、验证或测试患者；代码会过滤 `_ct_above` / `_ct_below` 变体，避免把 CT 切面变体当作独立病理患者。
3. 对 1536 个特征维度分别进行训练集 z-score 标准化。

```text
mu[j] = mean(训练集患者的第 j 维特征)
sd[j] = std(训练集患者的第 j 维特征, ddof=0) + 1e-8
z[p,j] = (feature[p,j] - mu[j]) / sd[j]
```

验证集和测试集复用训练集统计量；如果未先创建训练数据集，代码会从标签 CSV 的 train 组读取特征并计算统计量。这里没有额外的 PCA、特征维度筛选、L2 归一化、min-max 缩放或 patch 抽样。

当前标签文件为 `data/cvs/label_l_impact100_val58_r32_t36_mv40_v3_d3_m3.csv`，当前分组为 train 420 例、val 129 例、test1～test4 合计 100 例、exclude 197 例。即原始特征目录有 846 例，但并非全部进入当前模型训练及评估；exclude 组不被上述数据集加载。当前 CSV 中没有 `_ct_above` / `_ct_below` 或 `_aug` 变体记录。该分组属于患者层面的纳入设置，与 patch 筛选不同。

代码依据：[病理配置](../pathology_model/config.py)、[病理数据加载及标准化](../pathology_model/dataset.py)。

## ② UNI2 的具体版本、权重及是否微调

| 项目 | 核对结果 |
|---|---|
| 模型名称 | UNI2-h，Hugging Face 标识为 `MahmoodLab/UNI2-h`；LazySlide 注册名称为 `uni2` |
| 本地权重 | `feature_extraction/uni2/pytorch_model.bin`，约 2.6 GiB |
| 模型配置 | `feature_extraction/uni2/config.json`，`pretrained_cfg.tag = uni2-h`，特征维度 1536，`global_pool = token` |
| 构建名称 | `vit_giant_patch14_224`，通过自定义参数构建 UNI2-h 架构 |
| 关键参数 | 输入 224×224；Transformer 内部 patch/token 大小 14；depth=24；num_heads=24；embed_dim=1536；reg_tokens=8；SwiGLU；num_classes=0 |
| 权重加载 | 提取配置指向本地 `uni2` 目录，解析为 `pytorch_model.bin`；LazySlide 的 UNI2 类使用 `load_state_dict(..., strict=True)` 加载 |
| 是否微调 | 未微调；特征提取在 `eval()` 和 `torch.inference_mode()` 下运行，没有 UNI2 的梯度更新 |
| 下游训练 | 从预存 `.npy` 加载固定特征，仅优化病理 MLP 分类器 |

这里的 14 是 Transformer 内部图像分块大小，不是接收的 512×512 病理 patch 尺寸。

本次计算的本地权重 SHA-256 为：

```text
6e077eda234bebc595868d918d3458d9dd32a050199b0ff04443b2f46a0a3b1e
```

当前环境版本为 LazySlide 0.10.1、timm 1.0.26、PyTorch 2.11.0、torchvision 0.26.0。其中 LazySlide 和 timm 版本也与项目 `requirements.txt` 一致。现存提取摘要没有记录权重哈希、Hugging Face commit/revision 或完整历史环境版本，因此上述哈希标识当前本地文件，不能将其表述成已被历史提取日志逐例记录的权重哈希。

代码依据：[提取配置与模型构建](../feature_extraction/batch_tiles_to_uni2_temp.py)、[本地模型配置](../feature_extraction/uni2/config.json)、[本地模型说明](../feature_extraction/uni2/README.md)、[病理分类器](../pathology_model/model.py)、[病理训练入口](../pathology_model/train.py)。UNI2 类与实际图像变换另核对了当前 `wsi` 环境的 LazySlide 安装源码。

## ③ 多个 patch 如何汇总为患者最终的 1536 维特征

每个 patch 单独经过 UNI2，得到一个 1536 维向量。对患者 p 的 N 个 patch，记特征矩阵为：

```text
F_p.shape = (N, 1536)
```

随后执行：

```python
slide_features = tile_features.mean(axis=0, keepdims=True)
```

即对每个特征维度做等权算术平均：

```text
patient_feature[p,j] = (1 / N) × sum(F_p[i,j], i=1,...,N)
j = 1,...,1536
```

因此，无论患者有多少个 patch，输出都为 `(1,1536)`；下游读取后为 `(1536,)`。聚合发生在每个 patch 提取 1536 维特征之后，没有按 batch 先平均再等权平均，也没有对不同 patch 赋予学习得到的注意力权重。

该方法没有使用 attention-MIL、最大池化、top-k pooling 或根据 patch 位置进行空间加权。patch 坐标保存在 `tile_metadata.csv`，不参与均值计算。模型内部的 Transformer token pooling 与患者目录内多个病理 patch 之间的均值聚合是两个层次，不能混为一谈。

以患者 100 为例：

```text
6486 个 JPG patch
→ UNI2-h 分别提取特征
→ (6486, 1536) patch 特征矩阵
→ 沿 patch 维度等权取平均
→ (1, 1536) 患者级原始特征
→ 去掉首维并按训练集统计量做 z-score
→ (1536,) 下游分类器输入
```

这里的“患者级”以病例目录为单位：当前训练代码将每位患者映射到一个病例目录和一个聚合文件。没有额外的“多张 WSI 各自聚合后，再汇总到患者”的层级；若一个目录内包含多个来源切片的 patch，此实现会将目录内全部 patch 一起平均。不能仅凭患者级文件名断言每位患者只对应一张原始 WSI。

## ④ 本次验证结果与说明文件差异

- 核对全部 846 份 `summary.json`：模型字段均为 `uni2`，patch 特征维度均为 1536；每例 patch 数量最少 100、中位数 5949、最多 32754。
- 全部 846 例重新计算 `tile_features.mean(axis=0, keepdims=True)`，均与对应保存的 slide 特征逐元素完全一致，无不匹配。
- 全部摘要的输入来源父目录均为 `/hard0/carcinoma/jiandu/yansebiaozhunhua`。
- 部分摘要保留了历史输出路径 `feature_extraction/data/uni2_feature/ruojiandu/_default/...`，当前病理训练实际读取 `data/uni2_feature/jiandu/...`；以上聚合验证针对当前实际读取的文件，不能只凭历史输出目录名判断输入类别。
- `pathology_model/README.md` 中“各 split 独立标准化”的描述已与当前代码不符。当前代码是训练集计算统计量，验证及测试复用。
- README 中较深的 MLP 架构也不是当前执行结构；当前 `model.py` 实际为 `Dropout → Linear(1536,64) → ReLU → Linear(64,1)`。这属于下游分类器，不改变均值汇总得到的原始 1536 维特征。

本文件记录核对结果，未修改模型、特征文件或既有训练配置。

# Project Background
## python环境 使用conda工具 wsi这个环境
conda activate  wsi 

## 项目背景

本项目当前的目标是围绕口腔癌数据，搭建一个基于 `LazySlide + UNI2` 的病理基础模型特征提取 demo，用于从病理大图像中提取：

- tile-level features
- slide-level aggregated features

最终希望形成一个可复用的小型流程，支持后续：

- 更换病理基础模型
- 批量处理病例
- 与临床/生存/放射组学等其他模态做联合分析

## Current Data Situation

当前项目目录位于：

`/home/guozheng/wsi_research/feature_extraction`

用户提供的数据目录位于：

`/home/guozheng/wsi_research/feature_extraction/data`

该目录下目前有两类核心数据：

### 1. `WSI/`

这是当前病理分析主线使用的数据目录。

目录结构大致为：

`WSI/<class>/<case>/*.jpg`

例如：

- `WSI/jiandu/1/*.jpg`
- `WSI/jiandu/2/*.jpg`
- `WSI/ruojiandu/1/*.jpg`

目前确认到的特点：

- 这不是标准原始病理全切片文件，例如 `.svs`、`.ndpi`、`.tiff`
- 它已经是切好的 tile 数据
- 每个病例目录下有几千张 `.jpg`
- 文件名中带有坐标信息，例如：
  `1_46080_43520_10240_10240_10752_10752.jpg`
- 单个 tile 当前观测为 `512 x 512`
- 文件名中包含整张画布大小和 tile 的左上/右下坐标

这意味着：

- 不能直接把 `WSI/` 目录里的 `.jpg` 当成标准 WSI 交给 `LazySlide.open_wsi`
- 需要先把这些 tile 按坐标重建成一张大图，再转换成 `WSIData`

### 2. `Radiology/`

这是另一条数据线，不属于当前病理 WSI 特征提取主流程。

目前确认到的内容包括：

- `images/*.nii.gz`
- `ITH features/*.npy`
- `masks/...`

说明：

- 这是放射影像/影像组学数据
- 其核心格式是 `NIfTI`
- 不适合直接进入当前 `LazySlide + UNI2` 的病理 WSI 脚本

因此当前阶段：

- `WSI/` 用于病理特征提取
- `Radiology/` 暂不接入当前病理抽特征流程

## Current Technical Understanding

### LazySlide 的定位

`LazySlide` 不是单一模型，而是病理 AI 基础设施/工作流框架。它主要负责：

- WSI 打开与存储
- tissue detection
- tile generation
- feature extraction
- feature aggregation

### UNI2 的定位

`UNI2` 是病理基础模型之一，用于从病理 tile 中提取特征。

在当前项目里，`UNI2` 的作用是：

- 为每个 tile 提取 embedding
- 再将 tile embeddings 聚合成 slide-level representation

### WSIData 的定位

`WSIData` 是 `LazySlide` 使用的统一数据对象。

对于当前项目，可以把它理解为：

- 一个标准化的病理大图数据表示
- 便于后续做 tissue 检测、切块、特征提取、缓存和复用

当前项目里，`WSIData` 不是原始输入数据本身，而是需要从已有 tile 数据进一步转换得到的中间标准对象。

## Scripts in This Project

### 1. `demo_lazyslide_uni2.py`

作用：

- 使用 `LazySlide + UNI2` 对病理图像做特征提取
- 输出 tile-level 和 slide-level 特征

当前配置方式：

- 采用顶部 `CONFIG` 直接改参数
- 不使用 `argparse`

主要逻辑：

1. 打开 WSI
2. 检测 tissue
3. 生成 tiles
4. 用 `UNI2` 提取 tile 特征
5. 做 slide-level 聚合
6. 导出结果

注意：

- 这个脚本适合读取标准 WSI 或可被 `LazySlide` 直接打开的大图
- 不适合直接读取当前 `WSI/` 目录里的 `.jpg` tile 集合

### 2. `tiles_to_wsidata.py`

作用：

- 读取 `WSI/` 目录下每个病例的 `.jpg` tile
- 从文件名中解析 tile 坐标
- 把 tile 拼接回整张大图
- 将大图保存为 `.tif`
- 再转成 `LazySlide` 可读取的 `WSIData (.zarr)`

当前默认输入目录：

`D:\BaiduNetdiskDownload\oral carcinoma\oral carcinoma\WSI`

当前默认输出目录：

`D:\BaiduNetdiskDownload\oral carcinoma\oral carcinoma\converted_wsidata`

它是连接当前原始病理数据与 `demo_lazyslide_uni2.py` 的关键中间步骤。

## Recommended Workflow

当前推荐的项目流程如下：

1. 从 `WSI/` 目录读取病理 tile 数据
2. 用 `tiles_to_wsidata.py` 将每个病例重建成大图并转换成 `WSIData`
3. 用 `demo_lazyslide_uni2.py` 读取重建后的大图或对应 `WSIData`
4. 提取 `UNI2` 特征
5. 保存 tile-level 和 slide-level 特征

## Constraints and Risks

当前项目存在以下现实约束：

- 用户手头没有标准原始 `.svs/.ndpi` WSI，而是已经切好的 `.jpg` tile
- 因此需要先做重建，再进入标准病理 WSI 流程
- `UNI2` 是 gated Hugging Face model，通常需要：
  - 已获得 `MahmoodLab/UNI2-h` 访问权限
  - 有可用的 `hf_token`
  - 或者本地模型权重
- 大图重建和特征提取都可能占用较多磁盘、内存和显存

## Near-Term Next Steps

建议的近期工作顺序：

1. 先用 `tiles_to_wsidata.py` 成功处理一个病例
2. 确认生成的 `.tif` 和 `.zarr` 能被正常读取
3. 再让 `demo_lazyslide_uni2.py` 对该病例做 `UNI2` 特征提取
4. 成功后再扩展到批量病例
5. 后续再考虑将病理特征与：
   - `clinical.csv`
   - `survival.csv`
   - `Radiology/`
   做联合分析

## Summary

当前项目本质上是在做一条病理基础模型特征提取链路的搭建：

`tile jpg dataset -> reconstructed WSI -> WSIData -> LazySlide + UNI2 -> pathology features`

其中：

- `WSI/` 是病理输入主线
- `Radiology/` 是另一模态数据，暂不并入当前主流程
- `tiles_to_wsidata.py` 负责把现有 tile 数据转换成标准病理工作流可用格式
- `demo_lazyslide_uni2.py` 负责基于 `LazySlide + UNI2` 完成特征提取

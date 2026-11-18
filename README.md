# OSCC 多模态复发预测

## 模型与目录

| 模型 | 目录 | 输入 | 说明 |
| --- | --- | --- | --- |
| 临床模型 | [clin_model/](clin_model/) | 临床表格 | 候选特征 LASSO 筛选（train-only 10 折 CV，1-SE 规则）+ 多因素 logistic 重拟合 |
| CT 单模态 | [ct_model_Swin_tiny/](ct_model_Swin_tiny/) | 3 切片 CT | Swin-Tiny（timm 预训练）+ CBAM + MLP 分类头 |
| 病理单模态 | [pathology_model/](pathology_model/) | UNI2 特征 | `1536 → 512 → 128 → 32 → 1` MLP |
| 三模态融合 | [fusion_model/](fusion_model/) | 临床 + CT + 病理 | CMT-CPG 跨模态 Transformer |
| 双模态消融 | [fusion_ablation/](fusion_ablation/) | 临床-CT / 临床-病理 / CT-病理 | 从头训练融合层，另含同协议三模态 `full` 对照 |

 

## 项目结构

```text
wsi_research/
├── clin_model/            # 临床特征筛选与 logistic 回归
├── ct_model_Swin_tiny/    # CT 单模态（Swin-Tiny + CBAM）
├── pathology_model/       # 病理单模态（UNI2 特征 MLP）
├── fusion_model/          # 三模态融合 CMT-CPG
├── fusion_ablation/       # 双模态消融与三模态对照
├── residual_model/        # 残差/基线与对照实验
├── feature_extraction/    # WSI 切片预处理与 UNI2 特征提取
├── materials/             # 模型架构图与设计说明
├── pic/                   # 模型架构图（SVG）
└── project.md             # 课题设计说明
```

## 环境

Python 3.10+，推荐使用 conda 环境：

```bash
conda create -n wsi python=3.10
conda activate wsi
pip install torch torchvision timm scikit-learn pandas numpy matplotlib joblib statsmodels openpyxl
```

WSI 预处理另需 `lazyslide`、`openslide-python` 等，见 [feature_extraction/requirements.txt](feature_extraction/requirements.txt)。

> 国内网络可使用清华镜像：`pip install <包名> -i https://pypi.tuna.tsinghua.edu.cn/simple`


## 训练

在项目根目录、`wsi` 环境中执行：

```bash
conda activate wsi

# 临床模型：LASSO 筛选 + 多因素 logistic
python clin_model/train_lasso_selection.py

# CT 单模态
cd ct_model_Swin_tiny && python train.py && cd ..

# 病理单模态
cd pathology_model && python train.py && cd ..

# 三模态融合
cd fusion_model && python train.py && cd ..

# 双模态消融（依次训练三组）
python -m fusion_ablation.train --model all

# 同协议三模态对照
python -m fusion_ablation.train --model full
```

多随机种子训练与自定义超参数见 [fusion_ablation/README.md](fusion_ablation/README.md)。

## 权重

本仓库使用 **Git LFS** 管理权重文件。克隆后先初始化 LFS：

```bash
git lfs install
git lfs pull
```

| 模型 | 权重路径 | 大小 |
| --- | --- | --- |
| CT 单模态（seed 1085） | `ct_model_Swin_tiny/output/ct_swin_tiny_slice_agg/checkpoints/best_model_seed1085.pt` | 106 MB |
| 病理单模态（seed 2837） | `pathology_model/output/pathology_only/checkpoints/best_model_seed2837.pt` | 388 KB |
| 三模态融合（seed 1583） | `fusion_model/output/cmt_cpg/checkpoints/best_model_seed1583.pt` | 107 MB |
| 消融：临床-CT（seed 4977） | `fusion_ablation/output/clinical_ct/20261106_160432_454045_seed4977_best_model.pt` | 106 MB |
| 消融：临床-病理（seed 2941） | `fusion_ablation/output/clinical_pathology/20261106_161058_267473_seed2941_best_model.pt` | 708 KB |
| 消融：CT-病理（seed 3336） | `fusion_ablation/output/ct_pathology/20261106_154330_467644_seed3336_best_model.pt` | 107 MB |

 



## 引用与许可

论文发表信息与许可证待补充。

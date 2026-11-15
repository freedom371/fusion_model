# OSCC 多模态复发预测

口腔鳞状细胞癌（OSCC）多中心回顾性研究代码仓库。目标是基于术前临床特征、增强 CT 影像和全切片病理（WSI）特征，构建术后复发风险（RFS 二分类）预测模型，并进行多模态融合与消融比较。


## 研究概览

- **任务**：术后复发二分类预测（`label=1` 复发，`label=0` 无复发）
- **设计**：多中心回顾性研究，数据预先划分为 train / val / test，test 集为独立留出集
- **模态**：
  - 临床：术前临床及病理学特征（由 `clinical.csv` 派生）
  - CT：增强 CT 的 max / above / below 三张切片，按 3 通道输入
  - 病理：WSI 经 UNI2 提取的患者级 slide feature（1536 维）
- **融合**：CMT-CPG（Cross-Modal Transformer with Clinical-Prior Gating），临床、CT、病理三路 token 经 Transformer 交互后加权聚合

## 模型与目录

| 模型 | 目录 | 输入 | 说明 |
| --- | --- | --- | --- |
| 临床模型 | [clin_model/](clin_model/) | 临床表格 | 候选特征 LASSO 筛选（train-only 10 折 CV，1-SE 规则）+ 多因素 logistic 重拟合 |
| CT 单模态 | [ct_model_Swin_tiny/](ct_model_Swin_tiny/) | 3 切片 CT | Swin-Tiny（timm 预训练）+ CBAM + MLP 分类头 |
| 病理单模态 | [pathology_model/](pathology_model/) | UNI2 特征 | `1536 → 512 → 128 → 32 → 1` MLP |
| 三模态融合 | [fusion_model/](fusion_model/) | 临床 + CT + 病理 | CMT-CPG 跨模态 Transformer |
| 双模态消融 | [fusion_ablation/](fusion_ablation/) | 临床-CT / 临床-病理 / CT-病理 | 从头训练融合层，另含同协议三模态 `full` 对照 |
| 残差/基线模型 | [residual_model/](residual_model/) | CT + 病理 + 临床 | 对照实验 |

各子目录 README 包含该模型的详细结构、输入输出和训练参数。

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

## 数据准备

将数据放置为以下结构（内容不随仓库发布）：

| 路径 | 内容 |
| --- | --- |
| `data/cvs/*.csv` | 标签表（`编号`、`label`、`group` 等列）、临床特征表 |
| `data/images/ct5_cropped/` | 按病例组织的 CT 切片（`max` / `above` / `below`） |
| `data/uni2_feature/jiandu/` | UNI2 提取的病理 slide feature |

标签表需包含患者编号、二分类标签和划分分组（train/val/test）。各模型目录的 README 说明了所需的文件名和列格式；路径常量在对应 `config.py` / 脚本顶部，可按实际位置修改。

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

各模型 `output/` 下其余逐轮 checkpoint、训练日志和中间结果不纳入版本管理。

| 模型 | 权重路径 | 大小 |
| --- | --- | --- |
| CT 单模态（seed 1085） | `ct_model_Swin_tiny/output/ct_swin_tiny_slice_agg/checkpoints/best_model_seed1085.pt` | 106 MB |
| 病理单模态（seed 2837） | `pathology_model/output/pathology_only/checkpoints/best_model_seed2837.pt` | 388 KB |
| 三模态融合（seed 1583） | `fusion_model/output/cmt_cpg/checkpoints/best_model_seed1583.pt` | 107 MB |
| 消融：临床-CT（seed 4977） | `fusion_ablation/output/clinical_ct/20261106_160432_454045_seed4977_best_model.pt` | 106 MB |
| 消融：临床-病理（seed 2941） | `fusion_ablation/output/clinical_pathology/20261106_161058_267473_seed2941_best_model.pt` | 708 KB |
| 消融：CT-病理（seed 3336） | `fusion_ablation/output/ct_pathology/20261106_154330_467644_seed3336_best_model.pt` | 107 MB |

消融权重为每个消融组 2026-11-06 最后一次训练运行的权重，作为可复现示例；其余实验轮次的权重不发布。

## 评估口径与注意事项

- 所有按 epoch 训练的模型（CT、病理、融合、消融及残差模型）统一按 **test AUC** 保存最佳权重：所有轮次参与选模，AUC 并列时保留首次；前 6 轮仅豁免早停计数，不排除选模。
- **该选模策略会使 test 指标带有乐观偏倚**，test 结果不能视为完全独立的外部验证性能。若需严格评估，应改为按 val AUC 选模并仅在 test 上报告一次。
- 旧权重和已导出的概率不会随代码修改而更新，新规则需重新训练生效。
- 当前目标为二分类复发预测，输出 AUC、ACC、灵敏度、特异度等；未实现生存分析的 C-index、时间依赖 AUC 和校准曲线。`project.md` 中的 520/224/102 为早期设计口径，实际代码使用预先确定的统一划分（见各模型 README），二者不一致时以代码和数据文件为准。
- 本研究代码仅用于科研，不构成临床诊断工具。

## 引用与许可

论文发表信息与许可证待补充。

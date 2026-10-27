# 双模态融合消融

以现有 `fusion_model` 的 CMT-CPG 为基础，从头训练融合层，移除指定模态的编码器、输入、投影和 token。不是将已训练三模态模型的某一路输入置零，也不融合三个单模态模型的预测概率。

| 参数 `--model` | 输入 | 临床先验门控 |
| --- | --- | --- |
| `clinical_ct` | 临床 + CT | 保留，输出 2 个门值 |
| `clinical_pathology` | 临床 + 病理 | 保留，输出 2 个门值 |
| `ct_pathology` | CT + 病理 | 移除，直接对 Transformer token 做可学习加权聚合 |
| `full` | 临床 + CT + 病理 | 保留，作为同协议三模态对照 |
| `all` | 依次训练上面三个双模态模型 | 每组独立保存 |

## 网络结构

- CT：现有 Swin-Tiny + CBAM + `768 → 64 → 8`；三张切片作为 3 通道输入，128×128。
- 病理：现有 UNI2 slide feature `1536 → 64`。
- 临床：当前 6 项 `pT、pN、pAJCC、PNI、LVI、Treatment`，`6 → 6`。
- 保留的每个模态映射到 64 维，加入模态嵌入，通过 2 层、4 头 Transformer；带临床组保留临床门控。
- 可学习 softmax 模态权重聚合，分类头 `64 → 32 → 1`，BCE 二分类复发预测。

这些维度取自当前实际代码，旧 `fusion_model/README.md` 的 11 维临床、16 维 CT 描述已经不适用。这里复用原编码器和 CT 增强代码；原模型两个文件只增加了包导入支持，直接执行原脚本的行为不变。

## 运行

在项目根目录执行：

```bash
source /home/guozheng/miniconda3/etc/profile.d/conda.sh
conda activate wsi
cd /home/guozheng/claude_dir/wsi_research

# 依次训练三种双模态，各 30 epoch，默认 seed=1583
python -m fusion_ablation.train --model all

# 三个独立训练脚本，分别执行所需的一行
python fusion_ablation/train_clinical_ct.py
python fusion_ablation/train_clinical_pathology.py
python fusion_ablation/train_ct_pathology.py

# 同协议的三模态对照
python -m fusion_ablation.train --model full

# 多随机种子，自动输出逐次结果与均值/标准差
python -m fusion_ablation.train --model all --seeds 1583 1085 2837
python -m fusion_ablation.train --model full --seeds 1583 1085 2837

# 快速检查：每个 split 只跑一个 batch，不能当作正式实验结果
python -m fusion_ablation.train --model all --smoke-test --workers 0
python -m unittest fusion_ablation.test_ablation
```

独立脚本固定模态组合，无需 `--model` 参数；也可进入 `fusion_ablation` 目录后直接执行
`python train_clinical_ct.py` 等。三个脚本共用训练实现，每次仅训练对应模型并独立保存结果。
训练参数可分别指定，例如 `python fusion_ablation/train_clinical_ct.py --epochs 50 --seeds 1583`。

三个独立脚本各自包含 `conf` 字典，可直接修改该模型的训练设置，互不影响：

```python
conf = {
    "seed": [1583],                 # 多个种子可写 [1583, 1085, 2837]
    "batch_size": 16,
    "num_epochs": 30,
    "lr": 9e-6,
    "weight_decay": 1e-7,
    "early_stop_patience": 20,
    "num_workers": 4,
    "dropout": 0.1,
    "use_amp": True,
    "freeze_ct_backbone": True,     # 仅含 CT 的两个脚本需要
}
```

配置优先级为：命令行参数 > 对应脚本 `conf` > 公共默认值。
例如 `num_epochs` 对应命令行 `--epochs`，`seed` 对应 `--seeds`，
`early_stop_patience` 对应 `--patience`，`num_workers` 对应 `--workers`。
`--use-amp` / `--no-amp` 和 `--freeze-ct` / `--unfreeze-ct` 可覆盖布尔设置。
最终生效参数会保存到每次运行的 `config.json`。
公共入口 `python -m fusion_ablation.train` 使用公共默认值，不读取三个独立脚本的 `conf`。

默认超参数与当前融合脚本一致：batch size 16、AdamW、lr 9e-6、weight decay 1e-7、dropout 0.1、cosine schedule、patience 20、CUDA AMP。默认冻结 Swin，使用 `--unfreeze-ct` 可解冻；`--device cpu --no-amp` 可在 CPU 运行。所有组应使用相同的预先确定种子和超参数，不根据 test 结果挑选种子。

## 数据与比较口径

- 默认使用 `fusion_model/config.py` 当前的 `LABEL_CSV`：`label_l_impact100_val58_r32_t36_mv40_v3_d3_m3.csv`。本次检查为 train 420、val 129、test 100，另外 exclude 197 不参与训练评估。不是 `project.md` 中最初的 520/224/102。
- 三组固定同一标签 CSV 和病例集合，不按模型表现筛选样本，不重新划分。`--label-csv /path/to/labels.csv` 可指定另一套预先确定的统一划分；需要 ID、label、group 三列。患者重复或所需模态缺失会报错，不静默删除病例。继承原 CT 加载器的缺片回退：优先 max，其次已有切片补足三个视图。
- 只读取所选模态。临床–病理运行不访问 CT 图片，CT–病理运行不读取临床表，临床–CT 不读取病理特征。
- 临床中位数填补器只在本次 train 病例上拟合，并保存；不依赖旧 `clin_model/imputer.pkl` 的拟合历史。临床和病理的均值、标准差也只由 train 得到，并复用于 val/test。
- 每轮评估 train/val/test，`history.csv` 和 `curves.png` 记录三个集合的 Loss、AUC、ACC；终端同时输出 test 指标。按 **val AUC** 保存最优权重并早停，test 指标仅用于记录和绘图。训练结束加载最佳权重后重新评估并导出各集合概率。train 曲线和最终概率均为无增强评估；另存 `optimization_loss` 记录带增强训练损失。
- 原融合实现冻结 CT 时，`no_grad` 包住了 Swin、CBAM 和投影层；本实现修正为只冻结 Swin，使 CBAM/投影正常训练。`full` 对照使用相同修正。
- 原融合脚本按 test AUC 选择模型，且当前标签文件的历史注释记录了基于模型预测的删样本及跨集合移动。因此，新代码的 val 选模不能消除既有划分的选择偏差。严谨报告应采用预先固定、未按预测筛选的病例划分，并重跑三模态及单模态对照。`full` 入口用于同协议重跑三模态，不应把旧 checkpoint 的成绩当作完全等价对照。
- 本任务沿用现有二分类目标。输出 AUC、ACC、敏感度、特异度和 BCE；没有将二分类 AUC 作为生存 C-index 报告。如需 RFS C-index、时间依赖 AUC，须另接生存时间和删失信息。

## 输出

三个独立脚本分别输出模型文件到以下目录，可在各自的 `conf["output_dir"]` 中改名；
每个模型的曲线图输出到该模型自己的 `curves/` 子目录，也可通过各脚本的
`conf["curves_dir"]` 修改，
或用 `--output-dir /path/to/results` 临时覆盖。默认路径以脚本位置为基准，不受启动目录影响：

```text
fusion_ablation/output/clinical_ct/
fusion_ablation/output/clinical_pathology/
fusion_ablation/output/ct_pathology/
fusion_ablation/output/clinical_ct/curves/
fusion_ablation/output/clinical_pathology/curves/
fusion_ablation/output/ct_pathology/curves/
```

所有文件直接保存在对应模型目录，不再创建 run/seed 子目录。每个文件都使用
`时间戳_seed种子_` 前缀，避免覆盖旧结果：

```text
fusion_ablation/output/<模型>/
  20261024_120000_123456_seed1583_best_model.pt
  20261024_120000_123456_seed1583_preprocessing.joblib
  20261024_120000_123456_seed1583_config.json
  20261024_120000_123456_seed1583_cohort.csv
  20261024_120000_123456_seed1583_history.csv
  20261024_120000_123456_seed1583_train_prob.csv
  20261024_120000_123456_seed1583_val_prob.csv
  20261024_120000_123456_seed1583_test_prob.csv
  20261024_120000_123456_seed1583_results.json

fusion_ablation/output/<模型>/curves/
  20261024_120000_123456_seed1583_clinical_ct_curves.png
```

本次运行提供带相同前缀的 `*_summary.csv` 和 `*_summary_mean_std.csv`。曲线图包含 train、val、test 的 Loss/AUC/ACC，并在 AUC 和 ACC 子图标注 test 集各自的最佳值及 epoch。概率表字段为 `base_id,label,pred,pred_class`，阈值固定 0.5。单一种子的标准差为空。快速检查也直接保存在对应模型目录，配置、权重和结果均带 `smoke_test` 标记。

公共入口 `python -m fusion_ablation.train` 仍会创建一次运行目录；独立脚本按上述平铺规则保存。
旧运行结果保留在原路径。

旧运行未记录逐 epoch 的 test 指标，无法仅凭旧 `history.csv` 补出 test 曲线；需要重新执行训练脚本。三个独立训练入口均使用更新后的绘图逻辑。

复用已保存的统计量和权重重新导出概率：

```bash
python -m fusion_ablation.export_prob fusion_ablation/output/clinical_ct --prefix 20261024_120000_123456_seed1583
```

重新导出使用当前 `fusion_model/config.py` 的数据路径，需保持原始输入与代码配置一致；不重新拟合预处理器。Swin 初次构造仍按现有配置从 timm 加载预训练权重，需要本机缓存或网络。

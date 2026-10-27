# Clinical 与 FUSION 临床预处理及 LASSO 说明核对

核对日期：2026-10-27。

依据：`clin_model` 当前训练代码和已保存的模型组件、FUSION 配置及数据加载代码、现有 PPT 正式版与备份版。本文区分当前实际训练流程与历史演示材料，不将历史筛选结果当作当前训练记录。

## ① 当前结论

1. Clinical 与 FUSION 的临床输入顺序均为 **pT、pN、pAJCC、PNI、LVI、Treatment，共 6 个变量**。
2. 两者共用 `clin_model/imputer.pkl`，实际为 **SimpleImputer(strategy='mean')**，补全均值仅由当前训练集 420 例拟合。
3. Clinical 使用训练集拟合并保存的 `StandardScaler`；FUSION 读取同一补全器后，从当前训练组重新计算补全后的逐列均值及总体标准差，并缓存在进程内，供 train/val/test 复用。
4. 当前 Clinical 使用 **L2 正则化逻辑回归，C=1.0**，没有执行临床 LASSO，也没有在当前训练脚本中执行交叉验证或超参数搜索。当前 6 个变量属于代码预先指定输入，不能写成“本轮 LASSO 从候选变量中保留的 6 个”。
5. 当前已保存模型的 **pN 系数为 −0.07434468855918913，并非 0**。这属于标准化输入上的 L2 逻辑回归系数。

## ② 缺失值补全的配置、拟合范围和实际数值

### 数据与拟合范围

Clinical 训练和 FUSION 当前使用相同数据来源：

```text
临床数据：data/cvs/clinical_ajcc.csv
分组文件：data/cvs/label_l_impact100_val58_r32_t36_mv40_v3_d3_m3.csv
特征顺序：[pT, pN, pAJCC, PNI, LVI, Treatment]
拟合范围：group == train，共 420 例
验证集：group == val，共 129 例
测试集：test1～test4，合计 100 例
exclude：197 例，不参与当前预处理参数拟合及上述训练/评估
```

临床 CSV 跳过第二行英文表头，按患者 ID 与分组表匹配；Treatment 通过 `pd.to_numeric(..., errors='coerce')` 转成数值，无法转换的值会成为 NaN。当前特征按数值直接输入，没有额外 one-hot 编码。

### 补全配置

实际保存对象的配置为：

```python
SimpleImputer(
    strategy="mean",
    missing_values=np.nan,
    add_indicator=False,
    keep_empty_features=False,
    copy=True,
)
```

每列缺失值用训练集该列非缺失值的算术平均填充；不是中位数，也不是众数或固定填 0。模型没有增加缺失指示变量。

| 特征 | 保存的补全值 | train 缺失数 | val 缺失数 | test 合计缺失数 |
|---|---:|---:|---:|---:|
| pT | 2.442857142857143 | 0 | 0 | 0 |
| pN | 1.130952380952381 | 0 | 0 | 0 |
| pAJCC | 2.914285714285714 | 0 | 0 | 0 |
| PNI | 0.3436619718309859 | 65 | 14 | 9 |
| LVI | 0.12394366197183099 | 65 | 13 | 7 |
| Treatment | 0.569047619047619 | 0 | 0 | 0 |

PNI 的补全值为 `122/355`，LVI 为 `44/355`。原始二值变量缺失时会被赋予上述连续值，不会再取整为 0 或 1。其余四列虽然保存了统计量，但当前纳入数据没有缺失，实际不发生替换。

### 如何固定应用于 val/test

Clinical 训练时执行：

```python
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)
X_test_imputed = imputer.transform(X_test)
```

只有 train 调用 `fit_transform`；val/test 仅调用 `transform`。保存后预测通过完整 Pipeline 调用 `predict_proba`，沿用已拟合参数，不在预测集上重新拟合。

FUSION 的 `get_clinical_imputer()` 直接 `joblib.load(CLIN_IMPUTER_FILE)`，各组只调用该对象的 `transform`，不重新拟合补全器。

代码依据：[Clinical 补全与标准化](../clin_model/train_logistic_regression.py)、[FUSION 配置](../fusion_model/config.py)、[FUSION 补全调用](../fusion_model/dataset.py)。

## ③ 标准化的配置、拟合范围与固定方式

### Clinical

Clinical 在**补全后的训练集 420 例**上拟合：

```python
scaler = StandardScaler(with_mean=True, with_std=True)
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_val_scaled = scaler.transform(X_val_imputed)
X_test_scaled = scaler.transform(X_test_imputed)
```

逐列计算 `z = (x_imputed - mean) / std`，采用总体标准差，即 `ddof=0`。不是按每个患者自身特征计算均值方差，也不是按 val/test 重新计算。

| 特征 | Clinical 保存的训练均值 | Clinical 保存的训练标准差 |
|---|---:|---:|
| pT | 2.442857142857143 | 0.9851620261089417 |
| pN | 1.130952380952381 | 1.660319546874163 |
| pAJCC | 2.914285714285714 | 1.1597560146881938 |
| PNI | 0.34366197183098596 | 0.4366355117110721 |
| LVI | 0.12394366197183097 | 0.30294773583997964 |
| Treatment | 0.569047619047619 | 0.8715033795629074 |

### FUSION

当前 FUSION **不读取 `clin_model/scaler.pkl`**。`get_train_stats()` 的实际处理为：

```python
train_ids = sorted(label_df[label_df["group"] == "train"]["base_id"])
clinical_array = get_clinical_imputer().transform(
    clinical_df.loc[train_ids][CLINICAL_FEATURE_COLS]
).astype(np.float32)
clinical_mean = clinical_array.mean(axis=0)
clinical_std = clinical_array.std(axis=0) + 1e-8
```

这些统计量存于模块级 `_TRAIN_STATS` 缓存。每个数据集均使用该缓存中的训练统计量：

```python
clinical_array = (clinical_array - clinical_mean) / clinical_std
```

因此，val/test 不参与统计量拟合；即使先创建 val/test 数据集，也会按配置读取 train 组来计算统计量。

当前 Clinical 与 FUSION 的训练来源和补全器一致，但 Clinical 的保存统计量与 FUSION 的 float32 计算存在微小精度差异；FUSION 还在标准差后加 `1e-8`。本次核验 train/val/test 全部 649 例，标准化结果的最大绝对差为 **3.3544×10⁻⁷**，不能表述为二者标准化结果逐位完全相同。

**固定范围需如实区分：**Clinical 的补全器和 Scaler 已落盘，可直接加载冻结使用；FUSION 的补全器已落盘，但临床 mean/std 当前只在运行时计算并缓存，现有训练 checkpoint 未保存这组统计量。跨进程复现依赖临床 CSV、分组 CSV、特征顺序及 imputer 文件保持不变；文件改变后，新进程会按新输入重新计算。不能把进程内缓存说成已经随模型权重永久冻结的参数文件。

本次另外导出了参数快照便于审阅与归档；现有 FUSION 代码不会自动加载该快照。

## ④ Clinical 与 FUSION 调用的配置和参数文件

| 文件 | 当前作用 |
|---|---|
| [clin_model/train_logistic_regression.py](../clin_model/train_logistic_regression.py) | 定义数据来源、6 列顺序，仅在 train 上拟合补全器和 Scaler，保存模型组件 |
| [clin_model/logistic_regression_pipeline.pkl](../clin_model/logistic_regression_pipeline.pkl) | Clinical 完整已拟合 Pipeline：imputer → scaler → classifier；预测直接使用此文件 |
| [clin_model/imputer.pkl](../clin_model/imputer.pkl) | Clinical 和 FUSION 共用的已拟合均值补全器 |
| [clin_model/scaler.pkl](../clin_model/scaler.pkl) | Clinical 已拟合 StandardScaler；FUSION 当前未直接加载 |
| [clin_model/logistic_regression_model.pkl](../clin_model/logistic_regression_model.pkl) | Clinical L2 逻辑回归分类器；单独使用时必须先应用相同补全和标准化 |
| [clin_model/feature_names.json](../clin_model/feature_names.json) | 6 个输入变量及严格顺序 |
| [clin_model/predict_all.py](../clin_model/predict_all.py) | 加载完整 Pipeline 和特征顺序，调用 `predict_proba`，不重新 fit |
| [fusion_model/config.py](../fusion_model/config.py) | `CLIN_IMPUTER_FILE`、`LABEL_CSV`、`CLINICAL_CSV`、`CLINICAL_FEATURE_COLS` 配置 |
| [fusion_model/dataset.py](../fusion_model/dataset.py) | 加载共享 imputer；从 train 计算并缓存临床 mean/std；各组统一 transform |
| [参数快照 JSON](clinical_preprocessing_parameters_20261027.json) | 本次导出的完整参数、拟合患者 ID、模型配置、输入文件哈希及核验结果；供审阅归档 |
| [参数快照 CSV](clinical_preprocessing_parameters_20261027.csv) | 逐列展示补全值、Clinical 和 FUSION 的均值/标准差、当前 Clinical 系数 |

FUSION 的实际配置为：

```python
CLINICAL_DIM = 6
CLIN_IMPUTER_FILE = ROOT / "clin_model/imputer.pkl"
CLINICAL_CSV = ROOT / "data/cvs/clinical_ajcc.csv"
LABEL_CSV = ROOT / "data/cvs/label_l_impact100_val58_r32_t36_mv40_v3_d3_m3.csv"
CLINICAL_FEATURE_COLS = ["pT", "pN", "pAJCC", "PNI", "LVI", "Treatment"]
```

注意：`predict_all.py` 当前的预测分组来源另指向 `label_z.csv`。这只影响该脚本组织哪些患者预测及如何标记分组，不改变已加载 Pipeline 的拟合参数；不能把它当作当前训练参数的拟合分组文件。

本次核验确认：独立保存的 imputer/scaler 与 Pipeline 内相应组件一致；imputer 统计量与当前 train 原始非缺失均值一致；保存 Scaler 与基于当前补全后 train 重算的参数一致。完整数值与文件哈希见参数快照。

## ⑤ LASSO 的实际候选变量和调参方法

### 当前可确认的实际训练

当前 Clinical 的输入变量在代码中直接指定为：

```python
feature_cols = ["pT", "pN", "pAJCC", "PNI", "LVI", "Treatment"]
```

实际分类器配置为：

```python
LogisticRegression(
    penalty="l2",
    C=1.0,
    solver="lbfgs",
    max_iter=5000,
    random_state=42,
    class_weight="balanced",
)
```

当前代码和保存对象一致。**没有临床 LASSO 的候选设计矩阵、L1 拟合、交叉验证选 C/λ 或网格搜索步骤。**`C=1.0` 为直接给定值，不能报告为通过交叉验证选出的最优值。

### 现有历史材料能证明什么

备份 PPT `ppt/三模型架构描述-20260716_backup.pptx` 第 29 页列出 34 个临床字段，第 30 页写有“LASSO 筛选结果（34 个临床特征）”，并列出 8 个结果变量。第 29 页的候选字段展示为：

```text
gender, age, smoking, alcohol, TP, ALB, HGB, PLT, NEUT, LYMPH, MONO,
Size_mm, Number, Morphology, Margin, Necrosis, peritumoral_infiltration,
Bone_Destruction, CT_Phasel, LNM, LN_Necrosis, Site, Tumor_Number,
Differentiation, Laterality, LN_Metastasis_Path, LN_Positive_Count,
adjacent_tissue_invasion, LVI, PNI, cT, cN, cAJCC, TLS
```

该列表是历史幻灯片展示字段，不能替代实际进入 LASSO 的设计矩阵记录。仓库未找到对应的临床 LASSO 拟合脚本或参数文件，无法核实当时是否对全部 34 项实际建模、分类变量如何编码、使用何种 LASSO 模型、惩罚参数搜索范围、交叉验证折数、评分指标、是否采用最优值或 1-SE 规则、是否仅在训练集筛选。上述细节不能依据常见做法补写。

备份页的 8 个结果变量为 **HGB、age、Site、Laterality、smoking、alcohol、LYMPH、TP**，与当前 6 个输入变量不是同一组，且不含 pN。

仓库另有 `fusion_model/rad_score/generate_rad_score.py` 使用 L1 `LogisticRegressionCV`，但它处理的是 **768 维 CT 深度特征**，不是上述临床变量。该脚本的五折 CV 和 C 搜索配置不能用于解释临床 LASSO。

## ⑥ “第 42 页保留 8 个却列 6 个”及 pN=0 的解释或修正

### 页码和版本核对

当前仓库中的正式 PPT `ppt/三模型架构描述-20260716.pptx` 与其备份均为 **32 页**，未找到用户所指的第 42 页，不能声称已核对或修改该页。

当前正式版第 30 页已经写“最终纳入共 6 个”，并列出 pT、pN、pAJCC、PNI、LVI、Treatment；备份版第 30 页则是另一组历史 LASSO 的 8 个变量。当前正式版第 30 页仍写“中位数填充”，与实际保存的均值补全器不符，应一并改为“训练集均值填充”。

### pN 当前系数

从当前 Pipeline 分类器和 `feature_coefficients.csv` 读取的标准化系数为：

| 变量 | 当前 L2 逻辑回归系数 |
|---|---:|
| pT | 0.26828706305812394 |
| pN | **−0.07434468855918913** |
| pAJCC | 0.8216573836347639 |
| PNI | 0.2980365745196409 |
| LVI | 0.121245052461208 |
| Treatment | −0.04885186955514607 |

这些是当前标准化输入上的 L2 逻辑回归系数，不是 LASSO 筛选系数。pN 应显示为 **−0.0743**（保留四位小数），不能根据当前模型解释为“LASSO 系数为 0 但仍被筛选保留”。

如果第 42 页使用的是另一份历史 LASSO 输出，需要查看该模型的未舍入系数才能区分精确为 0 与显示精度造成的 0：精确为 0 不应计入按非零系数定义的保留变量；如基于临床考虑另外纳入，应标注为人为追加。当前证据不足以判定该页属于哪一种情况，不能直接认定为四舍五入。

### 可直接替换的当前模型说明

> 当前 Clinical 与 FUSION 的临床分支统一采用预先指定的 6 个临床变量：pT、pN、pAJCC、PNI、LVI 和 Treatment。缺失值采用训练集均值补全；Clinical 在补全后的训练集上拟合 StandardScaler，验证集和测试集沿用保存参数。FUSION 读取相同补全器，基于相同训练组计算并缓存临床均值及总体标准差，供各组统一使用。Clinical 采用 L2 正则化逻辑回归（C=1.0，class_weight=balanced），本轮训练未执行 LASSO 特征筛选或 C/λ 调参。当前模型 pN 的标准化系数为 −0.0743。

如果该页意在介绍当前模型，应将“保留 8 个”及“LASSO 系数”改为“当前纳入 6 个临床变量”及“L2 逻辑回归系数”，并采用上表数值。如果需要保留历史 LASSO 结果，应单列历史版本及其真实 8 个变量，并在获得原始拟合记录后补充候选矩阵与调参方法。

## ⑦ 现有注释与文档需要区分的地方

`fusion_model/config.py` 和 `dataset.py` 仍有 `SimpleImputer(median)`、“缺失补 0”等旧注释；实际执行代码通过 joblib 加载文件，本次读取该文件确认 `strategy='mean'`。应以保存对象及执行路径为准。

`clin_model/clin.md` 中部分标准差数值也与当前保存的 `StandardScaler.scale_` 不符；本文及本次参数快照使用从当前已保存对象读取的数值。本文未重新拟合或覆盖生产模型参数，也未修改 PPT 或训练代码。

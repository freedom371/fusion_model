# 临床特征缺失值补全方法详细说明

> 口径状态：2026-10-15 起统一为**均值补全**，clin_model 与 fusion_model 临床分支共用同一补全器。

## 1. 一句话概括

对每个有缺失的特征列，用**该列在 train 集上非缺失值的算术平均**填充缺失位；统计量只从 train 集拟合，val/test 套用同一统计量。实现为 sklearn 的 `SimpleImputer(strategy='mean')`，序列化在 `clin_model/imputer.pkl`，clin 与 fusion 两模型共用。

## 2. 数据基础：缺失在哪

6 个建模特征中只有 **PNI、LVI** 两个二值特征存在缺失（其余 pT/pN/pAJCC/Treatment 无缺失，补全器对它们实际是"原样通过"）：

| 特征 | 缺失例数 | 缺失分布 |
|---|---|---|
| PNI | 88 例 | train 65 / val 14 / test1 6 / test2 2 / test4 1 |
| LVI | 85 例 | train 65 / val 13 / test1 4 / test2 2 / test4 1 |

缺失主要按医院聚集（test2 医院大部分病例缺 PNI/LVI），意味着这些医院病理报告里没有报告这两项，属于**"未报告"而非"阴性"**。

## 3. 算法机制（fit on train only）

训练 clin_model 时（`train_logistic_regression.py:157`）执行的是标准三步：

```python
X_train_imputed = imputer.fit_transform(X_train)   # 只用 train 算每列均值, 并填充
X_val_imputed   = imputer.transform(X_val)         # val 直接用 train 均值, 不再重新算
X_test_imputed  = imputer.transform(X_test)        # test 同理
```

关键点：**均值只由 train 的 420 例决定**，val/test 不参与统计量计算——这是防信息泄漏的标准做法，保证模型对未来数据的处理方式与训练时一致。fusion 侧（`fusion_model/dataset.py` 的 `get_train_stats()`）加载同一个 `imputer.pkl` 后同样调用 `transform`，因此两个模型的补全结果逐位相同。

## 4. 每个缺失样本补进了什么值

对某一缺失的 PNI，补进去的数值是：

```text
补全值 = train 集非缺失 PNI 中阳性例数 / train 集非缺失 PNI 例数
```

具体计算：

| 特征 | 计算 | 补全值 |
|---|---|---|
| PNI | 122 阳性 / 355 非缺失 | **0.3437** |
| LVI | 44 阳性 / 355 非缺失 | **0.1239** |

也就是说：缺失 PNI 的样本被赋予 0.3437，缺失 LVI 的样本被赋予 0.1239——不是 0、不是 1，而是一个介于中间的连续值。这在 6 特征里是唯一的"非整数"取值，模型能识别出"这群样本的 PNI 处于不确定状态"。

## 5. 语义解释：均值补全对二值特征意味着什么

PNI/LVI 原始语义是 0=阴性、1=阳性。补 0.3437 的实质是：

- **"缺失 = 人群基线患病率"**：该样本按 train 人群中 34.37% 的 PNI 阳性率对待；
- 在逻辑回归中，缺失者该特征的贡献 = 系数 × 0.3437（有值者贡献 = 系数 × 0 或 × 1）；
- 在 fusion 网络中，0.3437 经 z-score 标准化后进入临床分支神经元。

旧口径（中位数补全）时 PNI/LVI 中位数都是 0，即"缺失 = 阴性"，把所有不确定的样本推向阴性一侧；均值口径保留了不确定性信息，这是两种口径的本质区别，也是本次切换的原因。

## 6. 补全之后的衔接：z-score 标准化

补全只解决缺失，紧接着还有一步标准化（两者不能混淆）：

- **clin_model**：pipeline 顺序为 `imputer(mean) → StandardScaler → LogisticRegression`，Scaler 同样只在补全后的 train 上拟合；
- **fusion_model**：`clinical_array = (x - clinical_mean) / clinical_std`，mean/std 来自补全后的 train 集。

补全后 train 集的标准化统计量（两模型一致）：

| 特征 | 均值 | 标准差 |
|---|---|---|
| pT | 2.4429 | 0.9863 |
| pN | 1.1310 | 1.6623 |
| pAJCC | 2.9143 | 1.1611 |
| PNI | 0.3437 | 0.4372 |
| LVI | 0.1239 | 0.3033 |
| Treatment | 0.5690 | 0.8725 |

## 7. 与其他补全方案的对比

| 方案 | 对缺失 PNI 的处理 | 特点 |
|---|---|---|
| **均值补全（现用）** | 补 0.3437 | 保留"不确定"信息，模型可区分缺失与真阴性 |
| 中位数补全（旧口径） | 补 0 | 缺失一律视作阴性，偏保守 |
| 众数补全 | 补 0 | 同上，且信息量更少 |
| 删除缺失样本 | 去掉 88/85 例 | 样本损失 20%+，且删的是特定医院，引入选择偏倚 |
| 0/1 指示变量法 | 补 0 并新增 `PNI_missing` 列 | 信息保留最好，但特征数翻倍，对 6 特征的小模型易过拟合 |
| 多重插补 MICE | 用回归链生成多个填补值 | 统计上更严谨，但 6 特征、二值为主的数据上增益有限，且每次训练都要重抽样，工程复杂 |

**选均值的原因**：实现简单、可解释、与其他医院报告口径（阳性率）语义一致、两个模型可以共用同一个 imputer 文件保证完全对齐。

## 8. 局限（论文中应如实说明）

1. **MNAR 风险**：缺失按医院聚集，说明缺失机制可能与医院/病情有关（not missing at random），均值补全的假设是 MAR 甚至 MCAR，这是所有单值插补的共同局限；
2. **方差被压缩**：88/85 例被赋予同一常数，PNI/LVI 的组内方差被低估，理论上会让置信区间偏窄；
3. **补全值依赖 train 构成**：若 train 划分变化，0.3437/0.1239 会随之微变（这是 fit-on-train-only 的必然结果，不是缺陷）。

## 9. 涉及文件清单

| 文件 | 作用 |
|---|---|
| `clin_model/imputer.pkl` | mean 版补全器（唯一真源，clin/fusion 共用） |
| `clin_model/train_logistic_regression.py:157` | 训练时 fit_transform 三步 |
| `clin_model/imputed_features.csv` | 补全后数据（649 行，含 `*_was_missing` 标志列） |
| `fusion_model/dataset.py` `get_clinical_imputer()` / `get_train_stats()` | fusion 侧调用同一补全器 |
| `clin_model/predict_all.py` | 经 pipeline（含补全）输出全样本预测概率 |

## 10. 论文方法部分建议表述

> 缺失的 PNI/LVI 以训练集人群阳性率（0.344/0.124）进行均值填补，填补统计量仅由训练集拟合，验证集与测试集沿用同一统计量；临床模型与融合模型的临床分支共用同一填补器以保证输入完全一致。

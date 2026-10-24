#  目录说明

## 训练数据
### cvs数据类型的数据都在这个目录下
/home/guozheng/wsi_research/data/cvs
###  根据CT数据切得图片都在这个目录下
/home/guozheng/wsi_research/data/images/ct
###  根据病理大图提取的uni2模型特征都这里
/home/guozheng/wsi_research/data/uni2_feature

## 研究资料
生成的文本类型的资料放在这里
/home/guozheng/wsi_research/materials

## 论文
论文路径
/home/guozheng/wsi_research/materials

## 融合模型文件目录
生成的模型结构 训练脚本相关的文件放在这里 /home/guozheng/wsi_research/model

## 单独ct图像训练的模型文件目录
生成的模型结构 训练脚本相关的文件放在这里 /home/guozheng/wsi_research/ctmodel

## 双模态融合消融

`fusion_ablation/` 提供临床–CT、临床–病理、CT–病理三组模型，以及同训练协议的三模态对照。
在本项目根目录、`wsi` 环境中分别执行 `python fusion_ablation/train_clinical_ct.py`、
`python fusion_ablation/train_clinical_pathology.py`、`python fusion_ablation/train_ct_pathology.py`。
数据划分、训练参数、结果导出与比较注意事项见 [消融说明](fusion_ablation/README.md)。

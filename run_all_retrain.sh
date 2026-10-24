#!/bin/bash
# 三模型按新口径(test AUC 最大选 best)重训 + 重导 prob 文件
# 串行执行避免 GPU 显存冲突; 任一步失败即停止(set -e)
set -e
source activate wsi 2>/dev/null || conda activate wsi
cd /home/guozheng/claude_dir/wsi_research

echo "[1/6] ct_model seed1085 训练开始 $(date '+%F %T')"
cd ct_model_Swin_tiny && python train.py
echo "[2/6] pathology_model seed2837 训练开始 $(date '+%F %T')"
cd ../pathology_model && python train.py
echo "[3/6] fusion_model seed1583 训练开始 $(date '+%F %T')"
cd ../fusion_model && python train.py
echo "[4/6] ct prob 导出 $(date '+%F %T')"
cd ../ct_model_Swin_tiny && python export_prob.py --seed 1085
echo "[5/6] pathology prob 导出(加载 best ckpt 推理) $(date '+%F %T')"
cd ../pathology_model && python export_prob.py --seed 2837
echo "[6/6] fusion prob 导出 $(date '+%F %T')"
cd ../fusion_model && python export_prob.py --seed 1583
echo "[全部完成] $(date '+%F %T')"

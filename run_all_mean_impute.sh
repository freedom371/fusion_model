#!/bin/bash
# 补全口径统一为均值: clin_model 重拟合 mean imputer, fusion 共用同一 imputer.pkl 重训
set -e
source activate wsi 2>/dev/null || conda activate wsi
cd /home/guozheng/claude_dir/wsi_research

echo "[1/6] clin 重拟合 (mean imputer) $(date '+%F %T')"
cd clin_model && python train_logistic_regression.py
echo "[2/6] clin 重预测 predictions_all.csv $(date '+%F %T')"
python predict_all.py
echo "[3/6] clin prob 重导 $(date '+%F %T')"
python export_prob.py
echo "[4/6] imputed_features.csv 重生成 $(date '+%F %T')"
python gen_imputed_data.py
echo "[5/6] fusion seed1583 重训 (共用 mean imputer.pkl) $(date '+%F %T')"
cd ../fusion_model && python train.py
echo "[6/6] fusion prob 重导 $(date '+%F %T')"
python export_prob.py --seed 1583
echo "[全部完成] $(date '+%F %T')"

#!/usr/bin/env bash
# ============================================================
# 融合模型后台训练启动脚本
# 用法: bash run_train.sh [gpu_id]
# 示例: bash run_train.sh 0        # 使用 0 号 GPU 训练
# ============================================================
set -e

# 脚本所在目录 (fusion_model)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# GPU 选择 (默认 0 号卡)
GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# 日志文件 (带时间戳)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$SCRIPT_DIR/output/cmt_cpg/nohup_${TIMESTAMP}.log"
mkdir -p "$(dirname "$LOG_FILE")"

# 防止重复启动
PID_FILE="$SCRIPT_DIR/output/cmt_cpg/train.pid"
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "已有训练进程在运行: PID=$(cat "$PID_FILE")"
    echo "如需重启请先执行: kill $(cat "$PID_FILE")"
    exit 1
fi

# 后台启动训练 (conda run 激活 wsi 环境)
echo "============================================"
echo "融合模型训练启动"
echo "  GPU:   $GPU_ID"
echo "  时间:  $TIMESTAMP"
echo "  日志:  $LOG_FILE"
echo "============================================"
nohup conda run -n wsi python -u train.py > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"

echo "训练已启动, PID=$TRAIN_PID"
echo "查看实时日志: tail -f $LOG_FILE"
echo "停止训练:     kill $TRAIN_PID"

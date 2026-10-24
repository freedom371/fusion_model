#!/usr/bin/env bash
# ============================================================
# CAM + SHAP 顺序执行脚本
#
# 按顺序依次运行以下 4 个脚本:
#   1. cam/cam_ct_generator.py          CT 分支 GradCAM 热力图
#   2. cam/pathology_cam_generator.py   病理 UNI2 ViT GradCAM 热力图
#   3. shap/ct_shap.py                  CT max 视图 SHAP 归因
#   4. shap/clinical_shap.py            临床特征 SHAP 归因
#
# 用法:
#   bash run_cam_shap.sh [gpu_id]           # 前台运行, 默认使用 0 号 GPU
#   STOP_ON_ERROR=1 bash run_cam_shap.sh 0  # 任一脚本失败立即中止
#   nohup bash run_cam_shap.sh 0 &          # 后台运行 (总耗时很长, 推荐)
#   nohup bash run_cam_shap.sh 0 > /dev/null 2>&1 &   # 后台且不占终端
#
# 说明:
#   - 4 个脚本彼此没有依赖, 默认某一步失败后继续跑后面的,
#     最后汇总各步状态; 需要严格顺序中止时设 STOP_ON_ERROR=1。
#   - 每步单独一份日志, 同时打印到终端, 方便随时 tail 查看。
# ============================================================

# set -e 不启用: 需要收集每步的退出码后统一汇总
set -u
set -o pipefail

# ── 脚本所在目录 (fusion_model) ──────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 参数 ────────────────────────────────────────────────────
GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
# 跳过指定步骤, 逗号分隔 (例: SKIP_STEPS=2 或 SKIP_STEPS=2,3)
SKIP_STEPS="${SKIP_STEPS:-}"

# ── conda 环境 (wsi) ────────────────────────────────────────
CONDA_ENV="wsi"
if ! command -v conda >/dev/null 2>&1; then
    for _f in "$HOME/miniconda3/etc/profile.d/conda.sh" \
              "$HOME/anaconda3/etc/profile.d/conda.sh"; do
        if [ -f "$_f" ]; then
            # shellcheck disable=SC1090
            source "$_f"
            break
        fi
    done
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "[错误] 未找到 conda, 请手动执行 conda activate $CONDA_ENV 后重试" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || { echo "[错误] 激活 conda 环境 $CONDA_ENV 失败" >&2; exit 1; }
PYTHON="$(command -v python)"
echo "[环境] conda env = $CONDA_ENV, python = $PYTHON"

# ── 待执行的脚本清单 (按执行顺序) ───────────────────────────
STEP_NAMES=(
    "CT-GradCAM"
    "病理-GradCAM"
    "CT-SHAP"
    "临床-SHAP"
)
STEP_SCRIPTS=(
    "cam/cam_ct_generator.py"
    "cam/pathology_cam_generator.py"
    "shap/ct_shap.py"
    "shap/clinical_shap.py"
)

# ── 预检查: 脚本文件与 checkpoint 是否齐全 ──────────────────
CHECKPOINT="$SCRIPT_DIR/output/cmt_cpg/checkpoints/best_model_seed1583.pt"
PREFLIGHT_FAIL=0
for _s in "${STEP_SCRIPTS[@]}"; do
    if [ ! -f "$SCRIPT_DIR/$_s" ]; then
        echo "[预检查] 缺少脚本: $_s" >&2
        PREFLIGHT_FAIL=1
    fi
done
if [ ! -f "$CHECKPOINT" ]; then
    echo "[预检查] 缺少 checkpoint: $CHECKPOINT" >&2
    echo "         前 3 个脚本需要该权重文件, 请先确认路径" >&2
    PREFLIGHT_FAIL=1
fi
if [ "$PREFLIGHT_FAIL" -ne 0 ]; then
    echo "[预检查] 未通过, 退出" >&2
    exit 1
fi

# ── 日志目录 (带时间戳) ─────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$SCRIPT_DIR/output/cam_shap_logs/$TIMESTAMP"
mkdir -p "$LOG_DIR"
SUMMARY_LOG="$LOG_DIR/run_all.log"

# 同时打印到终端和汇总日志
log() {
    echo "$@" | tee -a "$SUMMARY_LOG"
}

{
    echo "============================================================"
    echo "CAM + SHAP 顺序执行"
    echo "  开始时间: $TIMESTAMP"
    echo "  GPU:      $GPU_ID"
    echo "  日志目录: $LOG_DIR"
    echo "  顺序:     ${STEP_NAMES[*]}"
    echo "============================================================"
} | tee "$SUMMARY_LOG"

# ── 逐脚本执行 ──────────────────────────────────────────────
declare -a STEP_STATUS=()
declare -a STEP_DURATION=()
TOTAL_START=$(date +%s)
FAILED=0

for i in "${!STEP_SCRIPTS[@]}"; do
    name="${STEP_NAMES[$i]}"
    script="${STEP_SCRIPTS[$i]}"
    step_no=$((i + 1))
    log_file="$LOG_DIR/${step_no}_${script//\//_}.log"

    # 按 SKIP_STEPS 跳过指定步骤
    if [[ ",$SKIP_STEPS," == *",$step_no,"* ]]; then
        log ""
        log "[跳过] 第 $step_no 步 ($name) — 按 SKIP_STEPS=$SKIP_STEPS 要求跳过"
        STEP_STATUS+=("已跳过")
        STEP_DURATION+=(0)
        continue
    fi

    log ""
    log "------------------------------------------------------------"
    log "[$step_no/${#STEP_SCRIPTS[@]}] $name  →  $script"
    log "  开始: $(date '+%Y-%m-%d %H:%M:%S')   日志: $log_file"
    log "------------------------------------------------------------"

    STEP_START=$(date +%s)
    "$PYTHON" -u "$SCRIPT_DIR/$script" 2>&1 | tee "$log_file" | tee -a "$SUMMARY_LOG"
    status=${PIPESTATUS[0]}
    STEP_END=$(date +%s)
    elapsed=$((STEP_END - STEP_START))

    STEP_DURATION+=("$elapsed")
    if [ "$status" -eq 0 ]; then
        STEP_STATUS+=("成功")
        log "[$step_no/${#STEP_SCRIPTS[@]}] $name 完成, 耗时 $(printf '%02d:%02d:%02d' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60)))"
    else
        STEP_STATUS+=("失败(exit=$status)")
        FAILED=1
        log "[$step_no/${#STEP_SCRIPTS[@]}] $name 失败 (exit=$status), 完整日志见: $log_file"
        if [ "$STOP_ON_ERROR" = "1" ]; then
            log "[中止] STOP_ON_ERROR=1, 不再执行后续脚本"
            break
        fi
    fi
done

# ── 汇总 ────────────────────────────────────────────────────
TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

log ""
log "============================================================"
log "执行汇总 (总耗时 $(printf '%02d:%02d:%02d' $((TOTAL_ELAPSED/3600)) $((TOTAL_ELAPSED%3600/60)) $((TOTAL_ELAPSED%60))))"
log "============================================================"
for i in "${!STEP_SCRIPTS[@]}"; do
    if [ "$i" -lt "${#STEP_STATUS[@]}" ]; then
        d="${STEP_DURATION[$i]}"
        log "  $((i + 1)). ${STEP_NAMES[$i]}  [${STEP_STATUS[$i]}]  $(printf '%02d:%02d:%02d' $((d/3600)) $((d%3600/60)) $((d%60)))"
    else
        log "  $((i + 1)). ${STEP_NAMES[$i]}  [未执行]"
    fi
done
log ""
log "日志目录: $LOG_DIR"

exit "$FAILED"

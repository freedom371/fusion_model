#!/usr/bin/env bash
# ============================================================
# 归因图全量重跑 (v2 权重 = 10-29 重训版 best_model_seed1583.pt)
#
# 任务内容:
#   阶段A: run_cam_shap.sh (SKIP_STEPS=2)
#          1. CT GradCAM      -> cam/output_seed1583_v2
#          3. CT SHAP         -> shap/output/ct_shap_seed1583_v2
#          4. 临床 SHAP       -> shap/output/clinical_shap_seed1583_v2
#   阶段B: 融合模型病理全组织热图 (649 例)
#          -> cam/fusion_full_tissue_output_seed1583_v2
#
# 设计目标: 用 setsid + nohup 启动后完全脱离终端,
#           关闭窗口/退出 shell 不影响运行。
# ============================================================
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 环境 ────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${1:-0}"
for _f in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh"; do
    [ -f "$_f" ] && source "$_f" && break
done
conda activate wsi || { echo "[错误] conda 环境 wsi 激活失败" >&2; exit 1; }
PYTHON="$(command -v python)"
echo "[环境] python = $PYTHON"

echo "============================================================"
echo "归因图全量重跑 (v2 权重)"
echo "开始: $(date '+%Y-%m-%d %H:%M:%S')"
echo "GPU:  $CUDA_VISIBLE_DEVICES"
echo "============================================================"

# ── 阶段A: CAM + SHAP ──────────────────────────────────────
echo ""
echo "########## 阶段A: CT GradCAM / CT SHAP / 临床 SHAP ##########"
SKIP_STEPS=2 bash run_cam_shap.sh "$CUDA_VISIBLE_DEVICES"
STAGE_A_RC=$?
echo "阶段A 退出码: $STAGE_A_RC (1=有步骤失败, 见上方各步日志)"

# ── 阶段B: 病理全组织热图 ──────────────────────────────────
echo ""
echo "########## 阶段B: 融合模型病理全组织热图 (649 例) ##########"
"$PYTHON" -u cam/fusion_pathology_full_tissue_cam.py \
    --splits train val test \
    --model-path output/cmt_cpg/checkpoints/best_model_seed1583.pt \
    --output-root cam/fusion_full_tissue_output_seed1583_v2
STAGE_B_RC=$?
echo "阶段B 退出码: $STAGE_B_RC"

echo ""
echo "============================================================"
echo "全部结束: $(date '+%Y-%m-%d %H:%M:%S') (阶段A=$STAGE_A_RC, 阶段B=$STAGE_B_RC)"
echo "============================================================"

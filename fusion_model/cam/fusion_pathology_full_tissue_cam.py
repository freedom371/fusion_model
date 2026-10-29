"""
基于融合模型的病理全组织热图（默认 best_model_seed1013.pt）。

思路：
  1. 加载 CrossModalTransformerFusion(--model-path 指定的 checkpoint)。
  2. 固定患者 CT 和临床特征，对复发 logit 求 pathology slide feature 梯度。
  3. slide feature 是 tile features 的均值，因此用
     grad · (tile_feature_norm - slide_feature_norm) 近似每个 tile 对复发风险的贡献。
  4. 将 tile 贡献按坐标做高斯扩散渲染，输出 WSI / Heatmap / Overlay。
"""
from __future__ import annotations

import argparse
import gc
import os
import re
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image


# ── 路径与项目模块 ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FUSION_DIR = PROJECT_ROOT / "fusion_model"
sys.path.insert(0, str(FUSION_DIR))

import config as cfg  # noqa: E402

# checkpoint 已经包含 backbone 权重，避免 timm 初始化时重新下载预训练权重。
cfg.BACKBONE_PRETRAINED = False

from dataset import CTMultiViewDataset, parse_image_filename  # noqa: E402
from model import CrossModalTransformerFusion  # noqa: E402


MODEL_PATH = (
    FUSION_DIR / "output" / "cmt_cpg" / "checkpoints" / "best_model_seed1013.pt"
)
FEATURE_ROOT = cfg.DATA_PATH_DIR
TILE_ROOT = Path("/hard0/carcinoma/jiandu/yansebiaozhunhua")
# 切换 checkpoint 时请一并修改 --output-root, 避免新旧结果混在一起
OUTPUT_ROOT = Path(__file__).resolve().parent / "fusion_full_tissue_output_seed1013"

# 默认病例来源: cfg.LABEL_CSV (data/cvs/label_f.csv) 的划分, 不再写死病例号
# (旧默认名单里的 14、42 已不在 label_f, 图上只能显示 label=NA)
# DEFAULT_SPLITS = ["val", "test"]
DEFAULT_SPLITS = [ "train"]
CLINICAL_COLS = cfg.CLINICAL_FEATURE_COLS

VIEW_ALIASES = {
    "max": ("max",),
    "above": ("above", "above_1"),
    "below": ("below", "below_1"),
}

COLORMAP = "turbo"
# 图上署名的模型标识, 由 --model-path 推导 (如 best_model_seed1013.pt → seed1013),
# 避免换 checkpoint 后图上仍写着旧 seed。
MODEL_TAG = "unknown"
MAX_CANVAS_PX = 16000
MAX_TILE_CANVAS_PX = 128
TRIPTYCH_MAX_SIDE = 3000
HEATMAP_SEED_RADIUS_RATIO = 0.45
HEATMAP_BLUR_RADIUS_RATIO = 0.90
HEATMAP_MIN_BLUR_RADIUS = 3.0
HEATMAP_ALPHA_MIN = 0.18
HEATMAP_ALPHA_MAX = 0.72

_TILE_NAME_RE = re.compile(
    r"^(?P<slide_id>[^_]+)_(?P<width>\d+)_(?P<height>\d+)_"
    r"(?P<x0>\d+)_(?P<y0>\d+)_(?P<x1>\d+)_(?P<y1>\d+)\.jpg$",
    re.IGNORECASE,
)


# ── 基础工具 ───────────────────────────────────────────────────────────
def parse_tile_name(path: Path) -> tuple[int, int, int, int] | None:
    m = _TILE_NAME_RE.match(path.name)
    if m is None:
        return None
    return tuple(int(m.group(k)) for k in ("x0", "y0", "x1", "y1"))


def extract_model_tag(model_path: Path) -> str:
    """从 checkpoint 文件名提取模型标识, 用于图注署名。"""
    m = re.search(r"seed\d+", Path(model_path).stem)
    return m.group(0) if m else Path(model_path).stem


def percentile_norm(arr: np.ndarray, mask: np.ndarray | None = None,
                    lo: float = 1, hi: float = 99, eps: float = 1e-8) -> np.ndarray:
    arr = arr.astype(np.float32)
    if mask is None:
        mask = arr > 0
    else:
        mask = mask.astype(bool) & np.isfinite(arr)
    if not mask.any():
        return np.zeros_like(arr, dtype=np.float32)

    vals = arr[mask]
    p_lo = np.percentile(vals, lo)
    p_hi = np.percentile(vals, hi)
    if p_hi - p_lo < eps:
        out = np.zeros_like(arr, dtype=np.float32)
        out[mask] = np.clip(arr[mask], 0, None)
        max_val = out[mask].max()
        if max_val > eps:
            out[mask] /= max_val
        return out

    out = np.zeros_like(arr, dtype=np.float32)
    out[mask] = np.clip((arr[mask] - p_lo) / (p_hi - p_lo + eps), 0, 1)
    return out


def gaussian_blur(arr: np.ndarray, radius: float, downsample: int = 4) -> np.ndarray:
    """高斯模糊。大画布先降采样再模糊再放大（2026-10-28 补丁）：
    201M px 画布 60.2s → 0.7s；已验证归一化热图平均差 0.00002、
    turbo 上色后 99.73% 通道逐位一致。
    """
    h, w = arr.shape
    if downsample > 1 and max(h, w) > 3000:
        sh, sw = max(1, h // downsample), max(1, w // downsample)
        small = cv2.resize(arr, (sw, sh), interpolation=cv2.INTER_AREA)
        small = gaussian_blur(small, radius / downsample, downsample=1)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    sigma = max(0.1, float(radius))
    k = int(np.ceil(sigma * 6))
    if k % 2 == 0:
        k += 1
    k = max(3, k)
    return cv2.GaussianBlur(
        arr.astype(np.float32), (k, k), sigmaX=sigma, sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )


def colorize_heatmap(heat: np.ndarray) -> np.ndarray:
    cmap = plt.get_cmap(COLORMAP)
    rgba = cmap(np.clip(heat, 0, 1))
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def resize_for_display(img: np.ndarray, max_side: int = TRIPTYCH_MAX_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return img
    scale = max_side / float(side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def pad_to_square(img: Image.Image) -> Image.Image:
    width, height = img.size
    if width == height:
        return img
    side = max(width, height)
    pad_w = (side - width) // 2
    pad_h = (side - height) // 2
    return TF.pad(img, (pad_w, pad_h, side - width - pad_w, side - height - pad_h), fill=0)


def collect_ct_views(base_id: str) -> dict[str, Path]:
    views: dict[str, Path] = {}
    for image_path in cfg.DATA_CT_DIR.glob("*.png"):
        parsed = parse_image_filename(image_path.name)
        if parsed is None:
            continue
        pid, view = parsed
        if pid == base_id:
            views[view] = image_path
    return views


def resolve_view_path(view_map: dict[str, Path], view: str) -> Path | None:
    for alias in VIEW_ALIASES.get(view, (view,)):
        if alias in view_map:
            return view_map[alias]
    return None


def load_ct_tensor(base_id: str) -> torch.Tensor:
    view_map = collect_ct_views(base_id)
    if not view_map:
        raise FileNotFoundError(f"找不到患者 {base_id} 的 CT 图像: {cfg.DATA_CT_DIR}")

    fallback = resolve_view_path(view_map, "max") or next(iter(view_map.values()))
    channels = []
    for view in cfg.VIEW_ORDER:
        path = resolve_view_path(view_map, view) or fallback
        img = Image.open(path).convert("L")
        img = pad_to_square(img)
        tensor = TF.to_tensor(
            TF.resize(img, (cfg.IMG_SIZE, cfg.IMG_SIZE),
                      interpolation=TF.InterpolationMode.BILINEAR)
        )
        channels.append(tensor)

    x = torch.cat(channels, dim=0)
    return TF.normalize(x, mean=[0.5] * cfg.NUM_VIEWS, std=[0.5] * cfg.NUM_VIEWS)


def load_clinical_tensor(base_id: str, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    clinical_df = cfg.load_clinical_df()
    if base_id not in clinical_df.index:
        raise KeyError(f"找不到患者 {base_id} 的临床数据: {cfg.CLINICAL_CSV}")
    raw = clinical_df.loc[base_id][CLINICAL_COLS].values.astype(np.float32)
    raw = np.where(np.isnan(raw), mean, raw)
    norm = (raw - mean) / (std + 1e-8)
    return torch.from_numpy(norm.astype(np.float32))


def load_label_info(base_id: str) -> tuple[str, str]:
    label_path = cfg.LABEL_CSV
    if not label_path.exists():
        return "NA", "NA"
    label_df = pd.read_csv(label_path)
    if "base_id" not in label_df.columns:
        label_df["base_id"] = label_df["ID"].str.replace(r"\.nii\.gz$", "", regex=True)
    row = label_df[label_df["base_id"].astype(str) == str(base_id)]
    if row.empty:
        return "NA", "NA"
    label = str(row.iloc[0].get("label", "NA"))
    group = str(row.iloc[0].get("group", "NA"))
    return label, group


# ── 融合模型包装 ───────────────────────────────────────────────────────
def forward_with_aux(model: CrossModalTransformerFusion, image: torch.Tensor,
                     pathology: torch.Tensor, clinical: torch.Tensor):
    ct_feat = model.ct_encoder(image)
    path_feat = model.path_adapter(pathology)
    clinical_feat = model.clinical_adapter(clinical)

    ct_tok = model.ct_proj(ct_feat)
    path_tok = model.path_proj(path_feat)
    clinical_tok = model.clinical_proj(clinical_feat)

    tokens = torch.stack([ct_tok, path_tok, clinical_tok], dim=1)
    tokens = tokens + model.modality_embed
    tokens = model.cross_modal_transformer(tokens)

    clinical_query = tokens[:, 2, :]
    gate_logits = model.gate_generator(clinical_query)
    temperature = torch.exp(model.log_gate_temperature) + 1e-6
    gates = torch.sigmoid(gate_logits / temperature)
    gated_tokens = tokens * gates.unsqueeze(-1)

    weights = torch.softmax(model.modality_weights, dim=0)
    fused = (gated_tokens * weights.view(1, 3, 1)).sum(dim=1)
    logit = model.classifier(fused)

    aux = {
        "gates": gates.detach(),
        "modality_weights": weights.detach(),
    }
    return logit, aux


class FusionPathologyAttributor:
    def __init__(self, model_path: Path = MODEL_PATH,
                 device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)
        self.model = self._load_model()
        self._init_train_stats()

    def _load_model(self) -> CrossModalTransformerFusion:
        print(f"[模型] 加载: {self.model_path}")
        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        model = CrossModalTransformerFusion(dropout=0.3, freeze_ct_backbone=True).to(self.device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        print(f"[模型] epoch={ckpt.get('epoch', 'NA')} 设备={self.device}")
        if "val_metrics" in ckpt:
            print(f"[模型] val={ckpt['val_metrics']}")
        if "test_metrics" in ckpt:
            print(f"[模型] test={ckpt['test_metrics']}")
        return model

    def _init_train_stats(self):
        print("[统计量] 使用训练集计算病理/临床归一化参数...")
        ds_train = CTMultiViewDataset("train", use_aug=False)
        self.path_mean = ds_train.path_mean.astype(np.float32)
        self.path_std = ds_train.path_std.astype(np.float32)
        self.clinical_mean = ds_train.clinical_mean.astype(np.float32)
        self.clinical_std = ds_train.clinical_std.astype(np.float32)
        print(f"  pathology stats: {self.path_mean.shape}")
        print(f"  clinical stats: {self.clinical_mean.shape}")

    def load_case_inputs(self, base_id: str):
        feature_dir = FEATURE_ROOT / base_id
        slide_path = feature_dir / f"{base_id}_uni2_slide_features.npy"
        if not slide_path.exists():
            raise FileNotFoundError(f"找不到 slide feature: {slide_path}")

        slide_raw = np.load(slide_path).squeeze(0).astype(np.float32)
        slide_norm = (slide_raw - self.path_mean) / (self.path_std + 1e-8)

        image = load_ct_tensor(base_id).unsqueeze(0).to(self.device)
        pathology = torch.from_numpy(slide_norm).unsqueeze(0).to(self.device)
        clinical = load_clinical_tensor(
            base_id, self.clinical_mean, self.clinical_std
        ).unsqueeze(0).to(self.device)
        return image, pathology, clinical, slide_norm

    def compute_pathology_gradient(self, base_id: str):
        image, pathology, clinical, slide_norm = self.load_case_inputs(base_id)
        pathology = pathology.clone().detach().requires_grad_(True)

        self.model.zero_grad(set_to_none=True)
        logit, aux = forward_with_aux(self.model, image, pathology, clinical)
        target = logit.squeeze()
        target.backward()

        grad = pathology.grad.detach().squeeze(0).cpu().numpy().astype(np.float32)
        logit_value = float(logit.detach().cpu().item())
        prob_value = float(torch.sigmoid(logit.detach()).cpu().item())
        gates = aux["gates"].squeeze(0).cpu().numpy().astype(float)
        weights = aux["modality_weights"].cpu().numpy().astype(float)

        del image, pathology, clinical, logit
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return {
            "grad": grad,
            "slide_norm": slide_norm.astype(np.float32),
            "logit": logit_value,
            "prob": prob_value,
            "gates": gates,
            "weights": weights,
        }

    def compute_tile_scores(self, base_id: str):
        attr = self.compute_pathology_gradient(base_id)
        feature_dir = FEATURE_ROOT / base_id
        tile_feat_path = feature_dir / f"{base_id}_uni2_tile_features.npy"
        meta_path = feature_dir / "tile_metadata.csv"
        if not tile_feat_path.exists():
            raise FileNotFoundError(f"找不到 tile feature: {tile_feat_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"找不到 tile metadata: {meta_path}")

        tile_features = np.load(tile_feat_path).astype(np.float32)
        meta = pd.read_csv(meta_path)
        if len(meta) != tile_features.shape[0]:
            raise ValueError(
                f"tile 数量不一致: metadata={len(meta)}, features={tile_features.shape[0]}"
            )

        tile_norm = (tile_features - self.path_mean[None, :]) / (self.path_std[None, :] + 1e-8)
        centered = tile_norm - attr["slide_norm"][None, :]
        signed_scores = centered @ attr["grad"]

        positive_scores = np.maximum(signed_scores, 0)
        if np.count_nonzero(positive_scores) == 0:
            positive_scores = signed_scores - signed_scores.min()

        score_mask = positive_scores > 0
        norm_scores = percentile_norm(positive_scores, score_mask, lo=1, hi=99)

        meta = meta.copy()
        meta["signed_score"] = signed_scores.astype(np.float32)
        meta["positive_score"] = positive_scores.astype(np.float32)
        meta["norm_score"] = norm_scores.astype(np.float32)

        return meta, attr


# ── 空间渲染 ───────────────────────────────────────────────────────────
def make_tile_path_map(tile_dir: Path) -> dict[tuple[int, int, int, int], Path]:
    mapping = {}
    for path in sorted(tile_dir.glob("*.jpg")):
        coords = parse_tile_name(path)
        if coords is not None:
            mapping[coords] = path
    return mapping


def build_smooth_heatmap_and_tissue(meta: pd.DataFrame, tile_dir: Path,
                                    max_canvas_px: int):
    required = {"x0", "y0", "x1", "y1", "norm_score"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"metadata 缺少列: {sorted(missing)}")

    x_min = int(meta["x0"].min())
    y_min = int(meta["y0"].min())
    x_max = int(meta["x1"].max())
    y_max = int(meta["y1"].max())
    raw_w, raw_h = x_max - x_min, y_max - y_min

    tile_w = float((meta["x1"] - meta["x0"]).median())
    tile_h = float((meta["y1"] - meta["y0"]).median())
    ds_by_canvas = max(1, int(np.ceil(max(raw_w, raw_h) / max_canvas_px)))
    ds_by_tile = max(1, int(np.ceil(min(tile_w, tile_h) / MAX_TILE_CANVAS_PX)))
    ds = max(ds_by_canvas, ds_by_tile)
    cw = int(np.ceil(raw_w / ds))
    ch = int(np.ceil(raw_h / ds))

    tile_px = max(1.0, min(tile_w, tile_h) / ds)
    seed_radius = max(1, int(round(tile_px * HEATMAP_SEED_RADIUS_RATIO)))
    blur_radius = max(HEATMAP_MIN_BLUR_RADIUS, tile_px * HEATMAP_BLUR_RADIUS_RATIO)

    print(
        f"  画布: {raw_w}x{raw_h} -> ds={ds} -> {cw}x{ch}; "
        f"seed={seed_radius}px blur={blur_radius:.1f}px"
    )

    heat_seed = np.zeros((ch, cw), dtype=np.float32)
    weight_seed = np.zeros((ch, cw), dtype=np.float32)
    tissue_rgb = np.full((ch, cw, 3), 235, dtype=np.uint8)
    filled = np.zeros((ch, cw), dtype=np.bool_)

    tile_map = make_tile_path_map(tile_dir)
    sorted_tile_paths = sorted(tile_dir.glob("*.jpg"))

    for row_idx, row in enumerate(meta.itertuples(index=False)):
        x0, y0, x1, y1 = int(row.x0), int(row.y0), int(row.x1), int(row.y1)
        score = float(row.norm_score)

        cx0 = int((x0 - x_min) // ds)
        cy0 = int((y0 - y_min) // ds)
        cx1 = int(np.ceil((x1 - x_min) / ds))
        cy1 = int(np.ceil((y1 - y_min) / ds))
        cx1 = min(max(cx1, cx0 + 1), cw)
        cy1 = min(max(cy1, cy0 + 1), ch)

        center_x = int(round(((x0 + x1) * 0.5 - x_min) / ds))
        center_y = int(round(((y0 + y1) * 0.5 - y_min) / ds))
        x_start = max(0, center_x - seed_radius)
        x_end = min(cw, center_x + seed_radius + 1)
        y_start = max(0, center_y - seed_radius)
        y_end = min(ch, center_y + seed_radius + 1)
        if x_end > x_start and y_end > y_start and score > 0:
            yy, xx = np.ogrid[y_start:y_end, x_start:x_end]
            disk = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= seed_radius ** 2
            heat_seed[y_start:y_end, x_start:x_end][disk] += score
            weight_seed[y_start:y_end, x_start:x_end][disk] += 1.0

        tile_path = tile_map.get((x0, y0, x1, y1))
        if tile_path is None and row_idx < len(sorted_tile_paths):
            tile_path = sorted_tile_paths[row_idx]
        if tile_path is not None and tile_path.exists():
            try:
                tile_img = Image.open(tile_path).convert("RGB")
                tile_np = np.asarray(
                    tile_img.resize((cx1 - cx0, cy1 - cy0), Image.BILINEAR),
                    dtype=np.uint8,
                )
                tissue_rgb[cy0:cy1, cx0:cx1] = tile_np
                filled[cy0:cy1, cx0:cx1] = True
            except Exception as exc:
                print(f"  ! tile 读取失败: {tile_path.name}: {exc}")

    heat_blur = gaussian_blur(heat_seed, blur_radius)
    weight_blur = gaussian_blur(weight_seed, blur_radius)
    raw_heat = heat_blur / np.maximum(weight_blur, 1e-6)

    display_mask = filled
    value_mask = display_mask & (weight_blur > 1e-5)
    heat = percentile_norm(raw_heat, value_mask, lo=1, hi=99)
    heat[~display_mask] = 0.0

    return heat, display_mask, tissue_rgb


def make_gradcam_panel(tissue_u8: np.ndarray, heat: np.ndarray,
                       display_mask: np.ndarray) -> np.ndarray:
    tissue_f = tissue_u8.astype(np.float32)
    heat_rgb = colorize_heatmap(heat).astype(np.float32)
    out = tissue_f.copy()
    out[display_mask] = 0.30 * tissue_f[display_mask] + 0.70 * heat_rgb[display_mask]
    out[~display_mask] = np.array([235, 235, 235], dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_overlay(tissue_u8: np.ndarray, heat: np.ndarray,
                 display_mask: np.ndarray) -> np.ndarray:
    tissue_f = tissue_u8.astype(np.float32)
    heat_rgb = colorize_heatmap(heat).astype(np.float32)
    alpha = HEATMAP_ALPHA_MIN + heat * (HEATMAP_ALPHA_MAX - HEATMAP_ALPHA_MIN)
    alpha[~display_mask] = 0.0
    alpha_3ch = np.stack([alpha] * 3, axis=-1).astype(np.float32)
    blended = heat_rgb * alpha_3ch + tissue_f * (1 - alpha_3ch)
    blended[~display_mask] = np.array([235, 235, 235], dtype=np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def save_triptych(tissue: np.ndarray, heat_panel: np.ndarray, overlay: np.ndarray,
                  save_path: Path, base_id: str, info: dict):
    tissue = resize_for_display(tissue)
    heat_panel = resize_for_display(heat_panel)
    overlay = resize_for_display(overlay)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), gridspec_kw={"wspace": 0.025})
    for ax in axes:
        ax.axis("off")

    axes[0].imshow(tissue)
    axes[0].set_title("WSI", fontsize=14, pad=1)
    axes[1].imshow(heat_panel)
    axes[1].set_title("Fusion Pathology Heatmap", fontsize=14, pad=1)
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=14, pad=1)

    fig.subplots_adjust(left=0.012, right=0.905, top=0.88, bottom=0.045)
    pos = axes[2].get_position()
    cax = fig.add_axes([pos.x1 + 0.018, pos.y0, 0.010, pos.height])
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(COLORMAP))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=9, length=2)

    fig.text(
        0.5, 0.012,
        f"Patient {base_id} | prob={info['prob']:.4f} logit={info['logit']:.4f} "
        f"| label={info['label']} group={info['group']} | {info['model_tag']}",
        ha="center", fontsize=9, color="gray",
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=250, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def save_overlay_image(overlay: np.ndarray, save_path: Path, base_id: str, info: dict):
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    bar = np.full((44, overlay_bgr.shape[1], 3), 60, dtype=np.uint8)
    text = (
        f"Patient {base_id} | prob={info['prob']:.4f} | logit={info['logit']:.4f} "
        f"| label={info['label']} | group={info['group']} | fusion {info['model_tag']}"
    )
    cv2.putText(bar, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (220, 220, 220), 1, cv2.LINE_AA)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), np.vstack([overlay_bgr, bar]),
                [cv2.IMWRITE_PNG_COMPRESSION, 3])


# ── 主流程 ─────────────────────────────────────────────────────────────
def process_patient(attributor: FusionPathologyAttributor, base_id: str,
                    output_root: Path, max_canvas_px: int):
    feature_dir = FEATURE_ROOT / base_id
    tile_dir = TILE_ROOT / base_id
    if not feature_dir.exists():
        raise FileNotFoundError(f"feature 目录不存在: {feature_dir}")
    if not tile_dir.exists():
        raise FileNotFoundError(f"tile 目录不存在: {tile_dir}")

    print(f"\n{'=' * 60}")
    print(f"患者 {base_id}")
    print(f"{'=' * 60}")

    print("  [1/4] 计算融合模型病理梯度与 tile 贡献...")
    meta, attr = attributor.compute_tile_scores(base_id)
    label, group = load_label_info(base_id)
    info = {
        "logit": attr["logit"],
        "prob": attr["prob"],
        "label": label,
        "group": group,
        "model_tag": MODEL_TAG,
    }
    print(
        f"  pred_prob={attr['prob']:.6f}, logit={attr['logit']:.6f}, "
        f"label={label}, group={group}"
    )
    print(
        "  gates(ct,path,cli)="
        f"{attr['gates'][0]:.3f},{attr['gates'][1]:.3f},{attr['gates'][2]:.3f}; "
        "weights="
        f"{attr['weights'][0]:.3f},{attr['weights'][1]:.3f},{attr['weights'][2]:.3f}"
    )
    print(
        f"  tile scores: min={meta['signed_score'].min():.6f}, "
        f"p99={np.percentile(meta['signed_score'], 99):.6f}, "
        f"max={meta['signed_score'].max():.6f}, "
        f"positive={(meta['positive_score'] > 0).sum()}/{len(meta)}"
    )

    patient_out = output_root / base_id
    patient_out.mkdir(parents=True, exist_ok=True)
    score_path = patient_out / f"{base_id}_fusion_tile_scores.csv"
    meta.to_csv(score_path, index=False)
    print(f"  ✓ {score_path}")

    print("  [2/4] 重建全组织热图...")
    heat, display_mask, tissue_rgb = build_smooth_heatmap_and_tissue(
        meta, tile_dir, max_canvas_px=max_canvas_px,
    )

    print("  [3/4] 生成 overlay...")
    heat_panel = make_gradcam_panel(tissue_rgb, heat, display_mask)
    overlay = make_overlay(tissue_rgb, heat, display_mask)

    print("  [4/4] 保存图像...")
    triptych_path = patient_out / f"{base_id}_fusion_pathology_triptych.png"
    overlay_path = patient_out / f"{base_id}_fusion_pathology_overlay.png"
    save_triptych(tissue_rgb, heat_panel, overlay, triptych_path, base_id, info)
    save_overlay_image(overlay, overlay_path, base_id, info)
    print(f"  ✓ {triptych_path}")
    print(f"  ✓ {overlay_path}")

    del meta, heat, display_mask, tissue_rgb, heat_panel, overlay
    gc.collect()
    if attributor.device.type == "cuda":
        torch.cuda.empty_cache()


def load_label_table() -> pd.DataFrame:
    """读取 cfg.LABEL_CSV (label_f.csv), 统一加一列 base_id。"""
    label_df = pd.read_csv(cfg.LABEL_CSV)
    label_df["base_id"] = label_df["ID"].astype(str).str.replace(r"\.nii\.gz$", "", regex=True)
    return label_df


def label_f_base_ids() -> list[str]:
    """label_f.csv 里所有病例 ID。"""
    return load_label_table()["base_id"].drop_duplicates().tolist()


def collect_patients_by_split(splits: list[str]) -> list[str]:
    """按 label_f.csv 的 group 收集病例 (train / val / test*, test 自动含 test1-4)。"""
    label_df = load_label_table()

    groups = [g.lower() for g in splits]
    mask = label_df["group"].isin(groups)
    if "test" in groups:
        mask |= label_df["group"].str.startswith("test")

    bases = label_df.loc[mask, "base_id"].drop_duplicates().tolist()
    return sorted(bases, key=lambda x: int(x) if str(x).isdigit() else 0)


def warn_unknown_patients(patients: list[str]):
    """提示 --patients 里不在 label_f.csv 中的病例 (图上 label/group 会显示 NA)。"""
    known = set(label_f_base_ids())
    unknown = [str(p) for p in patients if str(p) not in known]
    if unknown:
        preview = " ".join(unknown[:10]) + (" ..." if len(unknown) > 10 else "")
        print(f"[警告] {len(unknown)}/{len(patients)} 例不在 {cfg.LABEL_CSV.name} 中, "
              f"图上 label/group 会是 NA: {preview}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="基于融合模型生成病理 WSI 全组织热图"
    )
    parser.add_argument("--patients", nargs="+", default=None,
                        help="患者 ID 列表; 不传则按 --splits (默认 val test) 从 label_f.csv 取")
    parser.add_argument("--splits", nargs="+", default=None,
                        help="按 label_f.csv 划分批量处理: val / test / train "
                             f"(默认 {' '.join(DEFAULT_SPLITS)})")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT,
                        help="输出目录")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH,
                        help="融合模型 checkpoint")
    parser.add_argument("--max-canvas-px", type=int, default=MAX_CANVAS_PX,
                        help="输出画布最长边像素上限")
    return parser.parse_args()


def main():
    global MODEL_TAG
    args = parse_args()
    MODEL_TAG = extract_model_tag(args.model_path)

    if args.patients:
        patients = [str(p) for p in args.patients]
        warn_unknown_patients(patients)
    else:
        splits = args.splits or DEFAULT_SPLITS
        patients = collect_patients_by_split(splits)
        print(f"按 {cfg.LABEL_CSV.name} 的划分 {splits} 收集病例: {len(patients)} 例")

    print("=" * 70)
    print("Fusion Pathology Full-Tissue Heatmap")
    print(f"模型: {args.model_path}")
    print(f"署名标识: {MODEL_TAG}")
    print(f"标签来源: {cfg.LABEL_CSV}")
    print(f"患者: {len(patients)} 例")
    print(f"输出: {args.output_root}")
    print("=" * 70)

    attributor = FusionPathologyAttributor(model_path=args.model_path)
    for idx, base_id in enumerate(patients, 1):
        print(f"\n[{idx}/{len(patients)}] {base_id}")
        try:
            process_patient(
                attributor=attributor,
                base_id=str(base_id),
                output_root=args.output_root,
                max_canvas_px=args.max_canvas_px,
            )
        except Exception as exc:
            print(f"  ✗ {base_id}: {exc}")
            import traceback
            traceback.print_exc()
            if attributor.device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"\n完成: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()

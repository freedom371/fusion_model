"""
病理 WSI 完整组织图 GradCAM 生成器（高斯扩散渲染版）。

核心改进（参考 bl_cam.py）：
  1. 高斯扩散渲染：tile CAM → 种子图 → 高斯模糊 → 连续热力图
  2. 百分位归一化（p1-p99 clipping）
  3. turbo 配色 + 自适应 alpha 混合
  4. 三栏输出：WSI / Grad-CAM / Overlay

Tile 命名规则:
  {patient_id}_{offset_x}_{offset_y}_{tile_x1}_{tile_y1}_{tile_x2}_{tile_y2}.jpg
"""
import os, sys, re, gc
from pathlib import Path

import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import timm
import torchvision.transforms.functional as TF
from pytorch_grad_cam import GradCAM

# ── 路径 ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNI2_WEIGHTS_PATH = PROJECT_ROOT / "feature_extraction" / "uni2" / "pytorch_model.bin"
TILE_ROOT = Path("/hard0/carcinoma/jiandu/yansebiaozhunhua")
OUTPUT_ROOT = Path(__file__).resolve().parent / "full_tissue_output"

UNI2_GRID_SIZE = 16
UNI2_NUM_PREFIX = 9
UNI2_EMBED_DIM = 1536
UNI2_IMG_SIZE = 224
UNI2_MEAN = [0.485, 0.456, 0.406]
UNI2_STD  = [0.229, 0.224, 0.225]

TARGET_PATIENTS = [
    "697", "420", "128", "89", "110",
    "44", "95", "88", "14", "42",
]

# ── 高斯扩散参数（参考 bl_cam.py）─────────────────────────────────────
HEATMAP_SEED_RADIUS_RATIO = 0.35   # 种子半径 / tile 半边长
HEATMAP_BLUR_RADIUS_RATIO = 0.75   # 模糊半径 / tile 半边长
HEATMAP_MIN_BLUR_RADIUS = 2.5
HEATMAP_ALPHA_MIN = 0.20            # overlay 最低 alpha
HEATMAP_ALPHA_MAX = 0.70            # overlay 最高 alpha
COLORMAP = "turbo"
MAX_CANVAS_PX = 20000

_TILE_NAME_RE = re.compile(
    r"^(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)\.jpg$"
)


# ── 工具函数 ────────────────────────────────────────────────────────────
def vit_reshape_transform(tensor, num_prefix=UNI2_NUM_PREFIX,
                           grid=UNI2_GRID_SIZE):
    result = tensor[:, num_prefix:, :].reshape(
        tensor.size(0), grid, grid, tensor.size(2))
    return result.permute(0, 3, 1, 2)


class L2NormTarget:
    def __call__(self, model_output):
        if model_output.ndim == 1:
            return model_output.norm(p=2)
        return model_output.norm(p=2, dim=1)


def parse_tile_coords(filename: str):
    m = _TILE_NAME_RE.match(filename)
    if m is None:
        return None
    return (int(m.group(4)), int(m.group(5)),
            int(m.group(6)), int(m.group(7)))


def percentile_norm(arr, lo=1, hi=99, eps=1e-8):
    """百分位归一化（p1-p99 clipping → [0,1]）。"""
    arr = arr.astype(np.float32)
    mask = arr > 0
    if not mask.any():
        return arr
    vals = arr[mask]
    p_lo = np.percentile(vals, lo)
    p_hi = np.percentile(vals, hi)
    if p_hi - p_lo < eps:
        return arr
    out = np.zeros_like(arr, dtype=np.float32)
    out[mask] = np.clip((arr[mask] - p_lo) / (p_hi - p_lo + eps), 0, 1)
    return out


def gaussian_blur(arr, radius):
    """高斯模糊，radius 为 sigma。"""
    arr = arr.astype(np.float32)
    radius = max(0.1, radius)
    sigma = float(radius)
    k = int(np.ceil(sigma * 6))
    if k % 2 == 0:
        k += 1
    k = max(3, k)
    return cv2.GaussianBlur(arr, (k, k), sigmaX=sigma, sigmaY=sigma,
                            borderType=cv2.BORDER_REFLECT_101)


class FullTileDataset(Dataset):
    def __init__(self, tile_paths):
        self.tile_paths = tile_paths
    def __len__(self):
        return len(self.tile_paths)
    def __getitem__(self, idx):
        path = self.tile_paths[idx]
        img = Image.open(path).convert("RGB")
        img = TF.resize(img, (UNI2_IMG_SIZE, UNI2_IMG_SIZE),
                        interpolation=TF.InterpolationMode.BILINEAR)
        tensor = TF.to_tensor(img)
        tensor = TF.normalize(tensor, mean=UNI2_MEAN, std=UNI2_STD)
        return tensor, path


# ── 完整组织图 CAM 生成器 ─────────────────────────────────────────────
class FullTissueCAMGenerator:

    def __init__(self, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_uni2()

    def _load_uni2(self):
        print(f"[UNI2] 加载: {UNI2_WEIGHTS_PATH}")
        model = timm.create_model(
            "vit_giant_patch14_224", pretrained=False,
            img_size=224, patch_size=14, depth=24, num_heads=24,
            init_values=1e-5, embed_dim=UNI2_EMBED_DIM,
            mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
            mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
            reg_tokens=8, global_pool="token")
        sd = torch.load(UNI2_WEIGHTS_PATH, map_location="cpu",
                        weights_only=False)
        model.load_state_dict(sd, strict=True)
        model.to(self.device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = True
        n = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[UNI2] {n:.1f}M 参数, {self.device}")
        return model

    def _get_target_layer(self):
        return [self.model.blocks[-1].norm1]

    # ── Step 1: CAM ──────────────────────────────────────────────────
    def generate_tile_cams(self, tile_paths, batch_size=4):
        results = {}
        loader = DataLoader(FullTileDataset(tile_paths),
                            batch_size=batch_size, shuffle=False,
                            num_workers=0)
        cam = GradCAM(model=self.model,
                      target_layers=self._get_target_layer(),
                      reshape_transform=vit_reshape_transform)
        for bt, bp in loader:
            bt = bt.to(self.device)
            cams = cam(input_tensor=bt, targets=[L2NormTarget()])
            for i, p in enumerate(bp):
                results[p] = np.clip(cams[i], 0, 1)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return results

    # ── Step 2: 空间重建（全尺寸 CAM + 轻量高斯边界平滑）─────────
    def build_smooth_heatmap(self, cam_results):
        """全尺寸 CAM 画布拼接 + 轻量高斯模糊（仅平滑 tile 边界）。

        保留 tile 内部 CAM 空间结构，同时消除硬边界。
        """
        # 解析坐标
        coords_map = {}
        for p in cam_results:
            c = parse_tile_coords(os.path.basename(p))
            if c is not None:
                coords_map[p] = c
        if not coords_map:
            raise ValueError("无法解析坐标")

        xs = [c[0] for c in coords_map.values()] + \
             [c[2] for c in coords_map.values()]
        ys = [c[1] for c in coords_map.values()] + \
             [c[3] for c in coords_map.values()]
        x_min, y_min = min(xs), min(ys)
        x_max, y_max = max(xs), max(ys)
        raw_w, raw_h = x_max - x_min, y_max - y_min

        ds = max(1, int(np.ceil(max(raw_w, raw_h) / MAX_CANVAS_PX)))
        cw, ch = raw_w // ds, raw_h // ds
        tile_canvas_sz = 512 // ds

        print(f"  画布: {raw_w}×{raw_h} → {ds}× → {cw}×{ch}, "
              f"tile {tile_canvas_sz}px")

        # 全尺寸 CAM 画布 + 组织 RGB
        cam_canvas = np.zeros((ch, cw), dtype=np.float32)
        tissue_rgb = np.zeros((ch, cw, 3), dtype=np.uint8)
        filled = np.zeros((ch, cw), dtype=np.bool_)
        n_active = 0

        for tile_path, cam_224 in cam_results.items():
            c = coords_map.get(tile_path)
            if c is None:
                continue
            x1, y1 = c[0], c[1]
            cx1 = (x1 - x_min) // ds
            cy1 = (y1 - y_min) // ds
            cx2 = (x1 + 512 - x_min) // ds
            cy2 = (y1 + 512 - y_min) // ds
            h, w = max(1, cy2 - cy1), max(1, cx2 - cx1)

            # 224×224 CAM → resize 到画布 tile 尺寸
            cam_r = cv2.resize(cam_224, (w, h), interpolation=cv2.INTER_LINEAR)
            # 逐 tile 归一化到 [0,1]
            local_max = cam_r.max()
            if local_max > 0.005:
                cam_r = cam_r / local_max
                n_active += 1
            else:
                cam_r = np.zeros_like(cam_r)
            # max pooling 覆盖
            np.maximum(cam_canvas[cy1:cy2, cx1:cx2], cam_r,
                       out=cam_canvas[cy1:cy2, cx1:cx2])

            # RGB 底图
            if not filled[cy1:cy2, cx1:cx2].all():
                try:
                    tile_img = Image.open(tile_path).convert("RGB")
                    tile_np = np.array(
                        tile_img.resize((w, h), Image.BILINEAR), dtype=np.uint8)
                    m = filled[cy1:cy2, cx1:cx2]
                    tissue_rgb[cy1:cy2, cx1:cx2][~m] = tile_np[~m]
                    filled[cy1:cy2, cx1:cx2] = True
                except Exception:
                    pass
            else:
                filled[cy1:cy2, cx1:cx2] = True

        # 轻量高斯模糊：半 tile 像素作为 sigma，仅平滑 tile 边界
        blur_sigma = max(0.5, tile_canvas_sz * 0.15)
        cam_smooth = gaussian_blur(cam_canvas, radius=blur_sigma)

        # 百分位归一化（只在有组织区域）
        display_mask = filled
        heat = percentile_norm(cam_smooth * display_mask.astype(np.float32))
        heat[~display_mask] = 0.0

        return heat, display_mask, tissue_rgb, n_active

    # ── Step 3: 可视化 ──────────────────────────────────────────────
    def colorize_heatmap(self, heat):
        """turbo colormap → uint8 RGB。"""
        cmap = plt.get_cmap(COLORMAP)
        rgba = cmap(np.clip(heat, 0, 1))
        return (rgba[:, :, :3] * 255).astype(np.uint8)

    def make_gradcam_panel(self, tissue_u8, heat, display_mask):
        """Grad-CAM 面板：淡化 WSI(30%) + 热力图(70%)，非组织区域留白。"""
        tissue_f = tissue_u8.astype(np.float32)
        heat_rgb = self.colorize_heatmap(heat).astype(np.float32)
        out = tissue_f.copy()
        blended = 0.3 * tissue_f + 0.7 * heat_rgb
        out[display_mask] = blended[display_mask]
        out[~display_mask] = np.array([235, 235, 235], dtype=np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)

    def make_overlay(self, tissue_u8, heat, display_mask):
        """自适应 alpha 叠加：alpha = 0.2 + heat * 0.5。"""
        tissue_f = tissue_u8.astype(np.float32)
        heat_rgb = self.colorize_heatmap(heat).astype(np.float32)
        alpha = HEATMAP_ALPHA_MIN + heat * (HEATMAP_ALPHA_MAX - HEATMAP_ALPHA_MIN)
        alpha_3ch = np.stack([alpha] * 3, axis=-1).astype(np.float32)
        blended = heat_rgb * alpha_3ch + tissue_f * (1 - alpha_3ch)
        blended[~display_mask] = np.array([235, 235, 235], dtype=np.float32)
        return np.clip(blended, 0, 255).astype(np.uint8)

    def save_triptych(self, tissue_u8, gradcam_panel, overlay,
                      path, base_id, n_tiles, n_active):
        """三栏对比图：WSI / Grad-CAM / Overlay + 色条。"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.2),
                                 gridspec_kw={"wspace": 0.025})
        for ax in axes:
            ax.axis("off")

        axes[0].imshow(tissue_u8)
        axes[0].set_title("WSI", fontsize=14, pad=1)

        axes[1].imshow(gradcam_panel)
        axes[1].set_title("Grad-CAM", fontsize=14, pad=1)

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay", fontsize=14, pad=1)

        # 右侧色条
        fig.subplots_adjust(left=0.012, right=0.905, top=0.88, bottom=0.035)
        pos = axes[2].get_position()
        cax = fig.add_axes([pos.x1 + 0.018, pos.y0, 0.010, pos.height])
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(COLORMAP))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_ticks([0, 0.5, 1.0])
        cbar.ax.tick_params(labelsize=9, length=2)

        # 底部信息
        fig.text(0.5, 0.01,
                 f"Patient {base_id}  |  {n_tiles} tiles  |  "
                 f"{n_active} with CAM activation  |  {COLORMAP}",
                 ha="center", fontsize=9, color="gray")

        plt.savefig(path, dpi=250, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)

    # ── 主流程 ──────────────────────────────────────────────────────
    def process_patient(self, base_id, output_dir, batch_size=4):
        tile_dir = TILE_ROOT / base_id
        if not tile_dir.exists():
            print(f"  ✗ 目录不存在: {tile_dir}")
            return
        all_tiles = sorted([str(p) for p in tile_dir.iterdir()
                            if p.suffix == ".jpg"])
        n_tiles = len(all_tiles)
        if n_tiles == 0:
            print(f"  ✗ 无 tile")
            return

        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{base_id}_cam_triptych.png")

        print(f"\n{'='*50}")
        print(f"患者 {base_id}: {n_tiles} tiles")
        print(f"{'='*50}")

        # 1. CAM
        print(f"  [1/4] CAM ({n_tiles} tiles)...")
        cam_results = self.generate_tile_cams(all_tiles, batch_size)
        print(f"  ✓ {len(cam_results)} CAMs")

        # 2. 高斯扩散热力图
        print(f"  [2/4] 高斯扩散渲染...")
        heat, display_mask, tissue_rgb, n_active = \
            self.build_smooth_heatmap(cam_results)
        print(f"  ✓ {n_active}/{n_tiles} tiles 有激活")

        # 3. 三栏图
        print(f"  [3/4] 生成三栏图...")
        gradcam_panel = self.make_gradcam_panel(tissue_rgb, heat, display_mask)
        overlay = self.make_overlay(tissue_rgb, heat, display_mask)
        self.save_triptych(tissue_rgb, gradcam_panel, overlay,
                           save_path, base_id, n_tiles, n_active)
        print(f"  ✓ {save_path}")

        # 4. 同时保存 overlay 原图（更大尺寸）
        overlay_path = os.path.join(output_dir, f"{base_id}_cam_overlay.png")
        # 用 cv2 无损保存 overlay 原图
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        # 底部信息条
        bar = np.full((40, overlay_bgr.shape[1], 3), 60, dtype=np.uint8)
        cv2.putText(bar,
                    f"Patient {base_id} | {n_tiles} tiles | "
                    f"{n_active} with CAM | colormap={COLORMAP}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.imwrite(overlay_path, np.vstack([overlay_bgr, bar]),
                    [cv2.IMWRITE_PNG_COMPRESSION, 3])
        print(f"  ✓ {overlay_path}")

        del cam_results, heat, display_mask, tissue_rgb
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


# ── main ───────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("病理完整组织图 CAM (高斯扩散渲染 + 三栏图)")
    print(f"患者 ({len(TARGET_PATIENTS)}): {TARGET_PATIENTS}")
    print(f"输出: {OUTPUT_ROOT}")
    print("=" * 60)

    gen = FullTissueCAMGenerator()
    for i, bid in enumerate(TARGET_PATIENTS):
        print(f"\n[{i+1}/{len(TARGET_PATIENTS)}] 患者 {bid}")
        try:
            gen.process_patient(bid, str(OUTPUT_ROOT), batch_size=4)
        except Exception as e:
            print(f"  ✗ {e}")
            import traceback
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"完成! {OUTPUT_ROOT.resolve()}")
    for f in sorted(OUTPUT_ROOT.glob("*.png")):
        print(f"  {f.name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

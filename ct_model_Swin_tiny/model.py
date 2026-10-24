import torch
import torch.nn as nn
import timm

from config import (
    BACKBONE_NAME,
    BACKBONE_PRETRAINED,
    BACKBONE_USE_LOCAL_WEIGHTS,
    BACKBONE_LOCAL_WEIGHTS,
    IMG_SIZE,
    USE_SE,
    SE_REDUCTION_RATIO,
    USE_CBAM,
    CBAM_REDUCTION_RATIO,
    CBAM_SPATIAL_KERNEL_SIZE,
)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        hidden_dim = max(channels // reduction_ratio, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        hidden_dim = max(channels // reduction_ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False),
        )
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_attn = self.mlp(self.avg_pool(x))
        max_attn = self.mlp(self.max_pool(x))
        return self.gate(avg_attn + max_attn)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError(f"CBAM spatial kernel must be 3 or 7, got {kernel_size}")
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_map, max_map], dim=1)
        return self.gate(self.conv(attn))


class CBAMBlock(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16, spatial_kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio=reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class CTDirectEncoder(nn.Module):
    """将三张 CT 切片视为 3 通道图像，使用 Swin Transformer Tiny 编码。

    timm 的 Swin forward_features 返回 (B, H, W, C) 的 NHWC 特征图，
    先 permute 到 (B, C, H, W) NCHW 再接入 SE/CBAM，最后全局平均池化。
    """

    def __init__(self, freeze_backbone: bool = True):
        super().__init__()
        self.freeze_backbone = freeze_backbone

        self.backbone = timm.create_model(
            BACKBONE_NAME,
            pretrained=BACKBONE_PRETRAINED,
            num_classes=0,
            img_size=IMG_SIZE,
        )

        # 本地权重加载（预留，当前使用 timm 内置预训练权重）
        if BACKBONE_USE_LOCAL_WEIGHTS and BACKBONE_LOCAL_WEIGHTS.exists():
            import safetensors.torch

            state_dict = safetensors.torch.load_file(str(BACKBONE_LOCAL_WEIGHTS))
            state_dict = {
                k: v for k, v in state_dict.items()
                if not k.startswith("head.")
            }
            try:
                missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
                print(
                    f"[{BACKBONE_NAME}] 本地权重已加载 "
                    f"(缺失={len(missing)}, 多余={len(unexpected)})"
                )
            except RuntimeError as exc:
                print(
                    f"[{BACKBONE_NAME}] 跳过本地权重（形状不匹配）: {exc}"
                )

        self.feature_dim = self.backbone.num_features

        # SE / CBAM 注意力模块（作用于 NCHW 特征图）
        attention_blocks = []
        if USE_SE:
            attention_blocks.append(
                SEBlock(
                    channels=self.feature_dim,
                    reduction_ratio=SE_REDUCTION_RATIO,
                )
            )
        if USE_CBAM:
            attention_blocks.append(
                CBAMBlock(
                    channels=self.feature_dim,
                    reduction_ratio=CBAM_REDUCTION_RATIO,
                    spatial_kernel_size=CBAM_SPATIAL_KERNEL_SIZE,
                )
            )
        self.attention = (
            nn.Sequential(*attention_blocks)
            if attention_blocks
            else nn.Identity()
        )

        # 冻结 backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        print(
            f"[CT Backbone] {BACKBONE_NAME} pretrained={BACKBONE_PRETRAINED} "
            f"dim={self.feature_dim} freeze={self.freeze_backbone} "
            f"se={USE_SE} cbam={USE_CBAM}"
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入形状规范化
        if x.dim() == 5:
            if x.shape[1] == 1:
                x = x.squeeze(1)
            elif x.shape[2] == 1:
                x = x.squeeze(2)
            else:
                raise ValueError(f"Expected a singleton axis in shape {tuple(x.shape)}")
        if x.dim() != 4:
            raise ValueError(f"Expected input shape (B, 3, H, W), got {tuple(x.shape)}")
        if x.shape[1] != 3:
            raise ValueError(f"Expected 3 CT slices as channels, got {x.shape[1]}")

        if self.freeze_backbone:
            with torch.no_grad():
                return self._extract_features(x)
        else:
            return self._extract_features(x)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """forward_features → NHWC 特征图 → permute 到 NCHW → SE/CBAM → 全局平均池化。"""
        # (B, 3, H, W) → (B, H', W', C)
        feat = self.backbone.forward_features(x)
        # NHWC → NCHW
        feat = feat.permute(0, 3, 1, 2).contiguous()
        # 2D 注意力
        feat = self.attention(feat)
        # 全局平均池化 → (B, C)
        return feat.mean(dim=(2, 3))


class CTOnlyModel(nn.Module):
    """3-slice CT as 3 channels -> Swin-Tiny + SE/CBAM -> MLP classifier."""

    def __init__(self, dropout: float = 0.3, freeze_backbone: bool = True):
        super().__init__()
        self.ct_encoder = CTDirectEncoder(freeze_backbone=freeze_backbone)
        self.classifier = nn.Sequential(
            nn.Linear(self.ct_encoder.feature_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 8),
            nn.LayerNorm(8),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(8, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.ct_encoder(image)
        return self.classifier(feat)

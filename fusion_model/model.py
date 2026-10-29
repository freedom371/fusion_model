import torch
import torch.nn as nn
import timm

if __package__:
    from .config import (
        BACKBONE_LOCAL_WEIGHTS,
        BACKBONE_NAME,
        BACKBONE_PRETRAINED,
        BACKBONE_USE_LOCAL_WEIGHTS,
        CBAM_REDUCTION_RATIO,
        CBAM_SPATIAL_KERNEL_SIZE,
        CLINICAL_DIM,
        FUSION_D_MODEL,
        FUSION_DIM_FF,
        FUSION_DROPOUT,
        FUSION_NHEAD,
        FUSION_NUM_LAYERS,
        IMG_SIZE,
        SE_REDUCTION_RATIO,
        USE_CBAM,
        USE_SE,
    )
else:
    from config import (
        BACKBONE_LOCAL_WEIGHTS,
        BACKBONE_NAME,
        BACKBONE_PRETRAINED,
        BACKBONE_USE_LOCAL_WEIGHTS,
        CBAM_REDUCTION_RATIO,
        CBAM_SPATIAL_KERNEL_SIZE,
        CLINICAL_DIM,
        FUSION_D_MODEL,
        FUSION_DIM_FF,
        FUSION_DROPOUT,
        FUSION_NHEAD,
        FUSION_NUM_LAYERS,
        IMG_SIZE,
        SE_REDUCTION_RATIO,
        USE_CBAM,
        USE_SE,
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


class CTEncoder(nn.Module):
    """Three-view CT -> Swin-Tiny -> optional SE/CBAM -> pooled feature."""

    def __init__(self, freeze_backbone: bool = True , dropout: float = 0.5):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.backbone = timm.create_model(
            BACKBONE_NAME,
            pretrained=BACKBONE_PRETRAINED,
            num_classes=0,
            img_size=IMG_SIZE,
        )

        if BACKBONE_USE_LOCAL_WEIGHTS and BACKBONE_LOCAL_WEIGHTS.exists():
            import safetensors.torch

            state_dict = safetensors.torch.load_file(str(BACKBONE_LOCAL_WEIGHTS))
            state_dict = {
                key: value
                for key, value in state_dict.items()
                if not key.startswith("head.")
            }
            try:
                missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
                print(
                    f"[{BACKBONE_NAME}] Local weights loaded "
                    f"(missing={len(missing)}, unexpected={len(unexpected)})"
                )
            except RuntimeError as exc:
                print(f"[{BACKBONE_NAME}] Skip local weights due to shape mismatch: {exc}")

        self.feature_dim = self.backbone.num_features

        attention_blocks = []
        if USE_SE:
            attention_blocks.append(SEBlock(self.feature_dim, reduction_ratio=SE_REDUCTION_RATIO))
        if USE_CBAM:
            attention_blocks.append(
                CBAMBlock(
                    self.feature_dim,
                    reduction_ratio=CBAM_REDUCTION_RATIO,
                    spatial_kernel_size=CBAM_SPATIAL_KERNEL_SIZE,
                )
            )
        self.attention = nn.Sequential(*attention_blocks) if attention_blocks else nn.Identity()

        # 维度调整投影层：backbone 特征维度 → CT_OUTPUT_DIM
        self.proj = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 8),
            nn.LayerNorm(8),
            nn.GELU(),
            # nn.Linear(self.feature_dim, CT_OUTPUT_DIM),
            # nn.BatchNorm1d(CT_OUTPUT_DIM),
            # nn.ReLU(inplace=True),
        )
        self.feature_dim = 8  # 与 self.proj 最后一层 nn.Linear(64, 8) 的输出维度一致

        if self.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
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
            raise ValueError(f"Expected 3 CT views as channels, got {x.shape[1]}")

        return self._extract_features(x)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # no_grad 只能包住 backbone,不能包住 attention/proj,
        # 否则冻结 backbone 时 CBAM 和 proj 参数收不到梯度,永远不会更新。
        if self.freeze_backbone:
            with torch.no_grad():
                feat = self.backbone.forward_features(x)
        else:
            feat = self.backbone.forward_features(x)
        feat = feat.permute(0, 3, 1, 2).contiguous()
        feat = self.attention(feat)
        feat = feat.mean(dim=(2, 3))
        return self.proj(feat)


class PathologyAdapter(nn.Module):
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1536, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClinicalAdapter(nn.Module):
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(CLINICAL_DIM, CLINICAL_DIM),
            nn.LayerNorm(CLINICAL_DIM),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossModalTransformerFusion(nn.Module):
    """跨模态 Transformer + 临床先验门控融合网络 (CMT-CPG).

    CT(feature_dim) + pathology(64) + clinical(6)
        → 统一投影到 d_model
        → 2 层 Transformer Encoder（全局跨模态 self-attention）
        → 临床 token 生成门控信号调制各模态贡献
        → 自适应聚合 → MLP 分类器
    """

    def __init__(
        self,
        dropout: float = FUSION_DROPOUT,
        freeze_ct_backbone: bool = True,
        d_model: int = FUSION_D_MODEL,
        nhead: int = FUSION_NHEAD,
        num_layers: int = FUSION_NUM_LAYERS,
        dim_feedforward: int = FUSION_DIM_FF,
    ):
        super().__init__()

        # ===== 模态编码器（复用现有模块，保持不变）=====
        self.ct_encoder = CTEncoder(freeze_backbone=freeze_ct_backbone, dropout= dropout)
        self.path_adapter = PathologyAdapter(dropout=dropout)
        self.clinical_adapter = ClinicalAdapter(dropout=dropout)

        ct_dim = self.ct_encoder.feature_dim  # CT_OUTPUT_DIM，如 16

        # ===== Step 1: 统一语义投影 =====
        self.ct_proj = nn.Sequential(
            nn.Linear(ct_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.path_proj = nn.Sequential(
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model),
        )
        self.clinical_proj = nn.Sequential(
            nn.Linear(CLINICAL_DIM, d_model),
            nn.LayerNorm(d_model),
        )

        # ===== Step 2: 可学习模态类型嵌入 =====
        self.modality_embed = nn.Parameter(torch.randn(1, 3, d_model) * 0.02)

        # ===== Step 3: 跨模态 Transformer Encoder =====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.cross_modal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # ===== Step 4: 临床先验门控生成器 =====
        self.gate_generator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),
        )
        self.log_gate_temperature = nn.Parameter(torch.zeros(1))

        # ===== Step 5: 自适应聚合 =====
        self.modality_weights = nn.Parameter(torch.ones(3) / 3.0)

        # ===== Step 6: 分类器 =====
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # 参数统计
        total_params = sum(p.numel() for p in self.parameters())
        print(
            f"[CMT-CPG] d_model={d_model} nhead={nhead} num_layers={num_layers} "
            f"dim_ff={dim_feedforward} total_params={total_params:,}"
        )

    def forward(
        self, image: torch.Tensor, pathology: torch.Tensor, clinical: torch.Tensor
    ) -> torch.Tensor:
        # ---- Step 0: 模态编码 ----
        ct_feat = self.ct_encoder(image)                # (B, ct_dim)
        path_feat = self.path_adapter(pathology)         # (B, 64)
        clinical_feat = self.clinical_adapter(clinical)  # (B, CLINICAL_DIM)

        # ---- Step 1: 统一投影 ----
        ct_tok = self.ct_proj(ct_feat)                   # (B, d_model)
        path_tok = self.path_proj(path_feat)             # (B, d_model)
        clinical_tok = self.clinical_proj(clinical_feat) # (B, d_model)

        # ---- Step 2: Token 构建 ----
        tokens = torch.stack([ct_tok, path_tok, clinical_tok], dim=1)  # (B, 3, d_model)
        tokens = tokens + self.modality_embed                          # (B, 3, d_model)

        # ---- Step 3: 跨模态 Transformer ----
        tokens = self.cross_modal_transformer(tokens)                 # (B, 3, d_model)

        # ---- Step 4: 临床先验门控 ----
        clinical_query = tokens[:, 2, :]                              # (B, d_model)
        gate_logits = self.gate_generator(clinical_query)             # (B, 3)
        temperature = torch.exp(self.log_gate_temperature) + 1e-6
        gates = torch.sigmoid(gate_logits / temperature)              # (B, 3)
        gated_tokens = tokens * gates.unsqueeze(-1)                   # (B, 3, d_model)

        # ---- Step 5: 自适应聚合 ----
        weights = torch.softmax(self.modality_weights, dim=0)         # (3,)
        fused = (gated_tokens * weights.view(1, 3, 1)).sum(dim=1)     # (B, d_model)

        # ---- Step 6: 分类 ----
        return self.classifier(fused)                                 # (B, 1)

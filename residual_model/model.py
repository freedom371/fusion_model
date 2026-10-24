import torch
import torch.nn as nn
import timm
from config import BACKBONE_NAME, BACKBONE_PRETRAINED, BACKBONE_USE_LOCAL_WEIGHTS, BACKBONE_LOCAL_WEIGHTS


class CTEncoder(nn.Module):
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.backbone = timm.create_model(
            BACKBONE_NAME, pretrained=BACKBONE_PRETRAINED, num_classes=0,
        )
        if BACKBONE_USE_LOCAL_WEIGHTS and BACKBONE_LOCAL_WEIGHTS.exists():
            import safetensors.torch
            state_dict = safetensors.torch.load_file(str(BACKBONE_LOCAL_WEIGHTS))
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            print(f"[{BACKBONE_NAME}] Local weights loaded (missing={len(missing)}, unexpected={len(unexpected)})")

        backbone_dim = self.backbone.num_features
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_dim, 16), nn.BatchNorm1d(16), nn.ReLU(inplace=True),
        )
        print(f"[CT Backbone] {BACKBONE_NAME} pretrained={BACKBONE_PRETRAINED} dim={backbone_dim}")

    def forward(self, x):
        if x.dim() == 5:
            x = x.squeeze(1)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        feat = self.backbone(x)
        return self.proj(feat)                       # (B, 32)


class PathologyAdapter(nn.Module):
    """病理特征：1536d → 64d"""
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1536, 64), nn.LayerNorm(64), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)                            # (B, 64)


class ClinicalAdapter(nn.Module):
    """临床特征：11d → 12d"""
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(11, 12), nn.LayerNorm(12), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)                            # (B, 12)


class MultiModalTransformer(nn.Module):
    """CT(16) + Path(64) + Clinical(12) → concat(92) → MLP → 分类"""
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.ct_encoder = CTEncoder()
        self.path_adapter = PathologyAdapter(dropout=dropout)
        self.clinical_adapter = ClinicalAdapter(dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(92, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, image, pathology, clinical):
        ct_feat = self.ct_encoder(image)               # (B, 32)
        path_feat = self.path_adapter(pathology)       # (B, 64)
        clinical_feat = self.clinical_adapter(clinical)  # (B, 12)
        fused = torch.cat([ct_feat, path_feat, clinical_feat], dim=1)  # (B, 108)
        return self.classifier(fused)                  # (B, 1)

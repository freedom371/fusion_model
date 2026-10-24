import torch
from torch import nn

from fusion_model.model import CTEncoder, PathologyAdapter, ClinicalAdapter
from .config import MODALITIES, base


class AblationCTEncoder(CTEncoder):
    """Freeze only Swin; retain gradients for CBAM and the CT projection."""

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("CT input must have shape (B, 3, H, W)")
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone.forward_features(image)
        else:
            features = self.backbone.forward_features(image)
        features = features.permute(0, 3, 1, 2).contiguous()
        features = self.attention(features).mean(dim=(2, 3))
        return self.proj(features)


class ModalityFusion(nn.Module):
    """CMT-CPG with physically absent branches for excluded modalities.

    Clinical variants keep clinical-prior gating. Without clinical inputs the
    gate is removed; Transformer tokens are aggregated by learned softmax weights.
    """

    def __init__(self, variant, dropout=0.1, freeze_ct_backbone=True):
        super().__init__()
        self.modalities = MODALITIES[variant]
        self.variant = variant
        dim = base.FUSION_D_MODEL
        encoders, dimensions = {}, {}
        if "ct" in self.modalities:
            encoders["ct"] = AblationCTEncoder(freeze_ct_backbone, dropout)
            dimensions["ct"] = encoders["ct"].feature_dim
        if "pathology" in self.modalities:
            encoders["pathology"] = PathologyAdapter(dropout)
            dimensions["pathology"] = 64
        if "clinical" in self.modalities:
            encoders["clinical"] = ClinicalAdapter(dropout)
            dimensions["clinical"] = base.CLINICAL_DIM
        self.encoders = nn.ModuleDict(encoders)
        self.projections = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(dimensions[name], dim), nn.LayerNorm(dim))
            for name in self.modalities
        })
        count = len(self.modalities)
        self.modality_embed = nn.Parameter(torch.randn(1, count, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=base.FUSION_NHEAD,
            dim_feedforward=base.FUSION_DIM_FF, dropout=dropout,
            activation="gelu", batch_first=True,
        )
        self.cross_modal_transformer = nn.TransformerEncoder(
            layer, num_layers=base.FUSION_NUM_LAYERS,
        )
        if "clinical" in self.modalities:
            self.gate_generator = nn.Sequential(
                nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, count),
            )
            self.log_gate_temperature = nn.Parameter(torch.zeros(1))
        self.modality_weights = nn.Parameter(torch.ones(count) / count)
        self.classifier = nn.Sequential(
            nn.Linear(dim, 32), nn.LayerNorm(32), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(32, 1),
        )

    def forward(self, image=None, pathology=None, clinical=None):
        inputs = {"ct": image, "pathology": pathology, "clinical": clinical}
        tokens = []
        for name in self.modalities:
            if inputs[name] is None:
                raise ValueError(f"Missing required modality: {name}")
            tokens.append(self.projections[name](self.encoders[name](inputs[name])))
        tokens = self.cross_modal_transformer(torch.stack(tokens, dim=1) + self.modality_embed)
        if "clinical" in self.modalities:
            query = tokens[:, self.modalities.index("clinical")]
            temperature = self.log_gate_temperature.exp() + 1e-6
            gates = torch.sigmoid(self.gate_generator(query) / temperature)
            tokens = tokens * gates.unsqueeze(-1)
        weights = self.modality_weights.softmax(dim=0)
        return self.classifier((tokens * weights[None, :, None]).sum(dim=1))

import torch
import torch.nn as nn
from config import PATH_DIM


class PathologyOnlyModel(nn.Module):
    """纯病理特征模型：1536d → 64 → 32 → 16 → 1"""

    def __init__(self, input_dim: int = PATH_DIM, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, 64),
            # nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            # nn.Dropout(dropout),
            # nn.Linear(512, 128),
            # nn.LayerNorm(128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(dropout),
            # nn.Linear(128, 32),
            # nn.LayerNorm(32),
            # nn.ReLU(inplace=True),
            # nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, pathology):
        return self.net(pathology)  # (B, 1)

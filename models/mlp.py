import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """
    Input(D) -> Linear(512) -> ReLU -> BatchNorm -> Dropout(0.3)
             -> Linear(128) -> ReLU -> BatchNorm -> Dropout(0.3)
             -> Linear(32)  -> ReLU -> BatchNorm
             -> Linear(C)   (logits)
    """
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2 for multiclass classification.")

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),

            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),

            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),

            nn.Linear(32, num_classes),  # logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D)
        return: (B, C) logits
        """
        return self.net(x)

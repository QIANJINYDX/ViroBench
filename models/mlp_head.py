import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional


class MLPHead(nn.Module):
    """
    通用微调头：
      - binary:    输出 1，训练时建议用 BCEWithLogitsLoss（forward 默认不加 Sigmoid）
      - multiclass:输出 C，训练时用 CrossEntropyLoss（forward 默认不加 Softmax）
      - regression:输出 K（默认 1），训练时用 MSELoss/HuberLoss

    结构：
      Input(D) → Linear(512) → ReLU → BatchNorm → Dropout(0.3)
                → Linear(128) → ReLU → BatchNorm → Dropout(0.3)
                → Linear(32)  → ReLU → BatchNorm
                → Linear(out) → (可选激活)
    """
    def __init__(
        self,
        input_dim: int,                       # D: embedding 维度
        task: Literal["binary", "multiclass", "regression"] = "multiclass",
        num_outputs: int = 1,                 # binary=1；multiclass=C；regression=K
        h1: int = 512,
        h2: int = 128,
        h3: int = 32,
        p_drop: float = 0.3,
    ):
        super().__init__()
        assert task in {"binary", "multiclass", "regression"}
        self.task = task
        self.num_outputs = int(num_outputs)
        if self.num_outputs < 100 and self.task == "multiclass":
            h1,h2,h3 = 512,128,64
        elif self.num_outputs < 1000 and self.task == "multiclass":
            h1,h2,h3 = 512,256,128

        self.block1 = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(inplace=True),
            nn.LayerNorm(h1),
            nn.Dropout(p_drop),
        )
        self.block2 = nn.Sequential(
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.LayerNorm(h2),
            nn.Dropout(p_drop),
        )
        self.block3 = nn.Sequential(
            nn.Linear(h2, h3),
            nn.ReLU(inplace=True),
            nn.LayerNorm(h3),
        )
        self.fc_out = nn.Linear(h3, self._out_dim())

        self._reset_parameters()

    def _out_dim(self) -> int:
        if self.task == "binary":
            return 1
        return self.num_outputs

    def _reset_parameters(self):
        # Kaiming 初始化适配 ReLU；BN 初始化为标准形式
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, apply_activation: bool = False) -> torch.Tensor:
        """
        返回：
          - binary/multiclass: logits（apply_activation=True 时：sigmoid/softmax 概率）
          - regression: 预测值
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        out = self.fc_out(x)

        if not apply_activation:
            return out

        if self.task == "binary":
            return torch.sigmoid(out)          # [B,1]
        elif self.task == "multiclass":
            return F.softmax(out, dim=-1)      # [B,C]
        else:  # regression
            return out                          # 线性输出

    # ------- 实用方法（可选） -------
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """推理：返回类别或数值。"""
        if self.task == "binary":
            probs = torch.sigmoid(self.forward(x, apply_activation=False))
            return (probs >= threshold).long().view(-1)
        elif self.task == "multiclass":
            logits = self.forward(x, apply_activation=False)
            return torch.argmax(logits, dim=-1)
        else:
            return self.forward(x, apply_activation=False)  # 回归值

    def get_default_criterion(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        class_weight: Optional[torch.Tensor] = None,
    ) -> nn.Module:
        """
        返回常用损失：
          - binary: BCEWithLogitsLoss(pos_weight=…)
          - multiclass: CrossEntropyLoss(weight=class_weight)
          - regression: MSELoss
        """
        if self.task == "binary":
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif self.task == "multiclass":
            return nn.CrossEntropyLoss(weight=class_weight)
        else:
            return nn.MSELoss()

# model/cnn.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Literal, Dict, Union, Mapping, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mlp_head import MLPHead


@dataclass
class CNNConfig:
    # tokens: PAD=0, A=1,C=2,G=3,T=4,N=5  => vocab_size=5 (not counting PAD)
    vocab_size: int = 5
    pad_idx: int = 0

    # embedding dim
    embed_dim: int = 64

    # ResNet-like backbone
    channels: Sequence[int] = (64, 128, 256)   # stage channels
    blocks_per_stage: Sequence[int] = (2, 2, 2)  # residual blocks per stage
    kernel_size: int = 7
    norm: Literal["bn", "gn"] = "bn"           # BN or GN
    gn_groups: int = 8                         # used if norm=="gn"
    dropout: float = 0.2

    # pooling / head
    global_pool: Literal["max", "avg"] = "avg"
    head_hidden: int = 256
    head_dropout: float = 0.3

    # optional: use a lightweight Squeeze-and-Excitation in blocks
    use_se: bool = False
    se_ratio: float = 0.25


def _get_norm(norm: str, c: int, gn_groups: int) -> nn.Module:
    if norm == "bn":
        return nn.BatchNorm1d(c)
    if norm == "gn":
        g = min(gn_groups, c)
        # ensure divisible; fallback to 1 group if not divisible
        if c % g != 0:
            g = 1
        return nn.GroupNorm(g, c)
    raise ValueError(f"Unsupported norm: {norm}")


class SE1D(nn.Module):
    """Squeeze-and-Excitation for 1D feature maps (B,C,L)."""

    def __init__(self, channels: int, ratio: float = 0.25):
        super().__init__()
        hidden = max(4, int(channels * ratio))
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,L)
        s = x.mean(dim=-1)  # (B,C)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))  # (B,C)
        return x * s.unsqueeze(-1)


class ResidualBlock1D(nn.Module):
    """
    Pre-activation-ish residual block:
      x -> norm -> relu -> conv -> norm -> relu -> conv -> (+ skip) -> dropout
    Supports downsample via stride in the first conv.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int,
        norm: str,
        gn_groups: int,
        dropout: float,
        use_se: bool = False,
        se_ratio: float = 0.25,
    ):
        super().__init__()
        pad = kernel_size // 2

        self.n1 = _get_norm(norm, in_ch, gn_groups)
        self.c1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                            stride=stride, padding=pad, bias=False)

        self.n2 = _get_norm(norm, out_ch, gn_groups)
        self.c2 = nn.Conv1d(
            out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=pad, bias=False)

        self.se = SE1D(out_ch, ratio=se_ratio) if use_se else None
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        if in_ch != out_ch or stride != 1:
            self.proj = nn.Conv1d(
                in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)

        out = F.relu(self.n1(x))
        out = self.c1(out)
        out = F.relu(self.n2(out))
        out = self.c2(out)

        if self.se is not None:
            out = self.se(out)

        out = out + identity
        out = self.drop(out)
        return out


class GenomeCNN1D(nn.Module):
    """
    ResNet-style 1D CNN for genome windows.

    Input:
      x: LongTensor (B, L), tokens in [0..vocab_size], 0 is PAD
    Output:
      - 单任务：logits Tensor, shape (B, out_dim)
      - 多任务：dict[str, Tensor]，每个 key 对应一个任务的 logits，shape (B, out_dim_task)

    多任务用法（例如同时预测 kingdom/phylum/class/order/family）：
      model = GenomeCNN1D(out_dim={
          "kingdom": 10,
          "phylum": 30,
          "class": 80,
          "order": 200,
          "family": 500,
      })
    """

    TaskOutDims = Union[int, Sequence[int], Mapping[str, int]]

    class _Head(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.head = MLPHead(
                input_dim=in_dim,
                task="multiclass",
                num_outputs=out_dim,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(x)

    def __init__(
        self,
        out_dim: TaskOutDims = 100,
        cfg: CNNConfig = CNNConfig(),
        task_names: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        if len(cfg.channels) != len(cfg.blocks_per_stage):
            raise ValueError(
                "cfg.channels and cfg.blocks_per_stage must have same length")

        self.cfg = cfg
        self.out_dim = out_dim  # int | list[int] | dict[str,int]

        # +1 because tokens include PAD=0
        self.embedding = nn.Embedding(
            num_embeddings=cfg.vocab_size + 1,
            embedding_dim=cfg.embed_dim,
            padding_idx=cfg.pad_idx,
        )

        # stem: (B,E,L) -> (B,C0,L/2)
        stem_out = cfg.channels[0]
        self.stem = nn.Sequential(
            nn.Conv1d(cfg.embed_dim, stem_out, kernel_size=cfg.kernel_size,
                      stride=2, padding=cfg.kernel_size // 2, bias=False),
            _get_norm(cfg.norm, stem_out, cfg.gn_groups),
            nn.ReLU(inplace=True),
        )

        # stages
        stages = []
        in_ch = stem_out
        for si, (out_ch, nblk) in enumerate(zip(cfg.channels, cfg.blocks_per_stage)):
            # downsample at stage start except stage0 already has stride2 in stem;
            # keep stage0 stride=1, later stages stride=2
            stride = 1 if si == 0 else 2
            blocks = []
            blocks.append(
                ResidualBlock1D(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    kernel_size=cfg.kernel_size,
                    stride=stride,
                    norm=cfg.norm,
                    gn_groups=cfg.gn_groups,
                    dropout=cfg.dropout,
                    use_se=cfg.use_se,
                    se_ratio=cfg.se_ratio,
                )
            )
            for _ in range(nblk - 1):
                blocks.append(
                    ResidualBlock1D(
                        in_ch=out_ch,
                        out_ch=out_ch,
                        kernel_size=cfg.kernel_size,
                        stride=1,
                        norm=cfg.norm,
                        gn_groups=cfg.gn_groups,
                        dropout=cfg.dropout,
                        use_se=cfg.use_se,
                        se_ratio=cfg.se_ratio,
                    )
                )
            stages.append(nn.Sequential(*blocks))
            in_ch = out_ch

        self.stages = nn.Sequential(*stages)

        # global pooling -> fixed length
        if cfg.global_pool == "avg":
            self.global_pool = nn.AdaptiveAvgPool1d(1)
        elif cfg.global_pool == "max":
            self.global_pool = nn.AdaptiveMaxPool1d(1)
        else:
            raise ValueError(f"Unsupported global_pool: {cfg.global_pool}")

        # heads (single-task or multi-task)
        self.is_multitask = not isinstance(out_dim, int)
        if isinstance(out_dim, int):
            if out_dim <= 0:
                raise ValueError(f"out_dim must be > 0, got {out_dim}")
            self.task_names = None
            self.head = self._Head(in_ch, out_dim)
            self.heads = None
        else:
            # sequence[int] or mapping[str,int]
            if isinstance(out_dim, Mapping):
                out_dims: Dict[str, int] = {
                    str(k): int(v) for k, v in out_dim.items()}
                names = list(out_dims.keys())
            else:
                dims = [int(x) for x in out_dim]
                if task_names is not None:
                    if len(task_names) != len(dims):
                        raise ValueError(
                            "task_names length must match out_dim length")
                    names = [str(x) for x in task_names]
                else:
                    names = [f"task{i}" for i in range(len(dims))]
                out_dims = {n: d for n, d in zip(names, dims)}

            for n, d in out_dims.items():
                if d <= 0:
                    raise ValueError(
                        f"out_dim for task '{n}' must be > 0, got {d}")

            self.task_names = names
            self.head = None
            self.heads = nn.ModuleDict(
                {n: self._Head(in_ch, out_dims[n]) for n in names}
            )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        返回 pooled 特征向量 (B, C)，便于外部自定义 head。
        """
        if x.dim() != 2:
            raise ValueError(f"Expected x shape (B, L), got {tuple(x.shape)}")
        x = x.long()

        # (B,L) -> (B,L,E) -> (B,E,L)
        x = self.embedding(x).transpose(1, 2)

        # stem + res stages
        x = self.stem(x)
        x = self.stages(x)

        # (B,C,L') -> (B,C,1) -> (B,C)
        return self.global_pool(x).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        if not self.is_multitask:
            assert self.head is not None
            return self.head(feat)
        assert self.heads is not None
        return {name: head(feat) for name, head in self.heads.items()}


if __name__ == "__main__":
    # sanity check
    cfg = CNNConfig(
        vocab_size=5,
        pad_idx=0,
        embed_dim=64,
        channels=(64, 128, 256),
        blocks_per_stage=(2, 2, 2),
        kernel_size=7,
        norm="bn",
        dropout=0.1,
        global_pool="avg",
        head_hidden=256,
        head_dropout=0.3,
        use_se=False,
    )
    # single-task
    model = GenomeCNN1D(out_dim=100, cfg=cfg)
    x = torch.randint(0, 6, (4, 512))  # 0..5
    y = model(x)
    print("logits:", y.shape)  # (4, 100)

    # multi-task
    mt = GenomeCNN1D(
        out_dim={"kingdom": 10, "phylum": 20, "family": 50}, cfg=cfg)
    y2 = mt(x)
    print("multitask keys:", list(y2.keys()))
    for k, v in y2.items():
        print(k, v.shape)

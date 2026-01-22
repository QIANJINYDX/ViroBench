# models/heads.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Literal, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import inspect

TaskType = Literal["binary", "multiclass", "regression"]
PoolingType = Literal["cls", "mean", "max"]

# -------------------------
# 小工具：构建 MLP 块
# -------------------------
class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu",
                 use_bn: bool = True, dropout: float = 0.0):
        super().__init__()
        act = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
        }[activation]
        mods = [nn.Linear(in_dim, out_dim), act]
        if use_bn:
            mods.append(nn.BatchNorm1d(out_dim))
        if dropout and dropout > 0:
            mods.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
def _move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        typ = type(obj)
        return typ(_move_to_device(v, device) for v in obj)
    return obj

# -------------------------
# 可配置 MLP 分类/回归头
# -------------------------
@dataclass
class HeadConfig:
    in_dim: int
    task: TaskType = "binary"
    num_classes: Optional[int] = None    # multiclass 时必填
    hidden_dims: List[int] = (512, 128, 32)
    activation: str = "relu"             # relu/gelu/silu/tanh
    use_bn: bool = True
    dropouts: Optional[List[float]] = (0.3, 0.3, 0.0)  # 与 hidden_dims 对齐；可为 None 全0
    # binary 专用控制：1-logit BCE 或 2-logits CE
    binary_as_single_logit: bool = True

class MLPHead(nn.Module):
    """
    输入: [B, D]
    输出:
      - binary (single-logit): [B]  (logits)
      - binary (two-class):    [B, 2]  (logits)
      - multiclass:            [B, C]  (logits)
      - regression:            [B]  (pred)
    训练时若传 labels，会自动计算对应的 loss。
    """
    def __init__(self, cfg: HeadConfig):
        super().__init__()
        self.cfg = cfg

        D = cfg.in_dim
        hidden = list(cfg.hidden_dims)
        drops = list(cfg.dropouts) if cfg.dropouts is not None else [0.0] * len(hidden)

        layers: List[nn.Module] = []
        for h, p in zip(hidden, drops):
            layers.append(MLPBlock(D, h, activation=cfg.activation, use_bn=cfg.use_bn, dropout=p))
            D = h
        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()

        # output dim
        if cfg.task == "regression":
            out_dim = 1
        elif cfg.task == "binary":
            out_dim = 1 if cfg.binary_as_single_logit else 2
        else:
            assert cfg.num_classes and cfg.num_classes > 1, "multiclass 需提供 num_classes > 1"
            out_dim = int(cfg.num_classes)

        self.out = nn.Linear(D, out_dim)

    @property
    def out_dim(self) -> int:
        return self.out.out_features

    def forward(
        self,
        feats: torch.Tensor,                     # [B, D]
        labels: Optional[torch.Tensor] = None,   # 训练时传入
        class_weights: Optional[torch.Tensor] = None,  # 多分类/二分类(2-logits)时可传
        pos_weight: Optional[torch.Tensor] = None,     # 二分类(single-logit)时可传
        reduction: str = "mean",
        return_probs: bool = False               # 推理可直接要概率
    ) -> Dict[str, Any]:
        # print(f"[INFO] feats = {feats.shape}")
        x = self.trunk(feats)
        logits = self.out(x)                     # [B, out_dim]

        out: Dict[str, Any] = {"logits": logits}

        # 推理概率（可选）
        if return_probs:
            if self.cfg.task == "binary":
                if self.cfg.binary_as_single_logit:
                    out["probs"] = torch.sigmoid(logits.view(-1))  # [B]
                else:
                    out["probs"] = F.softmax(logits, dim=-1)       # [B,2]
            elif self.cfg.task == "multiclass":
                out["probs"] = F.softmax(logits, dim=-1)           # [B,C]
            else:
                out["probs"] = logits.view(-1)                     # 回归直接返回预测值

        # 计算 loss（训练/评估）
        if labels is not None:
            if self.cfg.task == "regression":
                loss = F.mse_loss(logits.view(-1).float(), labels.float(), reduction=reduction)
            elif self.cfg.task == "binary":
                if self.cfg.binary_as_single_logit:
                    # BCE with logits
                    loss = F.binary_cross_entropy_with_logits(
                        logits.view(-1), labels.float().view(-1),
                        pos_weight=pos_weight, reduction=reduction
                    )
                else:
                    # 两类 CE，可带类权重
                    loss = F.cross_entropy(
                        logits, labels.long(),
                        weight=class_weights, reduction=reduction
                    )
            else:  # multiclass
                loss = F.cross_entropy(
                    logits, labels.long(),
                    weight=class_weights, reduction=reduction
                )
            out["loss"] = loss

        return out

# -------------------------
# 包装：把 backbone + pooling + 头 组成可训练模型
# -------------------------
class SequenceTaskModel(nn.Module):
    """
    将任意 HF/自定义 backbone 的序列输出通过池化送入 MLPHead。
    - 自动探测 backbone.forward 是否支持 attention_mask / return_dict / output_hidden_states
    - 自动解析输出: last_hidden_state / hidden_states[-1] / tuple[0] / tensor
    """
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,                 # MLPHead
        pooling: PoolingType = "cls",
        exclude_special: bool = True,
        tokenizer: Optional[Any] = None,
        freeze_backbone: bool = False,
        layer_name: str = "last_hidden_state",
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.pooling = pooling
        self.exclude_special = exclude_special
        self.tokenizer = tokenizer
        self.layer_name = layer_name
        # 探测 forward 支持的参数
        try:
            sig = inspect.signature(self.backbone.forward)
            self._accepts_attn = "attention_mask" in sig.parameters
            self._accepts_return_dict = "return_dict" in sig.parameters
            self._accepts_hidden = "output_hidden_states" in sig.parameters
        except Exception:
            self._accepts_attn = False
            self._accepts_return_dict = False
            self._accepts_hidden = False

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _pool(self, hidden: torch.Tensor, input_ids: torch.Tensor,
              attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # hidden: [B, L, H]
        if self.pooling == "cls":
            return hidden[:, 0, :]

        # 构造 mask
        if attention_mask is None:
            mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
        else:
            mask = attention_mask.bool()

        if self.exclude_special and self.tokenizer is not None:
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_id is not None:
                not_pad = input_ids.ne(pad_id)
                mask = mask & not_pad

        mask = mask.unsqueeze(-1)                     # [B,L,1]
        masked = hidden.masked_fill(~mask, 0.0)

        if self.pooling == "mean":
            denom = mask.sum(dim=1).clamp_min(1)
            return masked.sum(dim=1) / denom
        elif self.pooling == "max":
            masked = masked + (~mask) * (-1e9)
            return masked.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def _parse_hidden(self, outputs: Any) -> torch.Tensor:
        """
        将不同 backbones 的输出规范为最后一层隐藏态 [B,L,H]
        """
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            return outputs.hidden_states[-1]
        if isinstance(outputs, (tuple, list)) and len(outputs) > 0 and torch.is_tensor(outputs[0]):
            return outputs[0]
        if torch.is_tensor(outputs):
            # 有些模型直接返回 [B,L,H]
            return outputs
        raise RuntimeError("无法从 backbone 输出中解析 hidden states。")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
        pos_weight: Optional[torch.Tensor] = None,
        return_probs: bool = False,
        **backbone_kwargs,
    ) -> Dict[str, Any]:
        import inspect

        # ---- 设备对齐 ----
        try:
            dev = next(self.backbone.parameters()).device
        except StopIteration:
            dev = input_ids.device
        input_ids = input_ids.to(dev)
        if attention_mask is not None:
            attention_mask = attention_mask.to(dev)

        # ---- 解析 forward 的签名，判断能传哪些关键字 ----
        try:
            sig = inspect.signature(self.backbone.forward)
            params = sig.parameters
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            accepts_kw = has_var_kw or ("input_ids" in params)
        except Exception:
            # 拿不到签名就保守：认为可以关键字方式传 input_ids
            params, has_var_kw, accepts_kw = {}, True, True

        # 构建可传入的关键字集合
        def _can_pass(name: str) -> bool:
            return has_var_kw or (name in params)

        # ---- 组装调用参数（仅包含被 forward 接受的键）----
        call_kwargs = {}
        if accepts_kw and _can_pass("input_ids"):
            call_kwargs["input_ids"] = input_ids
        if attention_mask is not None and _can_pass("attention_mask"):
            call_kwargs["attention_mask"] = attention_mask
        # 只在被支持时才传 return_dict / output_hidden_states
        if _can_pass("return_dict"):
            call_kwargs["return_dict"] = True
        if _can_pass("output_hidden_states"):
            call_kwargs["output_hidden_states"] = True
        # 透传可能存在的可选键（同样要筛选）
        for opt in ("token_type_ids", "position_ids"):
            if opt in backbone_kwargs and _can_pass(opt):
                call_kwargs[opt] = backbone_kwargs[opt]

        # ---- 前向调用：不支持关键字/没有 input_ids 形参时，改用位置参数 ----
        if accepts_kw and (("input_ids" in call_kwargs) or has_var_kw):
            outputs = self.backbone(**call_kwargs)
        else:
            # 仅位置参数（例如某些自定义/非HF backbone，或 evo2）
            _,outputs = self.backbone(input_ids,return_embeddings=True,layer_names=[self.layer_name])
            outputs = outputs[self.layer_name][0, -1, :]
        # ---- 解析隐藏层 [B, L, H] ----
        try:
            hidden = self._parse_hidden(outputs)
        except Exception:
            if isinstance(outputs, torch.Tensor) and outputs.dim() == 3:
                hidden = outputs
            elif isinstance(outputs, (tuple, list)) and len(outputs) > 0 and torch.is_tensor(outputs[0]) and outputs[0].dim() == 3:
                hidden = outputs[0]
            else:
                raise RuntimeError(
                    "Backbone outputs 不能被解析为 [B, L, H] 的隐藏表示；请检查 _parse_hidden 或 backbone 返回格式。"
                )

        feats = self._pool(hidden, input_ids, attention_mask)

        # ---- 头部设备/精度对齐 ----
        if next(self.head.parameters(), None) is not None:
            p0 = next(self.head.parameters())
            if p0.device != feats.device:
                self.head.to(feats.device)
            if p0.dtype != feats.dtype:
                self.head.to(dtype=feats.dtype)

        labels_ = labels.to(feats.device) if labels is not None else None
        class_weights_ = class_weights.to(feats.device) if class_weights is not None else None
        pos_weight_ = pos_weight.to(feats.device) if pos_weight is not None else None

        head_out = self.head(
            feats,
            labels=labels_,
            class_weights=class_weights_,
            pos_weight=pos_weight_,
            return_probs=return_probs,
        )
        return head_out

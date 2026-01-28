# models/gena_lm.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Optional, Literal, Union

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from .base_model import BaseModel

Pooling = Literal["mean", "cls", "token"]


class GenaLMModel(BaseModel):
    """
    GENA_LM embedding 适配器（仅实现 get_embedding）

    - 官方加载方式：AutoTokenizer/AutoModel + trust_remote_code=True  :contentReference[oaicite:2]{index=2}
    - 修复点：有些情况下 AutoModel 会返回 MLM head，forward 输出 MaskedLMOutput，
      它没有 last_hidden_state，需要用 hidden_states[-1]（需 output_hidden_states=True） :contentReference[oaicite:3]{index=3}
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        device: Optional[str] = None,
        torch_dtype: Union[str, torch.dtype] = "auto",
        trust_remote_code: bool = True,
    ):
        super().__init__(model_name=model_name, model_path=model_path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=trust_remote_code,
        )

        # 关键：用 AutoModel（base encoder）。但即便意外加载到 MLM head，我们也能在 get_embedding 里兜底。
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=None if torch_dtype == "auto" else torch_dtype,
        ).to(self.device)
        self.model.eval()

        # 让 hidden_states 更稳：有些模型不传 output_hidden_states 就不给
        if hasattr(self.model, "config"):
            self.model.config.output_hidden_states = True

    @staticmethod
    def _clean_gene(seq: str) -> str:
        # GENA-LM 主要面向 DNA；把非 ACGTN(U) 的字符替换为 N，减少 tokenizer 报错概率
        s = (seq or "").strip().upper()
        if not s:
            return s
        allowed = set("ACGTNU")
        return "".join(ch if ch in allowed else "N" for ch in s)

    def _resolve_max_length(self, max_length: Optional[int]) -> Optional[int]:
        if max_length is not None:
            return int(max_length)

        cfg_mpe = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)
        tok_mml = getattr(self.tokenizer, "model_max_length", None)

        # tokenizer.model_max_length 有时是一个特别大的占位值（例如 1e30）
        if isinstance(tok_mml, int) and tok_mml > 0 and tok_mml < 1_000_000:
            return tok_mml
        if isinstance(cfg_mpe, int) and cfg_mpe > 0:
            return cfg_mpe
        return None  # 让 tokenizer 自己决定

    @staticmethod
    def _get_last_hidden(outputs) -> torch.Tensor:
        """
        兼容两类输出：
        1) BaseModelOutput / BaseModelOutputWithPooling: outputs.last_hidden_state
        2) MaskedLMOutput: 没有 last_hidden_state，需要 outputs.hidden_states[-1]
        """
        h = getattr(outputs, "last_hidden_state", None)
        if h is not None:
            return h

        hs = getattr(outputs, "hidden_states", None)
        if hs is not None and len(hs) > 0:
            return hs[-1]

        # 极端兜底：有些 remote code 会塞到 base_model_output
        bmo = getattr(outputs, "base_model_output", None)
        if bmo is not None:
            h2 = getattr(bmo, "last_hidden_state", None)
            if h2 is not None:
                return h2

        raise AttributeError(
            "Cannot find last hidden states from model outputs. "
            "Try passing output_hidden_states=True / setting model.config.output_hidden_states=True."
        )

    @torch.no_grad()
    def get_embedding(
        self,
        sequences: List[str],
        layer_name: str = "mean",
        batch_size: int = 16,
        truncation: bool = True,
        max_length: Optional[int] = None,
        pool: Pooling = "mean",
        return_numpy: bool = True,
    ):
        """
        Args:
            sequences: DNA 序列列表
            layer_name:
                - "mean": mean pooling（去掉 CLS/SEP + padding）
                - "cls" : CLS 向量
                - "token": token-level 矩阵（去掉 CLS/SEP + padding），返回 List[np.ndarray]
            batch_size: 推理 batch size
            truncation/max_length: tokenization 控制

        Returns:
            - layer_name in {"mean","cls"}: np.ndarray, shape (B, H)
            - layer_name == "token": List[np.ndarray], 每条 shape (L_i, H)
        """
        if not sequences:
            return np.zeros((0, 0), dtype=np.float32)

        ln = (layer_name or "mean").lower()
        if ln in ("mean", "avg", "mean_pool", "mean_pooling"):
            pooling: Pooling = "mean"
        elif ln in ("cls", "cls_pool", "cls_pooling"):
            pooling = "cls"
        elif ln in ("token", "tokens", "last_hidden", "last_hidden_state", "matrix"):
            pooling = "token"
        else:
            raise ValueError(f"Unsupported layer_name='{layer_name}'. Use mean/cls/token.")

        max_len = self._resolve_max_length(max_length)

        sent_vecs: List[np.ndarray] = []
        token_vecs: List[np.ndarray] = []

        for st in tqdm(range(0, len(sequences), batch_size), desc="Getting GENA-LM embeddings"):
            batch_raw = sequences[st : st + batch_size]
            batch = [self._clean_gene(s) for s in batch_raw]

            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=truncation,
                max_length=max_len,
                add_special_tokens=True,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            outputs = self.model(**enc, output_hidden_states=True, return_dict=True)
            h = self._get_last_hidden(outputs)  # (B, L, H)
            attn = enc.get("attention_mask", None)  # (B, L)

            if pooling == "cls":
                vec = h[:, 0, :]
                sent_vecs.append(vec.detach().cpu().numpy())
                continue

            if pooling == "mean":
                if attn is None:
                    core = h[:, 1:-1, :]
                    vec = core.mean(dim=1)
                    sent_vecs.append(vec.detach().cpu().numpy())
                else:
                    mask = attn.clone()
                    # 去 CLS
                    mask[:, 0] = 0
                    # 去 SEP：每行最后一个 attention=1 的位置通常是 SEP
                    lengths = attn.sum(dim=1)  # includes CLS/SEP
                    sep_pos = torch.clamp(lengths - 1, min=0)
                    for i in range(mask.size(0)):
                        mask[i, sep_pos[i]] = 0

                    mask_f = mask.unsqueeze(-1).to(h.dtype)  # (B, L, 1)
                    denom = mask_f.sum(dim=1).clamp_min(1.0)
                    vec = (h * mask_f).sum(dim=1) / denom
                    sent_vecs.append(vec.detach().cpu().numpy())
                continue

            # token-level
            if attn is None:
                for i in range(h.size(0)):
                    token_vecs.append(h[i, 1:-1, :].detach().cpu().numpy())
            else:
                lengths = attn.sum(dim=1).tolist()
                for i in range(h.size(0)):
                    L = int(lengths[i])
                    if L <= 2:
                        token_vecs.append(h[i, 0:0, :].detach().cpu().numpy())
                    else:
                        token_vecs.append(h[i, 1 : L - 1, :].detach().cpu().numpy())

        if pooling == "token":
            return token_vecs

        X = np.concatenate(sent_vecs, axis=0) if sent_vecs else np.zeros((0, 0), dtype=np.float32)
        return X


if __name__ == "__main__":
    # 用法测试：python -m models.gena_lm
    CKPT = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/gena-lm-bigbird-base-t2t"  # 或你的本地目录

    seqs = [
        "ATGCGTACGTTAGC" * 50,
        "ACGT" * 200,
        "NNNNACGTACGTNNNN" * 80,
    ]

    enc = GenaLMModel(
        model_name="gena-lm",
        model_path=CKPT,
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype="auto",
    )

    X = enc.get_embedding(seqs, layer_name="mean", batch_size=4, truncation=True)
    print("mean emb shape:", X.shape)
    print("mean emb[0][:10]:", X[0][:10])

    Xc = enc.get_embedding(seqs, layer_name="cls", batch_size=4, truncation=True)
    print("cls emb shape:", Xc.shape)

    tok = enc.get_embedding([seqs[0]], layer_name="token", batch_size=1, truncation=True)
    print("token emb shape for seq0:", tok[0].shape)

# models/hyenadna_local.py
# -*- coding: utf-8 -*-
"""
基于「自训练 / ckpt_to_hf 转换」的 HyenaDNA 权重，本地加载并提供：
- get_logits: 提取序列的 logits（完整模型输出，形状 (B,L,V)）
- get_embedding: 提取序列 embedding（mean/max/cls pool）
- generate: 自回归生成 DNA 序列

模型目录需包含（由 pretrain/hyena-dna/ckpt_to_hf.py 生成）：
  - config.json
  - pytorch_model.bin（或 weights.ckpt 中的 state_dict，key 带 model. 前缀）

依赖：需能导入 pretrain/hyena-dna 的 ConvLMHeadModel（将 pretrain_root 加入 PYTHONPATH 或传参）。

注意：本文件故意不使用 pathlib，以免在「python models/hyenadna_local.py」时因 models/nt.py
遮蔽标准库 nt 导致 pathlib 导入失败。路径统一用 os.path。
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

import os, inspect
from typing import List, Optional, Literal, Dict, Any, Union

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
)

from .base_model import BaseModel

import torch.nn.functional as F

import math
import warnings
from typing import Tuple

Pooling = Literal["mean", "max", "cls", None]


# ---------- 本地字符级 Tokenizer（与 pretrain hg38_char_tokenizer 一致） ----------
class CharacterTokenizer:
    """A/C/G/T/N 字符级 tokenizer，与 pretrain/hyena-dna 的 CharacterTokenizer 对齐。"""

    def __init__(
        self,
        characters: List[str] = None,
        model_max_length: int = 1026,
        padding_side: str = "left",
    ):
        characters = characters or ["A", "C", "G", "T", "N"]
        self.characters = characters
        self.model_max_length = model_max_length
        self.padding_side = padding_side
        self._vocab_str_to_int = {
            "[CLS]": 0, "[SEP]": 1, "[BOS]": 2, "[MASK]": 3,
            "[PAD]": 4, "[RESERVED]": 5, "[UNK]": 6,
            **{ch: i + 7 for i, ch in enumerate(characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
        self.pad_token_id = 4
        self.unk_token_id = 6

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> List[str]:
        return list(text.upper())

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        padding: bool = False,
        max_length: Optional[int] = None,
        truncation: bool = False,
        return_tensors: Optional[str] = None,
    ) -> Union[List[int], Dict[str, torch.Tensor]]:
        tokens = self._tokenize(text)
        ids = [self._vocab_str_to_int.get(c, self.unk_token_id) for c in tokens]
        if add_special_tokens:
            ids = ids + [1]  # [SEP]
        max_len = max_length or self.model_max_length
        if truncation and len(ids) > max_len:
            ids = ids[:max_len]
        if padding and len(ids) < max_len:
            pad_len = max_len - len(ids)
            if self.padding_side == "left":
                ids = [self.pad_token_id] * pad_len + ids
            else:
                ids = ids + [self.pad_token_id] * pad_len
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return ids

    def decode(self, ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        if torch.is_tensor(ids):
            ids = ids.tolist()
        if isinstance(ids[0], list):
            ids = ids[0]
        special = {0, 1, 2, 3, 4, 5, 6}
        chars = []
        for i in ids:
            if skip_special_tokens and i in special:
                continue
            chars.append(self._vocab_int_to_str.get(int(i), "[UNK]"))
        return "".join(chars)

    def __call__(
        self,
        text: Union[str, List[str]],
        return_tensors: Optional[str] = None,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        add_special_tokens: bool = False,
    ) -> Dict[str, Any]:
        if isinstance(text, str):
            text = [text]
        max_len = max_length or self.model_max_length
        all_ids = []
        for t in text:
            ids = self.encode(t, add_special_tokens=add_special_tokens, truncation=truncation, max_length=max_len)
            all_ids.append(ids if isinstance(ids, list) else ids["input_ids"][0].tolist())
        if padding:
            pad_len = max(len(x) for x in all_ids)
            if max_len is not None:
                pad_len = min(pad_len, max_len)
            padded = []
            for ids in all_ids:
                if len(ids) < pad_len:
                    if self.padding_side == "left":
                        ids = [self.pad_token_id] * (pad_len - len(ids)) + ids
                    else:
                        ids = ids + [self.pad_token_id] * (pad_len - len(ids))
                else:
                    ids = ids[:pad_len]
                padded.append(ids)
            all_ids = padded
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(all_ids, dtype=torch.long)}
        return {"input_ids": all_ids}


def _ensure_pretrain_import(pretrain_root: Optional[str] = None) -> None:
    root = pretrain_root or os.environ.get("HYENA_PRETRAIN_ROOT")
    if root:
        root = os.path.abspath(os.path.normpath(root))
        if root not in sys.path:
            sys.path.insert(0, root)


def _load_model_and_tokenizer(
    model_dir: Union[str, Any],
    device: str = "cuda:0",
    pretrain_root: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
):
    """
    从 ckpt_to_hf 输出的目录加载 ConvLMHeadModel 和 tokenizer。
    返回 (model, tokenizer, config).
    """
    model_dir = os.path.abspath(os.path.normpath(str(model_dir)))
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {model_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 权重文件：优先 pytorch_model.bin，否则 weights.ckpt 里的 state_dict
    state_dict = None
    pytorch_bin = os.path.join(model_dir, "pytorch_model.bin")
    weights_ckpt = os.path.join(model_dir, "weights.ckpt")
    if os.path.exists(pytorch_bin):
        state_dict = torch.load(pytorch_bin, map_location="cpu", weights_only=True)
    elif os.path.exists(weights_ckpt):
        ckpt = torch.load(weights_ckpt, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("state_dict")
    if state_dict is None:
        raise FileNotFoundError(f"No pytorch_model.bin or weights.ckpt in {model_dir}")

    # 只保留 model.* 的 key，并去掉 "model." 前缀以匹配 ConvLMHeadModel.state_dict()
    model_state = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            model_state[k[6:]] = v  # strip "model."
    if not model_state:
        model_state = state_dict

    _ensure_pretrain_import(pretrain_root)
    try:
        from src.models.sequence.long_conv_lm import ConvLMHeadModel
    except ImportError as e:
        raise ImportError(
            "ConvLMHeadModel 需要 pretrain/hyena-dna 在 PYTHONPATH 下。"
            "设置 HYENA_PRETRAIN_ROOT 或传入 pretrain_root，例如：\n"
            "  export HYENA_PRETRAIN_ROOT=/path/to/GeneShield/pretrain/hyena-dna"
        ) from e

    # 构建 layer 配置（ConvLMHeadModel 需要 layer 为 dict，且会传给 instantiate）
    # 不要往 layer_cfg 里加 d_model：Block 会以 mixer_cls(d_model) 传入，否则会报 multiple values for d_model
    layer_cfg = config.get("layer") or {}
    layer_cfg = dict(layer_cfg)
    if "_name_" not in layer_cfg:
        layer_cfg["_name_"] = "hyena"

    model = ConvLMHeadModel(
        d_model=config["d_model"],
        n_layer=config["n_layer"],
        d_inner=config.get("d_inner", 4 * config["d_model"]),
        vocab_size=config["vocab_size"],
        layer=layer_cfg,
        resid_dropout=config.get("resid_dropout", 0.0),
        embed_dropout=config.get("embed_dropout", 0.1),
        layer_norm_epsilon=config.get("layer_norm_epsilon", 1e-5),
        pad_vocab_size_multiple=config.get("pad_vocab_size_multiple", 8),
        residual_in_fp32=config.get("residual_in_fp32", True),
        checkpoint_mixer=config.get("checkpoint_mixer", False),
        checkpoint_mlp=config.get("checkpoint_mlp", False),
        process_group=None,
    )

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        print("[HyenaDNALocal] missing keys (expected if only backbone):", missing[:5], "..." if len(missing) > 5 else "")
    if unexpected:
        print("[HyenaDNALocal] unexpected keys:", unexpected[:5], "..." if len(unexpected) > 5 else "")

    model.eval()
    if dtype is not None:
        model = model.to(dtype)
    model = model.to(device)

    max_len = config.get("max_position_embeddings", 1024) + 2
    tokenizer_config_path = os.path.join(model_dir, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path, "r", encoding="utf-8") as f:
            tok_cfg = json.load(f)
        tokenizer = CharacterTokenizer(
            characters=tok_cfg.get("characters", ["A", "C", "G", "T", "N"]),
            model_max_length=tok_cfg.get("model_max_length", max_len),
            padding_side=tok_cfg.get("padding_side", "left"),
        )
    else:
        tokenizer = CharacterTokenizer(model_max_length=max_len, padding_side="left")

    return model, tokenizer, config


class HyenaDNALocal:
    """
    从「自训练 + ckpt_to_hf」得到的目录加载 HyenaDNA，提供 get_embedding 与 generate。
    参考官方 HyenaDNA 的 embedding / 生成接口。
    """

    def __init__(
        self,
        model_dir: Union[str, Any],
        device: Optional[str] = None,
        pretrain_root: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = torch.float32,
    ):
        self.model_dir = os.path.abspath(os.path.normpath(str(model_dir)))
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pretrain_root = pretrain_root
        self.dtype = torch_dtype

        self.model, self.tokenizer, self.config = _load_model_and_tokenizer(
            self.model_dir,
            device=self.device,
            pretrain_root=self.pretrain_root,
            dtype=self.dtype,
        )
        self.model_max_length = self.config.get("max_position_embeddings", 1024) + 2
        self.d_model = self.config.get("d_model", 128)

    @torch.no_grad()
    def get_logits(
        self,
        sequences: Union[str, List[str]],
        batch_size: int = 1,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_numpy: bool = False,
    ) -> Union[List[torch.Tensor], np.ndarray]:
        """
        提取序列的 logits（完整模型前向，非 backbone 的 hidden states）。
        返回 list 时：每条序列一个 tensor，形状 (L_i, vocab_size)；
        若 return_numpy=True：返回 padded 的 (N, max_L, vocab_size)。
        """
        if isinstance(sequences, str):
            sequences = [sequences]
        max_len = max_length or self.model_max_length
        all_logits = []

        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=truncation,
                max_length=max_len,
                add_special_tokens=False,
            )
            input_ids = enc["input_ids"].to(self.device)

            out = self.model(input_ids=input_ids)
            logits = out[0].logits if isinstance(out, (tuple, list)) and len(out) > 0 else out.logits
            # logits: (B, L, V)
            B, L, V = logits.shape
            for i in range(B):
                # 去掉 padding：按 input 有效长度截断（这里简化：用当前 batch 的 max 作为 L，不二次 pad）
                all_logits.append(logits[i].float().cpu())

        if return_numpy:
            max_L = max(t.size(0) for t in all_logits)
            V = all_logits[0].size(-1)
            pad_logits = np.zeros((len(all_logits), max_L, V), dtype=np.float32)
            for i, t in enumerate(all_logits):
                pad_logits[i, : t.size(0), :] = t.numpy()
            return pad_logits
        return all_logits

    @torch.no_grad()
    def get_embedding(
        self,
        sequences: Union[str, List[str]],
        batch_size: int = 1,
        pool: Pooling = "mean",
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_numpy: bool = True,
    ) -> Union[np.ndarray, List[torch.Tensor]]:
        """
        提取序列的 embedding，对齐官方 HyenaDNA 用法：
          tok_seq = tokenizer(sequence)[\"input_ids\"]
          tok_seq = torch.LongTensor(tok_seq).unsqueeze(0).to(device)
          with torch.inference_mode():
              embeddings = model(tok_seq)   # 此处官方为 backbone 输出 (B, L, d_model)

        - pool is None: 返回原始 per-position embeddings，形状 (B, L, d_model) 或 list[(L_i, d_model)]。
        - pool "mean" | "max" | "cls": 返回池化后的 (N, d_model)。
        """
        if isinstance(sequences, str):
            sequences = [sequences]
        max_len = max_length or self.model_max_length
        self.model.to(self.device)
        self.model.eval()

        all_embs = []
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            # 官方风格: tokenizer(sequence)["input_ids"] -> LongTensor -> device
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=truncation,
                max_length=max_len,
                add_special_tokens=False,
            )
            tok_seq = enc["input_ids"].long().to(self.device)

            with torch.inference_mode():
                # 官方: embeddings = model(tok_seq)；本地为 LM 故用 backbone 得到相同形状 (B, L, d_model)
                embeddings = self.model.backbone(tok_seq)

            B, L, H = embeddings.shape

            if pool is None:
                # 与官方一致：返回原始 embeddings，不池化
                for i in range(B):
                    all_embs.append(embeddings[i].float().cpu())
            else:
                if pool == "cls":
                    pooled = embeddings[:, 0, :]
                elif pool == "max":
                    pooled = embeddings.max(dim=1)[0]
                else:
                    pooled = embeddings.mean(dim=1)
                if return_numpy:
                    all_embs.append(pooled.float().cpu().numpy())
                else:
                    all_embs.extend([pooled[i] for i in range(B)])

        if pool is None:
            if return_numpy:
                max_L = max(t.size(0) for t in all_embs)
                out = np.zeros((len(all_embs), max_L, all_embs[0].size(-1)), dtype=np.float32)
                for i, t in enumerate(all_embs):
                    out[i, : t.size(0), :] = t.numpy()
                return out
            return all_embs

        if return_numpy:
            return np.concatenate(all_embs, axis=0)
        return all_embs

    @staticmethod
    def _top_k_top_p_filtering(
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
        min_tokens_to_keep: int = 1,
    ) -> torch.Tensor:
        if logits.dim() != 1:
            logits = logits.view(-1)
        if top_k > 0:
            top_k = min(max(int(top_k), min_tokens_to_keep), logits.size(-1))
            kth = torch.topk(logits, top_k).values[-1]
            logits = torch.where(logits < kth, torch.full_like(logits, -float("inf")), logits)
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(probs, dim=-1)
            cutoff = cumprobs > float(top_p)
            cutoff[:min_tokens_to_keep] = False
            sorted_logits = torch.where(cutoff, torch.full_like(sorted_logits, -float("inf")), sorted_logits)
            new_logits = torch.full_like(logits, -float("inf"))
            new_logits.scatter_(0, sorted_idx, sorted_logits)
            logits = new_logits
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt_seqs: Union[str, List[str]] = "ACGT",
        n_samples: int = 1,
        n_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int = 4,
        top_p: float = 1.0,
        max_prompt_tokens: Optional[int] = None,
        return_list: bool = True,
    ) -> Union[List[str], str]:
        """
        自回归生成 DNA 序列。prompt_seqs 为 str 或 List[str]；若为 str 则复制为 n_samples 条。
        """
        if isinstance(prompt_seqs, str):
            prompts = [prompt_seqs] * int(n_samples)
        else:
            prompts = list(prompt_seqs)
            if len(prompts) == 1 and int(n_samples) > 1:
                prompts = prompts * int(n_samples)

        self.model.eval()
        outputs: List[str] = []

        with torch.inference_mode():
            for p in prompts:
                enc = self.tokenizer.encode(p, add_special_tokens=False)
                if isinstance(enc, dict):
                    input_ids = enc["input_ids"].to(self.device)
                else:
                    input_ids = torch.tensor([enc], dtype=torch.long, device=self.device)

                if max_prompt_tokens is not None and input_ids.size(1) > int(max_prompt_tokens):
                    input_ids = input_ids[:, -int(max_prompt_tokens):]

                for _ in range(int(n_tokens)):
                    out = self.model(input_ids=input_ids)
                    # ConvLMHeadModel returns (CausalLMOutput(logits=...), None)
                    logits = out[0].logits if isinstance(out, (tuple, list)) and len(out) > 0 else out.logits
                    next_logits = logits[0, -1, :].float()
                    if temperature and float(temperature) > 0:
                        next_logits = next_logits / float(temperature)
                    next_logits = self._top_k_top_p_filtering(next_logits, top_k=int(top_k), top_p=float(top_p))
                    probs = F.softmax(next_logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                    input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)

                txt = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                txt = (txt or "").replace(" ", "").strip()
                outputs.append(txt)

        if return_list:
            return outputs
        return outputs[0] if outputs else ""
    # Lazy load: CausalLM for generate / PPL
    # =============================
    def _infer_model_input_device(self, model) -> torch.device:
        # device_map="auto" 时，inputs 通常送到 embedding 所在设备即可
        if hasattr(model, "device") and model.device is not None:
            try:
                return torch.device(model.device)
            except Exception:
                pass
        try:
            return next(model.parameters()).device
        except Exception:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _ensure_lm_model(self):
        """
        生成 / PPL 使用已加载的 ConvLMHeadModel（self.model）。
        本地 ckpt 的 config.model_type 为 "hyenadna"，不在 HuggingFace Auto 注册表中，
        故不使用 AutoModelForCausalLM.from_pretrained，直接复用 _load_model_and_tokenizer 加载的完整模型。
        """
        if hasattr(self, "lm_model") and self.lm_model is not None:
            return

        self.lm_model = self.model
        if getattr(self, "device_map", None) is None:
            self.lm_model.to(self.device)
        self.lm_model.eval()

    @staticmethod
    def _get_pad_id_fallback(tokenizer) -> int:
        """
        tokenizer 可能没有 pad_token_id。
        这里返回一个合法 id 用于右侧 padding（loss 会用长度 mask 排除 padding）。
        注意：不使用 (input_ids==pad_id) 来做 mask，避免 pad_id 与真实 token 冲突。
        """
        for name in ("pad_token_id", "pad_id"):
            if hasattr(tokenizer, name):
                v = getattr(tokenizer, name)
                if v is not None:
                    try:
                        return int(v)
                    except Exception:
                        pass

        # fallback: 用 'A' 的 token id
        try:
            ids = tokenizer("A", add_special_tokens=False)["input_ids"]
            if ids:
                return int(ids[0])
        except Exception:
            pass
        return 0

    # =============================
    # Generation (manual sampling)
    # =============================
    @staticmethod
    def _top_k_top_p_filtering(
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
        min_tokens_to_keep: int = 1,
    ) -> torch.Tensor:
        """
        logits: (V,) 或 (B, V)；对 (B,V) 在最后一维上逐行做 top-k/top-p
        """
        if logits.dim() == 1:
            # 单条
            next_logits = logits
            if top_k > 0:
                top_k = min(max(int(top_k), min_tokens_to_keep), next_logits.size(-1))
                kth = torch.topk(next_logits, top_k).values[-1]
                next_logits = torch.where(next_logits < kth, torch.full_like(next_logits, -float("inf")), next_logits)
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(probs, dim=-1)
                cutoff = cumprobs > float(top_p)
                cutoff[:min_tokens_to_keep] = False
                sorted_logits = torch.where(cutoff, torch.full_like(sorted_logits, -float("inf")), sorted_logits)
                new_logits = torch.full_like(next_logits, -float("inf"))
                new_logits.scatter_(0, sorted_idx, sorted_logits)
                next_logits = new_logits
            return next_logits

        # (B, V)
        B, V = logits.shape
        out = logits.clone()
        if top_k > 0:
            top_k = min(max(int(top_k), min_tokens_to_keep), V)
            kth = torch.topk(logits, top_k, dim=-1).values[:, -1]  # (B,)
            out = torch.where(logits < kth.unsqueeze(-1), torch.full_like(out, -float("inf")), out)
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(out, descending=True, dim=-1)
            probs = F.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(probs, dim=-1)
            cutoff = cumprobs > float(top_p)
            cutoff[:, :min_tokens_to_keep] = False
            sorted_logits = torch.where(cutoff, torch.full_like(sorted_logits, -float("inf")), sorted_logits)
            new_logits = torch.full_like(out, -float("inf"))
            new_logits.scatter_(1, sorted_idx, sorted_logits)
            out = new_logits
        return out

    @torch.no_grad()
    def generate(
        self,
        prompt_seqs: Union[str, List[str]] = "ACGT",
        n_samples: int = 1,
        n_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int = 4,
        top_p: float = 1.0,
        max_prompt_tokens: Optional[int] = None,
        return_list: bool = True,
    ) -> Union[List[str], str]:
        """
        HyenaDNA 序列生成（手写 sampling，不依赖 transformers.generate）
        - 支持 prompt 为 str 或 List[str]
        - 若 prompt 为 str：会复制为 n_samples 条生成
        - 当 len(prompts) > 1 时使用批量化生成（一次前向多条），提高 GPU 利用率
        """
        self._ensure_lm_model()
        model = self.lm_model
        tok = self.tokenizer
        dev = self._infer_model_input_device(model)
        pad_id = self._get_pad_id_fallback(tok)

        # 规范 prompts
        if isinstance(prompt_seqs, str):
            prompts = [prompt_seqs] * int(n_samples)
        else:
            prompts = list(prompt_seqs)
            if len(prompts) == 1 and int(n_samples) > 1:
                prompts = prompts * int(n_samples)

        n_tokens = int(n_tokens)
        max_prompt_tokens = int(max_prompt_tokens) if max_prompt_tokens is not None else None
        outputs: List[str] = []
        model.eval()

        def _decode_one(ids: torch.Tensor) -> str:
            txt = tok.decode(ids, skip_special_tokens=True)
            return (txt or "").replace(" ", "").strip()

        with torch.inference_mode():
            if len(prompts) == 1:
                # 单条：保持原有逻辑
                p = prompts[0]
                enc = tok(p, return_tensors="pt", add_special_tokens=False)
                input_ids = enc["input_ids"].to(dev)
                if max_prompt_tokens is not None and input_ids.size(1) > max_prompt_tokens:
                    input_ids = input_ids[:, -max_prompt_tokens:]
                for _ in range(n_tokens):
                    out = model(input_ids=input_ids)
                    out = out[0] if isinstance(out, (tuple, list)) and len(out) > 0 else out
                    logits = out.logits
                    next_logits = logits[0, -1, :].float()
                    if temperature is not None and float(temperature) > 0:
                        next_logits = next_logits / float(temperature)
                    next_logits = self._top_k_top_p_filtering(next_logits, top_k=int(top_k), top_p=float(top_p))
                    probs = F.softmax(next_logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                    input_ids = torch.cat([input_ids, next_id.view(1, 1)], dim=1)
                outputs.append(_decode_one(input_ids[0]))
            else:
                # 批量化：左 padding，一次前向 (B, L)
                enc_list = tok(prompts, return_tensors="pt", add_special_tokens=False, padding=True, truncation=False)
                if "input_ids" in enc_list:
                    input_ids = enc_list["input_ids"].to(dev)
                else:
                    # 若 tokenizer 不返回 padding，手动 pad
                    ids_list = []
                    for p in prompts:
                        raw = tok(p, add_special_tokens=False)["input_ids"]
                        if torch.is_tensor(raw):
                            raw = raw.tolist()
                        if raw and isinstance(raw[0], (list, torch.Tensor)):
                            raw = raw[0].tolist() if torch.is_tensor(raw[0]) else raw[0]
                        ids_list.append(raw)
                    max_len = max(len(ids) for ids in ids_list)
                    pad_id_int = int(pad_id) if torch.is_tensor(pad_id) else int(pad_id)
                    padded = []
                    for ids in ids_list:
                        pad_len = max_len - len(ids)
                        padded.append([pad_id_int] * pad_len + ids)
                    input_ids = torch.tensor(padded, dtype=torch.long, device=dev)

                if max_prompt_tokens is not None and input_ids.size(1) > max_prompt_tokens:
                    input_ids = input_ids[:, -max_prompt_tokens:]

                B = input_ids.size(0)
                for _ in range(n_tokens):
                    out = model(input_ids=input_ids)
                    out = out[0] if isinstance(out, (tuple, list)) and len(out) > 0 else out
                    logits = out.logits
                    next_logits = logits[:, -1, :].float()
                    if temperature is not None and float(temperature) > 0:
                        next_logits = next_logits / float(temperature)
                    next_logits = self._top_k_top_p_filtering(next_logits, top_k=int(top_k), top_p=float(top_p))
                    probs = F.softmax(next_logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                    input_ids = torch.cat([input_ids, next_id], dim=1)

                for i in range(B):
                    outputs.append(_decode_one(input_ids[i]))

        if return_list:
            return outputs
        return outputs[0] if outputs else ""

    # =============================
    # PPL (full / conditional)
    # =============================
    @staticmethod
    def _safe_exp(nll: float, max_nll: float = 700.0) -> float:
        if math.isnan(nll):
            return float("nan")
        if math.isinf(nll):
            return float("inf") if nll > 0 else 0.0
        if nll > max_nll:
            return float("inf")
        try:
            return math.exp(nll)
        except OverflowError:
            return float("inf")

    @torch.no_grad()
    def get_ppl(
        self,
        sequences: Union[str, List[str]],
        batch_size: int = 1,
        max_length: Optional[int] = None,
        prompt_len_chars: Optional[int] = 128,  # None => full ppl；否则 conditional ppl
        use_cuda: bool = True,
        ppl_mode: Literal["token", "char"] = "token",
        return_details: bool = True,
    ) -> Union[List[Dict[str, Any]], float, List[float]]:
        """
        HyenaDNA PPL (full / conditional)，并补齐统计字段：
        - token 口径：avg_nll_token = total_nll / token_count, ppl_token = exp(avg_nll_token)
        - char  口径：avg_nll_char  = total_nll / char_count,  ppl_char  = exp(avg_nll_char)
        - 主输出 ppl 由 ppl_mode 决定（token 或 char）

        conditional 规则（按字符切分）：
          prompt = seq[:prompt_len_chars]
          continuation = seq[prompt_len_chars:]
        只对 continuation 的 token 计算 NLL。
        """

        # ---- ensure causal LM ----
        self._ensure_lm_model()
        model = self.lm_model
        tok = self.tokenizer
        model.eval()

        # ---- normalize input ----
        if isinstance(sequences, str):
            seq_list = [sequences]
            is_single = True
        else:
            seq_list = list(sequences)
            is_single = False

        def _clean_seq(s: str) -> str:
            # 生成结果常见 "A T G ..."；PPL 计算前统一清理空格/换行
            return (s or "").replace(" ", "").replace("\n", "").replace("\t", "").strip().upper()

        seq_list = [_clean_seq(s) for s in seq_list]

        # ---- device ----
        target_device = torch.device("cuda:0") if (use_cuda and torch.cuda.is_available()) else torch.device("cpu")
        # device_map="auto" 时，模型可能分片；inputs 送到 model input device
        dev = self._infer_model_input_device(model)
        if dev.type == "cpu" and target_device.type == "cuda":
            # 尝试把整模型搬到 cuda（只有在 device_map=None 时通常可行）
            try:
                if getattr(self, "device_map", None) is None:
                    model.to(target_device)
                    dev = target_device
            except Exception:
                pass

        pad_id = self._get_pad_id_fallback(tok)

        # 序列长度不得超过模型支持的 max（config 中 max_position_embeddings + 2 或 layer.l_max），否则 backbone 内会维度不匹配
        effective_max_length = max_length if max_length is not None else getattr(self, "model_max_length", 1026)
        if effective_max_length is not None:
            effective_max_length = int(effective_max_length)

        # ---- pre-tokenize & prompt token lengths ----
        seq_infos: List[Tuple[int, str, List[int], int, int, int]] = []
        # tuple: (orig_i, seq, ids, prompt_tok_len, seq_char_len, cont_char_len)
        def _to_flat_ids(raw):
            """CharacterTokenizer 对单条 str 返回 input_ids 为 [[...]]，转为单条 int 列表。"""
            if not raw:
                return []
            x = raw[0] if isinstance(raw[0], (list, torch.Tensor)) else raw
            if torch.is_tensor(x):
                x = x.tolist()
            return list(map(int, x))

        for i, s in enumerate(seq_list):
            try:
                raw_ids = tok(s, add_special_tokens=False)["input_ids"]
                ids = _to_flat_ids(raw_ids)
                if effective_max_length is not None and len(ids) > effective_max_length:
                    ids = ids[:effective_max_length]

                seq_char_len = len(s)

                if prompt_len_chars is not None:
                    p_chars = min(int(prompt_len_chars), seq_char_len)
                    cont_char_len = max(seq_char_len - p_chars, 0)

                    p_str = s[:p_chars]
                    p_ids = _to_flat_ids(tok(p_str, add_special_tokens=False)["input_ids"])
                    prompt_tok_len = min(len(p_ids), len(ids))
                else:
                    p_chars = 0
                    cont_char_len = max(seq_char_len - 1, 0)  # full ppl: 预测从 token1 开始，字符上通常对应 L-1
                    prompt_tok_len = 0

                seq_infos.append((i, s, ids, int(prompt_tok_len), int(seq_char_len), int(cont_char_len)))
            except Exception as e:
                warnings.warn(f"Tokenize failed at {i}: {str(e)[:120]}")
                seq_infos.append((i, s, [], 0, len(s), 0))

        # length sort to save padding
        seq_infos_sorted = sorted(seq_infos, key=lambda x: len(x[2]))

        results: List[Dict[str, Any]] = []

        with torch.inference_mode():
            for st in range(0, len(seq_infos_sorted), batch_size):
                batch = seq_infos_sorted[st: st + batch_size]
                B = len(batch)
                lens = [len(x[2]) for x in batch]
                max_len = max(lens) if lens else 0

                # too short: cannot compute next-token loss
                if max_len <= 1:
                    for orig_i, s, ids, p_tok, seq_char_len, cont_char_len in batch:
                        results.append({
                            "sequence_id": orig_i,
                            "sequence_chars": seq_char_len,
                            "prompt_len_chars": int(prompt_len_chars) if prompt_len_chars is not None else 0,
                            "char_count": int(cont_char_len),

                            "sequence_tokens": len(ids),
                            "prompt_tokens": int(p_tok),
                            "token_count": 0,

                            "avg_nll_token": float("nan"),
                            "avg_nll_char": float("nan"),
                            "ppl_token": float("nan"),
                            "ppl_char": float("nan"),
                            "ppl": float("nan"),
                            "mode": ppl_mode,
                            "error": "too_short",
                        })
                    continue

                # pad + valid mask (do NOT rely on pad_id equality for mask)
                input_ids = torch.full((B, max_len), int(pad_id), dtype=torch.long)
                valid_mask = torch.zeros((B, max_len), dtype=torch.bool)
                prompt_lens_tok = [0] * B
                seq_char_lens = [0] * B
                cont_char_lens = [0] * B

                for r, (orig_i, s, ids, p_tok, seq_char_len, cont_char_len) in enumerate(batch):
                    L = len(ids)
                    if L > 0:
                        input_ids[r, :L] = torch.tensor(ids, dtype=torch.long)
                        valid_mask[r, :L] = True
                    prompt_lens_tok[r] = int(p_tok)
                    seq_char_lens[r] = int(seq_char_len)
                    cont_char_lens[r] = int(cont_char_len)

                input_ids = input_ids.to(dev, non_blocking=True)
                valid_mask = valid_mask.to(dev, non_blocking=True)

                out = model(input_ids=input_ids)
                out = out[0] if isinstance(out, (tuple, list)) and len(out) > 0 else out
                logits = out.logits  # (B, L, V)
                if logits.dim() != 3:
                    raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
                logits = logits.float()

                # shift
                shift_logits = logits[:, :-1, :].contiguous()     # (B, L-1, V)
                shift_labels = input_ids[:, 1:].contiguous()      # (B, L-1)
                shift_valid  = valid_mask[:, 1:].contiguous()     # (B, L-1)

                # build final mask: full or conditional
                if prompt_len_chars is None:
                    final_mask = shift_valid
                else:
                    cont_mask = torch.zeros_like(shift_valid)
                    for r in range(B):
                        seq_len_tok = int(valid_mask[r].sum().item())
                        if seq_len_tok <= 1:
                            continue
                        start = max(prompt_lens_tok[r] - 1, 0)  # shift index
                        end = seq_len_tok - 1                   # exclusive in shift space
                        if start < end:
                            cont_mask[r, start:end] = True
                    final_mask = shift_valid & cont_mask

                V = shift_logits.size(-1)
                token_nll = F.cross_entropy(
                    shift_logits.view(-1, V),
                    shift_labels.view(-1),
                    reduction="none",
                ).view(B, -1)  # (B, L-1)

                nll_sum = (token_nll * final_mask).sum(dim=1)   # (B,)
                tok_cnt = final_mask.sum(dim=1)                 # (B,)

                for r, (orig_i, s, ids, p_tok, seq_char_len, cont_char_len) in enumerate(batch):
                    c_tok = int(tok_cnt[r].item())
                    if c_tok <= 0:
                        results.append({
                            "sequence_id": orig_i,
                            "sequence_chars": int(seq_char_len),
                            "prompt_len_chars": int(prompt_len_chars) if prompt_len_chars is not None else 0,
                            "char_count": int(cont_char_len),

                            "sequence_tokens": len(ids),
                            "prompt_tokens": int(p_tok),
                            "token_count": 0,

                            "avg_nll_token": float("nan"),
                            "avg_nll_char": float("nan"),
                            "ppl_token": float("nan"),
                            "ppl_char": float("nan"),
                            "ppl": float("nan"),
                            "mode": ppl_mode,
                            "error": "no_continuation_tokens" if prompt_len_chars is not None else "no_valid_tokens",
                        })
                        continue

                    total_nll = float(nll_sum[r].item())
                    avg_nll_token = total_nll / float(c_tok)
                    ppl_token = float(self._safe_exp(avg_nll_token))

                    # char-level derived metric (for BPB / normalization)
                    c_char = int(cont_char_len) if prompt_len_chars is not None else max(int(seq_char_len) - 1, 0)
                    if c_char > 0:
                        avg_nll_char = total_nll / float(c_char)
                        ppl_char = float(self._safe_exp(avg_nll_char))
                    else:
                        avg_nll_char = float("nan")
                        ppl_char = float("nan")

                    ppl = ppl_char if ppl_mode == "char" else ppl_token

                    results.append({
                        "sequence_id": orig_i,

                        "sequence_chars": int(seq_char_len),
                        "prompt_len_chars": int(prompt_len_chars) if prompt_len_chars is not None else 0,
                        "char_count": int(c_char),

                        "sequence_tokens": len(ids),
                        "prompt_tokens": int(p_tok),
                        "token_count": int(c_tok),

                        "avg_nll_token": float(avg_nll_token),
                        "avg_nll_char": float(avg_nll_char),

                        "ppl_token": float(ppl_token),
                        "ppl_char": float(ppl_char),

                        "ppl": float(ppl),
                        "mode": ppl_mode,
                    })

        # restore original order
        results.sort(key=lambda x: x["sequence_id"])

        if return_details:
            return results

        ppl_list = [float(x.get("ppl", float("nan"))) for x in results]
        if is_single:
            return ppl_list[0] if ppl_list else float("nan")
        return ppl_list


if __name__ == "__main__":
    import argparse
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None, help="Path to converted HF dir (config.json + pytorch_model.bin)")
    parser.add_argument("--pretrain_root", type=str, default=None, help="Path to pretrain/hyena-dna (for ConvLMHeadModel)")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if not args.model_dir:
        default_dir = os.path.join(_root, "pretrain", "hyena-dna", "hyena_hg38_hf")
        if os.path.isdir(default_dir):
            args.model_dir = default_dir
        else:
            print("Usage: --model_dir /path/to/hyena_hg38_hf [--pretrain_root /path/to/pretrain/hyena-dna]")
            sys.exit(1)

    if args.pretrain_root is None:
        args.pretrain_root = os.path.join(_root, "pretrain", "hyena-dna")

    model = HyenaDNALocal(
        model_dir=args.model_dir,
        device=args.device,
        pretrain_root=args.pretrain_root,
    )

    # logits
    seqs = ["ACGT" * 64, "ATGC" * 64]
    logits_list = model.get_logits(seqs, batch_size=2, return_numpy=False)
    print("logits (list): num_seqs =", len(logits_list), "first shape:", logits_list[0].shape)
    logits_pad = model.get_logits(seqs, batch_size=2, return_numpy=True)
    print("logits (numpy padded) shape:", logits_pad.shape)

    # embedding（可选）
    emb = model.get_embedding(seqs, batch_size=2, pool="mean")
    print("embedding shape:", emb.shape)

    # generate
    gen = model.generate(prompt_seqs="ACGT", n_samples=2, n_tokens=50, temperature=1.0, top_k=4, top_p=1.0)
    print("generated:", gen)

    # PPL（与 hyenadna_model.py 末尾测试一致）
    seqs = ["ACGT" * 200, "ATGC" * 220]
    ppl_full = model.get_ppl(seqs, batch_size=2, return_details=False)
    print("ppl_full:", ppl_full)

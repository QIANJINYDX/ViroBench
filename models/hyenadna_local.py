# models/hyenadna_local.py
# -*- coding: utf-8 -*-
"""
基于「自训练 / ckpt_to_hf 转换」的 HyenaDNA 权重，本地加载并提供：
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

Pooling = Literal["mean", "max", "cls"]


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
        提取序列的向量表示。pool: "mean" | "max" | "cls"。
        返回 (N, d_model) 或张量列表。
        """
        if isinstance(sequences, str):
            sequences = [sequences]
        max_len = max_length or self.model_max_length
        all_embs = []

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

            hidden = self.model.backbone(input_ids)  # (B, L, d_model)
            B, L, H = hidden.shape

            if pool == "cls":
                pooled = hidden[:, 0, :]
            elif pool == "max":
                pooled = hidden.max(dim=1)[0]
            else:
                pooled = hidden.mean(dim=1)

            if return_numpy:
                all_embs.append(pooled.float().cpu().numpy())
            else:
                all_embs.extend([pooled[i] for i in range(B)])

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

    # embedding
    seqs = ["ACGT" * 64, "ATGC" * 64]
    emb = model.get_embedding(seqs, batch_size=2, pool="mean")
    print("embedding shape:", emb.shape)

    # generate
    gen = model.generate(prompt_seqs="ACGT", n_samples=2, n_tokens=50, temperature=1.0, top_k=4, top_p=1.0)
    print("generated:", gen)

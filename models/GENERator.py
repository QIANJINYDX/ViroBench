# models/generator_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Optional, Literal, Union, Dict, Any


try:
    import transformers
    from transformers import PreTrainedTokenizer

    if not hasattr(PreTrainedTokenizer, "_orig_convert_ids_to_tokens"):
        PreTrainedTokenizer._orig_convert_ids_to_tokens = PreTrainedTokenizer.convert_ids_to_tokens

    def _patched_convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        if not hasattr(self, "_added_tokens_decoder"):
            self._added_tokens_decoder = {}
        return self._orig_convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)

    PreTrainedTokenizer.convert_ids_to_tokens = _patched_convert_ids_to_tokens
    print("[GENERatorModel] Applied compatibility patch for DNAKmerTokenizer.")
except Exception as e:
    print(f"[GENERatorModel] Warning: Failed to apply patch: {e}")

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from .base_model import BaseModel

Pooling = Literal["mean", "last_token", "max", "eos"]


class GENERatorModel(BaseModel):
    """
    GENERator 适配器 (用于 embedding / PLL)

    关键修复：
    - 6-mer 对齐默认改为右侧补齐（不改变前缀 k-mer 切分）
    - pooling/pool 参数兼容
    - pooling 排除 special tokens（可用时）
    - last_token/eos 使用 last non-pad index，独立于 padding_side
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        device: Optional[str] = None,
        max_len: int = 16384,
        tokenizer_padding_side: Literal["left", "right"] = "right",
        kmer_size: int = 6,
        kmer_pad_char: str = "A",
        kmer_pad_side: Literal["left", "right"] = "right",  # ✅ 默认 right
    ):
        super().__init__(model_name, model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.user_max_len = int(max_len)
        self.max_len = int(max_len)

        self.tokenizer_padding_side = tokenizer_padding_side
        self.kmer_size = int(kmer_size)
        self.kmer_pad_char = str(kmer_pad_char)
        self.kmer_pad_side = kmer_pad_side

        self._load_model()

    def _load_model(self):
        print(f"Loading {self.model_name} from {self.model_path}...")

        # 1) tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        self.tokenizer.padding_side = self.tokenizer_padding_side
        self.tokenizer.truncation_side = self.tokenizer_padding_side

        if self.tokenizer.pad_token is None:
            # GENERator 常见做法：pad_token = eos_token
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"Set pad_token to eos_token: {self.tokenizer.pad_token}")

        # 2) config（用于拿 max_position_embeddings）
        try:
            cfg = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        except Exception:
            cfg = None

        # 3) model
        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=None,
        )
        self.model.config.use_cache = False
        self.model.to(self.device).eval()

        # 4) max_len 限制到模型支持的上限（如果有）
        model_max_pos = None
        if cfg is not None and hasattr(cfg, "max_position_embeddings"):
            model_max_pos = int(cfg.max_position_embeddings)
        elif hasattr(self.model.config, "max_position_embeddings"):
            model_max_pos = int(self.model.config.max_position_embeddings)

        if model_max_pos is not None and model_max_pos > 0:
            self.max_len = min(self.user_max_len, model_max_pos)
        else:
            self.max_len = self.user_max_len

        print(f"[{self.model_name}] loaded on {self.device} (dtype={dtype}, max_len={self.max_len})")

    def _preprocess(self, seq: str) -> str:
        """
        预处理（与官方一致）：
        1) 大写 + 去空白
        2) 6-mer 对齐：截断到长度为 6 的倍数（不补齐）
        """
        seq = seq.strip().upper()
        if len(seq) == 0:
            return seq
        return seq[: len(seq) // self.kmer_size * self.kmer_size]

    @staticmethod
    def _last_nonpad_index(attention_mask: torch.Tensor) -> torch.Tensor:
        """
        返回每个样本最后一个非 pad token 的 index（B,）
        attention_mask: (B, L) 0/1
        """
        # sum-1 就是最后一个 1 的位置（假设至少有一个 1）
        idx = attention_mask.long().sum(dim=1) - 1
        return torch.clamp(idx, min=0)

    @torch.no_grad()
    def get_embedding(
        self,
        sequences: Union[str, List[str]],
        batch_size: int = 8,
        pool: Pooling = "mean",
        pooling: Optional[Pooling] = None,  # 兼容历史参数名
        layer_index: int = -1,
        return_numpy: bool = True,
        exclude_special_tokens: bool = True,  # 保留兼容；官方实现不排除 special tokens
        **kwargs,
    ) -> Union[np.ndarray, torch.Tensor]:
        if pooling is not None:
            pool = pooling

        if isinstance(sequences, str):
            sequences = [sequences]

        # 官方实现：BOS + 截断到 6 的倍数
        bos = self.tokenizer.bos_token or ""
        processed = [bos + self._preprocess(s) for s in sequences]

        results: List[torch.Tensor] = []

        for i in tqdm(range(0, len(processed), batch_size), desc="GENERator Embedding"):
            batch = processed[i : i + batch_size]

            # 与官方一致：right padding, add_special_tokens, max_length
            encoded = self.tokenizer(
                batch,
                add_special_tokens=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_len,
            )

            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

            # 最后一层 hidden states（与官方一致）
            hidden_states = outputs.hidden_states[-1]  # (B, L, H)
            if layer_index != -1:
                li = layer_index
                if li < 0:
                    li = len(outputs.hidden_states) + li
                hidden_states = outputs.hidden_states[li]

            token_embeddings = hidden_states

            if pool == "mean":
                # 官方 Option 2: Mean pooling over all tokens (按 attention_mask)
                expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(torch.float32)
                sum_embeddings = torch.sum(token_embeddings * expanded_mask, dim=1)
                emb = sum_embeddings / expanded_mask.sum(dim=1).clamp(min=1)

            elif pool == "max":
                # 兼容：按 attention_mask，排除 pad
                valid_mask = attention_mask.bool().unsqueeze(-1)
                masked = token_embeddings.masked_fill(
                    ~valid_mask, torch.finfo(token_embeddings.dtype).min
                )
                emb = masked.max(dim=1).values

            elif pool in ("last_token", "eos"):
                # 官方 Option 1: Last token (EOS) embedding
                last_token_indices = attention_mask.sum(dim=1) - 1
                emb = token_embeddings[
                    torch.arange(token_embeddings.size(0), device=self.device),
                    last_token_indices,
                    :,
                ]

            else:
                raise ValueError(f"Unknown pooling strategy: {pool}")

            results.append(emb.detach().float().cpu())

            del input_ids, attention_mask, outputs, token_embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        final = torch.cat(results, dim=0)
        if return_numpy:
            return final.numpy()
        return final

    def embed_sequences(self, *args, **kwargs):
        # 兼容外部调用
        return self.get_embedding(*args, **kwargs)

    @torch.no_grad()
    def score_sequences(
        self,
        sequences: List[str],
        batch_size: int = 8,
    ) -> List[float]:
        """
        计算序列 Average Log-Likelihood（PLL）
        """
        processed = [self._preprocess(s) for s in sequences]
        scores: List[float] = []

        for i in tqdm(range(0, len(processed), batch_size), desc="Scoring with GENERator"):
            batch = processed[i : i + batch_size]

            encoded = self.tokenizer(
                batch,
                max_length=self.max_len,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=True,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )

            logits = outputs.logits  # (B, L, V)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            token_losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_labels.size())

            valid_nll = token_losses * shift_mask
            seq_nll = valid_nll.sum(dim=1)
            seq_lens = shift_mask.sum(dim=1).clamp(min=1)

            avg_log_likelihood = -seq_nll / seq_lens
            scores.extend(avg_log_likelihood.detach().cpu().tolist())

        return scores


if __name__ == "__main__":
    MODEL_PATH = "../../model_weight/GENERator-v2-prokaryote-1.2b-base"

    m = GENERatorModel(
        "GENERator",
        model_path=MODEL_PATH,
        tokenizer_padding_side="right",  # ✅ 与官方默认一致
        kmer_pad_side="right",           # ✅ 关键：不要左侧补齐
    )

    test_seqs = [
        "ACGTAGACGTAG",  # 12bp
        "ACGTAGACGT",    # 10bp -> pad 到 12bp（右侧补 A）
    ]

    embs = m.embed_sequences(test_seqs, pooling="last_token")
    print(f"\n[Embedding Shape]: {embs.shape}")

    scores = m.score_sequences(test_seqs, batch_size=2)
    print(f"\n[Likelihood Scores]: {scores}")

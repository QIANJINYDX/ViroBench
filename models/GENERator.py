# models/generator_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Optional, Literal, Union, Dict, Any

# ==========================================
# 兼容补丁: 解决 'DNAKmerTokenizer' 缺少 _added_tokens_decoder
#（尽量最小化对 transformers 的侵入）
# ==========================================
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
        预处理：
        1) 大写 + 去空白
        2) 6-mer 对齐：长度补齐到 k 的倍数
           ✅ 默认右侧补齐，避免平移整条 k-mer 切分
        """
        seq = seq.strip().upper()
        if len(seq) == 0:
            return seq

        r = len(seq) % self.kmer_size
        if r != 0:
            pad_len = self.kmer_size - r
            pad = self.kmer_pad_char * pad_len
            if self.kmer_pad_side == "right":
                seq = seq + pad
            else:
                seq = pad + seq
        return seq

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
        pooling: Optional[Pooling] = None,  # ✅ 兼容你之前的 pooling=...
        layer_index: int = -1,
        return_numpy: bool = True,
        exclude_special_tokens: bool = True,
        **kwargs,
    ) -> Union[np.ndarray, torch.Tensor]:

        if pooling is not None:
            pool = pooling  # 兼容历史参数名

        if isinstance(sequences, str):
            sequences = [sequences]

        processed = [self._preprocess(s) for s in sequences]

        results: List[torch.Tensor] = []

        for i in tqdm(range(0, len(processed), batch_size), desc="GENERator Embedding"):
            batch = processed[i : i + batch_size]

            # 尽量拿到 special_tokens_mask
            tok_kwargs: Dict[str, Any] = dict(
                max_length=self.max_len,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=True,
            )
            if exclude_special_tokens:
                tok_kwargs["return_special_tokens_mask"] = True

            encoded = self.tokenizer(batch, **tok_kwargs)

            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            special_mask = None
            if exclude_special_tokens and "special_tokens_mask" in encoded:
                special_mask = encoded["special_tokens_mask"].to(self.device)  # 1 表示 special

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            hidden_states = outputs.hidden_states
            li = layer_index
            if li < 0:
                li = len(hidden_states) + li
            token_embeddings = hidden_states[li]  # (B, L, H)

            # valid_mask：既要是非 pad，也可选择排除 special
            valid_mask = attention_mask.bool()
            if special_mask is not None:
                valid_mask = valid_mask & (~special_mask.bool())

            if pool == "mean":
                vm = valid_mask.unsqueeze(-1)  # (B, L, 1)
                denom = vm.sum(dim=1).clamp(min=1)  # (B, 1)
                emb = (token_embeddings * vm).sum(dim=1) / denom

            elif pool == "max":
                # masked positions -> -inf
                vm = valid_mask.unsqueeze(-1)
                masked = token_embeddings.masked_fill(~vm, torch.finfo(token_embeddings.dtype).min)
                emb = masked.max(dim=1).values

            elif pool in ("last_token", "eos"):
                # 默认：最后一个非 pad token
                last_idx = self._last_nonpad_index(attention_mask)  # 用 attention_mask（含 special）更稳
                if pool == "eos" and (self.tokenizer.eos_token_id is not None):
                    eos_id = int(self.tokenizer.eos_token_id)
                    # 找最后一个 eos（如果存在），否则回退 last_idx
                    is_eos = (input_ids == eos_id) & attention_mask.bool()
                    # 反向 argmax 找最后一个 True
                    rev = torch.flip(is_eos, dims=[1])
                    rev_pos = rev.float().argmax(dim=1)
                    has_eos = is_eos.any(dim=1)
                    eos_idx = input_ids.size(1) - 1 - rev_pos
                    idx = torch.where(has_eos, eos_idx, last_idx)
                else:
                    idx = last_idx

                emb = token_embeddings[torch.arange(token_embeddings.size(0), device=self.device), idx]

            else:
                raise ValueError(f"Unknown pooling strategy: {pool}")

            results.append(emb.detach().float().cpu())

            del input_ids, attention_mask, outputs, hidden_states
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
    MODEL_PATH = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/GENERator-v2-prokaryote-1.2b-base"

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

    embs = m.embed_sequences(test_seqs, pooling="mean")  # ✅ pooling 也支持
    print(f"\n[Embedding Shape]: {embs.shape}")

    scores = m.score_sequences(test_seqs, batch_size=2)
    print(f"\n[Likelihood Scores]: {scores}")

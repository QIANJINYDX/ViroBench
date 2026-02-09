# models/generator_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import math
import warnings
from typing import Any, Dict, List, Optional, Literal, Union, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# ==========================================
# 核心修复补丁: 解决 'DNAKmerTokenizer' object has no attribute '_added_tokens_decoder'
# ==========================================
try:
    import transformers
    from transformers import PreTrainedTokenizer

    # 保存原始方法
    if not hasattr(PreTrainedTokenizer, "_orig_convert_ids_to_tokens"):
        PreTrainedTokenizer._orig_convert_ids_to_tokens = PreTrainedTokenizer.convert_ids_to_tokens

    def _patched_convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        if not hasattr(self, "_added_tokens_decoder"):
            self._added_tokens_decoder = {}
        return self._orig_convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)

    PreTrainedTokenizer.convert_ids_to_tokens = _patched_convert_ids_to_tokens
    print("[GENERatorModel] Applied compatibility patch for DNAKmerTokenizer.")
except ImportError:
    pass
except Exception as e:
    print(f"[GENERatorModel] Warning: Failed to apply patch: {e}")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    raise ImportError("transformers not installed. Please install via `pip install transformers`.")

from .base_model import BaseModel

Pooling = Literal["mean", "last_token", "max"]
AlignMode = Literal["pad_left", "truncate_left"]
PPLMode = Literal["full", "conditional"]


class GENERatorModel(BaseModel):
    """
    GENERator V2 适配器（生成 + PPL）

    关键点（与官方示例一致）：
      - 6-mer 约束：输入长度需能被 6 整除，否则容易出现 <oov>，生成质量变差
      - 生成时建议手动 prepend BOS
      - decoder 模型建议 padding_side='left'
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        device: Optional[str] = None,
        max_len: int = 16384,
    ):
        super().__init__(model_name, model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = int(max_len)
        self._load_model()

    def _load_model(self):
        print(f"Loading {self.model_name} from {self.model_path}...")

        # 1) tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left",  # decoder 左填充
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"Set pad_token to eos_token: {self.tokenizer.pad_token}")

        # 2) model
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=None,
        )
        self.model.config.use_cache = True  # 生成建议开 cache（score/ppl 时可关）
        self.model.to(self.device).eval()

        print(f"[{self.model_name}] loaded on {self.device} (Type: {type(self.model).__name__})")
        print(f"[{self.model_name}] vocab={getattr(self.tokenizer, 'vocab_size', None)}, "
              f"pad_id={self.tokenizer.pad_token_id}, bos_id={self.tokenizer.bos_token_id}, eos_id={self.tokenizer.eos_token_id}")

    # -------------------------
    # helpers
    # -------------------------
    @staticmethod
    def _clean_seq_text(s: str) -> str:
        """清理 decode 后的空格/换行；只保留常见碱基字符（防止 tokenizer 用空格分隔 token）。"""
        if s is None:
            return ""
        s = s.replace(" ", "").replace("\n", "").replace("\r", "").strip()
        # 若你希望保留 N 以外的 IUPAC 扩展码，可自行扩展集合
        keep = set("ACGTNacgtn")
        s2 = "".join([ch for ch in s if ch in keep])
        return s2.upper() if s2 else s.upper()

    def _align_6mer(self, seq: str, mode: AlignMode = "pad_left") -> Tuple[str, int, int]:
        """
        让长度对齐到 6 的倍数：
          - pad_left: 左侧补 'A'（保留原序列内容，但会改变前端 token 对齐）
          - truncate_left: 左侧截断（丢掉开头若干 bp，但保持后续 token 对齐；更接近官方示例的“left truncate”思路）
        返回: (aligned_seq, n_added_left, n_truncated_left)
        """
        s = seq.strip().upper()
        r = len(s) % 6
        if r == 0:
            return s, 0, 0
        need = 6 - r
        if mode == "pad_left":
            return ("A" * need + s), need, 0
        elif mode == "truncate_left":
            # 去掉开头 r 个字符，使剩余长度可被 6 整除
            return s[r:], 0, r
        else:
            raise ValueError(f"Unknown align mode: {mode}")

    def _tokenize(
        self,
        seqs: List[str],
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            seqs,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
            padding=("longest" if padding else False),
            truncation=truncation,
            max_length=max_length,
        )
        # 关键：很多 tokenizer 会产生 token_type_ids，但 decoder 模型 generate/forward 不接收
        if "token_type_ids" in enc:
            enc.pop("token_type_ids")
        return {k: v.to(self.device) for k, v in enc.items()}

    def _prepend_bos(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bos_id = getattr(self.tokenizer, "bos_token_id", None)
        if bos_id is None:
            # 有些 tokenizer 没有 bos；就不加
            return input_ids, attention_mask
        B = input_ids.size(0)
        bos = torch.full((B, 1), int(bos_id), dtype=input_ids.dtype, device=input_ids.device)
        input_ids = torch.cat([bos, input_ids], dim=1)
        am = torch.ones((B, 1), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([am, attention_mask], dim=1)
        return input_ids, attention_mask

    # -------------------------
    # embedding / scoring（保留你原来的）
    # -------------------------
    @torch.no_grad()
    def get_embedding(
        self,
        sequences: Union[str, List[str]],
        batch_size: int = 8,
        pool: Pooling = "mean",
        layer_index: int = -1,
        return_numpy: bool = True,
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:

        if isinstance(sequences, str):
            sequences = [sequences]

        # 这里沿用你原来的预处理：pad_left 到 6 的倍数
        processed = []
        for s in sequences:
            s2, _, _ = self._align_6mer(s, mode="pad_left")
            processed.append(s2)

        results = []
        for i in tqdm(range(0, len(processed), batch_size), desc="GENERator Embedding"):
            batch_seqs = processed[i: i + batch_size]
            enc = self._tokenize(
                batch_seqs,
                add_special_tokens=False,
                max_length=self.max_len,
                padding=True,
                truncation=True,
            )

            outputs = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask", None),
                output_hidden_states=True
            )

            hidden_states = outputs.hidden_states
            idx = layer_index
            if idx < 0:
                idx = len(hidden_states) + idx
            token_embeddings = hidden_states[idx]

            attention_mask = enc.get("attention_mask", torch.ones_like(enc["input_ids"]))
            if pool == "mean":
                mask_exp = attention_mask.unsqueeze(-1).expand_as(token_embeddings)
                sum_emb = torch.sum(token_embeddings * mask_exp, dim=1)
                sum_mask = torch.clamp(mask_exp.sum(dim=1), min=1e-9)
                emb = sum_emb / sum_mask
            elif pool == "last_token":
                # left padding: 最后一个 token 是序列末尾
                emb = token_embeddings[:, -1, :]
            elif pool == "max":
                mask_exp = attention_mask.unsqueeze(-1).expand_as(token_embeddings)
                token_embeddings = token_embeddings.masked_fill(mask_exp == 0, -1e9)
                emb = torch.max(token_embeddings, dim=1)[0]
            else:
                raise ValueError(f"Unknown pooling strategy: {pool}")

            results.append(emb.detach().to(torch.float32).cpu())

        final = torch.cat(results, dim=0) if results else torch.empty(0, 0)
        return final.numpy() if return_numpy else final

    def embed_sequences(self, *args, **kwargs):
        return self.get_embedding(*args, **kwargs)

    @torch.no_grad()
    def score_sequences(self, sequences: List[str], batch_size: int = 8) -> List[float]:
        """
        返回 Average Log-Likelihood（越大越好）
        """
        # 仍沿用 pad_left 对齐
        processed = []
        for s in sequences:
            s2, _, _ = self._align_6mer(s, mode="pad_left")
            processed.append(s2)

        scores: List[float] = []
        for i in tqdm(range(0, len(processed), batch_size), desc="Scoring with GENERator"):
            batch_seqs = processed[i: i + batch_size]
            enc = self._tokenize(
                batch_seqs,
                add_special_tokens=False,
                max_length=self.max_len,
                padding=True,
                truncation=True,
            )

            input_ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))

            # 不用 outputs.loss（它是 batch mean），我们要 per-sample
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits  # [B, T, V]

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous().float()

            V = shift_logits.size(-1)
            token_loss = F.cross_entropy(
                shift_logits.reshape(-1, V),
                shift_labels.reshape(-1),
                reduction="none",
            ).view(shift_labels.size())

            nll = token_loss * shift_mask
            seq_nll = nll.sum(dim=1)
            seq_tok = shift_mask.sum(dim=1).clamp_min(1.0)

            avg_loglik = -(seq_nll / seq_tok)
            scores.extend(avg_loglik.detach().cpu().tolist())

        return scores

    # -------------------------
    # NEW: generation
    # -------------------------
    @torch.no_grad()
    def generate(
        self,
        prompt_seqs: Union[str, List[str]],
        n_tokens: int = 1536,  # 注意：此参数实际表示碱基数，不是 token 数
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        batch_size: int = 8,
        align_mode: AlignMode = "truncate_left",
        add_bos: bool = True,
        clean_output: bool = True,
    ) -> List[str]:
        """
        生成序列（Prompt + continuation）

        参数:
            n_tokens: 要生成的碱基数（注意：虽然参数名为 n_tokens，但实际传入的是碱基数）
                    - 由于 GENERator 使用 6-mer tokenization，1 个 token = 6 个碱基
                    - 如果碱基数能被 6 整除，则实际 token 数 = 碱基数 // 6
                    - 如果不能整除，则实际 token 数 = 碱基数 // 6 + 1（向上取整）
                    - 例如：传入 6 -> 1 token, 传入 7 -> 2 tokens, 传入 12 -> 2 tokens, 传入 13 -> 3 tokens

        align_mode 默认用 truncate_left，更贴近官方示例"让长度可被6整除并左截断"的用法。
        若你更想保留原始 prompt 前缀，可改为 pad_left。
        """
        # 将传入的碱基数（虽然参数名为 n_tokens）转换为真正的 token 数
        # 公式：(n_bases + 5) // 6 等价于向上取整到最近的 6 的倍数对应的 token 数
        n_bases = n_tokens  # 传入的 n_tokens 实际是碱基数
        actual_n_tokens = (n_bases + 5) // 6
        
        if isinstance(prompt_seqs, str):
            prompt_list = [prompt_seqs]
        else:
            prompt_list = list(prompt_seqs)

        # 预处理：对齐 6-mer
        aligned_prompts = []
        for s in prompt_list:
            s2, _, _ = self._align_6mer(s, mode=align_mode)
            aligned_prompts.append(s2)

        results: List[str] = []
        # 生成建议开 cache
        old_cache = getattr(self.model.config, "use_cache", True)
        self.model.config.use_cache = True

        for st in tqdm(range(0, len(aligned_prompts), batch_size), desc="GENERator generate"):
            batch = aligned_prompts[st: st + batch_size]

            # 注意：官方示例用 add_special_tokens=False，并手动 prepend BOS。
            enc = self._tokenize(
                batch,
                add_special_tokens=False,
                max_length=max(1, self.max_len - int(actual_n_tokens) - 4),
                padding=True,
                truncation=True,
            )

            input_ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))

            if add_bos:
                input_ids, attention_mask = self._prepend_bos(input_ids, attention_mask)

            gen_kwargs = dict(
                max_new_tokens=int(actual_n_tokens),
                do_sample=bool(do_sample),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                repetition_penalty=float(repetition_penalty),
                num_return_sequences=int(num_return_sequences),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

            gen_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs
            )

            decoded = self.tokenizer.batch_decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            if clean_output:
                decoded = [self._clean_seq_text(s) for s in decoded]

            results.extend(decoded)

        self.model.config.use_cache = old_cache
        return results

    @torch.no_grad()
    def get_ppl(
        self,
        sequences: Union[str, List[str]],
        batch_size: int = 8,
        mode: PPLMode = "full",              # "full" | "conditional"
        prompt_len_chars: int = 128,         # conditional 时用
        align_mode: AlignMode = "truncate_left",
        add_bos: bool = True,
        return_details: bool = True,
        *,
        # ---- 新增：长序列 sliding-window ----
        use_sliding_window: bool = True,
        max_window_tokens: Optional[int] = None,  # 默认 self.max_len
        stride: Optional[int] = None,             # 默认 min(1024, max_window_tokens//4)
        # ---- 新增：当不用 sliding 时，是否允许截断到 max_len ----
        truncate_if_needed: bool = True,
        # ---- 新增：conditional 时若截断导致 prompt 不在上下文，是否仍强行计算 ----
        allow_prompt_truncation: bool = False,
        # ---- 新增：是否排除 special token（一般 add_special_tokens=False 时可关）----
        exclude_special: bool = False,
    ) -> Union[List[float], float, List[Dict[str, Any]]]:
        """
        计算 PPL（token-level perplexity）。

        mode="full":
            计分整个序列（在 add_bos=True 时，相当于 P(x | BOS)）
        mode="conditional":
            只计分 continuation；prompt = 原始序列前 prompt_len_chars 个字符，
            会映射到对齐后的 aligned 序列，并强制对齐到 6 的倍数边界，再 token 化得到 prompt token 边界。

        sliding-window:
            对超长 token 序列使用窗口滚动估计：
              - 每次窗口长度 <= max_window_tokens
              - 每一步只对新增的 stride token 计分（其余作为上下文）
              - 每个 token 只计一次，避免重复计分

        返回：
            return_details=False:
                - 输入为 str -> float
                - 输入为 List[str] -> List[float]
            return_details=True:
                List[dict]，每条包含 ppl/avg_nll/token_count/窗口参数/错误信息等
        """
        if isinstance(sequences, str):
            seq_list = [sequences]
            single = True
        else:
            seq_list = list(sequences)
            single = False

        # window params
        if max_window_tokens is None:
            max_window_tokens = int(self.max_len)
        else:
            max_window_tokens = int(max_window_tokens)
        max_window_tokens = max(2, max_window_tokens)

        if stride is None:
            stride = min(1024, max(1, max_window_tokens // 4))
        stride = int(max(1, min(stride, max_window_tokens)))

        # 关 cache（PPL/score 省显存）
        old_cache = getattr(self.model.config, "use_cache", True)
        self.model.config.use_cache = False

        # special ids（可选）
        special_ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        if self.tokenizer.pad_token_id is not None:
            special_ids.add(int(self.tokenizer.pad_token_id))
        if self.tokenizer.bos_token_id is not None:
            special_ids.add(int(self.tokenizer.bos_token_id))
        if self.tokenizer.eos_token_id is not None:
            special_ids.add(int(self.tokenizer.eos_token_id))

        def _safe_exp(x: float, cap: float = 700.0) -> float:
            if math.isnan(x):
                return float("nan")
            if x > cap:
                return float("inf")
            try:
                return math.exp(x)
            except OverflowError:
                return float("inf")

        def _count_valid_from_labels(labels: torch.Tensor) -> int:
            """
            HF CausalLM 内部做 shift：
              shift_logits = logits[:, :-1]
              shift_labels = labels[:,  1:]
            所以有效计分 token 数 = (labels[:,1:] != -100).sum()
            """
            if labels.size(1) <= 1:
                return 0
            return int((labels[:, 1:] != -100).sum().item())

        def _tokenize_ids(seq: str, trunc: bool) -> torch.Tensor:
            enc = self.tokenizer(
                seq,
                return_tensors="pt",
                padding=False,
                truncation=bool(trunc),
                max_length=(int(max_window_tokens) if trunc else None),
                add_special_tokens=False,
                return_token_type_ids=False,
            )
            if isinstance(enc, dict) and "token_type_ids" in enc:
                enc.pop("token_type_ids", None)
            return enc["input_ids"][0].to(self.device)  # [L]

        def _prepend_bos_1d(ids: torch.Tensor) -> torch.Tensor:
            bos_id = getattr(self.tokenizer, "bos_token_id", None)
            if (not add_bos) or (bos_id is None):
                return ids
            bos = torch.tensor([int(bos_id)], device=ids.device, dtype=ids.dtype)
            return torch.cat([bos, ids], dim=0)

        def _map_prompt_chars_to_aligned_prefix_len(
            s_aligned: str, addL: int, truncL: int
        ) -> int:
            # prompt_len_chars 是基于“原始序列”的前缀
            if mode != "conditional":
                return 0
            if align_mode == "pad_left":
                mapped = min(int(prompt_len_chars) + int(addL), len(s_aligned))
            else:  # truncate_left
                mapped = max(int(prompt_len_chars) - int(truncL), 0)
                mapped = min(mapped, len(s_aligned))
            # 强制对齐到 6 的倍数，避免 prompt 内部出现不完整 6-mer
            mapped = mapped - (mapped % 6)
            return int(mapped)

        def _direct_nll_sum(ids_full: torch.Tensor, start_pos: int) -> Tuple[float, int, str]:
            """
            不用 sliding：一次 forward(labels=...) 计算 nll_sum 与 token_count
            若超长且 truncate_if_needed=True，则保留最后 max_window_tokens 个 token（左截断）。
            conditional 且 prompt 被截掉时，默认报错返回 NaN（除非 allow_prompt_truncation=True）。
            """
            L = int(ids_full.numel())
            if L <= 1 or start_pos >= L:
                return float("nan"), 0, "no_tokens"

            cut_begin = 0
            if L > max_window_tokens:
                if not truncate_if_needed:
                    return float("nan"), 0, "too_long_no_trunc"
                cut_begin = L - max_window_tokens
                # 如果 conditional 的 start_pos 落在被截掉的区域里，默认不算（避免错误结果）
                if (mode == "conditional") and (start_pos < cut_begin) and (not allow_prompt_truncation):
                    return float("nan"), 0, "prompt_out_of_context_due_to_truncation"
                ids = ids_full[cut_begin:]
                start_local = max(1, start_pos - cut_begin)
                tag = "direct_truncated"
            else:
                ids = ids_full
                start_local = max(1, start_pos)
                tag = "direct"

            ids2 = ids.unsqueeze(0)  # [1, T]
            labels = ids2.clone()
            labels[:, :start_local] = -100

            if exclude_special and len(special_ids) > 0:
                spec_mask = torch.zeros_like(labels, dtype=torch.bool)
                for sp in special_ids:
                    spec_mask |= (labels == sp)
                labels[spec_mask] = -100

            out = self.model(
                input_ids=ids2,
                attention_mask=torch.ones_like(ids2),
                labels=labels,
                use_cache=False,
                return_dict=True,
            )
            loss = out.loss
            if loss is None or torch.isnan(loss):
                return float("nan"), 0, tag

            valid = _count_valid_from_labels(labels)
            if valid <= 0:
                return float("nan"), 0, tag

            return float(loss.item()) * valid, valid, tag

        def _sliding_nll_sum(ids_full: torch.Tensor, start_pos: int) -> Tuple[float, int, str]:
            """
            sliding-window：窗口末尾对齐，每步只计分新增 stride 个 token。
            """
            L = int(ids_full.numel())
            if L <= 1 or start_pos >= L:
                return float("nan"), 0, "no_tokens"

            nll_sum = 0.0
            tok_cnt = 0

            prev_end = 0
            end = min(max_window_tokens, L)

            while True:
                begin = max(0, end - max_window_tokens)
                ids = ids_full[begin:end].unsqueeze(0)  # [1, W]
                W = int(ids.size(1))

                trg_len = end - prev_end  # 新增 token 数（全局）
                trg_len = max(0, min(trg_len, W))

                labels = ids.clone()

                # 1) 只对新增的 trg_len token 计分
                if trg_len < W:
                    labels[:, : (W - trg_len)] = -100

                # 2) conditional/full 起点：mask 全局位置 < start_pos
                #    global_idx = begin + j
                if start_pos > 0:
                    global_idx = torch.arange(begin, end, device=self.device)
                    labels[:, (global_idx < int(start_pos))] = -100

                # 3) special mask（可选）
                if exclude_special and len(special_ids) > 0:
                    spec_mask = torch.zeros_like(labels, dtype=torch.bool)
                    for sp in special_ids:
                        spec_mask |= (labels == sp)
                    labels[spec_mask] = -100

                out = self.model(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    labels=labels,
                    use_cache=False,
                    return_dict=True,
                )
                loss = out.loss
                if loss is None or torch.isnan(loss):
                    return float("nan"), 0, "sliding"

                valid = _count_valid_from_labels(labels)
                if valid > 0:
                    nll_sum += float(loss.item()) * valid
                    tok_cnt += int(valid)

                prev_end = end
                if end >= L:
                    break
                end = min(end + stride, L)

            if tok_cnt <= 0:
                return float("nan"), 0, "sliding"
            return nll_sum, tok_cnt, "sliding"

        # =========================
        # main loop
        # =========================
        ppl_list: List[float] = []
        out_details: List[Dict[str, Any]] = []

        for st in tqdm(range(0, len(seq_list), batch_size), desc="GENERator PPL (enhanced)"):
            batch = seq_list[st: st + batch_size]

            for raw in batch:
                # 6-mer 对齐
                s_aligned, addL, truncL = self._align_6mer(raw, mode=align_mode)

                # token ids（sliding 时不截断；direct 时可按需截断）
                ids0 = _tokenize_ids(s_aligned, trunc=(not use_sliding_window))
                ids0 = _prepend_bos_1d(ids0)

                L_tok = int(ids0.numel())

                # start_pos：在 input_ids 的 token index 空间里（0..L-1）
                # - full：从 token index 1 开始（跳过 BOS）
                # - conditional：从 prompt_len_total 开始（prompt token 本身不计分）
                if mode == "full":
                    start_pos = 1
                    prompt_len_tokens_total = 0
                    mapped_prompt_chars = None
                elif mode == "conditional":
                    mapped_prompt_chars = _map_prompt_chars_to_aligned_prefix_len(s_aligned, addL, truncL)
                    prefix = s_aligned[:mapped_prompt_chars]
                    p_ids = _tokenize_ids(prefix, trunc=True)  # prompt 不会太长，截断无所谓
                    p_len = int(p_ids.numel())
                    prompt_len_tokens_total = p_len + (1 if (add_bos and self.tokenizer.bos_token_id is not None) else 0)
                    start_pos = max(int(prompt_len_tokens_total), 1)
                else:
                    raise ValueError(f"Unknown mode={mode}")

                # 计算 nll_sum / token_count
                if use_sliding_window and (L_tok > max_window_tokens):
                    nll_sum, tok_cnt, used = _sliding_nll_sum(ids0, start_pos)
                else:
                    nll_sum, tok_cnt, used = _direct_nll_sum(ids0, start_pos)

                if tok_cnt <= 0 or (not math.isfinite(nll_sum)):
                    avg_nll = float("nan")
                    ppl = float("nan")
                    err = used if used not in {"direct", "direct_truncated", "sliding"} else "no_valid_tokens_for_ppl"
                else:
                    avg_nll = float(nll_sum) / float(tok_cnt)
                    ppl = float(_safe_exp(avg_nll))
                    err = None

                ppl_list.append(ppl)
                                # ---- bpb 需要：char_count / avg_nll_token ----
                # GENERator-v2: 6-mer non-overlap tokenization
                BASES_PER_TOKEN = 6
                char_count = int(tok_cnt) * BASES_PER_TOKEN if (tok_cnt is not None) else 0

                if return_details:
                    rec: Dict[str, Any] = {
                        "ppl": ppl,
                        "avg_nll": avg_nll,
                        "token_count": int(tok_cnt),
                        "mode": mode,
                        "used": used,
                        "sequence_tokens_total": int(L_tok),
                        "align_mode": align_mode,
                        "align_added_left_chars": int(addL),
                        "align_truncated_left_chars": int(truncL),
                        "use_sliding_window": bool(use_sliding_window),
                        "max_window_tokens": int(max_window_tokens),
                        "stride": int(stride),
                        "exclude_special": bool(exclude_special),
                        "error": err,

                        # ✅ bpb 兼容字段（Evo2-style）
                        "avg_nll_token": float(avg_nll),   # nat/token
                        "char_count": int(char_count),     # bases scored
                        "bases_per_token": int(BASES_PER_TOKEN),
                    }
                    if mode == "conditional":
                        rec.update({
                            "prompt_len_chars_raw": int(prompt_len_chars),
                            "prompt_len_chars_mapped_aligned": int(mapped_prompt_chars) if mapped_prompt_chars is not None else None,
                            "prompt_len_tokens_total": int(prompt_len_tokens_total),
                        })
                    out_details.append(rec)


                # if return_details:
                #     rec: Dict[str, Any] = {
                #         "ppl": ppl,
                #         "avg_nll": avg_nll,
                #         "token_count": int(tok_cnt),
                #         "mode": mode,
                #         "used": used,
                #         "sequence_tokens_total": int(L_tok),
                #         "align_mode": align_mode,
                #         "align_added_left_chars": int(addL),
                #         "align_truncated_left_chars": int(truncL),
                #         "use_sliding_window": bool(use_sliding_window),
                #         "max_window_tokens": int(max_window_tokens),
                #         "stride": int(stride),
                #         "exclude_special": bool(exclude_special),
                #         "error": err,
                #     }
                #     if mode == "conditional":
                #         rec.update({
                #             "prompt_len_chars_raw": int(prompt_len_chars),
                #             "prompt_len_chars_mapped_aligned": int(mapped_prompt_chars) if mapped_prompt_chars is not None else None,
                #             "prompt_len_tokens_total": int(prompt_len_tokens_total),
                #         })
                #     out_details.append(rec)

        self.model.config.use_cache = old_cache

        if return_details:
            return out_details

        if single:
            return ppl_list[0] if ppl_list else float("nan")
        return ppl_list



# -------------------------
# quick test
# -------------------------
if __name__ == "__main__":
    MODEL_PATH = "../../model_weight/GENERator-v2-eukaryote-1.2b-base"

    m = GENERatorModel("GENERator-v2", model_path=MODEL_PATH, device=None, max_len=16384)

    prompts = [
        "AGTTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT",   # 任意长度
        "ATGCGTACGTTAG"                           # 任意长度
    ]

    gen = m.generate(
        prompts,
        n_tokens=384,  # 传入 384 个碱基，内部会转换为 64 个 token (384 / 6 = 64)
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        do_sample=True,
        batch_size=2,
        align_mode="truncate_left",   # 更贴近官方示例
        add_bos=True,
        clean_output=True,
    )
    print("Generated:", gen)

    ppl_full = m.get_ppl(gen, batch_size=2, mode="full", return_details=False)
    print("PPL(full):", ppl_full)

    ppl_cond = m.get_ppl(gen, batch_size=2, mode="conditional", prompt_len_chars=128, return_details=True)
    print("PPL(conditional details):", ppl_cond)

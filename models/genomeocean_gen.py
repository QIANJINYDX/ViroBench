# models/genomeocean_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import math
from typing import List, Optional, Literal, Union, Dict, Any,Tuple

import torch
import numpy as np
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from .base_model import BaseModel

Pooling = Literal["mean", "last_token"]

# 官方：100M/500M=1024 tokens；4B=10240 tokens :contentReference[oaicite:1]{index=1}
MODEL_TOKEN_LIMITS = {
    "DOEJGI/GenomeOcean-100M": 1024,
    "DOEJGI/GenomeOcean-500M": 1024,
    "DOEJGI/GenomeOcean-4B": 10240,
    "pGenomeOcean/GenomeOcean-100M": 1024,
    "pGenomeOcean/GenomeOcean-500M": 1024,
    "pGenomeOcean/GenomeOcean-4B": 10240,
    "pGenomeOcean/GenomeOcean-4B-bgcFM": 10240,
    # 兼容你本地目录名
    "GenomeOcean-100M": 1024,
    "GenomeOcean-500M": 1024,
    "GenomeOcean-4B": 10240,
    "GenomeOcean-4B-bgcFM": 10240,
}


class PresenceFrequencyPenaltyLogitsProcessor(LogitsProcessor):
    """
    自定义 presence / frequency penalty（OpenAI 风格）：
      scores[token] -= presence_penalty * 1[token曾出现]
      scores[token] -= frequency_penalty * count(token出现次数)

    注意：这里用 input_ids（当前已生成的 token 序列）统计频次。
    """
    def __init__(self, presence_penalty: float = 0.0, frequency_penalty: float = 0.0):
        self.presence_penalty = float(presence_penalty)
        self.frequency_penalty = float(frequency_penalty)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if (self.presence_penalty == 0.0) and (self.frequency_penalty == 0.0):
            return scores

        # input_ids: [B, L], scores: [B, V]
        B, V = scores.shape
        # 统计每个 batch 里出现过的 token 以及频次
        for b in range(B):
            ids = input_ids[b]
            # bincount 需要 0..V-1
            cnt = torch.bincount(ids, minlength=V).to(scores.dtype)  # [V]
            if self.presence_penalty != 0.0:
                scores[b] = scores[b] - self.presence_penalty * (cnt > 0).to(scores.dtype)
            if self.frequency_penalty != 0.0:
                scores[b] = scores[b] - self.frequency_penalty * cnt
        return scores


class GenomeOceanModel(BaseModel):
    """
    GenomeOcean 适配器：Embedding + 生成 + (conditional) PPL
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        device: Optional[str] = None,
        max_len: Optional[int] = None,
    ):
        super().__init__(model_name, model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ---- 推断 max_len（兼容本地目录名）----
        if max_len is None:
            base_name = os.path.basename(model_path.rstrip("/"))
            # 先按字典匹配
            inferred = MODEL_TOKEN_LIMITS.get(model_path) or MODEL_TOKEN_LIMITS.get(base_name)
            # 再按字符串兜底
            if inferred is None:
                bn = base_name.lower()
                if "4b" in bn:
                    inferred = 10240
                elif "500m" in bn or "100m" in bn:
                    inferred = 1024
                else:
                    inferred = 10240
            self.max_len = int(inferred)
        else:
            self.max_len = int(max_len)

        self._load_model()

    def _load_model(self):
        print(f"Loading {self.model_name} from {self.model_path}...")

        # 1) Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            model_max_length=self.max_len,
            padding_side="left",
            use_fast=True,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pad_id = int(self.tokenizer.pad_token_id) if self.tokenizer.pad_token_id is not None else None
        self.eos_id = int(self.tokenizer.eos_token_id) if self.tokenizer.eos_token_id is not None else None

        # 2) Model（优先 CausalLM）
        try:
            dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map=None,
            )
            self.model.config.use_cache = True
            self.can_score = True
            print(f"Loaded as CausalLM (Class: {type(self.model).__name__})")
        except Exception as e:
            print(f"Warning: Could not load as CausalLM ({e}), falling back to AutoModel (no generate/PPL).")
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )
            self.model.config.use_cache = False
            self.can_score = False

        self.model.to(self.device).eval()
        print(f"[{self.model_name}] loaded on {self.device} (Max Len: {self.max_len})")

    def _preprocess(self, seq: str) -> str:
        # 去掉空格/换行，避免 decode 后出现空格干扰后续计算
        return "".join(seq.strip().upper().split())

    # --------------------
    # Generation (修复 token_type_ids)
    # --------------------
    @torch.no_grad()
    def generate(
        self,
        prompt_seqs: Union[str, List[str]],
        num_return_sequences: int = 1,
        min_new_tokens: int = 0,
        n_tokens: int = 128,
        temperature: float = 1.3,
        top_p: float = 0.7,
        top_k: int = -1,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        do_sample: bool = True,
        seed: Optional[int] = None,
        remove_spaces: bool = True,
        batch_size: int = 8,
    ) -> List[str]:
        if not getattr(self, "can_score", True):
            raise RuntimeError("Model loaded as AutoModel (not CausalLM). Cannot generate.")

        if isinstance(prompt_seqs, str):
            prompt_list = [prompt_seqs]
        else:
            prompt_list = list(prompt_seqs)

        prompt_list = [self._preprocess(p) for p in prompt_list]

        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        # 自定义 presence/frequency penalty（不依赖 transformers 内置）
        lp = LogitsProcessorList()
        if abs(presence_penalty) > 1e-12 or abs(frequency_penalty) > 1e-12:
            lp.append(PresenceFrequencyPenaltyLogitsProcessor(presence_penalty, frequency_penalty))

        # 展开到 num_return_sequences
        expanded: List[str] = []
        for p in prompt_list:
            expanded.extend([p] * int(num_return_sequences))

        outs: List[str] = []

        for st in tqdm(range(0, len(expanded), batch_size), desc="GenomeOcean generate"):
            batch_prompts = expanded[st : st + batch_size]

            max_prompt_len = max(1, int(self.max_len) - int(n_tokens))

            # ✅ 关键：return_token_type_ids=False
            enc = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_len,
                return_token_type_ids=False,
            ).to(self.device)

            # ✅ 关键：无论 tokenizer 是否尊重参数，都强制 pop 掉
            if isinstance(enc, dict) and "token_type_ids" in enc:
                enc.pop("token_type_ids", None)

            gen_kwargs = dict(
                max_new_tokens=int(n_tokens),
                min_new_tokens=int(min_new_tokens),
                do_sample=bool(do_sample),
                temperature=float(temperature),
                top_p=float(top_p),
                repetition_penalty=float(repetition_penalty),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
            if int(top_k) is not None and int(top_k) > 0:
                gen_kwargs["top_k"] = int(top_k)

            if len(lp) > 0:
                gen_kwargs["logits_processor"] = lp

            gen_ids = self.model.generate(**enc, **gen_kwargs)

            decoded = self.tokenizer.batch_decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if remove_spaces:
                decoded = ["".join(s.split()) for s in decoded]

            outs.extend(decoded)

        return outs

    # --------------------
    # PPL / conditional PPL
    # --------------------
    @staticmethod
    def _safe_exp(x: float, cap: float = 700.0) -> float:
        if math.isnan(x):
            return float("nan")
        if x > cap:
            return float("inf")
        try:
            return math.exp(x)
        except OverflowError:
            return float("inf")

    @torch.no_grad()
    def get_ppl(
        self,
        sequences: Union[str, List[str]],
        batch_size: int = 8,
        conditional: bool = False,
        prompt_len_chars: int = 128,
        add_special_tokens: bool = False,
        return_details: bool = False,
        *,
        # 长序列 sliding-window（强烈建议默认开启）
        use_sliding_window: bool = True,
        max_window_tokens: Optional[int] = None,   # 默认用 self.max_len
        stride: Optional[int] = None,              # 默认 min(512, max_window_tokens//4)
        # 是否排除 special token
        exclude_special: bool = False,
        # 当 use_sliding_window=False 且超长时是否截断（尾部保留）
        truncate_if_needed: bool = True,
        # conditional 在截断/窗口不足导致 prompt 条件不成立时，是否仍强行计算
        allow_prompt_truncation: bool = False,
    ) -> Union[float, List[float], List[Dict[str, Any]]]:
        """
        GenomeOcean PPL / conditional PPL（支持 sliding-window），并在 return_details=True 时输出：
          - avg_nll_token (nat/token)
          - token_count
          - char_count (bases)  # 供 bpb 换算

        说明：
          - avg_nll_token 是对“预测 token”（shift 后）的平均 NLL
          - char_count 是对“碱基字符”的计数，用于把 total_nll(nat) 换算到 nat/base，再换 bits/base
        """
        if not getattr(self, "can_score", True):
            raise RuntimeError("Model loaded as AutoModel (not CausalLM). Cannot compute PPL.")

        if isinstance(sequences, str):
            seq_list = [sequences]
            single = True
        else:
            seq_list = list(sequences)
            single = False

        seq_list = [self._preprocess(s) for s in seq_list]

        # window 参数
        if max_window_tokens is None:
            max_window_tokens = int(self.max_len)
        else:
            max_window_tokens = int(max_window_tokens)
        max_window_tokens = max(2, max_window_tokens)

        if stride is None:
            stride = min(512, max(1, max_window_tokens // 4))
        stride = int(max(1, min(stride, max_window_tokens)))

        # special mask helper
        def _special_mask(ids_1d: torch.Tensor) -> Optional[torch.Tensor]:
            if not exclude_special:
                return None
            try:
                spec = self.tokenizer.get_special_tokens_mask(
                    ids_1d.tolist(),
                    already_has_special_tokens=True,
                )
                return torch.tensor(spec, device=self.device, dtype=torch.bool)  # [L]
            except Exception:
                return None

        def _count_valid_from_labels(labels: torch.Tensor) -> int:
            # HF 内部 shift：真正计分的是 labels[:,1:] != -100
            if labels.size(1) <= 1:
                return 0
            return int((labels[:, 1:] != -100).sum().item())

        def _tokenize_one(s: str, trunc: bool) -> Dict[str, torch.Tensor]:
            enc = self.tokenizer(
                s,
                return_tensors="pt",
                padding=False,
                truncation=bool(trunc),
                max_length=(int(max_window_tokens) if trunc else None),
                add_special_tokens=bool(add_special_tokens),
                return_token_type_ids=False,
            )
            if isinstance(enc, dict) and "token_type_ids" in enc:
                enc.pop("token_type_ids", None)

            input_ids = enc["input_ids"].to(self.device)  # [1, L]
            attn_mask = enc.get("attention_mask", None)
            if attn_mask is None:
                attn_mask = torch.ones_like(input_ids)
            attn_mask = attn_mask.to(self.device)
            return {"input_ids": input_ids, "attention_mask": attn_mask}

        def _window_loss_sum(
            ids_full: torch.Tensor,          # [1, L]
            start_pos: int,                  # global token position 起点（>=1）
            spec_full: Optional[torch.Tensor],
        ) -> Tuple[float, int, str, bool]:
            """
            返回 (total_nll_sum_in_nat, token_count, used_mode, did_truncate_tail)
            """
            L = int(ids_full.size(1))
            if L <= 1 or start_pos >= L:
                return float("nan"), 0, "no_tokens", False

            # 不用 sliding：一次性算（必要时尾部截断）
            if (not use_sliding_window) or (L <= max_window_tokens):
                did_trunc = False

                if (L > max_window_tokens):
                    if not truncate_if_needed:
                        return float("nan"), 0, "too_long_no_sliding", True
                    # 保留尾部 max_window_tokens
                    cut_begin = L - max_window_tokens
                    did_trunc = True

                    # conditional：截断会破坏“给定完整 prompt 的条件”
                    if conditional and (not allow_prompt_truncation):
                        return float("nan"), 0, "prompt_truncated_no_sliding", True

                    ids = ids_full[:, cut_begin:]
                    spec = spec_full[cut_begin:] if spec_full is not None else None
                    start_pos_local = max(1, start_pos - cut_begin)
                    used = "direct_truncated"
                else:
                    ids = ids_full
                    spec = spec_full
                    start_pos_local = start_pos
                    used = "direct"

                labels = ids.clone()
                if start_pos_local > 0:
                    labels[:, :start_pos_local] = -100
                if spec is not None:
                    labels[:, spec] = -100

                out = self.model(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    labels=labels,
                    use_cache=False,
                )
                loss = out.loss
                if loss is None or torch.isnan(loss):
                    return float("nan"), 0, used, did_trunc

                valid = _count_valid_from_labels(labels)
                if valid <= 0:
                    return float("nan"), 0, used, did_trunc

                nll_sum = float(loss.item()) * valid
                return nll_sum, valid, used, did_trunc

            # sliding-window：每次只计新增 stride token
            nll_sum = 0.0
            tok_cnt = 0
            prev_end = 0
            used = "sliding"

            for begin in range(0, L, stride):
                end = min(begin + max_window_tokens, L)
                if end <= 1:
                    break

                trg_len = end - prev_end
                if trg_len <= 0:
                    prev_end = end
                    if end == L:
                        break
                    continue

                ids = ids_full[:, begin:end]  # [1, W]
                labels = ids.clone()

                W = int(ids.size(1))
                if trg_len < W:
                    labels[:, :(W - trg_len)] = -100

                # mask 全局位置 < start_pos
                if start_pos > 0:
                    j = torch.arange(begin, end, device=self.device)
                    labels[:, (j < start_pos)] = -100

                if spec_full is not None:
                    spec = spec_full[begin:end]
                    labels[:, spec] = -100

                out = self.model(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    labels=labels,
                    use_cache=False,
                )
                loss = out.loss
                if loss is None or torch.isnan(loss):
                    return float("nan"), 0, used, False

                valid = _count_valid_from_labels(labels)
                if valid > 0:
                    nll_sum += float(loss.item()) * valid
                    tok_cnt += valid

                prev_end = end
                if end == L:
                    break

            return nll_sum, tok_cnt, used, False

        details: List[Dict[str, Any]] = []
        ppl_list: List[float] = []

        for st in tqdm(range(0, len(seq_list), batch_size), desc="GenomeOcean PPL (bpb-ready)"):
            batch = seq_list[st: st + batch_size]

            for s in batch:
                enc = _tokenize_one(s, trunc=(not use_sliding_window))
                # print(enc)
                ids_full = enc["input_ids"]  # [1, L]
                L_tok = int(ids_full.size(1))

                # start_pos：token index 空间（0..L-1）
                prompt_tok_len = 0
                if conditional:
                    p = s[: int(prompt_len_chars)]
                    p_enc = _tokenize_one(p, trunc=True)
                    prompt_tok_len = int(p_enc["input_ids"].size(1))
                    start_pos = max(prompt_tok_len, 1)
                else:
                    start_pos = 1

                spec_full = _special_mask(ids_full[0]) if exclude_special else None

                nll_sum, tok_cnt, used_mode, did_trunc = _window_loss_sum(ids_full, start_pos, spec_full)

                if tok_cnt <= 0 or (not math.isfinite(nll_sum)):
                    avg_nll = float("nan")
                    ppl = float("nan")
                else:
                    avg_nll = float(nll_sum) / float(tok_cnt)
                    ppl = float(self._safe_exp(avg_nll))

                # -------- bpb 所需：char_count --------
                # 这里按“原始碱基字符”定义：
                if conditional:
                    char_count = max(int(len(s)) - int(prompt_len_chars), 0)
                else:
                    # next-token 计分对应长度约为 len(s)-1
                    char_count = max(int(len(s)) - 1, 0)

                # 若 direct_truncated 且 unconditional，为了更贴近“实际被计分的尾部”，可改成 decode 后长度
                # （可选：保守保持原始定义；如果你更想按截断尾部算 bpb，可以打开下面块）
                # if (not conditional) and did_trunc and (used_mode == "direct_truncated"):
                #     tail_text = self.tokenizer.decode(ids_full[0, -max_window_tokens:], skip_special_tokens=True,
                #                                     clean_up_tokenization_spaces=False)
                #     tail_text = "".join(tail_text.split())
                #     char_count = max(len(tail_text) - 1, 0)

                if return_details:
                    rec: Dict[str, Any] = {
                        # 兼容旧字段
                        "ppl": ppl,
                        "avg_nll": avg_nll,
                        "token_count": int(tok_cnt),
                        "mode": ("conditional_" + used_mode) if conditional else ("unconditional_" + used_mode),

                        # ✅ bpb 需要的字段（Evo2 风格）
                        "avg_nll_token": avg_nll,     # nat/token
                        "char_count": int(char_count),

                        # 额外诊断
                        "sequence_tokens": int(L_tok),
                        "sequence_chars": int(len(s)),
                        "prompt_len_chars": int(prompt_len_chars) if conditional else 0,
                        "prompt_len_tokens": int(prompt_tok_len) if conditional else 0,
                        "use_sliding_window": bool(use_sliding_window),
                        "max_window_tokens": int(max_window_tokens) if use_sliding_window else None,
                        "stride": int(stride) if use_sliding_window else None,
                        "exclude_special": bool(exclude_special),
                        "did_truncate_tail": bool(did_trunc),
                    }
                    details.append(rec)
                else:
                    ppl_list.append(ppl)

        if return_details:
            return details[0] if single else details
        return ppl_list[0] if single else ppl_list



if __name__ == "__main__":
    MODEL_PATH = "../../model_weight/GenomeOcean-500M"
    m = GenomeOceanModel("GenomeOcean", model_path=MODEL_PATH)

    prompts = ["ATG", "CCGTTAGG"]
    gen = m.generate(
        prompts,
        num_return_sequences=2,
        n_tokens=64,
        temperature=1.3,
        top_k=-1,
        top_p=0.7,
        presence_penalty=0.5,
        frequency_penalty=0.5,
        repetition_penalty=1.0,
        seed=123,
    )
    print("Generated:", gen)

    ppl_u = m.get_ppl(gen, conditional=False, batch_size=2)
    print("Unconditional PPL:", ppl_u)

    ppl_c = m.get_ppl(gen, conditional=True, prompt_len_chars=32, batch_size=2, return_details=True)
    print("Conditional details:", ppl_c)

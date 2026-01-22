# models/genos_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import math
from typing import List, Optional, Union, Literal, Tuple, Dict, Any

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from .base_model import BaseModel


PoolT = Literal["final", "mean", "max", "min"]


class GenosModel(BaseModel):
    """
    Genos 适配器：embedding / generation / perplexity (PPL)

    关键点：
      - Genos-1.2B 的 HF config 显示其 architectures=MixtralForCausalLM, vocab_size=128, pad_token_id=14,
        max_position_embeddings=1,048,576，因此 PPL 并不需要 <=4（因为词表不止 A/C/G/T）。:contentReference[oaicite:0]{index=0}
      - PPL 使用标准 CausalLM next-token NLL：PPL = exp(avg_nll)。可选 sliding-window(stride) 以处理长序列。:contentReference[oaicite:1]{index=1}

    Args:
        model_name: 逻辑名（e.g., "Genos-1.2B"）
        model_path: 本地权重目录（e.g., /data/model/Genos-1.2B）
        hf_home: 可选，显式设置 HF_HOME 缓存目录
        device: 'cuda:0' / 'cpu' / None(自动)
        use_flash_attention: 是否使用 flash_attention_2（需要环境支持）
        torch_dtype: 模型数据类型（默认读取 config.torch_dtype；否则用 bfloat16）
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        hf_home: Optional[str] = None,
        device: Optional[str] = None,
        use_flash_attention: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(model_name, model_path)
        self.hf_home = hf_home
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_flash_attention = use_flash_attention
        self.torch_dtype = torch_dtype  # None -> follow config / HF default
        self._load_model()

    def _load_model(self):
        if self.hf_home:
            os.environ["HF_HOME"] = self.hf_home

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
        }
        if self.torch_dtype is not None:
            model_kwargs["torch_dtype"] = self.torch_dtype

        if self.use_flash_attention:
            # 部分环境 / transformers 版本不支持该字段，失败则忽略
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            except Exception as e:
                print(f"[GenosModel] Warning: flash_attention_2 not available, using default. err={e}")

        # ✅ 必须用 CausalLM 才有 generate / loss / logits
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)

        self.model.to(self.device)
        self.model.eval()

        # 一些常用 id
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            # fallback：优先 eos，再 unk，再 0
            self.pad_id = getattr(self.tokenizer, "eos_token_id", None) \
                          or getattr(self.tokenizer, "unk_token_id", None) \
                          or 0

        self.eos_id = self.tokenizer.eos_token_id
        self.bos_id = self.tokenizer.bos_token_id

        # max len（Genos config 可能是 1,048,576；但实际推理请用 stride/window 控制显存）
        self.model_max_len = (
            getattr(self.model.config, "max_position_embeddings", None)
            or getattr(self.tokenizer, "model_max_length", None)
            or 131072
        )

        print(
            f"[GenosModel] loaded on {self.device}, "
            f"model_max_len={self.model_max_len}, "
            f"vocab_size={getattr(self.model.config, 'vocab_size', None)}, "
            f"pad_id={self.pad_id}"
        )

    # =========================
    # Embedding
    # =========================
    @torch.no_grad()
    def get_embedding(
        self,
        sequences: Union[str, List[str]],
        batch_size: int = 32,
        pool: PoolT = "mean",
        average_reverse_complement: bool = False,
        layer_index: int = -1,                   # -1: 最后一层
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        return_numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        提取 (N, D) 序列向量：
          - pool="final": 取最后一个非 PAD token 的 hidden
          - pool="mean"/"max"/"min": 对非 PAD token 进行池化
          - average_reverse_complement=True: 与反向互补平均
        """
        if isinstance(sequences, str):
            seq_list = [sequences]
        else:
            seq_list = list(sequences)

        tok = self.tokenizer
        model = self.model
        device = torch.device(self.device)

        def _revcomp(seq: str) -> str:
            tbl = str.maketrans("ACGTRYMKBDHVNacgtrymkbdhvn", "TGCAYRKMVHDBNtgcayrkmvhdbn")
            return seq.translate(tbl)[::-1]

        def _tokenize_batch(seqs: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            enc = tok(
                seqs,
                return_tensors="pt",
                padding=True,
                truncation=True if (max_length is not None) else False,
                max_length=max_length if (max_length is not None) else None,
                add_special_tokens=add_special_tokens,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
            lengths = attention_mask.sum(dim=1).long()
            return input_ids, attention_mask, lengths

        def _forward_hidden(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            # CausalLM 输出里 hidden_states 需要显式打开
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            hs = out.hidden_states
            if hs is None:
                raise RuntimeError("Model did not return hidden_states; ensure output_hidden_states=True works.")
            # hs: tuple(len_layers+1) each [B,T,D]
            idx = layer_index if layer_index >= 0 else (len(hs) + layer_index)
            idx = max(0, min(idx, len(hs) - 1))
            return hs[idx]

        def _pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
            # hidden: [B,T,D]
            if pool == "final":
                idx = torch.clamp(lengths - 1, min=0)
                bidx = torch.arange(hidden.size(0), device=hidden.device)
                return hidden[bidx, idx, :]
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # [B,T,1]

            if pool == "mean":
                summed = (hidden * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp_min(1.0)
                return summed / denom

            if pool == "max":
                masked = hidden.masked_fill(mask == 0, float("-inf"))
                v = masked.max(dim=1).values
                # 全 -inf 的极端情况兜底
                bad = torch.isinf(v).any(dim=1)
                if bad.any():
                    v[bad] = hidden[bad].max(dim=1).values
                return v

            if pool == "min":
                masked = hidden.masked_fill(mask == 0, float("inf"))
                v = masked.min(dim=1).values
                bad = torch.isinf(v).any(dim=1)
                if bad.any():
                    v[bad] = hidden[bad].min(dim=1).values
                return v

            raise ValueError(f"Unknown pool='{pool}'")

        out_vecs: List[torch.Tensor] = []
        use_rc = bool(average_reverse_complement)

        for st in tqdm(range(0, len(seq_list), batch_size), desc="Genos embedding"):
            batch = seq_list[st: st + batch_size]
            ids_f, am_f, len_f = _tokenize_batch(batch)
            hid_f = _forward_hidden(ids_f, am_f)
            vec_f = _pool_hidden(hid_f, am_f, len_f)

            if use_rc:
                rc_batch = [_revcomp(s) for s in batch]
                ids_r, am_r, len_r = _tokenize_batch(rc_batch)
                hid_r = _forward_hidden(ids_r, am_r)
                vec_r = _pool_hidden(hid_r, am_r, len_r)
                vec = 0.5 * (vec_f + vec_r)
            else:
                vec = vec_f

            out_vecs.append(vec.detach().to(torch.float32).cpu())

        out = torch.cat(out_vecs, dim=0) if out_vecs else torch.empty(0, 0)
        return out.numpy() if return_numpy else out

    # =========================
    # Generation
    # =========================
    @torch.no_grad()
    def generate(
        self,
        prompt_seqs: Union[str, List[str]],
        n_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        batch_size: int = 8,
        add_special_tokens: bool = True,
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """
        使用 HuggingFace generate 生成序列（返回：prompt+generated）
        """
        if isinstance(prompt_seqs, str):
            prompt_list = [prompt_seqs]
        else:
            prompt_list = list(prompt_seqs)

        tok = self.tokenizer
        model = self.model
        device = torch.device(self.device)

        # pad/eos 兜底
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else self.pad_id
        eos_id = tok.eos_token_id if tok.eos_token_id is not None else self.eos_id

        results: List[str] = []
        for st in tqdm(range(0, len(prompt_list), batch_size), desc="Genos generate"):
            batch = prompt_list[st: st + batch_size]
            enc = tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=add_special_tokens,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

            gen_kwargs = dict(
                max_new_tokens=int(n_tokens),
                do_sample=bool(do_sample),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                repetition_penalty=float(repetition_penalty),
                pad_token_id=int(pad_id),
                eos_token_id=int(eos_id) if eos_id is not None else None,
                use_cache=True,
            )

            out_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

            texts = tok.batch_decode(
                out_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=False,
            )
            texts = [t.replace(" ", "").replace("\n", "").replace("\t", "") for t in texts]
            results.extend(texts)

        return results

    # =========================
    # PPL / scoring
    # =========================
    @staticmethod
    def _safe_exp(x: float, max_x: float = 700.0) -> float:
        if math.isnan(x):
            return float("nan")
        if math.isinf(x):
            return float("inf") if x > 0 else 0.0
        if x > max_x:
            return float("inf")
        try:
            return math.exp(x)
        except OverflowError:
            return float("inf")

    @torch.no_grad()
    def get_ppl(
        self,
        sequences: Union[str, List[str]],
        *,
        prompt_len_chars: Optional[int] = None,     # None => full ppl；否则 conditional ppl
        ppl_mode: Literal["token", "char"] = "token",
        # sliding-window（强烈建议长序列启用）
        use_sliding_window: bool = True,
        max_window_tokens: int = 4096,
        stride: int = 1024,
        # tokenize
        add_special_tokens: bool = True,
        max_length_tokens: Optional[int] = None,    # 可选：截断评估长度（token）
        batch_size: int = 1,
        return_details: bool = True,
    ) -> Union[float, List[float], List[Dict[str, Any]]]:
        """
        Genos PPL（full / conditional）统一输出字段：
          - token_count：参与计分的 target token 数
          - char_count：参与计分的碱基字符数（conditional: continuation chars；full: len(seq)-1）
          - avg_nll_token：total_nll / token_count
          - avg_nll_char ：total_nll / char_count
          - ppl_token = exp(avg_nll_token)
          - ppl_char  = exp(avg_nll_char)
          - ppl（主输出）由 ppl_mode 决定

        conditional 定义（按字符切分）：
          prompt = seq[:prompt_len_chars]
          continuation = seq[prompt_len_chars:]
          只对 continuation 对应的 token 计分
        """
        import warnings
        import torch.nn.functional as F

        single = isinstance(sequences, str)
        seq_list = [sequences] if single else list(sequences)

        # 清理：去空格/换行（解决 "A T G ..." 这种输入）
        def _clean_seq(s: str) -> str:
            return (s or "").replace(" ", "").replace("\n", "").replace("\t", "").strip().upper()

        seq_list = [_clean_seq(s) for s in seq_list]

        tok = self.tokenizer
        model = self.model
        device = torch.device(self.device)
        model.eval()

        # window/stride 合法化
        max_window_tokens = int(max(2, max_window_tokens))
        stride = int(max(1, min(stride, max_window_tokens)))

        def _safe_exp(x: float, max_x: float = 700.0) -> float:
            if math.isnan(x):
                return float("nan")
            if math.isinf(x):
                return float("inf") if x > 0 else 0.0
            if x > max_x:
                return float("inf")
            try:
                return math.exp(x)
            except OverflowError:
                return float("inf")

        def _tokenize_one(s: str) -> Tuple[torch.Tensor, torch.Tensor]:
            enc = tok(
                s,
                return_tensors="pt",
                add_special_tokens=add_special_tokens,
                truncation=True if (max_length_tokens is not None) else False,
                max_length=int(max_length_tokens) if (max_length_tokens is not None) else None,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
            return input_ids, attention_mask

        def _prompt_tok_len_by_chars(s: str) -> int:
            if prompt_len_chars is None:
                return 0
            p = s[: int(prompt_len_chars)]
            ids_p, _am_p = _tokenize_one(p)
            return int(ids_p.size(1))

        def _window_nll(
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            start_label_pos: int,   # 全局 label 位置起点（>=1）；full:1；conditional:prompt_tok_len
        ) -> Tuple[float, int]:
            """
            sliding-window 计算 total_nll 与 token_count。
            label_pos 指的是原始 input_ids 的位置（0..L-1），但 loss 内部会 shift，
            所以 label_pos=0 其实不会被计入；这里用 start_label_pos>=1 控制。
            """
            L = int(input_ids.size(1))
            if L <= 1:
                return float("nan"), 0

            total_nll = 0.0
            total_tok = 0

            win = min(max_window_tokens, L)

            # 按 stride 滚动，每次仅计分窗口末尾 stride 个 label，且还要满足 >= start_label_pos
            for begin in range(0, L, stride):
                end = min(begin + win, L)
                W = end - begin
                if W <= 1:
                    break

                ids_w = input_ids[:, begin:end]         # [1, W]
                am_w = attention_mask[:, begin:end]     # [1, W]

                labels = ids_w.clone()
                labels[am_w == 0] = -100

                # 只计分窗口末尾 stride 个 label（避免重复计分）
                keep_from = max(0, W - stride)

                # 同时：只计分 global label_pos >= start_label_pos
                # window 内 pos -> global_pos = begin + pos
                # 所以需要 pos >= (start_label_pos - begin)
                keep_from = max(keep_from, start_label_pos - begin)

                # loss 内部会 shift，label[0] 不会被预测；但这里保守直接屏蔽到 keep_from
                if keep_from > 0:
                    labels[:, :keep_from] = -100

                # 统计有效 label 数（shift 后会少 1，但 HF loss 已处理；我们用 labels!=-100 近似即可）
                valid = int((labels != -100).sum().item())
                if valid <= 0:
                    if end == L:
                        break
                    continue

                out = model(
                    input_ids=ids_w,
                    attention_mask=am_w,
                    labels=labels,
                    use_cache=False,
                    return_dict=True,
                )
                loss = out.loss
                if loss is None or torch.isnan(loss):
                    return float("nan"), 0

                total_nll += float(loss.item()) * valid
                total_tok += valid

                if end == L:
                    break

            return float(total_nll), int(total_tok)

        def _direct_nll(
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            start_shift_pos: int,   # shift space 起点；full:0；conditional:prompt_tok_len-1
        ) -> Tuple[float, int]:
            """
            不用 sliding-window：一次前向，用 shift_logits + mask 精确计分
            返回 total_nll, token_count
            """
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            logits = out.logits
            if logits is None:
                return float("nan"), 0

            logits = logits.float()
            shift_logits = logits[:, :-1, :].contiguous()            # [1, L-1, V]
            shift_labels = input_ids[:, 1:].contiguous()             # [1, L-1]
            shift_valid  = attention_mask[:, 1:].bool().contiguous() # [1, L-1]

            start_shift_pos = max(int(start_shift_pos), 0)
            cont_mask = torch.zeros_like(shift_valid)
            if start_shift_pos < cont_mask.size(1):
                cont_mask[:, start_shift_pos:] = True

            final_mask = shift_valid & cont_mask
            V = shift_logits.size(-1)

            token_nll = F.cross_entropy(
                shift_logits.view(-1, V),
                shift_labels.view(-1),
                reduction="none",
            ).view(1, -1)

            nll_sum = float((token_nll * final_mask).sum().item())
            tok_cnt = int(final_mask.sum().item())
            if tok_cnt <= 0:
                return float("nan"), 0
            return nll_sum, tok_cnt

        details: List[Dict[str, Any]] = []
        ppl_list: List[float] = []

        for st in tqdm(range(0, len(seq_list), batch_size), desc="Genos PPL"):
            batch = seq_list[st: st + batch_size]
            for idx_in_batch, s in enumerate(batch):
                seq_id = st + idx_in_batch
                s_clean = s
                char_len = len(s_clean)

                input_ids, attention_mask = _tokenize_one(s_clean)
                L_tok = int(input_ids.size(1))


                # prompt token len（按 char 截 prompt，再 tokenize）
                prompt_tok_len = _prompt_tok_len_by_chars(s_clean) if (prompt_len_chars is not None) else 0

                # --- decide scoring range ---
                if prompt_len_chars is None:
                    # full ppl：从 label_pos=1 开始（对应预测 token1..）
                    start_label_pos = 1
                    start_shift_pos = 0
                    cont_char_count = max(char_len - 1, 0)
                    mode = "unconditional"
                else:
                    # conditional：从 label_pos = prompt_tok_len 开始（对应 shift 起点 prompt_tok_len-1）
                    start_label_pos = max(prompt_tok_len, 1)
                    start_shift_pos = max(prompt_tok_len - 1, 0)
                    cont_char_count = max(char_len - int(prompt_len_chars), 0)
                    mode = "conditional"

                # 若 prompt_tok_len >= L_tok，说明没有 continuation token 可算
                if prompt_len_chars is not None and prompt_tok_len >= L_tok:
                    rec = {
                        "sequence_id": seq_id,
                        "sequence_chars": int(char_len),
                        "prompt_len_chars": int(prompt_len_chars),
                        "char_count": int(cont_char_count),

                        "sequence_tokens": int(L_tok),
                        "prompt_tokens": int(prompt_tok_len),
                        "token_count": 0,

                        "avg_nll_token": float("nan"),
                        "avg_nll_char": float("nan"),
                        "ppl_token": float("nan"),
                        "ppl_char": float("nan"),
                        "ppl": float("nan"),
                        "mode": mode,
                        "error": "no_continuation_tokens",
                    }
                    details.append(rec)
                    ppl_list.append(float("nan"))
                    continue

                # --- compute total_nll & token_count ---
                if use_sliding_window:
                    total_nll, tok_cnt = _window_nll(input_ids, attention_mask, start_label_pos=start_label_pos)
                    calc_mode = mode + "_sliding"
                else:
                    total_nll, tok_cnt = _direct_nll(input_ids, attention_mask, start_shift_pos=start_shift_pos)
                    calc_mode = mode + "_direct"

                if tok_cnt <= 0 or (not math.isfinite(total_nll)):
                    rec = {
                        "sequence_id": seq_id,
                        "sequence_chars": int(char_len),
                        "prompt_len_chars": int(prompt_len_chars) if prompt_len_chars is not None else 0,
                        "char_count": int(cont_char_count),

                        "sequence_tokens": int(L_tok),
                        "prompt_tokens": int(prompt_tok_len),
                        # "token_count": int(tok_cnt),
                        "token_count": int(L_tok)-1, # 减去 bos 的 token

                        "avg_nll_token": float("nan"),
                        "avg_nll_char": float("nan"),
                        "ppl_token": float("nan"),
                        "ppl_char": float("nan"),
                        "ppl": float("nan"),
                        "mode": calc_mode,
                        "error": "no_valid_tokens",
                    }
                    details.append(rec)
                    ppl_list.append(float("nan"))
                    continue

                avg_nll_token = float(total_nll) / float(tok_cnt)
                ppl_token = float(_safe_exp(avg_nll_token))

                if cont_char_count > 0:
                    avg_nll_char = float(total_nll) / float(cont_char_count)
                    ppl_char = float(_safe_exp(avg_nll_char))
                else:
                    avg_nll_char = float("nan")
                    ppl_char = float("nan")

                ppl = ppl_char if ppl_mode == "char" else ppl_token

                rec = {
                    "sequence_id": seq_id,
                    "sequence_chars": int(char_len),
                    "prompt_len_chars": int(prompt_len_chars) if prompt_len_chars is not None else 0,
                    "char_count": int(cont_char_count),

                    "sequence_tokens": int(L_tok),
                    "prompt_tokens": int(prompt_tok_len),
                    "token_count": int(tok_cnt),

                    "avg_nll_token": float(avg_nll_token),
                    "avg_nll_char": float(avg_nll_char),
                    "ppl_token": float(ppl_token),
                    "ppl_char": float(ppl_char),
                    "ppl": float(ppl),
                    "mode": calc_mode,
                    "max_window_tokens": int(max_window_tokens) if use_sliding_window else None,
                    "stride": int(stride) if use_sliding_window else None,
                }

                details.append(rec)
                ppl_list.append(float(ppl))

        if return_details:
            return details

        return ppl_list[0] if single else ppl_list


    @torch.no_grad()
    def score_sequences(
        self,
        sequences: List[str],
        batch_size: int = 8,
        add_special_tokens: bool = True,
    ) -> List[float]:
        """
        返回每条序列的平均 log-prob（越大越好），与 PPL 互为单调变换：
           avg_logprob = -avg_nll
        """
        scores: List[float] = []
        tok = self.tokenizer
        model = self.model
        device = torch.device(self.device)

        for st in tqdm(range(0, len(sequences), batch_size), desc="Genos scoring"):
            batch = sequences[st: st + batch_size]
            enc = tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=add_special_tokens,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            logits = out.logits  # [B,T,V]
            if logits is None:
                scores.extend([float("nan")] * len(batch))
                continue

            logits = logits.float()
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_valid = attention_mask[:, 1:].bool().contiguous()

            B, Lm1, V = shift_logits.shape
            token_nll = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, V),
                shift_labels.view(-1),
                reduction="none",
            ).view(B, Lm1)

            nll_sum = (token_nll * shift_valid).sum(dim=1)
            n_tok = shift_valid.sum(dim=1).clamp_min(1)

            avg_nll = (nll_sum / n_tok).detach().cpu().numpy()
            # avg_logprob = -avg_nll
            scores.extend([float(-x) for x in avg_nll])

        return scores
    


# -------------------------
# Quick self-test
# -------------------------
if __name__ == "__main__":
    MODEL_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/Genos-1.2B"
    HF_HOME = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model"

    m = GenosModel(
        model_name="Genos-1.2B",
        model_path=MODEL_DIR,
        hf_home=HF_HOME,
        device=None,
        use_flash_attention=False,
        torch_dtype=torch.bfloat16,  # 也可以 None 跟随 config(float32)
    )

    seqs = ["ACGT" * 128, "AAAAACCCCGGGGTTTT" * 32]

    emb = m.get_embedding(seqs, pool="mean", batch_size=2)
    print("Embedding shape:", emb.shape)

    prompts = ["ATG", "CCG"]
    gen = m.generate(prompts, n_tokens=32, temperature=1.0, top_p=0.9, top_k=50, batch_size=2)
    print("Generated:", gen)

    ppl_list = m.get_ppl(seqs, max_window_tokens=2048, stride=512, batch_size=1, return_details=False)
    print("PPL:", ppl_list)

    # conditional ppl（只评估 continuation）
    cond = m.get_ppl(seqs, prompt_len_chars=32, batch_size=1, return_details=True)
    print("Conditional details:", cond)

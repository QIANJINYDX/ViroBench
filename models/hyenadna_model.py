# models/hyenadna_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, inspect
from typing import List, Optional, Literal, Dict, Any, Union

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)

from .base_model import BaseModel

from tqdm import tqdm
import torch.nn.functional as F

import math
import warnings
from typing import Tuple

Pooling = Literal["mean", "max", "cls"]

def _revcomp(seq: str) -> str:
    tbl = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(tbl)[::-1]


class HyenaDNAModel(BaseModel):
    """
    HyenaDNA 适配器（兼容部分实现不支持 attention_mask/return_dict 的 forward）
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        task: Literal["classification", "embedding"] = "classification",
        hf_home: Optional[str] = None,
        device: Optional[str] = None,
        device_map: Optional[str] = "auto",             # 与你示例一致
        torch_dtype: Optional[torch.dtype] = torch.bfloat16,
        trust_remote_code: bool = True,
    ):
        super().__init__(model_name, model_path)
        self.task = task
        self.hf_home = hf_home
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self._load_model()

    # ---------- 加载 ----------
    def _load_model(self):
        if self.hf_home:
            os.environ["HF_HOME"] = self.hf_home

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=self.trust_remote_code
        )

        common_kwargs: Dict[str, Any] = dict(
            trust_remote_code=self.trust_remote_code,
        )

        self._common_kwargs = common_kwargs

        if self.torch_dtype is not None:
            common_kwargs["torch_dtype"] = self.torch_dtype
        if self.device_map is not None:
            common_kwargs["device_map"] = self.device_map

        if self.task == "classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path, **common_kwargs
            )
        elif self.task == "embedding":
            self.model = AutoModel.from_pretrained(self.model_path, **common_kwargs)
        else:
            raise ValueError(f"Unknown task: {self.task}")

        if self.device_map is None:  # 非分布式映射时，放到单设备
            self.model.to(self.device)
        self.model.eval()

        # ---- 动态探测 forward 支持的参数 ----
        try:
            sig = inspect.signature(self.model.forward)
            self._accepts_attn = "attention_mask" in sig.parameters
            self._accepts_return_dict = "return_dict" in sig.parameters
            self._accepts_hidden = "output_hidden_states" in sig.parameters
        except Exception:
            self._accepts_attn = False
            self._accepts_return_dict = False
            self._accepts_hidden = False

        self.model_max_len = getattr(self.model.config, "max_position_embeddings", None) or \
                             getattr(self.tokenizer, "model_max_length", None)
        self.num_labels = getattr(self.model.config, "num_labels", None)

        print(f"[HyenaDNAModel] loaded: task={self.task}, device={self.device}, "
              f"device_map={self.device_map}, dtype={self.torch_dtype}, "
              f"max_len={self.model_max_len}, num_labels={self.num_labels}, "
              f"accepts(attn={self._accepts_attn}, return_dict={self._accepts_return_dict}, "
              f"output_hidden_states={self._accepts_hidden})")

    # ---------- Tokenize ----------
    def _tok_batch(
        self,
        sequences: List[str],
        truncation: bool = True,
        padding: bool = True,
        max_length: Optional[int] = None,
    ):
        enc = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=("longest" if padding else False),
            truncation=truncation,
            max_length=max_length,
        )
        input_ids = enc["input_ids"]
        # 某些 tokenizer 不返回 attention_mask；统一这里做兜底，但调用时仅在支持时传入
        attn_mask = enc.get("attention_mask", torch.ones_like(input_ids))
        return input_ids, attn_mask

    # ---------- 分类：概率/预测 ----------
    @torch.no_grad()
    def predict_proba(
        self,
        sequences: List[str],
        batch_size: int = 1,
        truncation: bool = True,
        max_length: Optional[int] = None,
        rc_augment: bool = False,
        return_numpy: bool = True,
    ):
        if self.task != "classification":
            raise RuntimeError("predict_proba 仅在 task='classification' 下可用。")

        all_probs: List[np.ndarray] = []
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            if rc_augment:
                batch_rc = [_revcomp(x) for x in batch]

            # 正向
            ids, mask = self._tok_batch(batch, truncation=truncation, max_length=max_length)
            dev = self.model.device if self.device_map is not None else self.device
            ids = ids.to(dev); mask = mask.to(dev)

            kwargs = {"input_ids": ids}
            if self._accepts_attn:
                kwargs["attention_mask"] = mask
            if self._accepts_return_dict:
                kwargs["return_dict"] = True

            out = self.model(**kwargs)

            # 兼容多种返回
            if hasattr(out, "logits"):
                logits = out.logits
            elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                logits = out[0]
            elif torch.is_tensor(out):
                logits = out
            else:
                raise RuntimeError("无法从模型输出中解析 logits。")

            probs = torch.softmax(logits.float(), dim=-1)  # (B, C)

            if rc_augment:
                ids_rc, mask_rc = self._tok_batch(batch_rc, truncation=truncation, max_length=max_length)
                ids_rc = ids_rc.to(dev); mask_rc = mask_rc.to(dev)
                kwargs_rc = {"input_ids": ids_rc}
                if self._accepts_attn:
                    kwargs_rc["attention_mask"] = mask_rc
                if self._accepts_return_dict:
                    kwargs_rc["return_dict"] = True
                out_rc = self.model(**kwargs_rc)
                if hasattr(out_rc, "logits"):
                    logits_rc = out_rc.logits
                elif isinstance(out_rc, (tuple, list)) and len(out_rc) > 0 and torch.is_tensor(out_rc[0]):
                    logits_rc = out_rc[0]
                elif torch.is_tensor(out_rc):
                    logits_rc = out_rc
                else:
                    raise RuntimeError("无法从 RC 输出中解析 logits。")
                probs_rc = torch.softmax(logits_rc.float(), dim=-1)
                probs = 0.5 * (probs + probs_rc)

            if return_numpy:
                all_probs.extend(probs.cpu().numpy())
            else:
                all_probs.extend([probs[i] for i in range(probs.shape[0])])

        return np.stack(all_probs, axis=0) if return_numpy else all_probs

    @torch.no_grad()
    def predict(self, sequences: List[str], **kwargs) -> List[int]:
        probs = self.predict_proba(sequences, **kwargs)
        return probs.argmax(axis=-1).tolist()
    @torch.no_grad()
    def score_sequences(self, sequences: List[str], batch_size: int = 256) -> List[float]:
        """
        对序列进行评分
        
        Args:
            sequences: 序列列表
            batch_size: 批处理大小
            
        Returns:
            每个序列的 log-likelihood 得分
        """
        all_scores = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc="Scoring"):
                batch_seqs = sequences[i:i + batch_size]
                batch_scores = self._score_batch(batch_seqs)
                all_scores.extend(batch_scores)
        
        return all_scores
    @torch.no_grad()
    def _score_batch(self, sequences: List[str]) -> List[float]:
        """对单个批次评分"""
        # 1. 分词和预处理
        input_ids, seq_lengths = self._prepare_batch(sequences)
        
        # 2. 模型前向传播
        kwargs = {"input_ids": input_ids}
        if self._accepts_hidden:
            kwargs["output_hidden_states"] = True
        if self._accepts_return_dict:
            kwargs["return_dict"] = True
        with torch.inference_mode():
            outputs = self.model(**kwargs)
        logits = outputs.hidden_states[-1]
        
        # 3. 计算 log probabilities
        # log_probs = F.log_softmax(logits, dim=-1)
        log_probs = self._compute_log_probabilities(logits, input_ids)
        
        # 4. 聚合得分
        scores = self._aggregate_scores(log_probs, seq_lengths)
        
        return scores
    @torch.no_grad()
    def _prepare_batch(self, sequences: List[str]):
        """分词和批处理"""
        seq_lengths = [len(seq) for seq in sequences]
        
        # 使用 tokenizer 编码
        all_token_ids = []
        for seq in sequences:
            token_ids = self.tokenizer.encode(seq, add_special_tokens=False)
            # 截断以适应模型最大长度
            if self.model_max_len is not None and len(token_ids) > self.model_max_len:
                token_ids = token_ids[:self.model_max_len]
            all_token_ids.append(token_ids)
        
        # Padding 到相同长度
        max_length = max(len(ids) for ids in all_token_ids)
        if self.model_max_len is not None:
            max_length = min(max_length, self.model_max_len)
            
        padded_ids = []
        # print("self.tokenizer.pad_token",self.tokenizer)
        for token_ids in all_token_ids:
            if len(token_ids) < max_length:
                # print("tokenizer",self.tokenizer)
                padded = token_ids + [4] * (max_length - len(token_ids))
            else:
                padded = token_ids[:max_length]
            padded_ids.append(padded)
        # print("padded_ids",padded_ids)
        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=self.model.device)
        return input_ids, seq_lengths
    @torch.no_grad()
    def _compute_log_probabilities(
        self,
        logits: torch.Tensor,          # [B, L, V]
        input_ids: torch.Tensor,       # [B, L]
        attention_mask: torch.Tensor | None = None  # [B, L]
    ) -> torch.Tensor:
        """计算逐 token 的对数概率，返回 [B, L_eff]"""
        if logits.dim() != 3:
            raise ValueError(f"期望 logits 形状 [B, L, V]，实际 {tuple(logits.shape)}")

        B, L_logits, V = logits.shape
        L_input = input_ids.size(1)
        # 对齐长度，按“右移”规则计算可用步数
        L_eff = min(L_logits - 1, L_input - 1)
        if L_eff <= 0:
            return logits.new_zeros((B, 0))

        # 右移对齐：logits 的位置 t 预测 targets 的位置 t+1
        log_probs = F.log_softmax(logits[:, :L_eff, :].float(), dim=-1)      # [B, L_eff, V]
        targets   = input_ids[:, 1:1 + L_eff]                                # [B, L_eff]

        token_log_probs = torch.gather(log_probs, 2, targets.unsqueeze(-1)).squeeze(-1)  # [B, L_eff]

        # 屏蔽 padding
        if attention_mask is not None:
            mask = attention_mask[:, 1:1 + L_eff].to(dtype=token_log_probs.dtype)        # [B, L_eff]
            token_log_probs = token_log_probs * mask

        # softmax_logprobs = torch.log_softmax(logits, dim=-1)
        # softmax_logprobs = softmax_logprobs[:, :-1] # Remove last prediction.
        # input_ids = input_ids[:, 1:] # Trim BOS added by tokenizer.
        # assert(softmax_logprobs.shape[1] == input_ids.shape[1])

        # token_log_probs = torch.gather(
        #     softmax_logprobs,       # Gather likelihoods...
        #     2,                      # along the vocab dimension...
        #     input_ids.unsqueeze(-1) # using the token ids to index.
        # ).squeeze(-1)
        return token_log_probs
    @torch.no_grad()
    def _aggregate_scores(self, log_probs: torch.Tensor, seq_lengths: List[int]):
        """聚合序列得分"""
        log_probs_np = log_probs.float().cpu().numpy()
        scores = []
        
        for idx, seq_len in enumerate(seq_lengths):
            # 只计算实际序列部分，排除 padding
            valid_length = min(seq_len - 1, log_probs_np.shape[1])
            if valid_length > 0:
                valid_log_probs = log_probs_np[idx][:valid_length]
                score = float(np.mean(valid_log_probs))  # 平均 log-likelihood
            else:
                score = 0.0
            scores.append(score)
        
        return scores

    # ---------- 嵌入提取 ----------
    @torch.no_grad()
    def embed_sequences(
        self,
        sequences: List[str],
        batch_size: int = 1,
        pooling: Pooling = "mean",
        exclude_special: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_numpy: bool = True,
    ):
        embs: List[np.ndarray] = []
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            ids, mask = self._tok_batch(batch, truncation=truncation, max_length=max_length)
            dev = self.model.device if self.device_map is not None else self.device
            ids = ids.to(dev); mask = mask.to(dev)

            kwargs = {"input_ids": ids}
            if self._accepts_attn:
                kwargs["attention_mask"] = mask
            # 只有在模型 forward 支持时才请求 hidden states
            if self._accepts_hidden:
                kwargs["output_hidden_states"] = True
            if self._accepts_return_dict:
                kwargs["return_dict"] = True

            out = self.model(**kwargs)

            # 解析隐藏态
            hidden = None
            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                hidden = out.last_hidden_state
            elif hasattr(out, "hidden_states") and out.hidden_states:
                hidden = out.hidden_states[-1]
            elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                hidden = out[0]
            elif torch.is_tensor(out):
                hidden = out
            if hidden is None:
                raise RuntimeError("无法获得隐藏态，请在 task='embedding' 下使用或确认模型支持 hidden states。")

            B, L, H = hidden.shape
            if pooling == "cls":
                pooled = hidden[:, 0, :]
            else:
                pooled_list = []
                for i in range(B):
                    mask_i = mask[i].bool()
                    if exclude_special:
                        spec = self.tokenizer.get_special_tokens_mask(
                            ids[i].tolist(), already_has_special_tokens=True
                        )
                        spec_mask = torch.tensor(spec, dtype=torch.bool, device=ids.device)
                        valid = mask_i & (~spec_mask)
                    else:
                        valid = mask_i
                    if not valid.any():
                        valid = mask_i  # 兜底
                    h = hidden[i][valid]
                    v = h.mean(dim=0) if pooling == "mean" else h.max(dim=0)[0]
                    pooled_list.append(v)
                pooled = torch.stack(pooled_list, dim=0)

            if return_numpy:
                embs.extend(pooled.float().cpu().numpy())
            else:
                embs.extend([pooled[i] for i in range(pooled.shape[0])])

        return np.stack(embs, axis=0) if return_numpy else embs
    @torch.no_grad()
    def get_embedding(
        self,
        sequences: Union[str, List[str]],
        batch_size: int = 1,
        pool: Pooling = "mean",           # "cls" | "mean" | "max"
        exclude_special: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_numpy: bool = True,
    ):
        """
        获取每条序列的向量表示，返回形状 (N, H)（或张量列表）。
        """
        # 规范输入为列表
        if isinstance(sequences, str):
            sequences = [sequences]
        else:
            sequences = list(sequences)

        embs: List[np.ndarray] = []
        for start in tqdm(range(0, len(sequences), batch_size), desc="Getting embedding"):
            batch = sequences[start: start + batch_size]
            ids, mask = self._tok_batch(batch, truncation=truncation, max_length=max_length)

            dev = self.model.device if getattr(self, "device_map", None) is not None else self.device
            ids = ids.to(dev)
            mask = mask.to(dev)

            kwargs = {"input_ids": ids}
            if getattr(self, "_accepts_attn", False):
                kwargs["attention_mask"] = mask
            if getattr(self, "_accepts_hidden", False):
                kwargs["output_hidden_states"] = True
            if getattr(self, "_accepts_return_dict", False):
                kwargs["return_dict"] = True

            out = self.model(**kwargs)

            # 解析隐藏态(优先 last_hidden_state，其次 hidden_states[-1]，再退化到 out[0]/out)
            hidden = None
            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                hidden = out.last_hidden_state
            elif hasattr(out, "hidden_states") and out.hidden_states:
                hidden = out.hidden_states[-1]
            elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                hidden = out[0]
            elif torch.is_tensor(out):
                hidden = out
            if hidden is None:
                raise RuntimeError("无法获得隐藏态，请在 task='embedding' 下使用或确认模型支持 hidden states。")

            # 池化
            if pool == "cls":
                pooled = hidden[:, 0, :]  # [B, H]
            else:
                pooled_list = []
                B = hidden.size(0)
                for i in range(B):
                    mask_i = mask[i].bool()
                    if exclude_special:
                        spec = self.tokenizer.get_special_tokens_mask(
                            ids[i].tolist(), already_has_special_tokens=True
                        )
                        spec_mask = torch.tensor(spec, dtype=torch.bool, device=ids.device)
                        valid = mask_i & (~spec_mask)
                    else:
                        valid = mask_i
                    if not valid.any():
                        valid = mask_i  # 兜底

                    h = hidden[i][valid]  # [L_valid, H]
                    v = h.mean(dim=0) if pool == "mean" else h.max(dim=0)[0]
                    pooled_list.append(v)
                pooled = torch.stack(pooled_list, dim=0)  # [B, H]

            if return_numpy:
                embs.extend(pooled.float().cpu().numpy())
            else:
                # 若需要 tensor，按条返回
                embs.extend([pooled[j] for j in range(pooled.shape[0])])

        return np.stack(embs, axis=0) if return_numpy else embs
        # =============================
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
        生成 / PPL 需要 CausalLM head。这里做 lazy load：
        - 不改变你原先 classification / embedding 的 self.model
        - 额外加载 self.lm_model
        """
        if hasattr(self, "lm_model") and self.lm_model is not None:
            return

        self.lm_model = AutoModelForCausalLM.from_pretrained(
            self.model_path, **getattr(self, "_common_kwargs", {"trust_remote_code": self.trust_remote_code})
        )
        if self.device_map is None:
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
        logits: (V,)
        """
        assert logits.dim() == 1

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
        HyenaDNA 序列生成（手写 sampling，不依赖 transformers.generate）
        - 支持 prompt 为 str 或 List[str]
        - 若 prompt 为 str：会复制为 n_samples 条生成
        """
        self._ensure_lm_model()
        model = self.lm_model
        tok = self.tokenizer
        dev = self._infer_model_input_device(model)

        # 规范 prompts
        if isinstance(prompt_seqs, str):
            prompts = [prompt_seqs] * int(n_samples)
        else:
            prompts = list(prompt_seqs)
            if len(prompts) == 1 and int(n_samples) > 1:
                prompts = prompts * int(n_samples)

        outputs: List[str] = []
        model.eval()

        with torch.inference_mode():
            for p in prompts:
                enc = tok(p, return_tensors="pt", add_special_tokens=False)
                input_ids = enc["input_ids"].to(dev)

                if max_prompt_tokens is not None and input_ids.size(1) > int(max_prompt_tokens):
                    input_ids = input_ids[:, -int(max_prompt_tokens):]

                for _ in range(int(n_tokens)):
                    out = model(input_ids=input_ids)
                    logits = out.logits  # (1, L, V)
                    next_logits = logits[0, -1, :].float()

                    if temperature is not None and float(temperature) > 0:
                        next_logits = next_logits / float(temperature)

                    next_logits = self._top_k_top_p_filtering(
                        next_logits, top_k=int(top_k), top_p=float(top_p)
                    )
                    probs = F.softmax(next_logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)  # (1,)

                    input_ids = torch.cat([input_ids, next_id.view(1, 1)], dim=1)

                txt = tok.decode(input_ids[0], skip_special_tokens=True)
                txt = txt.replace(" ", "")  # char-level tokenizer 有时会带空格
                outputs.append(txt)

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

        # ---- pre-tokenize & prompt token lengths ----
        seq_infos: List[Tuple[int, str, List[int], int, int, int]] = []
        # tuple: (orig_i, seq, ids, prompt_tok_len, seq_char_len, cont_char_len)
        for i, s in enumerate(seq_list):
            try:
                ids = tok(s, add_special_tokens=False)["input_ids"]
                ids = list(map(int, ids))
                if max_length is not None and len(ids) > int(max_length):
                    ids = ids[: int(max_length)]

                seq_char_len = len(s)

                if prompt_len_chars is not None:
                    p_chars = min(int(prompt_len_chars), seq_char_len)
                    cont_char_len = max(seq_char_len - p_chars, 0)

                    p_str = s[:p_chars]
                    p_ids = tok(p_str, add_special_tokens=False)["input_ids"]
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
            for st in tqdm(range(0, len(seq_infos_sorted), batch_size), desc="HyenaDNA PPL"):
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




# # # ---------- 自测 ----------
if __name__ == "__main__":
    CKPT = "../../model_weight/hyenadna-medium-160k-seqlen-hf"

    # 示例 1：分类推理（注意：若权重不含已训练分类头，输出将随机）
    clf = HyenaDNAModel(
        model_name="hyenadna-medium-160k-seqlen-hf",
        model_path=CKPT,
        task="classification",
        device_map="auto",
        torch_dtype=torch.float32,
    )

    max_length = 160_000
    seq = "ACTG" * (max_length // 4)
    seqs = [seq] * 2

    probs = clf.predict_proba(seqs, batch_size=1, truncation=True)
    print("scores (label 1 log-prob):", clf.score_sequences(seqs, batch_size=2)[:2])

    # 示例 2：embedding（如果仅做零样本/下游检索，建议用这个）
    enc = HyenaDNAModel(
        model_name="hyenadna-medium-160k-seqlen-hf",
        model_path=CKPT,
        task="embedding",
        device_map="auto",
        torch_dtype=torch.float32,
    )
    X = enc.get_embedding(seqs, pool="mean", truncation=True)
    print("emb shape:", X.shape)
    print("emb:", X)

    gen = enc.generate(prompt_seqs="ACGTACGT", n_samples=2, n_tokens=100, temperature=1.0, top_k=4, top_p=1.0)
    print("generated:", gen)

    seqs = ["ACGT" * 200, "ATGC" * 220]
    ppl_full = enc.get_ppl(seqs, batch_size=2, return_details=False)
    print("ppl_full:", ppl_full)



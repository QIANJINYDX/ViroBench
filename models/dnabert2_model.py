# models/dnabert2_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from typing import List, Optional, Literal, Union, Tuple

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

from .base_model import BaseModel

from tqdm import tqdm


Pooling = Literal["mean", "max", "cls"]


class DNABERT2Model(BaseModel):
    """
    DNABERT-2 适配器
    - 默认: 提取隐藏态 + 池化 得到序列向量 (AutoModel)
    - 可选: use_mlm_head=True 时使用 AutoModelForMaskedLM 做 PLL 评分（逐位遮罩）

    Args:
        model_name: 逻辑名
        model_path: 本地权重目录
        hf_home:    HF 缓存目录(可选)
        device:     'cuda:0' / 'cpu' / None(自动)
        use_mlm_head: 是否加载 MLM 头用于 PLL 评分
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        hf_home: Optional[str] = None,
        device: Optional[str] = None,
        use_mlm_head: bool = False,
    ):
        super().__init__(model_name, model_path)
        self.hf_home = hf_home
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_mlm_head = use_mlm_head
        self._load_model()

    # ---------- 加载 ----------
    def _load_model(self):
        if self.hf_home:
            os.environ["HF_HOME"] = self.hf_home

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        if self.use_mlm_head:
            self.model = AutoModelForMaskedLM.from_pretrained(
                self.model_path, trust_remote_code=True
            )
        else:
            self.model = AutoModel.from_pretrained(
                self.model_path, trust_remote_code=True
            )

        self.model.to(self.device).eval()

        # 基本属性
        self.model_max_len = getattr(self.model.config, "max_position_embeddings", None) or \
                             getattr(self.tokenizer, "model_max_length", None)

        # MLM 相关（仅在 use_mlm_head=True 时尝试）
        self.mask_id = None
        if self.use_mlm_head:
            self.mask_id = self.tokenizer.mask_token_id
            if self.mask_id is None:
                # 某些 tokenizer 自定义了 mask token 名
                cand = self.tokenizer.mask_token or "[MASK]"
                mid = self.tokenizer.convert_tokens_to_ids(cand)
                if isinstance(mid, int) and mid >= 0:
                    self.mask_id = mid
                else:
                    raise ValueError("找不到 mask_token_id，无法进行 PLL 评分。")

        print(f"[DNABERT2Model] loaded on {self.device}, "
              f"mlm_head={self.use_mlm_head}, model_max_len={self.model_max_len}")

    # ---------- 嵌入提取 ----------
    @torch.no_grad()
    def embed_sequences(
        self,
        sequences: List[str],
        batch_size: int = 32,
        pooling: Pooling = "mean",
        exclude_special: bool = True,
        truncation: bool = True,
        return_numpy: bool = True,
    ):
        """
        返回每条序列的向量表示（隐藏态池化）。
        pooling: 'mean' | 'max' | 'cls'
        exclude_special: mean/max 时是否排除 [CLS]/[SEP]/PAD 等
        """
        embs: List[np.ndarray] = []
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=truncation,
            )
            input_ids = enc["input_ids"].to(self.device)           # (B, L)
            attn_mask = enc.get("attention_mask")
            if attn_mask is None:
                attn_mask = torch.ones_like(input_ids)
            else:
                attn_mask = attn_mask.to(self.device)

            out = self.model(input_ids, attention_mask=attn_mask)
            # AutoModel: last_hidden_state / 自定义 models 也通常有 .last_hidden_state
            hidden = out[0] if isinstance(out, tuple) else out.last_hidden_state  # (B, L, H)
            B, L, H = hidden.shape

            if pooling == "cls":
                pooled = hidden[:, 0, :]  # (B, H)
            elif pooling == "mean":
                pooled = []
                for i in range(B):
                    ids_i = input_ids[i]
                    mask_i = attn_mask[i].bool()
                    if exclude_special:
                        spec = self.tokenizer.get_special_tokens_mask(
                            ids_i.tolist(), already_has_special_tokens=True
                        )
                        spec_mask = torch.tensor(spec, device=self.device, dtype=torch.bool)
                        valid = mask_i & (~spec_mask)
                    else:
                        valid = mask_i
                    if valid.any():
                        v = hidden[i][valid].mean(dim=0)
                    else:
                        v = hidden[i].mean(dim=0)  # 兜底
                    pooled.append(v)
                pooled = torch.stack(pooled, dim=0)  # (B, H)
            elif pooling == "max":
                pooled = []
                for i in range(B):
                    ids_i = input_ids[i]
                    mask_i = attn_mask[i].bool()
                    if exclude_special:
                        spec = self.tokenizer.get_special_tokens_mask(
                            ids_i.tolist(), already_has_special_tokens=True
                        )
                        spec_mask = torch.tensor(spec, device=self.device, dtype=torch.bool)
                        valid = mask_i & (~spec_mask)
                    else:
                        valid = mask_i
                    if valid.any():
                        v = hidden[i][valid].max(dim=0)[0]
                    else:
                        v = hidden[i].max(dim=0)[0]
                    pooled.append(v)
                pooled = torch.stack(pooled, dim=0)  # (B, H)
            else:
                raise ValueError(f"Unknown pooling: {pooling}")

            if return_numpy:
                embs.extend(pooled.float().cpu().numpy())
            else:
                embs.extend([pooled[i] for i in range(pooled.shape[0])])

        return np.stack(embs, axis=0) if return_numpy else embs

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
    
    def _score_batch(self, sequences: List[str]) -> List[float]:
        """对单个批次评分"""
        # 1. 分词和预处理
        input_ids, seq_lengths = self._prepare_batch(sequences)
        
        # 2. 模型前向传播
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits
        
        # 3. 计算 log probabilities
        log_probs = self._compute_log_probabilities(logits, input_ids)
        
        # 4. 聚合得分
        scores = self._aggregate_scores(log_probs, seq_lengths)
        
        return scores
    
    def _prepare_batch(self, sequences: List[str]):
        """分词和批处理"""
        seq_lengths = [len(seq) for seq in sequences]
        
        # 使用 tokenizer 编码
        all_token_ids = []
        for seq in sequences:
            token_ids = self.tokenizer.encode(seq, add_special_tokens=False)
            all_token_ids.append(token_ids)
        
        # Padding 到相同长度
        max_length = max(len(ids) for ids in all_token_ids)
        padded_ids = []
        
        for token_ids in all_token_ids:
            if len(token_ids) < max_length:
                padded = token_ids + [self.tokenizer.pad_token_id] * (max_length - len(token_ids))
            else:
                padded = token_ids[:max_length]
            padded_ids.append(padded)
        
        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=self.model.device)
        return input_ids, seq_lengths
    
    def _compute_log_probabilities(self, logits: torch.Tensor, input_ids: torch.Tensor):
        """计算 log probabilities"""
        # Softmax + log
        log_probs = torch.log_softmax(logits, dim=-1)
        log_probs = log_probs[:, :-1, :]  # 去掉最后一个位置
        target_ids = input_ids[:, 1:]     # 去掉第一个位置
        
        # 获取对应 token 的 log prob
        token_log_probs = torch.gather(
            log_probs, 2, target_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        return token_log_probs
    
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
    @torch.no_grad()
    def get_embedding(
        self,
        sequences: Union[str, List[str]],
        batch_size: int = 64,
        pool: Pooling = "cls",                  # "cls" | "mean" | "max"
        layer_index: int = -1,                     # -1 为最后一层；其余为索引
        average_reverse_complement: bool = False,  # 与反向互补平均
        exclude_special: bool = True,              # mean/max 时排除 [CLS]/[SEP]/PAD
        truncation: bool = True,
        max_length: Optional[int] = None,          # 不给则用 tokenizer/model 上限
        return_numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        返回每条序列一个向量 (N, H)。
        - pool="cls"/"mean"/"max"
        - layer_index: 取指定隐藏层（需要模型支持 hidden_states）
        - average_reverse_complement: 前向与反向互补分别取向量后做平均
        """

        device = self.model.device
        tok = self.tokenizer
        if isinstance(sequences, str):
            seq_list = [sequences]
        else:
            seq_list = list(sequences)

        # 反向互补
        def _revcomp(seq: str) -> str:
            tbl = str.maketrans("ACGTRYMKBDHVNacgtrymkbdhvn",
                                "TGCAYRKMVHDBNtgcayrkmvhdbn")
            return seq.translate(tbl)[::-1]

        def _tokenize(batch: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
            enc = tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=truncation,
                max_length=max_length or self.model_max_len,
            )
            input_ids = enc["input_ids"].to(device)
            attn_mask = enc.get("attention_mask")
            attn_mask = (attn_mask.to(device) if attn_mask is not None
                         else torch.ones_like(input_ids, device=device))
            return input_ids, attn_mask

        def _forward_hidden(input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
            # 需要中间层则打开 hidden_states；否则直接拿 last_hidden_state
            if layer_index == -1:
                try:
                    out = self.model(input_ids=input_ids, attention_mask=attn_mask, return_dict=True)
                    hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                except TypeError:
                    # 某些实现不收 attention_mask
                    out = self.model(input_ids=input_ids, return_dict=True)
                    hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                return hidden
            else:
                try:
                    out = self.model(input_ids=input_ids, attention_mask=attn_mask,
                                     output_hidden_states=True, return_dict=True)
                except TypeError:
                    out = self.model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
                return out.hidden_states[layer_index]

        def _pool(hidden: torch.Tensor, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
            # hidden: [B, T, H]
            if pool == "cls":
                return hidden[:, 0, :]  # [B, H]

            # 构造有效位置（排除 PAD / 特殊符号）
            valid = attn_mask.bool()  # [B, T]
            if exclude_special:
                spec_masks = []
                for ids in input_ids:
                    spec = tok.get_special_tokens_mask(ids.tolist(), already_has_special_tokens=True)
                    spec_masks.append(torch.tensor(spec, device=attn_mask.device, dtype=torch.bool))
                spec_mask = torch.stack(spec_masks, dim=0)
                valid = valid & (~spec_mask)

            if pool == "mean":
                m = valid.unsqueeze(-1).to(hidden.dtype)
                summed = (hidden * m).sum(dim=1)                 # [B, H]
                denom = m.sum(dim=1).clamp_min(1.0)              # [B, 1]
                return summed / denom
            elif pool == "max":
                masked = hidden.masked_fill(~valid.unsqueeze(-1), float("-inf"))
                # 若整行全是 -inf（极端情况），fallback 到未掩码 max
                pooled = masked.max(dim=1).values
                inf_mask = torch.isinf(pooled).any(dim=1)
                if inf_mask.any():
                    pooled[inf_mask] = hidden[inf_mask].max(dim=1).values
                return pooled
            else:
                raise ValueError(f"Unknown pool='{pool}'")

        outputs: List[torch.Tensor] = []
        for st in tqdm(range(0, len(seq_list), batch_size), desc="Getting embedding"):
            chunk = seq_list[st: st + batch_size]

            ids_f, am_f = _tokenize(chunk)
            hid_f = _forward_hidden(ids_f, am_f)
            vec_f = _pool(hid_f, ids_f, am_f)     # [B, H]

            if average_reverse_complement:
                rc_chunk = [_revcomp(s) for s in chunk]
                ids_r, am_r = _tokenize(rc_chunk)
                hid_r = _forward_hidden(ids_r, am_r)
                vec_r = _pool(hid_r, ids_r, am_r)
                vec = 0.5 * (vec_f + vec_r)
            else:
                vec = vec_f

            outputs.append(vec.detach().to(torch.float32).cpu())

        out = torch.cat(outputs, dim=0) if outputs else torch.empty(0, 0)
        return out.numpy() if return_numpy else out


# # ---------- 自测 ----------
if __name__ == "__main__":
    MODEL_DIR = "../../model_weight/DNABERT-S"
    HF_HOME = "../../model"

    m = DNABERT2Model(
        model_name="DNABERT-S",
        model_path=MODEL_DIR,
        hf_home=HF_HOME,
        device=None,          # 自动选 GPU/CPU
        use_mlm_head=False,   # 先跑嵌入提取（与您测试文件一致）
    )

    dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
    # embs = m.embed_sequences([dna], pooling="mean", exclude_special=True, truncation=True)
    # print("Embedding shape:", embs.shape)   # (1, hidden_size)

    # 如果你也下载了带 MLM 头的版本，可改为 use_mlm_head=True，再跑 PLL 评分：
    m_mlm = DNABERT2Model("DNABERT-2-117M", MODEL_DIR, HF_HOME, use_mlm_head=True)
    pll = m_mlm.score_sequences([dna], batch_size=128)
    print("PLL score:", pll)

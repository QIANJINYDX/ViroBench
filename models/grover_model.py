# models/grover_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import List, Optional, Literal, Union, Tuple

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

from tqdm import tqdm

from .base_model import BaseModel


Pooling = Literal["mean", "max", "cls"]


class GROVERModel(BaseModel):
    """
    GROVER (Genome Rules Obtained Via Extracted Representations) 适配器

    - 默认: 使用 AutoModel 提取隐藏态并做池化, 得到序列向量
    - 可选: use_mlm_head=True 时使用 AutoModelForMaskedLM 做 PLL 评分（逐位遮罩）

    官方模型使用示例 (来自 Hugging Face model card):
        tokenizer = AutoTokenizer.from_pretrained("PoetschLab/GROVER")
        model = AutoModelForMaskedLM.from_pretrained("PoetschLab/GROVER")
    :contentReference[oaicite:1]{index=1}

    GROVER 使用 Byte-Pair Encoding (BPE) tokenizer, 你只需要传入原始 DNA 序列字符串即可，
    tokenizer 会负责子词切分（BPE 规则保存在 tokenizer.json/vocab.txt 中）。

    Args:
        model_name: 逻辑名
        model_path: 本地权重目录或 HuggingFace 模型 ID（如 "PoetschLab/GROVER"）
        hf_home:    HF 缓存目录(可选)
        device:     'cuda:0' / 'cpu' / None(自动)
        use_mlm_head: 是否加载 MLM 头用于 PLL 评分
        vocab_path: 可选 tokenizer 路径（目录或 HF 模型 ID，建议传目录而不是单个 vocab.txt）
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        hf_home: Optional[str] = None,
        device: Optional[str] = None,
        use_mlm_head: bool = False,
        vocab_path: Optional[str] = None,
    ):
        super().__init__(model_name, model_path)
        self.hf_home = hf_home
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_mlm_head = use_mlm_head
        # 注意：这里的 vocab_path 应该是 tokenizer 目录或 HF 模型 ID，而不是单个 vocab.txt 文件
        self.vocab_path = vocab_path

        self._load_model()

    # ---------- 加载 ----------
    def _load_model(self):
        if self.hf_home:
            os.environ["HF_HOME"] = self.hf_home

        # 优先使用 vocab_path 为 tokenizer 的来源，否则用 model_path，再不行用官方 ID
        tok_source = self.vocab_path or self.model_path or "PoetschLab/GROVER"

        # 官方推荐使用 AutoTokenizer.from_pretrained("PoetschLab/GROVER")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_source,
            trust_remote_code=True,
        )

        model_source = self.model_path or tok_source

        if self.use_mlm_head:
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_source,
                trust_remote_code=True,
            )
        else:
            # 对于仅提取 embedding 的情况, 使用 AutoModel 即可
            self.model = AutoModel.from_pretrained(
                model_source,
                trust_remote_code=True,
            )

        self.model.to(self.device).eval()

        # 最大长度
        self.model_max_len = (
            getattr(self.model.config, "max_position_embeddings", None)
            or getattr(self.tokenizer, "model_max_length", None)
            or 512
        )

        # MLM 相关（仅在 use_mlm_head=True 时尝试）
        self.mask_id: Optional[int] = None
        if self.use_mlm_head:
            self.mask_id = self.tokenizer.mask_token_id
            if self.mask_id is None:
                cand = self.tokenizer.mask_token or "[MASK]"
                mid = self.tokenizer.convert_tokens_to_ids(cand)
                if isinstance(mid, int) and mid >= 0:
                    self.mask_id = mid
                else:
                    raise ValueError("找不到 mask_token_id，无法进行 PLL 评分。")

        print(
            f"[GROVERModel] loaded on {self.device}, "
            f"mlm_head={self.use_mlm_head}, "
            f"model_max_len={self.model_max_len}"
        )

    # ---------- 嵌入提取（简易封装） ----------
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
        return self.get_embedding(
            sequences=sequences,
            layer_name=None,
            batch_size=batch_size,
            pool=pooling,
            layer_index=-1,
            average_reverse_complement=False,
            exclude_special=exclude_special,
            truncation=truncation,
            max_length=None,
            return_numpy=return_numpy,
        )

    # ---------- PLL 评分 ----------
    def score_sequences(self, sequences: List[str], batch_size: int = 256) -> List[float]:
        """
        对序列进行 PLL (Pseudo-Log-Likelihood) 评分
        使用 MLM 方式：逐位遮罩，计算每个位置真实 token 的 log-probability

        Args:
            sequences: 原始 DNA 序列字符串列表（或任意字符串，交给 tokenizer 处理）
            batch_size: 每一次前向中同时遮罩的 token 数（注意：不是样本 batch 大小）

        Returns:
            每个序列的平均 log-likelihood（按 token 平均）
        """
        if not self.use_mlm_head:
            raise RuntimeError(
                "PLL scoring requires use_mlm_head=True. "
                "Please initialize GROVERModel with use_mlm_head=True."
            )

        if self.mask_id is None:
            raise RuntimeError("Mask token ID not found. Cannot perform PLL scoring.")

        all_scores: List[float] = []
        mask_batch_size = batch_size

        with torch.no_grad():
            for seq in tqdm(sequences, desc="Scoring sequences"):
                score = self._score_single_sequence(seq, mask_batch_size)
                all_scores.append(score)

        return all_scores

    def _score_single_sequence(self, sequence: str, mask_batch_size: int = 256) -> float:
        """
        对单条序列进行 PLL 评分

        Args:
            sequence: 输入序列（任意字符串，通常为原始 DNA 序列）
            mask_batch_size: 每次前向中同时遮罩的位置数量

        Returns:
            序列的平均 log-likelihood（按 token 平均）
        """
        # 1. 直接用 GROVER 的 BPE tokenizer 编码
        enc = self.tokenizer(
            sequence.strip(),
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.model_max_len,
            add_special_tokens=True,
        )

        input_ids = enc["input_ids"].to(self.device)  # (1, L)
        attn_mask = enc.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device)
        else:
            attn_mask = torch.ones_like(input_ids)

        # BERT 风格的 token_type_ids
        token_type_ids = torch.zeros_like(input_ids)

        seq_len = input_ids.shape[1]
        input_ids = input_ids.squeeze(0)      # (L,)
        attn_mask = attn_mask.squeeze(0)      # (L,)
        token_type_ids = token_type_ids.squeeze(0)

        # 2. 找出有效位置（排除特殊 token 和 padding）
        valid_positions: List[int] = []
        input_ids_list = input_ids.tolist()

        pad_id = self.tokenizer.pad_token_id
        cls_id = getattr(self.tokenizer, "cls_token_id", None)
        sep_id = getattr(self.tokenizer, "sep_token_id", None)

        for i in range(seq_len):
            if attn_mask[i].item() == 0:
                continue

            token_id = input_ids_list[i]

            is_special = False
            if pad_id is not None and token_id == pad_id:
                is_special = True
            elif cls_id is not None and token_id == cls_id:
                is_special = True
            elif sep_id is not None and token_id == sep_id:
                is_special = True

            if not is_special:
                valid_positions.append(i)

        if len(valid_positions) == 0:
            return 0.0

        # 3. 分块遮罩并计算 log-probability
        total_logprob = 0.0
        total_count = 0

        for start_idx in range(0, len(valid_positions), mask_batch_size):
            chunk_positions = valid_positions[start_idx : start_idx + mask_batch_size]
            chunk_size = len(chunk_positions)

            # 复制 input_ids 用于遮罩
            masked_ids = input_ids.unsqueeze(0).repeat(chunk_size, 1)          # (B, L)
            masked_attn = attn_mask.unsqueeze(0).repeat(chunk_size, 1)         # (B, L)
            masked_token_type = token_type_ids.unsqueeze(0).repeat(chunk_size, 1)  # (B, L)
            true_tokens: List[int] = []

            # 遮罩对应位置
            for b, pos in enumerate(chunk_positions):
                true_token = int(input_ids[pos].item())
                true_tokens.append(true_token)
                masked_ids[b, pos] = self.mask_id

            inputs = {
                "input_ids": masked_ids.to(self.device),
                "attention_mask": masked_attn.to(self.device),
                "token_type_ids": masked_token_type.to(self.device),
            }

            outputs = self.model(**inputs)

            # 提取 logits
            if hasattr(outputs, "logits"):
                logits = outputs.logits  # (B, L, V)
            elif isinstance(outputs, tuple):
                if len(outputs) >= 2:
                    logits = outputs[1]
                else:
                    logits = outputs[0]
            else:
                raise ValueError("Cannot extract logits from model outputs")

            # 获取遮罩位置的 logits
            batch_indices = torch.arange(chunk_size, device=self.device)
            pos_indices = torch.tensor(chunk_positions, device=self.device, dtype=torch.long)
            pos_logits = logits[batch_indices, pos_indices, :]  # (B, V)

            # 计算 log-probability
            log_probs = torch.log_softmax(pos_logits, dim=-1)  # (B, V)

            true_token_ids = torch.tensor(true_tokens, device=self.device, dtype=torch.long)
            token_log_probs = log_probs[batch_indices, true_token_ids]  # (B,)

            total_logprob += token_log_probs.sum().item()
            total_count += chunk_size

            # 清理显存
            del outputs, logits, pos_logits, log_probs, token_log_probs, true_token_ids
            if "cuda" in str(self.device):
                torch.cuda.empty_cache()

        avg_logprob = total_logprob / max(1, total_count)
        return float(avg_logprob)

    # ---------- 通用 embedding 接口 ----------
    @torch.no_grad()
    def get_embedding(
        self,
        sequences: Union[str, List[str]],
        layer_name: str = None,                # 为了兼容 BaseModel 接口，但实际使用 layer_index
        batch_size: int = 64,
        pool: Pooling = "cls",                 # "cls" | "mean" | "max"
        layer_index: int = -1,                 # -1 为最后一层；其余为索引
        average_reverse_complement: bool = False,  # 与反向互补平均
        exclude_special: bool = True,          # mean/max 时排除 [CLS]/[SEP]/PAD
        truncation: bool = True,
        max_length: Optional[int] = None,      # 不给则用 tokenizer/model 上限
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

        # 反向互补（如需）
        def _revcomp(seq: str) -> str:
            tbl = str.maketrans(
                "ACGTRYMKBDHVNacgtrymkbdhvn",
                "TGCAYRKMVHDBNtgcayrkmvhdbn",
            )
            return seq.translate(tbl)[::-1]

        def _tokenize(batch: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
            enc = tok(
                [s.strip() for s in batch],
                return_tensors="pt",
                padding=True,
                truncation=truncation,
                max_length=max_length or self.model_max_len or 512,
            )
            input_ids = enc["input_ids"].to(device)
            attn_mask = enc.get("attention_mask")
            attn_mask = (
                attn_mask.to(device)
                if attn_mask is not None
                else torch.ones_like(input_ids, device=device)
            )
            return input_ids, attn_mask

        def _forward_hidden(input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
            # 需要中间层则打开 hidden_states；否则直接拿 last_hidden_state
            if layer_index == -1:
                try:
                    out = self.model(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        return_dict=True,
                    )
                except TypeError:
                    out = self.model(input_ids=input_ids, return_dict=True)
                hidden = getattr(out, "last_hidden_state", out[0])
                return hidden
            else:
                try:
                    out = self.model(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                except TypeError:
                    out = self.model(
                        input_ids=input_ids,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                hidden_states = out.hidden_states
                return hidden_states[layer_index]

        def _pool(hidden: torch.Tensor, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
            # hidden: [B, T, H]
            if pool == "cls":
                return hidden[:, 0, :]  # [B, H]

            # 构造有效位置（排除 PAD / 特殊符号）
            valid = attn_mask.bool()  # [B, T]
            if exclude_special:
                spec_masks = []
                for ids in input_ids:
                    spec = tok.get_special_tokens_mask(
                        ids.tolist(),
                        already_has_special_tokens=True,
                    )
                    spec_masks.append(
                        torch.tensor(
                            spec,
                            device=attn_mask.device,
                            dtype=torch.bool,
                        )
                    )
                spec_mask = torch.stack(spec_masks, dim=0)
                valid = valid & (~spec_mask)

            if pool == "mean":
                m = valid.unsqueeze(-1).to(hidden.dtype)       # [B, T, 1]
                summed = (hidden * m).sum(dim=1)              # [B, H]
                denom = m.sum(dim=1).clamp_min(1.0)           # [B, 1]
                return summed / denom
            elif pool == "max":
                masked = hidden.masked_fill(
                    ~valid.unsqueeze(-1),
                    float("-inf"),
                )
                pooled = masked.max(dim=1).values             # [B, H]
                # 若整行全是 -inf（极端情况），fallback 到未掩码 max
                inf_mask = torch.isinf(pooled).any(dim=1)
                if inf_mask.any():
                    pooled[inf_mask] = hidden[inf_mask].max(dim=1).values
                return pooled
            else:
                raise ValueError(f"Unknown pool='{pool}'")

        outputs: List[torch.Tensor] = []
        for st in tqdm(range(0, len(seq_list), batch_size), desc="Getting embedding"):
            chunk = seq_list[st : st + batch_size]

            ids_f, am_f = _tokenize(chunk)
            hid_f = _forward_hidden(ids_f, am_f)
            vec_f = _pool(hid_f, ids_f, am_f)  # [B, H]

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
    # 根据你本地模型位置修改
    MODEL_DIR = "../../model_weight/GROVER"
    HF_HOME = "../../model"

    m = GROVERModel(
        model_name="GROVER",
        model_path=MODEL_DIR,   # 或直接 "PoetschLab/GROVER"
        hf_home=HF_HOME,
        device=None,            # 自动选 GPU/CPU
        use_mlm_head=False,     # 先跑嵌入提取
        vocab_path=None,        # 通常不需要单独指定
    )

    dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATTCGTTAGC"
    dna_list = [
        dna,
        "AGCGTACGTTAG",
        "AGTTTCCCGGAA",
    ]

    embs = m.embed_sequences(
        dna_list,
        pooling="cls",
        exclude_special=True,
        truncation=True,
    )
    print("Embedding shape:", embs.shape)   # (3, hidden_size)

    # 带 MLM 头版本，用于 PLL 评分
    m_mlm = GROVERModel(
        model_name="GROVER",
        model_path=MODEL_DIR,
        hf_home=HF_HOME,
        device=None,
        use_mlm_head=True,
        vocab_path=None,
    )
    pll = m_mlm.score_sequences(dna_list, batch_size=128)
    print("PLL score:", pll)

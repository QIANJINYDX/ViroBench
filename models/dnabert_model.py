# models/dnabert_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from typing import List, Optional, Literal, Union, Tuple

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

from tqdm import tqdm

# 假定 BaseModel 在同一包中
from .base_model import BaseModel


Pooling = Literal["mean", "max", "cls"]


class DNABERTModel(BaseModel):
    """
    DNABERT 适配器

    - 默认：加载模型时优先尝试 AutoModelForMaskedLM（带 MLM 头，可以做 PLL 评分），
            如果失败则退回到 AutoModel（仅做 embedding）。
    - 嵌入提取：无论是否有 MLM 头，都使用隐藏态做池化，得到序列级向量
    - PLL 评分：只有在成功加载 MLM 头、且存在有效 mask_token 时才可用

    说明：
        DNABERT-3/4/5/6 等第一代模型使用 k-mer token（k=3/4/5/6），
        通常需要先把原始 ATCG 序列转换成以空格分隔的 k-mer 文本，
        例如：ACGTACGT -> "ACGTAC CGTACG GTACGT"（k=6）。
        本类通过 auto_kmer=True 自动完成这一步。

    Args:
        model_name: 逻辑名（比如 "DNABERT-6"）
        model_path: 本地权重目录或 HuggingFace 模型名（如 "zhihan1996/DNA_bert_6"）
        hf_home:    HF 缓存目录(可选)，会写入环境变量 HF_HOME
        device:     'cuda:0' / 'cpu' / None(自动选择)
        kmer_size:  DNABERT 的 k（3/4/5/6），默认 6（适配 DNA_bert_6）
        auto_kmer:  是否自动把原始 ATCG 序列转换为 k-mer 文本
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        hf_home: Optional[str] = None,
        device: Optional[str] = None,
        kmer_size: Optional[int] = 6,
        auto_kmer: bool = True,
    ):
        super().__init__(model_name, model_path)
        self.hf_home = hf_home
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.kmer_size = kmer_size
        self.auto_kmer = auto_kmer

        # 自动检测是否有 MLM 头
        self.has_mlm_head: bool = False
        self.mask_id: Optional[int] = None

        # 如果没显式指定 k，尝试从名字里猜一个
        if self.kmer_size is None:
            for k in (3, 4, 5, 6):
                if str(k) in (model_name or "") or str(k) in (model_path or ""):
                    self.kmer_size = k
                    break

        self._load_model()

    # ---------- 加载 ----------
    def _load_model(self):
        if self.hf_home:
            os.environ["HF_HOME"] = self.hf_home

        # DNABERT tokenizer: 需要 trust_remote_code=True
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        # 优先尝试加载带 MLM 头的模型，失败再退回 AutoModel
        try:
            model = AutoModelForMaskedLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
            self.has_mlm_head = True
        except Exception as e:
            print(f"[DNABERTModel] AutoModelForMaskedLM load failed ({e}); "
                  f"falling back to AutoModel (no PLL scoring).")
            model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
            self.has_mlm_head = False

        self.model = model.to(self.device).eval()

        # 最大长度
        self.model_max_len = (
            getattr(self.model.config, "max_position_embeddings", None)
            or getattr(self.tokenizer, "model_max_length", None)
            or 512
        )

        # MLM 相关：只有在 has_mlm_head=True 时才尝试
        self.mask_id = None
        if self.has_mlm_head:
            self.mask_id = self.tokenizer.mask_token_id
            if self.mask_id is None:
                cand = self.tokenizer.mask_token or "[MASK]"
                mid = self.tokenizer.convert_tokens_to_ids(cand)
                if isinstance(mid, int) and mid >= 0:
                    self.mask_id = mid
                else:
                    # 找不到 mask token，则不能做 PLL
                    print(
                        "[DNABERTModel] Warning: cannot find mask_token_id; "
                        "PLL scoring will be disabled."
                    )
                    self.has_mlm_head = False

        print(
            f"[DNABERTModel] loaded on {self.device}, "
            f"has_mlm_head={self.has_mlm_head}, "
            f"k={self.kmer_size}, "
            f"model_max_len={self.model_max_len}"
        )

    # ---------- k-mer 预处理 ----------
    def _preprocess_sequence(self, seq: str) -> str:
        """
        将原始 ATCG 序列转成 k-mer 文本（如 'ACGT...' -> 'ACGTAC CGTACG ...'）。

        - 如果 auto_kmer=False 或 kmer_size=None，则只做大小写/碱基清洗。
        - 如果字符串中已经包含空格，认为已经是 k-mer 文本，直接返回。
        """
        s = seq.strip().upper().replace("U", "T")
        if not self.auto_kmer or self.kmer_size is None:
            return s

        # 已经是由空格分隔的 k-mer 文本，直接用
        if " " in s:
            return s

        k = int(self.kmer_size)
        if len(s) < k:
            # 太短，凑不出一个完整 k-mer，就直接用原序列
            return s

        kmers = [s[i: i + k] for i in range(len(s) - k + 1)]
        return " ".join(kmers)

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
    def score_sequences(self, sequences: List[str], batch_size: int = 1, mask_batch_size: int = 256) -> List[float]:
        """
        对序列进行 PLL (Pseudo-Log-Likelihood) 评分
        使用 MLM 方式：逐位遮罩，计算每个位置真实 token 的 log-probability

        Args:
            sequences: 原始 DNA 序列列表（ATCG），或已预处理的 k-mer 文本
            batch_size: 样本批处理大小（同时处理多少条序列，但在内部对于 PLL 仍是逐条计算）
            mask_batch_size: 每条序列每一次前向中同时遮罩的 token 数

        Returns:
            每个序列的平均 log-likelihood 得分（按 token 平均）
        """
        if not self.has_mlm_head or self.mask_id is None:
            raise RuntimeError(
                "Current DNABERTModel does not have a valid MLM head or MASK token. "
                "PLL scoring is unavailable. Please ensure you are using an "
                "MLM-style DNABERT checkpoint (e.g., zhihan1996/DNA_bert_6)."
            )

        all_scores: List[float] = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc="Scoring sequences"):
                batch_seqs = sequences[i : i + batch_size]
                # 统一做 k-mer 预处理
                batch_seqs = [self._preprocess_sequence(s) for s in batch_seqs]
                batch_scores = self._score_batch(batch_seqs, mask_batch_size)
                all_scores.extend(batch_scores)

        return all_scores

    def _score_batch(self, sequences: List[str], mask_batch_size: int) -> List[float]:
        scores = []
        for seq in sequences:
            scores.append(self._score_single_sequence(seq, mask_batch_size))
        return scores

    def _score_single_sequence(self, sequence: str, mask_batch_size: int = 256) -> float:
        """
        对单条序列进行 PLL 评分

        Args:
            sequence: 输入序列（已预处理的 k-mer 文本）
            mask_batch_size: 每次前向中同时遮罩的位置数量

        Returns:
            序列的平均 log-likelihood（按 token 平均）
        """
        # 1. k-mer 预处理 + 分词
        # proc_seq = self._preprocess_sequence(sequence)
        proc_seq = sequence

        try:
            enc = self.tokenizer(
                proc_seq,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.model_max_len,
                add_special_tokens=True,
            )
        except Exception:
            token_ids = self.tokenizer.encode(
                proc_seq,
                add_special_tokens=True,
                max_length=self.model_max_len,
                truncation=True,
            )
            enc = {
                "input_ids": torch.tensor([token_ids], dtype=torch.long),
                "attention_mask": torch.ones(1, len(token_ids), dtype=torch.long),
            }

        input_ids = enc["input_ids"].to(self.device)  # (1, L)
        attn_mask = enc.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device)
        else:
            attn_mask = torch.ones_like(input_ids)

        # DNABERT 基于 BERT，使用 token_type_ids
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
            # 跳过 padding
            if attn_mask[i].item() == 0:
                continue

            token_id = input_ids_list[i]

            # 跳过特殊 token
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
            chunk_positions = valid_positions[start_idx: start_idx + mask_batch_size]
            chunk_size = len(chunk_positions)

            # 复制 input_ids 用于遮罩
            masked_ids = input_ids.unsqueeze(0).repeat(chunk_size, 1)      # (B, L)
            masked_attn = attn_mask.unsqueeze(0).repeat(chunk_size, 1)     # (B, L)
            masked_token_type = token_type_ids.unsqueeze(0).repeat(chunk_size, 1)  # (B, L)
            true_tokens: List[int] = []

            # 遮罩对应位置
            for b, pos in enumerate(chunk_positions):
                true_token = int(input_ids[pos].item())
                true_tokens.append(true_token)
                masked_ids[b, pos] = self.mask_id

            # 构建模型输入
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
            # del outputs, logits, pos_logits, log_probs, token_log_probs, true_token_ids
            # if "cuda" in str(self.device):
            #     torch.cuda.empty_cache()

        # 4. 计算平均 log-probability
        avg_logprob = total_logprob / max(1, total_count)
        return float(avg_logprob)

    # ---------- 通用 embedding 接口 ----------
    @torch.no_grad()
    def get_embedding(
        self,
        sequences: Union[str, List[str]],
        layer_name: str = None,                # 为了兼容 BaseModel 接口，但实际使用 layer_index
        batch_size: int = 64,
        pool: Pooling = "mean",                # "cls" | "mean" | "max"
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

        # 反向互补
        def _revcomp(seq: str) -> str:
            tbl = str.maketrans(
                "ACGTRYMKBDHVNacgtrymkbdhvn",
                "TGCAYRKMVHDBNtgcayrkmvhdbn",
            )
            return seq.translate(tbl)[::-1]

        def _tokenize(batch: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
            # 先做 DNABERT 所需的 k-mer 预处理
            proc = [self._preprocess_sequence(s) for s in batch]

            enc = tok(
                proc,
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
            chunk = seq_list[st: st + batch_size]

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
    import time
    # 根据你本地模型位置修改
    MODEL_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/DNA_bert_3"
    HF_HOME = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/model"

    m = DNABERTModel(
        model_name="DNABERT-3",
        model_path=MODEL_DIR,
        hf_home=HF_HOME,
        device=None,          # 自动选 GPU/CPU
        kmer_size=6,
        auto_kmer=True,
    )

    dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATTCGTTAGC"
    dna_list = [
        dna,
        "AGCGTACGTTAG",
        "AGTTTCCCGGAA",
    ]

    start_time = time.time()
    embs = m.embed_sequences(
        dna_list,
        pooling="mean",
        exclude_special=True,
        truncation=True,
    )
    end_time = time.time()
    print(f"Embedding done in {end_time - start_time:.4f}s")
    print("Embedding shape:", embs.shape)   # (3, hidden_size)

    # 如果模型成功带有 MLM 头，则可以直接做 PLL 评分；否则会抛出 RuntimeError
    try:
        start_time = time.time()
        pll = m.score_sequences(dna_list, batch_size=128)
        end_time = time.time()
        print(f"PLL scoring done in {end_time - start_time:.4f}s")
        print("PLL score:", pll)
    except RuntimeError as e:
        print("PLL scoring not available:", e)

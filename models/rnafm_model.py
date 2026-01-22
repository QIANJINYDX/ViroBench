# models/rnafm_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import sys
from typing import List, Optional, Literal, Union, Tuple

import torch
import numpy as np
from tqdm import tqdm

# 你的基类
from .base_model import BaseModel


Pooling = Literal["mean", "max", "cls"]


class RNAFMModel(BaseModel):
    """
    RNA-FM 适配器（基于 multimolecule 实现）

    - 输入：DNA 序列字符串（A/C/G/T/N/…），内部自动转为 RNA（T -> U）
    - Backbone：RnaFmForMaskedLM（带 MLM 头），既能抽取 embedding，也能做 PLL 评分
    - 接口：尽量对齐 DNABERTModel，包括：
        - get_embedding(...) / embed_sequences(...)
        - score_sequences(...)

    Args:
        model_name: 逻辑名（比如 "RNA-FM"）
        model_path: 本地权重目录或 HF 名称（如 "multimolecule/rnafm"）
        hf_home:    HF 缓存目录(可选)，会写入环境变量 HF_HOME
        device:     'cuda:0' / 'cpu' / None(自动选择)
        replace_T_with_U: 是否在 tokenizer 里启用 T->U 替换；
                          我们自己已经做了一次 T->U，这里默认 False，避免重复处理
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        hf_home: Optional[str] = None,
        device: Optional[str] = None,
        replace_T_with_U: bool = False,
    ):
        super().__init__(model_name, model_path)
        self.hf_home = hf_home
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.replace_T_with_U = replace_T_with_U

        self._load_model()

    # ---------- 加载 ----------
    def _load_model(self):
        if self.hf_home:
            os.environ["HF_HOME"] = self.hf_home

        rna_tokenizer_cls, rna_fm_cls = self._import_multimolecule()

        # tokenizer：这里可以控制是否在内部做 T->U
        self.tokenizer = rna_tokenizer_cls.from_pretrained(
            self.model_path,
            replace_T_with_U=self.replace_T_with_U,
        )

        # 使用带 MLM 头的模型：同时支持 logits + hidden_states
        self.model = rna_fm_cls.from_pretrained(self.model_path)
        self.model.to(self.device).eval()

        self.model_max_len = getattr(self.model.config, "max_position_embeddings", 1024)

        # mask token id（用于 PLL）
        self.mask_id: Optional[int] = self.tokenizer.mask_token_id
        if self.mask_id is None:
            # multimolecule 的实现正常都会有 <mask>
            raise ValueError("RnaTokenizer 没有 mask_token_id，无法进行 PLL 评分。")

        print(
            f"[RNAFMModel] loaded on {self.device}, "
            f"model_max_len={self.model_max_len}"
        )

    def _import_multimolecule(self):
        """
        Import multimolecule while avoiding name collision with local `datasets` package.

        run_all.py prepends the project root to sys.path, which shadows the
        HuggingFace `datasets` module that multimolecule depends on. We temporarily
        remove the project root and purge a cached local `datasets` module.
        """
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_root_abs = os.path.abspath(project_root)
        removed_paths = [p for p in sys.path if os.path.abspath(p) == project_root_abs]
        if removed_paths:
            sys.path = [p for p in sys.path if os.path.abspath(p) != project_root_abs]
        removed_datasets = False
        datasets_mod = sys.modules.get("datasets")
        if datasets_mod is not None:
            mod_path = getattr(datasets_mod, "__file__", "")
            if mod_path and os.path.abspath(mod_path).startswith(project_root_abs):
                del sys.modules["datasets"]
                removed_datasets = True
        try:
            from multimolecule import RnaTokenizer, RnaFmForMaskedLM
        finally:
            if removed_paths:
                sys.path.insert(0, project_root)
            if removed_datasets:
                # Let future imports resolve normally; no re-insert.
                pass
        return RnaTokenizer, RnaFmForMaskedLM

    # ---------- DNA -> RNA 预处理 ----------
    def _preprocess_sequence(self, seq: str) -> str:
        """
        将 DNA 序列标准化为 RNA 序列：
        - 去掉首尾空白
        - 全部大写
        - T -> U
        其它字符（N、- 等）保留，由 tokenizer 处理。
        """
        s = seq.strip().upper()
        s = s.replace("T", "U")
        return s

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
    @torch.no_grad()
    def score_sequences(
        self,
        sequences: List[str],
        batch_size: int = 256,
    ) -> List[float]:
        """
        对序列进行 PLL (Pseudo-Log-Likelihood) 评分。

        使用 MLM 方式：对每条序列，逐位遮罩（按 batch 分块），
        计算真实 token 在被遮罩位置的 log-probability，并按 token 平均。

        Args:
            sequences: DNA 序列列表（ACGT...）
            batch_size: 每次前向中同时遮罩的位置数量（不是样本 batch）

        Returns:
            每个序列的平均 log-likelihood 得分（按 token 平均，log base e）
        """
        if self.mask_id is None:
            raise RuntimeError("mask_token_id 缺失，无法进行 PLL 评分。")

        all_scores: List[float] = []

        for seq in tqdm(sequences, desc="Scoring sequences with RNA-FM"):
            score = self._score_single_sequence(seq, batch_size)
            all_scores.append(score)

        return all_scores

    def _score_single_sequence(
        self,
        sequence: str,
        mask_batch_size: int = 256,
    ) -> float:
        """
        对单条 DNA 序列进行 PLL 评分。
        """
        # 1. 预处理：DNA -> RNA + 分词
        rna_seq = self._preprocess_sequence(sequence)

        enc = self.tokenizer(
            rna_seq,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.model_max_len,
        )

        input_ids = enc["input_ids"].to(self.device)  # (1, L)
        attn_mask = enc.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device)
        else:
            attn_mask = torch.ones_like(input_ids, device=self.device)

        # RNA-FM 不需要 token_type_ids，这里统一处理一下
        seq_len = input_ids.shape[1]
        input_ids = input_ids.squeeze(0)  # (L,)
        attn_mask = attn_mask.squeeze(0)  # (L,)

        # 2. 找出有效位置（attention_mask 为 1 且不是特殊 token）
        special_ids = set(self.tokenizer.all_special_ids or [])
        valid_positions: List[int] = []

        for i in range(seq_len):
            if attn_mask[i].item() == 0:
                continue
            tok_id = int(input_ids[i].item())
            if tok_id in special_ids:
                continue
            valid_positions.append(i)

        if len(valid_positions) == 0:
            return 0.0

        # 3. 分块遮罩并计算 log-probability
        total_logprob = 0.0
        total_count = 0

        for start_idx in range(0, len(valid_positions), mask_batch_size):
            chunk_positions = valid_positions[start_idx : start_idx + mask_batch_size]
            chunk_size = len(chunk_positions)

            # 复制一份 input_ids 作为 batch
            masked_ids = input_ids.unsqueeze(0).repeat(chunk_size, 1)   # (B, L)
            masked_attn = attn_mask.unsqueeze(0).repeat(chunk_size, 1)  # (B, L)

            true_tokens: List[int] = []

            # 在对应位置放 MASK
            for b, pos in enumerate(chunk_positions):
                true_tok = int(input_ids[pos].item())
                true_tokens.append(true_tok)
                masked_ids[b, pos] = self.mask_id

            outputs = self.model(
                input_ids=masked_ids,
                attention_mask=masked_attn,
                output_hidden_states=False,
                return_dict=True,
            )

            logits = outputs.logits  # (B, L, V)

            batch_indices = torch.arange(chunk_size, device=self.device)
            pos_indices = torch.tensor(chunk_positions, device=self.device, dtype=torch.long)

            pos_logits = logits[batch_indices, pos_indices, :]  # (B, V)
            log_probs = torch.log_softmax(pos_logits, dim=-1)   # (B, V)

            true_token_ids = torch.tensor(true_tokens, device=self.device, dtype=torch.long)
            token_log_probs = log_probs[batch_indices, true_token_ids]  # (B,)

            total_logprob += token_log_probs.sum().item()
            total_count += chunk_size

            # 显存清理
            del outputs, logits, pos_logits, log_probs, true_token_ids, token_log_probs
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
        pool: Pooling = "mean",                # "cls" | "mean" | "max"
        layer_index: int = -1,                 # -1 为最后一层；其余为索引
        average_reverse_complement: bool = False,  # 与反向互补平均（DNA 维度上做 RC）
        exclude_special: bool = True,          # mean/max 时排除特殊 token
        truncation: bool = True,
        max_length: Optional[int] = None,      # 不给则用模型上限
        return_numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        返回每条序列一个向量 (N, H)。
        - 输入 sequences 是 DNA 序列
        - pool="cls"/"mean"/"max"
        - layer_index: 取指定隐藏层（需要模型支持 hidden_states）
        - average_reverse_complement: 对 DNA 序列做 RC，再分别前向，最后向量平均
        """
        device = self.model.device
        tok = self.tokenizer

        if isinstance(sequences, str):
            seq_list = [sequences]
        else:
            seq_list = list(sequences)

        # 反向互补在 DNA 空间中做，再交给 _preprocess_sequence -> RNA
        def _revcomp(seq: str) -> str:
            tbl = str.maketrans(
                "ACGTRYMKBDHVNacgtrymkbdhvn",
                "TGCAYRKMVHDBNtgcayrkmvhdbn",
            )
            return seq.translate(tbl)[::-1]

        def _tokenize(batch: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
            # 先 DNA -> RNA
            proc = [self._preprocess_sequence(s) for s in batch]

            enc = tok(
                proc,
                return_tensors="pt",
                padding=True,
                truncation=truncation,
                max_length=max_length or self.model_max_len or 1024,
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
            # RnaFmForMaskedLM: 使用 hidden_states
            if layer_index == -1:
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    output_hidden_states=False,
                    return_dict=True,
                )
                # 没有 last_hidden_state 字段时，可以从 hidden_states[-1] 取
                if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                    hidden = out.last_hidden_state
                else:
                    # 回退：打开 hidden_states
                    out2 = self.model(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    hidden = out2.hidden_states[-1]
                return hidden
            else:
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = out.hidden_states
                return hidden_states[layer_index]

        def _pool(hidden: torch.Tensor, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
            # hidden: [B, T, H]
            if pool == "cls":
                # RNA-FM 的 tokenizer 通常会在开头有 BOS token，可以用作 cls
                return hidden[:, 0, :]  # [B, H]

            valid = attn_mask.bool()  # [B, T]
            if exclude_special:
                # 使用 tokenizer 的 special ids 掩掉 BOS/EOS/PAD/MASK 等
                special_ids = set(tok.all_special_ids or [])
                spec_masks = []
                for ids in input_ids:
                    # ids: [T]
                    spec = torch.zeros_like(ids, dtype=torch.bool)
                    for i, tid in enumerate(ids):
                        if int(tid.item()) in special_ids:
                            spec[i] = True
                    spec_masks.append(spec)
                spec_mask = torch.stack(spec_masks, dim=0)  # [B, T]
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
        for st in tqdm(range(0, len(seq_list), batch_size), desc="Getting RNA-FM embedding"):
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


# ---------- 自测 ----------
if __name__ == "__main__":
    # 根据你本地模型位置修改
    MODEL_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/rnafm"
    HF_HOME = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model"

    m = RNAFMModel(
        model_name="RNA-FM",
        model_path=MODEL_DIR,   # 也可以直接用 "multimolecule/rnafm"
        hf_home=HF_HOME,
        device=None,            # 自动选 GPU/CPU
    )

    dna_list = [
        "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATTCGTTAGC",
        "AGCGTACGTTAG",
        "AGTTTCCCGGAA",
    ]

    embs = m.embed_sequences(
        dna_list,
        pooling="mean",
        exclude_special=True,
        truncation=True,
    )
    print("Embedding shape:", embs.shape)   # (3, hidden_size)

    pll = m.score_sequences(dna_list, batch_size=128)
    print("PLL scores:", pll)

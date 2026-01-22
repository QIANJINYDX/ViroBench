# models/mprna_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from typing import List, Optional, Literal, Union, Tuple

import torch
import numpy as np
from tqdm import tqdm

# --- 修改 1: 使用通用的 HF Auto 类 ---
from transformers import AutoTokenizer, AutoModelForMaskedLM

# 你的基类
from .base_model import BaseModel

Pooling = Literal["mean", "max", "cls"]

class MPRNAModel(BaseModel):
    """
    MP-RNA (OmniGenome) 适配器
    
    - 兼容基于 ESM 结构或其他自定义结构的 RNA 模型
    - 自动识别 Tokenizer (通常是 EsmTokenizer)
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        hf_home: Optional[str] = None,
        device: Optional[str] = None,
        trust_remote_code: bool = True, # MP-RNA 可能需要信任远程代码
    ):
        super().__init__(model_name, model_path)
        self.hf_home = hf_home
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.trust_remote_code = trust_remote_code

        self._load_model()

    # ---------- 加载 ----------
    def _load_model(self):
        if self.hf_home:
            os.environ["HF_HOME"] = self.hf_home

        print(f"[{self.model_name}] Loading tokenizer from {self.model_path}...")
        
        # --- 修改 2: 使用 AutoTokenizer ---
        # 很多生物模型不像标准 NLP 模型那样处理 special tokens，所以尽量保持默认
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            do_lower_case=False, # DNA/RNA 通常区分大小写或全大写，MP-RNA 一般大写
        )

        print(f"[{self.model_name}] Loading model from {self.model_path}...")
        
        # --- 修改 3: 使用 AutoModelForMaskedLM ---
        # 这会自动映射到 OmniGenomeForMaskedLM 或 EsmForMaskedLM
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code
        )
        self.model.to(self.device).eval()

        # 获取最大长度，防止报错
        self.model_max_len = getattr(self.model.config, "max_position_embeddings", 1024)
        
        # 针对 ESM 结构的修正：ESM 有时候 max_pos 是 1026 (包含 cls/eos)，实际可用是 1024
        if self.model_max_len > 2048: # 有些模型这里是相对位置编码，值很大，取默认限制
            self.model_max_len = 1024

        # mask token id
        self.mask_id: Optional[int] = self.tokenizer.mask_token_id
        if self.mask_id is None:
            # 尝试手动获取标准 mask
            if hasattr(self.tokenizer, 'mask_token') and self.tokenizer.mask_token:
                self.mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
            else:
                print(f"[{self.model_name}] Warning: Could not find mask_token_id, PLL scoring usually requires it.")

        print(
            f"[{self.model_name}] Loaded on {self.device}. "
            f"Tokenizer: {self.tokenizer.__class__.__name__}, "
            f"Model: {self.model.__class__.__name__}, "
            f"Max Len: {self.model_max_len}"
        )

    # ---------- DNA -> RNA 预处理 ----------
    def _preprocess_sequence(self, seq: str) -> str:
        """
        MP-RNA 通常也接受 RNA 序列 (U)。
        如果是 ESM 结构的 tokenizer，通常词表中包含 A, C, G, U, T, N 等。
        为了统一，我们转为 RNA (T->U)。
        """
        s = seq.strip().upper()
        s = s.replace("T", "U")
        return s

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
        return self.get_embedding(
            sequences=sequences,
            batch_size=batch_size,
            pool=pooling,
            layer_index=-1,
            exclude_special=exclude_special,
            truncation=truncation,
            return_numpy=return_numpy,
        )

    # ---------- PLL 评分 (逻辑通用) ----------
    @torch.no_grad()
    def score_sequences(
        self,
        sequences: List[str],
        batch_size: int = 256,
    ) -> List[float]:
        if self.mask_id is None:
            raise RuntimeError("mask_token_id missing.")

        all_scores = []
        for seq in tqdm(sequences, desc=f"Scoring with {self.model_name}"):
            score = self._score_single_sequence(seq, batch_size)
            all_scores.append(score)
        return all_scores

    def _score_single_sequence(self, sequence: str, mask_batch_size: int) -> float:
        rna_seq = self._preprocess_sequence(sequence)
        
        # 很多 ESM tokenizer 需要空格分隔字符，但 AutoTokenizer 通常会自动处理
        # 如果发现 tokenizer 无法正确切分单个核苷酸，这里可能需要 list(rna_seq)
        
        enc = self.tokenizer(
            rna_seq,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.model_max_len,
            add_special_tokens=True 
        )

        input_ids = enc["input_ids"].to(self.device) # (1, L)
        attn_mask = enc.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device)
        else:
            attn_mask = torch.ones_like(input_ids, device=self.device)

        input_ids = input_ids.squeeze(0)
        attn_mask = attn_mask.squeeze(0)
        seq_len = input_ids.shape[0]

        # 找出有效位置 (非特殊token)
        special_ids = set(self.tokenizer.all_special_ids)
        valid_positions = []
        for i in range(seq_len):
            if attn_mask[i].item() == 0: continue
            if input_ids[i].item() in special_ids: continue
            valid_positions.append(i)

        if not valid_positions:
            return 0.0

        total_logprob = 0.0
        total_count = 0

        # 分块 MASK
        for start_idx in range(0, len(valid_positions), mask_batch_size):
            chunk = valid_positions[start_idx : start_idx + mask_batch_size]
            B = len(chunk)

            masked_ids = input_ids.unsqueeze(0).repeat(B, 1)
            masked_attn = attn_mask.unsqueeze(0).repeat(B, 1)

            true_tokens = []
            for b_idx, pos in enumerate(chunk):
                true_tokens.append(input_ids[pos].item())
                masked_ids[b_idx, pos] = self.mask_id

            outputs = self.model(
                input_ids=masked_ids,
                attention_mask=masked_attn,
                output_hidden_states=False,
                return_dict=True
            )
            logits = outputs.logits # (B, L, V)

            # 提取被 mask 位置的概率
            batch_idx = torch.arange(B, device=self.device)
            pos_idx = torch.tensor(chunk, device=self.device)
            
            target_logits = logits[batch_idx, pos_idx, :] # (B, V)
            log_probs = torch.log_softmax(target_logits, dim=-1)
            
            true_tok_ids = torch.tensor(true_tokens, device=self.device)
            token_vals = log_probs[batch_idx, true_tok_ids]

            total_logprob += token_vals.sum().item()
            total_count += B
            
            del outputs, logits
        
        return total_logprob / max(1, total_count)

    # ---------- 通用 Embedding ----------
    @torch.no_grad()
    def get_embedding(
        self,
        sequences: Union[str, List[str]],
        layer_name: str = None,
        batch_size: int = 64,
        pool: Pooling = "mean",
        layer_index: int = -1,
        average_reverse_complement: bool = False,
        exclude_special: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        
        if isinstance(sequences, str):
            sequences = [sequences]
        
        # 内部函数：Tokenizer -> Forward -> Pool
        def _process_batch(batch_strs):
            # 预处理
            proc_batch = [self._preprocess_sequence(s) for s in batch_strs]
            
            enc = self.tokenizer(
                proc_batch,
                return_tensors="pt",
                padding=True,
                truncation=truncation,
                max_length=max_length or self.model_max_len,
            )
            input_ids = enc["input_ids"].to(self.device)
            attn_mask = enc["attention_mask"].to(self.device)

            # 前向
            out = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            # 取 hidden states
            # 有些模型 hidden_states 是 tuple，有些在 output.hidden_states
            hs = out.hidden_states
            target_layer = hs[layer_index] # (B, L, H)

            # 池化
            if pool == "cls":
                # 假设 CLS 在 0 位，如果是 ESM 通常也是 0 (<s>)
                vec = target_layer[:, 0, :]
            else:
                # mean / max
                mask = attn_mask.bool()
                if exclude_special:
                    special = set(self.tokenizer.all_special_ids)
                    # 构建特殊字符 mask
                    spec_mask = torch.zeros_like(mask)
                    for r in range(input_ids.shape[0]):
                        for c in range(input_ids.shape[1]):
                            if input_ids[r, c].item() in special:
                                spec_mask[r, c] = True
                    mask = mask & (~spec_mask)
                
                if pool == "mean":
                    mask_f = mask.unsqueeze(-1).float()
                    summed = (target_layer * mask_f).sum(dim=1)
                    denom = mask_f.sum(dim=1).clamp_min(1.0)
                    vec = summed / denom
                elif pool == "max":
                    # fill invalid with -inf
                    fill_val = float("-inf")
                    tmp = target_layer.clone()
                    tmp[~mask] = fill_val
                    vec = tmp.max(dim=1).values
                    # 兜底
                    if torch.isinf(vec).any():
                        vec[torch.isinf(vec)] = 0.0
                else:
                    raise ValueError(f"Unknown pool type: {pool}")
            
            return vec.cpu()

        # 反向互补
        def _revcomp(seq):
            tbl = str.maketrans("ACGTRYMKBDHVN", "TGCAYRKMVHDBN")
            return seq.translate(tbl)[::-1]

        final_embs = []
        for i in tqdm(range(0, len(sequences), batch_size), desc="Embedding"):
            chunk = sequences[i : i + batch_size]
            
            vec_f = _process_batch(chunk)
            
            if average_reverse_complement:
                chunk_rc = [_revcomp(s) for s in chunk]
                vec_r = _process_batch(chunk_rc)
                vec = 0.5 * (vec_f + vec_r)
            else:
                vec = vec_f
            
            final_embs.append(vec)

        res = torch.cat(final_embs, dim=0) if final_embs else torch.empty(0)
        return res.numpy() if return_numpy else res


# ---------- 自测 ----------
if __name__ == "__main__":
    # 请根据实际情况修改路径
    MODEL_PATH = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/MP-RNA"
    
    # 假设你没有单独的 HF_HOME 需求，或者在外部设置好了
    # 实例化 MP-RNA 模型
    m = MPRNAModel(
        model_name="MP-RNA",
        model_path=MODEL_PATH,
        trust_remote_code=True # 关键：允许执行模型文件夹里的 Python 代码
    )

    test_seqs = [
        "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATTCGTTAGC",
        "AGCGTACGTTAG"
    ]

    # 测试 embedding
    embs = m.embed_sequences(test_seqs, pooling="mean")
    print(embs)
    print("Embedding shape:", embs.shape)

    # 测试 PLL
    scores = m.score_sequences(test_seqs, batch_size=32)
    print("PLL Scores:", scores)
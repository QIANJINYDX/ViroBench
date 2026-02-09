# models/ntv3_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import math
from typing import List, Optional, Literal, Union, Tuple

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
import json
from .base_model import BaseModel

Pooling = Literal["mean", "max", "cls"]

class NTV3Model(BaseModel):
    """
    Nucleotide Transformer v3 (NTv3) 适配器
    
    对应模型: InstaDeepAI/NTv3_8M_pre, InstaDeepAI/NTv3_100M_pre 等
    注意: NTv3 是 DNA 模型，输入序列应包含 T 而非 U。
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        hf_home: Optional[str] = None,
        device: Optional[str] = None,
        trust_remote_code: bool = True,
    ):
        super().__init__(model_name, model_path)
        self.hf_home = hf_home
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.trust_remote_code = trust_remote_code

        self._load_model()

    def _load_model(self):
        if self.hf_home:
            os.environ["HF_HOME"] = self.hf_home

        print(f"[{self.model_name}] Loading tokenizer from {self.model_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code,
            )
        except Exception as e:
            print(f"[{self.model_name}] AutoTokenizer failed, trying fallback to NucleotideTokenizer logic...")
            # NTv3 的 tokenizer 逻辑比较简单，如果自动加载失败，可以尝试手动指定
            # 但通常 trust_remote_code=True 应该能工作
            raise e

        print(f"[{self.model_name}] Loading model from {self.model_path}...")
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code
        )
        self.model.to(self.device).eval()

        # NTv3 架构特性：输入长度通常需要是对齐的 (比如 2^downsamples)
        # 获取下采样次数，默认为 8 (即 256 对齐)，小模型可能是 6 或 7
        self.num_downsamples = getattr(self.model.config, "num_downsamples", 8)
        self.align_factor = 2 ** self.num_downsamples
        
        # 最大长度：NTv3 支持很长，但显存有限，默认限制一下
        self.model_max_len = getattr(self.model.config, "max_position_embeddings", 2048)
        
        # 兼容性处理：有些 config 没写 max_pos
        if self.model_max_len > 10000: 
             # NTv3-multi-species 有些支持 12kb+，如果显存不够可以手动调小
            pass 

        # Mask Token
        self.mask_id = self.tokenizer.mask_token_id
        
        print(
            f"[{self.model_name}] Loaded on {self.device}. "
            f"Align Factor: {self.align_factor} (len must be multiple of this)"
        )

    # ---------- 预处理 (DNA) ----------
    def _preprocess_sequence(self, seq: str) -> str:
        """
        NTv3 是 DNA 模型，保持 T，转大写。
        """
        return seq.strip().upper() # 不做 T->U

    # ---------- 辅助：Padding 到对齐长度 ----------
    def _pad_to_alignment(self, batch_encoding):
        """
        确保 input_ids 长度是 align_factor 的倍数，否则 NTv3 会报错。
        """
        seq_len = batch_encoding["input_ids"].shape[1]
        remainder = seq_len % self.align_factor
        
        if remainder != 0:
            pad_len = self.align_factor - remainder
            # 获取 pad_token_id
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            
            # Pad input_ids
            input_ids = batch_encoding["input_ids"]
            pad_tensor = torch.full(
                (input_ids.shape[0], pad_len), 
                pad_id, 
                dtype=input_ids.dtype, 
                device=input_ids.device
            )
            batch_encoding["input_ids"] = torch.cat([input_ids, pad_tensor], dim=1)
            
            # Pad attention_mask
            if "attention_mask" in batch_encoding:
                mask = batch_encoding["attention_mask"]
                mask_pad = torch.zeros(
                    (mask.shape[0], pad_len), 
                    dtype=mask.dtype, 
                    device=mask.device
                )
                batch_encoding["attention_mask"] = torch.cat([mask, mask_pad], dim=1)
                
        return batch_encoding

    # ---------- Embedding ----------
    @torch.no_grad()
    def embed_sequences(
        self,
        sequences: List[str],
        batch_size: int = 16, # NTv3 显存占用较大，建议减小 batch
        pooling: Pooling = "mean",
        exclude_special: bool = True,
        truncation: bool = True,
        return_numpy: bool = True,
    ):
        return self.get_embedding(
            sequences=sequences,
            batch_size=batch_size,
            pool=pooling,
            exclude_special=exclude_special,
            truncation=truncation,
            return_numpy=return_numpy,
        )

    # ---------- PLL Scoring ----------
    @torch.no_grad()
    def score_sequences(
        self,
        sequences: List[str],
        batch_size: int = 256, # 这里指 mask batch size
    ) -> List[float]:
        if self.mask_id is None:
            raise RuntimeError("mask_token_id missing.")

        all_scores = []
        for seq in tqdm(sequences, desc=f"Scoring with {self.model_name}"):
            score = self._score_single_sequence(seq, batch_size)
            all_scores.append(score)
        return all_scores

    def _score_single_sequence(self, sequence: str, mask_batch_size: int) -> float:
        dna_seq = self._preprocess_sequence(sequence)
        
        # Tokenize
        enc = self.tokenizer(
            dna_seq,
            return_tensors="pt",
            truncation=True,
            max_length=self.model_max_len,
            padding=False 
        )
        
        original_len = enc["input_ids"].shape[1]
        
        # 移到 GPU
        input_ids = enc["input_ids"].to(self.device)
        attn_mask = enc.get("attention_mask")
        if attn_mask is None:
            attn_mask = torch.ones_like(input_ids)
        else:
            attn_mask = attn_mask.to(self.device)

        # 手动 Pad 以满足模型要求
        remainder = input_ids.shape[1] % self.align_factor
        if remainder != 0:
            pad_len = self.align_factor - remainder
            pad_id = self.tokenizer.pad_token_id or 0
            input_ids = torch.cat([input_ids, torch.full((1, pad_len), pad_id, device=self.device)], dim=1)
            attn_mask = torch.cat([attn_mask, torch.zeros((1, pad_len), device=self.device)], dim=1)

        input_ids = input_ids.squeeze(0) # (L_pad,)
        attn_mask = attn_mask.squeeze(0) # (L_pad,)
        
        # 寻找有效位置（只考虑原始长度内的非特殊token）
        # 特殊 token
        special_ids = set(self.tokenizer.all_special_ids)
        valid_positions = []
        
        # 只遍历到 original_len
        for i in range(original_len):
            if attn_mask[i].item() == 0: continue
            if input_ids[i].item() in special_ids: continue
            valid_positions.append(i)

        if not valid_positions:
            return 0.0

        total_logprob = 0.0
        total_count = 0

        # 分块 Mask
        for start_idx in range(0, len(valid_positions), mask_batch_size):
            chunk = valid_positions[start_idx : start_idx + mask_batch_size]
            B = len(chunk)
            L = input_ids.shape[0]

            masked_ids = input_ids.unsqueeze(0).repeat(B, 1) # (B, L_pad)
            masked_attn = attn_mask.unsqueeze(0).repeat(B, 1)

            true_tokens = []
            for k, pos in enumerate(chunk):
                true_tokens.append(input_ids[pos].item())
                masked_ids[k, pos] = self.mask_id

            # Forward
            outputs = self.model(
                input_ids=masked_ids,
                attention_mask=masked_attn,
                return_dict=True
            )
            logits = outputs.logits # (B, L_pad, V)

            # Gather results
            batch_idx = torch.arange(B, device=self.device)
            pos_idx = torch.tensor(chunk, device=self.device)
            
            target_logits = logits[batch_idx, pos_idx, :]
            log_probs = torch.log_softmax(target_logits, dim=-1)
            
            true_tok_ids = torch.tensor(true_tokens, device=self.device)
            token_val = log_probs[batch_idx, true_tok_ids]
            
            total_logprob += token_val.sum().item()
            total_count += B
            
        return total_logprob / max(1, total_count)

    @torch.no_grad()
    def get_embedding(
        self,
        sequences: Union[str, List[str]],
        batch_size: int = 16,
        pool: Pooling = "mean",
        layer_index: int = -1,
        exclude_special: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_numpy: bool = True,
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        
        if isinstance(sequences, str):
            sequences = [sequences]
            
        def _process_batch(batch_strs):
            proc = [self._preprocess_sequence(s) for s in batch_strs]
            
            enc = self.tokenizer(
                proc,
                return_tensors="pt",
                padding=True,
                truncation=truncation,
                max_length=max_length or self.model_max_len,
                pad_to_multiple_of=self.align_factor, 
                add_special_tokens=False,          
            )
            
            enc = {k: v.to(self.device) for k, v in enc.items()}
            
            input_ids = enc["input_ids"]
    
            if "attention_mask" in enc:
                attn_mask = enc["attention_mask"]
            else:
                pad_id = self.tokenizer.pad_token_id or 0
                attn_mask = (input_ids != pad_id).long()

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            hidden_states = outputs.hidden_states
            target_layer = hidden_states[layer_index] # (B, L_pad, H)

            mask = attn_mask.bool()
           
            if exclude_special:
                special_ids_list = self.tokenizer.all_special_ids
                special_ids_tensor = torch.tensor(special_ids_list, device=self.device)
                
                is_special = torch.isin(input_ids, special_ids_tensor)
                
                mask = mask & (~is_special)

            if pool == "cls":
              
                return target_layer[:, 0, :]
            
            elif pool == "mean":
                # (B, L, 1)
                m_f = mask.unsqueeze(-1).float()
                summed = (target_layer * m_f).sum(dim=1)
                denom = m_f.sum(dim=1).clamp_min(1.0)
                return summed / denom
            
            elif pool == "max":
                tmp = target_layer.clone()
                tmp[~mask] = float("-inf")
                vec = tmp.max(dim=1).values
                vec[torch.isinf(vec)] = 0.0
                return vec
            
            else:
                raise ValueError(f"Unknown pool {pool}")

        final_list = []
        for i in tqdm(range(0, len(sequences), batch_size), desc="Embedding"):
            chunk = sequences[i : i + batch_size]
            vec = _process_batch(chunk)
            final_list.append(vec.cpu())

        res = torch.cat(final_list, dim=0) if final_list else torch.empty(0)
        return res.numpy() if return_numpy else res


if __name__ == "__main__":
    # 测试
    # MODEL_DIR = "InstaDeepAI/NTv3_8M_pre" # 或者本地路径
    MODEL_DIR = "../../model_weight/NTv3_100M_pre"
    
    try:
        m = NTV3Model(
            model_name="NTv3_8M",
            model_path=MODEL_DIR,
            trust_remote_code=True
        )
        seqs = ["ACGTAGCTAGCTAGCTAG"]
        emb = m.embed_sequences(seqs)
        print("Embedding shape:", emb.shape)
        
        score = m.score_sequences(seqs)
        print("Score:", score)
    except Exception as e:
        print(f"Test failed: {e}")
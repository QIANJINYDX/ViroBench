# models/birnabert_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Optional, Literal, Union

try:
    from transformers import AutoModelForMaskedLM, AutoTokenizer, BertConfig
except ImportError:
    print("Error: transformers not installed. Please install via `pip install transformers`.")

from .base_model import BaseModel

Pooling = Literal["mean", "max", "cls"]

class BiRNABERTModel(BaseModel):
    """
    BiRNA-BERT 适配器 (最终稳定版)
    
    - 修复: 兼容 Base Model 返回 Tuple 的情况
    - 功能: 同时支持 Embedding 提取 (Identity Trick 等效) 和 PLL 评分
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        tokenizer_path: Optional[str] = None, 
        device: Optional[str] = None,
        max_len: int = 512,
    ):
        super().__init__(model_name, model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.tokenizer_path = tokenizer_path
        
        self._load_model()

    def _load_model(self):
        print(f"Loading {self.model_name}...")
        
        # 1. 加载 Tokenizer
        possible_paths = []
        if self.tokenizer_path: possible_paths.append(self.tokenizer_path)
        sub_tokenizer_dir = os.path.join(self.model_path, "TOKENIZER")
        if os.path.exists(sub_tokenizer_dir): possible_paths.append(sub_tokenizer_dir)
        possible_paths.append(self.model_path)
        possible_paths.append("buetnlpbio/birna-tokenizer")

        self.tokenizer = None
        for tp in possible_paths:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tp)
                print(f"[{self.model_name}] Success loading tokenizer from: {tp}")
                break
            except Exception:
                continue
        
        if self.tokenizer is None:
            raise ValueError(f"Could not load tokenizer from any of: {possible_paths}")

        # 2. 加载模型
        try:
            config = BertConfig.from_pretrained(self.model_path)
            # 强制开启 hidden_states
            config.output_hidden_states = True 
            
            self.model = AutoModelForMaskedLM.from_pretrained(
                self.model_path, 
                config=config,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Model loading error: {e}")
            raise

        self.model.to(self.device).eval()
        
        self.mask_token_id = self.tokenizer.mask_token_id
        if self.mask_token_id is None:
            self.mask_token_id = self.tokenizer.convert_tokens_to_ids("[MASK]")

        print(f"[{self.model_name}] loaded on {self.device}")

    def _preprocess(self, seq: str) -> str:
        return seq.strip().upper().replace("T", "U")

    # ---------- Embedding 接口 ----------
    @torch.no_grad()
    def get_embedding(
        self,
        sequences: Union[str, List[str]],
        batch_size: int = 32,
        pool: Pooling = "mean",
        return_numpy: bool = True,
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        
        if isinstance(sequences, str):
            sequences = [sequences]

        sequences = [self._preprocess(s) for s in sequences]
        results = []

        for i in tqdm(range(0, len(sequences), batch_size), desc="BiRNA-BERT Embedding"):
            batch_seqs = sequences[i : i + batch_size]
            
            encoded = self.tokenizer(
                batch_seqs,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            token_embeddings = None
            
            # 方法 A: 尝试通过 MaskedLM 获取 hidden_states
            # 注意: BiRNA-BERT 可能因为自定义代码导致 output_hidden_states 参数无效
            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                    token_embeddings = outputs.hidden_states[-1]
            except Exception:
                pass # 只要出错就走兜底
            
            # 方法 B: 兜底方案，直接调用 Base Model (Encoder)
            # 这里的修改解决了 'tuple' object has no attribute 'last_hidden_state'
            if token_embeddings is None:
                # 寻找内部的 bert 模型
                base_model = getattr(self.model, "bert", getattr(self.model, "base_model", None))
                
                if base_model is not None:
                    base_outputs = base_model(input_ids, attention_mask=attention_mask, return_dict=True)
                    
                    # === 关键修复开始 ===
                    if isinstance(base_outputs, tuple):
                        # 如果是 Tuple，第一个元素通常是 last_hidden_state
                        token_embeddings = base_outputs[0]
                    elif hasattr(base_outputs, "last_hidden_state"):
                        token_embeddings = base_outputs.last_hidden_state
                    else:
                        # 极少数情况，可能是 dict
                        try:
                            token_embeddings = base_outputs["last_hidden_state"]
                        except:
                            token_embeddings = base_outputs[0] # 最后的尝试
                    # === 关键修复结束 ===

            if token_embeddings is None:
                raise ValueError("Failed to extract embeddings: Could not retrieve hidden states from Model or Base Model.")

            # Pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

            if pool == "mean":
                sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                emb = sum_embeddings / sum_mask
            elif pool == "max":
                token_embeddings[mask_expanded == 0] = -1e9
                emb = torch.max(token_embeddings, 1)[0]
            elif pool == "cls":
                emb = token_embeddings[:, 0, :]
            else:
                raise ValueError(f"Unknown pool: {pool}")
            
            results.append(emb.cpu())

        final = torch.cat(results, dim=0)
        return final.numpy() if return_numpy else final

    def embed_sequences(self, *args, **kwargs):
        return self.get_embedding(*args, **kwargs)

    # ---------- PLL Scoring 接口 ----------
    @torch.no_grad()
    def score_sequences(
        self,
        sequences: List[str],
        batch_size: int = 256,
    ) -> List[float]:
        scores = []
        for seq in tqdm(sequences, desc="Scoring with BiRNA-BERT"):
            scores.append(self._score_single(seq, batch_size))
        return scores

    def _score_single(self, seq: str, mask_bs: int) -> float:
        seq = self._preprocess(seq)
        encoded = self.tokenizer(
            seq, 
            truncation=True, 
            max_length=self.max_len, 
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"][0].to(self.device)
        
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            input_ids.cpu().tolist(), already_has_special_tokens=True
        )
        valid_indices = [i for i, is_spec in enumerate(special_tokens_mask) if is_spec == 0]
        
        seq_len = len(valid_indices)
        if seq_len == 0: return 0.0

        total_logprob = 0.0
        
        for i in range(0, seq_len, mask_bs):
            chunk_indices = valid_indices[i : i + mask_bs]
            chunk_size = len(chunk_indices)

            batch_input = input_ids.unsqueeze(0).repeat(chunk_size, 1)
            targets = []
            for batch_idx, pos_idx in enumerate(chunk_indices):
                targets.append(batch_input[batch_idx, pos_idx].item())
                batch_input[batch_idx, pos_idx] = self.mask_token_id

            outputs = self.model(input_ids=batch_input, return_dict=True)
            
            # 兼容 Tuple 返回 (Logits 通常是第一个元素)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs.logits 

            pos_tensor = torch.tensor(chunk_indices, device=self.device).view(-1, 1, 1)
            pos_expanded = pos_tensor.expand(-1, 1, logits.size(-1))
            
            masked_logits = torch.gather(logits, 1, pos_expanded).squeeze(1)
            log_probs = torch.log_softmax(masked_logits, dim=-1)
            
            target_tensor = torch.tensor(targets, device=self.device).view(-1, 1)
            token_scores = torch.gather(log_probs, 1, target_tensor).squeeze(1)

            total_logprob += token_scores.sum().item()

        return total_logprob / seq_len


if __name__ == "__main__":
    MODEL_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/birna-bert"
    TOKENIZER_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/birna-tokenizer"

 
    m = BiRNABERTModel(
        "BiRNA-BERT", 
        model_path=MODEL_DIR, 
        tokenizer_path=TOKENIZER_DIR
    )

    test_seqs = [
        "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATTCGTTAGC", 
        "AGCGUACGUUAG" 
    ]

    embs = m.embed_sequences(test_seqs, pooling="mean")
    print(f"\n[Embedding Test] Shape: {embs.shape}")
    print(embs)

    scores = m.score_sequences(test_seqs, batch_size=10)
    print(f"\n[PLL Scoring Test] Scores: {scores}")
        
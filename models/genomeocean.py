# models/genomeocean_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Optional, Literal, Union

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
except ImportError:
    print("Error: transformers not installed. Please install via `pip install transformers`.")

from .base_model import BaseModel

Pooling = Literal["mean", "last_token"]

MODEL_TOKEN_LIMITS = {
    'DOEJGI/GenomeOcean-100M': 1024,
    'DOEJGI/GenomeOcean-500M': 1024,
    'DOEJGI/GenomeOcean-4B': 10240,
}

class GenomeOceanModel(BaseModel):
    """
    GenomeOcean 适配器 (修复评分功能版)
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
        
        if max_len is None:
            base_name = os.path.basename(model_path) if os.path.isdir(model_path) else model_path
            self.max_len = MODEL_TOKEN_LIMITS.get(model_path, MODEL_TOKEN_LIMITS.get(base_name, 10240))
        else:
            self.max_len = max_len

        self._load_model()

    def _load_model(self):
        print(f"Loading {self.model_name} from {self.model_path}...")
        
        # 1. 加载 Tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                model_max_length=self.max_len,
                padding_side="left", 
                use_fast=True,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            print(f"Tokenizer loading failed: {e}")
            raise

        # 2. 加载模型
        try:
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
            
            # 尝试加载为 CausalLM 以支持评分
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map=None 
            )
            self.model.config.use_cache = False
            self.can_score = True
            print(f"Loaded as CausalLM (Class: {type(self.model).__name__})")
            
        except Exception as e:
            print(f"Warning: Could not load as CausalLM ({e}), falling back to AutoModel.")
            try:
                self.model = AutoModel.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float32
                )
                self.model.config.use_cache = False
                self.can_score = False # 标记无法评分
                print(f"Loaded as Base AutoModel (Class: {type(self.model).__name__})")
            except Exception as e2:
                raise RuntimeError(f"Failed to load model: {e2}")

        self.model.to(self.device).eval()
        print(f"[{self.model_name}] loaded on {self.device} (Max Len: {self.max_len})")

    def _preprocess(self, seq: str) -> str:
        return seq.strip().upper()

    # ---------- Embedding 接口 ----------
    @torch.no_grad()
    def get_embedding(
        self,
        sequences: Union[str, List[str]],
        batch_size: int = 16,
        pool: Pooling = "mean", 
        layer_index: int = -1,
        return_numpy: bool = True,
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        
        if isinstance(sequences, str):
            sequences = [sequences]

        results = []
        for i in tqdm(range(0, len(sequences), batch_size), desc="GenomeOcean Embedding"):
            batch_seqs = sequences[i : i + batch_size]
            
            encoded = self.tokenizer(
                batch_seqs,
                max_length=self.max_len,
                return_tensors='pt',
                padding='longest',
                truncation=True
            )
            
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            hidden_states = outputs.hidden_states
            if layer_index < 0:
                layer_index = len(hidden_states) + layer_index
            
            token_embeddings = hidden_states[layer_index]

            if pool == "mean":
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(token_embeddings)
                sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                emb = sum_embeddings / sum_mask
                
            elif pool == "last_token":
                # Left Padding: 最后一个有效 token 在 -1
                if self.tokenizer.padding_side == "left":
                    emb = token_embeddings[:, -1, :]
                else:
                    sequence_lengths = attention_mask.sum(dim=1) - 1
                    emb = token_embeddings[torch.arange(token_embeddings.size(0), device=self.device), sequence_lengths]
            else:
                raise ValueError(f"Unknown pooling strategy: {pool}")
            
            results.append(emb.cpu())
            
            del input_ids, attention_mask, outputs, hidden_states, token_embeddings
            torch.cuda.empty_cache()

        final = torch.cat(results, dim=0)
        return final.numpy() if return_numpy else final

    def embed_sequences(self, *args, **kwargs):
        return self.get_embedding(*args, **kwargs)

    # ---------- PLL Scoring 接口 ----------
    @torch.no_grad()
    def score_sequences(
        self,
        sequences: List[str],
        batch_size: int = 16,
    ) -> List[float]:
        """
        计算序列的 Average Log-Likelihood (PLL)。
        """
        # [Fix] 移除 isinstance 检查，改为检查 flag 或方法
        if not getattr(self, "can_score", True): 
             print("Warning: Model marked as unable to score (loaded as AutoModel). Returning 0s.")
             return [0.0] * len(sequences)

        scores = []
        for i in tqdm(range(0, len(sequences), batch_size), desc="Scoring with GenomeOcean"):
            batch_seqs = sequences[i : i + batch_size]
            
            encoded = self.tokenizer(
                batch_seqs,
                max_length=self.max_len,
                return_tensors='pt',
                padding='longest',
                truncation=True
            )
            
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # Forward with labels to calculate loss (or logits)
            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids # 传入 labels 触发内部 loss 计算 (如果模型支持)
                )
                
                # 如果 outputs 有 logits，我们手动算 loss 以获取 per-sample score
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                    
                    # Shift logits and labels
                    # logits[i] 预测 input_ids[i+1]
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()
                    shift_mask = attention_mask[..., 1:].contiguous()
                    
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    token_losses = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)), 
                        shift_labels.view(-1)
                    )
                    token_losses = token_losses.view(shift_labels.size())
                    
                    # Mask padding
                    valid_nll = token_losses * shift_mask
                    seq_nll = valid_nll.sum(dim=1)
                    seq_lens = shift_mask.sum(dim=1)
                    
                    # Average Log Likelihood = -NLL / Length
                    avg_log_likelihood = -seq_nll / torch.clamp(seq_lens, min=1e-9)
                    scores.extend(avg_log_likelihood.cpu().tolist())
                    
                else:
                    print("Warning: Model output has no logits. Returning 0s.")
                    scores.extend([0.0] * len(batch_seqs))

            except Exception as e:
                print(f"Error during scoring: {e}")
                scores.extend([0.0] * len(batch_seqs))

        return scores


if __name__ == "__main__":
    MODEL_PATH = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/GenomeOcean-500M"
    
    print("=== Testing GenomeOcean Adapter (Fixed Scoring) ===")
    try:
        m = GenomeOceanModel(
            "GenomeOcean", 
            model_path=MODEL_PATH,
        )
        
        test_seqs = [
            "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATTCGTTAGC", 
            "AGCGUACGUUAG" 
        ]
        
        embs = m.embed_sequences(test_seqs, pooling="mean")
        print(f"\n[Embedding Shape]: {embs.shape}")
        
        scores = m.score_sequences(test_seqs, batch_size=2)
        print(f"\n[Likelihood Scores]: {scores}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nTest failed: {e}")
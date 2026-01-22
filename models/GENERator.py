# models/generator_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Optional, Literal, Union

# ==========================================
# 核心修复补丁: 解决 'DNAKmerTokenizer' object has no attribute '_added_tokens_decoder'
# ==========================================
try:
    import transformers
    from transformers import PreTrainedTokenizer
    
    # 保存原始方法
    if not hasattr(PreTrainedTokenizer, "_orig_convert_ids_to_tokens"):
        PreTrainedTokenizer._orig_convert_ids_to_tokens = PreTrainedTokenizer.convert_ids_to_tokens

    # 定义补丁方法
    def _patched_convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        # 如果缺少 _added_tokens_decoder 属性，则初始化一个空字典，防止报错
        if not hasattr(self, "_added_tokens_decoder"):
            self._added_tokens_decoder = {}
        # 调用原始方法
        return self._orig_convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)

    # 应用补丁
    PreTrainedTokenizer.convert_ids_to_tokens = _patched_convert_ids_to_tokens
    print("[GENERatorModel] Applied compatibility patch for DNAKmerTokenizer.")

except ImportError:
    pass
except Exception as e:
    print(f"[GENERatorModel] Warning: Failed to apply patch: {e}")

# ==========================================

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
except ImportError:
    print("Error: transformers not installed. Please install via `pip install transformers`.")

from .base_model import BaseModel

Pooling = Literal["mean", "last_token", "max"]

class GENERatorModel(BaseModel):
    """
    GENERator 适配器 (V2 兼容版)
    
    - 修复: 自动应用 Monkey Patch 解决 Tokenizer 初始化报错
    - 特性: 6-mer Tokenizer 支持 (自动补全 'A')
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        device: Optional[str] = None,
        max_len: int = 16384,
    ):
        super().__init__(model_name, model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        
        self._load_model()

    def _load_model(self):
        print(f"Loading {self.model_name} from {self.model_path}...")
        
        # 1. 加载 Tokenizer
        try:
            # 这里的 trust_remote_code=True 会加载远程的 DNAKmerTokenizer
            # 上面的补丁会确保它初始化时不崩
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side="left" # Decoder 模型左填充
            )
            
            # 确保有 pad_token (GENERator 可能没有默认 pad)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"Set pad_token to eos_token: {self.tokenizer.pad_token}")
                
        except Exception as e:
            print(f"Tokenizer loading failed: {e}")
            raise

        # 2. 加载模型
        try:
            # 尝试 bfloat16 加速
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map=None
            )
            self.model.config.use_cache = False
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            raise

        self.model.to(self.device).eval()
        print(f"[{self.model_name}] loaded on {self.device} (Type: {type(self.model).__name__})")

    def _preprocess(self, seq: str) -> str:
        """
        预处理逻辑：
        1. 转大写
        2. [关键] 检查长度是否为 6 的倍数，如果不是，在左侧补 'A' (遵循官方建议)
        """
        seq = seq.strip().upper()
        
        # GENERator 的 6-mer tokenizer 要求长度必须整除 6
        remainder = len(seq) % 6
        if remainder != 0:
            pad_len = 6 - remainder
            # 左侧补 A
            seq = 'A' * pad_len + seq
            
        return seq

    # ---------- Embedding 接口 ----------
    @torch.no_grad()
    def get_embedding(
        self,
        sequences: Union[str, List[str]],
        batch_size: int = 8, 
        pool: Pooling = "mean", 
        layer_index: int = -1,
        return_numpy: bool = True,
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        
        if isinstance(sequences, str):
            sequences = [sequences]

        # 预处理 (6-mer 对齐)
        processed_seqs = [self._preprocess(s) for s in sequences]

        results = []
        for i in tqdm(range(0, len(processed_seqs), batch_size), desc="GENERator Embedding"):
            batch_seqs = processed_seqs[i : i + batch_size]
            
            encoded = self.tokenizer(
                batch_seqs,
                max_length=self.max_len,
                return_tensors='pt',
                padding=True,
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
                if self.tokenizer.padding_side == "left":
                    # Left padding: 最后一个 token 是序列末尾
                    emb = token_embeddings[:, -1, :]
                else:
                    # Right padding
                    sequence_lengths = attention_mask.sum(dim=1) - 1
                    emb = token_embeddings[torch.arange(token_embeddings.size(0), device=self.device), sequence_lengths]
            
            elif pool == "max":
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(token_embeddings)
                token_embeddings[mask_expanded == 0] = -1e9
                emb = torch.max(token_embeddings, 1)[0]
                
            else:
                raise ValueError(f"Unknown pooling strategy: {pool}")
            
            results.append(emb.cpu())
            
            del input_ids, attention_mask, outputs, hidden_states
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
        batch_size: int = 8,
    ) -> List[float]:
        """
        计算序列的 Average Log-Likelihood (PLL)。
        """
        processed_seqs = [self._preprocess(s) for s in sequences]
        scores = []
        
        for i in tqdm(range(0, len(processed_seqs), batch_size), desc="Scoring with GENERator"):
            batch_seqs = processed_seqs[i : i + batch_size]
            
            encoded = self.tokenizer(
                batch_seqs,
                max_length=self.max_len,
                return_tensors='pt',
                padding=True,
                truncation=True
            )
            
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # Forward with labels
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            logits = outputs.logits 
            
            # Shift Logic
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            token_losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
            token_losses = token_losses.view(shift_labels.size())
            
            valid_nll = token_losses * shift_mask
            seq_nll = valid_nll.sum(dim=1)
            seq_lens = shift_mask.sum(dim=1)
            
            avg_log_likelihood = -seq_nll / torch.clamp(seq_lens, min=1e-9)
            scores.extend(avg_log_likelihood.cpu().tolist())

        return scores


if __name__ == "__main__":
    MODEL_PATH = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/GENERator-v2-prokaryote-1.2b-base"
 
    m = GENERatorModel(
        "GENERator", 
        model_path=MODEL_PATH,
    )

    test_seqs = [
        "ACGTAGACGTAG", # 12bp (6x2)
        "ACGTAGACGT"    # 10bp -> Will be padded to 12bp
    ]

    embs = m.embed_sequences(test_seqs, pooling="mean")
    print(f"\n[Embedding Shape]: {embs.shape}")
    print(embs)

    scores = m.score_sequences(test_seqs, batch_size=2)
    print(f"\n[Likelihood Scores]: {scores}")

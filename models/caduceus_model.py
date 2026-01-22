# models/caduceus_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import json
from typing import List, Optional, Literal, Union, Tuple, Dict, Sequence, Any

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, PreTrainedTokenizer

# 检查 mamba_ssm 依赖
try:
    import mamba_ssm
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("[Caduceus] Warning: 'mamba_ssm' not found. Caduceus models require this library.")

from .base_model import BaseModel

Pooling = Literal["mean", "max", "cls", "last"]

# =============================================================================
# 1. 本地 Tokenizer 实现 (提取自 tokenization_caduceus.py)
# =============================================================================
class CaduceusLocalTokenizer(PreTrainedTokenizer):
    """
    Caduceus 字符级分词器，提取自官方实现，避免 AutoTokenizer 联网问题。
    """
    model_input_names = ["input_ids"]

    def __init__(self,
                 model_max_length: int = 131072, # Caduceus 通常支持超长序列
                 characters: Sequence[str] = ("A", "C", "G", "T", "N"),
                 complement_map=None,
                 bos_token="[BOS]",
                 eos_token="[SEP]",
                 sep_token="[SEP]",
                 cls_token="[CLS]",
                 pad_token="[PAD]",
                 mask_token="[MASK]",
                 unk_token="[UNK]",
                 **kwargs):
        
        if complement_map is None:
            complement_map = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
        self.characters = characters
        self.model_max_length = model_max_length

        # 词表构建
        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[MASK]": 3,
            "[PAD]": 4,
            "[RESERVED]": 5,
            "[UNK]": 6,
            **{ch: i + 7 for i, ch in enumerate(self.characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
        
        # 兼容性处理
        add_prefix_space = kwargs.pop("add_prefix_space", False)
        padding_side = kwargs.pop("padding_side", "right") # Mamba 通常 right padding

        # 必须先初始化词表字典，再调用 super().__init__，因为父类会调用 get_vocab
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=add_prefix_space,
            model_max_length=model_max_length,
            padding_side=padding_side,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def get_vocab(self) -> Dict[str, int]:
        """
        必须实现此方法，否则 transformers 初始化时会报错
        """
        return dict(self._vocab_str_to_int)

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        # 简单的字符级分词，转大写
        return list(text.upper())

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple:
        return ()

# =============================================================================
# 2. Caduceus 模型适配器
# =============================================================================

class CaduceusModel(BaseModel):
    """
    Caduceus (Mamba-based DNA Model) 适配器
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

        if not HAS_MAMBA and "cuda" in self.device:
             print("[Caduceus] Warning: 'mamba_ssm' not installed. Running on CUDA might fail or be slow.")

        self._load_model()

    def _load_model(self):
        if self.hf_home:
            os.environ["HF_HOME"] = self.hf_home

        print(f"[{self.model_name}] Initializing Tokenizer...")
        # 尝试从 config.json 读取 max_length
        max_len = 131072
        try:
            with open(os.path.join(self.model_path, "config.json"), 'r') as f:
                cfg = json.load(f)
                # Caduceus config 可能包含 model_max_length 或 d_model
                if "model_max_length" in cfg:
                    max_len = cfg["model_max_length"]
        except:
            pass
        
        self.tokenizer = CaduceusLocalTokenizer(model_max_length=max_len)

        print(f"[{self.model_name}] Loading model from {self.model_path}...")
        try:
            # 使用 trust_remote_code=True 加载自定义模型代码
            self.model = AutoModelForMaskedLM.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code,
                local_files_only=True # 优先本地
            )
        except Exception as e:
            print(f"[{self.model_name}] Local load failed, trying standard load: {e}")
            self.model = AutoModelForMaskedLM.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code
            )

        self.model.to(self.device).eval()
        self.model_max_len = max_len
        self.mask_id = self.tokenizer.mask_token_id

        print(f"[{self.model_name}] Loaded on {self.device}.")

    # ---------- 预处理 ----------
    def _preprocess_sequence(self, seq: str) -> str:
        return seq.strip().upper()

    # ---------- Embedding ----------
    @torch.no_grad()
    def embed_sequences(
        self,
        sequences: List[str],
        batch_size: int = 16, # Mamba 推理显存通常较省，但 embedding 维度可能大
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
            
        final_list = []
        for i in tqdm(range(0, len(sequences), batch_size), desc="Embedding"):
            chunk = sequences[i : i + batch_size]
            proc_chunk = [self._preprocess_sequence(s) for s in chunk]
            
            enc = self.tokenizer(
                proc_chunk,
                return_tensors="pt",
                padding=True,
                truncation=truncation,
                max_length=max_length or self.model_max_len,
            )
            
            input_ids = enc["input_ids"].to(self.device)
            attn_mask = enc.get("attention_mask")
            if attn_mask is not None:
                attn_mask = attn_mask.to(self.device)

            # Forward
            # output_hidden_states=True 以获取 hidden states
            out = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Caduceus 的 hidden_states[-1] 是最后一层的输出
            hidden_states = out.hidden_states
            target = hidden_states[layer_index] # (B, L, D)

            # Pooling
            mask = attn_mask.bool() if attn_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)
            
            if exclude_special:
                # 排除 [CLS], [SEP], [PAD] 等
                special = set(self.tokenizer.all_special_ids)
                spec_mask = torch.zeros_like(mask)
                for r in range(input_ids.shape[0]):
                    for c in range(input_ids.shape[1]):
                        if input_ids[r, c].item() in special:
                            spec_mask[r, c] = True
                mask = mask & (~spec_mask)
            
            if pool == "cls":
                # Caduceus 有 [CLS] (id=0)，通常在开头
                vec = target[:, 0, :]
            elif pool == "mean":
                m_f = mask.unsqueeze(-1).float()
                summed = (target * m_f).sum(dim=1)
                denom = m_f.sum(dim=1).clamp_min(1.0)
                vec = summed / denom
            elif pool == "max":
                tmp = target.clone()
                tmp[~mask] = float("-inf")
                vec = tmp.max(dim=1).values
                vec[torch.isinf(vec)] = 0.0
            elif pool == "last":
                # 取最后一个有效 token
                lengths = mask.sum(dim=1) - 1
                lengths = lengths.clamp_min(0)
                vec = target[torch.arange(target.size(0)), lengths]
            else:
                raise ValueError(f"Unknown pool {pool}")

            final_list.append(vec.cpu())

        res = torch.cat(final_list, dim=0) if final_list else torch.empty(0)
        return res.numpy() if return_numpy else res

    # ---------- PLL Scoring ----------
    @torch.no_grad()
    def score_sequences(
        self,
        sequences: List[str],
        batch_size: int = 256,
    ) -> List[float]:
        all_scores = []
        for seq in tqdm(sequences, desc=f"Scoring with {self.model_name}"):
            score = self._score_single_sequence(seq, batch_size)
            all_scores.append(score)
        return all_scores

    def _score_single_sequence(self, sequence: str, mask_batch_size: int) -> float:
        proc_seq = self._preprocess_sequence(sequence)
        enc = self.tokenizer(
            proc_seq, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.model_max_len
        )
        input_ids = enc["input_ids"].to(self.device).squeeze(0) # (L,)
        seq_len = input_ids.shape[0]
        
        special_ids = set(self.tokenizer.all_special_ids)
        valid_positions = [i for i in range(seq_len) if input_ids[i].item() not in special_ids]
        
        if not valid_positions:
            return 0.0

        total_logprob = 0.0
        total_count = 0

        for start_idx in range(0, len(valid_positions), mask_batch_size):
            chunk = valid_positions[start_idx : start_idx + mask_batch_size]
            B = len(chunk)
            
            masked_ids = input_ids.unsqueeze(0).repeat(B, 1) # (B, L)
            
            true_tokens = []
            for k, pos in enumerate(chunk):
                true_tokens.append(input_ids[pos].item())
                masked_ids[k, pos] = self.mask_id
            
            out = self.model(input_ids=masked_ids)
            logits = out.logits # (B, L, V)
            
            batch_idx = torch.arange(B, device=self.device)
            pos_idx = torch.tensor(chunk, device=self.device)
            true_idx = torch.tensor(true_tokens, device=self.device)
            
            target_logits = logits[batch_idx, pos_idx, :]
            log_probs = torch.log_softmax(target_logits, dim=-1)
            val = log_probs[batch_idx, true_idx]
            
            total_logprob += val.sum().item()
            total_count += B
            
        return total_logprob / max(1, total_count)


if __name__ == "__main__":
    # 测试
    MODEL_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"
    
    m = CaduceusModel(
        model_name="caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
        model_path=MODEL_DIR,
        trust_remote_code=True
    )
    
    seqs = ["ACGTACGT"]
    emb = m.embed_sequences(seqs)
    print("Embedding shape:", emb.shape)
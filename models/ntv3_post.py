# models/ntv3_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import torch
import numpy as np
from typing import List, Optional, Literal, Union
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

Pooling = Literal["mean", "max", "cls"]

class NTV3Model:
    """
    Nucleotide Transformer v3 (NTv3) 通用适配器 - 官方最佳实践版
    支持:
      - 自动调用 model.encode_species() 处理物种ID
      - 兼容 Pre-trained (无 species) 和 Post-trained (有 species)
      - 自动对齐长度
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        hf_home: Optional[str] = None,
        device: Optional[str] = None,
        trust_remote_code: bool = True,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.hf_home = hf_home
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.trust_remote_code = trust_remote_code
        
        self._load_model()

    def _load_model(self):
        if self.hf_home:
            os.environ["HF_HOME"] = self.hf_home

        print(f"[{self.model_name}] Loading tokenizer from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )

        print(f"[{self.model_name}] Loading model from {self.model_path}...")
        
        # 官方示例使用的是 AutoModel，这对 Post-trained 最稳妥
        # 对于 Pre-trained，AutoModel 也能跑，虽然没有 MaskedLM 头，但我们主要是为了取 Embedding
        try:
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code
            )
        except Exception as e:
            # 如果 AutoModel 失败（极少见），尝试 MaskedLM
            print(f"AutoModel failed ({e}), trying AutoModelForMaskedLM...")
            self.model = AutoModelForMaskedLM.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code
            )

        self.model.to(self.device).eval()

        # 检测对齐因子
        self.align_factor = 2 ** getattr(self.model.config, "num_downsamples", 7)
        if self.align_factor < 16: self.align_factor = 128
        print(f"[{self.model_name}] Align Factor: {self.align_factor}")
        
        self.model_max_len = getattr(self.model.config, "max_position_embeddings", 1000)

        # 检测是否支持 encode_species 方法
        if hasattr(self.model, "encode_species"):
            print(f"[{self.model_name}] Detected helper method: model.encode_species().")
            self.use_encode_species = True
        else:
            print(f"[{self.model_name}] No encode_species method found (Pre-trained?).")
            self.use_encode_species = False

    def _preprocess_sequence(self, seq: str) -> str:
        return seq.strip().upper()

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
        species: str = "human",  # 注意：官方示例用的是 "human" 而不是 "Homo sapiens"
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        
        if isinstance(sequences, str):
            sequences = [sequences]
            
        def _process_batch(batch_strs):
            proc = [self._preprocess_sequence(s) for s in batch_strs]
            
            # 1. 长度安全对齐
            raw_max_len = max_length or self.model_max_len
            safe_max_len = (raw_max_len // self.align_factor) * self.align_factor
            if safe_max_len == 0: safe_max_len = self.align_factor

            # 2. Tokenize (官方参数：add_special_tokens=False, pad_to_multiple_of=128)
            enc = self.tokenizer(
                proc,
                return_tensors="pt",
                padding=True,
                truncation=truncation,
                max_length=safe_max_len,
                pad_to_multiple_of=self.align_factor,
                add_special_tokens=False,
            )
            
            # 3. 处理 species_ids
            species_tensor = None
            if self.use_encode_species:
                # 官方 API: model.encode_species(['human', ...]) -> Tensor
                # 我们这里构建一个长度为 batch 的 list
                batch_species_list = [species] * len(proc)
                try:
                    # 注意：encode_species 返回的 tensor 默认在 CPU，需要移到 GPU
                    # 并且有些版本直接返回 tensor，有些返回 list，通常是 tensor
                    species_ids = self.model.encode_species(batch_species_list)
                    species_tensor = species_ids.to(self.device)
                except Exception as e:
                    raise ValueError(f"Failed to encode species '{species}'. Error: {e}")

            # 4. 移到 GPU
            enc = {k: v.to(self.device) for k, v in enc.items()}
            input_ids = enc["input_ids"]
            
            # 5. Mask 处理
            if "attention_mask" in enc:
                attn_mask = enc["attention_mask"]
            else:
                pad_id = self.tokenizer.pad_token_id or 0
                attn_mask = (input_ids != pad_id).long()

            # 6. 构建 Forward 参数
            forward_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attn_mask,
                "output_hidden_states": True, # 必须加，否则拿不到 embedding
            }
            
            if species_tensor is not None:
                forward_kwargs["species_ids"] = species_tensor

            # 7. Forward
            outputs = self.model(**forward_kwargs)
            
            # 8. 提取 Embedding
            # 注意：Post-trained 模型返回的 output 可能包含 logits, bigwig_logits 等
            # 但只要传了 output_hidden_states=True，它一定有 hidden_states
            hidden_states = outputs.hidden_states
            target_layer = hidden_states[layer_index]

            # 9. Pooling
            mask = attn_mask.bool()
            
            if exclude_special:
                special_ids_list = self.tokenizer.all_special_ids
                if special_ids_list:
                    special_ids_tensor = torch.tensor(special_ids_list, device=self.device)
                    is_special = torch.isin(input_ids, special_ids_tensor)
                    mask = mask & (~is_special)

            if pool == "cls":
                return target_layer[:, 0, :]
            elif pool == "mean":
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

        # Batch 循环
        final_list = []
        for i in tqdm(range(0, len(sequences), batch_size), desc=f"Embedding"):
            chunk = sequences[i : i + batch_size]
            vec = _process_batch(chunk)
            final_list.append(vec.cpu())

        res = torch.cat(final_list, dim=0) if final_list else torch.empty(0)
        return res.numpy() if return_numpy else res

if __name__ == "__main__":
    MODEL_PATH = "../../model_weight/NTv3_100M_post"

    try:
        model = NTV3Model(
            model_name="NTv3_Official",
            model_path=MODEL_PATH,
            trust_remote_code=True
        )
        
        seqs = ["ACGTACGTACGT", "GGGGTTTTCCCCAAAA"]
        
        # 使用官方文档里的 "human" 而非 Latin name，
        # 因为 model.encode_species 内部通常做了映射
        emb = model.get_embedding(
            seqs, 
            batch_size=2, 
            species="human" 
        )
        
        print("\nSuccess!")
        print(f"Output shape: {emb.shape}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
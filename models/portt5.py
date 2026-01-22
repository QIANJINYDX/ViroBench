# models/prottrans_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import re
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Optional, Literal, Union

try:
    from transformers import (
        AutoTokenizer, 
        AutoModel, 
        AutoModelForMaskedLM, 
        T5EncoderModel,
        T5Tokenizer
    )
except ImportError:
    print("Error: transformers not installed. Please install via `pip install transformers`.")

from .base_model import BaseModel

Pooling = Literal["mean", "max", "cls", "per_residue"]

class ProtTransModel(BaseModel):
    """
    ProtTrans 适配器 (DNA 支持版)
    
    - 输入: DNA 序列 (ATCG)
    - 流程: DNA -> 翻译 -> 蛋白质 -> 加空格 -> ProtTrans -> 向量/评分
    - 支持: ProtBert, ProtT5
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        device: Optional[str] = None,
        max_len: int = 1024,
        half_precision: bool = False,
        translation_mode: Literal["first_orf", "fixed_frame"] = "first_orf",
        translation_frame: int = 0,
        stop_at_stop: bool = True,
    ):
        super().__init__(model_name, model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.half_precision = half_precision
        
        # 翻译参数
        self.translation_mode = translation_mode
        self.translation_frame = int(translation_frame) % 3
        self.stop_at_stop = stop_at_stop
        
        # 标准遗传密码表
        self._codon_table = {
            "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
            "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M", "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
            "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S", "CCT": "P", "CCC": "P",
            "CCA": "P", "CCG": "P", "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T", "GCT": "A", "GCC": "A",
            "GCA": "A", "GCG": "A", "TAT": "Y", "TAC": "Y", "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
            "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
            "TGT": "C", "TGC": "C", "TGG": "W", "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R",
            "AGG": "R", "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
            "TAA": "*", "TAG": "*", "TGA": "*"
        }
        
        self._load_model()

    def _load_model(self):
        print(f"Loading {self.model_name} from {self.model_path}...")
        
        # 1. 加载 Tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, do_lower_case=False)
        except Exception as e:
            # 自动降级处理 (针对 ProtT5)
            if "tiktoken" in str(e) or "Tiktoken" in str(e) or "SentencePiece" in str(e):
                print(f"Warning: Fast tokenizer failed ({e}). Falling back to slow T5Tokenizer...")
                try:
                    self.tokenizer = T5Tokenizer.from_pretrained(self.model_path, do_lower_case=False)
                except ImportError:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, do_lower_case=False, use_fast=False)
            else:
                print(f"Warning: Tokenizer loading error: {e}. Trying use_fast=False...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, do_lower_case=False, use_fast=False)

        # 2. 识别模型类型并加载
        is_t5 = "t5" in self.model_path.lower() or "t5" in self.model_name.lower()
        torch_dtype = torch.float16 if self.half_precision and torch.cuda.is_available() else torch.float32
        
        try:
            if is_t5:
                # ProtT5 (Encoder)
                print("Detected T5 architecture. Loading T5EncoderModel...")
                self.model = T5EncoderModel.from_pretrained(
                    self.model_path, 
                    torch_dtype=torch_dtype
                )
                self.model_type = "t5"
                self.can_score = False
            else:
                # ProtBert (MLM)
                print("Detected BERT architecture. Loading AutoModelForMaskedLM...")
                self.model = AutoModelForMaskedLM.from_pretrained(
                    self.model_path, 
                    torch_dtype=torch_dtype
                )
                self.model_type = "bert"
                self.can_score = True
                
        except Exception as e:
            print(f"Warning: Specialized loading failed ({e}). Falling back to AutoModel.")
            self.model = AutoModel.from_pretrained(self.model_path, torch_dtype=torch_dtype)
            self.model_type = "generic"
            self.can_score = False

        self.model.to(self.device).eval()
        self.mask_id = self.tokenizer.mask_token_id
        
        print(f"[{self.model_name}] loaded on {self.device} (Type: {self.model_type}, FP16: {self.half_precision})")

    # ---------- DNA 翻译工具函数 ----------
    @staticmethod
    def _clean_dna(seq: str) -> str:
        """统一大小写并只保留 A/C/G/T/N，U 替换为 T。"""
        s = seq.strip().upper().replace("U", "T")
        out = []
        for ch in s:
            if ch in ("A", "C", "G", "T", "N"):
                out.append(ch)
        return "".join(out)

    def _translate_dna_to_protein(self, dna_seq: str) -> str:
        """DNA -> Protein"""
        dna = self._clean_dna(dna_seq)
        if len(dna) < 3: return "X" # 序列太短

        # 确定起始位置
        if self.translation_mode == "first_orf":
            start = dna.find("ATG")
            if start == -1:
                start = 0
            dna = dna[start:]
        else: # fixed_frame
            dna = dna[self.translation_frame:]

        aa_list = []
        for i in range(0, len(dna) - 2, 3):
            codon = dna[i: i + 3]
            if len(codon) < 3: break
            
            aa = self._codon_table.get(codon, "X")
            if aa == "*": # 终止密码子
                if self.stop_at_stop:
                    break
                else:
                    aa = "X"
            aa_list.append(aa)

        if not aa_list: return "X"
        return "".join(aa_list)

    def _preprocess(self, seq: str) -> str:
        """ProtTrans 特有: 去空白 + 字符间加空格 (M K A L ...)"""
        seq = "".join(seq.split())
        return " ".join(list(seq))

    # ---------- Embedding 接口 ----------
    @torch.no_grad()
    def get_embedding(
        self,
        sequences: Union[str, List[str]],
        batch_size: int = 32,
        pool: Pooling = "mean",
        layer_index: int = -1,
        return_numpy: bool = True,
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        
        if isinstance(sequences, str):
            sequences = [sequences]

        # 1. 翻译 (DNA -> Protein)
        # 2. 预处理 (Protein -> M K A L ...)
        processed_seqs = []
        for dna in sequences:
            aa = self._translate_dna_to_protein(dna)
            processed = self._preprocess(aa)
            processed_seqs.append(processed)

        results = []
        # ProtT5-XL 显存占用大，自动调小 batch
        if "t5" in self.model_type and batch_size > 4:
            # print(f"Warning: ProtT5 is large. Reducing batch_size from {batch_size} to 4 to avoid OOM.")
            # batch_size = 4
            pass

        for i in tqdm(range(0, len(processed_seqs), batch_size), desc="ProtT5 Embedding"):
            batch_seqs = processed_seqs[i : i + batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_seqs,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )
            
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            # Forward
            if self.model_type == "t5":
                # T5EncoderModel 输出
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                token_embeddings = outputs.last_hidden_state
            else:
                # BERT / MaskedLM 输出
                outputs = self.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    output_hidden_states=True
                )
                if hasattr(outputs, "hidden_states"):
                    token_embeddings = outputs.hidden_states[layer_index]
                else:
                    token_embeddings = outputs.last_hidden_state # Fallback

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
            
            del input_ids, attention_mask, outputs, token_embeddings
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
        batch_size: int = 64,
    ) -> List[float]:
        """计算序列 PLL (输入为 DNA)"""
        if not self.can_score or self.mask_id is None:
            # ProtT5 无法评分，返回 0
            return [0.0] * len(sequences)

        # 翻译 + 预处理
        processed_seqs = []
        for dna in sequences:
            aa = self._translate_dna_to_protein(dna)
            processed = self._preprocess(aa)
            processed_seqs.append(processed)

        scores = []
        for seq in tqdm(processed_seqs, desc=f"Scoring with {self.model_name}"):
            score = self._score_single(seq, batch_size)
            scores.append(score)
        return scores

    def _score_single(self, seq: str, mask_bs: int) -> float:
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
                batch_input[batch_idx, pos_idx] = self.mask_id

            outputs = self.model(input_ids=batch_input, return_dict=True)
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
    MODEL_PATH = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/prot_t5_xl_uniref50"
    
    m = ProtTransModel(
        "ProtBert", 
        model_path=MODEL_PATH,
        half_precision=False, # BERT 通常用 float32
        translation_mode="first_orf" # 自动寻找 ATG
    )

    test_seqs = [
        # 随机序列
        "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATTCGTTAGC", 
        # 含 ATG 的序列 (对应 M...)
        "ATGGGCGAGGCCATGGGCCTGACCCAGCCCGCCGTGAGCCGCGCCGTGGCCCGCCTGGAGGAGCGCGTGGGCATCCGCATCTTCAACCGCACCGCCCGCGCCATCACCCTGACCGACGAGGGCCGCCGCTTCTACGAGGCCGTGGCCCCCCTGCTGGCCGGCATCGAGATGCACGGCTACCGCGTGAACGTGGAGGGCGTGGCCCAGCTGCTGGAGCTGTACGCCCGCGACATCCTGGCCGAGGGCCGCCTGGTGCAGCTGCTGCCCGAGTGGGCCGACTGA"
    ]

    # 1. Embedding
    embs = m.embed_sequences(test_seqs, pooling="mean")
    print(f"\n[Embedding Shape]: {embs.shape}")
       
# models/ankh_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Optional, Literal, Union

try:
    from transformers import T5EncoderModel, AutoTokenizer, T5Tokenizer
except ImportError:
    print("Error: transformers not installed. Please install via `pip install transformers`.")

from .base_model import BaseModel

Pooling = Literal["mean", "max", "cls"]

class AnkhModel(BaseModel):
    """
    Ankh 适配器 (DNA 支持版)
    
    - 输入: DNA 序列 (ATCG)
    - 流程: DNA -> 翻译 -> 蛋白质序列 -> Ankh (list of chars) -> Embedding
    - 注意: 仅支持特征提取，不支持 PLL 评分
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
            if "sentencepiece" in str(e).lower():
                print("Error: sentencepiece library is missing.")
                print("Please run: `pip install sentencepiece`")
                raise e
            else:
                print(f"Tokenizer loading failed: {e}. Trying T5Tokenizer directly...")
                self.tokenizer = T5Tokenizer.from_pretrained(self.model_path, do_lower_case=False)

        # 2. 加载模型
        torch_dtype = torch.float16 if self.half_precision and torch.cuda.is_available() else torch.float32
        
        try:
            self.model = T5EncoderModel.from_pretrained(
                self.model_path, 
                torch_dtype=torch_dtype
            )
        except Exception as e:
            print(f"Model loading failed: {e}")
            raise

        self.model.to(self.device).eval()
        print(f"[{self.model_name}] loaded on {self.device} (FP16: {self.half_precision})")

    # ---------- DNA 翻译工具函数 ----------
    @staticmethod
    def _clean_dna(seq: str) -> str:
        s = seq.strip().upper().replace("U", "T")
        out = [ch for ch in s if ch in ("A", "C", "G", "T", "N")]
        return "".join(out)

    def _translate_dna_to_protein(self, dna_seq: str) -> str:
        dna = self._clean_dna(dna_seq)
        if len(dna) < 3: return "X" 

        if self.translation_mode == "first_orf":
            start = dna.find("ATG")
            if start == -1: start = 0
            dna = dna[start:]
        else:
            dna = dna[self.translation_frame:]

        aa_list = []
        for i in range(0, len(dna) - 2, 3):
            codon = dna[i: i + 3]
            if len(codon) < 3: break
            aa = self._codon_table.get(codon, "X")
            if aa == "*":
                if self.stop_at_stop: break
                else: aa = "X"
            aa_list.append(aa)

        if not aa_list: return "X"
        return "".join(aa_list)

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
        for i in tqdm(range(0, len(sequences), batch_size), desc="Ankh Embedding"):
            batch_raw = sequences[i : i + batch_size]
            
            # [新增] 1. DNA -> Protein
            # [关键] Ankh 要求输入是字符列表，且 is_split_into_words=True
            processed_seqs = []
            for dna in batch_raw:
                aa = self._translate_dna_to_protein(dna)
                processed_seqs.append(list(aa))
            
            # 2. Tokenize
            encoded = self.tokenizer(
                processed_seqs,
                add_special_tokens=True,
                padding=True,
                is_split_into_words=True, # 必须开启
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            # 3. Forward
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state

            # 4. Pooling
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
        batch_size: int = 32,
    ) -> List[float]:
        # Ankh 是 Encoder-only，不支持 PLL
        print(f"Warning: {self.model_name} (T5 Encoder) does not support PLL scoring directly. Returning 0s.")
        return [0.0] * len(sequences)


if __name__ == "__main__":
    MODEL_PATH = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/ankh3-large" 
    
    print("=== Testing Ankh Adapter (DNA Input) ===")
    try:
        m = AnkhModel(
            "Ankh-Large", 
            model_path=MODEL_PATH,
            half_precision=True,
            translation_mode="first_orf" # 自动寻找 ATG
        )

        test_dna_seqs = [
            # 带 ATG 的 DNA -> 翻译为 Protein -> Ankh Embedding
            "ATGGGCGAGGCCATGGGCCTGACCCAGCCCGCCGTGAGCCGCGCCGTGGCCCGCCTGGAGGAGCGCGTGGGCATCCGCATCTTCAACCGCACCGCCCGCGCCATCACCCTGACCGACGAGGGCCGCCGCTTCTACGAGGCCGTGGCCCCCCTGCTGGCCGGCATCGAGATGCACGGCTACCGCGTGAACGTGGAGGGCGTGGCCCAGCTGCTGGAGCTGTACGCCCGCGACATCCTGGCCGAGGGCCGCCTGGTGCAGCTGCTGCCCGAGTGGGCCGACTGA",
            "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATTCGTTAGC"
        ]

        embs = m.embed_sequences(test_dna_seqs, pooling="mean")
        print(f"\n[Embedding Shape]: {embs.shape}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nTest failed: {e}")
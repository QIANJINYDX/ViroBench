# models/protgpt2_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Optional, Literal, Union

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Error: transformers not installed. Please install via `pip install transformers`.")

from .base_model import BaseModel

Pooling = Literal["mean", "last_token", "max"]

class ProtGPT2Model(BaseModel):
    """
    ProtGPT2 适配器（DNA → 蛋白 → ProtGPT2）
    
    - 外部接口：接受 DNA 序列（ATCG）
    - 内部流程：DNA -> 氨基酸序列 -> 格式化(换行符/特殊Token) -> ProtGPT2 -> 向量/评分
    - 架构：GPT-2 (Decoder-only Causal LM)
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        device: Optional[str] = None,
        translation_mode: Literal["first_orf", "fixed_frame"] = "first_orf",
        translation_frame: int = 0,
        stop_at_stop: bool = True,
        max_len: int = 1024,
    ):
        super().__init__(model_name, model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        
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
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            # GPT2 默认没有 pad_token，手动设置为 eos_token 以支持 batch
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"Set pad_token to eos_token: {self.tokenizer.eos_token}")
        except Exception as e:
            print(f"Tokenizer loading failed: {e}")
            raise

        # 2. 加载模型
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        except Exception as e:
            print(f"Model loading failed: {e}")
            raise

        self.model.to(self.device).eval()
        print(f"[{self.model_name}] loaded on {self.device}")

    # ---------- DNA 翻译逻辑 (复用 ESMModel 代码) ----------
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

    # ---------- ProtGPT2 特有预处理 ----------
    def _preprocess_protein(self, seq: str) -> str:
        """
        ProtGPT2 官方要求:
        1. 每 60 个氨基酸插入换行符 '\n' (模拟 FASTA)
        2. 序列包裹在 <|endoftext|> 中
        """
        seq = "".join(seq.split()) # 清理现有空白
        
        # 插入换行符
        seq_formatted = "\n".join([seq[i : i + 60] for i in range(0, len(seq), 60)])
        
        # 构造 prompt (参考官方 README)
        return f"{self.tokenizer.bos_token}{seq_formatted}{self.tokenizer.eos_token}"

    # ---------- Embedding 接口 ----------
    @torch.no_grad()
    def get_embedding(
        self,
        sequences: Union[str, List[str]],
        batch_size: int = 16,
        pool: Pooling = "last_token", # Causal LM 推荐 last_token 代表全序信息
        layer_index: int = -1,
        return_numpy: bool = True,
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        
        if isinstance(sequences, str):
            sequences = [sequences]

        results = []
        # 分批处理
        for i in tqdm(range(0, len(sequences), batch_size), desc="ProtGPT2 Embedding"):
            batch_raw = sequences[i : i + batch_size]
            
            # 1. 翻译 + 预处理
            processed_seqs = []
            for dna in batch_raw:
                aa = self._translate_dna_to_protein(dna)
                processed = self._preprocess_protein(aa)
                processed_seqs.append(processed)
            
            # 2. Tokenize
            encoded = self.tokenizer(
                processed_seqs,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )
            
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            # 3. Forward
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # 获取指定层的 Hidden States
            hidden_states = outputs.hidden_states
            if layer_index < 0:
                layer_index = len(hidden_states) + layer_index
            
            token_embeddings = hidden_states[layer_index] # [B, L, H]

            # 4. Pooling
            if pool == "last_token":
                # 找到最后一个有效 token (EOS) 的位置
                # 注意: 我们手动加了 EOS，且 padding 在右侧
                sequence_lengths = attention_mask.sum(dim=1) - 1
                emb = token_embeddings[torch.arange(token_embeddings.size(0), device=self.device), sequence_lengths]
                
            elif pool == "mean":
                mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                emb = sum_embeddings / sum_mask
            
            elif pool == "max":
                mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                token_embeddings[mask_expanded == 0] = -1e9
                emb = torch.max(token_embeddings, 1)[0]
                
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
        batch_size: int = 16,
    ) -> List[float]:
        """
        计算序列的 Average Log-Likelihood (PLL)。
        输入为 DNA 序列。
        """
        scores = []
        for i in tqdm(range(0, len(sequences), batch_size), desc="Scoring with ProtGPT2"):
            batch_raw = sequences[i : i + batch_size]
            
            # 翻译 + 预处理
            processed_seqs = []
            for dna in batch_raw:
                aa = self._translate_dna_to_protein(dna)
                processed = self._preprocess_protein(aa)
                processed_seqs.append(processed)
            
            encoded = self.tokenizer(
                processed_seqs,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )
            
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            
            # ProtGPT2 是 Causal LM，labels=input_ids 会自动触发 Shifted Loss 计算
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            logits = outputs.logits # [B, L, V]
            
            # 手动计算 Per-sample Log Likelihood
            # Shift: logits[t] 预测 input_ids[t+1]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            token_losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
            token_losses = token_losses.view(shift_labels.size())
            
            # Loss 是 Negative Log Likelihood (NLL)
            valid_nll = token_losses * shift_mask
            seq_nll = valid_nll.sum(dim=1)
            seq_lens = shift_mask.sum(dim=1)
            
            # 返回 Average Log Likelihood = -NLL / Length
            avg_log_likelihood = -seq_nll / torch.clamp(seq_lens, min=1e-9)
            scores.extend(avg_log_likelihood.cpu().tolist())

        return scores


if __name__ == "__main__":
    # 配置: 替换为本地路径或 HF ID (nferruz/ProtGPT2)
    MODEL_PATH = "../../model_weight/ProtGPT2"
    
    print("=== Testing ProtGPT2 Adapter (DNA Input) ===")
    try:
        if not os.path.exists(MODEL_PATH) and not "/" in MODEL_PATH:
             print("Warning: Model path not found, falling back to HF ID.")
             MODEL_PATH = "nferruz/ProtGPT2"

        m = ProtGPT2Model(
            "ProtGPT2", 
            model_path=MODEL_PATH,
            translation_mode="first_orf" # 自动寻找 ATG
        )
        
        # 输入 DNA 序列
        test_dna_seqs = [
            # 包含 ATG 的序列 (翻译后类似 MGEAM...)
            "ATGGGCGAGGCCATGGGCCTGACCCAGCCCGCCGTGAGCCGCGCCGTGGCCCGCCTGGAGGAGCGCGTGGGCATCCGCATCTTCAACCGCACCGCCCGCGCCATCACCCTGACCGACGAGGGCCGCCGCTTCTACGAGGCCGTGGCCCCCCTGCTGGCCGGCATCGAGATGCACGGCTACCGCGTGAACGTGGAGGGCGTGGCCCAGCTGCTGGAGCTGTACGCCCGCGACATCCTGGCCGAGGGCCGCCTGGTGCAGCTGCTGCCCGAGTGGGCCGACTGA", 
            # 随机短序列
            "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATTCGTTAGC"
        ]
        
        # 1. Embedding
        embs = m.embed_sequences(test_dna_seqs, pooling="last_token")
        print(f"\n[Embedding Shape]: {embs.shape}")
        
        # 2. Scoring
        scores = m.score_sequences(test_dna_seqs, batch_size=2)
        print(f"\n[PLL Scores (Log Likelihood)]: {scores}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nTest failed: {e}")
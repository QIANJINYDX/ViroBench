# models/progen2_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Optional, Literal, Union

# ==========================================
# 自动寻找并导入本地 progen 包
# ==========================================
def _register_progen():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths = [
        os.getcwd(),
        os.path.dirname(current_dir),
        os.path.dirname(os.path.dirname(current_dir)),
        "../../model_weight/progen2"
    ]
    
    for path in search_paths:
        potential_pkg = os.path.join(path, "progen")
        if os.path.exists(os.path.join(potential_pkg, "modeling_progen.py")):
            if path not in sys.path:
                sys.path.append(path)
            return True
            
    return False

HAS_LOCAL_PROGEN = _register_progen()

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
except ImportError:
    print("Error: transformers not installed.")

try:
    from tokenizers import Tokenizer
except ImportError:
    pass

from .base_model import BaseModel

Pooling = Literal["mean", "last_token", "max"]

class ProGen2Model(BaseModel):
    """
    ProGen2 适配器 (DNA 支持版)
    
    - 输入: DNA 序列 (ATCG)
    - 流程: DNA -> 翻译 -> 蛋白质序列 -> ProGen2 -> 向量/评分
    - 修复: 自动设置 pad_token
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
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                if self.tokenizer.vocab and "<|pad|>" in self.tokenizer.get_vocab():
                    self.tokenizer.pad_token = "<|pad|>"
                elif self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    
        except Exception as e:
            print(f"AutoTokenizer failed ({e}), trying generic Tokenizer.from_file...")
            json_path = os.path.join(self.model_path, "tokenizer.json")
            if os.path.exists(json_path):
                self.tokenizer = Tokenizer.from_file(json_path)
                pad_id = self.tokenizer.token_to_id("<|pad|>")
                if pad_id is None: pad_id = self.tokenizer.token_to_id("<|endoftext|>")
                self.tokenizer.pad_token_id = pad_id
                self.tokenizer.eos_token_id = self.tokenizer.token_to_id("<|endoftext|>")
            else:
                raise ValueError(f"Could not load tokenizer from {self.model_path}")

        # 2. 加载模型
        torch_dtype = torch.float16 if self.half_precision and torch.cuda.is_available() else torch.float32
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True
            )
        except Exception as e:
            print(f"AutoModel loading failed ({e}). Trying manual ProGenForCausalLM...")
            if HAS_LOCAL_PROGEN:
                try:
                    from progen.modeling_progen import ProGenForCausalLM
                    self.model = ProGenForCausalLM.from_pretrained(
                        self.model_path,
                        torch_dtype=torch_dtype
                    )
                except Exception as e2:
                    raise RuntimeError(f"Manual loading also failed: {e2}")
            else:
                raise e
        
        if hasattr(self.tokenizer, "vocab_size") and hasattr(self.model, "resize_token_embeddings"):
             if len(self.tokenizer) > self.model.config.vocab_size:
                 self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.to(self.device).eval()
        print(f"[{self.model_name}] loaded on {self.device} (FP16: {self.half_precision})")

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

    def _tokenize_batch(self, sequences):
        """兼容 Transformers 和 Tokenizers 库的统一分词"""
        if hasattr(self.tokenizer, "encode_batch") and not hasattr(self.tokenizer, "__call__"):
            encoded_batch = self.tokenizer.encode_batch(sequences)
            max_len = max(len(enc.ids) for enc in encoded_batch)
            max_len = min(max_len, self.max_len)
            
            input_ids = []
            attention_mask = []
            
            pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
            
            for enc in encoded_batch:
                ids = enc.ids[:max_len]
                mask = [1] * len(ids)
                pad_len = max_len - len(ids)
                ids = ids + [pad_id] * pad_len
                mask = mask + [0] * pad_len
                input_ids.append(ids)
                attention_mask.append(mask)
            
            return {
                "input_ids": torch.tensor(input_ids).to(self.device),
                "attention_mask": torch.tensor(attention_mask).to(self.device)
            }
        else:
            return self.tokenizer(
                sequences,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            ).to(self.device)

    # ---------- Embedding 接口 ----------
    @torch.no_grad()
    def get_embedding(
        self,
        sequences: Union[str, List[str]],
        batch_size: int = 16,
        pool: Pooling = "last_token", 
        layer_index: int = -1,
        return_numpy: bool = True,
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        
        if isinstance(sequences, str):
            sequences = [sequences]

        # [新增] 这里先进行 DNA -> Protein 翻译
        processed_seqs = [self._translate_dna_to_protein(s) for s in sequences]
        
        results = []
        for i in tqdm(range(0, len(processed_seqs), batch_size), desc="ProGen2 Embedding"):
            batch_seqs = processed_seqs[i : i + batch_size]
            
            encoded = self._tokenize_batch(batch_seqs)
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            hidden_states = outputs.hidden_states
            if layer_index < 0:
                layer_index = len(hidden_states) + layer_index
            
            token_embeddings = hidden_states[layer_index]

            if pool == "last_token":
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
        batch_size: int = 16,
    ) -> List[float]:
        # [新增] 先翻译 DNA -> Protein
        processed_seqs = [self._translate_dna_to_protein(s) for s in sequences]
        
        scores = []
        for i in tqdm(range(0, len(processed_seqs), batch_size), desc="Scoring with ProGen2"):
            batch_seqs = processed_seqs[i : i + batch_size]
            
            encoded = self._tokenize_batch(batch_seqs)
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            logits = outputs.logits
            
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
    MODEL_PATH = "../../model_weight/progen2-base"
    
    print("=== Testing ProGen2 Adapter (DNA Input) ===")
    try:
        m = ProGen2Model(
            "ProGen2", 
            model_path=MODEL_PATH,
            translation_mode="first_orf" # 自动寻找 ATG
        )
        
        # 输入 DNA 序列 (包含一个随机短序列和一个带 ATG 的序列)
        test_dna_seqs = [
            # 随机 (翻译后可能很短)
            "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATTCGTTAGC", 
            # 包含 ORF: ATG (M) -> ... -> TGA (Stop)
            "ATGGGCGAGGCCATGGGCCTGACCCAGCCCGCCGTGAGCCGCGCCGTGGCCCGCCTGGAGGAGCGCGTGGGCATCCGCATCTTCAACCGCACCGCCCGCGCCATCACCCTGACCGACGAGGGCCGCCGCTTCTACGAGGCCGTGGCCCCCCTGCTGGCCGGCATCGAGATGCACGGCTACCGCGTGAACGTGGAGGGCGTGGCCCAGCTGCTGGAGCTGTACGCCCGCGACATCCTGGCCGAGGGCCGCCTGGTGCAGCTGCTGCCCGAGTGGGCCGACTGA"
        ]
        
        embs = m.embed_sequences(test_dna_seqs, pooling="last_token")
        print(f"\n[Embedding Shape]: {embs.shape}")
        
        scores = m.score_sequences(test_dna_seqs, batch_size=4)
        print(f"\n[PLL Scores]: {scores}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nTest failed: {e}")
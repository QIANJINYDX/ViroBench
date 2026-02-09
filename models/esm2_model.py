# models/esm2_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import sys
from typing import List, Optional, Union
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForMaskedLM, EsmModel, AutoModelForMaskedLM, EsmTokenizer

from .base_model import BaseModel


Pooling = Literal["mean", "max", "cls"]


class ESM2Model(BaseModel):
    """
    ESM2 适配器（DNA → 蛋白 → ESM2）
    
    基于 Hugging Face transformers 库加载 ESM2 模型。
    
    - 外部接口：仍然接受 DNA 序列（ATCG）
    - 内部流程：DNA -> 氨基酸序列 -> ESM2 -> 序列向量/评分
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        device: Optional[str] = None,
        translation_mode: Literal["first_orf", "fixed_frame"] = "first_orf",
        translation_frame: int = 0,
        stop_at_stop: bool = True,
    ):
        """
        Args:
            model_name: 逻辑名（比如 "ESM2-650M"），仅用于日志
            model_path: 本地 ESM2 权重路径（Hugging Face 格式目录）
            device:     "cuda:0" / "cpu" / None(自动)
            translation_mode:
                - "first_orf": 从第一个 ATG 作为起始密码子开始翻译，遇到终止密码子停止
                - "fixed_frame": 按 translation_frame 指定的阅读框（0/1/2）直接按 3bp 一段翻译
            translation_frame: 在 "fixed_frame" 模式下使用，0/1/2
            stop_at_stop: 翻译时遇到终止密码子是否立刻停止（True）；
                          否则用 "X" 填充该位继续翻译
        """
        super().__init__(model_name, model_path)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        self.translation_mode = translation_mode
        self.translation_frame = int(translation_frame) % 3
        self.stop_at_stop = stop_at_stop

        # 模型相关对象
        self.tokenizer = None
        self.model = None

        # 标准遗传密码表（DNA 三联体 → 氨基酸）
        # 只处理 A/C/G/T，其余碱基统一翻译为 "X"
        self._codon_table = {
            # Phe
            "TTT": "F", "TTC": "F",
            # Leu
            "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
            # Ile / Met
            "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
            # Val
            "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
            # Ser
            "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S",
            # Pro
            "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
            # Thr
            "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
            # Ala
            "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
            # Tyr
            "TAT": "Y", "TAC": "Y",
            # His
            "CAT": "H", "CAC": "H",
            # Gln
            "CAA": "Q", "CAG": "Q",
            # Asn
            "AAT": "N", "AAC": "N",
            # Lys
            "AAA": "K", "AAG": "K",
            # Asp
            "GAT": "D", "GAC": "D",
            # Glu
            "GAA": "E", "GAG": "E",
            # Cys
            "TGT": "C", "TGC": "C",
            # Trp
            "TGG": "W",
            # Arg
            "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R", "AGG": "R",
            # Gly
            "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
            # Stop
            "TAA": "*", "TAG": "*", "TGA": "*",
        }

        self._load_model()

    def _load_model(self):
        print(f"[ESM2Model] Loading model from {self.model_path} ...")
        
        # 加载 tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        except Exception as e:
            print(f"[ESM2Model] AutoTokenizer failed ({e}), trying EsmTokenizer...")
            try:
                self.tokenizer = EsmTokenizer.from_pretrained(self.model_path)
            except Exception as e2:
                raise ImportError(f"[ESM2Model] Failed to load tokenizer from {self.model_path}: {e2}")

        # 加载模型 (带 LM head)
        # 优先尝试 AutoModelForMaskedLM 以支持 custom code (FastEsm)
        try:
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_path, trust_remote_code=True)
        except Exception as e:
            print(f"[ESM2Model] AutoModelForMaskedLM failed ({e}), trying EsmForMaskedLM...")
            try:
                self.model = EsmForMaskedLM.from_pretrained(self.model_path)
            except Exception as e2:
                print(f"[ESM2Model] EsmForMaskedLM failed ({e2}), trying EsmModel (no PLL support)...")
                self.model = EsmModel.from_pretrained(self.model_path)

        self.model = self.model.to(self.device).eval()

        # 获取最大长度
        self.model_max_len = (
            getattr(self.model.config, "max_position_embeddings", None)
            or getattr(self.tokenizer, "model_max_length", None)
            or 1024
        )
        
        print(
            f"[ESM2Model] loaded '{self.model_name}' on {self.device}, "
            f"max_len={self.model_max_len}"
        )

    # ---------- DNA → AA 翻译相关 ----------
    @staticmethod
    def _clean_dna(seq: str) -> str:
        """统一大小写并只保留 A/C/G/T/N，U 替换为 T。"""
        s = seq.strip().upper().replace("U", "T")
        out = []
        for ch in s:
            if ch in ("A", "C", "G", "T", "N"):
                out.append(ch)
        return "".join(out)

    @staticmethod
    def _revcomp(seq: str) -> str:
        """DNA 反向互补。"""
        tbl = str.maketrans(
            "ACGTRYMKBDHVNacgtrymkbdhvn",
            "TGCAYRKMVHDBNtgcayrkmvhdbn",
        )
        return seq.translate(tbl)[::-1]

    def _translate_dna_to_protein(self, dna_seq: str) -> str:
        """
        将 DNA 序列翻译为氨基酸序列（单条序列版）。
        """
        dna = self._clean_dna(dna_seq)
        if len(dna) < 3:
            return "X"  # 太短

        if self.translation_mode == "first_orf":
            start = dna.find("ATG")
            if start == -1:
                start = 0
            frame = start % 3
            dna = dna[start:]
        else:  # "fixed_frame"
            frame = self.translation_frame
            dna = dna[frame:]

        aa_list: List[str] = []
        for i in range(0, len(dna) - 2, 3):
            codon = dna[i: i + 3]
            if len(codon) < 3:
                break
            aa = self._codon_table.get(codon, "X")
            if aa == "*":  # 终止密码子
                if self.stop_at_stop:
                    break
                else:
                    aa = "X"
            aa_list.append(aa)

        if not aa_list:
            return "X"
        return "".join(aa_list)

    # ---------- PLL 评分 ----------
    def score_sequences(self, sequences: List[str], batch_size: int = 256) -> List[float]:
        """
        对序列进行 PLL (Pseudo-Log-Likelihood) 评分。
        DNA -> AA -> Masked LM Scoring
        """
        # Check if model supports Masked LM (has logits)
        # EsmForMaskedLM or FastEsmForMaskedLM
        if not hasattr(self.model, "config"):
             raise RuntimeError("[ESM2Model] Model has no config.")
             
        # 简单检查是否是 MaskedLM 模型 (通常会有 vocab_size 输出)
        # 或者直接检查类名
        class_name = self.model.__class__.__name__
        if "MaskedLM" not in class_name and not isinstance(self.model, EsmForMaskedLM):
             # 也有可能 custom model 不叫 MaskedLM 但有能力
             pass 

        all_scores: List[float] = []
        mask_batch_size = batch_size

        with torch.no_grad():
            for seq in tqdm(sequences, desc="Scoring sequences (ESM2)"):
                aa_seq = self._translate_dna_to_protein(seq)
                score = self._score_single_sequence(aa_seq, mask_batch_size)
                all_scores.append(score)

        return all_scores

    def _score_single_sequence(self, aa_sequence: str, mask_batch_size: int = 256) -> float:
        if len(aa_sequence) == 0:
            return 0.0

        # Tokenize
        inputs = self.tokenizer(
            aa_sequence, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.model_max_len
        )
        input_ids = inputs["input_ids"].to(self.device) # [1, T]
        attention_mask = inputs["attention_mask"].to(self.device) # [1, T]
        
        T = input_ids.size(1)
        token_ids = input_ids[0]
        
        # 找出有效位置 (排除 CLS, EOS, PAD)
        # ESM tokenizer: cls_token="<cls>", eos_token="<eos>", pad_token="<pad>"
        # 对应 id 一般为 0, 2, 1 (esm2_t33_650M_UR50D)
        # 也可以通过 tokenizer 属性获取
        
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            token_ids.tolist(), already_has_special_tokens=True
        ) # List[0/1]
        
        valid_positions = [i for i, is_spec in enumerate(special_tokens_mask) if is_spec == 0]

        if len(valid_positions) == 0:
            return 0.0

        total_logprob = 0.0
        total_count = 0
        
        mask_token_id = self.tokenizer.mask_token_id

        for start_idx in range(0, len(valid_positions), mask_batch_size):
            chunk_positions = valid_positions[start_idx : start_idx + mask_batch_size]
            chunk_size = len(chunk_positions)

            # 复制 batch
            masked_input_ids = input_ids.repeat(chunk_size, 1) # [B, T]
            masked_attention_mask = attention_mask.repeat(chunk_size, 1) # [B, T]
            
            true_tokens: List[int] = []

            # Apply masks
            for b, pos in enumerate(chunk_positions):
                true_token = int(token_ids[pos].item())
                true_tokens.append(true_token)
                masked_input_ids[b, pos] = mask_token_id
            
            # Forward
            outputs = self.model(
                input_ids=masked_input_ids, 
                attention_mask=masked_attention_mask
            )
            logits = outputs.logits # [B, T, V]

            # Gather logits at masked positions
            batch_indices = torch.arange(chunk_size, device=self.device)
            pos_indices = torch.tensor(chunk_positions, device=self.device, dtype=torch.long)
            
            pos_logits = logits[batch_indices, pos_indices, :] # [B, V]
            log_probs = torch.log_softmax(pos_logits, dim=-1) # [B, V]
            
            true_token_ids = torch.tensor(true_tokens, device=self.device, dtype=torch.long)
            token_log_probs = log_probs[batch_indices, true_token_ids] # [B]

            total_logprob += token_log_probs.sum().item()
            total_count += chunk_size

        avg_logprob = total_logprob / max(1, total_count)
        return float(avg_logprob)

    # ---------- Embedding 接口 ----------
    @torch.no_grad()
    def embed_sequences(
        self,
        sequences: List[str],
        batch_size: int = 16,
        pooling: Pooling = "mean",
        average_reverse_complement: bool = False,
        truncation: bool = True,
        return_numpy: bool = True,
    ):
        """
        直接输入 DNA 序列列表，返回每条序列的 ESM2 向量。
        """
        return self.get_embedding(
            sequences=sequences,
            batch_size=batch_size,
            pool=pooling,
            average_reverse_complement=average_reverse_complement,
            truncation=truncation,
            return_numpy=return_numpy,
        )

    @torch.no_grad()
    def get_embedding(
        self,
        sequences: Union[str, List[str]],
        layer_name: str = None, # Unused
        batch_size: int = 16,
        pool: Pooling = "mean",
        layer_index: int = -1, # -1: last hidden state
        average_reverse_complement: bool = False,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        
        if isinstance(sequences, str):
            seq_list = [sequences]
        else:
            seq_list = list(sequences)

        max_len = max_length or self.model_max_len
        
        def _aa_forward(aa_batch: List[str]) -> torch.Tensor:
            # Tokenize
            # 注意: ESM2 tokenizer 如果输入是 list，会自动 batch
            inputs = self.tokenizer(
                aa_batch, 
                return_tensors="pt", 
                padding=True, 
                truncation=truncation, 
                max_length=max_len
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            # Forward
            # 如果需要特定层，可能需要 output_hidden_states=True
            output_hidden_states = (layer_index != -1)
            
            outputs = self.model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states
            )
            
            if layer_index == -1:
                # EsmForMaskedLM 的 output[0] 是 logits, 但这里我们需要 hidden states
                # 如果是 EsmModel, output[0] 是 last_hidden_state
                # 如果是 EsmForMaskedLM, 并没有直接返回 last_hidden_state 在 output[0], 
                # 而是 output.logits. 我们需要获取 hidden_states[-1] 如果 output_hidden_states=True
                # 或者我们可以直接用 base model
                
                if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                     token_reps = outputs.hidden_states[-1]
                elif hasattr(self.model, "esm"): # EsmForMaskedLM has .esm (EsmModel)
                    # 重新 forward 一次 base model 比较浪费，最好 force output_hidden_states=True
                    # 但是 transformers 的 EsmForMaskedLM 默认 forward 不返回 last_hidden_state 除非 output_hidden_states=True
                    # 让我们确保 output_hidden_states=True
                    if not output_hidden_states:
                         outputs = self.model(
                            input_ids=input_ids, 
                            attention_mask=attention_mask,
                            output_hidden_states=True
                        )
                    token_reps = outputs.hidden_states[-1]
                else:
                    # EsmModel case
                    token_reps = outputs.last_hidden_state
            else:
                token_reps = outputs.hidden_states[layer_index] # 0 is embeddings

            # Pooling
            if pool == "cls":
                return token_reps[:, 0, :]
            
            # mean / max
            mask = attention_mask.bool()
            # Exclude special tokens (CLS, EOS) from mask if possible
            # 简单起见，假设 input_ids[0] 是 CLS, input_ids[-1] 是 EOS (对于 padded, EOS 在中间)
            # 用 special_tokens_mask 更稳妥
            special_mask = torch.zeros_like(mask)
            for i in range(input_ids.size(0)):
                 ids = input_ids[i].tolist()
                 specs = self.tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)
                 special_mask[i] = torch.tensor(specs, device=self.device, dtype=torch.bool)
            
            mask = mask & (~special_mask)
            
            if pool == "mean":
                m = mask.unsqueeze(-1).float()
                summed = (token_reps * m).sum(dim=1)
                denom = m.sum(dim=1).clamp_min(1.0)
                return summed / denom
            elif pool == "max":
                masked_reps = token_reps.masked_fill(~mask.unsqueeze(-1), float("-inf"))
                pooled = masked_reps.max(dim=1).values
                # handle all -inf
                inf_mask = torch.isinf(pooled).any(dim=1)
                if inf_mask.any():
                     pooled[inf_mask] = token_reps[inf_mask].max(dim=1).values
                return pooled
            else:
                raise ValueError(f"Unknown pool type: {pool}")

        outputs_list: List[torch.Tensor] = []
        for st in tqdm(range(0, len(seq_list), batch_size), desc="Getting ESM2 embedding"):
            batch_dna = seq_list[st: st + batch_size]
            
            aa_f = [self._translate_dna_to_protein(s) for s in batch_dna]
            vec_f = _aa_forward(aa_f)
            
            if average_reverse_complement:
                rc_dna = [self._revcomp(s) for s in batch_dna]
                aa_r = [self._translate_dna_to_protein(s) for s in rc_dna]
                vec_r = _aa_forward(aa_r)
                vec = 0.5 * (vec_f + vec_r)
            else:
                vec = vec_f
            
            outputs_list.append(vec.detach().cpu())

        if outputs_list:
            out = torch.cat(outputs_list, dim=0)
        else:
            out = torch.empty(0, 0)
            
        return out.numpy() if return_numpy else out

if __name__ == "__main__":
    MODEL_PATH = "../../model_weight/esm2_t6_8M_UR50D"
    MODEL_NAME = "esm2_t6_8M_UR50D"
    
    try:
        m = ESM2Model(
            model_name=MODEL_NAME,
            model_path=MODEL_PATH,
            device=None,
            translation_mode="first_orf"
        )
        
        dna_list = [
            "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG",
            "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATTCGTTAGC",
            "AACCC"
        ]
        
        print("\nTesting Embeddings...")
        embs = m.embed_sequences(dna_list, batch_size=2, pooling="mean")
        print("Embedding shape:", embs.shape)
        
        print("\nTesting PLL Scoring...")
        scores = m.score_sequences(dna_list, batch_size=2)
        print("Scores:", scores)
        
    except Exception as e:
        print(f"Test failed: {e}")


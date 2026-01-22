# models/gpn_brassicales_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import re
from typing import List, Optional, Union, Literal

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from tqdm import tqdm

# 关键：注册 GPN 自定义架构到 Transformers（GPNRoFormer）
import gpn.model

from .base_model import BaseModel

Pooling = Literal["mean", "max", "cls"]


def _call_or_value(obj) -> Optional[int]:
    """
    将 tokenizer 的 *_token_id 兼容为整数。
    可能是属性（int）也可能是可调用（返回 int）。
    """
    if obj is None:
        return None
    return int(obj() if callable(obj) else obj)


class GPNBrassicalesModel(BaseModel):
    """
    GPN-Brassicales 适配器：
      - 输入是 DNA 序列（A/C/G/T/N）
      - score_sequences: 伪对数似然（PLL），仅对 A/C/G/T 位计分；N/未知不计分
      - get_embedding: 对 last_hidden_state 做池化（mean/max/cls）
    """

    def __init__(
        self,
        model_name: str,
        model_path: str = "songlab/gpn-brassicales",
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        load_mlm_head: bool = True,
    ):
        """
        Args:
            model_name: 自定义名称
            model_path: HF 或本地路径（如 "songlab/gpn-brassicales"）
            device:     "cuda" | "cpu" | torch.device
            dtype:      torch.float16 / torch.bfloat16 / None
            load_mlm_head: 是否加载 MLM 头（score_sequences 需要）
        """
        super().__init__(model_name=model_name, model_path=model_path)

        self.device = torch.device(device) if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.dtype = dtype

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._setup_token_ids()

        # Backbone（用于 embedding）
        self.backbone = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        if self.dtype is not None:
            self.backbone = self.backbone.to(self.dtype)
        self.backbone.to(self.device).eval()

        # MLM 头（用于 PLL 评分）
        self.mlm: Optional[AutoModelForMaskedLM] = None
        if load_mlm_head:
            self.mlm = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True)
            if self.dtype is not None:
                self.mlm = self.mlm.to(self.dtype)
            self.mlm.to(self.device).eval()

        # 模型最大长度（确保是合理的整数）
        max_pos_emb = getattr(self.backbone.config, "max_position_embeddings", None)
        tokenizer_max_len = getattr(self.tokenizer, "model_max_length", None)
        
        # 尝试获取合理的最大长度值
        candidate_values = [max_pos_emb, tokenizer_max_len]
        self.model_max_len = 512  # 默认值
        
        for val in candidate_values:
            if val is not None:
                try:
                    val_int = int(val)
                    # 确保值在合理范围内（512 到 32768）
                    if 512 <= val_int <= 32768:
                        self.model_max_len = val_int
                        break
                    elif val_int > 32768:
                        # 如果值太大，使用一个合理的上限
                        self.model_max_len = 32768
                        break
                except (ValueError, TypeError):
                    continue

        print(f"[GPNBrassicalesModel] loaded on {self.device}, "
              f"mlm_head={load_mlm_head}, model_max_len={self.model_max_len}, "
              f"valid_base_ids={self._valid_base_ids}")

    # --------- 安全 token id 处理（兼容无 'N' 的 tokenizer） ---------
    def _setup_token_ids(self) -> None:
        # 获取 vocab
        vocab_obj = getattr(self.tokenizer, "vocab", None)
        if vocab_obj is None:
            # 尝试从 get_vocab() 方法获取
            if hasattr(self.tokenizer, "get_vocab"):
                vocab_obj = self.tokenizer.get_vocab()
        
        # 保存 vocab_obj 以便后续使用
        self._vocab_obj = vocab_obj
        
        if vocab_obj is None:
            vocab_list: List[str] = []
        elif isinstance(vocab_obj, dict):
            vocab_list = list(vocab_obj.keys())
        elif isinstance(vocab_obj, (list, tuple)):
            vocab_list = list(vocab_obj)
        else:
            vocab_list = list(vocab_obj)

        self._vocab_list: List[str] = vocab_list

        self._unk_id = _call_or_value(getattr(self.tokenizer, "unk_token_id", None))
        self._pad_id = _call_or_value(getattr(self.tokenizer, "pad_token_id", None))
        self._mask_id = _call_or_value(getattr(self.tokenizer, "mask_token_id", None))
        if self._mask_id is None:
            raise RuntimeError("Tokenizer does not provide mask_token_id (property or method).")

        def _safe_id(tok: str) -> int:
            # 尝试从 vocab 中查找
            if isinstance(vocab_obj, dict) and tok in vocab_obj:
                return vocab_obj[tok]
            if tok in self._vocab_list:
                return self._vocab_list.index(tok)
            if self._unk_id is not None:
                return int(self._unk_id)
            if self._pad_id is not None:
                return int(self._pad_id)
            return 0  # 最后兜底

        # 构建核苷酸映射（不要求有 'N'）
        self._nuc_to_id = {
            "A": _safe_id("A"),
            "C": _safe_id("C"),
            "G": _safe_id("G"),
            "T": _safe_id("T"),
            "N": _safe_id("N"),  # 若 vocab 无 'N'，将回退到 unk/pad/0
        }

        # 仅当 A/C/G/T 真正存在于 vocab 时，才将其纳入"可计分 token 集合"
        # 参考 basic_example.ipynb: tokenizer.get_vocab()[nc]
        self._valid_base_ids = set()
        for base in ("A", "C", "G", "T"):
            base_id = None
            # 优先使用 get_vocab() 方法（与 basic_example.ipynb 一致）
            if hasattr(self.tokenizer, "get_vocab"):
                vocab_dict = self.tokenizer.get_vocab()
                if isinstance(vocab_dict, dict) and base in vocab_dict:
                    base_id = vocab_dict[base]
            # 如果没有，尝试从 vocab_obj 获取
            if base_id is None and isinstance(vocab_obj, dict) and base in vocab_obj:
                base_id = vocab_obj[base]
            # 如果还没有，尝试从 vocab_list 获取
            if base_id is None and base in self._vocab_list:
                base_id = self._vocab_list.index(base)
            # 如果找到了，添加到 valid_base_ids
            if base_id is not None:
                self._valid_base_ids.add(int(base_id))

    # -------------------- 公共 API：评分（PLL） --------------------
    @torch.inference_mode()
    def score_sequences(self, sequences: List[str], batch_size: int = 256) -> List[float]:
        """
        伪对数似然（PLL）评分：
          - 对每条序列：逐位 mask，取真实碱基（A/C/G/T）的 log-prob，按位平均。
          - 'N' 或未知字符映射为 unk/pad/0，不参与计分。
        Returns:
          List[float]：每条序列的平均 log-prob（自然对数）
        """
        if self.mlm is None:
            raise RuntimeError("MLM head is not loaded (load_mlm_head=False).")

        all_scores = []
        for seq in tqdm(sequences, desc="Scoring sequences"):
            score = self._score_single_sequence(seq, batch_size)
            all_scores.append(score)

        return all_scores

    def _score_single_sequence(self, sequence: str, mask_batch_size: int = 256) -> float:
        """
        对单条序列进行 PLL 评分
        """
        # 1. Tokenize 序列
        dna = sequence.strip().upper()
        if not re.fullmatch(r"[ACGTN]+", dna):
            raise ValueError(
                f"Sequence contains invalid characters: {sequence!r}. Only A/C/G/T/N are allowed."
            )

        # 确保 max_length 在合理范围内
        effective_max_len = self.model_max_len
        if effective_max_len > 32768:
            effective_max_len = 32768
        
        enc = self.tokenizer(
            dna,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=effective_max_len,
            return_attention_mask=True,
        )
        input_ids = enc["input_ids"].to(self.device).squeeze(0)  # (L,)
        attn_mask = enc.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device).squeeze(0)
        else:
            attn_mask = torch.ones_like(input_ids)

        seq_len = input_ids.shape[0]

        # 2. 找出有效位置（排除特殊 token 和 padding）
        # 注意：GPN tokenizer 可能使用 BPE，所以我们对所有非特殊 token 的位置都计分
        # 参考 basic_example.ipynb，他们对所有位置都计分（只要不是特殊 token）
        valid_positions = []
        input_ids_list = input_ids.tolist()

        pad_id = self._pad_id
        cls_id = getattr(self.tokenizer, "cls_token_id", None)
        sep_id = getattr(self.tokenizer, "sep_token_id", None)
        mask_id = self._mask_id

        for i in range(seq_len):
            if attn_mask[i].item() == 0:
                continue

            token_id = input_ids_list[i]

            # 跳过特殊 token（PAD, CLS, SEP, MASK）
            is_special = False
            if pad_id is not None and token_id == pad_id:
                is_special = True
            elif cls_id is not None and token_id == cls_id:
                is_special = True
            elif sep_id is not None and token_id == sep_id:
                is_special = True
            elif mask_id is not None and token_id == mask_id:
                is_special = True

            # 对所有非特殊 token 的位置计分
            if not is_special:
                valid_positions.append(i)

        if len(valid_positions) == 0:
            # 打印调试信息
            print(f"DEBUG: No valid positions found!")
            print(f"  Sequence length: {seq_len}")
            print(f"  First 20 token IDs: {input_ids_list[:20]}")
            print(f"  Pad/CLS/SEP/Mask IDs: {pad_id}/{cls_id}/{sep_id}/{mask_id}")
            # 尝试解码 token 看看实际内容
            try:
                if hasattr(self.tokenizer, "decode"):
                    print(f"  First 20 decoded tokens: {[self.tokenizer.decode([tid]) for tid in input_ids_list[:20]]}")
            except Exception as e:
                print(f"  Failed to decode tokens: {e}")
            return float("nan")

        # 3. 分块遮罩并计算 log-probability
        total_logprob = 0.0
        total_count = 0

        for start_idx in range(0, len(valid_positions), mask_batch_size):
            chunk_positions = valid_positions[start_idx:start_idx + mask_batch_size]
            chunk_size = len(chunk_positions)

            # 复制 input_ids 用于遮罩
            masked_ids = input_ids.unsqueeze(0).repeat(chunk_size, 1)  # (B, L)
            masked_attn = attn_mask.unsqueeze(0).repeat(chunk_size, 1)  # (B, L)
            true_tokens = []

            # 遮罩对应位置
            for b, pos in enumerate(chunk_positions):
                true_token = int(input_ids[pos].item())
                true_tokens.append(true_token)
                masked_ids[b, pos] = self._mask_id

            # 模型前向传播（参考 basic_example.ipynb 的方式）
            # 在 basic_example.ipynb 中：all_logits = model_for_mlm(input_ids=input_ids).logits
            all_logits = self.mlm(input_ids=masked_ids).logits  # (B, L, V)

            # 获取遮罩位置的 logits（参考 basic_example.ipynb：all_logits[0, pos, ...]）
            batch_indices = torch.arange(chunk_size, device=self.device)
            pos_indices = torch.tensor(chunk_positions, device=self.device, dtype=torch.long)
            pos_logits = all_logits[batch_indices, pos_indices, :]  # (B, V)

            # 计算 log-probability（参考 basic_example.ipynb：使用 softmax 计算概率）
            # 在 basic_example.ipynb 中：probs = torch.nn.functional.softmax(logits, dim=0).numpy()
            # 我们使用 log_softmax 来获取 log-probability
            log_probs = F.log_softmax(pos_logits, dim=-1)  # (B, V)

            # 获取真实 token 的 log-probability
            true_token_ids = torch.tensor(true_tokens, device=self.device, dtype=torch.long)
            token_log_probs = log_probs[batch_indices, true_token_ids]  # (B,)

            total_logprob += token_log_probs.sum().item()
            total_count += chunk_size

            # 清理显存
            del all_logits, pos_logits, log_probs, token_log_probs
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # 计算平均 log-probability
        avg_logprob = total_logprob / max(1, total_count)
        return float(avg_logprob)

    # -------------------- 公共 API：Embedding --------------------
    @torch.inference_mode()
    def get_embedding(
        self,
        sequences: Union[str, List[str]],
        layer_name: str = None,  # 为了兼容 BaseModel 接口，但实际使用 pool 和 layer_index
        batch_size: int = 64,
        pool: Pooling = "mean",
        layer_index: int = -1,
        exclude_special: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_numpy: bool = True,
    ) -> Union[np.ndarray, List[List[float]]]:
        """
        提取序列级 embedding。

        Args:
            sequences: 输入序列列表
            layer_name: 兼容 BaseModel 接口（未使用）
            batch_size: 前向批大小
            pool: 池化方式 {"mean", "max", "cls"}
            layer_index: 取指定隐藏层（-1 为最后一层）
            exclude_special: mean/max 时是否排除特殊 token
            truncation: 是否截断
            max_length: 最大长度
            return_numpy: 是否返回 numpy.ndarray（否则返回 Python list）

        Returns:
            - 如果 return_numpy=True: (N, H) numpy array
            - 否则: List[List[float]]，每个序列一个向量
        """
        if isinstance(sequences, str):
            seq_list = [sequences]
        else:
            seq_list = list(sequences)

        all_vecs: List[np.ndarray] = []

        for start in tqdm(range(0, len(seq_list), batch_size), desc="Getting embedding"):
            batch = seq_list[start:start + batch_size]

            # Tokenize - 确保 max_length 在合理范围内
            effective_max_len = max_length or self.model_max_len
            # 限制最大长度在 32768 以内，避免溢出
            if effective_max_len > 32768:
                effective_max_len = 32768
            
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=truncation,
                max_length=effective_max_len,
                return_attention_mask=True,
            )
            input_ids = enc["input_ids"].to(self.device)  # (B, L)
            attn_mask = enc.get("attention_mask")
            if attn_mask is not None:
                attn_mask = attn_mask.to(self.device)
            else:
                attn_mask = torch.ones_like(input_ids)

            # 前向传播
            if layer_index == -1:
                outputs = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    return_dict=True,
                )
                hidden = outputs.last_hidden_state  # (B, L, H)
            else:
                outputs = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden = outputs.hidden_states[layer_index]  # (B, L, H)

            # 池化
            if pool == "cls":
                vecs = hidden[:, 0, :]  # (B, H)
            elif pool == "mean":
                if exclude_special:
                    # 排除特殊 token
                    valid = attn_mask.bool()  # (B, L)
                    spec_masks = []
                    for ids in input_ids:
                        spec = self.tokenizer.get_special_tokens_mask(
                            ids.tolist(), already_has_special_tokens=True
                        )
                        spec_masks.append(torch.tensor(spec, device=self.device, dtype=torch.bool))
                    spec_mask = torch.stack(spec_masks, dim=0)
                    valid = valid & (~spec_mask)

                    m = valid.unsqueeze(-1).to(hidden.dtype)  # (B, L, 1)
                    summed = (hidden * m).sum(dim=1)  # (B, H)
                    denom = m.sum(dim=1).clamp_min(1.0)  # (B, 1)
                    vecs = summed / denom
                else:
                    vecs = hidden.mean(dim=1)  # (B, H)
            elif pool == "max":
                if exclude_special:
                    valid = attn_mask.bool()  # (B, L)
                    spec_masks = []
                    for ids in input_ids:
                        spec = self.tokenizer.get_special_tokens_mask(
                            ids.tolist(), already_has_special_tokens=True
                        )
                        spec_masks.append(torch.tensor(spec, device=self.device, dtype=torch.bool))
                    spec_mask = torch.stack(spec_masks, dim=0)
                    valid = valid & (~spec_mask)

                    masked = hidden.masked_fill(~valid.unsqueeze(-1), float("-inf"))
                    vecs = masked.max(dim=1).values  # (B, H)
                    # 处理全为 -inf 的情况
                    inf_mask = torch.isinf(vecs).any(dim=1)
                    if inf_mask.any():
                        vecs[inf_mask] = hidden[inf_mask].max(dim=1).values
                else:
                    vecs, _ = hidden.max(dim=1)  # (B, H)
            else:
                raise ValueError(f"Unknown pool: {pool}")

            all_vecs.append(vecs.detach().cpu().numpy())

            del outputs, hidden, vecs
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        embs = np.concatenate(all_vecs, axis=0)  # (N, H)

        if return_numpy:
            return embs
        else:
            return embs.tolist()


# # ---------- 自测 ----------
# if __name__ == "__main__":
#     MODEL_PATH = "/mnt/s3mount/model_weight/gpn-brassicales"
#     # 或本地路径: MODEL_PATH = "/mnt/s3mount/model_weight/gpn-brassicales"

#     m = GPNBrassicalesModel(
#         model_name="gpn-brassicales",
#         model_path=MODEL_PATH,
#         device=None,          # 自动选 GPU/CPU
#         load_mlm_head=True,   # 需要 PLL 评分时设为 True
#     )

#     # 测试序列（来自 basic_example.ipynb）
#     seq = "CGGGTTAAAAATCTAGTTGTTATTATTAAAGGAAATAAAATATCCTCATAAAACAATTTGTTGTAATCTATCTTTGGGCTAATGTTCTTATCCTACAAGACGAACCCTGACCGTATTCGTCGTAGAAAAAAAATTGCTTCGATCCCATCATTGAGTTCAATAATCGGCGCACAAAGGCCGATTCATAAAAACTCTAGGCCCATTAAAGTAAAGCCCATTCTCAACCCTATCCAGTCTCCCTGTATATATATATTTACGACACCAACCCAGCGTTGATATTTAATTTTCTTCAGTCAGAGATTTCGAAACCCTAGTCGATTTCGAGATCCAACTAACTCTGCTCCTTATCTCAGGTAAAATTCTCGCTCGAGAACTCAATTGCTTATCCAAAGTTCCAACTGAAGATGCTTTCCTACTGAATCTTAGGTTAATGTTTTGGATTTGGAATCTTACCCGAAATTTCTCTGCAGCTTGTTGAATTTGCGAAGTATGGGAGACGCTAGAGACAACGAAGCCTACGAGGAGGAGCTCTTGGACTATGAAGAAGAAGACGAGAAGGTCCCAGATTCTGGAAACAAAGTTAACGGCGAAGCTGTGAAAAAGTGAGTTTTATGGTTTCCTCGATATGTTTCATGTATACTACTGTGTGTTTAAATTTGTCGATTCTTAGATTACTACTTGATAACAAGTAGCAGTATGT"
#     seqs = [seq, "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATTCGTTAGC"]

#     # 测试 embedding 提取
#     embs = m.get_embedding(seqs, pool="mean", batch_size=32, return_numpy=True)
#     print("Embedding shape:", embs.shape)  # (2, hidden_size)
#     print(embs)

#     # 测试 PLL 评分
#     scores = m.score_sequences(seqs, batch_size=128)
#     print("PLL scores:", scores)


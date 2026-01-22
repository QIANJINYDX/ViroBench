# models/nucleotide_transformer_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, inspect
from typing import List, Optional, Literal, Dict, Any, Union

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM

from .base_model import BaseModel

from tqdm import tqdm

Pooling = Literal["mean", "max", "cls"]

def _revcomp(seq: str) -> str:
    tbl = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(tbl)[::-1]


class NucleotideTransformerModel(BaseModel):
    """
    Nucleotide Transformer 适配器 (e.g., nucleotide-transformer-2.5b-multi-species)

    - embed_sequences: 提取最后一层隐藏态并池化（mean/max/cls）
    - score_sequences: 伪对数似然 PLL（逐位 [MASK]，需要 MLM 头）
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        hf_home: Optional[str] = None,
        device: Optional[str] = None,
        device_map: Optional[str] = None,        # 如 "auto"
        torch_dtype: Optional[torch.dtype] = None,  # 可设 torch.bfloat16/float16
        trust_remote_code: bool = False,
    ):
        super().__init__(model_name, model_path)
        self.hf_home = hf_home
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self._load_model()

    # ---------- 加载 ----------
    def _load_model(self):
        if self.hf_home:
            os.environ["HF_HOME"] = self.hf_home

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=self.trust_remote_code
        )

        common_kwargs: Dict[str, Any] = dict(trust_remote_code=self.trust_remote_code)
        if self.torch_dtype is not None:
            common_kwargs["torch_dtype"] = self.torch_dtype
        if self.device_map is not None:
            common_kwargs["device_map"] = self.device_map

        # 使用带 MLM 头的权重（既能取 hidden_states 也能做 PLL）
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_path, **common_kwargs)

        if self.device_map is None:
            self.model.to(self.device)
        self.model.eval()

        # 探测 forward 支持参数
        try:
            sig = inspect.signature(self.model.forward)
            self._accepts_attn = "attention_mask" in sig.parameters
            self._accepts_enc_attn = "encoder_attention_mask" in sig.parameters
            self._accepts_ret = "return_dict" in sig.parameters
            self._accepts_hidden = "output_hidden_states" in sig.parameters
        except Exception:
            self._accepts_attn = True
            self._accepts_enc_attn = False
            self._accepts_ret = True
            self._accepts_hidden = True

        # 记录 mask_id（用于 PLL）
        self.mask_id = self.tokenizer.mask_token_id
        if self.mask_id is None:
            cand = self.tokenizer.mask_token or "[MASK]"
            mid = self.tokenizer.convert_tokens_to_ids(cand)
            self.mask_id = mid if isinstance(mid, int) and mid >= 0 else None

        self.model_max_len = getattr(self.tokenizer, "model_max_length", None)
        print(f"[NT] loaded on {self.device} | device_map={self.device_map} | "
              f"dtype={self.torch_dtype} | max_len={self.model_max_len} | "
              f"accepts(attn={self._accepts_attn}, enc_attn={self._accepts_enc_attn}, "
              f"return_dict={self._accepts_ret}, hidden={self._accepts_hidden}) | mask_id={self.mask_id}")

    def score_sequences(self, sequences: List[str], batch_size: int = 256) -> List[float]:
        """
        对序列进行评分
        
        Args:
            sequences: 序列列表
            batch_size: 批处理大小
            
        Returns:
            每个序列的 log-likelihood 得分
        """
        all_scores = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc="Scoring"):
                batch_seqs = sequences[i:i + batch_size]
                batch_scores = self._score_batch(batch_seqs)
                all_scores.extend(batch_scores)
        
        return all_scores
    
    def _score_batch(self, sequences: List[str]) -> List[float]:
        """对单个批次评分"""
        # 1. 分词和预处理
        input_ids, _ = self._prepare_batch(sequences)
        
        # 创建 attention_mask（参考官方实现）
        attention_mask = (input_ids != self.tokenizer.pad_token_id).to(input_ids.device)

        # 2. 模型前向传播（添加 attention_mask 避免警告）
        try:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_attention_mask=attention_mask,
            )
        except TypeError:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        logits = outputs.logits
        logits = logits.float()

        # 3. 直接对 logits 求平均作为得分（仅对有效位置）
        # 使用 attention_mask 掩码掉 padding 位置
        attn_mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
        masked_logits = logits * attn_mask_expanded  # [B, L, V]
        # 对每个序列，计算有效位置的平均 logits
        scores = []
        for i in range(logits.shape[0]):
            valid_mask = attention_mask[i].bool()
            if valid_mask.any():
                valid_logits = logits[i][valid_mask]  # [L_valid, V]
                score = float(valid_logits.mean().item())
            else:
                score = 0.0
            scores.append(score)

        return scores
    
    def _prepare_batch(self, sequences: List[str]):
        """分词和批处理"""
        seq_lengths = [len(seq) for seq in sequences]
        
        # 使用 tokenizer 编码
        all_token_ids = []
        for seq in sequences:
            # tokens_ids = self.tokenizer.batch_encode_plus(sequences, return_tensors="pt", padding="max_length", max_length = max_length)["input_ids"]

            token_ids = self.tokenizer.encode(seq, add_special_tokens=False)
            all_token_ids.append(token_ids)
        # Padding 到相同长度
        max_length = min(max(len(ids) for ids in all_token_ids),1000)
        padded_ids = []
        
        for token_ids in all_token_ids:
            if len(token_ids) < max_length:
                padded = token_ids + [self.tokenizer.pad_token_id] * (max_length - len(token_ids))
            else:
                padded = token_ids[:max_length]
            padded_ids.append(padded)
        
        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=self.model.device)
        return input_ids, seq_lengths
    
    def _compute_log_probabilities(self, logits: torch.Tensor, input_ids: torch.Tensor):
        """计算 log probabilities"""
        # Softmax + log
        log_probs = torch.log_softmax(logits, dim=-1)
        log_probs = log_probs[:, :-1, :]  # 去掉最后一个位置
        target_ids = input_ids[:, 1:]     # 去掉第一个位置
        
        # 获取对应 token 的 log prob
        token_log_probs = torch.gather(
            log_probs, 2, target_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        return token_log_probs
    
    def _aggregate_scores(self, log_probs: torch.Tensor, seq_lengths: List[int]):
        """聚合序列得分"""
        log_probs_np = log_probs.float().cpu().numpy()
        scores = []
        
        for idx, seq_len in enumerate(seq_lengths):
            # 只计算实际序列部分，排除 padding
            valid_length = min(seq_len - 1, log_probs_np.shape[1])
            if valid_length > 0:
                valid_log_probs = log_probs_np[idx][:valid_length]
                score = float(np.mean(valid_log_probs))  # 平均 log-likelihood
            else:
                score = 0.0
            scores.append(score)
        
        return scores
    @torch.no_grad()
    def get_embedding(
        self,
        sequences: Union[str, List[str]],
        batch_size: int = 8,
        pool: Pooling = "mean",           # "cls" | "mean" | "max"
        exclude_special: bool = True,        # mean/max 时是否排除 [CLS]/[SEP]/PAD
        max_length: Optional[int] = None,    # None=用 tokenizer.model_max_length（NT 默认1000）
        return_numpy: bool = True,
    ):
        """
        取 Nucleotide Transformer 的序列向量，返回形状 (N, H)。
        要求：self.tokenizer = AutoTokenizer(...); self.model = AutoModelForMaskedLM(...)
        """
        # 统一为列表
        if isinstance(sequences, str):
            sequences = [sequences]
        else:
            sequences = list(sequences)

        # 设备
        try:
            dev = next(self.model.parameters()).device
        except StopIteration:
            dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 选择长度：未显式传入则用模型卡建议的最大长度（HF 模型卡同样示例）:contentReference[oaicite:1]{index=1}
        if max_length is None:
            max_length = getattr(self.tokenizer, "model_max_length", 1000)

        embs: List[np.ndarray] = []
        self.model.eval()

        for st in tqdm(range(0, len(sequences), batch_size), desc="Getting embedding"):
            batch = sequences[st: st + batch_size]

            # 编码（按 max_length 右侧 padding/truncation，参考官方实现）
            # 注意：参考代码没有指定 add_special_tokens，使用默认值
            enc = self.tokenizer.batch_encode_plus(
                batch,
                return_tensors="pt",
                padding="max_length",
                max_length=max_length,
            )
            input_ids = enc["input_ids"].to(dev)                       # [B, L]
            # 官方示例用 "pad对比" 作为 mask
            attn_mask = (input_ids != self.tokenizer.pad_token_id).to(dev)  # [B, L]

            # 前向传播（参考官方实现）
            out = None
            try:
                out = self.model(
                    input_ids,
                    attention_mask=attn_mask,
                    encoder_attention_mask=attn_mask,  # NT 某些权重要求该参数
                    output_hidden_states=True,
                )
            except TypeError:
                # 兼容老版本：骨干在 .esm
                out = self.model.esm(
                    input_ids,
                    attention_mask=attn_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )

            # 取最后一层 token 级表示（参考官方实现）
            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                hidden = out.last_hidden_state                        # [B, L, H]
            elif isinstance(out, dict) and "hidden_states" in out:
                hidden = out["hidden_states"][-1]                     # [B, L, H]
            else:
                raise RuntimeError("无法从模型输出中获取隐藏态（hidden states）。")

            # 池化（参考官方实现的方式）
            if pool == "cls":
                pooled = hidden[:, 0, :]  # [B, H]
            elif pool == "mean":
                # 参考官方实现：使用 attention_mask 加权平均
                # attention_mask: [B, L], hidden: [B, L, H]
                # 扩展 attention_mask 到 [B, L, 1] 以便广播
                attn_mask_expanded = torch.unsqueeze(attn_mask, dim=-1).float()  # [B, L, 1]
                hidden_float = hidden.float()  # 确保是 float 类型
                # 加权求和：sum(attention_mask * embeddings, axis=-2) / sum(attention_mask, axis=1)
                mask_sum = torch.sum(attn_mask.float(), dim=1, keepdim=True)  # [B, 1]
                # 避免除零：如果 mask_sum 为 0，则使用 1（虽然不应该发生）
                mask_sum = torch.clamp(mask_sum, min=1.0)
                pooled = torch.sum(attn_mask_expanded * hidden_float, dim=-2) / mask_sum  # [B, H]
            else:  # max pooling
                B, L, H = hidden.shape
                pooled_list = []
                for i in range(B):
                    valid = attn_mask[i].bool()  # 先用 PAD 掩掉
                    if exclude_special:
                        # 排除 [CLS]/[SEP]/PAD 等特殊符号
                        spec = self.tokenizer.get_special_tokens_mask(
                            input_ids[i].tolist(), already_has_special_tokens=True
                        )
                        spec_mask = torch.tensor(spec, dtype=torch.bool, device=dev)
                        valid = valid & (~spec_mask)
                    if not valid.any():
                        valid = attn_mask[i].bool()  # 兜底

                    tok_vecs = hidden[i][valid]     # [L_valid, H]
                    v = tok_vecs.max(dim=0)[0]
                    pooled_list.append(v)
                pooled = torch.stack(pooled_list, dim=0)  # [B, H]

            if return_numpy:
                embs.extend(pooled.float().cpu().numpy())
            else:
                embs.extend([pooled[j] for j in range(pooled.shape[0])])
            
            # print(embs[0].shape)

        return np.stack(embs, axis=0) if return_numpy else embs
# python -m models.nucleotide_transformer_model
# ---------- 自测 ----------
if __name__ == "__main__":
    MODEL_DIR = "/mnt/s3mount/model_weight/cur/nucleotide-transformer-2.5b-1000g"
    HF_HOME = "/mnt/s3mount/model_weight/cache"

    m = NucleotideTransformerModel(
        model_name="nt-2.5b-1000g",
        model_path=MODEL_DIR,
        hf_home=HF_HOME,
        device_map=None,                 # 如需分片可设为 "auto"
        torch_dtype=None,                # 可设为 torch.bfloat16
        trust_remote_code=False,
    )

    # 与你给的推理函数一致的两条序列
    seqs = ["ACct", "ATTTCTCTCTCTCTCTGAGATCGATCGATCGAT"]

    # 1) PLL 评分（逐位 [MASK]）
    pll = m.score_sequences(seqs, batch_size=128)
    embedding = m.get_embedding(seqs, batch_size=128)
    print("PLL scores:", pll)
    print("Embedding:", embedding)

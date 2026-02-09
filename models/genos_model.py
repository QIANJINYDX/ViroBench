# models/genos_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from typing import List, Optional, Sequence, Union, Literal
from typing import Tuple

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

from .base_model import BaseModel

from tqdm import tqdm
import math


class GenosModel(BaseModel):
    """
    Genos 适配器：用于生物序列的embedding提取和评分。
    - 使用 AutoModel 加载预训练的 Genos 模型
    - 支持提取各层的 hidden states 作为 embedding
    - 支持多种池化方法（mean, final, max, min）
    - 支持序列评分（基于 log-likelihood）

    Args:
        model_name: 逻辑名（e.g., "Genos-1.2B"）
        model_path: 本地权重目录（e.g., /data/model/Genos-1.2B）
        hf_home: 可选，显式设置 HF_HOME 缓存目录
        device: 'cuda:0' / 'cpu' / None(自动)
        use_flash_attention: 是否使用 flash_attention_2（需要环境支持）
        torch_dtype: 模型数据类型，默认 torch.bfloat16
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        hf_home: Optional[str] = None,
        device: Optional[str] = None,
        use_flash_attention: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(model_name, model_path)
        self.hf_home = hf_home
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_flash_attention = use_flash_attention
        self.torch_dtype = torch_dtype or torch.bfloat16
        self._load_model()

    def _load_model(self):
        if self.hf_home:
            os.environ["HF_HOME"] = self.hf_home
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        
        # 准备模型加载参数
        model_kwargs = {
            "output_hidden_states": True,
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": True,
        }
        
        # 如果支持且启用，添加 flash_attention
        if self.use_flash_attention:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            except Exception as e:
                print(f"[GenosModel] Warning: flash_attention_2 not available, using default: {e}")
        
        # 加载模型
        self.model = AutoModel.from_pretrained(self.model_path, **model_kwargs)
        
        # 移动到设备
        if self.device.startswith("cuda"):
            self.model = self.model.to(self.device)
        self.model.eval()

        # 检查是否需要/支持 attention_mask
        import inspect
        try:
            sig = inspect.signature(self.model.forward)
            self._supports_attn = "attention_mask" in sig.parameters
        except Exception:
            self._supports_attn = True  # 大多数模型都支持

        self.model_max_len = getattr(self.model.config, "max_position_embeddings", None) or \
                            getattr(self.tokenizer, "model_max_length", None) or 131072  # Genos默认128k
        print(f"[GenosModel] loaded on {self.device}, "
              f"model_max_len={self.model_max_len}, "
              f"supports_attn={self._supports_attn}, "
              f"torch_dtype={self.torch_dtype}")

    def score_sequences(self, sequences: List[str], batch_size: int = 256) -> List[float]:
        """
        对序列进行评分（基于语言模型的log-likelihood）
        
        Args:
            sequences: 序列列表
            batch_size: 批处理大小
            
        Returns:
            每个序列的平均 log-likelihood 得分
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
        input_ids, attention_mask, seq_lengths = self._prepare_batch(sequences)
        
        # 2. 模型前向传播
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        # 3. 获取最后一层的hidden states用于计算logits（如果模型有lm_head）
        # 注意：Genos可能没有直接的lm_head，这里尝试获取logits或使用hidden states的某种度量
        if hasattr(outputs, "logits") and outputs.logits is not None:
            logits = outputs.logits
            # 计算 log probabilities
            log_probs = self._compute_log_probabilities(logits, input_ids, attention_mask)
        else:
            # 如果没有logits，使用hidden states的某种聚合作为分数
            # 这里使用hidden states的norm作为替代指标
            hidden_states = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else outputs.last_hidden_state
            log_probs = self._compute_scores_from_hidden(hidden_states, attention_mask)
        
        # 4. 聚合得分
        scores = self._aggregate_scores(log_probs, seq_lengths, attention_mask)
        
        return scores
    
    def _prepare_batch(self, sequences: List[str]):
        """分词和批处理"""
        seq_lengths = [len(seq) for seq in sequences]
        
        # 使用 tokenizer 编码
        encoded = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.model_max_len,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        else:
            attention_mask = torch.ones_like(input_ids)
        
        return input_ids, attention_mask, seq_lengths
    
    def _compute_log_probabilities(
        self, 
        logits: torch.Tensor, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ):
        """计算 log probabilities（基于语言模型的方式）"""
        # logits: [B, T, V]
        # input_ids: [B, T]
        # attention_mask: [B, T]
        
        # 计算每个位置的log-prob
        log_probs = torch.log_softmax(logits, dim=-1)  # [B, T, V]
        
        # 获取对应token的log-prob
        # 对于语言模型，我们预测下一个token，所以需要shift
        # 但Genos可能是双向的，这里简化处理：计算当前位置的log-prob
        target_ids = input_ids.unsqueeze(-1)  # [B, T, 1]
        token_log_probs = torch.gather(log_probs, 2, target_ids).squeeze(-1)  # [B, T]
        
        # 应用attention mask
        token_log_probs = token_log_probs * attention_mask.float()
        
        return token_log_probs
    
    def _compute_scores_from_hidden(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ):
        """从hidden states计算分数（当没有logits时的替代方法）"""
        # 使用hidden states的某种聚合作为分数
        # 这里使用mean pooling后的norm作为替代指标
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # [B, T, 1]
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)  # [B, D]
        # 返回一个标量分数（这里简化处理，实际可能需要更复杂的计算）
        scores = torch.norm(pooled, dim=-1, keepdim=True)  # [B, 1]
        # 扩展为 [B, T] 形状以兼容聚合函数
        scores = scores.expand(-1, hidden_states.size(1)) * attention_mask.float()
        return scores
    
    def _aggregate_scores(
        self, 
        log_probs: torch.Tensor, 
        seq_lengths: List[int],
        attention_mask: torch.Tensor
    ):
        """聚合序列得分"""
        log_probs_np = log_probs.float().cpu().numpy()
        mask_np = attention_mask.cpu().numpy()
        scores = []
        
        for idx in range(log_probs_np.shape[0]):
            # 只计算实际序列部分，排除 padding
            valid_mask = mask_np[idx] > 0
            if valid_mask.sum() > 0:
                valid_log_probs = log_probs_np[idx][valid_mask]
                score = float(np.mean(valid_log_probs))  # 平均 log-likelihood
            else:
                score = 0.0
            scores.append(score)
        
        return scores

    @torch.no_grad()
    def get_embedding(
        self,
        sequences: Union[str, List[str]],
        batch_size: int = 64,
        pool: Literal["final", "mean", "max", "min"] = "mean",
        average_reverse_complement: bool = False,
        layer_index: int = -1,                   # -1: 最后一层
        add_special_tokens: bool = False,        # 是否添加特殊token
        max_length: Optional[int] = None,        # 最大长度限制
        return_numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        提取 (N, D) 序列向量。
        - pool="final": 取每条序列最后一个非 PAD token 的隐藏向量
        - pool="mean":  对非 PAD token 做均值池化
        - pool="max":   对非 PAD token 做最大池化
        - pool="min":   对非 PAD token 做最小池化
        - average_reverse_complement=True: 与反向互补结果做平均
        - layer_index: 选择隐藏层（-1表示最后一层）
        """

        device = next(self.model.parameters()).device
        tok = self.tokenizer

        # pad_id：若未设置，回退到 unk/eos/0
        pad_id = tok.pad_token_id
        if pad_id is None:
            pad_id = getattr(tok, "unk_token_id", None) or getattr(tok, "eos_token_id", None) or 0

        # 规范输入
        if isinstance(sequences, str):
            seq_list = [sequences]
        else:
            seq_list = list(sequences)

        def _revcomp(seq: str) -> str:
            """反向互补序列"""
            tbl = str.maketrans("ACGTRYMKBDHVNacgtrymkbdhvn",
                                "TGCAYRKMVHDBNtgcayrkmvhdbn")
            return seq.translate(tbl)[::-1]

        def _tokenize_batch(seqs: List[str], max_len: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """对批次进行tokenize"""
            # 使用tokenizer进行编码
            encoded = tok(
                seqs,
                padding=True,
                truncation=max_len is not None,
                max_length=max_len if max_len else self.model_max_len,
                add_special_tokens=add_special_tokens,
                return_tensors="pt"
            )
            
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)
            
            # 计算实际长度
            lengths = attention_mask.sum(dim=1).long()
            
            return input_ids, attention_mask, lengths

        def _forward_hidden(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            """前向传播获取hidden states"""
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # 获取指定层的hidden states
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                if layer_index != -1:
                    if layer_index < len(outputs.hidden_states):
                        return outputs.hidden_states[layer_index]
                    else:
                        raise ValueError(f"layer_index {layer_index} out of range (total {len(outputs.hidden_states)} layers)")
                return outputs.hidden_states[-1]
            elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                return outputs.last_hidden_state
            else:
                raise RuntimeError("无法从模型输出中获取隐藏表示。")

        def _pool_hidden(
            hidden: torch.Tensor, 
            attn_mask_local: torch.Tensor, 
            lengths: torch.Tensor
        ) -> torch.Tensor:
            """池化hidden states"""
            if pool == "final":
                idx = torch.clamp(lengths - 1, min=0)
                batch_idx = torch.arange(hidden.size(0), device=hidden.device)
                return hidden[batch_idx, idx, :]  # [B, D]
            elif pool == "mean":
                mask = attn_mask_local.unsqueeze(-1).to(hidden.dtype)  # [B, T, 1]
                summed = (hidden * mask).sum(dim=1)                    # [B, D]
                denom = mask.sum(dim=1).clamp_min(1.0)                 # [B, 1]
                return summed / denom
            elif pool == "max":
                mask = attn_mask_local.unsqueeze(-1).to(hidden.dtype)  # [B, T, 1]
                # 将padding位置设为很小的值，然后取max
                masked_hidden = hidden * mask + (1 - mask) * (-1e9)
                return masked_hidden.max(dim=1)[0]  # [B, D]
            elif pool == "min":
                mask = attn_mask_local.unsqueeze(-1).to(hidden.dtype)  # [B, T, 1]
                # 将padding位置设为很大的值，然后取min
                masked_hidden = hidden * mask + (1 - mask) * 1e9
                return masked_hidden.min(dim=1)[0]  # [B, D]
            else:
                raise ValueError(f"Unknown pool='{pool}', choose from ['final','mean','max','min'].")

        all_vecs: List[torch.Tensor] = []
        use_rc = bool(average_reverse_complement)

        for st in tqdm(range(0, len(seq_list), batch_size), desc="Getting embedding"):
            chunk = seq_list[st: st + batch_size]

            ids_f, am_f, len_f = _tokenize_batch(chunk, max_length)
            hid_f = _forward_hidden(ids_f, am_f)
            vec_f = _pool_hidden(hid_f, am_f, len_f)

            if use_rc:
                rc_chunk = [_revcomp(s) for s in chunk]
                ids_r, am_r, len_r = _tokenize_batch(rc_chunk, max_length)
                hid_r = _forward_hidden(ids_r, am_r)
                vec_r = _pool_hidden(hid_r, am_r, len_r)
                vec = 0.5 * (vec_f + vec_r)
            else:
                vec = vec_f

            all_vecs.append(vec.detach().to(torch.float32).cpu())

        out = torch.cat(all_vecs, dim=0) if all_vecs else torch.empty(0, 0)
        return out.numpy() if return_numpy else out
    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        batch_size: int = 8
    ) -> List[str]:
        """
        生成基因序列
        
        Args:
            prompts: 起始序列列表 (Prompt)
            max_new_tokens: 最大生成长度
            temperature: 采样温度
            do_sample: 是否采样 (False为贪婪搜索)
            
        Returns:
            包含完整序列（Prompt + Generated）的列表
        """
        results = []
        
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
            batch_prompts = prompts[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=self.model_max_len - max_new_tokens
            ).to(self.device)

            gen_config = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True
            }

            outputs = self.model.generate(**inputs, **gen_config)
            
            # 解码
            batch_results = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            results.extend(batch_results)
            
        return results
    @torch.no_grad()
    def calculate_ppl(self, sequences: List[str], batch_size: int = 8, stride: int = 512) -> float:
        """
        计算给定序列集合的平均困惑度 (Perplexity)
        
        Args:
            sequences: 待评估的序列列表
            batch_size: 处理批次大小
            stride: 窗口移动步长（用于处理超长序列，简单起见通常设为 max_len）
        
        Returns:
            所有序列的平均 PPL
        """
        nlls = [] # Negative Log Likelihoods
        total_tokens = 0

        # 为了准确计算，通常建议逐个或小Batch处理，并使用 labels 计算 Loss
        # 这里使用标准 HuggingFace CausalLM Loss 计算逻辑
        
        for i in tqdm(range(0, len(sequences), batch_size), desc="Calculating PPL"):
            batch_seqs = sequences[i:i + batch_size]
            
            encodings = self.tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.model_max_len
            ).to(self.device)

            input_ids = encodings.input_ids
            attention_mask = encodings.attention_mask
            
            # 标签就是输入ID，HuggingFace 模型内部会自动 shift logits
            target_ids = input_ids.clone()
            # 将 padding 部分的 label 设为 -100，这样计算 loss 时会忽略它们
            target_ids[attention_mask == 0] = -100

            outputs = self.model(input_ids, attention_mask=attention_mask, labels=target_ids)
            
            # outputs.loss 是这个 batch 的平均 CrossEntropy Loss
            # 我们需要还原回总 NLL 来加权平均
            # CrossEntropyLoss(reduction='mean') * valid_token_count
            
            # 计算该 batch 中有效的 token 数量 (非 padding)
            valid_tokens = (target_ids != -100).sum().item()
            
            # 如果 batch 全是 padding (异常情况)，跳过
            if valid_tokens == 0:
                continue

            # 还原总 loss
            batch_nll = outputs.loss.item() * valid_tokens
            
            nlls.append(batch_nll)
            total_tokens += valid_tokens

        if total_tokens == 0:
            return float('nan')

        avg_nll = sum(nlls) / total_tokens
        ppl = math.exp(avg_nll)
        return ppl


# # ---------- 自测 ----------
if __name__ == "__main__":
    # Genos模型路径
    MODEL_DIR = "../../model_weight/Genos-1.2B"  # 替换为实际路径
    HF_HOME = "../../model"  # 可选

    m = GenosModel(
        model_name="Genos-1.2B",
        model_path=MODEL_DIR,
        hf_home=HF_HOME,
        device=None,   # 自动
        use_flash_attention=False,  # 根据环境设置
    )

    # 测试embedding提取
    seqs = ["ACGT" * 128, "AAAAACCCCGGGGTTTT"]
    emb = m.get_embedding(seqs, pool="mean", batch_size=32)
    print(f"Embedding shape: {emb.shape}")

    # 测试评分（如果模型支持）
    # scores = m.score_sequences(seqs, batch_size=32)
    # for s, v in zip(seqs, scores):
    #     print(f"{s[:20]}... -> score = {v:.6f}")
    
    # 1. 测试生成
    prompts = ["ATG", "CCG"]
    generated = m.generate(prompts, max_new_tokens=10)
    print("Generated:", generated)
    
    # 2. 测试 PPL
    ppl = m.calculate_ppl(generated)
    print(f"PPL: {ppl}")

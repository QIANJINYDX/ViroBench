from __future__ import annotations
import os
import json
import sys

import collections
import collections.abc
# 手动将 abc 下的类挂载回 collections，骗过 attrdict
for name in ['Mapping', 'MutableMapping', 'Sequence']:
    if not hasattr(collections, name):
        setattr(collections, name, getattr(collections.abc, name))

ABS_RNABERT_PATH = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model/RNABERT"
if os.path.exists(ABS_RNABERT_PATH):
    if ABS_RNABERT_PATH not in sys.path:
        sys.path.append(ABS_RNABERT_PATH)
        print(f"Added {ABS_RNABERT_PATH} to sys.path")
else:
    print(f"Warning: Path not found: {ABS_RNABERT_PATH}")

import torch
import numpy as np
from tqdm import tqdm
from typing import List, Optional, Literal, Union, Tuple
from attrdict import AttrDict

from utils.bert import BertModel, BertForMaskedLM

from .base_model import BaseModel

Pooling = Literal["mean", "max", "cls"]


class RNABERTModel(BaseModel):
    """
    RNABERT 适配器
    
    - 基于 RNABERT 源码 (PyTorch implementation)
    - 预处理：自动将 DNA 转为 RNA (T -> U)，并映射为 RNABERT 的整数词表
    - 词表映射 (基于 dataload.py k=1): PAD=0, MASK=1, A=2, U=3, G=4, C=5
    
    Args:
        model_name: 逻辑名 (如 "RNABERT")
        model_path: 预训练权重文件路径 (.pth)
        config_path: 配置文件路径 (RNA_bert_config.json)
        device: 'cuda:0' / 'cpu'
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        config_path: str,
        device: Optional[str] = None,
        max_len: int = 440,
    ):
        super().__init__(model_name, model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config_path = config_path
        # RNABERT 默认 max_position_embeddings 为 440
        self.model_max_len = max_len

        # RNABERT 词表映射 (参考 dataload.py make_dict k=1)
        # 0 是 padding (convert 函数逻辑)
        # 1 是 MASK (make_dict 插入头部)
        # A, U, G, C 依次为 2, 3, 4, 5
        self.vocab_map = {"PAD": 0, "MASK": 1, "A": 2, "U": 3, "G": 4, "C": 5}
        self.mask_id = 1
        self.pad_id = 0

        self._load_model()

    def _load_model(self):
        # 1. 加载配置
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            json_obj = json.load(f)
            self.config = AttrDict(json_obj)
        
        # RNABERT 特殊逻辑：hidden_size 需要根据 heads * multiple 计算
        # 参考 MLM_SFP.py 中的 objective 函数
        if not hasattr(self.config, 'hidden_size'):
            self.config.hidden_size = self.config.num_attention_heads * self.config.multiple

        # 2. 初始化模型结构
        # RNABERT 使用 BertForMaskedLM 包装 BertModel
        # 我们使用 ForMaskedLM 以便同时支持 Embedding 提取 (通过内部 bert) 和 PLL 评分 (通过 cls 头)
        self.bert = BertModel(self.config)
        self.model = BertForMaskedLM(self.config, self.bert)

        # 3. 加载权重
        if os.path.exists(self.model_path):
            print(f"Loading weights from {self.model_path}...")
            state_dict = torch.load(self.model_path, map_location=self.device)
            
            # 处理可能的 DataParallel 'module.' 前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            self.model.load_state_dict(new_state_dict)
        else:
            print(f"Warning: Model weights not found at {self.model_path}, utilizing random initialization.")

        self.model.to(self.device).eval()
        print(f"[{self.model_name}] loaded on {self.device}, max_len={self.model_max_len}")

    def _preprocess_sequence(self, seq: str) -> str:
        """
        DNA -> RNA，大写，去除无关字符
        """
        s = seq.strip().upper()
        s = s.replace("T", "U")
        # RNABERT 仅训练了 A, U, G, C。这里不处理非标准碱基，tokenize 时会转为 0 (PAD) 或忽略
        return s

    def _tokenize(self, batch: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将序列列表转换为 input_ids 和 attention_mask
        """
        input_ids = []
        masks = []

        for seq in batch:
            rna_seq = self._preprocess_sequence(seq)
            
            # 字符转 ID，未知字符(如N)映射为 PAD(0)
            ids = [self.vocab_map.get(c, self.pad_id) for c in rna_seq]
            
            # 截断
            if len(ids) > self.model_max_len:
                ids = ids[:self.model_max_len]
            
            length = len(ids)
            # 填充
            padding_len = self.model_max_len - length
            ids_padded = ids + [self.pad_id] * padding_len
            
            # Mask: 1 for valid, 0 for pad (RNABERT utils/bert.py 处理逻辑)
            # 实际上 utils/bert.py 中 forward 会把 mask 0 的位置变成很大的负数
            mask = [1] * length + [0] * padding_len
            
            input_ids.append(ids_padded)
            masks.append(mask)

        return (torch.tensor(input_ids, dtype=torch.long, device=self.device),
                torch.tensor(masks, dtype=torch.long, device=self.device))

    # ---------- Embedding 接口 ----------
    @torch.no_grad()
    def get_embedding(
        self,
        sequences: Union[str, List[str]],
        batch_size: int = 64,
        pool: Pooling = "mean",
        layer_index: int = -1, # RNABERT 源码仅方便获取最后一层
        return_numpy: bool = True,
        **kwargs # 兼容接口参数
    ) -> Union[np.ndarray, torch.Tensor]:
        
        if isinstance(sequences, str):
            sequences = [sequences]

        results = []
        for i in tqdm(range(0, len(sequences), batch_size), desc="RNABERT Embedding"):
            batch_seqs = sequences[i : i + batch_size]
            input_ids, attn_mask = self._tokenize(batch_seqs)

            # Forward
            # BertForMaskedLM.forward 返回: (prediction_scores, prediction_scores_ss, encoded_layers)
            # encoded_layers 在 attention_show_flg=False 且 output_all_encoded_layers=False 时为最后一层 tensor [B, L, H]
            _, _, last_hidden_state = self.model(
                input_ids, 
                attention_mask=attn_mask, 
                attention_show_flg=False
            )
            
            # Pooling
            # attn_mask shape: [B, L] -> 扩展为 [B, L, 1]
            mask_expanded = attn_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            
            if pool == "mean":
                sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
                sum_mask = mask_expanded.sum(1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                embedding = sum_embeddings / sum_mask
                
            elif pool == "max":
                # 将 padding 区域设为极小值
                last_hidden_state[mask_expanded == 0] = -1e9
                embedding = torch.max(last_hidden_state, 1)[0]
                
            elif pool == "cls":
                # RNABERT 数据处理不像 BERT 那样自动加 CLS token
                # 这里的 "cls" pooling 我们取第一个 token (index 0)
                embedding = last_hidden_state[:, 0, :]
                
            else:
                raise ValueError(f"Unknown pooling method: {pool}")

            results.append(embedding.cpu())

        final_res = torch.cat(results, dim=0)
        return final_res.numpy() if return_numpy else final_res

    # 兼容接口别名
    def embed_sequences(self, *args, **kwargs):
        return self.get_embedding(*args, **kwargs)

    # ---------- PLL Scoring 接口 ----------
    @torch.no_grad()
    def score_sequences(
        self,
        sequences: List[str],
        batch_size: int = 256,
    ) -> List[float]:
        """
        计算伪对数似然 (PLL) 分数
        """
        scores = []
        for seq in tqdm(sequences, desc="Scoring sequences with RNABERT"):
            scores.append(self._score_single_sequence(seq, batch_size))
        return scores

    def _score_single_sequence(self, seq: str, mask_batch_size: int) -> float:
        # 对单条序列进行处理
        input_ids_batch, attn_mask_batch = self._tokenize([seq])
        input_ids = input_ids_batch[0] # [L]
        attn_mask = attn_mask_batch[0] # [L]
        
        # 有效长度 (非 padding 部分)
        seq_len = int(attn_mask.sum().item())
        if seq_len == 0:
            return 0.0

        # 需要 Mask 的位置索引
        valid_positions = list(range(seq_len))
        
        total_logprob = 0.0
        total_count = 0

        # 分块 Mask
        for i in range(0, len(valid_positions), mask_batch_size):
            chunk_positions = valid_positions[i : i + mask_batch_size]
            chunk_size = len(chunk_positions)

            # 复制 input_ids 构造 batch [Chunk, L]
            masked_input = input_ids.unsqueeze(0).repeat(chunk_size, 1)
            masked_attn = attn_mask.unsqueeze(0).repeat(chunk_size, 1)
            
            target_ids = []

            # 在对角线位置打 Mask
            for batch_idx, pos in enumerate(chunk_positions):
                target_ids.append(masked_input[batch_idx, pos].item())
                masked_input[batch_idx, pos] = self.mask_id

            # Forward
            # prediction_scores: [Chunk, L, Vocab]
            prediction_scores, _, _ = self.model(
                masked_input, 
                attention_mask=masked_attn, 
                attention_show_flg=False
            )
            
            # 提取 Mask 位置的 logits
            # pos_indices: [Chunk]
            pos_indices = torch.tensor(chunk_positions, device=self.device).view(-1, 1, 1)
            # 扩展为 [Chunk, 1, Vocab] 以便 gather
            pos_indices_expanded = pos_indices.expand(-1, 1, prediction_scores.size(-1))
            
            # masked_logits: [Chunk, 1, Vocab] -> squeeze -> [Chunk, Vocab]
            masked_logits = torch.gather(prediction_scores, 1, pos_indices_expanded).squeeze(1)
            
            # Log Softmax
            log_probs = torch.log_softmax(masked_logits, dim=-1)

            # 获取目标 token 的 log_prob
            target_ids_tensor = torch.tensor(target_ids, device=self.device).view(-1, 1)
            token_log_probs = torch.gather(log_probs, 1, target_ids_tensor).squeeze(1) # [Chunk]

            total_logprob += token_log_probs.sum().item()
            total_count += chunk_size
        
        return total_logprob / max(1, total_count)


# ---------- 自测部分 ----------
if __name__ == "__main__":
    # 请根据实际路径修改
    MODEL_PATH = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model/bert_mul_2.pth" 
    CONFIG_PATH = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model/RNABERT/RNA_bert_config.json"

    m = RNABERTModel(
        model_name="RNABERT",
        model_path=MODEL_PATH,
        config_path=CONFIG_PATH,
    )

    dna_list = [
        "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATTCGTTAGC", # DNA
        "AGCGUACGUUAG", # RNA
    ]

    # 测试 Embedding
    embs = m.embed_sequences(dna_list, pooling="mean")
    print("Embedding shape:", embs.shape) # 应为 (2, 120)
    print(embs)

    # 测试 PLL
    scores = m.score_sequences(dna_list, batch_size=10)
    print("PLL scores:", scores)

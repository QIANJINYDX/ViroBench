# models/gpn_star_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import re
from typing import List, Tuple, Optional, Union, Literal
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForMaskedLM, AutoConfig
from tqdm import tqdm

# 关键：注册 GPN 自定义架构到 Transformers（GPNStar）
import gpn.model
from gpn.star.data import GenomeMSA, Tokenizer

from .base_model import BaseModel

Pooling = Literal["mean", "max", "cls"]

_COORD_RE = re.compile(
    r"^(?:chr)?(?P<chrom>[0-9XYM]+):(?P<start>\d+)-(?P<end>\d+):(?P<strand>[+-])$",
    re.IGNORECASE
)

def _is_coord(s: str) -> bool:
    return _COORD_RE.match(s) is not None

def _parse_coord(s: str) -> Tuple[str, int, int, str]:
    m = _COORD_RE.match(s)
    if not m:
        raise ValueError(f"Invalid locus format: {s!r}. Expected 'chr6:31575665-31575793:+'")
    chrom = m.group("chrom")
    start = int(m.group("start"))
    end = int(m.group("end"))
    strand = m.group("strand")
    return chrom, start, end, strand

def _call_or_value(obj) -> Optional[int]:
    """
    将 tokenizer 的 *_token_id 兼容为整数。
    可能是属性（int）也可能是可调用（返回 int）。
    """
    if obj is None:
        return None
    return int(obj() if callable(obj) else obj)


class GPNStarModel(BaseModel):
    """
    GPN-Star 适配器：
      - 输入可以是 DNA 序列（A/C/G/T/N）或坐标字符串：'chr6:31575665-31575793:+'
      - score_sequences: 伪对数似然（PLL），仅对 A/C/G/T 位计分；N/未知不计分
      - get_embedding: 对 last_hidden_state 做池化（mean/max/cls）
    """

    def __init__(
        self,
        model_name: str,
        model_path: str = "songlab/gpn-star-hg38-v100-200m",
        msa_paths: Optional[Union[str, List[str], dict]] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        load_mlm_head: bool = True,
        target_species: int = 0,
    ):
        """
        Args:
            model_name: 自定义名称
            model_path: HF 或本地路径（如 "songlab/gpn-star-hg38-v100-200m"）
            msa_paths:  GenomeMSA 数据源（坐标输入时必须提供）
                       - str: 单个路径
                       - List[str]: 多个路径列表
                       - dict: {n_species: path} 字典，如 {100: "/path/to/100.zarr"}
            device:     "cuda" | "cpu" | torch.device
            dtype:      torch.float16 / torch.bfloat16 / None
            load_mlm_head: 是否加载 MLM 头（score_sequences 需要）
            target_species: target species 的索引（通常是 0，表示第一个物种）
        """
        super().__init__(model_name=model_name, model_path=model_path)

        self.device = torch.device(device) if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.dtype = dtype
        self.target_species = target_species

        # 加载配置并设置 phylo_dist_path
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        phylo_dist_path = os.path.join(model_path, "phylo_dist")
        if os.path.exists(phylo_dist_path):
            config.phylo_dist_path = phylo_dist_path
        else:
            print(f"[GPNStarModel] Warning: phylo_dist_path not found at {phylo_dist_path}")

        # Backbone（用于 embedding）
        self.backbone = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True)
        if self.dtype is not None:
            self.backbone = self.backbone.to(self.dtype)
        self.backbone.to(self.device).eval()

        # MLM 头（用于 PLL 评分）
        self.mlm: Optional[AutoModelForMaskedLM] = None
        if load_mlm_head:
            self.mlm = AutoModelForMaskedLM.from_pretrained(model_path, config=config, trust_remote_code=True)
            if self.dtype is not None:
                self.mlm = self.mlm.to(self.dtype)
            self.mlm.to(self.device).eval()

        # 可选：MSA 数据源（用于坐标输入）
        self.genome_msa_list: List[GenomeMSA] = []
        if msa_paths is not None:
            if isinstance(msa_paths, str):
                # 单个路径
                self.genome_msa_list = [GenomeMSA(msa_paths, in_memory=False)]
            elif isinstance(msa_paths, list):
                # 多个路径列表
                self.genome_msa_list = [GenomeMSA(path, in_memory=False) for path in msa_paths]
            elif isinstance(msa_paths, dict):
                # 字典格式：{n_species: path}
                self.genome_msa_list = [
                    GenomeMSA(path, n_species=n_species, in_memory=False)
                    for n_species, path in msa_paths.items()
                ]
            else:
                raise ValueError(f"Unsupported msa_paths type: {type(msa_paths)}")

        # Tokenizer 与安全映射
        self.tokenizer = Tokenizer()
        self._setup_token_ids()

        print(f"[GPNStarModel] loaded on {self.device}, "
              f"mlm_head={load_mlm_head}, target_species={target_species}, "
              f"num_msa_sources={len(self.genome_msa_list)}")

    # --------- 安全 token id 处理 ---------
    def _setup_token_ids(self) -> None:
        # 获取 vocab
        vocab_obj = getattr(self.tokenizer, "vocab", None)
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
            if tok in self._vocab_list:
                return self._vocab_list.index(tok)
            if self._unk_id is not None:
                return int(self._unk_id)
            if self._pad_id is not None:
                return int(self._pad_id)
            return 0  # 最后兜底

        # 构建核苷酸映射
        self._nuc_to_id = {
            "A": _safe_id("A"),
            "C": _safe_id("C"),
            "G": _safe_id("G"),
            "T": _safe_id("T"),
            "N": _safe_id("N"),
        }

        # 仅当 A/C/G/T 真正存在于 vocab 时，才将其纳入"可计分 token 集合"
        self._valid_base_ids = {
            self._vocab_list.index(b) for b in ("A", "C", "G", "T") if b in self._vocab_list
        }

    # -------------------- 内部：输入组装 --------------------
    def _build_inputs(self, seq_or_locus: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将输入转换为 (input_ids, source_ids, target_species)：
          - input_ids:   (1, L, 1) long - target species 的序列
          - source_ids:  (1, L, S) long - 所有物种的 MSA
          - target_species: (1, 1) int - target species 的索引
        """
        if _is_coord(seq_or_locus):
            if len(self.genome_msa_list) == 0:
                raise RuntimeError("Received a genomic locus but msa_paths was not provided.")
            
            chrom, start, end, strand = _parse_coord(seq_or_locus)
            
            # 从所有 MSA 源获取数据并连接
            msa_list = []
            for genome_msa in self.genome_msa_list:
                msa_np = genome_msa.get_msa(chrom, start, end, strand=strand, tokenize=True)  # (L, S)
                msa_list.append(msa_np)
            
            if len(msa_list) > 1:
                # 连接多个 MSA（沿最后一个维度）
                msa = np.concatenate(msa_list, axis=-1)  # (L, S_total)
            else:
                msa = msa_list[0]  # (L, S)
            
            # 转换为 tensor: (1, L, S)
            msa_tensor = torch.tensor(np.expand_dims(msa, 0).astype(np.int64), device=self.device)
            
            # input_ids 是 target species（第一个物种），source_ids 是所有物种
            input_ids = msa_tensor[:, :, self.target_species:self.target_species+1]  # (1, L, 1)
            source_ids = msa_tensor  # (1, L, S)
            target_species = torch.tensor([[self.target_species]], dtype=torch.long, device=self.device)
            
            return input_ids, source_ids, target_species

        # 纯序列输入：允许 A/C/G/T/N；N 走回退 id，且不参与计分
        dna = seq_or_locus.strip().upper()
        if not re.fullmatch(r"[ACGTN]+", dna):
            raise ValueError(
                f"Sequence contains invalid characters: {seq_or_locus!r}. Only A/C/G/T/N are allowed."
            )

        ids = [self._nuc_to_id.get(ch, self._nuc_to_id["N"]) for ch in dna]
        input_ids = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0).unsqueeze(-1)  # (1, L, 1)
        
        # 对于纯序列输入，source_ids 与 input_ids 相同（只有一个物种）
        source_ids = input_ids  # (1, L, 1)
        target_species = torch.tensor([[self.target_species]], dtype=torch.long, device=self.device)
        
        return input_ids, source_ids, target_species

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
        # 1. 构建输入
        input_ids, source_ids, target_species = self._build_inputs(sequence)
        seq_len = input_ids.shape[1]

        # 2. 找出有效位置（仅对 A/C/G/T 计分）
        input_ids_cpu = input_ids[0, :, 0].detach().cpu().tolist()
        valid_positions = [i for i, tok in enumerate(input_ids_cpu) if tok in self._valid_base_ids]

        if len(valid_positions) == 0:
            return float("nan")

        # 3. 分块遮罩并计算 log-probability
        total_logprob = 0.0
        total_count = 0

        for start_idx in range(0, len(valid_positions), mask_batch_size):
            chunk_positions = valid_positions[start_idx:start_idx + mask_batch_size]
            chunk_size = len(chunk_positions)

            # 复制 input_ids 和 source_ids 用于遮罩
            masked_input_ids = input_ids.repeat(chunk_size, 1, 1)  # (B, L, 1)
            masked_source_ids = source_ids.repeat(chunk_size, 1, 1)  # (B, L, S)
            target_species_batch = target_species.repeat(chunk_size, 1)  # (B, 1)
            true_tokens = []

            # 遮罩对应位置
            for b, pos in enumerate(chunk_positions):
                true_token = int(input_ids[0, pos, 0].item())
                true_tokens.append(true_token)
                masked_input_ids[b, pos, 0] = self._mask_id

            # 模型前向传播
            # GPN-Star 模型需要 input_ids, source_ids, target_species
            all_logits = self.mlm(
                input_ids=masked_input_ids,
                source_ids=masked_source_ids,
                target_species=target_species_batch.cpu().numpy()
            ).logits  # (B, L, 1, V)

            # 获取遮罩位置的 logits
            batch_indices = torch.arange(chunk_size, device=self.device)
            pos_indices = torch.tensor(chunk_positions, device=self.device, dtype=torch.long)
            pos_logits = all_logits[batch_indices, pos_indices, 0, :]  # (B, V)

            # 计算 log-probability
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
            sequences: 输入序列/坐标列表
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

            # 逐条构建输入 -> 对齐到不同长度时 pad
            input_ids_list, source_ids_list, target_species_list = [], [], []
            max_len = 0
            
            for item in batch:
                input_ids, source_ids, target_species = self._build_inputs(item)
                input_ids_list.append(input_ids[0])  # (L, 1)
                source_ids_list.append(source_ids[0])  # (L, S)
                target_species_list.append(target_species[0])  # (1,)
                max_len = max(max_len, input_ids.size(1))

            # pad 到 max_len
            pad_id = self._pad_id if self._pad_id is not None else 0
            input_ids_padded = []
            source_ids_padded = []
            
            for input_ids, source_ids in zip(input_ids_list, source_ids_list):
                L = input_ids.size(0)
                pad_len = max_len - L
                if pad_len > 0:
                    input_ids = torch.cat([
                        input_ids,
                        torch.full((pad_len, 1), pad_id, dtype=torch.long, device=self.device)
                    ], dim=0)
                    source_ids = torch.cat([
                        source_ids,
                        torch.full((pad_len, source_ids.size(1)), pad_id, dtype=torch.long, device=self.device)
                    ], dim=0)
                input_ids_padded.append(input_ids.unsqueeze(0))
                source_ids_padded.append(source_ids.unsqueeze(0))

            input_ids_batch = torch.cat(input_ids_padded, dim=0)  # (B, L, 1)
            source_ids_batch = torch.cat(source_ids_padded, dim=0)  # (B, L, S)
            target_species_batch = torch.stack(target_species_list, dim=0)  # (B, 1)

            # 前向传播
            if layer_index == -1:
                outputs = self.backbone(
                    input_ids=input_ids_batch,
                    source_ids=source_ids_batch,
                    target_species=target_species_batch.cpu().numpy(),
                    return_dict=True,
                )
                hidden = outputs.last_hidden_state  # (B, L, H)
            else:
                outputs = self.backbone(
                    input_ids=input_ids_batch,
                    source_ids=source_ids_batch,
                    target_species=target_species_batch.cpu().numpy(),
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden = outputs.hidden_states[layer_index]  # (B, L, H)

            # 池化
            if pool == "cls":
                vecs = hidden[:, 0, :]  # (B, H)
            elif pool == "mean":
                # 对于 GPN-Star，我们只对 target species 的位置做 mean pooling
                # 由于 input_ids 已经是 target species，我们可以直接对 hidden 做 mean
                vecs = hidden.mean(dim=1)  # (B, H)
            elif pool == "max":
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
#     MODEL_PATH = "/mnt/s3mount/model_weight/gpn-star-hg38-v100-200m"
#     MSA_PATH = "/mnt/s3mount/peijunlin/gpn_msa/multiz100way/99.zarr"
#     
#     m = GPNStarModel(
#         model_name="gpn-star",
#         model_path=MODEL_PATH,
#         msa_paths={100: MSA_PATH},  # 或单个路径: msa_paths=MSA_PATH
#         device=None,          # 自动选 GPU/CPU
#         load_mlm_head=True,   # 需要 PLL 评分时设为 True
#         target_species=0,     # target species 索引
#     )
# 
#     # 测试坐标输入
#     seqs = ["chr6:31575665-31575793:+"]
#     scores = m.score_sequences(seqs, batch_size=128)
#     print("PLL scores:", scores)
# 
#     # 测试 embedding 提取
#     embs = m.get_embedding(seqs, pool="mean", batch_size=32, return_numpy=True)
#     print("Embedding shape:", embs.shape)  # (1, hidden_size)


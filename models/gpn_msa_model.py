# models/gpn_msa_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import re
from typing import List, Tuple, Optional, Union, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForMaskedLM
from tqdm import tqdm
# 关键：注册 GPN 自定义架构到 Transformers（GPNRoFormer）
import gpn.model
from gpn.data import GenomeMSA, Tokenizer

try:
    from .base_model import BaseModel
except ImportError:
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    from base_model import BaseModel  # 兼容单文件运行

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

class GPNMSAModel(BaseModel):
    """
    GPN-MSA 适配器：
      - 输入可以是 DNA 序列（A/C/G/T/N）或坐标字符串：'chr6:31575665-31575793:+'
      - score_sequences: 伪对数似然（PLL），仅对 A/C/G/T 位计分；N/未知不计分
      - get_embedding: 对 last_hidden_state 做 mean pooling
    """

    def __init__(
        self,
        model_name: str,
        model_path: str = "songlab/gpn-msa-sapiens",
        msa_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        load_mlm_head: bool = True,
    ):
        """
        Args:
            model_name: 自定义名称
            model_path: HF 或本地路径（如 "songlab/gpn-msa-sapiens"）
            msa_path:   GenomeMSA 数据源（坐标输入时必须提供）
            device:     "cuda" | "cpu" | torch.device
            dtype:      torch.float16 / torch.bfloat16 / None
            load_mlm_head: 是否加载 MLM 头（score_sequences 需要）
        """
        super().__init__(model_name=model_name, model_path=model_path)

        self.device = torch.device(device) if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.dtype = dtype

        # Backbone（用于 embedding）
        self.backbone = AutoModel.from_pretrained(model_path)
        if self.dtype is not None:
            self.backbone = self.backbone.to(self.dtype)
        self.backbone.to(self.device).eval()

        # MLM 头（用于 PLL 评分）
        self.mlm: Optional[AutoModelForMaskedLM] = None
        if load_mlm_head:
            self.mlm = AutoModelForMaskedLM.from_pretrained(model_path)
            if self.dtype is not None:
                self.mlm = self.mlm.to(self.dtype)
            self.mlm.to(self.device).eval()

        # 可选：MSA 数据源（用于坐标输入）
        self.genome_msa: Optional[GenomeMSA] = GenomeMSA(msa_path) if msa_path is not None else None

        # Tokenizer 与安全映射
        self.tokenizer = Tokenizer()
        self._setup_token_ids()

    # --------- 安全 token id 处理（兼容无 'N' 的 tokenizer） ---------
    def _setup_token_ids(self) -> None:
        # 尝试读取 vocab：多数实现为 list；也兼容为 dict.keys()
        vocab_obj = getattr(self.tokenizer, "vocab", None)
        if vocab_obj is None:
            # 兜底为空列表（之后全部走 unk/pad 回退）
            vocab_list: List[str] = []
        elif isinstance(vocab_obj, dict):
            vocab_list = list(vocab_obj.keys())
        elif isinstance(vocab_obj, (list, tuple)):
            vocab_list = list(vocab_obj)
        else:
            # 未知类型，尽量转 list
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

        # 构建核苷酸映射（不要求有 'N'）
        self._nuc_to_id = {
            "A": _safe_id("A"),
            "C": _safe_id("C"),
            "G": _safe_id("G"),
            "T": _safe_id("T"),
            "N": _safe_id("N"),  # 若 vocab 无 'N'，将回退到 unk/pad/0
        }

        # 仅当 A/C/G/T 真正存在于 vocab 时，才将其纳入“可计分 token 集合”
        self._valid_base_ids = {
            self._vocab_list.index(b) for b in ("A", "C", "G", "T") if b in self._vocab_list
        }

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

        out: List[float] = []
        for item in sequences:
            input_ids, aux = self._build_inputs(item)  # (1, L), (1, L, S-1 or 0)
            L = int(input_ids.size(1))

            # 仅对 A/C/G/T 的位点计分
            ids_cpu = input_ids[0].detach().cpu().tolist()
            valid_positions = [i for i, tok in enumerate(ids_cpu) if tok in self._valid_base_ids]

            if len(valid_positions) == 0:
                out.append(float("nan"))
                continue

            # 分块掩码，降低显存占用
            total_logprob, total_cnt = 0.0, 0
            for start in range(0, len(valid_positions), batch_size):
                chunk = valid_positions[start:start + batch_size]
                bsz = len(chunk)

                ids_batch = input_ids.repeat(bsz, 1)  # (B, L)
                aux_batch = aux.repeat(bsz, 1, 1)     # (B, L, S-1 or 0)
                true_tokens: List[int] = []

                # 掩码对应位置
                for b, pos in enumerate(chunk):
                    true_tok = int(ids_batch[b, pos].item())
                    true_tokens.append(true_tok)
                    ids_batch[b, pos] = int(self._mask_id)
                kwargs = {"input_ids": ids_batch}
                if aux is not None and aux.dim() == 3 and aux.size(-1) > 0:
                    kwargs["aux_features"] = aux_batch  # 只有在最后一维 > 0 才传
                outputs = self.mlm(**kwargs)
                logits = outputs.logits  # (B, L, V)

                row_idx = torch.arange(bsz, device=self.device)
                col_idx = torch.tensor(chunk, device=self.device, dtype=torch.long)
                pos_logits = logits[row_idx, col_idx, :]  # (B, V)
                log_probs = F.log_softmax(pos_logits, dim=-1)  # (B, V)

                true_idx = torch.tensor(true_tokens, device=self.device, dtype=torch.long)
                gathered = log_probs[row_idx, true_idx]        # (B,)

                total_logprob += gathered.sum().item()
                total_cnt += bsz

                # 释放临时张量
                del outputs, logits, pos_logits, log_probs, true_idx, gathered
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

            avg_logprob = total_logprob / max(1, total_cnt)
            out.append(avg_logprob)

        return out

    # -------------------- 公共 API：Embedding --------------------
    @torch.inference_mode()
    def get_embedding(
        self,
        sequences: List[str],
        pool: str = "mean",
        batch_size: int = 64,
        return_numpy: bool = False,
    ) -> Union[List[List[float]], np.ndarray]:
        """
        提取序列级 embedding。

        Args:
            sequences: 输入序列/坐标列表
            pool: 池化方式 {"mean", "max", "cls"}
            batch_size: 前向批大小
            return_numpy: 是否返回 numpy.ndarray（否则返回 Python list）

        Returns:
            - 如果 return_numpy=True: (N, H) numpy array
            - 否则: List[List[float]]，每个序列一个向量
        """
        pool = pool.lower()
        if pool not in {"mean", "max", "cls"}:
            raise ValueError(f"Unsupported pool: {pool}")

        all_vecs: List[np.ndarray] = []
        # 按 batch 切分
        for start in tqdm(range(0, len(sequences), batch_size), desc="Getting embedding"):
            batch = sequences[start:start + batch_size]

            # 逐条构建输入 -> 对齐到不同长度时 pad
            input_ids_list, aux_list = [], []
            max_len = 0
            for item in batch:
                ids, aux = self._build_inputs(item)   # (1, L), (1, L, S-1)
                input_ids_list.append(ids[0])        # (L,)
                aux_list.append(aux[0])              # (L, S-1)
                max_len = max(max_len, ids.size(1))

            # pad 到 max_len
            pad_id = self._pad_id if self._pad_id is not None else 0
            input_ids_padded = []
            aux_padded = []
            for ids, aux in zip(input_ids_list, aux_list):
                L = ids.size(0)
                pad_len = max_len - L
                if pad_len > 0:
                    ids = torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long, device=self.device)], dim=0)
                    aux = torch.cat([aux,
                                     torch.full((pad_len, aux.size(1)), pad_id, dtype=torch.long, device=self.device)], dim=0)
                input_ids_padded.append(ids.unsqueeze(0))
                aux_padded.append(aux.unsqueeze(0))

            input_ids_batch = torch.cat(input_ids_padded, dim=0)  # (B, L)
            aux_batch = torch.cat(aux_padded, dim=0)              # (B, L, S-1)

            outputs = self.backbone(input_ids=input_ids_batch, aux_features=aux_batch)
            hs = outputs.last_hidden_state  # (B, L, H)

            if pool == "mean":
                vecs = hs.mean(dim=1)
            elif pool == "max":
                vecs, _ = hs.max(dim=1)
            else:  # "cls" -> 取第一个位置
                vecs = hs[:, 0, :]

            all_vecs.append(vecs.detach().cpu().numpy())

            del outputs, hs, vecs
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        embs = np.concatenate(all_vecs, axis=0)  # (N, H)

        if return_numpy:
            return embs
        else:
            return embs.tolist()
    # -------------------- 内部：输入组装 --------------------
    def _build_inputs(self, seq_or_locus: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将输入转换为 (input_ids, aux_features)：
          - input_ids:   (1, L) long
          - aux_features:(1, L, S-1) long；若无 MSA 则为 (1, L, 0)
        """
        if _is_coord(seq_or_locus):
            if self.genome_msa is None:
                raise RuntimeError("Received a genomic locus but msa_path was not provided.")
            chrom, start, end, strand = _parse_coord(seq_or_locus)
            msa_np = self.genome_msa.get_msa(chrom, start, end, strand=strand, tokenize=True)  # (L, S)
            msa = torch.tensor(np.expand_dims(msa_np, 0).astype(np.int64), device=self.device)
            input_ids = msa[:, :, 0]   # (1, L)
            aux = msa[:, :, 1:]        # (1, L, S-1)
            return input_ids, aux

        # 纯序列输入：允许 A/C/G/T/N；N 走回退 id，且不参与计分
        dna = seq_or_locus.strip().upper()
        if not re.fullmatch(r"[ACGTN]+", dna):
            raise ValueError(
                f"Sequence contains invalid characters: {seq_or_locus!r}. Only A/C/G/T/N are allowed."
            )

        ids = [self._nuc_to_id.get(ch, self._nuc_to_id["N"]) for ch in dna]
        input_ids = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)  # (1, L)

        L = int(input_ids.size(1))
        aux = torch.empty((1, L, 0), dtype=torch.long, device=self.device)  # 无多物种时为空维度
        return input_ids, aux


# model = GPNMSAModel(
#     model_name="gpn-msa",
#     model_path="/mnt/s3mount/model_weight/gpn-msa-sapiens",
#     msa_path="/mnt/s3mount/peijunlin/gpn_msa/peijunlin/89.zarr",            # 不提供也行
#     device="cuda"             # 或 "cpu"
# )

# seqs = ["chr6:31575665-31575793:+"]
# scores = model.score_sequences(seqs)     # 伪对数似然平均 log-prob
# embeds = model.get_embedding([seqs[0]]) # 单条序列 embedding（List[float]）

# print(scores)
# print(embeds)
# models/rinalmo_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import sys
from typing import List, Optional, Literal, Union, Tuple

import torch
import numpy as np
from tqdm import tqdm

from .base_model import BaseModel


Pooling = Literal["mean", "max", "cls"]


class RiNALMoModel(BaseModel):
    """
    RiNALMo 适配器（基于 multimolecule/Hugging Face 实现）

    - 默认: 使用 RiNALMoModel 提取隐藏态并池化，得到序列级向量
    - 可选: use_mlm_head=True 时使用 RiNALMoForMaskedLM 做 PLL 评分（逐位遮罩）

    说明：
        本类假定输入仍是 DNA 序列 (A/C/G/T/...)，内部使用 RnaTokenizer，
        自动将 T -> U，并做大写转换等预处理。

    Args:
        model_name: 逻辑名（比如 "RiNALMo-giga"）
        model_path: HuggingFace 模型名或本地目录
                    例如 "multimolecule/rinalmo-giga"
        hf_home:    HF 缓存目录(可选)，会写入环境变量 HF_HOME
        device:     'cuda:0' / 'cpu' / None(自动选择)
        use_mlm_head: 是否加载 MLM 头用于 PLL 评分
    """

    def __init__(
        self,
        model_name: str,
        model_path: str = "multimolecule/rinalmo-giga",
        hf_home: Optional[str] = None,
        device: Optional[str] = None,
        use_mlm_head: bool = False,
    ):
        super().__init__(model_name, model_path)
        self.hf_home = hf_home
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_mlm_head = use_mlm_head

        self.tokenizer: RnaTokenizer
        self.model: torch.nn.Module
        self.model_max_len: int = 1024
        self.mask_id: Optional[int] = None

        self._load_model()

    # ---------- 加载 ----------
    def _load_model(self):
        if self.hf_home:
            os.environ.setdefault("HF_HOME", self.hf_home)

        rna_tokenizer_cls, hf_rinalmo_cls, rinalmo_mlm_cls = self._import_multimolecule()

        # RnaTokenizer 会自动：
        # - 大写序列
        # - 将 T 替换为 U（replace_T_with_U=True）
        self.tokenizer = rna_tokenizer_cls.from_pretrained(self.model_path)

        if self.use_mlm_head:
            self.model = rinalmo_mlm_cls.from_pretrained(self.model_path)
        else:
            self.model = hf_rinalmo_cls.from_pretrained(self.model_path)

        self.model.to(self.device).eval()

        # 最大长度
        self.model_max_len = (
            getattr(self.model.config, "max_position_embeddings", None)
            or getattr(self.tokenizer, "model_max_length", None)
            or 1024
        )

        # MLM 相关（仅在 use_mlm_head=True 时）
        self.mask_id = None
        if self.use_mlm_head:
            self.mask_id = self.tokenizer.mask_token_id
            if self.mask_id is None:
                cand = self.tokenizer.mask_token or "<mask>"
                mid = self.tokenizer.convert_tokens_to_ids(cand)
                if isinstance(mid, int) and mid >= 0:
                    self.mask_id = mid
                else:
                    raise ValueError("找不到 mask_token_id，无法进行 PLL 评分。")

        print(
            f"[RiNALMoModel] loaded on {self.device}, "
            f"mlm_head={self.use_mlm_head}, "
            f"model_max_len={self.model_max_len}"
        )

    def _import_multimolecule(self):
        """
        Import multimolecule while avoiding name collision with local `datasets` package.

        run_all.py prepends the project root to sys.path, which shadows the
        HuggingFace `datasets` module that multimolecule depends on. We temporarily
        remove the project root and purge a cached local `datasets` module.
        """
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_root_abs = os.path.abspath(project_root)
        removed_paths = [p for p in sys.path if os.path.abspath(p) == project_root_abs]
        if removed_paths:
            sys.path = [p for p in sys.path if os.path.abspath(p) != project_root_abs]
        removed_datasets = False
        datasets_mod = sys.modules.get("datasets")
        if datasets_mod is not None:
            mod_path = getattr(datasets_mod, "__file__", "")
            if mod_path and os.path.abspath(mod_path).startswith(project_root_abs):
                del sys.modules["datasets"]
                removed_datasets = True
        try:
            import multimolecule  # noqa: F401
            from multimolecule import (
                RnaTokenizer,
                RiNALMoModel as HFRiNALMoModel,
                RiNALMoForMaskedLM,
            )
        finally:
            if removed_paths:
                sys.path.insert(0, project_root)
            if removed_datasets:
                # Let future imports resolve normally; no re-insert.
                pass
        return RnaTokenizer, HFRiNALMoModel, RiNALMoForMaskedLM

    # ---------- 简单预处理 ----------
    def _preprocess_sequence(self, seq: str) -> str:
        """
        将原始 DNA/RNA 序列做简单清洗：
        - 去掉首尾空白
        - 转大写
        RnaTokenizer 内部会负责 T->U 等操作。
        """
        return seq.strip().upper()

    # ---------- 嵌入提取（简易封装） ----------
    @torch.no_grad()
    def embed_sequences(
        self,
        sequences: List[str],
        batch_size: int = 32,
        pooling: Pooling = "mean",
        exclude_special: bool = True,
        truncation: bool = True,
        return_numpy: bool = True,
    ):
        """
        返回每条序列的向量表示（隐藏态池化）。
        pooling: 'mean' | 'max' | 'cls'
        """
        return self.get_embedding(
            sequences=sequences,
            layer_name=None,
            batch_size=batch_size,
            pool=pooling,
            layer_index=-1,
            average_reverse_complement=False,
            exclude_special=exclude_special,
            truncation=truncation,
            max_length=None,
            return_numpy=return_numpy,
        )

    # ---------- PLL 评分 ----------
    def score_sequences(self, sequences: List[str], batch_size: int = 256) -> List[float]:
        """
        对序列进行 PLL (Pseudo-Log-Likelihood) 评分
        使用 MLM 方式：逐位遮罩，计算每个位置真实 token 的 log-probability

        Args:
            sequences: 原始 DNA/RNA 序列列表
            batch_size: 每一次前向中同时遮罩的 token 数（不是样本 batch）

        Returns:
            每个序列的平均 log-likelihood 得分（按 token 平均）
        """
        if not self.use_mlm_head:
            raise RuntimeError(
                "PLL scoring requires use_mlm_head=True. "
                "Please initialize RiNALMoModel with use_mlm_head=True."
            )

        if self.mask_id is None:
            raise RuntimeError("Mask token ID not found. Cannot perform PLL scoring.")

        all_scores: List[float] = []
        mask_batch_size = batch_size

        with torch.no_grad():
            for seq in tqdm(sequences, desc="Scoring sequences (RiNALMo)"):
                score = self._score_single_sequence(seq, mask_batch_size)
                all_scores.append(score)

        return all_scores

    def _score_single_sequence(self, sequence: str, mask_batch_size: int = 256) -> float:
        """
        对单条序列进行 PLL 评分

        Args:
            sequence: 输入序列（DNA/RNA）
            mask_batch_size: 每次前向中同时遮罩的位置数量

        Returns:
            序列的平均 log-likelihood（按 token 平均）
        """
        # 1. 预处理 + 分词
        proc_seq = self._preprocess_sequence(sequence)

        try:
            enc = self.tokenizer(
                proc_seq,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.model_max_len,
                add_special_tokens=True,
            )
        except Exception:
            # 保险 fallback
            token_ids = self.tokenizer.encode(
                proc_seq,
                add_special_tokens=True,
                max_length=self.model_max_len,
                truncation=True,
            )
            enc = {
                "input_ids": torch.tensor([token_ids], dtype=torch.long),
                "attention_mask": torch.ones(1, len(token_ids), dtype=torch.long),
            }

        input_ids = enc["input_ids"].to(self.device)  # (1, L)
        attn_mask = enc.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device)
        else:
            attn_mask = torch.ones_like(input_ids)

        seq_len = input_ids.shape[1]
        input_ids = input_ids.squeeze(0)   # (L,)
        attn_mask = attn_mask.squeeze(0)   # (L,)

        # 2. 找出有效位置（排除特殊 token 和 padding）
        valid_positions: List[int] = []
        input_ids_list = input_ids.tolist()

        pad_id = self.tokenizer.pad_token_id
        cls_id = getattr(self.tokenizer, "cls_token_id", None)
        sep_id = getattr(self.tokenizer, "sep_token_id", None)

        for i in range(seq_len):
            if attn_mask[i].item() == 0:
                continue

            token_id = input_ids_list[i]

            is_special = False
            if pad_id is not None and token_id == pad_id:
                is_special = True
            elif cls_id is not None and token_id == cls_id:
                is_special = True
            elif sep_id is not None and token_id == sep_id:
                is_special = True

            if not is_special:
                valid_positions.append(i)

        if len(valid_positions) == 0:
            return 0.0

        # 3. 分块遮罩并计算 log-probability
        total_logprob = 0.0
        total_count = 0

        for start_idx in range(0, len(valid_positions), mask_batch_size):
            chunk_positions = valid_positions[start_idx : start_idx + mask_batch_size]
            chunk_size = len(chunk_positions)

            # 复制 input_ids 用于遮罩
            masked_ids = input_ids.unsqueeze(0).repeat(chunk_size, 1)   # (B, L)
            masked_attn = attn_mask.unsqueeze(0).repeat(chunk_size, 1)  # (B, L)
            true_tokens: List[int] = []

            # 遮罩对应位置
            for b, pos in enumerate(chunk_positions):
                true_token = int(input_ids[pos].item())
                true_tokens.append(true_token)
                masked_ids[b, pos] = self.mask_id

            inputs = {
                "input_ids": masked_ids.to(self.device),
                "attention_mask": masked_attn.to(self.device),
            }

            outputs = self.model(**inputs)

            # 提取 logits
            if hasattr(outputs, "logits"):
                logits = outputs.logits  # (B, L, V)
            elif isinstance(outputs, tuple):
                if len(outputs) >= 2:
                    logits = outputs[1]
                else:
                    logits = outputs[0]
            else:
                raise ValueError("Cannot extract logits from model outputs")

            # 获取遮罩位置的 logits
            batch_indices = torch.arange(chunk_size, device=self.device)
            pos_indices = torch.tensor(chunk_positions, device=self.device, dtype=torch.long)
            pos_logits = logits[batch_indices, pos_indices, :]  # (B, V)

            # 计算 log-probability
            log_probs = torch.log_softmax(pos_logits, dim=-1)  # (B, V)

            true_token_ids = torch.tensor(true_tokens, device=self.device, dtype=torch.long)
            token_log_probs = log_probs[batch_indices, true_token_ids]  # (B,)

            total_logprob += token_log_probs.sum().item()
            total_count += chunk_size

            # 清理显存
            del outputs, logits, pos_logits, log_probs, token_log_probs, true_token_ids
            if "cuda" in str(self.device):
                torch.cuda.empty_cache()

        avg_logprob = total_logprob / max(1, total_count)
        return float(avg_logprob)

    # ---------- 通用 embedding 接口 ----------
    @torch.no_grad()
    def get_embedding(
        self,
        sequences: Union[str, List[str]],
        layer_name: str = None,                # 为了兼容 BaseModel 接口，但实际使用 layer_index
        batch_size: int = 64,
        pool: Pooling = "mean",                # "cls" | "mean" | "max"
        layer_index: int = -1,                 # -1 为最后一层；其余为索引
        average_reverse_complement: bool = False,  # 与反向互补平均（按 DNA 互补规则）
        exclude_special: bool = True,          # mean/max 时排除 [CLS]/[SEP]/PAD
        truncation: bool = True,
        max_length: Optional[int] = None,      # 不给则用 tokenizer/model 上限
        return_numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        返回每条序列一个向量 (N, H)。
        - pool="cls"/"mean"/"max"
        - layer_index: 取指定隐藏层（需要模型支持 hidden_states）
        - average_reverse_complement: 前向与反向互补分别取向量后做平均
        """
        device = self.model.device
        tok = self.tokenizer

        if isinstance(sequences, str):
            seq_list = [sequences]
        else:
            seq_list = list(sequences)

        # DNA 反向互补（兼容 U/T）
        def _revcomp(seq: str) -> str:
            tbl = str.maketrans(
                "ACGUTRYMKBDHVNacgutrymkbdhvn",
                "TGCAAYRKMVHDBNtgcaaYrkmvhdbn",
            )
            return seq.translate(tbl)[::-1]

        def _tokenize(batch: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
            proc = [self._preprocess_sequence(s) for s in batch]

            enc = tok(
                proc,
                return_tensors="pt",
                padding=True,
                truncation=truncation,
                max_length=max_length or self.model_max_len or 1024,
            )
            input_ids = enc["input_ids"].to(device)
            attn_mask = enc.get("attention_mask")
            attn_mask = (
                attn_mask.to(device)
                if attn_mask is not None
                else torch.ones_like(input_ids, device=device)
            )
            return input_ids, attn_mask

        def _forward_hidden(input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
            # 需要中间层则打开 hidden_states；否则直接拿 last_hidden_state
            if layer_index == -1:
                try:
                    out = self.model(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        return_dict=True,
                    )
                except TypeError:
                    out = self.model(input_ids=input_ids, return_dict=True)
                hidden = getattr(out, "last_hidden_state", out[0])
                return hidden
            else:
                try:
                    out = self.model(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                except TypeError:
                    out = self.model(
                        input_ids=input_ids,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                hidden_states = out.hidden_states
                return hidden_states[layer_index]

        def _pool(hidden: torch.Tensor, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
            # hidden: [B, T, H]
            if pool == "cls":
                return hidden[:, 0, :]  # [B, H]

            # 构造有效位置（排除 PAD / 特殊符号）
            valid = attn_mask.bool()  # [B, T]
            if exclude_special:
                spec_masks = []
                for ids in input_ids:
                    spec = tok.get_special_tokens_mask(
                        ids.tolist(),
                        already_has_special_tokens=True,
                    )
                    spec_masks.append(
                        torch.tensor(
                            spec,
                            device=attn_mask.device,
                            dtype=torch.bool,
                        )
                    )
                spec_mask = torch.stack(spec_masks, dim=0)
                valid = valid & (~spec_mask)

            if pool == "mean":
                m = valid.unsqueeze(-1).to(hidden.dtype)       # [B, T, 1]
                summed = (hidden * m).sum(dim=1)              # [B, H]
                denom = m.sum(dim=1).clamp_min(1.0)           # [B, 1]
                return summed / denom
            elif pool == "max":
                masked = hidden.masked_fill(
                    ~valid.unsqueeze(-1),
                    float("-inf"),
                )
                pooled = masked.max(dim=1).values             # [B, H]
                # 若整行全是 -inf（极端情况），fallback 到未掩码 max
                inf_mask = torch.isinf(pooled).any(dim=1)
                if inf_mask.any():
                    pooled[inf_mask] = hidden[inf_mask].max(dim=1).values
                return pooled
            else:
                raise ValueError(f"Unknown pool='{pool}'")

        outputs: List[torch.Tensor] = []
        for st in tqdm(range(0, len(seq_list), batch_size), desc="Getting embedding (RiNALMo)"):
            chunk = seq_list[st : st + batch_size]

            ids_f, am_f = _tokenize(chunk)
            hid_f = _forward_hidden(ids_f, am_f)
            vec_f = _pool(hid_f, ids_f, am_f)  # [B, H]

            if average_reverse_complement:
                rc_chunk = [_revcomp(s) for s in chunk]
                ids_r, am_r = _tokenize(rc_chunk)
                hid_r = _forward_hidden(ids_r, am_r)
                vec_r = _pool(hid_r, ids_r, am_r)
                vec = 0.5 * (vec_f + vec_r)
            else:
                vec = vec_f

            outputs.append(vec.detach().to(torch.float32).cpu())

        out = torch.cat(outputs, dim=0) if outputs else torch.empty(0, 0)
        return out.numpy() if return_numpy else out


# ---------- 自测 ----------
if __name__ == "__main__":
    # 根据你本地/远程模型位置修改
    # 1) 直接用 HF 名称：
    # MODEL_ID = "multimolecule/rinalmo-giga"
    # 2) 或者使用已经下载到本地的目录：
    MODEL_ID = "../../model_weight/rinalmo-mega"
    HF_HOME = "../../model"  # 可按需修改/置空

    # 1. 序列嵌入
    m = RiNALMoModel(
        model_name="RiNALMo-giga",
        model_path=MODEL_ID,
        hf_home=HF_HOME,
        device=None,          # 自动选 GPU/CPU
        use_mlm_head=False,   # 先跑嵌入提取
    )

    dna_list = [
        "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATTCGTTAGC",
        "AGCGTACGTTAG",
        "AGTTTCCCGGAA",
    ]

    embs = m.embed_sequences(
        dna_list,
        pooling="mean",
        exclude_special=True,
        truncation=True,
    )
    print("Embedding shape:", embs.shape)   # (3, hidden_size)

    # 2. PLL 评分（需要 MLM 头）
    m_mlm = RiNALMoModel(
        model_name="RiNALMo-giga",
        model_path=MODEL_ID,
        hf_home=HF_HOME,
        device=None,
        use_mlm_head=True,
    )
    pll = m_mlm.score_sequences(dna_list, batch_size=128)
    print("PLL score:", pll)

# models/lucavirus_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Optional, Literal, Union, Dict
import os
import sys
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from .base_model import BaseModel

Pooling = Literal["mean", "cls", "token"]


class LucaVirusModel(BaseModel):
    """
    LucaVirus embedding 适配器（仅实现 get_embedding）

    严格参考官方示例：
      inputs = tokenizer(seq, seq_type="gene"/"prot", return_tensors="pt", add_special_tokens=True)
      outputs = model(**inputs)
      last_hidden = outputs.last_hidden_state  # [B, L, H]
      mean pooling: last_hidden[:, 1:-1, :].mean(dim=1)
      cls  pooling: last_hidden[:, 0, :]
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        device: Optional[str] = None,
        torch_dtype: Union[str, torch.dtype] = "auto",
        force_download: bool = False,
    ):
        super().__init__(model_name=model_name, model_path=model_path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # 将模型路径添加到 sys.path，以便 transformers 能够正确导入自定义模块
        # 这对于包含特殊字符（如连字符和点号）的模型路径很重要
        model_path_abs = os.path.abspath(self.model_path)
        if model_path_abs not in sys.path:
            sys.path.insert(0, model_path_abs)
        print(f"model_path_abs: {model_path_abs}")
        print(f"self.model_path: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            force_download=force_download,
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            task_level="token_level",
            task_type="embedding",
            trust_remote_code=True,
            force_download=force_download,
            torch_dtype=None if torch_dtype == "auto" else torch_dtype,
        )
        self.model.to(self.device)
        self.model.eval()

        # pad id：优先 pad_token_id，其次 padding_idx，最后 0
        self.pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if self.pad_id is None:
            self.pad_id = getattr(self.tokenizer, "padding_idx", 0)
        if self.pad_id is None:
            self.pad_id = 0

    @staticmethod
    def _infer_seq_type(seq: str) -> str:
        s = seq.strip().upper()
        gene_vocab = set("ACGTUN-")
        return "gene" if set(s).issubset(gene_vocab) else "prot"

    def _tokenize_one(self, seq: str, seq_type: str) -> Dict[str, torch.Tensor]:
        """
        完全按官方示例对单条序列 tokenize。
        返回 dict(tensor): 每个 tensor shape 为 [1, L]
        """
        inputs = self.tokenizer(
            seq,
            seq_type=seq_type,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _pad_batch(self, feats: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        将若干条单样本 tokenization 结果手动 pad 成 batch。
        兼容 input_ids / attention_mask / token_type_ids（如果某个 key 不存在就跳过）
        """
        max_len = max(int(f["input_ids"].shape[1]) for f in feats)
        bsz = len(feats)

        batch: Dict[str, torch.Tensor] = {}

        # input_ids
        input_ids = torch.full((bsz, max_len), int(self.pad_id), dtype=torch.long, device=self.device)
        for i, f in enumerate(feats):
            ids = f["input_ids"][0]
            L = int(ids.shape[0])
            input_ids[i, :L] = ids.long()
        batch["input_ids"] = input_ids

        # attention_mask（若不存在则构造）
        if "attention_mask" in feats[0]:
            attn = torch.zeros((bsz, max_len), dtype=torch.long, device=self.device)
            for i, f in enumerate(feats):
                m = f["attention_mask"][0]
                L = int(m.shape[0])
                attn[i, :L] = m.long()
            batch["attention_mask"] = attn
        else:
            batch["attention_mask"] = (batch["input_ids"] != int(self.pad_id)).long()

        # token_type_ids（可选）
        if "token_type_ids" in feats[0]:
            tti = torch.zeros((bsz, max_len), dtype=torch.long, device=self.device)
            for i, f in enumerate(feats):
                t = f["token_type_ids"][0]
                L = int(t.shape[0])
                tti[i, :L] = t.long()
            batch["token_type_ids"] = tti

        return batch

    @torch.no_grad()
    def get_embedding(
        self,
        sequences: List[str],
        layer_name: str = "mean",
        batch_size: int = 16,
        seq_type: Optional[Literal["gene", "prot"]] = "gene",
        pool: Literal["mean", "cls", "token"] = "mean",
        return_numpy: bool = True,
    ) -> List[np.ndarray]:
        if not sequences:
            return []

        ln = (layer_name or "mean").lower()
        if ln in ("mean", "avg", "mean_pool", "mean_pooling"):
            pooling: Pooling = "mean"
        elif ln in ("cls", "cls_pool", "cls_pooling"):
            pooling = "cls"
        elif ln in ("token", "tokens", "matrix", "last_hidden", "last_hidden_state"):
            pooling = "token"
        else:
            raise ValueError(f"Unsupported layer_name='{layer_name}'. Use: mean/cls/token")

        def _stype(s: str) -> str:
            return seq_type if seq_type is not None else self._infer_seq_type(s)

        outs: List[np.ndarray] = []

        for start in tqdm(range(0, len(sequences), batch_size), desc="Getting LucaVirus embeddings"):
            batch_seqs = sequences[start : start + batch_size]

            # 逐条 tokenization（官方单条写法），然后 pad 成 batch
            feats = [self._tokenize_one(s, _stype(s)) for s in batch_seqs]
            enc = self._pad_batch(feats)

            outputs = self.model(**enc)
            h = outputs.last_hidden_state  # [B, L, H]
            attn = enc.get("attention_mask", None)

            if pooling == "cls":
                vec = h[:, 0, :]  # CLS
                outs.extend(vec.detach().cpu().numpy())
                continue

            if pooling == "mean":
                # 官方示例对单条是 [0, 1:-1, :] mean，这里我们对 batch 做同等逻辑；
                # 同时若有 padding，用 mask 排除 padding，并且排除 CLS/SEP
                if attn is None:
                    vec = h[:, 1:-1, :].mean(dim=1)
                    outs.extend(vec.detach().cpu().numpy())
                else:
                    mask = attn.clone()
                    mask[:, 0] = 0  # drop CLS
                    lengths = attn.sum(dim=1)  # include CLS/SEP
                    last_pos = torch.clamp(lengths - 1, min=0)  # SEP index
                    for i in range(mask.size(0)):
                        mask[i, int(last_pos[i].item())] = 0  # drop SEP

                    mask_f = mask.unsqueeze(-1).to(h.dtype)
                    denom = mask_f.sum(dim=1).clamp_min(1.0)
                    vec = (h * mask_f).sum(dim=1) / denom
                    outs.extend(vec.detach().cpu().numpy())
                continue

            # token-level：去 CLS/SEP + 去 padding
            if attn is None:
                for i in range(h.size(0)):
                    outs.append(h[i, 1:-1, :].detach().cpu().numpy())
            else:
                lengths = attn.sum(dim=1).tolist()
                for i in range(h.size(0)):
                    L = int(lengths[i])
                    if L <= 2:
                        outs.append(h[i, 0:0, :].detach().cpu().numpy())
                    else:
                        outs.append(h[i, 1 : L - 1, :].detach().cpu().numpy())

        return outs


# ------------------------------------------------------------
# 官方风格测试：python -m models.lucavirus_model
# ------------------------------------------------------------
if __name__ == "__main__":
    CKPT = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/LucaVirus-default-step3.8M"

    enc = LucaVirusModel(
        model_name="lucavirus-default",
        model_path=CKPT,
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype="auto",
        force_download=False,
    )

    # gene
    nucleotide_sequence = "ATGCGTACGTTAGC"
    e_mean = enc.get_embedding([nucleotide_sequence], layer_name="mean", batch_size=1, seq_type="gene")[0]
    e_cls = enc.get_embedding([nucleotide_sequence], layer_name="cls", batch_size=1, seq_type="gene")[0]
    print("gene len:", len(nucleotide_sequence))
    print("gene mean emb shape:", e_mean.shape)
    print("gene cls  emb shape:", e_cls.shape)
    print("gene mean emb[:10]:", e_mean[:10])
    print("-" * 50)

    # prot
    protein_sequence = "MKTLLILTAVVLL"
    p_mean = enc.get_embedding([protein_sequence], layer_name="mean", batch_size=1, seq_type="prot")[0]
    p_cls = enc.get_embedding([protein_sequence], layer_name="cls", batch_size=1, seq_type="prot")[0]
    print("prot len:", len(protein_sequence))
    print("prot mean emb shape:", p_mean.shape)
    print("prot cls  emb shape:", p_cls.shape)
    print("prot mean emb[:10]:", p_mean[:10])
    print("-" * 50)

    # token-level
    tok = enc.get_embedding([nucleotide_sequence], layer_name="token", batch_size=1, seq_type="gene")[0]
    print("gene token emb shape (no CLS/SEP):", tok.shape)

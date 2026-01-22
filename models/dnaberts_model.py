from __future__ import annotations

from typing import List, Optional, Union, Literal

import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from .base_model import BaseModel

Pooling = Literal["mean", "max", "cls"]


class DNABERTSModel(BaseModel):
    """
    DNABERT-S 适配器（官方示例：zhihan1996/DNABERT-S）

    说明：
    - DNABERT-S 直接对原始 DNA 序列进行 BPE/Tokenizer 编码
    - 默认使用 mean pooling 生成序列级 embedding
    """

    def __init__(
        self,
        model_name: str,
        model_path: str = "zhihan1996/DNABERT-S",
        device: Optional[str] = None,
        hf_home: Optional[str] = None,
        max_length: Optional[int] = None,
    ):
        super().__init__(model_name, model_path)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hf_home = hf_home
        self.max_length = max_length

        self._load_model()

    def _load_model(self):
        if self.hf_home:
            os.environ["HF_HOME"] = self.hf_home

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        ).to(self.device).eval()

        self.model_max_len = (
            self.max_length
            or getattr(self.model.config, "max_position_embeddings", None)
            or getattr(self.tokenizer, "model_max_length", None)
            or 512
        )

        print(
            f"[DNABERTSModel] loaded on {self.device}, model_max_len={self.model_max_len}"
        )

    def _preprocess(self, seq: str) -> str:
        s = "".join(seq.split()).upper().replace("U", "T")
        return s

    @torch.no_grad()
    def get_embedding(
        self,
        sequences: Union[str, List[str]],
        layer_name: Optional[str] = None,
        batch_size: int = 64,
        pool: Pooling = "mean",
        exclude_special: bool = True,
        truncation: bool = True,
        return_numpy: bool = True,
        max_length: Optional[int] = None,
        **kwargs,
    ) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(sequences, str):
            sequences = [sequences]

        processed = [self._preprocess(seq) for seq in sequences]
        results = []

        for i in tqdm(range(0, len(processed), batch_size), desc="DNABERT-S Embedding"):
            batch_seqs = processed[i : i + batch_size]

            encoded = self.tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=truncation,
                max_length=max_length or self.model_max_len,
            )

            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )[0]
            token_embeddings = outputs

            if pool == "cls":
                emb = token_embeddings[:, 0, :]
            else:
                mask = attention_mask.clone()
                if exclude_special:
                    special_ids = {
                        tid
                        for tid in [
                            self.tokenizer.cls_token_id,
                            self.tokenizer.sep_token_id,
                            self.tokenizer.pad_token_id,
                        ]
                        if tid is not None
                    }
                    if special_ids:
                        for sid in special_ids:
                            mask = mask * (input_ids != sid)

                mask = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                if pool == "mean":
                    sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
                    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                    emb = sum_embeddings / sum_mask
                elif pool == "max":
                    token_embeddings[mask == 0] = -1e9
                    emb = torch.max(token_embeddings, dim=1)[0]
                else:
                    raise ValueError(f"Unknown pooling method: {pool}")

            results.append(emb.cpu())

            del input_ids, attention_mask, outputs, token_embeddings
            torch.cuda.empty_cache()

        final = torch.cat(results, dim=0)
        return final.numpy() if return_numpy else final

    def embed_sequences(self, *args, **kwargs):
        return self.get_embedding(*args, **kwargs)


if __name__ == "__main__":
    # 使用示例
    model = DNABERTSModel(
        model_name="DNABERT-S",
        model_path="zhihan1996/DNABERT-S",
    )

    dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
    embedding = model.get_embedding([dna], pool="mean", return_numpy=False)
    print(embedding.shape)  # torch.Size([1, 768])

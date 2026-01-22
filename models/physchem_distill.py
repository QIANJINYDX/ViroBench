import os
import sys
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from .base_model import BaseModel

# Add Caduceus path
CADUCEUS_PATH = (
    "/mnt/shared-storage-user/dnacoding/jiangfeifei/VEP-evaluator/models/caduceus"
)
if CADUCEUS_PATH not in sys.path:
    sys.path.append(CADUCEUS_PATH)

try:
    from caduceus.modeling_caduceus import Caduceus, CaduceusPreTrainedModel
    from caduceus.configuration_caduceus import CaduceusConfig
except ImportError:
    # Fallback or mock for linter/IDE if path is not accessible during editing
    print("Warning: Could not import Caduceus. Make sure the path is correct.")

    class CaduceusPreTrainedModel(nn.Module):
        pass

    class Caduceus(nn.Module):
        pass

    class CaduceusConfig:
        pass


class DNATokenizer:
    def __init__(self):
        # Standard mapping. N (unknown) maps to 0 usually, or a specific token.
        # Let's use: PAD=0, N=1, A=2, C=3, G=4, T=5
        self.token_map = {
            'N': 1,
            'A': 2,
            'C': 3,
            'G': 4,
            'T': 5,
        }
        self.pad_token_id = 0
        self.vocab_size = 6

    def encode(self, sequence: str, max_len: int = None) -> torch.LongTensor:
        """
        Converts a DNA string to a tensor of indices.
        """
        sequence = sequence.upper()
        ids = [self.token_map.get(base, 1) for base in sequence] # Default to N=1 if unknown char
        
        if max_len is not None:
            if len(ids) > max_len:
                ids = ids[:max_len]
            else:
                # Pad
                ids = ids + [self.pad_token_id] * (max_len - len(ids))
        
        return torch.tensor(ids, dtype=torch.long)

    def batch_encode(self, sequences: List[str], max_len: int = None) -> torch.LongTensor:
        """
        Batch encodes a list of sequences.
        """
        if max_len is None:
            max_len = max(len(s) for s in sequences)
            
        tensors = [self.encode(s, max_len) for s in sequences]
        return torch.stack(tensors)


class RegressionHead(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CaduceusForMultiHeadPhysChemRegression(CaduceusPreTrainedModel):
    """
    Caduceus backbone + multiple regression heads.
    Each feature (defined by group_slices) has its own regression head.
    """

    def __init__(
        self,
        config: CaduceusConfig,
        d_out: int,
        loss_weight: float = 100.0,  # 按照百分比
        group_slices=None,
        head_hidden_dim: int = 64,
    ):
        super().__init__(config)
        self.backbone = Caduceus(config)
        self.d_model = config.d_model
        self.loss_weight = loss_weight
        self.group_slices = group_slices
        
        # Initialize multiple regression heads if slices are provided
        if self.group_slices:
            self.heads = nn.ModuleDict()
            for name, start, end in self.group_slices:
                head_d_out = end - start
                if head_d_out > 0:
                    self.heads[name] = RegressionHead(self.d_model, head_d_out, hidden_dim=head_hidden_dim)
        else:
            # Fallback to a single head named "all" if no groups are defined
            self.heads = nn.ModuleDict({
                "all": RegressionHead(self.d_model, d_out, hidden_dim=head_hidden_dim)
            })

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels_physchem: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Backbone forward
        outputs = self.backbone(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        # Extract hidden states (Batch, Seq, Dim)
        if isinstance(outputs, dict) or hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs[0]

        # Global mean pooling over sequence length -> (Batch, Dim)
        h_pooled = hidden_states.mean(dim=1)

        # Multi-head Prediction
        preds_list = []
        loss = None
        
        if self.group_slices:
            # Process each head according to group_slices order
            group_losses = []
            loss_fct = nn.MSELoss()
            
            for name, start, end in self.group_slices:
                if end <= start:
                    continue
                
                if name in self.heads:
                    head = self.heads[name]
                    pred_g = head(h_pooled)
                    preds_list.append(pred_g)

                    if labels_physchem is not None:
                        label_g = labels_physchem[:, start:end]
                        group_losses.append(loss_fct(pred_g, label_g))
            
            if preds_list:
                preds = torch.cat(preds_list, dim=1)
            else:
                # Should not happen if slices are valid
                preds = torch.zeros(h_pooled.size(0), 0, device=h_pooled.device)

            if labels_physchem is not None:
                if group_losses:
                    loss = self.loss_weight * sum(group_losses) / len(group_losses)
                else:
                    loss = torch.tensor(0.0, device=h_pooled.device)

        else:
            # Single head fallback
            head = self.heads["all"]
            preds = head(h_pooled)
            
            if labels_physchem is not None:
                loss_fct = nn.MSELoss()
                loss = self.loss_weight * loss_fct(preds, labels_physchem)

        if not return_dict:
            output = (preds,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return {
            "loss": loss,
            "preds": preds,
            "hidden_states": outputs.hidden_states
            if hasattr(outputs, "hidden_states")
            else None,
        }


class PhyschemDistillModel(BaseModel):
    """
    Adapter that exposes the distilled physchem Caduceus checkpoint through the
    common BaseModel interface, providing both scalar scores and pooled
    embeddings.
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = torch.float32,
        head_hidden_dim: int = 128,
        score_reduce: Literal["mean", "sum", "max", "first", "l2"] = "mean",
        max_length: Optional[int] = None,
    ):
        super().__init__(model_name, model_path)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.head_hidden_dim = head_hidden_dim
        self.score_reduce = score_reduce.lower()
        self.explicit_max_length = max_length
        self.tokenizer = DNATokenizer()

        self._load_model()

    def _load_model(self) -> None:
        if "CaduceusConfig" not in globals():
            raise ImportError("Caduceus modules are unavailable; check CADUCEUS_PATH.")

        self.config = CaduceusConfig.from_pretrained(self.model_path)
        self.group_slices = self._normalize_group_slices(
            getattr(self.config, "physchem_group_slices", None)
        )
        d_out = getattr(self.config, "physchem_dim", None)
        if d_out is None and self.group_slices:
            d_out = max(slice_[2] for slice_ in self.group_slices)
        if d_out is None:
            raise ValueError("Could not determine physchem output dimension from config.")

        self.model = CaduceusForMultiHeadPhysChemRegression(
            self.config,
            d_out=d_out,
            group_slices=self.group_slices,
            head_hidden_dim=self.head_hidden_dim,
        )

        weight_file = os.path.join(self.model_path, "pytorch_model.bin")
        if not os.path.exists(weight_file):
            raise FileNotFoundError(f"Missing checkpoint weights: {weight_file}")

        state = torch.load(weight_file, map_location="cpu")
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[PhyschemDistillModel] Missing keys when loading state_dict: {missing}")
        if unexpected:
            print(
                f"[PhyschemDistillModel] Unexpected keys when loading state_dict: {unexpected}"
            )

        if self.torch_dtype is not None:
            self.model = self.model.to(self.torch_dtype)

        self.model.to(self.device)
        self.model.eval()

        self.max_length = self.explicit_max_length or getattr(
            self.config, "max_position_embeddings", None
        )

    @staticmethod
    def _normalize_group_slices(
        group_slices: Optional[List[List[Union[str, int]]]]
    ) -> Optional[List[Tuple[str, int, int]]]:
        if not group_slices:
            return None
        normalized: List[Tuple[str, int, int]] = []
        for item in group_slices:
            if len(item) != 3:
                continue
            name, start, end = item
            normalized.append((str(name), int(start), int(end)))
        return normalized or None

    def _prepare_batch(
        self, sequences: List[str], max_length: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        if not sequences:
            raise ValueError("`sequences` cannot be empty.")

        target_len = max(len(seq) for seq in sequences)
        if max_length is not None:
            target_len = min(target_len, int(max_length))
        if self.max_length is not None:
            target_len = min(target_len, int(self.max_length))

        input_ids = self.tokenizer.batch_encode(sequences, max_len=target_len)
        attn_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return input_ids.to(self.device), attn_mask.to(self.device)

    @staticmethod
    def _reduce_preds(preds: Tensor, reduce: str) -> Tensor:
        reduce = reduce.lower()
        preds = preds.to(torch.float32)

        if reduce == "mean":
            return preds.mean(dim=-1)
        if reduce == "sum":
            return preds.sum(dim=-1)
        if reduce == "max":
            return preds.max(dim=-1).values
        if reduce == "first":
            return preds[:, 0]
        if reduce == "l2":
            return torch.norm(preds, p=2, dim=-1)
        raise ValueError(f"Unsupported reduction strategy: {reduce}")

    def _pool_hidden(
        self, hidden: Tensor, attn_mask: Tensor, pooling: Literal["mean", "last", "max", "cls"]
    ) -> Tensor:
        pooling = pooling.lower()
        mask = attn_mask.unsqueeze(-1).to(hidden.dtype)

        if pooling == "mean":
            summed = (hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            return summed / denom
        if pooling == "max":
            masked_hidden = hidden.masked_fill(mask == 0, float("-inf"))
            return masked_hidden.max(dim=1).values
        if pooling == "last":
            lengths = mask.sum(dim=1).squeeze(-1).long().clamp_min(1)
            idx = lengths - 1
            batch_range = torch.arange(hidden.size(0), device=hidden.device)
            return hidden[batch_range, idx]
        if pooling == "cls":
            return hidden[:, 0, :]
        raise ValueError(f"Unsupported pooling method: {pooling}")

    @torch.no_grad()
    def predict_physchem(
        self,
        sequences: Union[str, List[str]],
        batch_size: int = 64,
        max_length: Optional[int] = None,
        return_numpy: bool = True,
    ):
        seq_list = [sequences] if isinstance(sequences, str) else list(sequences)
        if not seq_list:
            return np.empty((0, 0)) if return_numpy else torch.empty(0, 0)

        preds_chunk: List[Tensor] = []
        iterator = tqdm(
            range(0, len(seq_list), batch_size),
            desc="Physchem regression",
            leave=False,
        )
        for start in iterator:
            batch = seq_list[start : start + batch_size]
            input_ids, _ = self._prepare_batch(batch, max_length=max_length)
            outputs = self.model(input_ids=input_ids, return_dict=True)
            batch_preds = outputs["preds"] if isinstance(outputs, dict) else outputs[0]
            preds_chunk.append(batch_preds.detach().to(torch.float32).cpu())

        preds_tensor = torch.cat(preds_chunk, dim=0)
        return preds_tensor.numpy() if return_numpy else preds_tensor

    @torch.no_grad()
    def score_sequences(
        self,
        sequences: List[str],
        batch_size: int = 64,
        aggregate: Optional[str] = None,
        max_length: Optional[int] = None,
    ) -> List[float]:
        if not sequences:
            return []

        reduce = (aggregate or self.score_reduce).lower()
        scores: List[float] = []

        iterator = tqdm(
            range(0, len(sequences), batch_size),
            desc="Physchem scoring",
            leave=False,
        )
        for start in iterator:
            batch = sequences[start : start + batch_size]
            input_ids, _ = self._prepare_batch(batch, max_length=max_length)
            outputs = self.model(input_ids=input_ids, return_dict=True)
            preds = outputs["preds"] if isinstance(outputs, dict) else outputs[0]
            batch_scores = self._reduce_preds(preds, reduce)
            scores.extend(batch_scores.detach().cpu().tolist())

        return scores

    @torch.no_grad()
    def get_embedding(
        self,
        sequences: Union[str, List[str]],
        batch_size: int = 32,
        pool: Optional[Literal["mean", "last", "max", "cls"]] = None,
        pooling: Literal["mean", "last", "max", "cls"] = "mean",
        layer: Optional[int] = -1,
        layer_name: Optional[str] = None,
        max_length: Optional[int] = None,
        return_numpy: bool = True,
    ):
        seq_list = [sequences] if isinstance(sequences, str) else list(sequences)
        if not seq_list:
            return np.empty((0, 0)) if return_numpy else torch.empty(0, 0)

        effective_pool = (pool or pooling or "mean").lower()
        layer_index = layer if layer is not None else -1
        if layer_name not in (None, ""):
            try:
                layer_index = int(layer_name)
            except ValueError:
                print(
                    f"[PhyschemDistillModel] layer_name '{layer_name}' is not supported; "
                    "falling back to default layer."
                )
                layer_index = -1

        pooled_vecs: List[Tensor] = []

        iterator = tqdm(
            range(0, len(seq_list), batch_size),
            desc="Physchem embedding",
            leave=False,
        )
        for start in iterator:
            batch = seq_list[start : start + batch_size]
            input_ids, attn_mask = self._prepare_batch(batch, max_length=max_length)
            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

            preds = None
            if isinstance(outputs, dict):
                preds = outputs.get("preds")
            elif isinstance(outputs, (tuple, list)) and outputs:
                preds = outputs[0]

            if preds is None:
                raise RuntimeError("PhyschemDistillModel forward did not return preds.")

            pooled_vecs.append(preds.detach().to(torch.float32).cpu())

        embeddings = torch.cat(pooled_vecs, dim=0)
        return embeddings.numpy() if return_numpy else embeddings

# python -m models.physchem_distill
# # # ---------- 自测 ----------
if __name__ == "__main__":
    # 根据你本地模型位置修改
    MODEL_DIR = "/mnt/shared-storage-user/dnacoding/jiangfeifei/VEP-evaluator/models/Physchem-distill"

    m = PhyschemDistillModel(
        model_name="physchem-distill",
        model_path=MODEL_DIR,
        device=None,  # 自动挑选 GPU/CPU
        score_reduce="mean",
    )

    toy_sequences = [
        "ACGTACGTACGT",
        "AAAACCCCGGGGTTTT",
        "TGCATGCATGCA",
    ]

    # 1) 获取完整理化性质向量
    physchem = m.predict_physchem(toy_sequences, batch_size=2)
    print("physchem shape:", physchem.shape)  # (N, physchem_dim)

    # 2) 聚合成单个分数（默认为 mean）
    scores = m.score_sequences(toy_sequences, batch_size=2)
    print("scores:", scores)

    # 3) 提取序列表示（可选 mean/max/last/cls）
    embeddings = m.get_embedding(
        toy_sequences,
        batch_size=2,
        pooling="mean",
        layer=-1,
    )
    print("embedding shape:", embeddings.shape)  # (N, hidden_dim)

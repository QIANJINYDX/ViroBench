try:
    from .base_model import BaseModel
except ImportError:
    # Allow running as a script directly
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from model.base_model import BaseModel
from typing import List, Union
import torch
import numpy as np
from tqdm import tqdm
import math
import warnings
from typing import List, Union, Dict, Any, Tuple
import torch.nn.functional as F


class Evo2Model(BaseModel):
    def __init__(self, model_name: str, model_path: str):
        super().__init__(model_name, model_path)
        import sys
        sys.path.append("/mnt/oss_chenxin/wuyucheng/workspace/evo2_val_total")
        from evo2 import Evo2
        self.evo2_instance = Evo2(model_name=model_name, local_path=model_path)
        self.evo2_instance.model.eval()

    @torch.no_grad()
    def get_embedding(
        self,
        sequences: Union[str, List[str]],
        layer_name: str = "blocks.28.mlp.l3",
        batch_size: int = 64,
        return_numpy: bool = True,
        pool: bool = False,# 占位用，无实际作用
    ) -> Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor]:
        """
        按 Evo2 官方示例方式获取 embeddings：
        - tokenize 原始序列
        - 构造 shape=(1, L) 的 input_ids（不做 padding）
        - 调用 evo2(input_ids, return_embeddings=True, layer_names=[layer_name])

        Returns:
            - 若输入是单条字符串：返回该序列的 token 级 embeddings，shape=(1, L, D)
            - 若输入是 List[str]：
                - 若所有序列的 L 相同：返回 shape=(N, L, D)
                - 否则返回一个 list，每个元素为 shape=(1, L_i, D)
        """
        # device：尽量贴近官方示例，优先使用 cuda:0（若可用）
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 规范输入
        if isinstance(sequences, str):
            seq_list = [sequences]
            is_single = True
        else:
            seq_list = list(sequences)
            is_single = False

        tokenizer = self.evo2_instance.tokenizer
        self.evo2_instance.model.eval()

        outs: List[torch.Tensor] = []
        total = len(seq_list)
        if total == 0:
            if return_numpy:
                return np.empty((0, 0, 0))
            return torch.empty(0, 0, 0)

        for i in tqdm(
            range(0, total, batch_size),
            desc="Evo2 get_embedding",
            total=(total + batch_size - 1) // batch_size,
        ):
            chunk = seq_list[i: i + batch_size]
            for sequence in chunk:
                input_ids = torch.tensor(
                    tokenizer.tokenize(sequence),
                    dtype=torch.long,
                ).unsqueeze(0).to(dev)

                _outputs, embeddings = self.evo2_instance(
                    input_ids, return_embeddings=True, layer_names=[layer_name]
                )
                if layer_name not in embeddings:
                    raise KeyError(
                        f"Layer '{layer_name}' not found in embeddings.")

                emb = embeddings[layer_name]
                outs.append(emb.detach().to(torch.float32).cpu())

        if is_single:
            out = outs[0] if outs else torch.empty(0, 0, 0)
            return out.numpy() if return_numpy else out

        if not outs:
            out_multi = torch.empty(0, 0, 0)
            return out_multi.numpy() if return_numpy else out_multi

        # 若所有序列长度一致，则拼成一个 tensor/ndarray，方便直接看 shape
        first_shape = tuple(outs[0].shape)
        if all(tuple(t.shape) == first_shape for t in outs):
            stacked = torch.cat(outs, dim=0)  # (N, L, D)
            return stacked.numpy() if return_numpy else stacked

        # 否则保留 list（不同长度无法直接 stack，除非做 padding）
        if return_numpy:
            return [t.numpy() for t in outs]
        return outs

    @torch.no_grad()
    def generate(
        self,
        prompt_seqs: List[str] = ["ACGT"],
        n_tokens: int = 400,
        temperature: float = 1.0,
        top_k: int = 4,
    ):
        """
        按 Evo2 官方示例方式进行生成：

            output = evo2_model.generate(
                prompt_seqs=["ACGT"], n_tokens=400, temperature=1.0, top_k=4
            )
            print(output.sequences[0])
        """
        self.evo2_instance.model.eval()
        output = self.evo2_instance.generate(
            prompt_seqs=prompt_seqs,
            n_tokens=n_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        generated_seqs = output.sequences
        return generated_seqs
    @staticmethod
    def _get_pad_id(tokenizer) -> int:
        """Evo2 CharLevelTokenizer 可能没 pad_id；用一个合法 token 作为 padding（不计入loss）"""
        for name in ["pad_id", "pad_token_id"]:
            if hasattr(tokenizer, name):
                v = getattr(tokenizer, name)
                if v is not None:
                    return int(v)
        # fallback: 用 'A' 的 token id
        aid = tokenizer.tokenize("A")
        if isinstance(aid, (list, tuple)) and len(aid) > 0:
            return int(aid[0])
        raise RuntimeError("Cannot infer pad_id from tokenizer.")

    @staticmethod
    def _extract_logits(model_out: Any) -> torch.Tensor:
        """
        兼容不同返回形式：
        - tensor
        - (tensor, None)
        - ((tensor, None), other) 之类的嵌套
        """
        if torch.is_tensor(model_out):
            return model_out

        if isinstance(model_out, (list, tuple)):
            if len(model_out) >= 1 and torch.is_tensor(model_out[0]):
                return model_out[0]
            if len(model_out) >= 1 and isinstance(model_out[0], (list, tuple)):
                inner = model_out[0]
                if len(inner) >= 1 and torch.is_tensor(inner[0]):
                    return inner[0]

        raise RuntimeError(
            f"Cannot extract logits from model output type={type(model_out)}"
        )

    @staticmethod
    def _safe_exp(nll: float, max_nll: float = 700.0) -> float:
        """安全计算 exp(nll)，避免溢出。"""
        if math.isnan(nll):
            return float("nan")
        if math.isinf(nll):
            return float("inf") if nll > 0 else 0.0
        if nll > max_nll:
            warnings.warn(
                f"NLL value {nll:.2f} exceeds max_nll {max_nll}, returning inf for PPL"
            )
            return float("inf")
        try:
            return math.exp(nll)
        except OverflowError:
            return float("inf")
    @torch.no_grad()
    def get_ppl(
        self,
        sequences: Union[str, List[str]],
        prompt_len_chars: int = 128,
        batch_size: int = 8,
        use_cuda: bool = True,
        sort_by_length: bool = True,
        prepend_bos: bool = False,
        return_details: bool = True,
    ) -> Union[List[Dict[str, Any]], float, List[float]]:
        """
        Evo2 conditional perplexity（按字符切分）：
        prompt = seq[:prompt_len_chars]
        continuation = seq[prompt_len_chars:]
        只对 continuation 部分计算 token-level 平均 NLL，并返回：
        - avg_nll_token: 平均 NLL（per token, nat）
        - token_count: continuation 的有效 token 数（参与loss）
        - char_count: continuation 的字符数（len(seq)-prompt_len_chars, clamp>=0）
        - ppl: exp(avg_nll_token)

        return_details=True  -> List[dict]
        return_details=False -> ppl 列表（与输入顺序对齐）；若输入是 str 返回 float
        """
        # -------- normalize input --------
        if isinstance(sequences, str):
            seq_list = [sequences]
            is_single = True
        else:
            seq_list = list(sequences)
            is_single = False

        evo2_model = self.evo2_instance
        model = evo2_model.model
        tokenizer = evo2_model.tokenizer
        pad_id = int(self._get_pad_id(tokenizer))
        model.eval()

        # -------- device --------
        target_device = torch.device("cuda:0") if (use_cuda and torch.cuda.is_available()) else torch.device("cpu")
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = target_device

        device = model_device
        if device.type == "cpu" and target_device.type == "cuda":
            try:
                model.to(target_device)
                device = target_device
            except Exception as e:
                warnings.warn(f"Failed to move model to CUDA, fallback CPU. err={str(e)[:120]}")
                device = model_device

        # -------- tokenize all sequences once --------
        # 保存 (orig_idx, seq_str, ids, prompt_chars, cont_chars)
        seq_infos: List[Tuple[int, str, List[int], int, int]] = []
        for i, s in enumerate(seq_list):
            s = s if s is not None else ""
            try:
                ids = list(map(int, tokenizer.tokenize(s)))
                if prepend_bos and hasattr(tokenizer, "bos_id") and tokenizer.bos_id is not None:
                    bos = int(tokenizer.bos_id)
                    ids = [bos] + ids

                p_chars = max(0, int(prompt_len_chars))
                cont_chars = max(0, len(s) - p_chars)  # 只按字符定义 continuation 长度
                seq_infos.append((i, s, ids, p_chars, cont_chars))
            except Exception as e:
                warnings.warn(f"Failed to tokenize sequence {i}: {str(e)}")
                seq_infos.append((i, s, [], max(0, int(prompt_len_chars)), 0))

        if sort_by_length:
            seq_infos.sort(key=lambda x: len(x[2]))

        results: List[Dict[str, Any]] = []

        with torch.inference_mode():
            for b_start in tqdm(range(0, len(seq_infos), batch_size), desc="Evo2 PPL batches"):
                batch = seq_infos[b_start:b_start + batch_size]
                B = len(batch)
                lens = [len(x[2]) for x in batch]
                max_len = max(lens) if lens else 0

                if max_len == 0:
                    for orig_i, s, ids, p_chars, cont_chars in batch:
                        results.append({
                            "sequence_id": orig_i,
                            "avg_nll_token": float("nan"),
                            "token_count": 0,
                            "char_count": int(cont_chars),
                            "ppl": float("nan"),
                            "error": "empty_sequence",
                        })
                    continue

                # pad batch
                input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
                valid_mask = torch.zeros((B, max_len), dtype=torch.bool)

                # 这里不再用 prompt_token_len（避免重新 tokenize 前缀造成不一致）
                # 我们只保留每条的 cont_chars，然后基于 token 数推断对应 token 段
                cont_chars_list: List[int] = [0] * B
                seq_token_lens: List[int] = [0] * B

                for r, (_orig_i, _s, ids, _p_chars, cont_chars) in enumerate(batch):
                    L = len(ids)
                    seq_token_lens[r] = L
                    cont_chars_list[r] = int(cont_chars)
                    if L == 0:
                        continue
                    input_ids[r, :L] = torch.tensor(ids, dtype=torch.long)
                    valid_mask[r, :L] = True

                input_ids = input_ids.to(device, non_blocking=True)
                valid_mask = valid_mask.to(device, non_blocking=True)

                # forward -> logits
                out = evo2_model(input_ids)
                logits = self._extract_logits(out)  # (B, L, V)
                if logits.dim() != 3:
                    raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)} (expect B,L,V)")
                logits = logits.float()

                # causal shift: predict token t+1 from token t
                shift_logits = logits[:, :-1, :].contiguous()   # (B, L-1, V)
                shift_labels = input_ids[:, 1:].contiguous()    # (B, L-1)
                shift_valid  = valid_mask[:, 1:].contiguous()   # (B, L-1)

                # token-level loss (NLL) for every position
                V = shift_logits.size(-1)
                token_nll = F.cross_entropy(
                    shift_logits.view(-1, V),
                    shift_labels.view(-1),
                    reduction="none",
                ).view(B, -1)  # (B, L-1)

                # -------- build continuation mask (char-based) --------
                # 对 char-level tokenizer：token_len ~= char_len (+bos?)
                # 我们按 “末尾 cont_chars 个字符” 来选 continuation 的 labels 区间。
                # 对应到 shift_labels 的位置索引是 [start, end)：
                #   seq_token_len = L
                #   shift_labels 长度 = L-1, 对应原 token index 1..L-1
                # continuation 的 token labels 数（期望）≈ cont_chars（char-level）
                # start = (L-1) - cont_tok_cnt
                cont_mask = torch.zeros_like(shift_valid)

                for r in range(B):
                    L = int(seq_token_lens[r])          # token count (含可能 bos)
                    if L <= 1:
                        continue

                    # 目标：只计算 continuation 部分的 “label positions”
                    # cont_chars_list[r] 是按原始字符串算的字符数（不含 bos）
                    cont_tok_cnt = int(cont_chars_list[r])

                    # clamp：不能超过 (L-1)
                    cont_tok_cnt = max(0, min(cont_tok_cnt, L - 1))
                    if cont_tok_cnt == 0:
                        continue

                    end = L - 1
                    start = end - cont_tok_cnt
                    if start < end:
                        cont_mask[r, start:end] = True

                final_mask = shift_valid & cont_mask

                # sum/mean per sample
                nll_sum = (token_nll * final_mask).sum(dim=1)          # (B,)
                tok_cnt = final_mask.sum(dim=1)                         # (B,)

                for r, (orig_i, s, ids, p_chars, cont_chars) in enumerate(batch):
                    c = int(tok_cnt[r].item())
                    if c <= 0:
                        results.append({
                            "sequence_id": orig_i,
                            "avg_nll_token": float("nan"),
                            "token_count": 0,
                            "char_count": int(cont_chars),
                            "ppl": float("nan"),
                            "error": "no_continuation_tokens",
                        })
                    else:
                        avg_nll_token = float((nll_sum[r] / tok_cnt[r]).item())
                        ppl = float(self._safe_exp(avg_nll_token))
                        results.append({
                            "sequence_id": orig_i,
                            "avg_nll_token": avg_nll_token,
                            "token_count": c,
                            "char_count": int(cont_chars),
                            "ppl": ppl,
                        })

        # restore original order
        results.sort(key=lambda x: x["sequence_id"])

        if return_details:
            return results

        ppl_list = [float(x.get("ppl", float("nan"))) for x in results]
        if is_single:
            return ppl_list[0] if ppl_list else float("nan")
        return ppl_list


# # # ========== 可选：脚本直接运行的快速自测 ==========
if __name__ == "__main__":
    # 按你的真实路径修改
    model_name = "evo2_7b_base"
    model_path = f"/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/{model_name}/{model_name}.pt"

    model = Evo2Model(
        model_name=model_name,
        model_path=model_path,
    )
    sequences = ["ACGT"*100, "ATGC"*100]
    layer_name = "blocks.28.mlp.l3"

    emb = model.get_embedding(
        sequences, layer_name=layer_name, batch_size=1, return_numpy=True)
    print("Embeddings shape: ", emb.shape)

    output = model.generate(
        prompt_seqs=sequences, n_tokens=8, temperature=1.0, top_k=4)
    print(output)

    # 2) 只要 ppl 数值（与输入顺序对齐）
    ppl_vals = model.get_ppl(
        sequences,
        prompt_len_chars=128,
        batch_size=2,
        use_cuda=True,
        return_details=True,
    )
    print("ppl:", ppl_vals)

# evo1_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Optional, Tuple, Union, Literal
import torch.nn as nn
import numpy as np
import torch
from tqdm import tqdm
import inspect
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
PoolT = Literal["final", "mean"]

# 兼容两种使用方式：
# 1) 作为包导入：from model.evo1_model import Evo1Model
# 2) 作为脚本运行：python model/evo1_model.py
try:
    from .base_model import BaseModel
except ImportError:
    # Allow running as a script directly
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from model.base_model import BaseModel
import math
import warnings
from typing import Dict, Any
import torch.nn.functional as F



def _nested_getattr(root: nn.Module, dotted: str):
    obj = root
    for part in dotted.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def _find_blocklist(model: nn.Module) -> Tuple[Optional[nn.ModuleList], Optional[str]]:
    """
    试图在常见位置找到 block 的 ModuleList，并返回 (module_list, name)。
    """
    candidates = [
        "blocks", "layers", "h",
        "transformer.blocks", "transformer.layers", "transformer.h",
        "encoder.layers", "model.layers",
    ]
    for name in candidates:
        ml = _nested_getattr(model, name)
        if isinstance(ml, nn.ModuleList) and len(ml) > 0:
            return ml, name
    # 兜底：遍历找第一个像样的 ModuleList
    for m in model.modules():
        if isinstance(m, nn.ModuleList) and len(m) > 0:
            return m, None
    return None, None


class Evo1Model(BaseModel):
    """
    Evo-1 模型适配器

    继承 BaseModel，实现 score_sequences 接口，返回每条序列的平均 log-likelihood。
    使用方式：
        m = Evo1Model(
            model_name='evo-1-131k-base',
            model_path='/mnt/s3mount/model_weight/evo-1-131k-base',
            config_path='/mnt/s3mount/model_weight/evo-1-131k-base/evo-1-131k-base_inference.yml',
            hf_home='/mnt/s3mount/model_weight/cache',    # 可选
            device=None                                   # 'cuda:0' | 'cpu' | None 自动判定
        )
        scores = m.score_sequences(['ACGT', 'AAAACCCCGGGGTTTT'])
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        config_path: Optional[str] = None,
        hf_home: Optional[str] = None,
        device: Optional[str] = None,
    ):
        super().__init__(model_name, model_path)
        self.config_path = config_path
        self.hf_home = hf_home
        self.device = self._pick_device(device)
        self._load_model()

    @staticmethod
    def _pick_device(device: Optional[str]) -> str:
        if device is not None:
            return device
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        """加载 Evo-1 模型与分词器（借助你的 Evo 封装）。"""
        if self.hf_home:
            os.environ["HF_HOME"] = self.hf_home  # 与你的测试脚本一致
        
        # 修复 transformers 动态模块加载问题
        # 清理可能损坏的 transformers_modules 缓存，避免 ModuleNotFoundError: No module named 'transformers_modules.evo-1'
        import shutil
        import glob
        cache_base = self.hf_home if self.hf_home else os.path.expanduser("~/.cache")
        transformers_cache_dir = os.path.join(cache_base, "huggingface", "transformers")
        if os.path.exists(transformers_cache_dir):
            # 尝试清理可能损坏的 transformers_modules 缓存
            transformers_modules_dir = os.path.join(transformers_cache_dir, "transformers_modules")
            if os.path.exists(transformers_modules_dir):
                # 清理与当前模型相关的缓存（evo-1 开头的所有缓存）
                model_cache_pattern = os.path.join(transformers_modules_dir, "evo-1*")
                for cache_path in glob.glob(model_cache_pattern):
                    try:
                        if os.path.isdir(cache_path):
                            shutil.rmtree(cache_path)
                            print(f"[Evo1Model] Cleaned corrupted cache: {cache_path}")
                    except Exception as e:
                        print(f"[Evo1Model] Warning: Failed to clean cache {cache_path}: {e}")
        
        # 设置环境变量，避免 transformers 动态模块加载问题
        # 强制使用本地配置文件，不尝试动态导入 transformers_modules
        os.environ["HF_HUB_OFFLINE"] = "1"  # 强制离线模式，只使用本地文件

        try:
            from evo import Evo  # 你环境里的 evo 包
        except Exception as e:
            raise RuntimeError(
                f"无法导入 evo 包，请确认已安装并在当前环境可见：{e}"
            )

        # 实例化你已有的 Evo 封装
        self.evo_model = Evo(
            model_name=self.model_name,
            device=self.device,
            config_path=self.config_path,
            hf_model_name=self.model_path,   # 你的本地权重目录
        )

        # 取出底层模型与 tokenizer
        self.model = self.evo_model.model
        self.tokenizer = self.evo_model.tokenizer

        # 统一设置为 eval & 设备
        self.model.to(self.device)
        self.model.eval()

        # 尝试获取 pad_id（若不存在，将在 batch 评分时退化为单条）
        self.pad_id = self._detect_pad_id(self.tokenizer)

        # 检测是否支持基于 cache 的块式前向（仅做占位，默认仍整段前向）
        self._supports_cache_forward, self._cache_param_name = self._detect_cache_forward(
            self.model)

        print(
            f"[Evo1Model] Loaded on {self.device} | pad_id={self.pad_id} | supports_cache={self._supports_cache_forward}")

    @staticmethod
    def _detect_pad_id(tokenizer) -> Optional[int]:
        # 常见属性名尝试
        for name in ("pad_id", "pad_token_id"):
            if hasattr(tokenizer, name):
                val = getattr(tokenizer, name)
                if isinstance(val, int) and val >= 0:
                    return val
        # 某些 tokenizer 可能有 token_to_id 方法
        tok2id = getattr(tokenizer, "token_to_id", None)
        if callable(tok2id):
            for cand in ("<pad>", "[PAD]", "<PAD>"):
                try:
                    pid = tok2id(cand)
                    if isinstance(pid, int) and pid >= 0:
                        return pid
                except Exception:
                    pass
        return None  # 未能可靠获取

    @staticmethod
    def _detect_cache_forward(model) -> Tuple[bool, Optional[str]]:
        """检查 forward 是否有 cache / past_key_values / kv_cache 等参数。"""
        try:
            sig = inspect.signature(model.forward)
            for pname in ("cache", "past_key_values", "past_kv", "kv_cache", "prev_cache"):
                if pname in sig.parameters:
                    return True, pname
        except Exception:
            pass
        return False, None

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
            for i in tqdm(range(0, len(sequences), batch_size), desc="Scoring sequences"):
                batch_seqs = sequences[i:i + batch_size]
                batch_scores = self._score_batch(batch_seqs)
                all_scores.extend(batch_scores)

        return all_scores

    def _score_batch(self, sequences: List[str]) -> List[float]:
        """对单个批次评分"""
        # 1. 分词和预处理
        input_ids, seq_lengths = self._prepare_batch(sequences)

        # 2. 模型前向传播
        logits, _ = self.model(input_ids)

        # 3. 计算 log probabilities
        log_probs = self._compute_log_probabilities(logits, input_ids)

        # 4. 聚合得分
        scores = self._aggregate_scores(log_probs, seq_lengths)

        return scores

    def _prepare_batch(self, sequences: List[str]):
        """分词和批处理"""
        seq_lengths = [len(seq) for seq in sequences]

        all_token_ids = []
        for seq in sequences:
            ids = self.tokenizer.tokenize(seq)
            # 若返回的是字符串 token，则转成 id
            if ids and isinstance(ids[0], str) and hasattr(self.tokenizer, "convert_tokens_to_ids"):
                ids = self.tokenizer.convert_tokens_to_ids(ids)
            ids = [int(x) for x in ids]  # 统一为纯 int
            all_token_ids.append(ids)

        # Padding 到相同长度（并确保 pad_id 为 int）
        pad_id = int(getattr(self.tokenizer, "pad_token_id",
                     getattr(self.tokenizer, "pad_id", 0)) or 0)
        max_length = max(len(ids) for ids in all_token_ids)

        # （可选，若你有模型长度上限）
        model_max_len = getattr(self, "model_max_len", None)
        if model_max_len is not None:
            max_length = min(max_length, int(model_max_len))
        max_length = min(max_length, getattr(self, "seq_len_cap", 16384))

        padded_ids = []
        for ids in all_token_ids:
            if len(ids) < max_length:
                padded = ids + [pad_id] * (max_length - len(ids))
            else:
                padded = ids[:max_length]
            padded_ids.append(padded)

        input_ids = torch.tensor(
            padded_ids, dtype=torch.long, device=self.device)
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
    def generate(
        self,
        prompt_seqs: Union[str, List[str]] = "ACGT",
        # 兼容旧入参名（你项目里可能之前用的是 prompt=...）
        prompt: Optional[Union[str, List[str]]] = None,
        n_samples: int = 3,
        n_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 4,
        top_p: float = 1.0,
        cached_generation: bool = True,
        batched: bool = True,
        prepend_bos: bool = False,
        device: Optional[str] = None,
        verbose: int = 1,
    ):
        """
        基于 Evo 官方 `scripts.generate` 逻辑的生成接口。

        官方脚本核心调用：
            output_seqs, output_scores = evo.generate(
                [prompt] * n_samples, model, tokenizer,
                n_tokens=..., temperature=..., top_k=..., top_p=...,
                cached_generation=..., batched=..., prepend_bos=...,
                device=..., verbose=...
            )

        Args:
            prompt_seqs: 单个 prompt（str）或 prompts 列表（List[str]）。
                - 若为 str：会复制为 [prompt] * n_samples
                - 若为 List[str]：
                    - len == 1：同样复制为 n_samples
                    - len == n_samples：直接使用
                    - 其他长度：将原样传给官方 generate（由其自行处理/报错）
            prompt: **兼容别名**；若传入，将覆盖 prompt_seqs（但不允许两者同时显式传入）。
            n_samples: 一次采样多少条（当 prompt 为 str 时生效）
            n_tokens: 生成 token 数
            temperature/top_k/top_p: 采样参数
            cached_generation: 是否使用 KV cache
            batched: 是否使用 batch generation
            prepend_bos: 是否在输入前加 BOS
            device: 覆盖设备（默认使用初始化时的 self.device）
            verbose: 官方 generate 的 verbose

        Returns:
            (output_seqs, output_scores)：与官方 `evo.generate` 一致
        """
        try:
            from evo import generate as evo_generate  # 官方函数
        except Exception as e:
            raise RuntimeError(f"无法导入 evo.generate，请确认 evo 包可用：{e}")

        dev = device or getattr(self, "device", None) or (
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        # 确保模型在正确设备且为 eval 模式
        self.model.to(dev)
        self.model.eval()

        # prompt / prompt_seqs 统一入口
        if prompt is not None:
            # 若用户显式传了 prompt，同时也显式传了 prompt_seqs（不是默认值），则报错避免歧义
            # 由于 Python 无法区分“是否显式传默认值”，这里做最小化的安全检查：
            # 当 prompt_seqs 不是默认 "ACGT" 且 prompt 也不为空时，认为两者都想生效。
            if prompt_seqs != "ACGT":
                raise ValueError("请勿同时传入 prompt 与 prompt_seqs；二者是同义参数。")
            prompt_seqs = prompt

        if isinstance(prompt_seqs, str):
            prompts = [prompt_seqs] * int(n_samples)
        else:
            prompt_list = list(prompt_seqs)
            if len(prompt_list) == 1 and int(n_samples) > 1:
                prompts = prompt_list * int(n_samples)
            else:
                prompts = prompt_list

        output_seqs, output_scores = evo_generate(
            prompts,
            self.model,
            self.tokenizer,
            n_tokens=int(n_tokens),
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            cached_generation=bool(cached_generation),
            batched=bool(batched),
            prepend_bos=bool(prepend_bos),
            device=str(dev),
            verbose=int(verbose),
        )
        return output_seqs

    @torch.no_grad()
    def get_embedding(
        self,
        sequences: Union[str, List[str]],
        layer_index: int = 4,                 # 按“索引”取层，支持负索引
        batch_size: int = 64,
        pool: PoolT = "final",                  # "final" | "mean"
        prepend_bos: bool = False,
        return_numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:

        # 设备
        try:
            dev = next(self.model.parameters()).device
        except Exception:
            dev = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")

        # 规范输入
        if isinstance(sequences, str):
            seq_list = [sequences]
        else:
            seq_list = list(sequences)

        tok = self.tokenizer
        pad_id = getattr(tok, "pad_id", 0)
        bos_id = getattr(tok, "eod_id", None) or getattr(tok, "bos_id", None)

        # 找到 block 列表
        blocklist, blocklist_name = _find_blocklist(self.model)
        if blocklist is None or len(blocklist) == 0:
            raise RuntimeError(
                "未能在 Evo1 模型中找到 blocks/layers（ModuleList）。请检查模型结构。")

        # 负索引转正
        if layer_index < 0:
            layer_index = len(blocklist) + layer_index
        if not (0 <= layer_index < len(blocklist)):
            raise IndexError(
                f"layer_index 越界：{layer_index}（共有 {len(blocklist)} 层）")

        all_vecs: List[torch.Tensor] = []

        # 池化函数
        def _pool_from(emb: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
            # emb: [B, T, D], input_ids: [B, T]
            B, T, D = emb.shape
            if pool == "final":
                lengths = (input_ids != pad_id).long().sum(
                    dim=1).clamp_min(1)   # [B]
                idx = (lengths - 1).clamp_min(0)
                bidx = torch.arange(B, device=emb.device)
                # [B, D]
                return emb[bidx, idx, :]
            elif pool == "mean":
                # [B, T, 1]
                mask = (input_ids != pad_id).unsqueeze(-1)
                denom = mask.sum(dim=1).clamp_min(
                    1)                              # [B, 1]
                # [B, D]
                return (emb * mask).sum(dim=1) / denom
            else:
                raise ValueError("pool 仅支持 'final' 或 'mean'")

        # 小批处理
        for st in tqdm(range(0, len(seq_list), batch_size), desc="Getting embedding"):
            chunk = seq_list[st: st + batch_size]
            # tokenize -> 动态 pad 到本 batch 最大长度
            token_lists: List[List[int]] = []
            for s in chunk:
                ids = tok.tokenize(s)  # List[int]
                if prepend_bos and bos_id is not None:
                    ids = [bos_id] + ids
                token_lists.append(ids)
            max_len = max(len(x) for x in token_lists) if token_lists else 1
            B = len(token_lists)
            input_ids = torch.full((B, max_len), pad_id,
                                   dtype=torch.long, device=dev)
            for i, ids in enumerate(token_lists):
                if ids:
                    input_ids[i, :len(ids)] = torch.tensor(
                        ids, dtype=torch.long, device=dev)

            # 注册 hook 到目标 block
            captured: List[torch.Tensor] = []

            def _hook(_module, _inp, _out):
                # 兼容 (hidden,*) / Tensor
                if isinstance(_out, (tuple, list)):
                    val = _out[0]
                else:
                    val = _out
                if not torch.is_tensor(val):
                    # 少数实现可能返回 dict，尝试拿第一个 tensor
                    if isinstance(val, dict) and len(val) > 0:
                        for v in val.values():
                            if torch.is_tensor(v):
                                val = v
                                break
                captured.append(val.detach())

            handle = blocklist[layer_index].register_forward_hook(_hook)

            try:
                # 只走最小可行前向：不传多余参数
                logits, _ = self.model(input_ids)
            finally:
                handle.remove()

            if not captured:
                raise RuntimeError(
                    f"hook 未捕获到层输出（层={layer_index}）。"
                    f"{' 你可以打印模型结构确认 block 名称：' + str(blocklist_name) if blocklist_name else ''}"
                )

            emb = captured[-1]  # [B, T, D]
            if emb.dim() != 3:
                raise RuntimeError(f"捕获到的张量维度异常：{emb.shape}（期望 [B, T, D]）")

            pooled = _pool_from(emb, input_ids)
            all_vecs.append(pooled.to(torch.float32).cpu())

        out = torch.cat(all_vecs, dim=0) if all_vecs else torch.empty(0, 0)
        return out.numpy() if return_numpy else out
        # =============================
    # PPL helpers
    # =============================
    @staticmethod
    def _get_pad_id_fallback(tokenizer) -> int:
        """尽量从 tokenizer 推断 pad_id；若无则 fallback 到 'A' 的 token id"""
        for name in ("pad_id", "pad_token_id"):
            if hasattr(tokenizer, name):
                v = getattr(tokenizer, name)
                if v is not None:
                    try:
                        return int(v)
                    except Exception:
                        pass

        # 某些 tokenizer 可能提供 token_to_id
        tok2id = getattr(tokenizer, "token_to_id", None)
        if callable(tok2id):
            for cand in ("<pad>", "[PAD]", "<PAD>"):
                try:
                    pid = tok2id(cand)
                    if pid is not None:
                        return int(pid)
                except Exception:
                    pass

        # fallback: 用 'A'
        try:
            aid = tokenizer.tokenize("A")
            if isinstance(aid, (list, tuple)) and len(aid) > 0:
                return int(aid[0])
        except Exception:
            pass

        # 最后兜底
        return 0

    @staticmethod
    def _extract_logits(model_out: Any) -> torch.Tensor:
        """
        兼容不同返回形式：
        - tensor
        - (tensor, cache)
        - ((tensor, cache), other) 之类嵌套
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

        raise RuntimeError(f"Cannot extract logits from model output type={type(model_out)}")

    @staticmethod
    def _safe_exp(nll: float, max_nll: float = 700.0) -> float:
        """安全 exp，避免溢出"""
        if math.isnan(nll):
            return float("nan")
        if math.isinf(nll):
            return float("inf") if nll > 0 else 0.0
        if nll > max_nll:
            warnings.warn(f"NLL {nll:.2f} > {max_nll}, return inf PPL")
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
        prepend_bos: bool = True,
        return_details: bool = True,
        ppl_mode: Literal["token", "char"] = "token",
    ) -> Union[List[Dict[str, Any]], float, List[float]]:
        """
        Evo1 conditional perplexity（与 Evo2 口径对齐 + 补全统计字段）

        切分方式（按字符）：
          prompt = seq[:prompt_len_chars]
          continuation = seq[prompt_len_chars:]

        计算方式：
        - char 口径（严格官方对齐）：使用 evo.scoring.logits_to_logprobs 得到每个字符位置的 logprob
          avg_nll_char = -mean(logprob_chars over continuation chars)
          ppl_char = exp(avg_nll_char)

        - token 口径（更通用，更适合跨 tokenizer 比较）：对 continuation 的 token 做 CE
          avg_nll_token = mean(CE over continuation tokens)
          ppl_token = exp(avg_nll_token)

        返回字段（return_details=True）：
          - ppl: 按 ppl_mode 返回（token 或 char）
          - ppl_token, avg_nll_token, token_count
          - ppl_char,  avg_nll_char,  char_count
          - 还有 prompt_len_chars 等信息
        """
        # ---------- normalize input ----------
        if isinstance(sequences, str):
            seq_list = [sequences]
            is_single = True
        else:
            seq_list = list(sequences)
            is_single = False

        # ---------- device ----------
        target_device = torch.device("cuda:0") if (use_cuda and torch.cuda.is_available()) else torch.device("cpu")
        self.model.to(target_device)
        self.model.eval()

        # ---------- import official scoring helpers ----------
        from evo.scoring import prepare_batch, logits_to_logprobs  # 官方实现

        # ---------- build (padded) batch with official align ----------
        # prepare_batch 会 pad，并且 prepend_bos=True 时会加 eod/bos，logits_to_logprobs(trim_bos=True) 会对齐到字符位置
        input_ids, seq_lengths_chars = prepare_batch(
            seq_list, self.tokenizer, prepend_bos=bool(prepend_bos), device=str(target_device)
        )

        # forward
        logits, _ = self.model(input_ids)          # [B, L, V]
        logits = logits.float()                    # CE 用 float 更稳
        B, L, V = logits.shape

        # ============================================================
        # (A) 官方逐字符 logprob：用于 ppl_char / avg_nll_char / char_count
        # ============================================================
        # logprobs_chars: [B, max_chars]，位置 0..len(seq)-1 对齐到原始字符（trim_bos=True）
        logprobs_chars = logits_to_logprobs(logits, input_ids, trim_bos=True)

        # ============================================================
        # (B) token 口径 NLL：用于 ppl_token / avg_nll_token / token_count
        # ============================================================
        # causal shift
        shift_logits = logits[:, :-1, :].contiguous()   # [B, L-1, V]
        shift_labels = input_ids[:, 1:].contiguous()    # [B, L-1]

        # 有效 token mask（排除 pad）
        pad_id = self._get_pad_id_fallback(self.tokenizer)
        valid_mask = (shift_labels != pad_id)           # [B, L-1]

        # 计算每个 token 的 CE
        token_ce = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            reduction="none",
        ).view(B, L - 1)                                # [B, L-1]

        # continuation 的 token mask：按 “prompt_len_chars 对应的 prompt token 数” 来截断
        # 注意：prompt_len_chars 是按字符切分，所以要用 tokenizer.tokenize(prompt) 得到 prompt token 数
        results: List[Dict[str, Any]] = []

        for i, s in enumerate(seq_list):
            s = s or ""
            full_chars = len(s)

            # 以官方 seq_lengths_chars 为准（有时可能被截断）
            # seq_lengths_chars[i] 通常等于 len(s)
            L_chars = int(seq_lengths_chars[i])
            if L_chars <= 0:
                d = {
                    "sequence_id": i,
                    "char_count": 0,
                    "token_count": 0,
                    "avg_nll_char": float("nan"),
                    "avg_nll_token": float("nan"),
                    "ppl_char": float("nan"),
                    "ppl_token": float("nan"),
                    "ppl": float("nan"),
                    "error": "empty_sequence",
                }
                results.append(d)
                continue

            p_chars = min(int(prompt_len_chars), L_chars)
            cont_chars = L_chars - p_chars

            # ---------- char-level stats ----------
            if cont_chars <= 0:
                avg_nll_char = float("nan")
                ppl_char = float("nan")
            else:
                mean_logprob_char = logprobs_chars[i, p_chars:L_chars].mean().item()
                avg_nll_char = float(-mean_logprob_char)
                ppl_char = float(math.exp(avg_nll_char))

            # ---------- token-level continuation mask ----------
            prompt_str = s[:p_chars]
            try:
                # tokenizer.tokenize 返回 list[int] 或 list[str]
                p_ids = self.tokenizer.tokenize(prompt_str)
                if p_ids and isinstance(p_ids[0], str) and hasattr(self.tokenizer, "convert_tokens_to_ids"):
                    p_ids = self.tokenizer.convert_tokens_to_ids(p_ids)
                prompt_len_tokens = int(len(p_ids))
            except Exception:
                prompt_len_tokens = 0

            # 由于是 next-token prediction，shift 的位置 t 对应预测 input_ids[t+1]
            # continuation 从 “第 prompt_len_tokens 个真实 token” 开始被预测
            # 在 shift 维度上，起点是 (prompt_len_tokens - 1)
            start = max(prompt_len_tokens - 1, 0)

            # 该样本的有效 token 长度（不含 pad）
            # valid_mask 是基于 shift_labels，所以它的长度是 L-1
            seq_tok_len = int(valid_mask[i].sum().item())  # shift 中有效 token 数
            # 在 shift 序列中可用的最大 end 是 seq_tok_len（exclusive）
            end = seq_tok_len

            if end <= start:
                avg_nll_token = float("nan")
                ppl_token = float("nan")
                tok_cnt = 0
            else:
                cont_mask = torch.zeros_like(valid_mask[i])
                cont_mask[start:end] = True
                final_mask = valid_mask[i] & cont_mask

                tok_cnt = int(final_mask.sum().item())
                if tok_cnt <= 0:
                    avg_nll_token = float("nan")
                    ppl_token = float("nan")
                else:
                    nll_sum = (token_ce[i] * final_mask).sum().item()
                    avg_nll_token = float(nll_sum / tok_cnt)
                    ppl_token = float(self._safe_exp(avg_nll_token))

            # ---------- choose ppl ----------
            if ppl_mode == "char":
                ppl = ppl_char
            else:
                ppl = ppl_token

            results.append({
                "sequence_id": i,

                # counts
                "sequence_chars": full_chars,
                "prompt_len_chars": p_chars,
                "char_count": cont_chars,

                "prompt_len_tokens": prompt_len_tokens,
                "token_count": tok_cnt-1,# 减去 bos 的 token

                # char-metric (official-aligned)
                "avg_nll_char": avg_nll_char,
                "ppl_char": ppl_char,

                # token-metric (cross-token comparable)
                "avg_nll_token": avg_nll_token,
                "ppl_token": ppl_token,

                # main output
                "ppl": ppl,
                "mode": ppl_mode,
            })

        # ---------- return ----------
        if return_details:
            return results

        ppl_list = [float(x.get("ppl", float("nan"))) for x in results]
        if is_single:
            return ppl_list[0] if ppl_list else float("nan")
        return ppl_list


    @torch.no_grad()
    def conditional_ppl_official(
        self,
        seqs,
        model,
        tokenizer,
        prompt_len_chars=128,
        device="cuda:0",
    ):
        """
        用 evo.scoring 的官方对齐方式计算 conditional PPL：
        prompt = seq[:prompt_len_chars]
        continuation = seq[prompt_len_chars:]
        只对 continuation 的每字符 logprob 求均值 -> ppl = exp(-mean_logprob)
        """
        from evo.scoring import prepare_batch, logits_to_logprobs  # 官方实现

        if isinstance(seqs, str):
            seqs = [seqs]
            single = True
        else:
            seqs = list(seqs)
            single = False

        # 官方 prepare_batch：prepend_bos=True, bos=tokenizer.eod_id, padding=tokenizer.pad_id
        input_ids, seq_lengths = prepare_batch(
            seqs, tokenizer, prepend_bos=True, device=device
        )

        logits, _ = model(input_ids)  # (B, L, V)
        logprobs = logits_to_logprobs(logits, input_ids, trim_bos=True)  # (B, max_seq_len_chars)

        out = []
        for i, s in enumerate(seqs):
            L = int(seq_lengths[i])              # 官方这里就是 len(seq)（字符数）
            p = min(int(prompt_len_chars), L)    # continuation 从 p 开始
            cont_len = L - p
            if cont_len <= 0:
                out.append(float("nan"))
                continue

            # logprobs[i] 的位置 0..L-1 对应每个字符的 logprob（已按官方 BOS 对齐）
            mean_logprob = logprobs[i, p:L].mean().item()
            ppl = math.exp(-mean_logprob)
            out.append(ppl)

        return out[0] if single else out



# # ========== 可选：脚本直接运行的快速自测 ==========
if __name__ == "__main__":
    # 按你的真实路径修改
    HF_HOME = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/cache"
    os.environ["HF_HOME"] = HF_HOME

    MODEL_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/evo-1-131k-base"
    CFG_PATH = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/evo-1-131k-base/evo-1-131k-base_inference.yml"

    model = Evo1Model(
        model_name="evo-1-131k-base",
        model_path=MODEL_DIR,
        config_path=CFG_PATH,
        hf_home=HF_HOME,
        device=None,  # 自动选择
    )
    # s = "AG"*4096
    # seqs = [s]
    sequences = ["ACGT"*100, "ATGC"*100]
    scores = model.get_embedding(sequences, batch_size=2)
    print(scores)
    print(scores.shape)
    output_seqs, output_scores = model.generate(
        prompt_seqs=sequences, n_tokens=8, temperature=1.0, top_k=4)
    print(output_seqs)

    ppl_vals = model.get_ppl(
        sequences,
        prompt_len_chars=128,
        batch_size=2,
        use_cuda=True,
        return_details=False,
    )
    print("ppl:", ppl_vals)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    ppl2 = model.conditional_ppl_official(
        sequences,
        model=model.model,
        tokenizer=model.tokenizer,
        prompt_len_chars=128,
        device=dev,
    )
    print("official-style conditional ppl:", ppl2)

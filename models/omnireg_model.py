# models/omnireg_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
from typing import List, Optional, Literal, Union, Tuple,Dict,Any

import torch
import numpy as np
from torch import nn

from transformers import AutoTokenizer

from tqdm import tqdm

from .base_model import BaseModel

Pooling = Literal["mean", "max", "cls"]


class OmniRegGPTModel(BaseModel):
    """
    OmniReg-GPT 适配器

    - 使用 GENA-LM tokenizer 对 DNA 序列进行编码
    - Backbone 使用 OmniReg-GPT 提供的 Hybrid Transformer（从其仓库中 import）
    - 提供通用 embedding 接口 + 自回归 log-likelihood 评分接口 (score_sequences)

    重要说明：
        1. 该类不重新实现 OmniReg-GPT 的网络结构，而是假定你已经
           `git clone https://github.com/wawpaopao/OmniReg-GPT.git`
           并能从中 import 对应的模型类（这里默认叫 `HybridTransformer`，你可按实际仓库改）。
        2. 预训练权重需要从 Zenodo 下载，并用 torch.load 加载 state_dict。
        3. 由于官方代码细节无法完全获知，这里对 HybridTransformer 的构造参数、
           checkpoint 的 key 名做了尽量通用的处理，如有不符，你只需改 _build_model/_load_state_dict 两个函数。

    Args:
        model_name: 逻辑名（比如 "OmniReg-GPT"）
        model_path: 预训练权重路径（.pt/.pth），或包含 state_dict 的目录/文件
        tokenizer_path: GENA-LM tokenizer 本地路径（如 "./gena-lm-bert-large-t2t"）
        omnireg_repo_path: OmniReg-GPT 仓库路径，用于 sys.path.append 后 import
        hf_home: HF 缓存目录 (可选)，会写入环境变量 HF_HOME，用于 tokenizer
        device: "cuda:0" / "cpu" / None(自动选择)
        max_length: 最大 token 长度；为 None 时优先读 model.config.max_position_embeddings，
                    再退而求其次用 tokenizer.model_max_length
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        tokenizer_path: str,
        omnireg_repo_path: Optional[str] = None,
        hf_home: Optional[str] = None,
        device: Optional[str] = None,
        max_length: Optional[int] = None,
    ):
        super().__init__(model_name, model_path)

        self.hf_home = hf_home
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer_path = tokenizer_path
        self.omnireg_repo_path = omnireg_repo_path
        self.max_length = max_length

        # ---- 环境变量 / sys.path 配置 ----
        if self.hf_home:
            os.environ["HF_HOME"] = self.hf_home

        if self.omnireg_repo_path:
            if self.omnireg_repo_path not in sys.path:
                sys.path.append(self.omnireg_repo_path)
        # print(sys.path)

        # ---- 加载 tokenizer 和模型 ----
        self._load_tokenizer()
        self._load_model()

    # =========================
    # 加载 tokenizer & 模型
    # =========================
    def _load_tokenizer(self):
        print(f"[OmniRegGPTModel] DEBUG: Loading tokenizer from: {self.tokenizer_path}")
        
        # 简化策略：GENA-LM tokenizer 不需要远程代码，直接加载本地文件
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                use_fast=True,
                trust_remote_code=False,  # 明确关闭远程代码信任
                local_files_only=True,    # 明确只读本地
            )
            print("[OmniRegGPTModel] DEBUG: Successfully loaded tokenizer with local_files_only=True")
            return
        except Exception as e:
            print(f"[OmniRegGPTModel] DEBUG: AutoTokenizer load failed: {e}")
            
        # 回退策略：如果上面失败，尝试直接用 PreTrainedTokenizerFast
        try:
            from transformers import PreTrainedTokenizerFast
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
                self.tokenizer_path,
                local_files_only=True,
            )
            print("[OmniRegGPTModel] DEBUG: Successfully loaded tokenizer with PreTrainedTokenizerFast")
            return
        except Exception as e:
            print(f"[OmniRegGPTModel] DEBUG: PreTrainedTokenizerFast load failed: {e}")

        raise RuntimeError(f"Failed to load tokenizer from {self.tokenizer_path}")

    def _infer_config_from_checkpoint(self, ckpt_path: str) -> dict:
        """
        从 checkpoint 的 state_dict 中推断模型配置。
        
        根据官方实现，OmniReg-GPT 使用以下配置：
        - num_tokens=32000
        - dim=1024
        - depth=12
        - heads=16
        - dim_head=64
        - hierarchies=(1, 8)  # 两个 hierarchy，压缩因子为 1 和 8
        """
        print(f"[OmniRegGPTModel] inferring config from checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        
        # 尝试从常见包装中提取 state_dict
        if isinstance(state, dict):
            for key in ["model", "model_state_dict", "state_dict"]:
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        
        if not isinstance(state, dict):
            raise ValueError(
                f"Checkpoint at {ckpt_path} is not a valid state_dict or wrapped dict."
            )
        
        config = {}
        
        # 从 token_emb.weight 推断 dim（最准确的方法）
        # token_emb.weight 的形状是 [num_tokens, dim]
        if "token_emb.weight" in state:
            config["dim"] = state["token_emb.weight"].shape[1]
        else:
            # 默认值（根据官方实现）
            config["dim"] = 1024
        
        # 从 last_merge.net.0.gamma 推断 hierarchy 数量
        # last_merge 的输入维度是 sum(dims)，即所有 hierarchy 的 dim 之和
        # 如果只有一个 hierarchy，sum(dims) = dim
        # 如果有多个 hierarchy，sum(dims) = dim * num_hierarchies
        if "last_merge.net.0.gamma" in state:
            dim_sum = state["last_merge.net.0.gamma"].shape[0]
            dim = config["dim"]
            num_hierarchies = dim_sum // dim
            
            if num_hierarchies == 1:
                config["hierarchies"] = 1
            elif num_hierarchies == 2:
                # 根据官方实现，通常是 (1, 8)
                config["hierarchies"] = (1, 8)
            else:
                # 如果有其他数量的 hierarchy，使用默认值
                config["hierarchies"] = (1, 8)
        else:
            # 如果找不到 last_merge，尝试通过 compressors 推断
            compressor_keys = [k for k in state.keys() if "compressors." in k]
            if compressor_keys:
                # 有 compressors，说明有多个 hierarchy
                compressor_indices = set()
                for key in compressor_keys:
                    parts = key.split(".")
                    if len(parts) >= 2 and parts[1].isdigit():
                        compressor_indices.add(int(parts[1]))
                
                # 根据官方实现，通常是 (1, 8)
                if len(compressor_indices) >= 1:
                    config["hierarchies"] = (1, 8)
                else:
                    config["hierarchies"] = (1, 8)
            else:
                # 没有 compressors，只有一个 hierarchy
                config["hierarchies"] = 1
        
        # 从 to_logits 推断 vocab_size
        if "to_logits.weight" in state:
            config["num_tokens"] = state["to_logits.weight"].shape[0]
        else:
            config["num_tokens"] = self.tokenizer.vocab_size
        
        # 尝试推断 depth（层数）
        # 通过计算 layers 的数量
        layer_keys = [k for k in state.keys() if k.startswith("layers.")]
        if layer_keys:
            # 找到最大的层索引
            max_layer_idx = -1
            for key in layer_keys:
                parts = key.split(".")
                if len(parts) >= 2 and parts[1].isdigit():
                    max_layer_idx = max(max_layer_idx, int(parts[1]))
            if max_layer_idx >= 0:
                config["depth"] = max_layer_idx + 1
            else:
                config["depth"] = 12  # 默认值（根据官方实现）
        else:
            config["depth"] = 12
        
        # 推断 heads：从 attention 的 to_qkv 权重推断
        # to_qkv 的权重形状是 [dim_inner * 3, dim]，其中 dim_inner = heads * dim_head
        attn_keys = [k for k in state.keys() if "attn.to_qkv.weight" in k]
        if attn_keys:
            # 取第一个 attention 层
            qkv_weight = state[attn_keys[0]]
            dim_inner = qkv_weight.shape[0] // 3
            dim = qkv_weight.shape[1]
            # 假设 dim_head = 64（根据官方实现）
            config["dim_head"] = 64
            config["heads"] = dim_inner // config["dim_head"]
        else:
            config["heads"] = 16  # 默认值（根据官方实现）
            config["dim_head"] = 64
        
        # 推断 window_sizes（如果有多个 hierarchy）
        # 根据官方实现，通常是 (64, None) 或 (16, None)
        # 这里使用默认值，可以根据实际需要调整
        if isinstance(config["hierarchies"], tuple) and len(config["hierarchies"]) > 1:
            config["window_sizes"] = (64, None)  # 默认值，可根据实际调整
        else:
            config["window_sizes"] = None
        
        print(f"[OmniRegGPTModel] inferred config: dim={config.get('dim')}, depth={config.get('depth')}, "
              f"heads={config.get('heads')}, num_tokens={config.get('num_tokens')}, "
              f"hierarchies={config.get('hierarchies')}")
        
        return config

    def _build_model(self, inferred_config: Optional[dict] = None) -> nn.Module:
        """
        构建 OmniReg-GPT 模型骨干。

        ⚠️ 此处只能做通用实现：
            - 默认从 omnireg_repo_path 下的 hybrid_transformer.py 中 import HierarchicalTransformer
            - 构造参数 config 可简单地用一个字典/命名空间，也可以你自己手动改为官方的配置类。

        如果官方仓库中类名或构造方式不同，你只需要修改这里。
        
        Args:
            inferred_config: 从 checkpoint 推断出的配置字典，如果提供则优先使用
        """
        try:
            from hybrid_transformer import HierarchicalTransformer  # type: ignore
        except ImportError as e:
            raise ImportError(
                "无法 import 'HierarchicalTransformer'。请确认 "
                "omnireg_repo_path 设置正确且 OmniReg-GPT 仓库已在 sys.path 中，"
                "或按实际代码修改 _build_model 中的导入逻辑。"
            ) from e

        # 如果提供了推断的配置，优先使用；否则使用默认值
        if inferred_config:
            inferred_hierarchies = inferred_config.get("hierarchies", (1, 8))
            # OmniReg-GPT 的实现假设至少存在两个 hierarchy，
            # 否则 forward 里会访问空列表导致 list index out of range。
            if inferred_hierarchies == 1 or inferred_hierarchies == (1,):
                print(
                    "[OmniRegGPTModel] Warning: inferred hierarchies=1 is not "
                    "supported by HybridTransformer; fallback to (1, 8)."
                )
                inferred_hierarchies = (1, 8)

            config = dict(
                num_tokens=inferred_config.get("num_tokens", self.tokenizer.vocab_size),
                dim=inferred_config.get("dim", 1024),
                depth=inferred_config.get("depth", 12),
                seq_len=self.max_length or 16384,
                heads=inferred_config.get("heads", 16),
                dim_head=inferred_config.get("dim_head", 64),
                ff_mult=4,
                hierarchies=inferred_hierarchies,
                window_sizes=inferred_config.get("window_sizes", (64, None)),
                hierarchical_stride=1,
                use_flash_attn=False,
                recon_loss_weight=0.1,
                prophet_loss_weight=0.0,
            )
        else:
            # 默认配置（根据官方实现）
            config = dict(
                num_tokens=self.tokenizer.vocab_size,
                dim=1024,  # 根据官方实现
                depth=12,  # 根据官方实现
                seq_len=self.max_length or 16384,
                heads=16,
                dim_head=64,
                ff_mult=4,
                hierarchies=(1, 8),  # 根据官方实现，两个 hierarchy
                window_sizes=(64, None),  # 根据官方实现
                hierarchical_stride=1,
                use_flash_attn=False,
                recon_loss_weight=0.1,
                prophet_loss_weight=0.0,
            )

        # HierarchicalTransformer 接受 **config
        model = HierarchicalTransformer(**config)  # 你可以按官方构造接口调整此行
        return model

    def _load_state_dict(self, model: nn.Module, ckpt_path: str):
        """
        加载预训练权重。

        - 优先尝试直接当作 state_dict
        - 否则尝试常见 key: 'model', 'model_state_dict', 'state_dict'
        - 确保权重数据类型与模型一致（转换为 float32）
        """
        print(f"[OmniRegGPTModel] loading weights from {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")

        if isinstance(state, dict):
            # 常见几种包装
            for key in ["model", "model_state_dict", "state_dict"]:
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break

        if not isinstance(state, dict):
            raise ValueError(
                f"Checkpoint at {ckpt_path} is not a valid state_dict or wrapped dict."
            )

        # 检查 checkpoint 中权重的数据类型
        # 保持 checkpoint 的原始数据类型，不进行转换
        # 这样模型会使用与 checkpoint 相同的数据类型
        sample_tensor = next((v for v in state.values() if isinstance(v, torch.Tensor)), None)
        if sample_tensor is not None:
            checkpoint_dtype = sample_tensor.dtype
            print(f"[OmniRegGPTModel] Checkpoint dtype: {checkpoint_dtype}")
        else:
            checkpoint_dtype = None

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[OmniRegGPTModel] Warning: missing keys: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
        if unexpected:
            print(f"[OmniRegGPTModel] Warning: unexpected keys: {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")

    def _load_model(self):
        # 1. 先从 checkpoint 推断配置
        try:
            inferred_config = self._infer_config_from_checkpoint(self.model_path)
        except Exception as e:
            print(f"[OmniRegGPTModel] Warning: failed to infer config from checkpoint: {e}")
            print("[OmniRegGPTModel] Using default config instead.")
            inferred_config = None
        
        # 2. 使用推断的配置构建模型
        self.model = self._build_model(inferred_config)
        
        # 3. 加载权重
        self._load_state_dict(self.model, self.model_path)

        # 4. 移动到设备
        self.model.to(self.device)
        
        # =======================================================
        # 【关键修复】将模型转为 FP16 (Half) 以匹配内部计算精度
        # =======================================================
        # OmniReg 的 Attention 层会将输入强制转为 FP16。
        # 为了避免 "Half != Float" 的错误，我们将权重也统一为 FP16。
        print(f"[OmniRegGPTModel] Converting model to FP16 (Half) to match internal activation dtype...")
        self.model.half() 
        
        self.model.eval()

        # (可选) 再次检查并打印
        param_dtypes = {p.dtype for p in self.model.parameters()}
        print(f"[OmniRegGPTModel] Model params dtypes: {param_dtypes}")

        # 5. 计算最大长度 (保持原有逻辑)
        model_seq_len = getattr(self.model, "seq_len", None)
        tokenizer_max_len = getattr(self.tokenizer, "model_max_length", None)
        
        if tokenizer_max_len is not None and tokenizer_max_len > 1000000:
            tokenizer_max_len = None
        
        self.model_max_len = (
            model_seq_len
            or self.max_length
            or tokenizer_max_len
            or 16384
        )
        
        if self.model_max_len > 1000000:
            print(f"[OmniRegGPTModel] Warning: model_max_len={self.model_max_len} is too large, using 16384 instead")
            self.model_max_len = 16384

        print(
            f"[OmniRegGPTModel] loaded on {self.device}, "
            f"model_max_len={self.model_max_len}, "
            f"vocab_size={self.tokenizer.vocab_size}"
        )

    # =========================
    # 预处理 DNA 序列
    # =========================
    def _preprocess_sequence(self, seq: str) -> str:
        """
        对输入 DNA 序列做最基本的清洗：
        - 去空格 / 换行
        - 大写
        - U -> T
        - 非 ACGTN 统一映射为 N
        """
        s = seq.strip().upper().replace("U", "T")
        out_chars = []
        for ch in s:
            if ch in {"A", "C", "G", "T", "N"}:
                out_chars.append(ch)
            else:
                out_chars.append("N")
        return "".join(out_chars)

    def _tokenize_batch(
        self,
        sequences: List[str],
        truncation: bool = True,
        max_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """将一批 DNA 序列转成 input_ids 和 attention_mask。"""
        proc = [self._preprocess_sequence(s) for s in sequences]

        enc = self.tokenizer(
            proc,
            return_tensors="pt",
            padding=True,
            truncation=truncation,
            max_length=max_length or self.model_max_len,
        )
        input_ids = enc["input_ids"]
        attn_mask = enc.get("attention_mask")
        if attn_mask is None:
            attn_mask = torch.ones_like(input_ids)
        return input_ids.to(self.device), attn_mask.to(self.device)

    # =========================
    # 通用 embedding 接口
    # =========================
    @torch.no_grad()
    def get_embedding(
        self,
        sequences: Union[str, List[str]],
        layer_name: str = None,  # 为了兼容 BaseModel 接口，这里不用
        batch_size: int = 8,
        pool: Pooling = "mean",  # "cls" | "mean" | "max"
        layer_index: int = -1,   # 若支持 hidden_states，可选中间层；否则忽略
        average_reverse_complement: bool = False,  # 是否与反向互补平均
        exclude_special: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        返回每条序列的向量表示 (N, H)。

        对于 GPT 类模型，一般使用最后一层 hidden state:
            - pool="cls" 时取第一个 token 的隐层
            - pool="mean"/"max" 时对有效 token（去除 PAD / 特殊符号）做池化
        """
        if isinstance(sequences, str):
            seq_list = [sequences]
        else:
            seq_list = list(sequences)

        device = self.device
        tok = self.tokenizer
        model = self.model

        def _revcomp(seq: str) -> str:
            tbl = str.maketrans(
                "ACGTRYMKBDHVNacgtrymkbdhvn",
                "TGCAYRKMVHDBNtgcayrkmvhdbn",
            )
            return seq.translate(tbl)[::-1]

        def _forward_hidden(input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
            """
            获取 HierarchicalTransformer 的 hidden states。
            
            HierarchicalTransformer.forward() 只接受 input_ids，不接受 attention_mask。
            它内部会做 ids[:, :-1], ids[:, 1:] 切片，所以返回的 embedding 长度是 T-1。
            为了保持维度一致，我们在输入末尾添加一个 dummy token。
            """
            # HierarchicalTransformer 不接受 attention_mask，只接受 input_ids
            # 注意：forward 方法内部会做 ids[:, :-1], ids[:, 1:] 切片
            # 所以如果我们传入 [B, T]，模型会使用 ids[:, :-1] = [B, T-1] 计算 embedding
            # 返回的 embedding 形状是 [B, T-1, H]
            # 
            # 为了保持维度一致（返回 [B, T, H]），我们在输入末尾添加一个 dummy token
            # 这样模型使用 ids[:, :-1] 时，正好对应原始的 T 个 token
            
            # 在末尾添加一个 dummy token（使用最后一个 token 的重复，或 pad_token）
            B, T = input_ids.shape
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else input_ids[:, -1]
            dummy_token = torch.full((B, 1), pad_id, device=input_ids.device, dtype=input_ids.dtype)
            input_ids_with_dummy = torch.cat([input_ids, dummy_token], dim=1)  # [B, T+1]
            
            # 调用模型获取 hierarchical embeddings
            # return_hierarchical_embeds=True 返回所有 hierarchy 的 embedding 列表
            # return_loss=False 表示不计算 loss
            embeds = model(
                input_ids=input_ids_with_dummy,
                return_loss=False,
                return_hierarchical_embeds=True
            )
            
            # embeds 是一个列表，包含所有 hierarchy 的 embedding
            # 模型使用 ids[:, :-1]，所以返回的 embedding 形状是 [B, T, H]（对应原始的 T 个 token）
            # 根据官方实现，最后一个 hierarchy（索引 -1）通常是用于预测的最高分辨率 embedding
            if isinstance(embeds, (list, tuple)):
                # 使用最后一个 hierarchy 的 embedding（最高分辨率）
                hidden = embeds[-1]  # [B, T, H]
            else:
                # 如果返回的不是列表，直接使用
                hidden = embeds
            
            return hidden  # [B, T, H]

        def _pool(
            hidden: torch.Tensor,
            input_ids: torch.Tensor,
            attn_mask: torch.Tensor,
        ) -> torch.Tensor:
            """
            hidden: [B, T, H]
            """
            if pool == "cls":
                # 对于 GPT，可以用第一个 token 的表示
                return hidden[:, 0, :]

            # 有效位置：基于 attention_mask
            valid = attn_mask.bool()  # [B, T]

            if exclude_special:
                spec_masks = []
                for ids in input_ids:
                    spec = tok.get_special_tokens_mask(
                        ids.tolist(),
                        already_has_special_tokens=True,
                    )
                    spec_masks.append(
                        torch.tensor(spec, device=attn_mask.device, dtype=torch.bool)
                    )
                spec_mask = torch.stack(spec_masks, dim=0)  # [B, T]
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
                # 极端情况下 fallback
                inf_mask = torch.isinf(pooled).any(dim=1)
                if inf_mask.any():
                    pooled[inf_mask] = hidden[inf_mask].max(dim=1).values
                return pooled
            else:
                raise ValueError(f"Unknown pool='{pool}'")

        outputs: List[torch.Tensor] = []

        for st in tqdm(range(0, len(seq_list), batch_size), desc="Getting OmniReg embeddings"):
            batch = seq_list[st : st + batch_size]

            input_ids, attn_mask = self._tokenize_batch(
                batch,
                truncation=truncation,
                max_length=max_length,
            )

            hid_f = _forward_hidden(input_ids, attn_mask)
            vec_f = _pool(hid_f, input_ids, attn_mask)  # [B, H]

            if average_reverse_complement:
                rc_batch = [_revcomp(s) for s in batch]
                rc_input_ids, rc_attn_mask = self._tokenize_batch(
                    rc_batch,
                    truncation=truncation,
                    max_length=max_length,
                )
                hid_r = _forward_hidden(rc_input_ids, rc_attn_mask)
                vec_r = _pool(hid_r, rc_input_ids, rc_attn_mask)
                vec = 0.5 * (vec_f + vec_r)
            else:
                vec = vec_f

            outputs.append(vec.detach().to(torch.float32).cpu())

        out = torch.cat(outputs, dim=0) if outputs else torch.empty(0, 0)
        return out.numpy() if return_numpy else out

    @torch.no_grad()
    def embed_sequences(
        self,
        sequences: List[str],
        batch_size: int = 8,
        pooling: Pooling = "mean",
        exclude_special: bool = True,
        truncation: bool = True,
        return_numpy: bool = True,
    ):
        """
        简化版嵌入接口（与 DNABERTModel 风格一致）
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

    # =========================
    # 自回归 log-likelihood 评分
    # =========================
    @torch.no_grad()
    def score_sequences(
        self,
        sequences: List[str],
        batch_size: int = 1,
    ) -> List[float]:
        """
        对序列进行自回归 log-likelihood 评分：

        - 标准 GPT 方式：log p(x_2..T | x_1..T-1) 的平均 log-prob (per token)
        - 忽略 padding / 特殊 token 的位置

        Returns:
            每条序列的平均 log-prob 得分（越大越好）
        """
        scores: List[float] = []

        pad_id = self.tokenizer.pad_token_id
        # 有些 tokenizer 没有 pad_id，这种情况下我们只依赖 attention_mask
        has_pad = pad_id is not None

        for st in tqdm(range(0, len(sequences), batch_size), desc="Scoring sequences (OmniReg-GPT)"):
            batch = sequences[st : st + batch_size]
            input_ids, attn_mask = self._tokenize_batch(
                batch,
                truncation=True,
                max_length=None,
            )  # [B, T]

            # 构造 labels: 右移一位，对应 next-token prediction
            # logits at t 预测 token at t+1
            # input (model internal): x_0 x_1 ... x_{T-2}
            # logits output: pred_1 pred_2 ... pred_{T-1}
            # targets: x_1 x_2 ... x_{T-1}
            
            # 注意：HierarchicalTransformer 内部对输入做了 ids[:, :-1] 切片
            # 所以 logits 的长度是 T-1
            # 对应的 labels 应该是 input_ids[:, 1:]
            labels = input_ids[:, 1:].clone()

            # 如果有 pad_id，则 pad 位置 mask 掉
            if has_pad:
                labels[input_ids[:, 1:] == pad_id] = -100

            # 前向推理，获取 logits
            # HierarchicalTransformer.forward() 不接受 attention_mask
            # 它内部会做 ids[:, :-1], ids[:, 1:] 切片用于自回归预测
            # 所以我们需要传入完整的 input_ids（包括最后一个token用于预测）
            out = self.model(
                input_ids=input_ids,
                return_loss=False,
            )

            if hasattr(out, "logits"):
                logits = out.logits  # [B, T, V]
            elif isinstance(out, torch.Tensor):
                logits = out
            elif isinstance(out, (tuple, list)):
                logits = out[0]  # 约定 [0] 为 logits 或 last_hidden_state
            else:
                # 如果模型输出的是 hidden states，而不是 logits，你需要在这里加上额外的输出层
                raise ValueError(
                    "Model forward output does not contain logits. "
                    "You may need to adapt OmniReg-GPT forward to return logits."
                )

            log_probs = torch.log_softmax(logits, dim=-1)  # [B, T, V]

            # 只统计 labels != -100 的位置
            # 展开成一维方便索引
            B, T, V = log_probs.shape
            log_probs_flat = log_probs.view(B * T, V)
            labels_flat = labels.view(B * T)

            valid_mask = labels_flat != -100
            if valid_mask.sum() == 0:
                scores.extend([0.0] * len(batch))
                continue

            idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
            selected_log_probs = log_probs_flat[idx, labels_flat[idx]]  # [N_valid]

            # 把每条样本的 token log-prob 再按样本聚合、求平均
            # 这里简化做法：按 batch 内每条样本单独计算
            offset = 0
            for b in range(len(batch)):
                # 当前样本的长度
                valid_positions = (labels[b] != -100).nonzero(as_tuple=False).view(-1)
                if len(valid_positions) == 0:
                    scores.append(0.0)
                    continue

                n = len(valid_positions)
                sample_log_probs = selected_log_probs[offset : offset + n]
                offset += n

                avg_logprob = sample_log_probs.mean().item()
                scores.append(float(avg_logprob))

        return scores
        # =============================
    # PPL helpers + get_ppl
    # =============================
    @staticmethod
    def _extract_logits_from_out(out):
        """
        尽量兼容 OmniReg-GPT / HierarchicalTransformer 的多种返回：
        - out.logits
        - Tensor
        - (Tensor, ...)
        - [Tensor, ...]
        - dict 里某个 Tensor
        """
        if hasattr(out, "logits") and torch.is_tensor(out.logits):
            return out.logits

        if torch.is_tensor(out):
            return out

        if isinstance(out, (tuple, list)):
            for x in out:
                if torch.is_tensor(x):
                    return x
                if hasattr(x, "logits") and torch.is_tensor(x.logits):
                    return x.logits

        if isinstance(out, dict):
            for v in out.values():
                if torch.is_tensor(v):
                    return v
                if hasattr(v, "logits") and torch.is_tensor(v.logits):
                    return v.logits

        raise RuntimeError(f"Cannot extract logits from model output type={type(out)}")

    @torch.no_grad()
    def get_ppl(
        self,
        sequences: Union[str, List[str]],
        *,
        # batch 控制：sliding-window 时这里主要控制外层循环的进度分组（每条序列仍按 window 逐条算）
        batch_size: int = 1,
        truncation: bool = True,
        max_length: Optional[int] = None,     # token max_length（用于 tokenizer truncation）
        exclude_special: bool = True,

        # conditional ppl（二选一）
        prompt_len_chars: Optional[int] = None,
        prompt_len_tokens: Optional[int] = None,

        # tokenizer 行为（强烈建议 conditional + chars 时用 False）
        add_special_tokens: bool = False,

        # sliding window（强烈建议长序列打开）
        use_sliding_window: bool = True,
        max_window_tokens: int = 4096,
        stride: int = 1024,

        # 输出控制
        ppl_mode: Literal["token", "char"] = "token",  # 主输出 ppl 用 token 还是 char
        return_details: bool = True,
    ):
        """
        OmniReg-GPT PPL 计算（next-token NLL）：

        - full PPL: 预测整条序列 (token position 1..L-1) 的平均 NLL -> exp
        - conditional PPL:
            * prompt_len_chars: prompt=seq[:prompt_len_chars]，只对 continuation 对应 token 计 NLL
            * prompt_len_tokens: 只对 token continuation 计 NLL
          注意：prompt_len_chars 与 prompt_len_tokens 只能二选一。

        sliding window：
        - 以 stride 为步长，每次用最多 max_window_tokens 的上下文来计算下一段 token 的 NLL，
          每个 token 只计一次，不重复计分，适合超长序列避免 OOM。

        返回：
          return_details=True -> List[dict]（每条序列一个 dict）
          return_details=False ->
              - 输入是 str: float
              - 输入是 list: List[float]
        """
        import math
        import warnings
        import torch.nn.functional as F

        if (prompt_len_chars is not None) and (prompt_len_tokens is not None):
            raise ValueError("prompt_len_chars 与 prompt_len_tokens 只能二选一。")

        # 如果用 prompt_len_chars，且 add_special_tokens=True，很容易导致 prompt token 长度不对应（[CLS]/[SEP]）
        if prompt_len_chars is not None and add_special_tokens:
            warnings.warn(
                "[OmniRegGPTModel.get_ppl] prompt_len_chars 建议配合 add_special_tokens=False。"
                "已自动将 add_special_tokens 设为 False 以避免 [CLS]/[SEP] 干扰。"
            )
            add_special_tokens = False

        # 规范输入
        single = isinstance(sequences, str)
        seq_list = [sequences] if single else list(sequences)

        # window/stride 合法化
        max_window_tokens = int(max(2, max_window_tokens))
        stride = int(max(1, min(stride, max_window_tokens)))

        def _safe_exp(x: float, max_x: float = 700.0) -> float:
            if math.isnan(x):
                return float("nan")
            if math.isinf(x):
                return float("inf") if x > 0 else 0.0
            if x > max_x:
                return float("inf")
            try:
                return math.exp(x)
            except OverflowError:
                return float("inf")

        def _tokenize_one(s: str) -> torch.Tensor:
            s = self._preprocess_sequence(s)
            enc = self.tokenizer(
                s,
                return_tensors="pt",
                padding=False,
                truncation=bool(truncation),
                max_length=(int(max_length) if max_length is not None else None),
                add_special_tokens=bool(add_special_tokens),
            )
            return enc["input_ids"].to(self.device)  # [1, L]

        def _special_mask(ids_1d: torch.Tensor) -> Optional[torch.Tensor]:
            if not exclude_special:
                return None
            spec = self.tokenizer.get_special_tokens_mask(
                ids_1d.tolist(),
                already_has_special_tokens=True,
            )
            return torch.tensor(spec, device=self.device, dtype=torch.bool)  # [L]

        def _extract_shift_logits(window_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            给定 window_ids: [1, W]
            返回：
              shift_logits: [1, L_eff, V]  (预测 targets)
              targets    : [1, L_eff]      (window_ids[:, 1:...])
            兼容：
              logits.shape[1] == W     => shift_logits = logits[:, :-1]
              logits.shape[1] == W-1   => shift_logits = logits
            """
            out = self.model(input_ids=window_ids, return_loss=False)
            logits = self._extract_logits_from_out(out)  # [1, T_out, V]
            if logits.dim() != 3:
                raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")

            logits = logits.float()
            W = int(window_ids.size(1))
            T_out = int(logits.size(1))
            if T_out == W:
                shift_logits = logits[:, :-1, :]              # [1, W-1, V]
                targets = window_ids[:, 1:]                   # [1, W-1]
            elif T_out == W - 1:
                shift_logits = logits                          # [1, W-1, V]
                targets = window_ids[:, 1:]                    # [1, W-1]
            else:
                # 兜底：取最小可对齐长度
                L_eff = min(T_out, W - 1)
                if L_eff <= 0:
                    return logits.new_zeros((1, 0, logits.size(-1))), window_ids.new_zeros((1, 0))
                shift_logits = logits[:, :L_eff, :]
                targets = window_ids[:, 1:1 + L_eff]
            return shift_logits, targets

        def _nll_on_window(
            window_ids: torch.Tensor,             # [1, W]
            global_begin: int,                    # window_ids 对应原序列 token 的起点位置（global token index）
            score_from_global_pos: int,           # 需要计分的 global token pos 起点（label pos，>=1）
            score_to_global_pos: int,             # 需要计分的 global token pos 终点（不含）
        ) -> Tuple[float, int]:
            """
            计算这个 window 内指定 global token pos 区间的 NLL 之和（只计一次）。
            约定：
              - global token pos p 的预测对应 shift index j = p - (global_begin + 1)
              - shift index j 预测 token at global pos (global_begin + 1 + j)
            """
            shift_logits, targets = _extract_shift_logits(window_ids)   # [1, L_eff, V], [1, L_eff]
            L_eff = int(targets.size(1))
            if L_eff <= 0:
                return float("nan"), 0

            # shift 空间的 mask：选中 [score_from_global_pos, score_to_global_pos) 对应的 j
            # p = global_begin + 1 + j  => j = p - (global_begin + 1)
            j_start = int(score_from_global_pos) - (int(global_begin) + 1)
            j_end   = int(score_to_global_pos) - (int(global_begin) + 1)

            j_start = max(j_start, 0)
            j_end = min(j_end, L_eff)
            if j_start >= j_end:
                return 0.0, 0

            mask = torch.zeros((L_eff,), device=self.device, dtype=torch.bool)
            mask[j_start:j_end] = True

            # 排除 special token（按 targets 对应的位置：window_ids 的 1..）
            spec = _special_mask(window_ids[0])
            if spec is not None:
                # targets 对应 window_ids[1:]
                spec_shift = spec[1:1 + L_eff]
                mask = mask & (~spec_shift)

            if mask.sum().item() <= 0:
                return 0.0, 0

            log_probs = F.log_softmax(shift_logits[0], dim=-1)  # [L_eff, V]
            tgt = targets[0]                                   # [L_eff]
            token_logp = log_probs.gather(1, tgt.unsqueeze(1)).squeeze(1)  # [L_eff]
            token_nll = -token_logp

            nll_sum = float(token_nll[mask].sum().item())
            tok_cnt = int(mask.sum().item())
            return nll_sum, tok_cnt

        def _compute_one(seq: str) -> Dict[str, Any]:
            seq_proc = self._preprocess_sequence(seq)
            char_len = len(seq_proc)

            ids = _tokenize_one(seq_proc)  # [1, L]
            L = int(ids.size(1))

            # 没有可预测 token
            if L <= 1:
                return {
                    "sequence_chars": int(char_len),
                    "sequence_tokens": int(L),
                    "prompt_len_chars": int(prompt_len_chars) if prompt_len_chars is not None else 0,
                    "prompt_tokens": 0,
                    "char_count": 0,
                    "token_count": 0,
                    "avg_nll_token": float("nan"),
                    "avg_nll_char": float("nan"),
                    "ppl_token": float("nan"),
                    "ppl_char": float("nan"),
                    "ppl": float("nan"),
                    "mode": "too_short",
                    "error": "too_short",
                }

            # prompt token 长度
            if prompt_len_tokens is not None:
                prompt_tok_len = int(prompt_len_tokens)
            elif prompt_len_chars is not None:
                p = seq_proc[: int(prompt_len_chars)]
                p_ids = _tokenize_one(p)
                prompt_tok_len = int(p_ids.size(1))
            else:
                prompt_tok_len = 0

            # 需要计分的 global token pos 范围（label pos）：[start_pos, L)
            # full: start_pos=1（预测 token1..）
            # conditional: start_pos=max(prompt_tok_len,1)
            start_pos = 1 if (prompt_len_chars is None and prompt_len_tokens is None) else max(int(prompt_tok_len), 1)

            if start_pos >= L:
                # 没有 continuation token
                cont_chars = max(char_len - (int(prompt_len_chars) if prompt_len_chars is not None else 0), 0)
                return {
                    "sequence_chars": int(char_len),
                    "sequence_tokens": int(L),
                    "prompt_len_chars": int(prompt_len_chars) if prompt_len_chars is not None else 0,
                    "prompt_tokens": int(prompt_tok_len),
                    "char_count": int(cont_chars),
                    "token_count": 0,
                    "avg_nll_token": float("nan"),
                    "avg_nll_char": float("nan"),
                    "ppl_token": float("nan"),
                    "ppl_char": float("nan"),
                    "ppl": float("nan"),
                    "mode": "conditional",
                    "error": "no_continuation_tokens",
                }

            # char_count：full 用 (len-1)，conditional 用 continuation chars
            if prompt_len_chars is None and prompt_len_tokens is None:
                char_count = max(char_len - 1, 0)
                mode = "unconditional"
            else:
                # 若用 tokens 做 conditional，char_count 只能近似（这里按 chars 提供时更准）
                if prompt_len_chars is not None:
                    char_count = max(char_len - int(prompt_len_chars), 0)
                else:
                    char_count = max(char_len - 1, 0)
                mode = "conditional"

            total_nll = 0.0
            total_tok = 0

            if use_sliding_window and L > max_window_tokens:
                # sliding：每次计分 stride 个 token（global pos），用最多 max_window_tokens 的上下文
                cur = int(start_pos)
                while cur < L:
                    nxt = min(cur + stride, L)  # 计分 global token pos [cur, nxt)
                    # window 必须包含到 token(nxt-1)，并包含它前面的上下文
                    context_begin = max(0, nxt - max_window_tokens)
                    window_ids = ids[:, context_begin:nxt]  # [1, W]，W<=max_window_tokens

                    nll_sum, tok_cnt = _nll_on_window(
                        window_ids=window_ids,
                        global_begin=context_begin,
                        score_from_global_pos=cur,
                        score_to_global_pos=nxt,
                    )
                    total_nll += float(nll_sum)
                    total_tok += int(tok_cnt)
                    cur = nxt

                calc_mode = mode + "_sliding"
            else:
                # direct：一次前向算完，然后用 mask 截取 [start_pos, L)
                # 这里复用 window 逻辑：window 就是全长
                window_ids = ids  # [1, L]
                nll_sum, tok_cnt = _nll_on_window(
                    window_ids=window_ids,
                    global_begin=0,
                    score_from_global_pos=start_pos,
                    score_to_global_pos=L,
                )
                total_nll = float(nll_sum)
                total_tok = int(tok_cnt)
                calc_mode = mode + "_direct"

            if total_tok <= 0 or (not math.isfinite(total_nll)):
                return {
                    "sequence_chars": int(char_len),
                    "sequence_tokens": int(L),
                    "prompt_len_chars": int(prompt_len_chars) if prompt_len_chars is not None else 0,
                    "prompt_tokens": int(prompt_tok_len),
                    "char_count": int(char_count),
                    "token_count": int(total_tok),
                    "avg_nll_token": float("nan"),
                    "avg_nll_char": float("nan"),
                    "ppl_token": float("nan"),
                    "ppl_char": float("nan"),
                    "ppl": float("nan"),
                    "mode": calc_mode,
                    "error": "no_valid_tokens",
                }

            avg_nll_token = float(total_nll) / float(total_tok)
            ppl_token = float(_safe_exp(avg_nll_token))

            if char_count > 0:
                avg_nll_char = float(total_nll) / float(char_count)
                ppl_char = float(_safe_exp(avg_nll_char))
            else:
                avg_nll_char = float("nan")
                ppl_char = float("nan")

            ppl = ppl_char if ppl_mode == "char" else ppl_token

            return {
                "sequence_chars": int(char_len),
                "sequence_tokens": int(L),
                "prompt_len_chars": int(prompt_len_chars) if prompt_len_chars is not None else 0,
                "prompt_tokens": int(prompt_tok_len),
                "char_count": int(char_count),
                "token_count": int(total_tok),
                "avg_nll_token": float(avg_nll_token),
                "avg_nll_char": float(avg_nll_char),
                "ppl_token": float(ppl_token),
                "ppl_char": float(ppl_char),
                "ppl": float(ppl),
                "mode": calc_mode,
                "max_window_tokens": int(max_window_tokens) if use_sliding_window else None,
                "stride": int(stride) if use_sliding_window else None,
            }

        # 主循环
        results: List[Dict[str, Any]] = []
        for st in tqdm(range(0, len(seq_list), batch_size), desc="OmniReg-GPT PPL"):
            batch = seq_list[st: st + batch_size]
            for s in batch:
                results.append(_compute_one(s))

        if return_details:
            return results

        ppl_list = [float(r.get("ppl", float("nan"))) for r in results]
        return ppl_list[0] if single else ppl_list


    # =============================
    # Generation (sampling)
    # =============================
    @staticmethod
    def _top_k_top_p_filtering(
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """
        logits: [B, V]
        """
        if top_k is not None and int(top_k) > 0:
            top_k = int(top_k)
            top_k = min(top_k, logits.size(-1))
            vals, _ = torch.topk(logits, top_k, dim=-1)
            kth = vals[:, -1].unsqueeze(-1)
            logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)

        if top_p is not None and float(top_p) < 1.0:
            top_p = float(top_p)
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumprobs = probs.cumsum(dim=-1)

            sorted_mask = cumprobs > top_p
            sorted_mask[:, 0] = False  # 至少保留 1 个
            mask = torch.zeros_like(sorted_mask).scatter(1, sorted_idx, sorted_mask)
            logits = logits.masked_fill(mask, float("-inf"))

        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt_seqs: Union[str, List[str]] = "ACGT",
        n_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        eos_token_id: Optional[int] = None,
        add_special_tokens: bool = False,
        clean_spaces: bool = True,
        batch_size: int = 8,
        return_scores: bool = False,
    ):
        """
        纯 PyTorch 逐步自回归生成（不依赖 transformers.generate），适配“模型 forward 只吃 input_ids”。

        关键点：
        - HierarchicalTransformer 内部做 ids[:, :-1] 来预测 ids[:, 1:]
        - 为了拿到“下一 token”的 logits，我们在末尾 append 一个 dummy token，
          这样 logits 的最后一位就对应 next-token 分布。

        返回：
        - sequences: List[str]
        - scores: List[float] (可选) 每条生成 token 的平均 logprob
        """
        if isinstance(prompt_seqs, str):
            prompts = [prompt_seqs] * int(num_return_sequences)
        else:
            prompts = list(prompt_seqs)
            if len(prompts) == 1 and int(num_return_sequences) > 1:
                prompts = prompts * int(num_return_sequences)

        prompts = [self._preprocess_sequence(s) for s in prompts]

        if eos_token_id is None:
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = 0
        pad_id = int(pad_id)

        # tokenize each prompt without padding（避免 pad 进入上下文）
        prompt_ids_list = []
        for s in prompts:
            enc = self.tokenizer(
                s,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.model_max_len,
                add_special_tokens=add_special_tokens,
            )
            ids = enc["input_ids"][0].to(self.device)  # [L]
            prompt_ids_list.append(ids)

        # 分组：只对“相同长度”的 prompt 做 batch（保证无 padding）
        groups = {}
        for idx, ids in enumerate(prompt_ids_list):
            L = int(ids.numel())
            groups.setdefault(L, []).append(idx)

        all_out_text = [None] * len(prompts)
        all_scores = [None] * len(prompts)

        for L, indices in groups.items():
            # 再按 batch_size 切
            for st in range(0, len(indices), batch_size):
                batch_idx = indices[st: st + batch_size]
                cur = torch.stack([prompt_ids_list[i] for i in batch_idx], dim=0)  # [B, L]
                B = cur.size(0)

                # 截断生成长度，别超过模型上限
                max_new = int(n_tokens)
                if cur.size(1) + max_new > int(self.model_max_len):
                    max_new = max(0, int(self.model_max_len) - cur.size(1))

                finished = torch.zeros((B,), device=self.device, dtype=torch.bool)
                sum_logp = torch.zeros((B,), device=self.device, dtype=torch.float32)
                gen_cnt = torch.zeros((B,), device=self.device, dtype=torch.float32)

                for _ in range(max_new):
                    # append dummy token to get next-token logits
                    dummy = torch.full((B, 1), pad_id, device=self.device, dtype=cur.dtype)
                    inp = torch.cat([cur, dummy], dim=1)  # [B, L+1]

                    out = self.model(input_ids=inp, return_loss=False)
                    logits = self._extract_logits_from_out(out)  # [B, T, V] or [B, T-1, V]
                    if logits.dim() != 3:
                        raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")

                    # 取“最后一位”的分布作为 next-token
                    next_logits = logits[:, -1, :].float()  # [B, V]

                    # temperature
                    temp = float(temperature)
                    if temp <= 0:
                        temp = 1e-6
                    next_logits = next_logits / temp

                    if do_sample:
                        flt = self._top_k_top_p_filtering(next_logits, top_k=top_k, top_p=top_p)
                        probs = torch.softmax(flt, dim=-1)
                        next_tok = torch.multinomial(probs, num_samples=1).squeeze(1)  # [B]
                    else:
                        next_tok = torch.argmax(next_logits, dim=-1)

                    # logprob（用未过滤的 next_logits 计算更“真实”）
                    logp = torch.log_softmax(next_logits, dim=-1).gather(1, next_tok.unsqueeze(1)).squeeze(1)
                    # 已完成的不再累计
                    logp = torch.where(finished, torch.zeros_like(logp), logp)

                    sum_logp += logp
                    gen_cnt += (~finished).float()

                    cur = torch.cat([cur, next_tok.unsqueeze(1)], dim=1)

                    if eos_token_id is not None:
                        finished = finished | (next_tok == int(eos_token_id))
                        if bool(finished.all()):
                            break

                # decode
                for bi, orig_i in enumerate(batch_idx):
                    ids = cur[bi].detach().cpu().tolist()
                    txt = self.tokenizer.decode(ids, skip_special_tokens=True)
                    if clean_spaces:
                        txt = txt.replace(" ", "")
                    all_out_text[orig_i] = txt

                    if return_scores:
                        denom = max(gen_cnt[bi].item(), 1.0)
                        all_scores[orig_i] = float((sum_logp[bi] / denom).item())

        if return_scores:
            return all_out_text, all_scores
        return all_out_text



# =========================
# 自测
# =========================
if __name__ == "__main__":
    # 根据你的实际路径修改
    MODEL_CKPT = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model/pytorch_model.bin"
    TOKENIZER_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/gena-lm-bert-base-t2t"
    OMNIREG_REPO = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model/OmniReg-GPT"
    HF_HOME = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model"

    m = OmniRegGPTModel(
        model_name="OmniReg-GPT",
        model_path=MODEL_CKPT,
        tokenizer_path=TOKENIZER_DIR,
        omnireg_repo_path=OMNIREG_REPO,
        hf_home=HF_HOME,
        device=None,      # 自动 gpu / cpu
        max_length=16384, # 可按实际模型改
    )

    dna_list = [
        "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATTCGTTAGC"*10,
        "AGCGTACGTTAG"*20,
        "AGTTTCCCGGAA"*20,
    ]

    embs = m.embed_sequences(
        dna_list,
        pooling="mean",
        batch_size=2,
    )
    print("Embedding shape:", embs.shape)

    scores = m.score_sequences(dna_list, batch_size=1)
    print("AR log-likelihood scores:", scores)

    
    ppl_cond = m.get_ppl(
        dna_list,
        prompt_len_chars=128,
        batch_size=2,
        return_details=False,
        add_special_tokens=False,
    )
    print("Conditional PPL:", ppl_cond)
    gen_seqs, gen_scores = m.generate(
        prompt_seqs="ACGT"*50,
        n_tokens=50,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        num_return_sequences=4,
        add_special_tokens=False,
        return_scores=True,
    )
    print("Generated sequences:", gen_seqs)
    print("Generated scores:", gen_scores)
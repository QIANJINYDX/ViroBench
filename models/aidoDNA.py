# models/aido_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import torch
import numpy as np
from typing import List, Optional, Literal, Union, Tuple
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForMaskedLM, 
    AutoConfig
)

# 尝试从本地导入 BaseModel，如果没有则定义一个简单的占位符以防报错
try:
    from .base_model import BaseModel
except ImportError:
    class BaseModel:
        def __init__(self, model_name, model_path):
            self.model_name = model_name
            self.model_path = model_path

Pooling = Literal["mean", "max", "cls"]

class AIDOModel(BaseModel):
    """
    AIDO.DNA (基于 RNABert 架构) 适配器
    
    核心功能：
    1. 支持指定 'rnabert_code_path' 来手动注册 RNABert 架构，解决 KeyError: 'rnabert'。
    2. 提供统一的 embed_sequences 和 score_sequences 接口。
    
    Args:
        model_name: 模型别名
        model_path: 模型权重路径 (包含 config.json, pytorch_model.bin)
        rnabert_code_path: [关键] 包含 configuration_rnabert.py 的文件夹路径
        hf_home: HuggingFace 缓存目录
        device: 'cuda' / 'cpu'
        use_mlm_head: 是否加载 MaskedLM 头用于 PLL 评分
        trust_remote_code: 是否信任远程代码 (通常设为 True)
        torch_dtype: 加载精度 ('auto', torch.float16, torch.bfloat16)
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        rnabert_code_path: Optional[str] = None, 
        hf_home: Optional[str] = None,
        device: Optional[str] = None,
        use_mlm_head: bool = False,
        trust_remote_code: bool = True,
        torch_dtype: Union[str, torch.dtype] = "auto", 
    ):
        super().__init__(model_name, model_path)
        self.hf_home = hf_home
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_mlm_head = use_mlm_head
        self.trust_remote_code = trust_remote_code
        self.torch_dtype = torch_dtype
        
        # 确定代码查找路径：用户指定 > 默认 fallback 到 model_path
        self.rnabert_code_path = rnabert_code_path or model_path

        # 1. 先注册模型架构
        self._register_rnabert_manual()
        
        # 2. 再加载模型
        self._load_model()

    def _register_rnabert_manual(self):
        """
        [终极修复 V3] 动态加载 RNABert 全套组件。
        解决: 
        1. KeyError: 'RNABertConfig'
        2. ImportError: attempted relative import
        3. AttributeError: 'NoneType' object has no attribute '__dict__' (dataclasses 报错)
        """
        # 1. 避免重复注册
        try:
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING
            if "rnabert" in CONFIG_MAPPING:
                return
        except Exception:
            pass

        print(f"[{self.model_name}] Registering 'rnabert' from: {self.rnabert_code_path}")
        
        # 2. 临时加入路径
        if self.rnabert_code_path not in sys.path:
            sys.path.insert(0, self.rnabert_code_path)
        
        try:
            # --- 步骤 A: 导入 Config & Tokenizer ---
            import configuration_rnabert
            RNABertConfig = configuration_rnabert.RNABertConfig
            
            import tokenization_rnabert
            RNABertTokenizer = tokenization_rnabert.RNABertTokenizer

            # --- 步骤 B: 导入 Modeling (含 dataclass 修复) ---
            import types
            modeling_path = os.path.join(self.rnabert_code_path, "modeling_rnabert.py")
            
            if os.path.exists(modeling_path):
                # 1. 读取源码并修复相对导入
                with open(modeling_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                source_code = source_code.replace("from .configuration_rnabert", "from configuration_rnabert")
                
                # 2. 创建动态模块
                module_name = "modeling_rnabert"
                modeling_mod = types.ModuleType(module_name)
                
                # 3. [关键修复] 将模块注入 sys.modules
                # dataclasses 需要在 sys.modules 中找到它，否则会报 AttributeError
                sys.modules[module_name] = modeling_mod
                
                # 4. 注入依赖
                modeling_mod.RNABertConfig = RNABertConfig
                modeling_mod.__file__ = modeling_path
                
                try:
                    # 5. 执行代码
                    exec(source_code, modeling_mod.__dict__)
                    
                    # 6. 提取模型类
                    RNABertModel = modeling_mod.RNABertModel
                    RNABertForMaskedLM = modeling_mod.RNABertForMaskedLM
                    
                    # --- 步骤 C: 全面注册到 Auto 类 ---
                    AutoConfig.register("rnabert", RNABertConfig)
                    AutoModel.register(RNABertConfig, RNABertModel)
                    AutoModelForMaskedLM.register(RNABertConfig, RNABertForMaskedLM)
                    AutoTokenizer.register(RNABertConfig, slow_tokenizer_class=RNABertTokenizer)
                    
                    print(f"[{self.model_name}] ✅ 成功注册所有组件 (含 dataclass 支持)。")
                    
                except Exception as exec_err:
                    # 如果执行失败，清理 sys.modules 以免污染环境
                    if module_name in sys.modules:
                        del sys.modules[module_name]
                    raise exec_err

            else:
                print(f"⚠️ Warning: modeling_rnabert.py not found in {self.rnabert_code_path}")

        except Exception as e:
            print(f"[{self.model_name}] ❌ 注册失败: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 清理 path
            if self.rnabert_code_path in sys.path:
                sys.path.remove(self.rnabert_code_path)

            

    def _load_model(self):
        if self.hf_home:
            os.environ["HF_HOME"] = self.hf_home

        print(f"[{self.model_name}] Loading tokenizer from {self.model_path}...")
        
        # 1. 尝试寻找 vocab.txt
        # 默认去权重目录找，如果找不到，就去代码目录找
        vocab_path = os.path.join(self.model_path, "vocab.txt")
        tokenizer_kwargs = {"trust_remote_code": self.trust_remote_code}
        
        if not os.path.exists(vocab_path):
            # 尝试在代码目录找
            code_vocab = os.path.join(self.rnabert_code_path, "vocab.txt")
            if os.path.exists(code_vocab):
                print(f"[{self.model_name}] ℹ️ 权重目录无 vocab.txt，使用代码目录的字典: {code_vocab}")
                tokenizer_kwargs["vocab_file"] = code_vocab
            else:
                print(f"[{self.model_name}] ⚠️ 警告: 到处都找不到 vocab.txt，Tokenizer 加载可能会失败。")

        # 2. 加载 Tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                **tokenizer_kwargs
            )
        except Exception as e:
            print(f"[{self.model_name}] AutoTokenizer error: {e}")
            print(f"[{self.model_name}] Trying direct class loading...")
            # 最后的兜底：直接实例化注册好的类
            try:
                # 此时 CONFIG_MAPPING 里应该已经有 RNABertConfig 了
                # 我们尝试直接从模块导入
                import tokenization_rnabert
                vocab_file = tokenizer_kwargs.get("vocab_file", vocab_path)
                self.tokenizer = tokenization_rnabert.RNABertTokenizer(vocab_file=vocab_file)
                print(f"[{self.model_name}] ✅ Manually loaded RNABertTokenizer.")
            except Exception as manual_e:
                raise RuntimeError(f"无法加载 Tokenizer，请确认 vocab.txt 存在。错误: {manual_e}")

        print(f"[{self.model_name}] Loading model from {self.model_path}...")
        
        load_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": self.torch_dtype,
        }

        if self.use_mlm_head:
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_path, **load_kwargs)
        else:
            self.model = AutoModel.from_pretrained(self.model_path, **load_kwargs)

        self.model.to(self.device)
        self.model.eval()

        # 3. 设置最大长度
        self.model_max_len = (
            getattr(self.model.config, "max_position_embeddings", None)
            or getattr(self.tokenizer, "model_max_length", None)
            or 512
        )
        
        # 4. 设置 Mask ID
        self.mask_id = self.tokenizer.mask_token_id
        if self.mask_id is None and self.use_mlm_head:
             if hasattr(self.tokenizer, "mask_token") and self.tokenizer.mask_token:
                 self.mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        print(f"[{self.model_name}] Loaded. MaxLen: {self.model_max_len}")

    def _preprocess_sequence(self, seq: str) -> str:
        # 统一转大写，去空白
        return seq.strip().upper()

    @torch.no_grad()
    def embed_sequences(
        self,
        sequences: List[str],
        batch_size: int = 16,
        pooling: Pooling = "mean",
        exclude_special: bool = True,
        truncation: bool = True,
        return_numpy: bool = True,
    ):
        """
        提取序列 Embedding
        """
        return self.get_embedding(
            sequences=sequences,
            batch_size=batch_size,
            pool=pooling,
            exclude_special=exclude_special,
            truncation=truncation,
            return_numpy=return_numpy,
        )

    def score_sequences(self, sequences: List[str], batch_size: int = 256) -> List[float]:
        """
        计算 PLL (Pseudo-Log-Likelihood) 分数
        """
        if not self.use_mlm_head:
            raise RuntimeError("PLL scoring requires use_mlm_head=True")
        
        if self.mask_id is None:
            raise RuntimeError("Mask token ID not found. Cannot perform PLL scoring.")

        all_scores = []
        # 注意：这里的 batch_size 指的是 mask 的 batch size
        for seq in tqdm(sequences, desc=f"Scoring ({self.model_name})"):
            score = self._score_single_sequence(seq, batch_size)
            all_scores.append(score)
        return all_scores

    def _score_single_sequence(self, sequence: str, mask_batch_size: int) -> float:
        # 1. Tokenize
        sequence = self._preprocess_sequence(sequence)
        enc = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.model_max_len,
            add_special_tokens=True, # AIDO 通常有 CLS/SEP
        )
        
        input_ids = enc["input_ids"].to(self.device).squeeze(0) # (L,)
        
        # 2. 确定有效位置 (排除特殊 token)
        special_mask = self.tokenizer.get_special_tokens_mask(
            input_ids.cpu().tolist(), already_has_special_tokens=True
        )
        valid_indices = [i for i, is_spec in enumerate(special_mask) if is_spec == 0]
        
        if not valid_indices:
            return 0.0

        # 3. 分块 Mask 并计算 Loss
        total_logprob = 0.0
        total_count = 0
        
        for i in range(0, len(valid_indices), mask_batch_size):
            chunk_indices = valid_indices[i : i + mask_batch_size]
            B = len(chunk_indices)
            
            # 复制 batch
            masked_input = input_ids.repeat(B, 1) # (B, L)
            
            # 构造 batch mask
            batch_pos = torch.arange(B, device=self.device)
            target_pos = torch.tensor(chunk_indices, device=self.device)
            
            # 记录真实 token
            labels = masked_input[batch_pos, target_pos].clone()
            
            # 应用 Mask
            masked_input[batch_pos, target_pos] = self.mask_id
            
            # Forward
            with torch.no_grad():
                out = self.model(masked_input)
                logits = out.logits # (B, L, V)
            
            # 取出被 mask 位置的 logits
            target_logits = logits[batch_pos, target_pos, :] # (B, V)
            log_probs = torch.log_softmax(target_logits, dim=-1)
            
            # 取出真实 token 的 log_prob
            token_log_prob = log_probs[batch_pos, labels]
            
            total_logprob += token_log_prob.sum().item()
            total_count += B
            
        return total_logprob / max(total_count, 1)

    @torch.no_grad()
    def get_embedding(
        self,
        sequences: Union[str, List[str]],
        batch_size: int = 16,
        pool: Pooling = "mean",
        exclude_special: bool = True,
        truncation: bool = True,
        return_numpy: bool = True,
        **kwargs
    ):
        if isinstance(sequences, str):
            sequences = [sequences]
            
        def _process_batch(batch_seqs):
            # AIDO Tokenizer 会处理 k-mer 等逻辑
            enc = self.tokenizer(
                [self._preprocess_sequence(s) for s in batch_seqs],
                return_tensors="pt",
                padding=True,
                truncation=truncation,
                max_length=self.model_max_len,
            )
            return enc.to(self.device)

        outputs = []
        for i in tqdm(range(0, len(sequences), batch_size), desc=f"Embedding"):
            batch = sequences[i : i + batch_size]
            inputs = _process_batch(batch)
            
            out = self.model(**inputs, output_hidden_states=True)
            # 取最后一层 hidden state
            hidden = out.hidden_states[-1] # (B, L, D)
            
            # Pooling Logic
            mask = inputs["attention_mask"].bool()
            
            if exclude_special:
                # 排除特殊字符 ([CLS], [SEP])
                # 简便方法：利用 tokenizer 的 special_tokens_mask
                # 但为了速度，这里通常可以直接假设首尾是特殊字符(如果是BERT结构)
                # 更严谨的做法是构造 special mask
                input_ids_cpu = inputs["input_ids"].cpu().tolist()
                spec_mask = []
                for seq_ids in input_ids_cpu:
                    spec_mask.append(self.tokenizer.get_special_tokens_mask(seq_ids, already_has_special_tokens=True))
                spec_mask = torch.tensor(spec_mask, device=self.device).bool()
                mask = mask & (~spec_mask)

            if pool == "mean":
                m_f = mask.unsqueeze(-1).float()
                summed = (hidden * m_f).sum(dim=1)
                denom = m_f.sum(dim=1).clamp(min=1e-9)
                emb = summed / denom
            elif pool == "cls":
                # 通常是第0个位置
                emb = hidden[:, 0, :]
            elif pool == "max":
                hidden_masked = hidden.clone()
                hidden_masked[~mask] = -1e9
                emb = hidden_masked.max(dim=1).values
            else:
                raise ValueError(f"Unknown pool type: {pool}")
            
            outputs.append(emb.cpu())

        res = torch.cat(outputs, dim=0)
        return res.to(torch.float32).numpy() if return_numpy else res


if __name__ == "__main__":
    import os

    MODEL_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/AIDO.DNA-7B"
    
    REPO_ROOT = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model/ModelGenerator" 
    
    CODE_DIR = os.path.join(REPO_ROOT, "huggingface", "aido.rna", "aido_rna", "models")

    m = AIDOModel(
        model_name="AIDO_7B",
        model_path=MODEL_DIR,
        rnabert_code_path=CODE_DIR,  # <--- 这里传入路径，它就会自己去读了
        trust_remote_code=True,
        device="cuda",
        torch_dtype="auto"
    )

    seqs = ["ACGTACGT", "GGCCAA"]
    emb = m.embed_sequences(seqs)
    print("Success!", emb.shape)

# models/aido_rna.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import types
import torch
import numpy as np
from typing import List, Optional, Literal, Union
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForMaskedLM, 
    AutoConfig
)

# 简单的 BaseModel 占位
try:
    from .base_model import BaseModel
except ImportError:
    class BaseModel:
        def __init__(self, model_name, model_path):
            self.model_name = model_name
            self.model_path = model_path

Pooling = Literal["mean", "max", "cls"]

class AIDORNAModel(BaseModel):
    """
    AIDO.RNA 适配器 (基于 RNABert 架构)
    
    核心特性:
    1. [自动转换] 输入 DNA (A,T,C,G) 会自动转换为 RNA (A,U,C,G) 以适配模型。
    2. [架构注册] 内置 RNABert 架构的动态注册修复 (解决 ImportError/KeyError)。
    3. [智能加载] 自动寻找 vocab.txt。
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
        
        # 默认 fallback 到 model_path
        self.rnabert_code_path = rnabert_code_path or model_path

        # 1. 注册架构
        self._register_rnabert_manual()
        
        # 2. 加载模型
        self._load_model()

    def _register_rnabert_manual(self):
        """
        [终极修复 V3] 动态加载 RNABert 组件。
        """
        try:
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING
            if "rnabert" in CONFIG_MAPPING:
                return
        except Exception:
            pass

        print(f"[{self.model_name}] Registering 'rnabert' from: {self.rnabert_code_path}")
        
        if self.rnabert_code_path not in sys.path:
            sys.path.insert(0, self.rnabert_code_path)
        
        try:
            import configuration_rnabert
            RNABertConfig = configuration_rnabert.RNABertConfig
            
            import tokenization_rnabert
            RNABertTokenizer = tokenization_rnabert.RNABertTokenizer

            # 修复 Modeling 的相对导入和 dataclass
            import types
            modeling_path = os.path.join(self.rnabert_code_path, "modeling_rnabert.py")
            
            if os.path.exists(modeling_path):
                with open(modeling_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                source_code = source_code.replace("from .configuration_rnabert", "from configuration_rnabert")
                
                module_name = "modeling_rnabert"
                modeling_mod = types.ModuleType(module_name)
                sys.modules[module_name] = modeling_mod # dataclass fix
                
                modeling_mod.RNABertConfig = RNABertConfig
                modeling_mod.__file__ = modeling_path
                
                exec(source_code, modeling_mod.__dict__)
                
                RNABertModel = modeling_mod.RNABertModel
                RNABertForMaskedLM = modeling_mod.RNABertForMaskedLM
                
                # 注册
                AutoConfig.register("rnabert", RNABertConfig)
                AutoModel.register(RNABertConfig, RNABertModel)
                AutoModelForMaskedLM.register(RNABertConfig, RNABertForMaskedLM)
                AutoTokenizer.register(RNABertConfig, slow_tokenizer_class=RNABertTokenizer)
                
                print(f"[{self.model_name}] ✅ 架构注册成功。")
            else:
                print(f"⚠️ Warning: modeling_rnabert.py not found in {self.rnabert_code_path}")

        except Exception as e:
            print(f"[{self.model_name}] ❌ 注册警告: {e}")
        finally:
            if self.rnabert_code_path in sys.path:
                sys.path.remove(self.rnabert_code_path)

    def _load_model(self):
        if self.hf_home:
            os.environ["HF_HOME"] = self.hf_home

        print(f"[{self.model_name}] Loading tokenizer...")
        
        # 自动寻找 vocab.txt
        vocab_path = os.path.join(self.model_path, "vocab.txt")
        tokenizer_kwargs = {"trust_remote_code": self.trust_remote_code}
        
        if not os.path.exists(vocab_path):
            code_vocab = os.path.join(self.rnabert_code_path, "vocab.txt")
            if os.path.exists(code_vocab):
                print(f"[{self.model_name}] Using vocab from code dir: {code_vocab}")
                tokenizer_kwargs["vocab_file"] = code_vocab

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, **tokenizer_kwargs)
        except Exception as e:
            print(f"[{self.model_name}] AutoTokenizer failed ({e}), trying manual load...")
            import tokenization_rnabert
            vocab = tokenizer_kwargs.get("vocab_file", vocab_path)
            self.tokenizer = tokenization_rnabert.RNABertTokenizer(vocab_file=vocab)

        print(f"[{self.model_name}] Loading model...")
        load_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": self.torch_dtype,
        }

        if self.use_mlm_head:
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_path, **load_kwargs)
        else:
            self.model = AutoModel.from_pretrained(self.model_path, **load_kwargs)

        self.model.to(self.device).eval()

        self.model_max_len = getattr(self.model.config, "max_position_embeddings", 1024)
        print(f"[{self.model_name}] Loaded. MaxLen: {self.model_max_len}")

    # ================= [ 核心修改点 ] =================
    def _preprocess_sequence(self, seq: str) -> str:
        """
        处理输入序列：
        1. 转大写
        2. 将 DNA 的 'T' 替换为 RNA 的 'U'
        """
        seq = seq.strip().upper()
        return seq.replace("T", "U") 
    # =================================================

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
        if isinstance(sequences, str): sequences = [sequences]
            
        def _process_batch(batch_seqs):
            # 在这里调用 _preprocess_sequence 会自动完成 DNA->RNA 转换
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
            hidden = out.hidden_states[-1]
            
            mask = inputs["attention_mask"].bool()
            
            # 排除特殊 token (CLS, SEP)
            if exclude_special:
                # 简单高效的排除首尾 (因为 RNABert 通常是 CLS...SEP)
                # 如果你想更严谨，可以用 tokenizer.get_special_tokens_mask
                # 但对于大批量，直接切片通常更快且足够
                # 这里为了严谨还是用 mask 逻辑
                input_ids_cpu = inputs["input_ids"].cpu().tolist()
                spec_mask = []
                for seq_ids in input_ids_cpu:
                    spec_mask.append(self.tokenizer.get_special_tokens_mask(seq_ids, already_has_special_tokens=True))
                spec_mask = torch.tensor(spec_mask, device=self.device).bool()
                mask = mask & (~spec_mask)

            if pool == "mean":
                m_f = mask.unsqueeze(-1).float()
                emb = (hidden * m_f).sum(dim=1) / m_f.sum(dim=1).clamp(min=1e-9)
            elif pool == "cls":
                emb = hidden[:, 0, :]
            elif pool == "max":
                hidden_masked = hidden.clone()
                hidden_masked[~mask] = -1e9
                emb = hidden_masked.max(dim=1).values
            
            outputs.append(emb.cpu())

        res = torch.cat(outputs, dim=0)
        return res.to(torch.float32).numpy() if return_numpy else res

    # 为了兼容旧接口
    embed_sequences = get_embedding

# ================= 使用示例 =================
if __name__ == "__main__":
    import os
    
    # 1. 权重路径 (AIDO.RNA 模型)
    # 例如: "/inspire/.../AIDO.RNA-1.6B"
    MODEL_DIR = "../../model_weight/AIDO.RNA-1.6B" 
    
    # 2. 代码路径 (通常和 AIDO.DNA 用的是同一套代码库)
    CODE_DIR = "../../model/ModelGenerator/huggingface/aido.rna/aido_rna/models"

    # 如果路径不存在，仅做代码演示
    if not os.path.exists(MODEL_DIR):
        print(f"提示: 请修改 MODEL_DIR 为真实的 RNA 模型路径。")
    
    try:
        m = AIDORNAModel(
            model_name="AIDO_RNA",
            model_path=MODEL_DIR,
            rnabert_code_path=CODE_DIR, # 同样需要这个来加载 RNABert 架构
            trust_remote_code=True,
            torch_dtype="auto"
        )
        
        # 输入是 DNA，代码内部会自动转 RNA
        seqs = ["ACGTACGT", "GGCCAA"] 
        print(f"\n输入序列 (DNA): {seqs}")
        
        emb = m.embed_sequences(seqs)
        print(f"输出 Embedding Shape: {emb.shape}")
        
    except Exception as e:
        print(f"Error: {e}")
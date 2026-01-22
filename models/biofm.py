# -*- coding: utf-8 -*-
import os
import sys
import torch
import numpy as np
from typing import List, Union, Optional
from tqdm import tqdm

class BioFMModel:
    """
    BioFM 纯序列适配器
    跳过 VCF/Annotation 流程，直接对 DNA 序列提取 Embedding。
    """
    def __init__(
        self,
        model_path: str = "m42-health/BioFM-265M",
        tokenizer_path: str = "m42-health/BioFM-265M",
        device: Optional[str] = None,
        max_length: int = 1024, # BioFM 上下文长度通常较长，但也需注意显存
    ):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        
        self._load_model()

    def _load_model(self):
        annotated_model_cls, annotation_tokenizer_cls = self._import_biofm_eval()
        print(f"[BioFM] Loading tokenizer from {self.tokenizer_path}...")
        # BioFM 的 tokenizer 比较特殊，专门处理生物信息
        self.tokenizer = annotation_tokenizer_cls.from_pretrained(self.tokenizer_path)

        print(f"[BioFM] Loading model from {self.model_path}...")
        # 加载模型，使用 bfloat16 以节省显存 (和官方示例一致)
        self.model = annotated_model_cls.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        )
        self.model.to(self.device).eval()
        print(f"[BioFM] Loaded on {self.device}.")

    def _import_biofm_eval(self):
        """
        Import biofm_eval while avoiding name collision with local `datasets` package.

        run_all.py prepends the project root to sys.path, which shadows the
        HuggingFace `datasets` module that biofm_eval depends on. We temporarily
        remove the project root so `datasets` resolves to the HF package.
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
            from biofm_eval import AnnotatedModel, AnnotationTokenizer
        finally:
            if removed_paths:
                sys.path.insert(0, project_root)
            if removed_datasets:
                # Let future imports resolve normally; no re-insert.
                pass
        return AnnotatedModel, AnnotationTokenizer

    @torch.no_grad()
    def get_embedding(
        self,
        sequences: Union[str, List[str]],
        batch_size: int = 8,
        layer_index: int = -1, # 取最后一层
        return_numpy: bool = True,
        pool: str = "mean",# 占位用
        layer_name : str = None, # 占位用
    ) -> Union[np.ndarray, torch.Tensor]:
        
        if isinstance(sequences, str):
            sequences = [sequences]

        # 批处理函数
        def _process_batch(batch_seqs):
            # 1. Tokenize
            # BioFM 的 tokenizer 可以直接处理文本列表
            # 注意：BioFM 可能不自动处理大小写，建议转大写
            batch_seqs = [s.upper() for s in batch_seqs]
            
            inputs = self.tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            
            # 移到 GPU
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 2. Forward
            # output_hidden_states=True 确保我们能拿到 embedding
            outputs = self.model(**inputs, output_hidden_states=True)
            
            # 3. 提取特征
            # hidden_states 是一个 tuple，包含每一层的输出
            # shape: (Batch, Sequence_Length, Hidden_Dim)
            hidden_states = outputs.hidden_states
            target_layer = hidden_states[layer_index]
            
            # 4. Pooling (Mean Pooling)
            # 利用 attention_mask 排除 padding 的部分
            mask = inputs["attention_mask"] # (Batch, Seq_Len)
            
            # 将 mask 扩展维度以匹配 embedding: (B, L, 1)
            mask_expanded = mask.unsqueeze(-1).to(target_layer.dtype)
            
            # 乘法屏蔽 padding 位置
            summed = torch.sum(target_layer * mask_expanded, dim=1)
            # 计算有效长度 (防除零)
            lengths = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            
            mean_embedding = summed / lengths
            return mean_embedding

        # 执行 Batch 循环
        all_embeddings = []
        for i in tqdm(range(0, len(sequences), batch_size), desc="BioFM Embedding"):
            batch = sequences[i : i + batch_size]
            emb = _process_batch(batch)
            all_embeddings.append(emb.cpu())
            
        # 拼接结果
        result = torch.cat(all_embeddings, dim=0)
        if return_numpy:
            # NumPy 不支持 bfloat16，必须先转 float32
            return result.to(torch.float32).numpy()
        else:
            return result

# ================= 使用示例 =================
if __name__ == "__main__":
    # 替换为你本地的权重路径，或者直接用 huggingface ID
    MODEL_PATH = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/BioFM-265M"
    
    # 初始化
    biofm = BioFMModel(model_path=MODEL_PATH, tokenizer_path=MODEL_PATH)
    
    # 你的纯 DNA 序列
    dna_sequences = [
        "ATGCTAGCTAGCTAGCTACGATCGATCGATCGTAGC", 
        "GGGGTTTTAAAACCCC"
    ]
    
    # 提取 Embedding
    embeddings = biofm.get_embedding(dna_sequences)
    
    print("\nShape:", embeddings.shape) 
    # 预期输出: (2, 1024) (假设 BioFM-265M 的 hidden_dim 是 1024)
    # print(embeddings)
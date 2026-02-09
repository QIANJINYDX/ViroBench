#!/usr/bin/env python3
import argparse
import sys
import os
import random
import re
import traceback
from typing import Dict
import torch
import pandas as pd
import time
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from collections import Counter


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Put project root at the front so local packages (e.g. `datasets/`) win over site-packages.
sys.path.insert(0, PROJECT_ROOT)

import torch.nn as nn
from models.mlp_head import MLPHead
from datasets.virus_datasets import VirusSplitDatasets
from evaluators.finetune import FineTuneSeqEvaluator

class MultiTaskMLPHead(nn.Module):
    def __init__(self, input_dim: int, task_out_dims: Dict[str, int]):
        super().__init__()
        self.task_names = list(task_out_dims.keys())
        self.heads = nn.ModuleDict({
            name: MLPHead(
                input_dim=input_dim,
                task="multiclass",
                num_outputs=int(task_out_dims[name]),
            )
            for name in self.task_names
        })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {name: head(x) for name, head in self.heads.items()}



def save_timing_records(timing_records, csv_path):
    """保存时间记录到CSV文件（追加模式）"""
    if not timing_records:
        print("[INFO] 没有时间记录需要保存")
        return

    df_new = pd.DataFrame(timing_records)

    # 检查文件是否存在
    if os.path.exists(csv_path):
        # 读取现有数据（跳过idx列）
        df_existing = pd.read_csv(csv_path)
        # 如果存在idx列，删除它
        if 'idx' in df_existing.columns:
            df_existing = df_existing.drop(columns=['idx'])
        # 合并数据
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new

    # 重新生成idx列（放在第一列）
    df.insert(0, 'idx', range(1, len(df) + 1))

    # 保存到指定路径
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"[INFO] 时间记录已保存到: {csv_path}")

    # 打印本次耗时统计
    current_time = df_new['用时(秒)'].sum()
    print(f"[INFO] 本次总耗时: {current_time:.2f} 秒 ({current_time/60:.2f} 分钟)")
    print(f"[INFO] 累计记录条数: {len(df)}")


def _infer_embedding_dim(model, layer_name, pool, default_dim, sample_length=512):
    """Try to infer embedding dimension via a single forward pass."""
    sample_length = max(4, sample_length)
    sample_seq = "".join(random.choices("ACGT", k=sample_length))
    try:
        embedding = model.get_embedding(
            [sample_seq],
            layer_name=layer_name,
            pool=pool if pool is not None else "mean",
            batch_size=1,
            return_numpy=True,
        )
        if embedding is not None and getattr(embedding, "shape", None):
            return int(embedding.shape[-1])
    except Exception as exc:
        print(f"[WARN] 自动检测 hidden_size 失败，将使用默认值 {default_dim}。原因: {exc}")
    return default_dim


def _infer_embedding_dim_hyenadna(model, pool, default_dim, sample_length=512):
    """Try to infer embedding dimension for HyenaDNA model (doesn't support layer_name)."""
    sample_length = max(4, sample_length)
    sample_seq = "".join(random.choices("ACGT", k=sample_length))
    try:
        embedding = model.get_embedding(
            [sample_seq],
            pool=pool if pool is not None else "mean",
            batch_size=1,
            return_numpy=True,
        )
        if embedding is not None and getattr(embedding, "shape", None):
            return int(embedding.shape[-1])
    except Exception as exc:
        print(f"[WARN] 自动检测 hidden_size 失败，将使用默认值 {default_dim}。原因: {exc}")
    return default_dim


def _infer_embedding_dim_nucleotide_transformer(model, pool, default_dim, sample_length=512):
    """Try to infer embedding dimension for NucleotideTransformer model (doesn't support layer_name)."""
    sample_length = max(4, sample_length)
    sample_seq = "".join(random.choices("ACGT", k=sample_length))
    try:
        embedding = model.get_embedding(
            [sample_seq],
            pool=pool if pool is not None else "mean",
            batch_size=1,
            return_numpy=True,
        )
        if embedding is not None and getattr(embedding, "shape", None):
            return int(embedding.shape[-1])
    except Exception as exc:
        print(f"[WARN] 自动检测 hidden_size 失败，将使用默认值 {default_dim}。原因: {exc}")
    return default_dim


def _infer_embedding_dim_genos(model, pool, default_dim, sample_length=512):
    """Try to infer embedding dimension for Genos model (doesn't support layer_name)."""
    sample_length = max(4, sample_length)
    sample_seq = "".join(random.choices("ACGT", k=sample_length))
    try:
        embedding = model.get_embedding(
            [sample_seq],
            pool=pool if pool is not None else "mean",
            batch_size=1,
            return_numpy=True,
        )
        if embedding is not None and getattr(embedding, "shape", None):
            return int(embedding.shape[-1])
    except Exception as exc:
        print(f"[WARN] 自动检测 hidden_size 失败，将使用默认值 {default_dim}。原因: {exc}")
    return default_dim


def run(
    model_name: str,
    dataset_name: str,
    model_dir: str = None,
    args_hidden_size: int = 0,
    results_root_name: str = "results",
    timing_records: list = None,
    timing_log_csv: str = None,
    window_len: int = 0,
    train_num_windows: int = 0,
    eval_num_windows: int = -1,
    lr: float = 1e-4,
    num_workers: int = 96,
    early_stopping_patience: int = 20,
    early_stopping_metric: str = "mcc",
    head_batch_size: int = 0,
    emb_batch_size_override: int = 0,
    force_recompute_embeddings: bool = False,
    save_predictions: bool = False,
) -> None:
    print(f"[INFO] model_name = {model_name}")
    print(f"[INFO] dataset_name = {dataset_name}")
    print(f"[INFO] model_dir = {model_dir}，仅NemotronH系列模型生效")
    # 初始化时间记录列表
    if timing_records is None:
        timing_records = []
        is_root_call = True
    else:
        is_root_call = False

    # 设置默认的timing_log_csv路径
    if timing_log_csv is None:
        timing_log_csv = os.path.join(PROJECT_ROOT, "time_log.csv")
    print(f"[INFO] timing_log_csv = {timing_log_csv}")

    if os.path.isabs(results_root_name):
        results_root = results_root_name
    else:
        results_root = os.path.join(PROJECT_ROOT, results_root_name)
    os.makedirs(results_root, exist_ok=True)
    print(f"[INFO] results_root = {results_root}")
    data_batch_size = 1024
    max_length = None
    emb_layer_name = None
    if model_name == "caduceus-ph" or model_name == "caduceus-ps":
        print(f"[INFO] model_name = {model_name}")
        # seqlen-131k_d_model-256_n_layer-16
        from models import CaduceusModel
        HF_HOME = "../../model_weight/cache"
        os.environ["HF_HOME"] = HF_HOME
        MODEL_DIR = f"../../model_weight/{model_name}"
        model = CaduceusModel(
            model_name=model_name,
            model_path=MODEL_DIR,
            hf_home=HF_HOME,
            device=None,   # 自动
        )
        if model_name == "caduceus-ph":
            hidden_size = 256
        elif model_name == "caduceus-ps":
            hidden_size = 512
        batch_size = 16
        emb_pool = "mean"
        max_length = 131072
    elif model_name == "DNABERT-2-117M":
        from models import DNABERT2Model
        MODEL_DIR = f"../../model_weight/{model_name}"
        HF_HOME = "../../model_weight/cache"
        model = DNABERT2Model(model_name, MODEL_DIR,
                              HF_HOME, use_mlm_head=True)
        hidden_size = 4096
        batch_size = 16
        emb_pool = "mean"
    elif model_name == "DNABERT-3" or model_name == "DNABERT-4" or model_name == "DNABERT-5" or model_name == "DNABERT-6":
        from models import DNABERTModel
        # 提取 k-mer 值（例如 "DNABERT-6" -> "6"）
        kmer = model_name.split("-")[-1]
        MODEL_DIR = f"../../model_weight/DNA_bert_{kmer}"
        HF_HOME = "../../model_weight/cache"
        model = DNABERTModel(
            model_name=f"DNABERT-{kmer}",
            model_path=MODEL_DIR,
            hf_home=HF_HOME,
            device=None,          # 自动选 GPU/CPU
            # use_mlm_head=False,   # 先跑嵌入提取
            kmer_size=kmer,
            auto_kmer=True,
        )
        batch_size = 16
        emb_pool = "cls"
        max_length = 512
        # 动态检测 hidden_size
        hidden_size = _infer_embedding_dim(
            model,
            layer_name=None,  # DNABERT 使用 layer_index，但 layer_name 参数会被忽略
            pool=emb_pool,
            default_dim=768,  # DNABERT 的默认 hidden_size
            sample_length=min(512, max_length or 512),
        )
    elif model_name == "DNABERT-S":
        from models.dnaberts_model import DNABERTSModel
        # 提取 k-mer 值（例如 "DNABERT-6" -> "6"）
        kmer = model_name.split("-")[-1]
        MODEL_DIR = f"../../model_weight/DNABERT-S"
        HF_HOME = "../../model_weight/cache"
        model = DNABERTSModel(
            model_name=f"DNABERT-S",
            model_path=MODEL_DIR,
        )
        batch_size = 16
        emb_pool = "mean"
        max_length = 512
        # 动态检测 hidden_size
        # hidden_size = _infer_embedding_dim(
        #     model,
        #     layer_name=None,  # DNABERT 使用 layer_index，但 layer_name 参数会被忽略
        #     pool=emb_pool,
        #     default_dim=768,  # DNABERT 的默认 hidden_size
        #     sample_length=min(512, max_length or 512),
        # )
        hidden_size = 768
    elif model_name == "evo-1-8k-base" or model_name == "evo-1-131k-base" or model_name == "evo-1.5-8k-base":
        from models import Evo1Model
        MODEL_DIR = f"../../model_weight/{model_name}"
        CFG_PATH = f"../../model_weight/{model_name}/{model_name}_inference.yml"
        HF_HOME = "../../model_weight/cache"
        model = Evo1Model(model_name, MODEL_DIR,
                          CFG_PATH, HF_HOME, device=None)
        hidden_size = 4096
        batch_size = 1
        emb_pool = "final"
    elif model_name == "evo2_1b_base" or model_name == "evo2_7b_base" or model_name == "evo2_40b_base" or model_name == "evo2_40b" or model_name == "evo2_7b":
        from models import Evo2Model
        MODEL_DIR = f"../../model_weight/{model_name}/{model_name}.pt"
        model = Evo2Model(model_name, MODEL_DIR)
        if model_name == "evo2_1b_base":
            # 更保守的默认 batch_size，避免 embedding 阶段显存/内存被杀
            batch_size = 8
            hidden_size = 1920
            emb_layer_name = "blocks.24"
            max_length = 8192
        elif model_name == "evo2_7b_base" or model_name == "evo2_7b":
            batch_size = 16
            hidden_size = 4096
            emb_layer_name = "blocks.26"
            max_length = 1000000
        elif model_name == "evo2_40b" or model_name == "evo2_40b_base":
            batch_size = 1
            hidden_size = 8192
            emb_layer_name = "blocks.20"
            max_length = 1000000
        emb_pool = "final"
    elif model_name == "gpn-msa":
        from models import GPNMSAModel
        MODEL_DIR = "../../model_weight/gpn-msa-sapiens"
        MSA_PATH = f"/mnt/s3mount/peijunlin/gpn_msa/peijunlin/89.zarr"
        model = GPNMSAModel(model_name, MODEL_DIR, MSA_PATH, device="cuda")
        hidden_size = 768
        batch_size = 16
        emb_pool = "mean"
    elif model_name == "LucaOne-default-step36M" or model_name == "LucaOne-gene-step36.8M":
        # python script/run_all.py --model_name LucaOne-gene-step36.8M --dataset_name DNA-taxon-genus
        from models.lucaonce import LucaOneModel
        if model_name == "LucaOne-default-step36M":
            CKPT = "../../model_weight/LucaOne-default-step36M"
        elif model_name == "LucaOne-gene-step36.8M":
            CKPT = "../../model_weight/LucaOne-gene-step36.8M"
        model = LucaOneModel(
            model_name=model_name,
            model_path=CKPT,
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype="auto",
        )
        # hidden_size = 2560
        batch_size = 16
        emb_pool = "mean"
        hidden_size = _infer_embedding_dim(
            model,
            layer_name=None,  # DNABERT 使用 layer_index，但 layer_name 参数会被忽略
            pool=emb_pool,
            default_dim=2560,  # DNABERT 的默认 hidden_size
            sample_length=min(512, max_length or 512),
        )
    elif model_name == "LucaVirus-default-step3.8M" or model_name == "LucaVirus-gene-step3.8M":
        # python script/run_all.py --model_name LucaVirus-default-step3.8M --dataset_name DNA-taxon-genus
        from models.lucavirus import LucaVirusModel
        if model_name == "LucaVirus-default-step3.8M":
            CKPT = "../../model_weight/LucaVirus-default-step3.8M"
        elif model_name == "LucaVirus-gene-step3.8M":
            CKPT = "../../model_weight/LucaVirus-gene-step3.8M"
        model = LucaVirusModel(
            model_name="lucavirus-default",
            model_path=CKPT,
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype="auto",
            force_download=False,
        )
        batch_size = 16
        emb_pool = "mean"
        hidden_size = _infer_embedding_dim(
            model,
            layer_name=None,  # DNABERT 使用 layer_index，但 layer_name 参数会被忽略
            pool=emb_pool,
            default_dim=2560,  # DNABERT 的默认 hidden_size
            sample_length=min(512, max_length or 512),
        )

    elif model_name == "gena-lm-bigbird-base-t2t" or model_name == "gena-lm-bert-base-t2t" or model_name == "gena-lm-bert-large-t2t":
        # python script/run_all.py --model_name gena-lm-bigbird-base-t2t --dataset_name DNA-taxon-genus
        # python script/run_all.py --model_name gena-lm-bert-base-t2t --dataset_name DNA-taxon-genus
        # python script/run_all.py --model_name gena-lm-bert-large-t2t --dataset_name DNA-taxon-genus
        from models.gena_lm import GenaLMModel
        if model_name == "gena-lm-bigbird-base-t2t":
            CKPT = "../../model_weight/gena-lm-bigbird-base-t2t"
        elif model_name == "gena-lm-bert-base-t2t":
            CKPT = "../../model_weight/gena-lm-bert-base-t2t"
        elif model_name == "gena-lm-bert-large-t2t":
            CKPT = "../../model_weight/gena-lm-bert-large-t2t"
        model = GenaLMModel(
            model_name="gena-lm",
            model_path=CKPT,
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype="auto",
        )
        # hidden_size = 2560
        batch_size = 16
        emb_pool = "mean"
        hidden_size = _infer_embedding_dim(
            model,
            layer_name=None,  # DNABERT 使用 layer_index，但 layer_name 参数会被忽略
            pool=emb_pool,
            default_dim=2560,  # DNABERT 的默认 hidden_size
            sample_length=min(512, max_length or 512),
        )
    elif model_name == "hyenadna" or model_name == "hyenadna-tiny-16k" or model_name == "hyenadna-tiny-1k" or model_name == "hyenadna-small-32k" or model_name == "hyenadna-medium-160k" or model_name == "hyenadna-medium-450k" or model_name == "hyenadna-large-1m":
        from models import HyenaDNAModel
        HF_HOME = "../../model_weight/cache"
        os.environ["HF_HOME"] = HF_HOME

        if model_name == "hyenadna-tiny-16k":
            MODEL_DIR = "../../model_weight/hyenadna-tiny-16k-seqlen-d128-hf"
        elif model_name == "hyenadna-tiny-1k":
            MODEL_DIR = "../../model_weight/hyenadna-tiny-1k-seqlen-hf"
        elif model_name == "hyenadna-small-32k":
            MODEL_DIR = "../../model_weight/hyenadna-small-32k-seqlen-hf"
        elif model_name == "hyenadna-medium-160k":
            MODEL_DIR = "../../model_weight/hyenadna-medium-160k-seqlen-hf"
        elif model_name == "hyenadna-medium-450k":
            MODEL_DIR = "../../model_weight/hyenadna-medium-450k-seqlen-hf"
        elif model_name == "hyenadna-large-1m":
            MODEL_DIR = "../../model_weight/hyenadna-large-1m-seqlen-hf"

        # 从路径中提取模型名称（去掉路径前缀和可能的后缀）
        # 例如: ../../model_weight/hyenadna-large-1m-seqlen-hf -> hyenadna-large-1m
        model_dir_name = os.path.basename(os.path.normpath(MODEL_DIR))
        # 提取模型名称（去掉 -seqlen-hf 等后缀）
        if "-seqlen-hf" in model_dir_name:
            hyenadna_model_name = model_dir_name.replace("-seqlen-hf", "")
        elif "-seqlen-d" in model_dir_name:
            # 处理类似 hyenadna-tiny-16k-seqlen-d128-hf 的情况
            hyenadna_model_name = re.sub(
                r"-seqlen-d\d+-hf$", "", model_dir_name)
        else:
            hyenadna_model_name = model_dir_name

        model = HyenaDNAModel(
            model_name=hyenadna_model_name,
            model_path=MODEL_DIR,
            task="classification",
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        # 重新设置model_name方便保存（使用目录名，仅在提供了model_dir时更新）
        # if model_dir:
        #     model_name = model_dir_name
        batch_size = 64
        emb_pool = "final"
        max_length = 160000
        # 动态检测 hidden_size（HyenaDNA 使用专门的检测函数，因为不支持 layer_name 参数）
        hidden_size = _infer_embedding_dim_hyenadna(
            model,
            pool=emb_pool,
            default_dim=256,
            sample_length=min(512, max_length or 512),
        )
    elif "hyena_local" in model_name:
        MODEL_DIR = None
        from models.hyenadna_local import HyenaDNALocal
        if model_name == "hyena_local-12M-mini-virus":
            MODEL_DIR = "../../GeneShield/pretrain/hyena-dna/hyena_local-12M-mini-virus"
        elif model_name == "hyena_local-12M-virus":
            MODEL_DIR = "../../GeneShield/pretrain/hyena-dna/hyena_local-12M-virus"
        elif model_name == "hyena_local-test":
            MODEL_DIR = "../../GeneShield/pretrain/hyena-dna/hyena_local-test"
        elif model_name == "hyena_local-436k-virus":
            MODEL_DIR = "../../GeneShield/pretrain/hyena-dna/hyena_local-436k-virus"
        elif model_name == "hyena_local-3.2M-virus":
            MODEL_DIR = "../../GeneShield/pretrain/hyena-dna/hyena_local-3.2M-virus"
        elif model_name == "hyena_local-253M":
            MODEL_DIR = "../../GeneShield/pretrain/hyena-dna/hyena_local-253M"
        if MODEL_DIR is None:
            MODEL_DIR = model_dir
            normalized_model_dir = os.path.normpath(MODEL_DIR)
            last_part = os.path.basename(normalized_model_dir)
            if last_part == "hf":
                time_part = os.path.basename(os.path.dirname(normalized_model_dir))
                date_part = os.path.basename(os.path.dirname(os.path.dirname(normalized_model_dir)))
                if time_part and date_part:
                    model_name = f"{date_part}_{time_part}"
                else:
                    model_name = last_part
            else:
                model_name = last_part
        
        print("---------------------------------------当前模型---------------------------------------")
        print(model_name)
        print("模型路径：⬆️⬇️⬆️⬇️",MODEL_DIR)
        print("---------------------------------------当前模型---------------------------------------")
        model = HyenaDNALocal(
            model_dir=MODEL_DIR,
            device="cuda",
            pretrain_root="../../GeneShield/pretrain/hyena-dna",
        )
        batch_size = 64
        emb_pool = "final"
        max_length = 160000
        hidden_size = _infer_embedding_dim_hyenadna(
            model,
            pool=emb_pool,
            default_dim=256,
            sample_length=min(512, max_length or 512),
        )
    elif model_name == "ViroHyena-436k" or model_name == "ViroHyena-1m" or model_name == "ViroHyena-6m" or model_name == "ViroHyena-253m":
        MODEL_DIR = None
        from models.hyenadna_local import HyenaDNALocal
        if model_name == "ViroHyena-436k":
            MODEL_DIR = "../../GeneShield/pretrain/hyena-dna/ViroHyena-436k"
        elif model_name == "ViroHyena-1m":
            MODEL_DIR = "../../GeneShield/pretrain/hyena-dna/ViroHyena-1m"
        elif model_name == "ViroHyena-6m":
            MODEL_DIR = "../../GeneShield/pretrain/hyena-dna/ViroHyena-6m"
        elif model_name == "ViroHyena-253m":
            MODEL_DIR = "../../GeneShield/pretrain/hyena-dna/ViroHyena-253m"
        
        print("---------------------------------------当前模型---------------------------------------")
        print(model_name)
        print("模型路径：⬆️⬇️⬆️⬇️",MODEL_DIR)
        print("---------------------------------------当前模型---------------------------------------")
        model = HyenaDNALocal(
            model_dir=MODEL_DIR,
            device="cuda",
            pretrain_root="../../GeneShield/pretrain/hyena-dna",
        )
        batch_size = 64
        emb_pool = "final"
        max_length = 160000
        hidden_size = _infer_embedding_dim_hyenadna(
            model,
            pool=emb_pool,
            default_dim=256,
            sample_length=min(512, max_length or 512),
        )

    elif model_name == "nt-500m-human" or model_name == "nt-500m-1000g" or model_name == "nt-2.5b-1000g" or model_name == "nt-2.5b-ms" or model_name == "ntv2-50m-ms-3kmer" or model_name == "ntv2-50m-ms" or model_name == "ntv2-100m-ms" or model_name == "ntv2-250m-ms" or model_name == "ntv2-500m-ms":
        from models import NucleotideTransformerModel
        HF_HOME = "../../model_weight/cache"
        os.environ["HF_HOME"] = HF_HOME

        nt_model_name = model_name
        if model_name == "nt-500m-human":
            MODEL_DIR = "../../model_weight/nucleotide-transformer-500m-human-ref"
        elif model_name == "nt-500m-1000g":
            MODEL_DIR = "../../model_weight/nucleotide-transformer-500m-1000g"
        elif model_name == "nt-2.5b-1000g":
            MODEL_DIR = "../../model_weight/nucleotide-transformer-2.5b-1000g"
        elif model_name == "nt-2.5b-ms":
            MODEL_DIR = "../../model_weight/nucleotide-transformer-2.5b-multi-species"
        elif model_name == "ntv2-50m-ms-3kmer":
            MODEL_DIR = "../../model_weight/nucleotide-transformer-v2-50m-3mer-multi-species"
        elif model_name == "ntv2-50m-ms":
            MODEL_DIR = "../../model_weight/nucleotide-transformer-v2-50m-multi-species"
        elif model_name == "ntv2-100m-ms":
            MODEL_DIR = "../../model_weight/nucleotide-transformer-v2-100m-multi-species"
        elif model_name == "ntv2-250m-ms":
            MODEL_DIR = "../../model_weight/nucleotide-transformer-v2-250m-multi-species"
        elif model_name == "ntv2-500m-ms":
            MODEL_DIR = "../../model_weight/nucleotide-transformer-v2-500m-multi-species"

        model = NucleotideTransformerModel(
            model_name=nt_model_name,
            model_path=MODEL_DIR,
            hf_home=HF_HOME,
            device_map=None,                 # 如需分片可设为 "auto"
            torch_dtype=None,                # 可设为 torch.bfloat16
            trust_remote_code=True,
        )

        batch_size = 16
        emb_pool = "cls"
        max_length = 6000
        # 动态检测 hidden_size（NucleotideTransformer 使用专门的检测函数，因为不支持 layer_name 参数）
        hidden_size = _infer_embedding_dim_nucleotide_transformer(
            model,
            pool=emb_pool,
            default_dim=2560,
            sample_length=min(512, max_length or 512),
        )
    elif model_name == "ntv3-8m-pre" or model_name == "ntv3-100m-pre" or model_name == "ntv3-650m-pre":
        from models.ntv3 import NTV3Model
        if model_name == "ntv3-8m-pre":
            MODEL_DIR = "../../model_weight/NTv3_8M_pre"
        elif model_name == "ntv3-100m-pre":
            MODEL_DIR = "../../model_weight/NTv3_100M_pre"
        elif model_name == "ntv3-650m-pre":
            MODEL_DIR = "../../model_weight/NTv3_650M_pre"
        model = NTV3Model(
            model_name=model_name,
            model_path=MODEL_DIR,
            trust_remote_code=True,
        )
        batch_size = 16
        emb_pool = "mean"
        max_length = 6000
        # 动态检测 hidden_size（NucleotideTransformer 使用专门的检测函数，因为不支持 layer_name 参数）
        hidden_size = _infer_embedding_dim_nucleotide_transformer(
            model,
            pool=emb_pool,
            default_dim=2560,
            sample_length=min(512, max_length or 512),
        )
    elif model_name == "ntv3-100m-post" or model_name == "ntv3-650m-post":
        from models.ntv3_post import NTV3Model
        if model_name == "ntv3-100m-post":
            MODEL_DIR = "../../model_weight/NTv3_100M_post"
        elif model_name == "ntv3-650m-post":
            MODEL_DIR = "../../model_weight/NTv3_650M_post"
        model = NTV3Model(
            model_name=model_name,
            model_path=MODEL_DIR,
            trust_remote_code=True,
        )
        batch_size = 16
        emb_pool = "mean"
        max_length = 6000
        # 动态检测 hidden_size（NucleotideTransformer 使用专门的检测函数，因为不支持 layer_name 参数）
        hidden_size = _infer_embedding_dim_nucleotide_transformer(
            model,
            pool=emb_pool,
            default_dim=2560,
            sample_length=min(512, max_length or 512),
        )
    elif model_name == "Genos-1.2B" or model_name == "Genos-10B" or model_name == "Genos-10B-v2":
        from models import GenosModel
        HF_HOME = "../../model_weight/cache"
        os.environ["HF_HOME"] = HF_HOME

        # 设置模型路径
        if model_name == "Genos-1.2B":
            MODEL_DIR = "../../model_weight/Genos-1.2B"
            default_hidden_size = 1024  # Genos-1.2B 的默认 hidden_size
            batch_size = 16
        elif model_name == "Genos-10B":
            MODEL_DIR = "../../model_weight/Genos-10B"
            default_hidden_size = 2048  # Genos-10B 的默认 hidden_size（需要根据实际模型调整）
            batch_size = 8  # 10B 模型更大，使用较小的 batch_size
        elif model_name == "Genos-10B-v2":
            MODEL_DIR = "../../model_weight/Genos-10B-v2/"
            default_hidden_size = 2048  # Genos-10B-V2 的默认 hidden_size（需要根据实际模型调整）
            batch_size = 8  # 10B-V2 模型更大，使用较小的 batch_size
        model = GenosModel(
            model_name=model_name,
            model_path=MODEL_DIR,
            hf_home=HF_HOME,
            device=None,   # 自动
            use_flash_attention=False,  # 根据环境设置，如果支持可以设为 True
        )
        emb_pool = "mean"  # Genos 主要使用 mean pooling
        max_length = 131072  # Genos 支持最大 128k 长度

        # 动态检测 hidden_size
        hidden_size = _infer_embedding_dim_genos(
            model,
            pool=emb_pool,
            default_dim=default_hidden_size,
            sample_length=min(512, max_length or 512),
        )
    elif model_name == "gpn-brassicales":
        from models import GPNBrassicalesModel
        HF_HOME = "../../model_weight/cache"
        os.environ["HF_HOME"] = HF_HOME
        MODEL_DIR = "../../model_weight/gpn-brassicales"
        model = GPNBrassicalesModel(
            model_name=model_name,
            model_path=MODEL_DIR,
            device=None,  # 自动
            load_mlm_head=True,  # 需要 PLL 评分
        )
        hidden_size = 768  # GPN-Brassicales 的默认 hidden_size
        batch_size = 16
        emb_pool = "mean"
        max_length = 128  # 根据模型配置调整
        # 动态检测 hidden_size
        hidden_size = _infer_embedding_dim(
            model,
            layer_name=None,  # GPN-Brassicales 使用 pool 参数
            pool=emb_pool,
            default_dim=768,
            sample_length=min(128, max_length or 128),
        )
    elif model_name == "GROVER" or model_name == "grover":
        from models import GROVERModel
        HF_HOME = "../../model_weight/cache"
        os.environ["HF_HOME"] = HF_HOME
        MODEL_DIR = "../../model_weight/GROVER"
        model = GROVERModel(
            model_name="GROVER",
            model_path=MODEL_DIR,
            hf_home=HF_HOME,
            device=None,  # 自动
            use_mlm_head=True,  # 需要 PLL 评分
        )
        hidden_size = 768  # GROVER 的默认 hidden_size
        batch_size = 16
        emb_pool = "mean"
        max_length = 512  # GROVER 的默认最大长度
        # 动态检测 hidden_size
        hidden_size = _infer_embedding_dim(
            model,
            layer_name=None,  # GROVER 使用 pool 参数
            pool=emb_pool,
            default_dim=768,
            sample_length=min(512, max_length or 512),
        )
    elif model_name == "GenomeOcean-100M" or model_name == "GenomeOcean-500M" or model_name =="GenomeOcean-4B":
        from models.genomeocean import GenomeOceanModel
        if model_name == "GenomeOcean-100M":
            MODEL_DIR = "../../model_weight/GenomeOcean-100M"
        elif model_name == "GenomeOcean-500M":
            MODEL_DIR = "../../model_weight/GenomeOcean-500M"
        elif model_name == "GenomeOcean-4B":
            MODEL_DIR = "../../model_weight/GenomeOcean-4B"
        model = GenomeOceanModel(
            model_name=model_name,
            model_path=MODEL_DIR)
        batch_size = 16
        emb_pool = "mean"
        max_length = 16384
        hidden_size = _infer_embedding_dim(
            model,
            layer_name=None,
            pool=emb_pool,
            default_dim=1024,
            sample_length=min(512, max_length or 512),
        )
    elif model_name == "GENERator-v2-eukaryote-1.2b-base" or model_name == "GENERator-v2-eukaryote-3b-base" or model_name == "GENERator-v2-prokaryote-1.2b-base" or model_name == "GENERator-v2-prokaryote-3b-base":
        from models.GENERator import GENERatorModel
        if model_name == "GENERator-v2-eukaryote-1.2b-base":
            MODEL_DIR = "../../model_weight/GENERator-v2-eukaryote-1.2b-base"
        elif model_name == "GENERator-v2-eukaryote-3b-base":
            MODEL_DIR = "../../model_weight/GENERator-v2-eukaryote-3b-base"
        elif model_name == "GENERator-v2-prokaryote-1.2b-base":
            MODEL_DIR = "../../model_weight/GENERator-v2-prokaryote-1.2b-base"
        elif model_name == "GENERator-v2-prokaryote-3b-base":
            MODEL_DIR = "../../model_weight/GENERator-v2-prokaryote-3b-base"
        model = GENERatorModel(
            model_name=model_name,
            model_path=MODEL_DIR,
            tokenizer_padding_side="right",  # ✅ 与官方默认一致
            kmer_pad_side="right",           # ✅ 关键：不要左侧补齐
            )
        batch_size = 16
        # 与官方 sequence_understanding 一致：使用 last non-padded token (EOS) embedding，不用 max
        emb_pool = "mean"
        max_length = 16384
        # GENERator config 中 hidden_size=2048；_infer 可能因 BFloat16 失败回退到 1024，导致分类头维度错误、MCC≈0
        _cfg = getattr(getattr(model, "model", None), "config", None)
        hidden_size = getattr(_cfg, "hidden_size", None) if _cfg is not None else None
        if hidden_size is None:
            hidden_size = _infer_embedding_dim(
                model,
                layer_name=None,
                pool=emb_pool,
                default_dim=2048,  # 1.2b/3b 均为 2048，与 config.json 一致
                sample_length=min(512, max_length or 512),
            )
    elif model_name == "BioFM-265M":
        from models.biofm import BioFMModel
        MODEL_DIR = "../../model_weight/BioFM-265M"
        model = BioFMModel(
            tokenizer_path=MODEL_DIR,
            model_path=MODEL_DIR)
        batch_size = 16
        emb_pool = "mean"
        max_length = 16384
        hidden_size = _infer_embedding_dim(
            model,
            layer_name=None,
            pool=emb_pool,
            default_dim=1024,
            sample_length=min(512, max_length or 512),
        )
    elif model_name == "AIDO.DNA-300M" or model_name == "AIDO.DNA-7B":
        from models.aidoDNA import AIDOModel
        if model_name == "AIDO.DNA-300M":
            MODEL_DIR = "../../model_weight/AIDO.DNA-300M"
        elif model_name == "AIDO.DNA-7B":
            MODEL_DIR = "../../model_weight/AIDO.DNA-7B"
        REPO_ROOT = "../../model/ModelGenerator" 
        
        CODE_DIR = os.path.join(REPO_ROOT, "huggingface", "aido.rna", "aido_rna", "models")

        model = AIDOModel(
            model_name=model_name,
            model_path=MODEL_DIR,
            rnabert_code_path=CODE_DIR,  # <--- 这里传入路径，它就会自己去读了
            trust_remote_code=True,
            device="cuda",
            torch_dtype="auto"
        )
        batch_size = 16
        emb_pool = "mean"
        max_length = 16384
        hidden_size = _infer_embedding_dim(
            model,
            layer_name=None,
            pool=emb_pool,
            default_dim=1024,
            sample_length=min(512, max_length or 512),
        )
    elif model_name == "AIDO.RNA-650M" or model_name == "AIDO.RNA-1.6B" or model_name == "AIDO.RNA-650M-CDS" or model_name == "AIDO.RNA-1.6B-CDS":
        from models.aidoRNA import AIDORNAModel
        if model_name == "AIDO.RNA-650M":
            MODEL_DIR = "../../model_weight/AIDO.RNA-650M"
        elif model_name == "AIDO.RNA-1.6B":
            MODEL_DIR = "../../model_weight/AIDO.RNA-1.6B"
        elif model_name == "AIDO.RNA-650M-CDS":
            MODEL_DIR = "../../model_weight/AIDO.RNA-650M-CDS"
        elif model_name == "AIDO.RNA-1.6B-CDS":
            MODEL_DIR = "../../model_weight/AIDO.RNA-1.6B-CDS"
        REPO_ROOT = "../../model/ModelGenerator" 
        CODE_DIR = os.path.join(REPO_ROOT, "huggingface", "aido.rna", "aido_rna", "models")
        
        model = AIDORNAModel(
            model_name=model_name,
            model_path=MODEL_DIR,
            rnabert_code_path=CODE_DIR, # 同样需要这个来加载 RNABert 架构
            trust_remote_code=True,
            torch_dtype="auto"
        )
        batch_size = 16
        emb_pool = "mean"
        max_length = 16384
        hidden_size = _infer_embedding_dim(
            model,
            layer_name=None,
            pool=emb_pool,
            default_dim=1024,
            sample_length=min(512, max_length or 512),
        )
    elif model_name == "OmniReg-bigbird" or model_name =="OmniReg-base" or model_name == "OmniReg-large":
        from models.omnireg_model import OmniRegGPTModel

        if model_name == "OmniReg-bigbird":
            MODEL_CKPT = "../../model_weight/gena-lm-bigbird-base-t2t/pytorch_model.bin"
            TOKENIZER_DIR = "../../model_weight/omnireg_bigbird"
        elif model_name == "OmniReg-base":
            MODEL_CKPT = "../../model_weight/gena-lm-bert-base-t2t/pytorch_model.bin"
            TOKENIZER_DIR = "../../model_weight/gena-lm-bert-base-t2t"
        elif model_name == "OmniReg-large":
            MODEL_CKPT = "../../model_weight/gena-lm-bert-large-t2t/pytorch_model.bin"
            TOKENIZER_DIR = "../../model_weight/gena-lm-bert-large-t2t"
        TOKENIZER_DIR = "../../model_weight/gena-lm-bert-large-t2t"
        OMNIREG_REPO = os.path.join(PROJECT_ROOT, "models", "OmniReg-GPT")
        HF_HOME = "../../model_weight/cache"

        model = OmniRegGPTModel(
            model_name=model_name,
            model_path=MODEL_CKPT,
            tokenizer_path=TOKENIZER_DIR,
            omnireg_repo_path=OMNIREG_REPO,
            hf_home=HF_HOME,
            device=None,
            max_length=16384,
        )
        batch_size = 4
        emb_pool = "mean"
        max_length = 16384
        hidden_size = _infer_embedding_dim(
            model,
            layer_name=None,
            pool=emb_pool,
            default_dim=1024,
            sample_length=min(512, max_length or 512),
        )
    elif model_name == "RNA-FM":
        from models.rnafm_model import RNAFMModel
        MODEL_DIR = "../../model_weight/rnafm"
        HF_HOME = "../../model_weight/cache"
        model = RNAFMModel(
            model_name=model_name,
            model_path=MODEL_DIR,
            hf_home=HF_HOME,
            device=None,
        )
        batch_size = 1
        emb_pool = "mean"
        max_length = 1022
        hidden_size = _infer_embedding_dim(
            model,
            layer_name=None,
            pool=emb_pool,
            default_dim=640,
            sample_length=min(512, max_length or 512),
        )
    elif model_name == "RiNALMo" or model_name == "RiNALMo-giga":
        from models.rinalmo_model import RiNALMoModel
        MODEL_DIR = "../../model_weight/rinalmo-mega"
        HF_HOME = "../../model"
        model = RiNALMoModel(
            model_name=model_name,
            model_path=MODEL_DIR,
            hf_home=HF_HOME,
            device=None,
            use_mlm_head=True,
        )
        batch_size = 16
        emb_pool = "mean"
        max_length = 1024
        hidden_size = _infer_embedding_dim(
            model,
            layer_name=None,
            pool=emb_pool,
            default_dim=1024,
            sample_length=min(512, max_length or 512),
        )
    elif model_name == "BiRNA-BERT":
        from models.birna_bert import BiRNABERTModel
        MODEL_DIR = "../../model_weight/birna-bert"
        TOKENIZER_DIR = "../../model_weight/birna-tokenizer"

    
        model = BiRNABERTModel(
            model_name=model_name, 
            model_path=MODEL_DIR, 
            tokenizer_path=TOKENIZER_DIR
        )
        batch_size = 16
        emb_pool = "mean"
        max_length = 16384
        hidden_size = _infer_embedding_dim(
            model,
            layer_name=None,
            pool=emb_pool,
            default_dim=1024,
            sample_length=min(512, max_length or 512),
        )
    elif model_name == "RNABERT":
        from models.rnabert import RNABERTModel
        MODEL_PATH = "../../model/bert_mul_2.pth" 
        CONFIG_PATH = "../../model/RNABERT/RNA_bert_config.json"

        model = RNABERTModel(
            model_name=model_name,
            model_path=MODEL_PATH,
            config_path=CONFIG_PATH,
        )
        batch_size = 16
        emb_pool = "mean"
        max_length = 16384
        hidden_size = _infer_embedding_dim(
            model,
            layer_name=None,
            pool=emb_pool,
            default_dim=1024,
            sample_length=min(512, max_length or 512),
        )
    elif model_name== "MP-RNA":
        from models.mp_rna_model import MPRNAModel
        MODEL_PATH = "../../model_weight/MP-RNA"
    
        # 假设你没有单独的 HF_HOME 需求，或者在外部设置好了
        # 实例化 MP-RNA 模型
        model = MPRNAModel(
            model_name="MP-RNA",
            model_path=MODEL_PATH,
            trust_remote_code=True # 关键：允许执行模型文件夹里的 Python 代码
        )
        batch_size = 16
        emb_pool = "mean"
        max_length = 16384
        hidden_size = _infer_embedding_dim(
            model,
            layer_name=None,
            pool=emb_pool,
            default_dim=1024,
            sample_length=min(512, max_length or 512),
        )
    elif model_name.startswith("ESM-1b") or model_name == "ESM":
        from models.esm_model import ESMModel
        if model_dir:
            MODEL_PATH = model_dir
        else:
            MODEL_PATH = "../../model_weight/esm-1b/esm1b_t33_650M_UR50S.pt"

        model = ESMModel(
            model_name=model_name,
            model_path=MODEL_PATH,
            device=None,
            translation_mode="first_orf",
        )
        batch_size = 16
        emb_pool = "mean"
        max_length = 1022
        hidden_size = _infer_embedding_dim(
            model,
            layer_name=None,
            pool=emb_pool,
            default_dim=1280,
            sample_length=min(512, max_length or 512),
        )
    elif model_name.startswith("ESM-2") or model_name == "ESM2":
        from models.esm2_model import ESM2Model
        if model_dir:
            MODEL_PATH = model_dir
        else:
            MODEL_PATH = "../../model_weight/esm2_t33_650M_UR50D"
        HF_HOME = "../../model_weight/cache"
        os.environ["HF_HOME"] = HF_HOME

        model = ESM2Model(
            model_name=model_name,
            model_path=MODEL_PATH,
            device=None,
            translation_mode="first_orf",
        )
        batch_size = 16
        emb_pool = "mean"
        max_length = 1024
        hidden_size = _infer_embedding_dim(
            model,
            layer_name=None,
            pool=emb_pool,
            default_dim=1280,
            sample_length=min(512, max_length or 512),
        )
    elif model_name in {"physchem-distill", "PhyschemDistill"}:
        from models import PhyschemDistillModel
        MODEL_DIR = model_dir or os.path.join(
            PROJECT_ROOT, "models", "Physchem-distill")
        model = PhyschemDistillModel(
            model_name="physchem-distill",
            model_path=MODEL_DIR,
            device=None,
            score_reduce="mean",
        )
        batch_size = 128
        emb_pool = "mean"
        max_length = model.max_length
        hidden_size = getattr(getattr(model, "config", None), "d_model", 128)
        hidden_size = 21639
    elif model_name == "CNN":
        from models.cnn import CNNConfig, GenomeCNN1D
        cfg = CNNConfig()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = GenomeCNN1D(out_dim=1, cfg=cfg).to(device)
        batch_size = 256
        emb_pool = "mean"
        max_length = None
        hidden_size = cfg.channels[-1]
    else:
        raise ValueError(f"[ERROR] model_name = {model_name} not found")

    if args_hidden_size != 0:
        hidden_size = args_hidden_size
    print(f"[INFO] hidden_size = {hidden_size}")
    if int(window_len) <= 0:
        raise ValueError("--window_len 必须为正整数（用于切分窗口）")

    # 支持12个子数据集：{na_type}-{label}-{check}
    # na_type: ALL, DNA, RNA
    # label: host, taxon
    # check: genus, times
    all_list = ["ALL", "DNA", "RNA"]
    LABELS = ["host", "taxon"]
    check_list = ["genus", "times"]
    
    # 解析数据集名称
    parts = dataset_name.split("-")
    if len(parts) == 3:
        na_type, label, check = parts
        if na_type not in all_list or label not in LABELS or check not in check_list:
            raise ValueError(f"[ERROR] 无效的数据集名称: {dataset_name}。格式应为: {{na_type}}-{{label}}-{{check}}，其中 na_type∈{all_list}, label∈{LABELS}, check∈{check_list}")
    else:
        raise ValueError(f"[ERROR] 无效的数据集名称: {dataset_name}。格式应为: {{na_type}}-{{label}}-{{check}}，其中 na_type∈{all_list}, label∈{LABELS}, check∈{check_list}")

    # 构建数据集路径
    split_dir = f"../../GeneShield/data/all_viral/cls_data/{na_type}/{label}/{check}"
    
    # 根据label类型确定标签列
    if label == "taxon":
        labels = ["kingdom", "phylum", "class", "order", "family"]
    elif label == "host":
        labels = ["host_label"]
    else:
        raise ValueError(f"[ERROR] 无效的label类型: {label}")

    print(f"[INFO] Dataset: {dataset_name}")
    print(f"[INFO] split_dir = {split_dir}")
    print(f"[INFO] window_len = {int(window_len)}, train_num_windows = {int(train_num_windows)}, eval_num_windows = {int(eval_num_windows)}")
    
    base = VirusSplitDatasets(
        split_dir,
        label_cols=labels,
        return_format="dict",
        attach_sequences=True,
    )
    win = base.make_windowed(
        window_len=int(window_len),
        train_num_windows=int(train_num_windows),
        eval_num_windows=int(eval_num_windows),
        seed=42,
        return_format="dict",
    )

    print("[OK] base sizes (train/val/test)="f"{len(base.train)}/{len(base.val)}/{len(base.test)}")
    print("[OK] windowed sizes (train/val/test)="f"{len(win.train)}/{len(win.val)}/{len(win.test)}")
    print("[OK] label2id keys:", list(base.label2id.keys()))
    
    # 调试：检查原始数据集的标签分布
    print("[DEBUG] 检查原始数据集的标签分布...")
    for task_name in labels:
        # 检查训练集
        train_labels_base = []
        for i in range(min(1000, len(base.train))):  # 只检查前1000个样本以节省时间
            sample = base.train[i]
            if isinstance(sample, dict):
                if "labels" in sample:
                    labels_val = sample["labels"]
                    if isinstance(labels_val, np.ndarray):
                        task_idx = labels.index(task_name)
                        if task_idx < len(labels_val):
                            train_labels_base.append(int(labels_val[task_idx]))
                    elif isinstance(labels_val, (int, np.integer)):
                        if labels.index(task_name) == 0:
                            train_labels_base.append(int(labels_val))
        
        # 检查验证集
        val_labels_base = []
        for i in range(len(base.val)):
            sample = base.val[i]
            if isinstance(sample, dict):
                if "labels" in sample:
                    labels_val = sample["labels"]
                    if isinstance(labels_val, np.ndarray):
                        task_idx = labels.index(task_name)
                        if task_idx < len(labels_val):
                            val_labels_base.append(int(labels_val[task_idx]))
                    elif isinstance(labels_val, (int, np.integer)):
                        if labels.index(task_name) == 0:
                            val_labels_base.append(int(labels_val))
        
        if train_labels_base:
            train_unique = len(np.unique(train_labels_base))
            train_counter = Counter(train_labels_base)
            print(f"[DEBUG] base_train_{task_name}: n={len(train_labels_base)} (sampled), unique={train_unique}, top5={train_counter.most_common(5)}")
        if val_labels_base:
            val_unique = len(np.unique(val_labels_base))
            val_counter = Counter(val_labels_base)
            print(f"[DEBUG] base_val_{task_name}: n={len(val_labels_base)}, unique={val_unique}, top5={val_counter.most_common(5)}")
            if val_unique == 1:
                print(f"[WARN] base_val_{task_name}: 原始验证集所有标签都相同! label={val_labels_base[0]}")
                print(f"[WARN] 这会导致MCC=0，因为验证集没有类别多样性。")
                print(f"[WARN] 建议检查数据划分，确保验证集包含所有类别的样本。")

    # 多任务输出维度（每个分类层级一个 head）
    task_dims = {name: len(base.label2id[name]) for name in labels}

    # 计算类别权重以处理样本不平衡问题
    print("[INFO] 计算类别权重以处理样本不平衡...")
    class_weights_dict = {}
    for task_idx, task_name in enumerate(labels):
        # 从训练数据集中提取该任务的标签
        train_labels = []
        for i in range(len(win.train)):
            sample = win.train[i]
            if isinstance(sample, dict):
                # 数据集返回dict格式，标签在"labels"字段中（numpy数组）或单独的列中
                if "labels" in sample:
                    labels_val = sample["labels"]
                    if isinstance(labels_val, np.ndarray):
                        # 多任务：labels是数组，索引对应任务顺序
                        if task_idx < len(labels_val):
                            label_val = labels_val[task_idx]
                            if isinstance(label_val, (int, np.integer)):
                                train_labels.append(int(label_val))
                            elif isinstance(label_val, torch.Tensor):
                                train_labels.append(int(label_val.item()))
                    elif isinstance(labels_val, (int, np.integer)):
                        # 单任务：labels是单个值
                        if task_idx == 0:
                            train_labels.append(int(labels_val))
                # 也尝试从单独的列中获取（如果存在）
                label_key = f"{task_name}__id"
                if label_key in sample and not train_labels:
                    label_val = sample[label_key]
                    if isinstance(label_val, (int, np.integer)):
                        train_labels.append(int(label_val))
                    elif isinstance(label_val, torch.Tensor):
                        train_labels.append(int(label_val.item()))
                    elif isinstance(label_val, np.ndarray):
                        train_labels.append(int(label_val.item()))
        
        # 调试：检查提取的标签
        if train_labels:
            train_labels_array = np.array(train_labels)
            unique_labels = np.unique(train_labels_array)
            print(f"[DEBUG] {task_name}: 提取了 {len(train_labels)} 个标签, 唯一值数量={len(unique_labels)}, 唯一值={unique_labels[:10]}")
            if len(unique_labels) == 1:
                print(f"[WARN] {task_name}: 所有训练标签都相同! label={unique_labels[0]}")
        
        if train_labels:
            train_labels_array = np.array(train_labels)
            # 过滤掉无效标签（-1表示缺失）
            valid_mask = train_labels_array >= 0
            if valid_mask.sum() > 0:
                valid_labels = train_labels_array[valid_mask]
                # 计算类别权重（使用balanced策略）
                unique_labels = np.unique(valid_labels)
                class_weights = compute_class_weight(
                    'balanced',
                    classes=unique_labels,
                    y=valid_labels
                )
                # 创建完整的权重向量（包括所有类别，即使某些类别在训练集中没有出现）
                num_classes = len(base.label2id[task_name])
                weight_tensor = torch.ones(num_classes, dtype=torch.float32)
                for idx, label_id in enumerate(unique_labels):
                    if 0 <= label_id < num_classes:
                        weight_tensor[label_id] = float(class_weights[idx])
                
                class_weights_dict[task_name] = weight_tensor
                print(f"[INFO] {task_name}: 类别数量={num_classes}, 训练样本数={len(valid_labels)}, 权重范围=[{weight_tensor.min():.3f}, {weight_tensor.max():.3f}]")
            else:
                print(f"[WARN] {task_name}: 训练集中没有有效标签，跳过类别权重计算")
        else:
            print(f"[WARN] {task_name}: 无法从训练集中提取标签，跳过类别权重计算")
    
    # 如果没有多任务，使用单个权重张量；否则使用字典
    if len(labels) == 1:
        class_weights = class_weights_dict.get(labels[0], None)
    else:
        class_weights = class_weights_dict if class_weights_dict else None

    if model_name == "CNN":
        from models.cnn import CNNConfig, GenomeCNN1D
        cfg = CNNConfig()
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        model = GenomeCNN1D(out_dim=task_dims, cfg=cfg).to(device)
        hidden_size = cfg.channels[-1]
        head = model
        embedder = None
    else:
        head = MultiTaskMLPHead(hidden_size, task_dims)
        embedder = model

    if head_batch_size <= 0:
        head_batch_size = 64
    if emb_batch_size_override > 0:
        emb_batch_size = emb_batch_size_override
    else:
        emb_batch_size = batch_size

    output_dir = os.path.join(
        results_root, f"Classification/{dataset_name}/{model_name}/{window_len}_{train_num_windows}_{eval_num_windows}/{lr}/")
    embedding_save_dir= os.path.join(
        results_root, f"Classification/{dataset_name}/{model_name}/{window_len}_{train_num_windows}_{eval_num_windows}/embeddings/")
    plot_image = os.path.join(
        results_root, f"Classification/{dataset_name}/{model_name}/{window_len}_{train_num_windows}_{eval_num_windows}/plots/")
    evaluator = FineTuneSeqEvaluator(
        embedder=embedder,
        model=head,
        train_ds=win.train,
        val_ds=win.val,
        test_ds=win.test,
        output_dir=output_dir,
        embedding_save_dir=embedding_save_dir,
        task="multiclass",
        lr=lr,
        weight_decay=0.01,
        num_epochs=300,
        batch_size=head_batch_size,
        emb_pool=emb_pool,
        emb_batch_size=emb_batch_size,
        emb_layer_name=emb_layer_name,
        multitask=True,
        task_names=labels,
        num_workers=num_workers,
        early_stopping_patience=early_stopping_patience,
        early_stopping_metric=early_stopping_metric,
        force_recompute_embeddings=force_recompute_embeddings,
        class_weights=class_weights,  # 传入类别权重
        save_predictions=save_predictions,  # 是否保存预测结果
    )
    summary = evaluator.run()
    print("[OK] multitask summary:", summary.get("output_dir"))

    # 记录并保存 embedding 提取与分类头训练用时
    emb_sec = summary.get("time_embedding_extract_sec") or 0.0
    head_sec = summary.get("time_head_training_sec") or 0.0
    timing_records.append({
        "阶段": "提取embedding",
        "用时(秒)": emb_sec,
        "model_name": model_name,
        "dataset_name": dataset_name,
    })
    timing_records.append({
        "阶段": "训练分类头",
        "用时(秒)": head_sec,
        "model_name": model_name,
        "dataset_name": dataset_name,
    })
    if timing_log_csv:
        save_timing_records(timing_records, timing_log_csv)
    print(f"[INFO] 提取embedding用时: {emb_sec:.2f} 秒 ({emb_sec/60:.2f} 分钟)")
    print(f"[INFO] 训练分类头用时: {head_sec:.2f} 秒 ({head_sec/60:.2f} 分钟)")

"""

ssd
/inspire/ssd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/results
CNN:evo2
python script/run_all.py --model_name CNN --dataset_name c2-genus
evo2_1b_base:evo2
# evo2
python script/run_all.py --model_name evo2_1b_base --dataset_name DNA-taxon-genus
# evo1
python script/run_all.py --model_name evo-1-8k-base --dataset_name c2-genus
# evo1.5
python script/run_all.py --model_name evo-1.5-8k-base --dataset_name c2-genus
# hyenadna
python script/run_all.py --model_name hyena_local --dataset_name RNA-taxon-genus

python script/run_all.py --model_name hyena_local-12M-mini-virus --dataset_name DNA-taxon-genus --force_recompute_embeddings True

python script/run_all.py --model_name hyena_local-test --dataset_name DNA-taxon-genus --force_recompute_embeddings True

python script/run_all.py --model_name hyena_local --dataset_name DNA-taxon-genus --force_recompute_embeddings True

python script/run_all.py --model_name nt-2.5b-1000g --dataset_name DNA-taxon-genus

python script/run_all.py --model_name hyena_local-3.2M-virus --dataset_name DNA-taxon-genus --save_predictions True

python script/run_all.py --model_name LucaVirus-default-step3.8M --dataset_name ALL-taxon-genus --save_predictions True

python script/run_all.py --model_name RNA-FM --dataset_name ALL-taxon-genus --save_predictions True --force_recompute_embeddings True

# 快速测试
python script/run_all.py --model_name nt-500m-human --dataset_name DNA-taxon-genus --force_recompute_embeddings True --train_num_windows 2 --eval_num_windows 4
python script/run_all.py --model_name GENERator-v2-eukaryote-1.2b-base --dataset_name DNA-taxon-genus --force_recompute_embeddings True --early_stopping_patience 100 

python script/run_all.py --model_name GENERator-v2-prokaryote-1.2b-base --dataset_name DNA-taxon-genus --force_recompute_embeddings True --train_num_windows 2 --eval_num_windows 4 --lr 1e-6


python script/run_all.py --model_name hyena_local --dataset_name RNA-host-times --force_recompute_embeddings True --model_dir ../../GeneShield/pretrain/hyena-dna/hyena_local-test --window_len 1024 --train_num_windows 4 --eval_num_windows 32

python script/run_all.py --model_name hyena_local --dataset_name RNA-host-times --force_recompute_embeddings True --model_dir ../../GeneShield/pretrain/hyena-dna/hyena_local-test --window_len 512


"""

def main():
    parser = argparse.ArgumentParser(
        description="接收两个参数：model_name 和 dataset_name"
    )
    parser.add_argument("--model_name", required=True,
                        help="模型名称，例如 qwen3-14b")
    parser.add_argument("--dataset_name", required=True,
                        help="数据集名称，例如 ceval；传 all 将依次评估所有已支持的数据集")
    parser.add_argument("--model_dir", required=False,
                        help="模型路径，Nemotron和HyenaDNA系列模型需要")
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=0,
        help="模型隐层维度/embedding 投影维度（正整数），默认 768"
    )
    parser.add_argument(
        "--results_root_name",
        type=str,
        default="results",
        help="结果保存目录名称，默认为 results",
    )
    parser.add_argument(
        "--timing_log_csv",
        type=str,
        default=None,
        help="时间记录CSV文件路径，默认为项目根目录下的 time_log.csv",
    )
    parser.add_argument(
        "--window_len",
        type=int,
        default=512,
        help="序列切分窗口长度（C1 任务使用；val/test 覆盖全部窗口）",
    )
    parser.add_argument(
        "--train_num_windows",
        type=int,
        default=8,
        help="训练集：每条序列随机采样的窗口个数（C1 任务使用；0 表示训练也覆盖全部窗口）",
    )
    parser.add_argument(
        "--eval_num_windows",
        type=int,
        default=64,
        help="验证/测试集：每条序列随机采样的窗口个数（-1 表示全覆盖）",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="学习率（默认 1e-4）",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="DataLoader 进程数（默认 8）",
    )
    parser.add_argument(
        "--head_batch_size",
        type=int,
        default=0,
        help="分类头训练 batch_size（<=0 使用默认 4096）",
    )
    parser.add_argument(
        "--emb_batch_size",
        type=int,
        default=0,
        help="embedding 提取 batch_size（<=0 使用模型默认）",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=30,
        help="Early stopping patience（默认 30）",
    )
    parser.add_argument(
        "--early_stopping_metric",
        type=str,
        default="mcc",
        choices=["mcc", "accuracy", "acc", "f1_macro", "f1", "auprc"],
        help="早停指标（默认 mcc，可选：mcc, accuracy/acc, f1_macro/f1, auprc）",
    )
    parser.add_argument(
        "--force_recompute_embeddings",
        type=bool,
        default=False,
        help="是否强制重新计算 embedding（忽略缓存）",
    )
    parser.add_argument(
        "--save_predictions",
        type=bool,
        default=False,
        help="是否保存预测结果",
    )
    args = parser.parse_args()

    # 确定结果保存目录
    results_root_name = args.results_root_name

    run(
        args.model_name,
        args.dataset_name,
        args.model_dir,
        args.hidden_size,
        results_root_name,
        timing_records=None,
        timing_log_csv=args.timing_log_csv,
        window_len=args.window_len,
        train_num_windows=args.train_num_windows,
        eval_num_windows=args.eval_num_windows,
        lr=args.lr,
        num_workers=args.num_workers,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_metric=args.early_stopping_metric,
        head_batch_size=args.head_batch_size,
        emb_batch_size_override=args.emb_batch_size,
        force_recompute_embeddings=args.force_recompute_embeddings,
        save_predictions=args.save_predictions,
    )


if __name__ == "__main__":
    main()

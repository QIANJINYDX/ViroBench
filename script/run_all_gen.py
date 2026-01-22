#!/usr/bin/env python3
import argparse
import sys
import os
import random
import re
import traceback
import math
from typing import Dict
import torch
import pandas as pd
import time
from datetime import datetime

# from CUR_TEST.plot import json_path


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Put project root at the front so local packages (e.g. `datasets/`) win over site-packages.
sys.path.insert(0, PROJECT_ROOT)

import torch.nn as nn
from models.mlp_head import MLPHead
from datasets.virus_datasets import VirusSplitDatasets
from evaluators.finetune import FineTuneSeqEvaluator
from datasets.gen_datasets import GenDataset
from evaluators.gen import GenEvaluator


def run(
    model_name: str,
    dataset_name: str,
    results_root_name: str = "results",
    prompt_len: int = -1,
    temperature: float = 1.0,
    top_k: int = 4,
) -> None:
    print(f"[INFO] model_name = {model_name}")
    print(f"[INFO] dataset_name = {dataset_name}")
    if os.path.isabs(results_root_name):
        results_root = results_root_name
    else:
        results_root = os.path.join(PROJECT_ROOT, results_root_name)
    os.makedirs(results_root, exist_ok=True)
    print(f"[INFO] results_root = {results_root}")
    
    if model_name == "evo-1-8k-base" or model_name == "evo-1-131k-base" or model_name == "evo-1.5-8k-base":
        from models import Evo1Model
        MODEL_DIR = f"/mnt/s3mount/model_weight/{model_name}"
        CFG_PATH = f"/mnt/s3mount/model_weight/{model_name}/{model_name}_inference.yml"
        HF_HOME = "/mnt/s3mount/model_weight/cache"
        model = Evo1Model(model_name, MODEL_DIR,
                          CFG_PATH, HF_HOME, device=None)
    elif model_name == "evo2_1b_base" or model_name == "evo2_7b_base" or model_name == "evo2_40b_base" or model_name == "evo2_40b" or model_name == "evo2_7b":
        from models import Evo2Model
        MODEL_DIR = f"/mnt/s3mount/wuyucheng/weight/{model_name}/{model_name}.pt"
        model = Evo2Model(model_name, MODEL_DIR)
    elif model_name == "hyenadna" or model_name == "hyenadna-tiny-16k" or model_name == "hyenadna-tiny-1k" or model_name == "hyenadna-small-32k" or model_name == "hyenadna-medium-160k" or model_name == "hyenadna-medium-450k" or model_name == "hyenadna-large-1m":
        from models import HyenaDNAModel
        HF_HOME = "/mnt/s3mount/model_weight/cache"
        os.environ["HF_HOME"] = HF_HOME

        if model_name == "hyenadna-tiny-16k":
            MODEL_DIR = "/mnt/s3mount/model_weight/hyenadna-tiny-16k-seqlen-d128-hf"
        elif model_name == "hyenadna-tiny-1k":
            MODEL_DIR = "/mnt/s3mount/model_weight/hyenadna-tiny-1k-seqlen-hf"
        elif model_name == "hyenadna-small-32k":
            MODEL_DIR = "/mnt/s3mount/model_weight/hyenadna-small-32k-seqlen-hf"
        elif model_name == "hyenadna-medium-160k":
            MODEL_DIR = "/mnt/s3mount/model_weight/hyenadna-medium-160k-seqlen-hf"
        elif model_name == "hyenadna-medium-450k":
            MODEL_DIR = "/mnt/s3mount/model_weight/hyenadna-medium-450k-seqlen-hf"
        elif model_name == "hyenadna-large-1m":
            MODEL_DIR = "/mnt/s3mount/model_weight/hyenadna-large-1m-seqlen-hf"

        # 从路径中提取模型名称（去掉路径前缀和可能的后缀）
        # 例如: /mnt/s3mount/model_weight/hyenadna-large-1m-seqlen-hf -> hyenadna-large-1m
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
    elif model_name == "Genos-1.2B" or model_name == "Genos-10B" or model_name == "Genos-10B-v2":
        from models.genos_model_gen import GenosModel
        HF_HOME = "/mnt/s3mount/model_weight/cache"
        os.environ["HF_HOME"] = HF_HOME

        # 设置模型路径
        if model_name == "Genos-1.2B":
            MODEL_DIR = "/mnt/s3mount/model_weight/Genos-1.2B"
            default_hidden_size = 1024  # Genos-1.2B 的默认 hidden_size
            batch_size = 16
        elif model_name == "Genos-10B":
            MODEL_DIR = "/mnt/s3mount/model_weight/Genos-10B"
            default_hidden_size = 2048  # Genos-10B 的默认 hidden_size（需要根据实际模型调整）
            batch_size = 8  # 10B 模型更大，使用较小的 batch_size
        elif model_name == "Genos-10B-v2":
            MODEL_DIR = "/mnt/s3mount/model_weight/Genos-10B-v2/"
            default_hidden_size = 2048  # Genos-10B-V2 的默认 hidden_size（需要根据实际模型调整）
            batch_size = 8  # 10B-V2 模型更大，使用较小的 batch_size
        model = GenosModel(
            model_name=model_name,
            model_path=MODEL_DIR,
            hf_home=HF_HOME,
            device=None,   # 自动
            use_flash_attention=False,  # 根据环境设置，如果支持可以设为 True
        )
    elif model_name == "GenomeOcean-100M" or model_name == "GenomeOcean-500M" or model_name =="GenomeOcean-4B":
        from models.genomeocean_gen import GenomeOceanModel
        if model_name == "GenomeOcean-100M":
            MODEL_DIR = "/mnt/s3mount/model_weight/GenomeOcean-100M"
        elif model_name == "GenomeOcean-500M":
            MODEL_DIR = "/mnt/s3mount/model_weight/GenomeOcean-500M"
        elif model_name == "GenomeOcean-4B":
            MODEL_DIR = "/mnt/s3mount/model_weight/GenomeOcean-4B"
        model = GenomeOceanModel(
            model_name=model_name,
            model_path=MODEL_DIR)

    elif model_name == "GENERator-v2-eukaryote-1.2b-base" or model_name == "GENERator-v2-eukaryote-3b-base" or model_name == "GENERator-v2-prokaryote-1.2b-base" or model_name == "GENERator-v2-prokaryote-3b-base":
        from models.GENERator_gen import GENERatorModel
        if model_name == "GENERator-v2-eukaryote-1.2b-base":
            MODEL_DIR = "/mnt/s3mount/model_weight/GENERator-v2-eukaryote-1.2b-base"
        elif model_name == "GENERator-v2-eukaryote-3b-base":
            MODEL_DIR = "/mnt/s3mount/model_weight/GENERator-v2-eukaryote-3b-base"
        elif model_name == "GENERator-v2-prokaryote-1.2b-base":
            MODEL_DIR = "/mnt/s3mount/model_weight/GENERator-v2-prokaryote-1.2b-base"
        elif model_name == "GENERator-v2-prokaryote-3b-base":
            MODEL_DIR = "/mnt/s3mount/model_weight/GENERator-v2-prokaryote-3b-base"
        model = GENERatorModel(
            model_name=model_name,
            model_path=MODEL_DIR)
    elif model_name == "OmniReg-bigbird" or model_name =="OmniReg-base" or model_name == "OmniReg-large":
        from models.omnireg_model import OmniRegGPTModel

        if model_name == "OmniReg-bigbird":
            MODEL_CKPT = "/mnt/s3mount/model_weight/gena-lm-bigbird-base-t2t/pytorch_model.bin"
            TOKENIZER_DIR = "/mnt/s3mount/model_weight/omnireg_bigbird"
        elif model_name == "OmniReg-base":
            MODEL_CKPT = "/mnt/s3mount/model_weight/gena-lm-bert-base-t2t/pytorch_model.bin"
            TOKENIZER_DIR = "/mnt/s3mount/model_weight/gena-lm-bert-base-t2t"
        elif model_name == "OmniReg-large":
            MODEL_CKPT = "/mnt/s3mount/model_weight/gena-lm-bert-large-t2t/pytorch_model.bin"
            TOKENIZER_DIR = "/mnt/s3mount/model_weight/gena-lm-bert-large-t2t"
        TOKENIZER_DIR = "/mnt/s3mount/model_weight/gena-lm-bert-large-t2t"
        OMNIREG_REPO = os.path.join(PROJECT_ROOT, "models", "OmniReg-GPT")
        HF_HOME = "/mnt/s3mount/model_weight/cache"

        model = OmniRegGPTModel(
            model_name=model_name,
            model_path=MODEL_CKPT,
            tokenizer_path=TOKENIZER_DIR,
            omnireg_repo_path=OMNIREG_REPO,
            hf_home=HF_HOME,
            device=None,
            max_length=16384,
        )

    if dataset_name == "cds-short":
        dataset = GenDataset(jsonl_path = "/mnt/shared-storage-user/dnacoding/yedongxin/DNAFM/GeneShield/data/gen_data/cds_gen/short_sequences.jsonl")
    elif dataset_name == "cds-medium":
        dataset = GenDataset(jsonl_path = "/mnt/shared-storage-user/dnacoding/yedongxin/DNAFM/GeneShield/data/gen_data/cds_gen/medium_sequences.jsonl")
    elif dataset_name == "cds-long":
        dataset = GenDataset(jsonl_path = "/mnt/shared-storage-user/dnacoding/yedongxin/DNAFM/GeneShield/data/gen_data/cds_gen/long_sequences.jsonl")

    # 生成序列并评估
    # 直接使用 GenDataset，无需适配器（GenEvaluator 已支持 (idx, sequence, taxid) 格式）
    output_dir = os.path.join(
        results_root, f"{dataset_name}/{model_name}/")
    
    gen_evaluator = GenEvaluator(
        model=model,
        dataset=dataset,              # 直接传入 GenDataset，无需 train/val/test 分割
        output_dir=output_dir,
        prompt_len=prompt_len,         # -1 表示使用序列长度的一半作为 prompt，>=0 表示固定长度
        batch_size=1,                 # 可以根据模型大小调整
        temperature=temperature,
        top_k=top_k,
        save_per_sample=True,
        save_summary=True,
    )
    
    print(f"[INFO] Starting generation evaluation...")
    print(f"[INFO] Dataset size: {len(dataset)}")
    print(f"[INFO] Output directory: {output_dir}")
    prompt_len_desc = "half of sequence length" if prompt_len == -1 else f"{prompt_len} bases"
    print(f"[INFO] Generation parameters: prompt_len={prompt_len} ({prompt_len_desc}), temperature={temperature}, top_k={top_k}")
    gen_summary = gen_evaluator.run()
    print(f"[OK] Generation evaluation completed. Summary saved to: {gen_summary.get('output_dir')}")
    
    # 打印统计信息
    if "all_statistics" in gen_summary:
        stats = gen_summary["all_statistics"]
        print(f"[INFO] Generation Evaluation Statistics:")
        
        # 打印 exact_match_acc 统计
        if "exact_match_acc" in stats:
            acc_stats = stats["exact_match_acc"]
            mean_val = acc_stats.get('mean')
            median_val = acc_stats.get('median')
            std_val = acc_stats.get('std')
            print(f"  Exact Match Accuracy:")
            print(f"    Mean: {mean_val:.6f}" if mean_val is not None and not math.isnan(mean_val) else f"    Mean: N/A")
            print(f"    Median: {median_val:.6f}" if median_val is not None and not math.isnan(median_val) else f"    Median: N/A")
            print(f"    Std: {std_val:.6f}" if std_val is not None and not math.isnan(std_val) else f"    Std: N/A")
        
        # 打印 edit_distance 统计
        if "edit_distance" in stats:
            ed_stats = stats["edit_distance"]
            mean_val = ed_stats.get('mean')
            median_val = ed_stats.get('median')
            std_val = ed_stats.get('std')
            print(f"  Edit Distance (normalized):")
            print(f"    Mean: {mean_val:.6f}" if mean_val is not None and not math.isnan(mean_val) else f"    Mean: N/A")
            print(f"    Median: {median_val:.6f}" if median_val is not None and not math.isnan(median_val) else f"    Median: N/A")
            print(f"    Std: {std_val:.6f}" if std_val is not None and not math.isnan(std_val) else f"    Std: N/A")
        
        # 打印 alignment_identity 统计
        if "alignment_identity" in stats:
            align_stats = stats["alignment_identity"]
            mean_val = align_stats.get('mean')
            median_val = align_stats.get('median')
            std_val = align_stats.get('std')
            print(f"  Alignment Identity:")
            print(f"    Mean: {mean_val:.6f}" if mean_val is not None and not math.isnan(mean_val) else f"    Mean: N/A")
            print(f"    Median: {median_val:.6f}" if median_val is not None and not math.isnan(median_val) else f"    Median: N/A")
            print(f"    Std: {std_val:.6f}" if std_val is not None and not math.isnan(std_val) else f"    Std: N/A")
        
        print(f"  Valid samples: {stats.get('valid_count', 0)} / {stats.get('count', 0)}")

"""
python script/run_all_gen.py --model_name evo2_1b_base --dataset_name cds-short
python script/run_all_gen.py --model_name evo-1-8k-base --dataset_name cds-short
# python script/run_all_gen.py --model_name evo-1.5-8k-base --dataset_name cds-short
python script/run_all_gen.py --model_name hyenadna-tiny-16k --dataset_name cds-short
python script/run_all_gen.py --model_name Genos-1.2B --dataset_name cds-short

"""

def main():
    parser = argparse.ArgumentParser(
        description="接收两个参数：model_name 和 dataset_name"
    )
    parser.add_argument("--model_name", required=True,
                        help="模型名称，例如 qwen3-14b")
    parser.add_argument("--dataset_name", required=True,
                        help="数据集名称，例如 ceval；传 all 将依次评估所有已支持的数据集")
    parser.add_argument("--results_root_name", required=False,
                        help="结果保存目录名称，默认为 results",
                        default="/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/results/Generate")
    parser.add_argument("--prompt_len", type=int, default=129,
                        help="Prompt 长度，-1 表示使用序列长度的一半，>=0 表示固定长度，默认 -1")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="生成温度参数，控制生成的随机性，默认 1.0")
    parser.add_argument("--top_k", type=int, default=4,
                        help="Top-k 采样参数，限制采样候选数量，默认 4")
    args = parser.parse_args()

    # 确定结果保存目录
    results_root_name = args.results_root_name

    run(
        args.model_name,
        args.dataset_name,
        results_root_name,
        prompt_len=args.prompt_len,
        temperature=args.temperature,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()

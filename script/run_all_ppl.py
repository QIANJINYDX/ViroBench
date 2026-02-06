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
from evaluators.bpb import BPBEvaluator


def run(
    model_name: str,
    dataset_name: str,
    results_root_name: str = "results",
    model_dir: str | None = None,
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
        MODEL_DIR = f"/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/{model_name}"
        CFG_PATH = f"/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/{model_name}/{model_name}_inference.yml"
        HF_HOME = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/cache"
        model = Evo1Model(model_name, MODEL_DIR,
                          CFG_PATH, HF_HOME, device=None)
        
    elif model_name == "evo2_1b_base" or model_name == "evo2_7b_base" or model_name == "evo2_40b_base" or model_name == "evo2_40b" or model_name == "evo2_7b":
        from models import Evo2Model
        MODEL_DIR = f"/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/{model_name}/{model_name}.pt"
        model = Evo2Model(model_name, MODEL_DIR)
    elif model_name == "hyenadna" or model_name == "hyenadna-tiny-16k" or model_name == "hyenadna-tiny-1k" or model_name == "hyenadna-small-32k" or model_name == "hyenadna-medium-160k" or model_name == "hyenadna-medium-450k" or model_name == "hyenadna-large-1m":
        from models import HyenaDNAModel
        HF_HOME = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/cache"
        os.environ["HF_HOME"] = HF_HOME

        if model_name == "hyenadna-tiny-16k":
            MODEL_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/hyenadna-tiny-16k-seqlen-d128-hf"
        elif model_name == "hyenadna-tiny-1k":
            MODEL_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/hyenadna-tiny-1k-seqlen-hf"
        elif model_name == "hyenadna-small-32k":
            MODEL_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/hyenadna-small-32k-seqlen-hf"
        elif model_name == "hyenadna-medium-160k":
            MODEL_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/hyenadna-medium-160k-seqlen-hf"
        elif model_name == "hyenadna-medium-450k":
            MODEL_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/hyenadna-medium-450k-seqlen-hf"
        elif model_name == "hyenadna-large-1m":
            MODEL_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/hyenadna-large-1m-seqlen-hf"

        # 从路径中提取模型名称（去掉路径前缀和可能的后缀）
        # 例如: /inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/hyenadna-large-1m-seqlen-hf -> hyenadna-large-1m
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
        HF_HOME = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/cache"
        os.environ["HF_HOME"] = HF_HOME

        # 设置模型路径
        if model_name == "Genos-1.2B":
            MODEL_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/Genos-1.2B"
            default_hidden_size = 1024  # Genos-1.2B 的默认 hidden_size
            batch_size = 16
        elif model_name == "Genos-10B":
            MODEL_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/Genos-10B"
            default_hidden_size = 2048  # Genos-10B 的默认 hidden_size（需要根据实际模型调整）
            batch_size = 8  # 10B 模型更大，使用较小的 batch_size
        elif model_name == "Genos-10B-v2":
            MODEL_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/Genos-10B-v2/"
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
            MODEL_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/GenomeOcean-100M"
        elif model_name == "GenomeOcean-500M":
            MODEL_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/GenomeOcean-500M"
        elif model_name == "GenomeOcean-4B":
            MODEL_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/GenomeOcean-4B"
        model = GenomeOceanModel(
            model_name=model_name,
            model_path=MODEL_DIR)

    elif model_name == "GENERator-v2-eukaryote-1.2b-base" or model_name == "GENERator-v2-eukaryote-3b-base" or model_name == "GENERator-v2-prokaryote-1.2b-base" or model_name == "GENERator-v2-prokaryote-3b-base":
        from models.GENERator_gen import GENERatorModel
        if model_name == "GENERator-v2-eukaryote-1.2b-base":
            MODEL_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/GENERator-v2-eukaryote-1.2b-base"
        elif model_name == "GENERator-v2-eukaryote-3b-base":
            MODEL_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/GENERator-v2-eukaryote-3b-base"
        elif model_name == "GENERator-v2-prokaryote-1.2b-base":
            MODEL_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/GENERator-v2-prokaryote-1.2b-base"
        elif model_name == "GENERator-v2-prokaryote-3b-base":
            MODEL_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/GENERator-v2-prokaryote-3b-base"
        model = GENERatorModel(
            model_name=model_name,
            model_path=MODEL_DIR)
    elif model_name == "OmniReg-bigbird" or model_name =="OmniReg-base" or model_name == "OmniReg-large":
        from models.omnireg_model import OmniRegGPTModel

        if model_name == "OmniReg-bigbird":
            MODEL_CKPT = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/gena-lm-bigbird-base-t2t/pytorch_model.bin"
            TOKENIZER_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/omnireg_bigbird"
        elif model_name == "OmniReg-base":
            MODEL_CKPT = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/gena-lm-bert-base-t2t/pytorch_model.bin"
            TOKENIZER_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/gena-lm-bert-base-t2t"
        elif model_name == "OmniReg-large":
            MODEL_CKPT = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/gena-lm-bert-large-t2t/pytorch_model.bin"
            TOKENIZER_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/gena-lm-bert-large-t2t"
        TOKENIZER_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/gena-lm-bert-large-t2t"
        OMNIREG_REPO = os.path.join(PROJECT_ROOT, "models", "OmniReg-GPT")
        HF_HOME = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/cache"

        model = OmniRegGPTModel(
            model_name=model_name,
            model_path=MODEL_CKPT,
            tokenizer_path=TOKENIZER_DIR,
            omnireg_repo_path=OMNIREG_REPO,
            hf_home=HF_HOME,
            device=None,
            max_length=16384,
        )
    elif "hyena_local" in model_name:
        from models.hyenadna_local import HyenaDNALocal
        if "hyena_local" == model_name:
            MODEL_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/pretrain/hyena-dna/hyena_hg38_hf"
        elif model_name == "hyena_local-12M-mini-virus":
            MODEL_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/pretrain/hyena-dna/hyena_local-12M-mini-virus"
        elif model_name == "hyena_local-12M-virus":
            MODEL_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/pretrain/hyena-dna/hyena_local-12M-virus"
        elif model_name == "hyena_local-test":
            MODEL_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/pretrain/hyena-dna/hyena_local-test"
        elif model_name == "hyena_local-436k-virus":
            MODEL_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/pretrain/hyena-dna/hyena_local-436k-virus"
        elif model_name == "hyena_local-3.2M-virus":
            MODEL_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/pretrain/hyena-dna/hyena_local-3.2M-virus"
        elif model_name == "hyena_local-253M":
            MODEL_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/pretrain/hyena-dna/hyena_local-253M"
        else:
            if model_dir is None:
                raise ValueError(
                    f"未预定义模型 {model_name} 的路径，请通过 --model_dir 指定模型目录"
                )
            MODEL_DIR = model_dir

        model = HyenaDNALocal(
            model_dir=MODEL_DIR,
            device="cuda",
            pretrain_root="/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/pretrain/hyena-dna",
        )

    if dataset_name == "genome-short":
        dataset = GenDataset(jsonl_path = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/all_viral/gen_data/genome_gen/short_sequences.jsonl")
    elif dataset_name == "genome-medium":
        dataset = GenDataset(jsonl_path = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/all_viral/gen_data/genome_gen/medium_sequences.jsonl")
    elif dataset_name == "genome-long":
        dataset = GenDataset(jsonl_path = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/all_viral/gen_data/genome_gen/long_sequences.jsonl")

    

    # 计算BPB
    # 直接使用 GenDataset，无需适配器（BPBEvaluator 已支持 (idx, sequence) 格式）
    output_dir = os.path.join(
        results_root, f"{dataset_name}/{model_name}/")
    
    bpb_evaluator = BPBEvaluator(
        model=model,
        dataset=dataset,              # 直接传入 GenDataset，无需 train/val/test 分割
        output_dir=output_dir,
        batch_size=1,               # 可以根据模型大小调整
        save_per_sample=True,
        save_summary=True,
    )
    
    print(f"[INFO] Starting BPB computation...")
    print(f"[INFO] Dataset size: {len(dataset)}")
    print(f"[INFO] Output directory: {output_dir}")
    bpb_summary = bpb_evaluator.run()
    print(f"[OK] BPB computation completed. Summary saved to: {bpb_summary.get('output_dir')}")
    
    # 打印统计信息
    if "all_statistics" in bpb_summary:
        stats = bpb_summary["all_statistics"]
        print(f"[INFO] BPB Statistics:")
        mean_val = stats.get('mean')
        median_val = stats.get('median')
        std_val = stats.get('std')
        min_val = stats.get('min')
        max_val = stats.get('max')
        print(f"  Mean: {mean_val:.6f}" if mean_val is not None and not math.isnan(mean_val) else f"  Mean: N/A")
        print(f"  Median: {median_val:.6f}" if median_val is not None and not math.isnan(median_val) else f"  Median: N/A")
        print(f"  Std: {std_val:.6f}" if std_val is not None and not math.isnan(std_val) else f"  Std: N/A")
        print(f"  Min: {min_val:.6f}" if min_val is not None and not math.isnan(min_val) else f"  Min: N/A")
        print(f"  Max: {max_val:.6f}" if max_val is not None and not math.isnan(max_val) else f"  Max: N/A")
        print(f"  Valid samples: {stats.get('valid_count', 0)} / {stats.get('count', 0)}")

"""
python script/run_all_ppl.py --model_name hyena_local-253M --dataset_name genome-short
python script/run_all_ppl.py --model_name evo2_1b_base --dataset_name genome-short
python script/run_all_ppl.py --model_name evo-1-8k-base --dataset_name genome-short
python script/run_all_ppl.py --model_name evo-1.5-8k-base --dataset_name genome-short
python script/run_all_ppl.py --model_name hyenadna-tiny-16k --dataset_name genome-short
python script/run_all_ppl.py --model_name Genos-1.2B --dataset_name genome-long
python script/run_all_ppl.py --model_name OmniReg-bigbird --dataset_name genome-short
python script/run_all_ppl.py --model_name GENERator-v2-eukaryote-1.2b-base --dataset_name genome-short

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
                        default="/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/results/Bpb")
    parser.add_argument("--model_dir", required=False, default=None,
                        help="模型目录路径，用于 hyena_local 等未在脚本中预定义路径的模型")
    args = parser.parse_args()

    # 确定结果保存目录
    results_root_name = args.results_root_name

    run(
        args.model_name,
        args.dataset_name,
        results_root_name,
        model_dir=args.model_dir,
    )


if __name__ == "__main__":
    main()

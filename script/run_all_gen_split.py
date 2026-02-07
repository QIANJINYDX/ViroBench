#!/usr/bin/env python3
import argparse
import sys
import os
import random
import re
import traceback
import math
from typing import Dict, Optional
import torch
import pandas as pd
import time
from datetime import datetime
import multiprocessing
import json

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

MODEL_WEIGHT = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight"
DATASET_PATH = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/all_viral/gen_data"


def _build_dataset(dataset_name: str, split_index: Optional[int]) -> GenDataset:
    """根据 dataset_name 和 split_index 构建 GenDataset（不加载模型）。"""
    if dataset_name == "cds-short":
        jsonl_path = f"{DATASET_PATH}/split_gen/short_sequences-{split_index}.jsonl" if split_index is not None else f"{DATASET_PATH}/cds_gen/short_sequences.jsonl"
        return GenDataset(jsonl_path=jsonl_path)
    elif dataset_name == "cds-medium":
        jsonl_path = f"{DATASET_PATH}/split_gen/medium_sequences-{split_index}.jsonl" if split_index is not None else f"{DATASET_PATH}/cds_gen/medium_sequences.jsonl"
        return GenDataset(jsonl_path=jsonl_path)
    elif dataset_name == "cds-long":
        jsonl_path = f"{DATASET_PATH}/split_gen/long_sequences-{split_index}.jsonl" if split_index is not None else f"{DATASET_PATH}/cds_gen/long_sequences.jsonl"
        return GenDataset(jsonl_path=jsonl_path)
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")


def run(
    model_name: str,
    dataset_name: str,
    results_root_name: str = "results",
    prompt_len: int = -1,
    temperature: float = 1.0,
    top_k: int = 4,
    split_index: int = None,
    model_dir: str = None,
    gen_batch_size: int = 4,
    num_workers: int = 1,
    worker_id: Optional[int] = None,
) -> None:
    print(f"[INFO] model_name = {model_name}")
    print(f"[INFO] dataset_name = {dataset_name}")
    if os.path.isabs(results_root_name):
        results_root = results_root_name
    else:
        results_root = os.path.join(PROJECT_ROOT, results_root_name)
    os.makedirs(results_root, exist_ok=True)
    print(f"[INFO] results_root = {results_root}")

    # 先构建 dataset，便于多 worker 时主进程只做 spawn+merge、子进程只处理子集
    dataset = _build_dataset(dataset_name, split_index)
    n_total = len(dataset)

    # 多 worker 主进程：只 spawn 子进程并合并结果，不加载模型
    if num_workers > 1 and worker_id is None:
        output_dir = os.path.join(
            results_root, f"{dataset_name}/{model_name}/{split_index if split_index is not None else 'all'}/")
        os.makedirs(output_dir, exist_ok=True)

        def _worker_fn(wid: int) -> None:
            env = os.environ.copy()
            # 多卡时每个进程用一张卡（按 worker_id 轮询）
            if torch.cuda.is_available() and torch.cuda.device_count() >= num_workers:
                env["CUDA_VISIBLE_DEVICES"] = str(wid % torch.cuda.device_count())
            run(
                model_name=model_name,
                dataset_name=dataset_name,
                results_root_name=results_root_name,
                prompt_len=prompt_len,
                temperature=temperature,
                top_k=top_k,
                split_index=split_index,
                model_dir=model_dir,
                gen_batch_size=gen_batch_size,
                num_workers=num_workers,
                worker_id=wid,
            )

        procs = [multiprocessing.Process(target=_worker_fn, args=(i,)) for i in range(num_workers)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()

        # 合并各 worker 的 jsonl，按 sequence_index 排序，写 all_gen_per_sample.jsonl 并生成 summary
        all_details = []
        for i in range(num_workers):
            p = os.path.join(output_dir, f"all_gen_per_sample_worker{i}.jsonl")
            if not os.path.isfile(p):
                continue
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_details.append(json.loads(line))
        all_details.sort(key=lambda d: d.get("sequence_index", 0))
        per_sample_path = os.path.join(output_dir, "all_gen_per_sample.jsonl")
        with open(per_sample_path, "w", encoding="utf-8") as f:
            for d in all_details:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"[INFO] Merged {len(all_details)} samples to {per_sample_path}")

        # 用 GenEvaluator 的统计逻辑计算合并后的 summary
        dummy_eval = GenEvaluator(model=None, dataset=[], output_dir=output_dir, batch_size=1)
        stats = dummy_eval._compute_statistics(all_details)
        summary = {
            "output_dir": output_dir,
            "prompt_len": prompt_len,
            "temperature": temperature,
            "top_k": top_k,
            "enable_kmer_spectrum": getattr(dummy_eval, "enable_kmer_spectrum", True),
            "kmer_k": getattr(dummy_eval, "kmer_k", -1),
            "kmer_k_min": getattr(dummy_eval, "kmer_k_min", 1),
            "kmer_k_max": getattr(dummy_eval, "kmer_k_max", 13),
            "all_statistics": stats,
            "all_size": len(all_details),
        }
        with open(os.path.join(output_dir, "gen_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[OK] Generation evaluation (multi-worker) completed. Summary saved.")
        stats = summary.get("all_statistics", {})
        if stats:
            print(f"[INFO] Generation Evaluation Statistics:")
            for metric in ("exact_match_acc", "edit_distance", "alignment_identity"):
                if metric in stats:
                    s = stats[metric]
                    for k, v in (("Mean", s.get("mean")), ("Median", s.get("median")), ("Std", s.get("std"))):
                        if v is not None and not math.isnan(v):
                            print(f"  {metric} {k}: {v:.6f}")
            print(f"  Valid samples: {stats.get('valid_count', 0)} / {stats.get('count', 0)}")
        return

    # 多 worker 子进程或单进程：只处理本 worker 的 indices
    if num_workers > 1 and worker_id is not None:
        indices = list(range(worker_id, n_total, num_workers))
        if not indices:
            print(f"[INFO] Worker {worker_id}: no indices, skip.")
            return
        print(f"[INFO] Worker {worker_id}: processing {len(indices)} / {n_total} samples.")
    else:
        indices = None

    if model_name == "evo-1-8k-base" or model_name == "evo-1-131k-base" or model_name == "evo-1.5-8k-base":
        from models import Evo1Model
        MODEL_DIR = f"{MODEL_WEIGHT}/{model_name}"
        CFG_PATH = f"{MODEL_WEIGHT}/{model_name}/{model_name}_inference.yml"
        HF_HOME = f"{MODEL_WEIGHT}/cache"
        model = Evo1Model(model_name, MODEL_DIR,
                          CFG_PATH, HF_HOME, device=None)
    elif model_name == "evo2_1b_base" or model_name == "evo2_7b_base" or model_name == "evo2_40b_base" or model_name == "evo2_40b" or model_name == "evo2_7b":
        from models import Evo2Model
        MODEL_DIR = f"{MODEL_WEIGHT}/{model_name}/{model_name}.pt"
        model = Evo2Model(model_name, MODEL_DIR)
    elif model_name == "hyenadna" or model_name == "hyenadna-tiny-16k" or model_name == "hyenadna-tiny-1k" or model_name == "hyenadna-small-32k" or model_name == "hyenadna-medium-160k" or model_name == "hyenadna-medium-450k" or model_name == "hyenadna-large-1m":
        from models import HyenaDNAModel
        HF_HOME = f"{MODEL_WEIGHT}/cache"
        os.environ["HF_HOME"] = HF_HOME

        if model_name == "hyenadna-tiny-16k":
            MODEL_DIR = f"{MODEL_WEIGHT}/hyenadna-tiny-16k-seqlen-d128-hf"
        elif model_name == "hyenadna-tiny-1k":
            MODEL_DIR = f"{MODEL_WEIGHT}/hyenadna-tiny-1k-seqlen-hf"
        elif model_name == "hyenadna-small-32k":
            MODEL_DIR = f"{MODEL_WEIGHT}/hyenadna-small-32k-seqlen-hf"
        elif model_name == "hyenadna-medium-160k":
            MODEL_DIR = f"{MODEL_WEIGHT}/hyenadna-medium-160k-seqlen-hf"
        elif model_name == "hyenadna-medium-450k":
            MODEL_DIR = f"{MODEL_WEIGHT}/hyenadna-medium-450k-seqlen-hf"
        elif model_name == "hyenadna-large-1m":
            MODEL_DIR = f"{MODEL_WEIGHT}/hyenadna-large-1m-seqlen-hf"

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
    elif "hyena_local" in model_name:
        MODEL_DIR = None
        from models.hyenadna_local import HyenaDNALocal
        if model_name == "hyena_local-12M-mini-virus":
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
        if MODEL_DIR is None:
            if model_dir is None:
                raise ValueError(
                    "model_name contains 'hyena_local' but no built-in path is set. "
                    "Please pass --model_dir <path_to_model>."
                )
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

        model = HyenaDNALocal(
            model_dir=MODEL_DIR,
            device="cuda",
            pretrain_root="/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/pretrain/hyena-dna",
        )
    
    elif model_name == "Genos-1.2B" or model_name == "Genos-10B" or model_name == "Genos-10B-v2":
        from models.genos_model_gen import GenosModel
        HF_HOME = f"{MODEL_WEIGHT}/cache"
        os.environ["HF_HOME"] = HF_HOME

        # 设置模型路径
        if model_name == "Genos-1.2B":
            MODEL_DIR = f"{MODEL_WEIGHT}/Genos-1.2B"
            default_hidden_size = 1024  # Genos-1.2B 的默认 hidden_size
            batch_size = 16
        elif model_name == "Genos-10B":
            MODEL_DIR = f"{MODEL_WEIGHT}/Genos-10B"
            default_hidden_size = 2048  # Genos-10B 的默认 hidden_size（需要根据实际模型调整）
            batch_size = 8  # 10B 模型更大，使用较小的 batch_size
        elif model_name == "Genos-10B-v2":
            MODEL_DIR = f"{MODEL_WEIGHT}/Genos-10B-v2/Genos-10B-v2/"
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
            MODEL_DIR = f"{MODEL_WEIGHT}/GenomeOcean-100M"
        elif model_name == "GenomeOcean-500M":
            MODEL_DIR = f"{MODEL_WEIGHT}/GenomeOcean-500M"
        elif model_name == "GenomeOcean-4B":
            MODEL_DIR = f"{MODEL_WEIGHT}/GenomeOcean-4B"
        model = GenomeOceanModel(
            model_name=model_name,
            model_path=MODEL_DIR)

    elif model_name == "GENERator-v2-eukaryote-1.2b-base" or model_name == "GENERator-v2-eukaryote-3b-base" or model_name == "GENERator-v2-prokaryote-1.2b-base" or model_name == "GENERator-v2-prokaryote-3b-base":
        from models.GENERator_gen import GENERatorModel
        if model_name == "GENERator-v2-eukaryote-1.2b-base":
            MODEL_DIR = f"{MODEL_WEIGHT}/GENERator-v2-eukaryote-1.2b-base"
        elif model_name == "GENERator-v2-eukaryote-3b-base":
            MODEL_DIR = f"{MODEL_WEIGHT}/GENERator-v2-eukaryote-3b-base"
        elif model_name == "GENERator-v2-prokaryote-1.2b-base":
            MODEL_DIR = f"{MODEL_WEIGHT}/GENERator-v2-prokaryote-1.2b-base"
        elif model_name == "GENERator-v2-prokaryote-3b-base":
            MODEL_DIR = f"{MODEL_WEIGHT}/GENERator-v2-prokaryote-3b-base"
        model = GENERatorModel(
            model_name=model_name,
            model_path=MODEL_DIR)
    elif model_name == "OmniReg-bigbird" or model_name =="OmniReg-base" or model_name == "OmniReg-large":
        from models.omnireg_model import OmniRegGPTModel

        if model_name == "OmniReg-bigbird":
            MODEL_CKPT = f"{MODEL_WEIGHT}/gena-lm-bigbird-base-t2t/pytorch_model.bin"
            TOKENIZER_DIR = f"{MODEL_WEIGHT}/omnireg_bigbird"
        elif model_name == "OmniReg-base":
            MODEL_CKPT = f"{MODEL_WEIGHT}/gena-lm-bert-base-t2t/pytorch_model.bin"
            TOKENIZER_DIR = f"{MODEL_WEIGHT}/gena-lm-bert-base-t2t"
        elif model_name == "OmniReg-large":
            MODEL_CKPT = f"{MODEL_WEIGHT}/gena-lm-bert-large-t2t/pytorch_model.bin"
            TOKENIZER_DIR = f"{MODEL_WEIGHT}/gena-lm-bert-large-t2t"
        TOKENIZER_DIR = f"{MODEL_WEIGHT}/gena-lm-bert-large-t2t"
        OMNIREG_REPO = os.path.join(PROJECT_ROOT, "models", "OmniReg-GPT")
        HF_HOME = f"{MODEL_WEIGHT}/cache"

        model = OmniRegGPTModel(
            model_name=model_name,
            model_path=MODEL_CKPT,
            tokenizer_path=TOKENIZER_DIR,
            omnireg_repo_path=OMNIREG_REPO,
            hf_home=HF_HOME,
            device=None,
            max_length=16384,
        )
    elif model_name == "ViroHyena-436k" or model_name == "ViroHyena-1m" or model_name == "ViroHyena-6m" or model_name == "ViroHyena-253m":
        MODEL_DIR = None
        from models.hyenadna_local import HyenaDNALocal
        if model_name == "ViroHyena-436k":
            MODEL_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/pretrain/hyena-dna/ViroHyena-436k"
        elif model_name == "ViroHyena-1m":
            MODEL_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/pretrain/hyena-dna/ViroHyena-1m"
        elif model_name == "ViroHyena-6m":
            MODEL_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/pretrain/hyena-dna/ViroHyena-6m"
        elif model_name == "ViroHyena-253m":
            MODEL_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/pretrain/hyena-dna/ViroHyena-253m"
        
        print("---------------------------------------当前模型---------------------------------------")
        print(model_name)
        print("模型路径：⬆️⬇️⬆️⬇️",MODEL_DIR)
        print("---------------------------------------当前模型---------------------------------------")
        model = HyenaDNALocal(
            model_dir=MODEL_DIR,
            device="cuda",
            pretrain_root="/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/pretrain/hyena-dna",
        )

    # dataset 已在 run() 开头构建

    # 生成序列并评估
    # 直接使用 GenDataset，无需适配器（GenEvaluator 已支持 (idx, sequence, taxid) 格式）
    output_dir = os.path.join(
        results_root, f"{dataset_name}/{model_name}/{split_index if split_index is not None else 'all'}/")
    
    gen_evaluator = GenEvaluator(
        model=model,
        dataset=dataset,
        output_dir=output_dir,
        prompt_len=prompt_len,
        batch_size=gen_batch_size,
        temperature=temperature,
        top_k=top_k,
        save_per_sample=True,
        save_summary=True,
        indices=indices,
        worker_id=worker_id,
    )
    
    print(f"[INFO] Starting generation evaluation...")
    print(f"[INFO] Dataset size: {len(dataset)}")
    print(f"[INFO] Output directory: {output_dir}")
    prompt_len_desc = "half of sequence length" if prompt_len == -1 else f"{prompt_len} bases"
    print(f"[INFO] Generation parameters: prompt_len={prompt_len} ({prompt_len_desc}), temperature={temperature}, top_k={top_k}, gen_batch_size={gen_batch_size}")
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
python script/run_all_gen_split.py --model_name OmniReg-bigbird --dataset_name cds-medium --split_index 1
python script/run_all_gen_split.py --model_name evo-1-8k-base --dataset_name cds-short --split_index 1
# python script/run_all_gen_split.py --model_name evo-1.5-8k-base --dataset_name cds-short --split_index 1
python script/run_all_gen_split.py --model_name hyenadna-tiny-16k --dataset_name cds-short --split_index 1
python script/run_all_gen_split.py --model_name Genos-1.2B --dataset_name cds-short --split_index 1
python script/run_all_gen_split.py --model_name ViroHyena-436k --dataset_name cds-long --split_index 1
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
                        default="results/Generate")
    parser.add_argument("--prompt_len", type=int, default=129,
                        help="Prompt 长度，-1 表示使用序列长度的一半，>=0 表示固定长度，默认 -1")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="生成温度参数，控制生成的随机性，默认 1.0")
    parser.add_argument("--top_k", type=int, default=4,
                        help="Top-k 采样参数，限制采样候选数量，默认 4")
    parser.add_argument("--split_index", type=int, default=None,
                        help="切分文件的索引（1-10），如果指定则使用 split_gen 目录下对应的文件，默认 None 使用原始文件")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="hyena_local 类模型的自定义路径，当 model_name 不在内置列表时必传")
    parser.add_argument("--gen_batch_size", type=int, default=1,
                        help="生成时每批处理的序列数；>1 时多条序列一次前向，提高 GPU 利用率；长序列若显存不足可设为 1，默认 4")
    parser.add_argument("--num_workers", type=int, default=32,
                        help="并行 worker 数（多进程）；>1 时按样本划分到各 worker，多卡时自动按 worker_id 分配 GPU，默认 1")
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
        split_index=args.split_index,
        model_dir=args.model_dir,
        gen_batch_size=args.gen_batch_size,
        num_workers=args.num_workers,
        worker_id=None,
    )


if __name__ == "__main__":
    main()

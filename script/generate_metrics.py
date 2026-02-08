# -*- coding: utf-8 -*-
"""
对生成结果 JSONL 计算评估指标：
  - edit_distance（归一化编辑距离）
  - exact_match_acc（字符级准确率，长度一致时）
  - kmer_JSD（k-mer 频谱 JSD）
  - kmer_KS（k-mer 频谱 KS 统计量）
  - is_CDS（生成序列是否为合法 CDS：仅末尾为终止密码子，中间无提前终止）

依赖：与 evaluators/gen.py 相同（numpy, Levenshtein, tqdm 等），在项目根目录运行。
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

# 复用 evaluators/gen.py 中的指标计算
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluators.gen import (
    evaluate_similarity,
    kmer_spectrum_metrics,
)

# 标准终止密码子（DNA）
_STOP_CODONS = frozenset({"TAA", "TAG", "TGA"})


def is_valid_cds(seq: str) -> bool:
    """
    判断序列是否为合法 CDS（编码区）：
    - 长度为 3 的倍数
    - 末尾恰好为终止密码子（TAA / TAG / TGA）
    - 中间（除最后一个密码子外）不出现终止密码子
    """
    if not seq or len(seq) < 3:
        return False
    seq_upper = seq.upper()
    if len(seq_upper) % 3 != 0:
        return False
    # 最后一个密码子必须是终止密码子
    last_codon = seq_upper[-3:]
    if last_codon not in _STOP_CODONS:
        return False
    # 中间不能出现终止密码子
    for i in range(0, len(seq_upper) - 3, 3):
        codon = seq_upper[i : i + 3]
        if codon in _STOP_CODONS:
            return False
    return True


def _get_generated_continuation(row: Dict[str, Any]) -> Optional[str]:
    """
    从一行中解析出“生成续写部分”（与 ground_truth 等长的 continuation）。
    兼容两种 JSONL 格式：
    - generated_sequence（可为续写或 prompt+续写，会去掉 prompt）
    - generated_continuation / generated_tail（已是续写）或 generated_full（会去掉 prompt）
    """
    prompt = row.get("prompt") if isinstance(row.get("prompt"), str) else None

    # 优先使用 generated_sequence（部分 pipeline 导出时使用）
    val = row.get("generated_sequence")
    if val is not None and isinstance(val, str):
        if prompt and val.startswith(prompt):
            return val[len(prompt) :]
        return val

    # 否则使用 run_all_gen_split / run_all_gen 产出的字段
    for key in ("generated_continuation", "generated_tail"):
        val = row.get(key)
        if val is not None and isinstance(val, str):
            return val
    val = row.get("generated_full")
    if val is not None and isinstance(val, str) and prompt and val.startswith(prompt):
        return val[len(prompt) :]
    if val is not None and isinstance(val, str):
        return val
    return None


def get_ground_truth_and_generated(row: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    从一行结果中解析出 ground_truth 与用于比较的生成序列（与 gt 等长或截断到 gt 长度）。
    支持 generated_sequence 或 generated_continuation / generated_tail / generated_full。
    """
    gt = row.get("ground_truth")
    if gt is None or not isinstance(gt, str):
        return None, None

    gen_cont = _get_generated_continuation(row)
    if gen_cont is None:
        return gt, None
    gen_compare = gen_cont[: len(gt)]
    return gt, gen_compare


def get_generated_sequence_for_cds(row: Dict[str, Any]) -> Optional[str]:
    """
    获取用于 is_CDS 判断的“生成续写序列”（continuation 部分）。
    支持 generated_sequence 或 generated_continuation / generated_tail / generated_full。
    """
    return _get_generated_continuation(row)


def compute_metrics_for_sample(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    对单条样本计算 5 个指标，返回包含原始字段 + 5 个指标的结果字典。
    """
    out = dict(row)
    gt, gen_compare = get_ground_truth_and_generated(row)
    gen_for_cds = get_generated_sequence_for_cds(row)

    if gt is None:
        out["edit_distance"] = float("nan")
        out["exact_match_acc"] = float("nan")
        out["kmer_JSD"] = float("nan")
        out["kmer_KS"] = float("nan")
        out["is_CDS"] = False
        return out

    if gen_compare is None:
        out["edit_distance"] = float("nan")
        out["exact_match_acc"] = float("nan")
        out["kmer_JSD"] = float("nan")
        out["kmer_KS"] = float("nan")
        out["is_CDS"] = is_valid_cds(gen_for_cds) if gen_for_cds else False
        return out

    # 空序列会导致 Bio.Align / kmer 报错，直接赋默认值
    if len(gt) == 0 or len(gen_compare) == 0:
        out["edit_distance"] = float("nan")
        out["exact_match_acc"] = 1.0 if (len(gt) == 0 and len(gen_compare) == 0) else 0.0
        out["kmer_JSD"] = float("nan")
        out["kmer_KS"] = float("nan")
        out["is_CDS"] = is_valid_cds(gen_for_cds) if gen_for_cds else False
        return out

    # 1) edit_distance, exact_match_acc
    sim = evaluate_similarity(gt, gen_compare)
    out["edit_distance"] = sim["edit_distance"]
    out["exact_match_acc"] = sim["exact_match_acc"]

    # 2) kmer_JSD, kmer_KS
    kmer_metrics = kmer_spectrum_metrics(gt, gen_compare, k=None, k_min=1, k_max=13)
    out["kmer_JSD"] = kmer_metrics["kmer_JSD"]
    out["kmer_KS"] = kmer_metrics["kmer_KS"]

    # 3) is_CDS（基于生成续写序列）
    out["is_CDS"] = is_valid_cds(gen_for_cds) if gen_for_cds else False

    return out


def compute_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """对数值指标做均值/中位数/标准差等汇总；is_CDS 做比例."""
    summary: Dict[str, Any] = {"count": len(records)}

    numeric_keys = ["edit_distance", "exact_match_acc", "kmer_JSD", "kmer_KS"]
    for key in numeric_keys:
        values = []
        for r in records:
            v = r.get(key)
            if v is not None and isinstance(v, (int, float)) and math.isfinite(v):
                values.append(float(v))
        if not values:
            summary[key] = {
                "mean": float("nan"),
                "median": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
            }
        else:
            arr = np.asarray(values, dtype=np.float64)
            summary[key] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

    cds_valid = sum(1 for r in records if r.get("is_CDS") is True)
    summary["is_CDS"] = {
        "count": cds_valid,
        "ratio": cds_valid / len(records) if records else 0.0,
    }
    return summary


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def collect_jsonl_paths(input_path: str) -> List[str]:
    """输入为文件则返回 [path]；为目录则返回目录下所有 *gen*per_sample*.jsonl 或 *all_gen*.jsonl."""
    if os.path.isfile(input_path):
        return [input_path]
    if not os.path.isdir(input_path):
        return []
    paths = []
    for root, _dirs, files in os.walk(input_path):
        for f in files:
            if f.endswith(".jsonl") and ("gen" in f and "per_sample" in f or "all_gen" in f):
                paths.append(os.path.join(root, f))
    return sorted(paths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="对生成结果 JSONL 计算 edit_distance, exact_match_acc, kmer_JSD, kmer_KS, is_CDS"
    )
    parser.add_argument(
        "input",
        type=str,
        help="输入 JSONL 文件路径或目录（目录时会递归查找 *gen*per_sample*.jsonl / *all_gen*.jsonl）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="输出 JSONL 路径（默认：在输入同目录下生成 <basename>_metrics.jsonl）",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help="汇总统计 JSON 输出路径（默认：与输出同目录的 <basename>_metrics_summary.json）",
    )
    parser.add_argument(
        "--no-per-sample",
        action="store_true",
        help="不写 per-sample JSONL，只写 summary",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="并行进程数（默认 1；设为 0 表示使用 CPU 核心数）",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=None,
        metavar="N",
        help="每个进程一次处理的样本数（默认自动，多进程时可用以调优）",
    )
    args = parser.parse_args()

    paths = collect_jsonl_paths(args.input)
    if not paths:
        print("未找到任何 JSONL 文件", file=sys.stderr)
        sys.exit(1)

    for input_path in paths:
        print(f"处理: {input_path}")
        records = load_jsonl(input_path)
        if not records:
            print(f"  跳过（无有效行）")
            continue

        n_jobs = args.jobs if args.jobs > 0 else max(1, multiprocessing.cpu_count())
        if n_jobs > 1:
            print(f"  并行进程数: {n_jobs}")
        if n_jobs <= 1:
            results = []
            for row in tqdm(records, desc="指标计算"):
                results.append(compute_metrics_for_sample(row))
        else:
            chunksize = args.chunksize
            if chunksize is None:
                # 默认 chunksize：使每个进程约处理多批，减少通信次数
                chunksize = max(1, len(records) // (n_jobs * 4))
            with multiprocessing.Pool(processes=n_jobs) as pool:
                results = list(
                    tqdm(
                        pool.imap(compute_metrics_for_sample, records, chunksize=chunksize),
                        total=len(records),
                        desc="指标计算",
                    )
                )

        summary = compute_summary(results)
        summary["source_file"] = input_path
        summary["num_samples"] = len(results)

        base = os.path.splitext(os.path.basename(input_path))[0]
        out_dir = os.path.dirname(input_path)
        default_out = os.path.join(out_dir, f"{base}_metrics.jsonl")
        default_summary = os.path.join(out_dir, f"{base}_metrics_summary.json")

        out_path = args.output or default_out
        summary_path = args.summary or default_summary

        # 若同时处理多个文件，output/summary 只对第一个生效，其余用默认
        if input_path != paths[0]:
            out_path = default_out
            summary_path = default_summary

        if not args.no_per_sample:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  已写 per-sample: {out_path}")

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  已写 summary: {summary_path}")
        print(f"  edit_distance mean={summary['edit_distance']['mean']:.4f} "
              f"exact_match_acc mean={summary['exact_match_acc']['mean']:.4f} "
              f"kmer_JSD mean={summary['kmer_JSD']['mean']:.4f} "
              f"kmer_KS mean={summary['kmer_KS']['mean']:.4f} "
              f"is_CDS ratio={summary['is_CDS']['ratio']:.4f}")


if __name__ == "__main__":
    main()

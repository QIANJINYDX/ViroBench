# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import json
import math
import warnings
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
from tqdm.auto import tqdm
import Levenshtein
from Bio import Align


# =========================
# Prompt utilities
# =========================
def calc_prompt_len(seq_len: int, prompt_len_arg: int) -> int:
    """
    prompt_len_arg < 0  -> 使用 seq_len // 2
    prompt_len_arg >= 0 -> 使用固定 prompt_len_arg
    保证最终 prompt_len 在 [1, seq_len-1]，避免 gt 为空
    """
    if seq_len <= 1:
        return 0
    if prompt_len_arg is None or prompt_len_arg < 0:
        pl = seq_len // 2
    else:
        pl = int(prompt_len_arg)

    if pl < 1:
        pl = 1
    if pl >= seq_len:
        pl = seq_len - 1
    return pl


def _get_singleton_aligner() -> Align.PairwiseAligner:
    """避免每个样本都重复创建 aligner（会慢）"""
    if not hasattr(_get_singleton_aligner, "_aligner"):
        aligner = Align.PairwiseAligner()
        aligner.mode = "global"
        aligner.match_score = 1
        aligner.mismatch_score = -1
        aligner.open_gap_score = -1
        aligner.extend_gap_score = -1
        _get_singleton_aligner._aligner = aligner
    return _get_singleton_aligner._aligner


# =========================
# Baseline similarity metrics
# =========================
def evaluate_similarity(ground_truth: str, generated: str) -> Dict[str, float]:
    """
    比较两个 DNA 序列的相似度（ground_truth vs generated）
    """
    metrics: Dict[str, float] = {}

    # 1) 字符级准确率（长度一致时）
    if len(ground_truth) == len(generated) and len(ground_truth) > 0:
        matches = sum(1 for a, b in zip(ground_truth, generated) if a == b)
        metrics["exact_match_acc"] = matches / len(ground_truth)
    else:
        metrics["exact_match_acc"] = 0.0

    # # 2) 编辑距离（归一化）
    if len(ground_truth) > 0:
        edit_dist = Levenshtein.distance(ground_truth, generated)
        metrics["edit_distance"] = edit_dist / len(ground_truth)
    else:
        metrics["edit_distance"] = 0.0

    # 3) 全局比对得分（归一化）
    if len(ground_truth) > 0:
        aligner = _get_singleton_aligner()
        score = aligner.score(ground_truth, generated)
        max_score = len(ground_truth)
        metrics["alignment_identity"] = (score / max_score) if max_score > 0 else 0.0
    else:
        metrics["alignment_identity"] = 0.0

    return metrics


# =========================
# k-mer spectrum metrics (paper-style)
# =========================
_DNA_UPPER = set("ACGT")


def _effective_length_acgt_upper(seq: str) -> int:
    """只统计大写 A/C/G/T（小写当 masked，不计入有效长度）"""
    if not seq:
        return 0
    return sum(1 for ch in seq if ch in _DNA_UPPER)


def choose_k_from_length(length: int, k_min: int = 1, k_max: int = 13) -> int:
    """
    论文启发式：k = 0.7 * log4(length)
    这里使用 round，并 clamp 到 [k_min, k_max]
    """
    if length <= 0:
        return k_min
    k = int(round(0.7 * (math.log(length) / math.log(4))))
    if k < k_min:
        k = k_min
    if k > k_max:
        k = k_max
    return k


def count_kmers_single_contig(seq: str, k: int) -> Dict[str, int]:
    """
    对单条序列统计 k-mer 次数：
    - 仅接受窗口全为大写 A/C/G/T；遇到 N/小写/其他字符 -> 跳过该窗口
    """
    counts: Dict[str, int] = {}
    L = len(seq)
    if L < k or k <= 0:
        return counts

    # 朴素实现：够用且稳；如需更快可换 rolling hash
    for i in range(L - k + 1):
        kmer = seq[i:i + k]
        ok = True
        for ch in kmer:
            if ch not in _DNA_UPPER:
                ok = False
                break
        if not ok:
            continue
        counts[kmer] = counts.get(kmer, 0) + 1
    return counts


def kmer_spectrum_pmf(seq: str, k: int) -> Optional[np.ndarray]:
    """
    构建论文定义的 k-mer 频谱 PMF：
      f_i = 出现次数为 i 的 k-mer “类型数”
      p_i = f_i / 4^k
    其中 i=0 也包含（nullomers）
    返回 pmf 向量，长度为 (max_count + 1)
    """
    if k <= 0:
        return None
    total_types = 4 ** k

    counts = count_kmers_single_contig(seq, k)
    observed_types = len(counts)

    # 统计 f_i（i>=1）
    # 用 numpy bincount 更快一些
    if observed_types == 0:
        # 全部都是 0 次
        pmf = np.zeros(1, dtype=np.float64)
        pmf[0] = 1.0
        return pmf

    cvals = np.fromiter(counts.values(), dtype=np.int64)
    max_c = int(cvals.max())
    hist = np.bincount(cvals, minlength=max_c + 1).astype(np.float64)  # hist[i] = #types with count=i, for i>=0 but hist[0] currently counts count==0 among observed (none)
    # hist[0] 应该是 0（因为 observed 的 count 不会为0），把 nullomer 加进去
    hist[0] = float(total_types - observed_types)

    pmf = hist / float(total_types)
    # 数值稳健：避免 -0.0
    pmf = np.maximum(pmf, 0.0)
    # 归一化校验（可选）
    s = pmf.sum()
    if s > 0:
        pmf = pmf / s
    return pmf


def _jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen–Shannon divergence（base-2）"""
    m = 0.5 * (p + q)
    p2 = np.clip(p, eps, 1.0)
    q2 = np.clip(q, eps, 1.0)
    m2 = np.clip(m, eps, 1.0)
    return float(0.5 * (np.sum(p2 * np.log2(p2 / m2)) + np.sum(q2 * np.log2(q2 / m2))))


def kmer_spectrum_metrics(
    seq_a: str,
    seq_b: str,
    k: Optional[int] = None,
    k_min: int = 1,
    k_max: int = 13,
) -> Dict[str, float]:
    """
    计算论文里用于比较 k-mer spectrum 的指标：
      - KS statistic（CDF 最大差）
      - JSD
      - EMD（CDF L1 差：sum |CDF_a - CDF_b|）
    """
    # 自动选 k：用两条序列有效长度（仅大写ACGT）较短者
    if k is None or k <= 0:
        eff_len = min(_effective_length_acgt_upper(seq_a), _effective_length_acgt_upper(seq_b))
        k = choose_k_from_length(eff_len, k_min=k_min, k_max=k_max)

    pmf_a = kmer_spectrum_pmf(seq_a, k)
    pmf_b = kmer_spectrum_pmf(seq_b, k)
    if pmf_a is None or pmf_b is None:
        return {
            "kmer_k": float("nan"),
            "kmer_KS": float("nan"),
            "kmer_JSD": float("nan"),
            "kmer_EMD": float("nan"),
        }

    # 对齐长度（不同 max_count 时补零）
    m = max(len(pmf_a), len(pmf_b))
    pa = np.pad(pmf_a, (0, m - len(pmf_a)))
    pb = np.pad(pmf_b, (0, m - len(pmf_b)))

    cdf_a = np.cumsum(pa)
    cdf_b = np.cumsum(pb)

    ks = float(np.max(np.abs(cdf_a - cdf_b)))
    jsd = _jsd(pa, pb)
    emd = float(np.sum(np.abs(cdf_a - cdf_b)))

    return {
        "kmer_k": float(k),
        "kmer_KS": ks,
        "kmer_JSD": jsd,
        "kmer_EMD": emd,
    }


# =========================
# Evaluator
# =========================
class GenEvaluator:
    """
    计算数据集的生成能力并保存结果。

    流程：
      1) 从数据集提取序列
      2) 将序列分成 prompt 和 ground truth continuation
      3) 调用模型的 generate() 方法生成 continuation
      4) 评估生成的序列与 ground truth 的相似度
      5) 保存结果到 JSON/JSONL 文件

    支持多种数据集格式：
    - (idx, sequence, taxid): GenDataset 格式
    - (idx, sequence): GenDataset 旧格式（taxid 为 None）
    - dict: {"sequence": ..., "taxid": ...}
    """

    def __init__(
        self,
        model,
        dataset,
        output_dir: str = "./runs/gen",
        # 生成参数
        prompt_len: int = -1,
        batch_size: int = 128,
        temperature: float = 1.0,
        top_k: int = 4,
        # k-mer spectrum 评估参数
        enable_kmer_spectrum: bool = True,
        kmer_k: int = -1,              # <=0 表示自动选 k
        kmer_k_min: int = 1,
        kmer_k_max: int = 13,
        # 保存选项
        save_per_sample: bool = True,
        save_summary: bool = True,
        seed: int = 2025,
    ):
        self.model = model
        self.dataset = dataset
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.prompt_len = prompt_len
        self.batch_size = batch_size
        self.temperature = temperature
        self.top_k = top_k

        self.enable_kmer_spectrum = enable_kmer_spectrum
        self.kmer_k = kmer_k
        self.kmer_k_min = kmer_k_min
        self.kmer_k_max = kmer_k_max

        self.save_per_sample = save_per_sample
        self.save_summary = save_summary
        self.seed = seed

    def _extract_sequences(self, dataset) -> Tuple[List[str], List[Any]]:
        sequences: List[str] = []
        taxids: List[Any] = []

        for i in range(len(dataset)):
            item = dataset[i]

            # dict 格式
            if isinstance(item, dict):
                if "sequence" not in item:
                    raise KeyError("dataset item dict missing key: sequence")
                sequences.append(item["sequence"])
                taxids.append(item.get("taxid"))
                continue

            # tuple/list 格式
            if not isinstance(item, (tuple, list)):
                raise ValueError("dataset item must be tuple/list/dict")

            if len(item) == 3:
                first, second, third = item
                if isinstance(first, int) and isinstance(second, str):
                    sequences.append(second)
                    taxids.append(third)
                else:
                    raise ValueError(
                        f"无法识别 3 元素 tuple 格式: first={type(first)}, second={type(second)}, third={type(third)}"
                    )
            elif len(item) == 2:
                first, second = item
                if isinstance(first, int) and isinstance(second, str):
                    sequences.append(second)
                    taxids.append(None)
                elif isinstance(first, str):
                    sequences.append(first)
                    taxids.append(None)
                else:
                    raise ValueError(
                        f"无法识别 2 元素 tuple 格式: first={type(first)}, second={type(second)}"
                    )
            else:
                raise ValueError(
                    f"数据集 __getitem__ 返回了 {len(item)} 个值，期望 2/3 或 dict"
                )

        return sequences, taxids

    def _generate_for_split(self, dataset, split_name: str) -> List[Dict[str, Any]]:
        sequences, taxids = self._extract_sequences(dataset)

        if not sequences:
            return []

        print(f"[{split_name}] Generating sequences for {len(sequences)} samples...")

        if not hasattr(self.model, "generate"):
            raise AttributeError(
                f"Model {type(self.model)} does not have generate method. "
                "Please ensure the model provides generate(prompt_seqs, n_tokens, ...)."
            )

        # 准备 prompt / gt
        prompt_list: List[str] = []
        n_tokens_list: List[int] = []
        ground_truth_list: List[str] = []
        valid_indices: List[int] = []

        for i, seq in enumerate(sequences):
            if not isinstance(seq, str) or len(seq) <= 1:
                continue

            pl = calc_prompt_len(len(seq), self.prompt_len)
            if pl <= 0:
                continue

            prompt = seq[:pl]
            gt = seq[pl:]
            if len(gt) == 0:
                continue

            prompt_list.append(prompt)
            ground_truth_list.append(gt)
            n_tokens_list.append(len(gt))
            valid_indices.append(i)

        if not prompt_list:
            print(f"[{split_name}] No valid sequences for generation.")
            return []
        
        # 小批量测试

        # prompt_list = prompt_list[:5]
        # ground_truth_list = ground_truth_list[:5]
        # n_tokens_list = n_tokens_list[:5]
        # valid_indices = valid_indices[:5]

        print(f"[{split_name}] Valid sequences: {len(prompt_list)}")

        all_details: List[Dict[str, Any]] = []


        for start_idx in tqdm(range(0, len(prompt_list), self.batch_size), desc=f"{split_name} Gen"):
            batch_prompts = prompt_list[start_idx:start_idx + self.batch_size]
            batch_gt = ground_truth_list[start_idx:start_idx + self.batch_size]
            batch_n_tokens = n_tokens_list[start_idx:start_idx + self.batch_size]
            batch_valid_indices = valid_indices[start_idx:start_idx + self.batch_size]

            n_tokens = max(batch_n_tokens) if batch_n_tokens else 0

            try:
                generated_seqs = self.model.generate(
                    prompt_seqs=batch_prompts,
                    n_tokens=n_tokens,
                    temperature=self.temperature,
                    top_k=self.top_k,
                )

                if not isinstance(generated_seqs, list):
                    if hasattr(generated_seqs, "sequences"):
                        generated_seqs = list(generated_seqs.sequences)
                    else:
                        raise ValueError(f"Unexpected generate output type: {type(generated_seqs)}")

                for i, (prompt, gt, n_tok, orig_idx) in enumerate(
                    zip(batch_prompts, batch_gt, batch_n_tokens, batch_valid_indices)
                ):
                    gen_full = generated_seqs[i] if i < len(generated_seqs) else ""
                    gen_full = gen_full if isinstance(gen_full, str) else ""

                    # ✅ 兼容两种 generate 返回：可能包含 prompt，也可能只返回 continuation
                    if gen_full.startswith(prompt):
                        gen_cont = gen_full[len(prompt):]
                    else:
                        gen_cont = gen_full

                    # 截断到 gt 等长
                    gen_tail = gen_cont[:len(gt)]

                    # 传统相似度
                    metrics = evaluate_similarity(gt, gen_tail)

                    # ✅ 追加论文 k-mer spectrum 指标
                    if self.enable_kmer_spectrum:
                        kk = None if (self.kmer_k is None or self.kmer_k <= 0) else int(self.kmer_k)
                        kmer_metrics = kmer_spectrum_metrics(
                            gt,
                            gen_tail,
                            k=kk,
                            k_min=self.kmer_k_min,
                            k_max=self.kmer_k_max,
                        )
                        metrics.update(kmer_metrics)

                    taxid = taxids[orig_idx] if orig_idx < len(taxids) else None

                    sample_detail = {
                        "sequence_index": orig_idx,
                        "taxid": taxid,
                        "prompt": prompt,
                        "ground_truth": gt,
                        "generated_full": gen_full,
                        "generated_continuation": gen_cont,
                        "generated_tail": gen_tail,
                        "prompt_length": len(prompt),
                        "ground_truth_length": len(gt),
                        "generated_full_length": len(gen_full),
                        "generated_continuation_length": len(gen_cont),
                        "generated_tail_length": len(gen_tail),
                        **metrics,
                    }

                    all_details.append(sample_detail)

            except Exception as e:
                warnings.warn(f"Error processing batch starting at {start_idx}: {str(e)}")
                for i, (prompt, gt, n_tok, orig_idx) in enumerate(
                    zip(batch_prompts, batch_gt, batch_n_tokens, batch_valid_indices)
                ):
                    taxid = taxids[orig_idx] if orig_idx < len(taxids) else None
                    all_details.append({
                        "sequence_index": orig_idx,
                        "taxid": taxid,
                        "prompt": prompt,
                        "ground_truth": gt,
                        "generated_full": "",
                        "generated_continuation": "",
                        "generated_tail": "",
                        "prompt_length": len(prompt),
                        "ground_truth_length": len(gt),
                        "generated_full_length": 0,
                        "generated_continuation_length": 0,
                        "generated_tail_length": 0,
                        "exact_match_acc": float("nan"),
                        "edit_distance": float("nan"),
                        "alignment_identity": float("nan"),
                        # kmer placeholders
                        "kmer_k": float("nan"),
                        "kmer_KS": float("nan"),
                        "kmer_JSD": float("nan"),
                        "kmer_EMD": float("nan"),
                        "error": str(e)[:200],
                    })

        return all_details

    def _compute_statistics(self, details: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算生成评估的统计信息（忽略 nan）。"""
        metrics_list = {
            "exact_match_acc": [],
            "edit_distance": [],
            "alignment_identity": [],
        }

        # ✅ 追加 k-mer spectrum 统计
        metrics_list_extra = {
            "kmer_k": [],
            "kmer_KS": [],
            "kmer_JSD": [],
            "kmer_EMD": [],
        }

        for detail in details:
            for metric_name in metrics_list.keys():
                val = detail.get(metric_name)
                if val is not None and isinstance(val, (int, float)) and math.isfinite(val):
                    metrics_list[metric_name].append(float(val))

            for metric_name in metrics_list_extra.keys():
                val = detail.get(metric_name)
                if val is not None and isinstance(val, (int, float)) and math.isfinite(val):
                    metrics_list_extra[metric_name].append(float(val))

        stats: Dict[str, Any] = {}

        def _summ(values: List[float]) -> Dict[str, float]:
            if not values:
                return {
                    "mean": float("nan"),
                    "median": float("nan"),
                    "std": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                }
            arr = np.asarray(values, dtype=np.float64)
            return {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

        for metric_name, values in metrics_list.items():
            stats[metric_name] = _summ(values)

        for metric_name, values in metrics_list_extra.items():
            stats[metric_name] = _summ(values)

        stats["count"] = len(details)
        stats["valid_count"] = len([d for d in details if "error" not in d])

        return stats

    def run(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "output_dir": self.output_dir,
            "prompt_len": self.prompt_len,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "enable_kmer_spectrum": self.enable_kmer_spectrum,
            "kmer_k": self.kmer_k,
            "kmer_k_min": self.kmer_k_min,
            "kmer_k_max": self.kmer_k_max,
        }

        split_name = "all"
        details = self._generate_for_split(self.dataset, split_name)

        stats = self._compute_statistics(details)
        results[f"{split_name}_statistics"] = stats
        results[f"{split_name}_size"] = len(details)

        if self.save_per_sample:
            per_sample_path = os.path.join(self.output_dir, f"{split_name}_gen_per_sample.jsonl")
            with open(per_sample_path, "w", encoding="utf-8") as f:
                for detail in details:
                    f.write(json.dumps(detail, ensure_ascii=False) + "\n")
            print(f"[{split_name}] Saved per-sample results to {per_sample_path}")

        if self.save_summary:
            summary_path = os.path.join(self.output_dir, "gen_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Saved summary to {summary_path}")

        return results


# ===== 示例 =====
if __name__ == "__main__":
    # 示例：评估相似度 + kmer spectrum
    gt = "ACGTACGTACGTACGTACGTACGT"
    gen = "ACGTACGTACGAACGTACGTACGT"  # 其中一个位置不一样

    base = evaluate_similarity(gt, gen)
    kmer = kmer_spectrum_metrics(gt, gen, k=None, k_min=1, k_max=13)

    print({**base, **kmer})

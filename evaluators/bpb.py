# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import json
import math
import warnings
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
from tqdm.auto import tqdm


def ppl_details_to_bpb(details):
    """
    details: Evo2Model.get_ppl(..., return_details=True) 的返回
             每个元素至少包含: avg_nll_token, token_count, char_count
    返回: 每条样本的 bpb（float），无法计算则为 nan
    """
    out = []
    ln2 = math.log(2.0)

    for d in details:
        avg = d.get("avg_nll_token", float("nan"))
        tok = int(d.get("token_count", 0) or 0)
        ch  = int(d.get("char_count", 0) or 0)

        if ch <= 0 or tok <= 0 or not math.isfinite(avg):
            out.append(float("nan"))
            continue

        total_nll = avg * tok                 # nat
        nll_per_base = total_nll / ch         # nat/base
        bpb = nll_per_base / ln2              # bits/base
        out.append(float(bpb))

    return out


class BPBEvaluator:
    """
    计算数据集的 BPB (bits per base) 值并保存结果。
    
    流程：
      1) 从数据集提取序列
      2) 调用模型的 get_ppl(..., return_details=True) 获取详细信息
      3) 使用 ppl_details_to_bpb 转换为 BPB 值
      4) 保存结果到 JSON/JSONL 文件
    
    支持多种数据集格式：
    - (idx, sequence, taxid): GenDataset 格式
    - (idx, sequence): GenDataset 旧格式（taxid 为 None）
    - dict: {"sequence": ..., "taxid": ...}
    """

    def __init__(
        self,
        model,                                 # 模型实例，需提供 .get_ppl(sequences, return_details=True) 方法
        dataset,                               # 数据集
        output_dir: str = "./runs/bpb",
        # 计算参数
        batch_size: int = 128,
        # 保存选项
        save_per_sample: bool = True,         # 是否保存每个样本的详细结果
        save_summary: bool = True,             # 是否保存汇总统计
        seed: int = 2025,
    ):
        self.model = model
        self.dataset = dataset
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.batch_size = batch_size
        self.save_per_sample = save_per_sample
        self.save_summary = save_summary
        self.seed = seed

    def _extract_sequences(self, dataset) -> Tuple[List[str], List[Any]]:
        """
        从数据集提取序列和 taxid。
        支持多种格式：
        - (idx, sequence, taxid): GenDataset 格式
        - (idx, sequence): GenDataset 旧格式（taxid 为 None）
        - (sequence, label): 标准格式（忽略 label，taxid 为 None）
        - dict: {"sequence": ..., "taxid": ...}
        
        返回:
          sequences: List[str]
          taxids: List[Any] (可能是 int 或 None)
        """
        sequences: List[str] = []
        taxids: List[Any] = []

        for i in range(len(dataset)):
            item = dataset[i]
            
            # 支持 dict 格式
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
                # (idx, sequence, taxid) 格式 - GenDataset 格式
                first, second, third = item
                if isinstance(first, int) and isinstance(second, str):
                    sequences.append(second)
                    taxids.append(third)
                else:
                    raise ValueError(
                        f"无法识别 3 元素 tuple 格式: first={type(first)}, second={type(second)}, third={type(third)}")
            elif len(item) == 2:
                # 可能是 (idx, sequence) 或 (sequence, label)
                first, second = item
                if isinstance(first, int) and isinstance(second, str):
                    # (idx, sequence) 格式 - GenDataset 旧格式
                    sequences.append(second)
                    taxids.append(None)
                elif isinstance(first, str):
                    # (sequence, label) 格式 - 只取 sequence，忽略 label
                    sequences.append(first)
                    taxids.append(None)
                else:
                    raise ValueError(
                        f"无法识别 2 元素 tuple 格式: first={type(first)}, second={type(second)}")
            else:
                raise ValueError(
                    f"数据集 __getitem__ 返回了 {len(item)} 个值，期望 2/3 或 dict")

        return sequences, taxids

    def _compute_bpb_for_split(
        self,
        dataset,
        split_name: str,
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        """
        对单个数据集计算 BPB。
        返回:
          bpb_values: List[float] - 每个样本的 BPB 值
          details: List[Dict] - 每个样本的详细信息
        """
        sequences, taxids = self._extract_sequences(dataset)
        
        if not sequences:
            return [], []

        print(f"[{split_name}] Computing BPB for {len(sequences)} sequences...")

        # 检查模型是否有 get_ppl 方法
        if not hasattr(self.model, "get_ppl"):
            raise AttributeError(
                f"Model {type(self.model)} does not have get_ppl method. "
                "Please ensure the model provides get_ppl(sequences, return_details=True)."
            )

        # 分批处理序列
        all_details: List[Dict[str, Any]] = []
        all_bpb: List[float] = []

        # 小批量测试
        # sequences=sequences[:10]

        for start_idx in tqdm(range(0, len(sequences), self.batch_size), desc=f"{split_name} BPB"):
            batch_seqs = sequences[start_idx:start_idx + self.batch_size]
            
            try:
                # 调用模型的 get_ppl 方法
                details = self.model.get_ppl(
                    batch_seqs,
                    return_details=True,
                )
                # print(details)
                # exit()
                # 转换为 BPB
                batch_bpb = ppl_details_to_bpb(details)
                
                # 保存详细信息
                batch_taxids = taxids[start_idx:start_idx + self.batch_size]
                for i, (seq, taxid, detail, bpb) in enumerate(
                    zip(batch_seqs, batch_taxids, details, batch_bpb)
                ):
                    sample_detail = {
                        "sequence_index": start_idx + i,
                        "taxid": taxid,
                        "bpb": float(bpb),
                        "avg_nll_token": detail.get("avg_nll_token"),
                        "token_count": detail.get("token_count"),
                        "char_count": detail.get("char_count"),
                    }
                    # 保留其他可能的字段
                    for k, v in detail.items():
                        if k not in sample_detail:
                            if isinstance(v, (int, float, str, bool, type(None))):
                                sample_detail[k] = v
                            elif isinstance(v, np.number):
                                sample_detail[k] = float(v)
                    
                    all_details.append(sample_detail)
                    all_bpb.append(bpb)

            except Exception as e:
                warnings.warn(f"Error processing batch starting at {start_idx}: {str(e)}")
                # 为失败的批次添加占位符
                batch_taxids = taxids[start_idx:start_idx + self.batch_size]
                for i in range(len(batch_seqs)):
                    all_details.append({
                        "sequence_index": start_idx + i,
                        "taxid": batch_taxids[i] if i < len(batch_taxids) else None,
                        "bpb": float("nan"),
                        "error": str(e)[:200],
                    })
                    all_bpb.append(float("nan"))

        return all_bpb, all_details


    def _compute_statistics(self, bpb_values: List[float]) -> Dict[str, float]:
        """计算 BPB 的统计信息（忽略 nan）。"""
        valid_bpb = [b for b in bpb_values if math.isfinite(b)]
        
        if not valid_bpb:
            return {
                "mean": float("nan"),
                "median": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "count": 0,
                "valid_count": 0,
            }

        return {
            "mean": float(np.mean(valid_bpb)),
            "median": float(np.median(valid_bpb)),
            "std": float(np.std(valid_bpb)),
            "min": float(np.min(valid_bpb)),
            "max": float(np.max(valid_bpb)),
            "count": len(bpb_values),
            "valid_count": len(valid_bpb),
        }

    def run(self) -> Dict[str, Any]:
        """
        主流程：计算数据集的 BPB 并保存结果。
        """
        results: Dict[str, Any] = {
            "output_dir": self.output_dir,
        }

        # 处理数据集
        split_name = "all"
        bpb_values, details = self._compute_bpb_for_split(self.dataset, split_name)
        
        # 统计信息
        stats = self._compute_statistics(bpb_values)
        results[f"{split_name}_statistics"] = stats
        results[f"{split_name}_size"] = len(details)

        # 保存每个样本的详细结果
        if self.save_per_sample:
            per_sample_path = os.path.join(self.output_dir, f"{split_name}_bpb_per_sample.jsonl")
            with open(per_sample_path, "w", encoding="utf-8") as f:
                for detail in details:
                    f.write(json.dumps(detail, ensure_ascii=False) + "\n")
            print(f"[{split_name}] Saved per-sample results to {per_sample_path}")

        # 保存汇总 JSON
        if self.save_summary:
            summary_path = os.path.join(self.output_dir, "bpb_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Saved summary to {summary_path}")

        return results


# ===== 示例 =====
if __name__ == "__main__":
    details = [
        {"avg_nll_token": 1.1522243023, "token_count": 512, "char_count": 512},
        {"avg_nll_token": 0.0508479675, "token_count": 480, "char_count": 512},
    ]
    print(ppl_details_to_bpb(details))
    # 解释：
    # 第1条：token_count==char_count -> bpb ≈ avg_nll_token / ln2
    # 第2条：token_count < char_count -> 每个 base 的平均 NLL 更小（被"摊薄"）

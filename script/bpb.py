#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据现有 all_bpb_per_sample.jsonl 重新计算 BPB。

规则：
- 若 char_count == sequence_chars - 128：保留原 BPB
- 否则：令 char_count = sequence_chars - 128，用公式 bpb = (avg_nll_token * token_count) / (char_count * ln2) 重新计算

遍历目录：{root}/子文件夹/模型名/all_bpb_per_sample.jsonl
处理完后写入新文件：all_bpb_per_sample_new.jsonl、bpb_summary_new.json（不覆盖原文件）。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

LN2 = math.log(2.0)
EXPECTED_OFFSET = 128  # char_count 期望为 sequence_chars - 128
BPB_DECIMALS = 4  # 结果保留小数位数


def recompute_bpb_one(d: Dict[str, Any]) -> tuple[float | None, bool]:
    """
    对单条样本决定是否重算 BPB。
    返回 (bpb_value, changed)。
    - 若缺少必要字段或 sequence_chars - 128 <= 0，返回 (原 bpb, False)
    - 若 char_count == sequence_chars - 128，返回 (原 bpb, False)
    - 否则用新 char_count 重算，返回 (新 bpb, True)
    """
    orig_bpb = d.get("bpb")
    char_count = d.get("char_count")
    if char_count is not None and char_count != "":
        try:
            seq_chars = int(char_count) + 128
        except (TypeError, ValueError):
            seq_chars = d.get("sequence_chars")
    else:
        seq_chars = d.get("sequence_chars")
    avg_nll = d.get("avg_nll_token")
    tok_count = d.get("token_count", 0) or 0

    if seq_chars is None:
        v = orig_bpb if isinstance(orig_bpb, (int, float)) and math.isfinite(orig_bpb) else None
        return (round(v, BPB_DECIMALS) if v is not None else None, False)

    try:
        seq_chars = int(seq_chars)
    except (TypeError, ValueError):
        v = orig_bpb if isinstance(orig_bpb, (int, float)) and math.isfinite(orig_bpb) else None
        return (round(v, BPB_DECIMALS) if v is not None else None, False)

    expected_ch = seq_chars - EXPECTED_OFFSET
    if expected_ch <= 0:
        v = orig_bpb if isinstance(orig_bpb, (int, float)) and math.isfinite(orig_bpb) else None
        return (round(v, BPB_DECIMALS) if v is not None else None, False)

    if char_count is not None:
        try:
            char_count = int(char_count)
        except (TypeError, ValueError):
            char_count = None
    # 仅在未截断且 char_count 已等于 expected_ch 时保留原 BPB；截断时 (tok_count < expected_ch) 必须重算
    if char_count == expected_ch and tok_count >= expected_ch:
        v = orig_bpb if isinstance(orig_bpb, (int, float)) and math.isfinite(orig_bpb) else None
        return (round(v, BPB_DECIMALS) if v is not None else None, False)

    if avg_nll is None or not math.isfinite(avg_nll) or not tok_count or tok_count <= 0:
        v = orig_bpb if isinstance(orig_bpb, (int, float)) and math.isfinite(orig_bpb) else None
        return (round(v, BPB_DECIMALS) if v is not None else None, False)

    # 截断时只计分了 tok_count 个碱基，分母用实际计分碱基数
    effective_ch = min(expected_ch, int(tok_count))
    total_nll = float(avg_nll) * int(tok_count)
    nll_per_base = total_nll / effective_ch
    new_bpb = nll_per_base / LN2
    return (round(float(new_bpb), BPB_DECIMALS), True)


def compute_statistics(bpb_values: List[float]) -> Dict[str, float]:
    """忽略 nan，计算 mean/median/std/min/max/count/valid_count。"""
    valid = [float(b) for b in bpb_values if isinstance(b, (int, float)) and math.isfinite(b)]
    n = len(bpb_values)
    if not valid:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "count": n,
            "valid_count": 0,
        }
    s = sum(valid)
    m = len(valid)
    mean = s / m
    sorted_v = sorted(valid)
    median = sorted_v[m // 2] if m % 2 else (sorted_v[m // 2 - 1] + sorted_v[m // 2]) / 2.0
    variance = sum((x - mean) ** 2 for x in valid) / m
    std = math.sqrt(variance)
    min_v, max_v = min(valid), max(valid)
    return {
        "mean": round(mean, BPB_DECIMALS),
        "median": round(median, BPB_DECIMALS),
        "std": round(std, BPB_DECIMALS),
        "min": round(min_v, BPB_DECIMALS),
        "max": round(max_v, BPB_DECIMALS),
        "count": n,
        "valid_count": len(valid),
    }


def process_jsonl_path(jsonl_path: str, subfolder: str, model_name: str) -> Dict[str, Any]:
    """
    处理单个模型的 all_bpb_per_sample.jsonl：按行读入，按规则重算 BPB，写入 all_bpb_per_sample_new.jsonl。
    """
    out_jsonl = os.path.join(os.path.dirname(jsonl_path), "all_bpb_per_sample_new.jsonl")
    results = {"updated": 0, "unchanged": 0, "skipped": 0, "bpb_values": [], "output_path": out_jsonl}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines: List[str] = []
    bpb_values: List[float] = []

    for line in tqdm(lines, desc=f"  [{subfolder}] {model_name}", leave=False, unit=" samples"):
        line = line.strip()
        if not line:
            new_lines.append("")
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            new_lines.append(line)
            results["skipped"] += 1
            continue

        bpb_val, changed = recompute_bpb_one(d)
        if changed:
            d["char_count"] = max(0, (d.get("sequence_chars") or 0) - EXPECTED_OFFSET)
            d["bpb"] = bpb_val
            results["updated"] += 1
        else:
            results["unchanged"] += 1
        # 统计用最终 bpb（可能为原值或新值），写入时统一保留四位小数
        final_bpb = d.get("bpb")
        if isinstance(final_bpb, (int, float)) and math.isfinite(final_bpb):
            final_bpb = round(final_bpb, BPB_DECIMALS)
            d["bpb"] = final_bpb
            bpb_values.append(final_bpb)
        new_lines.append(json.dumps(d, ensure_ascii=False))

    results["bpb_values"] = bpb_values

    with open(out_jsonl, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + ("\n" if new_lines else ""))

    return results


def find_model_dirs(root: str) -> List[tuple[str, str, str]]:
    """
    返回 (subfolder, model_name, jsonl_path) 列表。
    root 下直接子目录为 subfolder，每个 subfolder 下直接子目录为 model，model 下需有 all_bpb_per_sample.jsonl。
    """
    out: List[tuple[str, str, str]] = []
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        return out
    for sub in sorted(os.listdir(root)):
        sub_path = os.path.join(root, sub)
        if not os.path.isdir(sub_path) or sub.startswith("."):
            continue
        for name in sorted(os.listdir(sub_path)):
            model_path = os.path.join(sub_path, name)
            if not os.path.isdir(model_path):
                continue
            jsonl = os.path.join(model_path, "all_bpb_per_sample.jsonl")
            if os.path.isfile(jsonl):
                out.append((sub, name, jsonl))
    return out


def write_summary_csv(root: str, collected: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    """
    collected[model_name][subfolder] = {min, max, median, mean}
    输出 CSV：第一列 model_name，其余列为各子数据集的 min, max, median, mean。
    子数据集顺序固定为 genome-short, genome-medium, genome-long（仅输出存在的）。
    """
    if not collected:
        return ""
    subfolders_order = ["genome-short", "genome-medium", "genome-long"]
    subfolders = [s for s in subfolders_order if any(s in collected[m] for m in collected)]
    if not subfolders:
        subfolders = sorted({s for m in collected for s in collected[m]})
    headers = ["model_name"]
    for sub in subfolders:
        headers.extend([f"{sub}_min", f"{sub}_max", f"{sub}_median", f"{sub}_mean"])
    rows: List[Tuple[str, ...]] = []
    for model_name in sorted(collected.keys()):
        row: List[Any] = [model_name]
        for sub in subfolders:
            st = collected[model_name].get(sub)
            if st is None:
                row.extend(["", "", "", ""])
            else:
                def _v(k: str) -> str:
                    v = st.get(k)
                    if v is None or (isinstance(v, float) and not math.isfinite(v)):
                        return ""
                    if isinstance(v, (int, float)):
                        return f"{round(float(v), BPB_DECIMALS):.{BPB_DECIMALS}f}"
                    return str(v)
                row.extend([_v("min"), _v("max"), _v("median"), _v("mean")])
        rows.append(tuple(row))
    out_path = os.path.join(root, "bpb_summary_table_new.csv")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Recompute BPB from existing per-sample jsonl (char_count = sequence_chars - 128).")
    parser.add_argument(
        "root",
        nargs="?",
        default="/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/results/Bpb",
        help="Root folder containing subfolders (e.g. genome-long, genome-medium, genome-short), each with model dirs.",
    )
    args = parser.parse_args()
    root = os.path.abspath(args.root)

    tasks = find_model_dirs(root)
    if not tasks:
        print(f"[WARN] No all_bpb_per_sample.jsonl found under {root}")
        return

    print(f"[INFO] Found {len(tasks)} model(s) under {root}\n")

    # collected[model_name][subfolder] = {min, max, median, mean} 用于最后输出总表
    collected: Dict[str, Dict[str, Dict[str, float]]] = {}

    for subfolder, model_name, jsonl_path in tqdm(tasks, desc="Subfolder / Model", unit=" model"):
        model_dir = os.path.dirname(jsonl_path)
        summary_path_new = os.path.join(model_dir, "bpb_summary_new.json")

        # 若已存在 bpb_summary_new.json 则跳过，不重复更新
        if os.path.isfile(summary_path_new):
            try:
                with open(summary_path_new, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                stats = summary.get("all_statistics") or {}
            except Exception:
                stats = {}
            if model_name not in collected:
                collected[model_name] = {}
            collected[model_name][subfolder] = {
                "min": stats.get("min"),
                "max": stats.get("max"),
                "median": stats.get("median"),
                "mean": stats.get("mean"),
            }
            tqdm.write(f"\n>>> Subfolder: {subfolder}  |  Model: {model_name}  |  (skip, bpb_summary_new.json exists)")
            continue

        tqdm.write(f"\n>>> Subfolder: {subfolder}  |  Model: {model_name}  |  {jsonl_path}")
        res = process_jsonl_path(jsonl_path, subfolder, model_name)
        tqdm.write(f"    Updated: {res['updated']}, Unchanged: {res['unchanged']}, Skipped: {res['skipped']}")
        tqdm.write(f"    Written: {res['output_path']}")

        # 写入 bpb_summary_new.json
        stats = compute_statistics(res["bpb_values"])
        summary_orig = os.path.join(model_dir, "bpb_summary.json")
        if os.path.isfile(summary_orig):
            with open(summary_orig, "r", encoding="utf-8") as f:
                summary = json.load(f)
        else:
            summary = {"output_dir": model_dir}
        summary["all_statistics"] = stats
        summary["all_size"] = len(res["bpb_values"])
        with open(summary_path_new, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        tqdm.write(f"    Written: {summary_path_new}")
        tqdm.write(f"    Summary: mean={stats['mean']:.4f}, median={stats['median']:.4f}, valid={stats['valid_count']}")

        # 收集该模型在该子数据集下的统计，用于总表
        if model_name not in collected:
            collected[model_name] = {}
        collected[model_name][subfolder] = {
            "min": stats.get("min"),
            "max": stats.get("max"),
            "median": stats.get("median"),
            "mean": stats.get("mean"),
        }

    # 输出总表 CSV
    table_path = write_summary_csv(root, collected)
    if table_path:
        print(f"\n[INFO] Summary table written: {table_path}")

    print("\n[OK] Done.")


if __name__ == "__main__":
    main()

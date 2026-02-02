#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 JSONL（如 all_gen_per_sample_metrics.jsonl）加载生成评估结果，
按模型和 group（short/medium/long）聚合指标，输出 CSV。

CSV 结构：
- 第一列：model（模型名）
- 其余列：按 group 分块，每块为 cds-short / cds-medium / cds-long，
  每块内列：alignment_identity, edit_distance, exact_match_acc, kmer_EMD, kmer_JSD, kmer_KS, CDS success rate
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

# 每组下的指标列（与用户要求一致）
METRIC_KEYS = [
    "edit_distance",
    "exact_match_acc",
    "kmer_JSD",
    "kmer_KS",
]

# CDS success rate 由 is_CDS 聚合得到，单独处理
CDS_RATE_KEY = "CDS success rate"

# group 显示顺序与表头前缀（cds-short, cds-medium, cds-long）
GROUP_ORDER = ["short", "medium", "long"]
GROUP_PREFIX = {"short": "cds-short", "medium": "cds-medium", "long": "cds-long"}


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """加载 JSONL，每行一个 JSON 对象。"""
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Skip invalid JSON line: {e}", file=sys.stderr)
    return records


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        try:
            f = float(v)
            if f != f:  # nan
                return None
            return f
        except (TypeError, ValueError):
            return None
    return None


def aggregate_by_model_and_group(
    records: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    """
    按 (model_name, group) 聚合：对数值指标求均值，CDS success rate = mean(is_CDS)。
    返回: { model_name: { group: { "metric_name": value, "CDS success rate": value } } }
    """
    # model -> group -> list of values per metric
    metric_values: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    # model -> group -> list of is_CDS (bool)
    cds_flags: Dict[str, Dict[str, List[bool]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for r in records:
        model = r.get("model_name")
        group = r.get("group")
        if not model or not group:
            continue
        g = (group.lower() if isinstance(group, str) else str(group)).strip()
        if g not in GROUP_ORDER:
            continue

        for key in METRIC_KEYS:
            val = _safe_float(r.get(key))
            if val is not None:
                metric_values[model][g][key].append(val)

        is_cds = r.get("is_CDS")
        if isinstance(is_cds, bool):
            cds_flags[model][g].append(is_cds)

    # 收集所有出现过的 (model, group)
    all_models = set(metric_values.keys()) | set(cds_flags.keys())
    result: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for model in all_models:
        result[model] = {}
        for g in GROUP_ORDER:
            row: Dict[str, Optional[float]] = {}
            for key in METRIC_KEYS:
                vals = metric_values.get(model, {}).get(g, {}).get(key)
                if vals:
                    row[key] = sum(vals) / len(vals)
                else:
                    row[key] = None
            cds_list = cds_flags.get(model, {}).get(g, [])
            row[CDS_RATE_KEY] = (sum(cds_list) / len(cds_list)) if cds_list else None
            result[model][g] = row
    return result


def build_csv_header() -> List[str]:
    """表头：model, 然后按 cds-short, cds-medium, cds-long 各一组指标，组与组之间加一空列。"""
    header = ["model"]
    for i, g in enumerate(GROUP_ORDER):
        if i > 0:
            header.append("")  # 组与组之间空列
        prefix = GROUP_PREFIX[g]
        for key in METRIC_KEYS + [CDS_RATE_KEY]:
            header.append(f"{prefix}_{key}")
    return header


def format_cell(value: Optional[float]) -> str:
    if value is None:
        return ""
    return str(value)


def build_csv_rows(
    aggregated: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    model_order: Optional[List[str]] = None,
) -> List[List[str]]:
    """按表头顺序生成行。若提供 model_order 则按该顺序输出，否则按 key 排序。"""
    header = build_csv_header()
    rows = []
    models = list(aggregated.keys())
    if model_order is not None:
        # 保留 order 里有的且存在于数据的，再补上多出来的
        ordered = [m for m in model_order if m in aggregated]
        rest = sorted(set(models) - set(ordered))
        models = ordered + rest
    else:
        models = sorted(models)

    for model in models:
        groups_data = aggregated[model]
        row = [model]
        for i, g in enumerate(GROUP_ORDER):
            if i > 0:
                row.append("")  # 组与组之间空列
            group_row = groups_data.get(g, {})
            for key in METRIC_KEYS + [CDS_RATE_KEY]:
                row.append(format_cell(group_row.get(key)))
        rows.append(row)
    return rows


def write_csv(output_path: str, rows: List[List[str]], header: List[str]) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="从 JSONL 加载生成评估结果，按模型与 short/medium/long 聚合后输出 CSV"
    )
    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        default=None,
        help="输入 JSONL 路径（默认: results/Generate/all_gen_per_sample_metrics.jsonl）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="输出 CSV 路径（默认: 与输入同目录，文件名为 <basename>.csv）",
    )
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_jsonl = os.path.join(
        project_root, "results", "Generate", "all_gen_per_sample_metrics.jsonl"
    )
    input_path = args.input or default_jsonl

    if not os.path.exists(input_path):
        print(f"[ERROR] Input not found: {input_path}", file=sys.stderr)
        return 1

    records = load_jsonl(input_path)
    if not records:
        print(f"[WARN] No records in {input_path}", file=sys.stderr)
        return 1

    aggregated = aggregate_by_model_and_group(records)
    if not aggregated:
        print("[WARN] No (model_name, group) pairs found.", file=sys.stderr)
        return 1

    header = build_csv_header()
    rows = build_csv_rows(aggregated)

    if args.output:
        output_path = args.output
    else:
        base = os.path.splitext(os.path.basename(input_path))[0]
        if base.endswith("_metrics"):
            base = base
        else:
            base = base + "_metrics"
        output_path = os.path.join(os.path.dirname(input_path), base + ".csv")

    write_csv(output_path, rows, header)
    print(f"[INFO] Saved CSV to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

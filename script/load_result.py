#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import statistics
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SUPPORTED_TASKS = {
    "RNA-taxon-genus": "results/Classification/RNA-taxon-genus",
    "RNA-taxon-times": "results/Classification/RNA-taxon-times",
    "RNA-host-genus": "results/Classification/RNA-host-genus",
    "RNA-host-times": "results/Classification/RNA-host-times",
    "DNA-taxon-genus": "results/Classification/DNA-taxon-genus",
    "DNA-taxon-times": "results/Classification/DNA-taxon-times",
    "DNA-host-genus": "results/Classification/DNA-host-genus",
    "DNA-host-times": "results/Classification/DNA-host-times",
    "ALL-taxon-genus": "results/Classification/ALL-taxon-genus",
    "ALL-taxon-times": "results/Classification/ALL-taxon-times",
    "ALL-host-genus": "results/Classification/ALL-host-genus",
    "ALL-host-times": "results/Classification/ALL-host-times",
}
TASK_ORDER = [
    "RNA-taxon-genus",
    "RNA-taxon-times",
    "RNA-host-genus",
    "RNA-host-times",
    "DNA-taxon-genus",
    "DNA-taxon-times",
    "DNA-host-genus",
    "DNA-host-times",
    "ALL-taxon-genus",
    "ALL-taxon-times",
    "ALL-host-genus",
    "ALL-host-times",
]

# # 模型显示顺序（None 表示空行分隔）
# MODEL_ORDER = [
#     "CNN",
#     None,  # 空行
#     "evo2_1b_base",
#     "evo2_7b_base",
#     "evo2_7b",
#     "evo2_40b_base",
#     "evo2_40b",
#     None,  # 空行
#     "evo-1-8k-base",
#     "evo-1-131k-base",
#     None,  # 空行
#     "evo-1.5-8k-base",
#     None,  # 空行
#     "nt-500m-human",
#     "nt-500m-1000g",
#     "nt-2.5b-1000g",
#     "nt-2.5b-ms",
#     None,  # 空行
#     "ntv2-50m-ms-3kmer",
#     "ntv2-50m-ms",
#     "ntv2-100m-ms",
#     "ntv2-250m-ms",
#     "ntv2-500m-ms",
#     None,  # 空行
#     "ntv3-8m-pre",
#     "ntv3-100m-pre",
#     "ntv3-650m-pre",
#     "ntv3-100m-post",
#     "ntv3-650m-post",
#     None,  # 空行
#     "caduceus-ph",
#     "caduceus-ps",
#     None,  # 空行
#     "DNABERT-3",
#     "DNABERT-4",
#     "DNABERT-5",
#     "DNABERT-6",
#     "DNABERT-S",
#     "DNABERT-2-117M",
#     None,  # 空行
#     "hyenadna-tiny-16k",
#     "hyenadna-tiny-1k",
#     "hyenadna-small-32k",
#     "hyenadna-medium-160k",
#     "hyenadna-medium-450k",
#     "hyenadna-large-1m",
#     None,  # 空行
#     "Genos-1.2B",
#     "Genos-10B",
#     "Genos-10B-v2",
#     None,  # 空行
#     "OmniReg-base",
#     None,  # 空行
#     "gena-lm-bert-base-t2t",
#     "gena-lm-bert-large-t2t",
#     "gena-lm-bigbird-base-t2t",
#     None,  # 空行
#     "GROVER",
#     None,  # 空行
#     "GenomeOcean-100M",
#     "GenomeOcean-500M",
#     "GenomeOcean-4B",
#     None,  # 空行
#     "GENERator-v2-eukaryote-1.2b-base",
#     "GENERator-v2-eukaryote-3b-base",
#     "GENERator-v2-prokaryote-1.2b-base",
#     "GENERator-v2-prokaryote-3b-base",
#     None,  # 空行
#     "AIDO.DNA-300M",
#     "AIDO.DNA-7B",
#     None,  # 空行
#     "AIDO.RNA-650M",
#     "AIDO.RNA-1.6B",
#     "AIDO.RNA-650M-CDS",
#     "AIDO.RNA-1.6B-CDS",
#     None,  # 空行
#     "LucaOne-default-step36M",
#     "LucaOne-gene-step36.8M",
#     None,  # 空行
#     "LucaVirus-default-step3.8M",
#     "LucaVirus-gene-step3.8M",
#     None,  # 空行
#     "RNA-FM",
#     "RiNALMo",
#     "BiRNA-BERT",
#     "RNABERT",
#     "MP-RNA",
# ]

# 模型显示顺序（None 表示空行分隔）
MODEL_ORDER = [
    "CNN",
    None,  # 空行
    "LucaOne-default-step36M",
    "LucaOne-gene-step36.8M",
    "LucaVirus-default-step3.8M",
    "LucaVirus-gene-step3.8M",
    "DNABERT-S",
    "GenomeOcean-100M",
    "GenomeOcean-500M",
    "GenomeOcean-4B",
    None,  # 空行
    "evo-1-8k-base",
    "evo-1-131k-base",
    "evo-1.5-8k-base",
    "evo2_1b_base",
    "evo2_7b_base",
    "evo2_7b",
    "evo2_40b_base",
    "evo2_40b",
    "ntv3-8m-pre",
    "ntv3-100m-pre",
    "ntv3-650m-pre",
    "ntv3-100m-post",
    "ntv3-650m-post",
    None,  # 空行
    "AIDO.DNA-300M",
    "AIDO.DNA-7B",
    "caduceus-ph",
    "caduceus-ps",
    "Genos-1.2B",
    "Genos-10B",
    "Genos-10B-v2",
    "hyenadna-tiny-16k",
    "hyenadna-tiny-1k",
    "hyenadna-small-32k",
    "hyenadna-medium-160k",
    "hyenadna-medium-450k",
    "hyenadna-large-1m",
    "DNABERT-2-117M",
    "gena-lm-bert-base-t2t",
    "gena-lm-bert-large-t2t",
    "gena-lm-bigbird-base-t2t",
    "GROVER",
    "OmniReg-base",
    "DNABERT-3",
    "DNABERT-4",
    "DNABERT-5",
    "DNABERT-6",
    "nt-500m-human",
    "nt-500m-1000g",
    "nt-2.5b-1000g",
    "nt-2.5b-ms",
    "ntv2-50m-ms-3kmer",
    "ntv2-50m-ms",
    "ntv2-100m-ms",
    "ntv2-250m-ms",
    "ntv2-500m-ms",
    "GENERator-v2-eukaryote-1.2b-base",
    "GENERator-v2-eukaryote-3b-base",
    "GENERator-v2-prokaryote-1.2b-base",
    "GENERator-v2-prokaryote-3b-base",
    None,  # 空行
    "AIDO.RNA-650M",
    "AIDO.RNA-1.6B",
    "AIDO.RNA-650M-CDS",
    "AIDO.RNA-1.6B-CDS",
    "RNA-FM",
    "RiNALMo",
    "BiRNA-BERT",
    None,  # 空行
    "RNABERT",
    "MP-RNA",
]

def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_task_names(summary: dict) -> list:
    task_names = summary.get("training_args", {}).get("task_names")
    if task_names:
        return list(task_names)
    metrics_by_task = summary.get("test_metrics_by_task", {})
    return list(metrics_by_task.keys())


def _normalize_metric_name(metric: str) -> str:
    """规范化指标名称：acc -> accuracy, f1 -> f1_macro"""
    if metric == "acc":
        return "accuracy"
    elif metric == "f1":
        return "f1_macro"
    return metric


def _extract_metrics(summary: dict, metrics: list) -> dict:
    metrics_by_task = summary.get("test_metrics_by_task")
    if metrics_by_task is None:
        raise KeyError("Missing test_metrics_by_task in finetune_summary.json")
    task_names = _get_task_names(summary)
    result = {}
    for task in task_names:
        task_metrics = metrics_by_task.get(task, {})
        task_result = {}
        for metric in metrics:
            # 查找时使用规范化后的名称，但保存时使用原始名称
            normalized_metric = _normalize_metric_name(metric)
            task_result[metric] = task_metrics.get(normalized_metric)
        result[task] = task_result
    return result


def _get_model_order_index(model_name: str) -> int:
    """获取模型在排序列表中的索引，用于排序。不在列表中的模型返回一个很大的数字（排在最后）"""
    try:
        idx = MODEL_ORDER.index(model_name)
        return idx
    except ValueError:
        # 不在列表中的模型，排在最后（使用一个很大的数字）
        return len(MODEL_ORDER) + 1000


def _compute_mean_std(values: list) -> tuple:
    """计算平均值和标准差。如果只有一个值，返回该值和0。如果没有值，返回None, None"""
    if not values:
        return None, None
    # 转为原生 Python float，避免 numpy 等类型导致 statistics.stdev 报错
    try:
        floats = [float(x) for x in values]
    except (ValueError, TypeError):
        return None, None
    if not floats:
        return None, None
    if len(floats) == 1:
        return floats[0], 0.0
    mean = statistics.mean(floats)
    std = statistics.stdev(floats) if len(floats) > 1 else 0.0
    return mean, std


def _format_mean_std(mean: float, std: float, precision: int = 2) -> str:
    """格式化平均值(标准差)为字符串，将数值乘以100转换为百分比形式"""
    if mean is None:
        return ""
    return f"{mean * 100:.{precision}f}({std * 100:.{precision}f})"


def _parse_mean_std(value_str: str) -> tuple:
    """解析 "82.92(12.80)" 格式的字符串，返回 (mean, std) 元组（已除以100）
    
    Args:
        value_str: 格式为 "mean(std)" 的字符串
    
    Returns:
        (mean, std) 元组，如果解析失败返回 (None, None)
    """
    if not value_str or not isinstance(value_str, str):
        return None, None
    try:
        # 匹配 "82.92(12.80)" 格式
        match = re.match(r'^([\d.]+)\(([\d.]+)\)$', value_str.strip())
        if match:
            mean_str, std_str = match.groups()
            mean = float(mean_str) / 100.0  # 转换回 0-1 范围
            std = float(std_str) / 100.0
            return mean, std
    except (ValueError, AttributeError):
        pass
    return None, None


def _compute_task_average(task_values: list) -> str:
    """计算多个任务的平均值
    
    Args:
        task_values: 任务值列表，每个元素是 "mean(std)" 格式的字符串
    
    Returns:
        平均值的字符串表示，格式为 "mean(std)"
    """
    if not task_values:
        return ""
    
    means = []
    stds = []
    for val in task_values:
        if val:
            mean, std = _parse_mean_std(val)
            if mean is not None:
                means.append(mean)
                stds.append(std)
    
    if not means:
        return ""
    
    # 计算平均值
    avg_mean = statistics.mean(means)
    # 对于标准差，使用合并标准差公式（简化版：取平均值）
    avg_std = statistics.mean(stds) if stds else 0.0
    
    return _format_mean_std(avg_mean, avg_std)


def _select_closest_two_lrs(lr_metrics_list: list, task_names: list, metric: str) -> list:
    """选择在所有任务上结果最接近的两个学习率
    
    Args:
        lr_metrics_list: 学习率指标列表，每个元素是 (lr_value, {task: {metric: value}})
        task_names: 任务名称列表
        metric: 指标名称
    
    Returns:
        选定的学习率指标列表（最多2个）
    """
    if len(lr_metrics_list) <= 2:
        return lr_metrics_list
    
    # 为每个学习率计算在所有任务上的平均指标值
    lr_avg_scores = []  # [(index, avg_score, valid_tasks_count), ...]
    for idx, (lr_value, metrics) in enumerate(lr_metrics_list):
        task_scores = []
        for task in task_names:
            task_metrics = metrics.get(task, {})
            metric_val = task_metrics.get(metric)
            if metric_val is not None:
                try:
                    task_scores.append(float(metric_val))
                except (ValueError, TypeError):
                    pass
        
        if task_scores:
            avg_score = statistics.mean(task_scores)
            lr_avg_scores.append((idx, avg_score, len(task_scores)))
    
    if len(lr_avg_scores) < 2:
        # 如果有效学习率少于2个，返回前两个
        return lr_metrics_list[:2]
    
    # 找到平均指标值最接近的两个学习率
    min_distance = float('inf')
    best_pair = None
    for i in range(len(lr_avg_scores)):
        for j in range(i + 1, len(lr_avg_scores)):
            distance = abs(lr_avg_scores[i][1] - lr_avg_scores[j][1])
            if distance < min_distance:
                min_distance = distance
                best_pair = (lr_avg_scores[i][0], lr_avg_scores[j][0])
    
    if best_pair:
        selected = [lr_metrics_list[best_pair[0]], lr_metrics_list[best_pair[1]]]
        return selected
    
    # 如果找不到，返回前两个
    return lr_metrics_list[:2]


def _merge_lr_metrics(all_lr_metrics: list, task_names: list, metric: str) -> dict:
    """合并多个学习率的指标结果，计算平均值(标准差)
    如果学习率数量大于2，会自动选择结果最接近的两个学习率
    
    Args:
        all_lr_metrics: 所有学习率的指标列表，每个元素是 {task: {metric: value}}
        task_names: 任务名称列表
        metric: 指标名称
    
    Returns:
        合并后的指标字典，格式为 {task: {metric: "平均值(标准差)"}}
    """
    merged = {}
    for task in task_names:
        task_values = []
        for lr_metrics in all_lr_metrics:
            task_metrics = lr_metrics.get(task, {})
            metric_val = task_metrics.get(metric)
            if metric_val is not None:
                try:
                    task_values.append(float(metric_val))
                except (ValueError, TypeError):
                    pass
        
        mean, std = _compute_mean_std(task_values)
        if mean is not None:
            merged[task] = {metric: _format_mean_std(mean, std)}
        else:
            merged[task] = {metric: None}
    
    return merged


def _collect_model_rows(results_dir: str, metric: str = "mcc") -> tuple:
    # 第一步：收集所有模型的结果（按模型名称分组）
    model_results = {}  # {model_name: [rows]}
    model_dirs = [
        name for name in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, name))
    ]
    
    task_names = None
    for model_name in model_dirs:
        model_dir = os.path.join(results_dir, model_name)
        # 遍历配置目录: {window_len}_{train_num_windows}_{eval_num_windows}
        config_dirs = [
            name for name in os.listdir(model_dir)
            if os.path.isdir(os.path.join(model_dir, name)) and not name.startswith(".")
        ]
        config_dirs.sort()
        
        model_rows = []
        for config_dir_name in config_dirs:
            config_dir = os.path.join(model_dir, config_dir_name)
            # 跳过 embeddings 目录
            if config_dir_name == "embeddings":
                continue
            # 仅保留 512_8_64 配置
            if config_dir_name != "512_8_64":
                continue
            
            # 遍历 lr 目录，收集所有学习率的结果
            lr_dirs = [
                name for name in os.listdir(config_dir)
                if os.path.isdir(os.path.join(config_dir, name))
            ]
            lr_dirs.sort()
            
            # 收集该配置下所有学习率的结果
            config_lr_metrics = []  # 存储 (lr_value, {task: {metric: value}}) 元组列表
            
            for lr_dir_name in lr_dirs:
                lr_dir = os.path.join(config_dir, lr_dir_name)
                summary_path = os.path.join(lr_dir, "finetune_summary.json")
                if not os.path.exists(summary_path):
                    print(f"[WARN] Missing finetune_summary.json: {summary_path}")
                    continue
                
                try:
                    # 解析学习率值
                    try:
                        lr_value = float(lr_dir_name)
                    except ValueError:
                        # 如果无法解析为浮点数，使用目录名作为标识
                        lr_value = lr_dir_name
                    
                    summary = _load_json(summary_path)
                    metrics = _extract_metrics(summary, [metric])
                    if task_names is None:
                        task_names = _get_task_names(summary)
                    
                    # 保存该学习率的指标结果和学习率值
                    config_lr_metrics.append((lr_value, metrics))
                except Exception as exc:
                    print(f"[WARN] Failed to load {summary_path}: {exc}")
                    continue
            
            # 处理收集到的学习率结果
            if config_lr_metrics:
                # 如果学习率数量大于2，选择结果最接近的两个学习率
                if len(config_lr_metrics) > 2:
                    selected_lr_metrics = _select_closest_two_lrs(config_lr_metrics, task_names, metric)
                    if len(selected_lr_metrics) < len(config_lr_metrics):
                        print(f"[INFO] Model {model_name}/{config_dir_name}: selected {len(selected_lr_metrics)} closest learning rates from {len(config_lr_metrics)} total")
                    config_lr_metrics = selected_lr_metrics
                
                # 提取指标部分（去掉学习率值）
                metrics_only = [metrics for _, metrics in config_lr_metrics]
                # 统一使用平均值(标准差)格式（单个学习率时标准差为0）
                merged_metrics = _merge_lr_metrics(metrics_only, task_names, metric)
                model_display_name = f"{model_name}/{config_dir_name}"
                row = {"model": model_display_name, "metrics": merged_metrics}
                model_rows.append(row)
                if len(config_lr_metrics) > 1:
                    lr_values = [str(lr) for lr, _ in config_lr_metrics]
                    print(f"[INFO] Model {model_name}/{config_dir_name}: merged {len(config_lr_metrics)} learning rates ({', '.join(lr_values)}) into mean(std) format")
        
        if model_rows:
            model_results[model_name] = model_rows
    
    # 第二步：按照 MODEL_ORDER 的顺序输出，遇到 None 时插入空行
    all_rows = []
    for item in MODEL_ORDER:
        if item is None:
            # 插入空行
            all_rows.append(None)
        elif item in model_results:
            # 添加该模型的所有结果
            all_rows.extend(model_results[item])
        else:
            # 模型不存在，但也要预留位置（插入一个只有模型名称的空行）
            placeholder_row = {
                "model": item,  # 只显示模型名称
                "metrics": {}  # 空的指标
            }
            all_rows.append(placeholder_row)
    
    # 第三步：添加不在 MODEL_ORDER 中的模型（排在最后）
    for model_name in sorted(model_dirs):
        if model_name not in MODEL_ORDER and model_name in model_results:
            all_rows.extend(model_results[model_name])

    if task_names is None:
        task_names = []
    
    # 收集未运行的模型（在MODEL_ORDER中但不在model_results中）
    missing_models = []
    for item in MODEL_ORDER:
        if item is not None and item not in model_results:
            missing_models.append(item)
    
    return task_names, all_rows, missing_models


def _build_header(task_names: list, metrics: list, average_tasks: bool = False) -> list:
    header = ["model"]
    use_prefix = len(task_names) > 1
    for task in task_names:
        for metric in metrics:
            col_name = f"{task}_{metric}" if use_prefix else metric
            header.append(col_name)
    
    # 如果启用平均且是多任务，添加平均值列
    if average_tasks and len(task_names) > 1:
        for metric in metrics:
            header.append(f"average_{metric}")
    
    return header


def _build_row(task_names: list, row_data: dict, metrics: list, average_tasks: bool = False) -> list:
    row = [row_data["model"]]
    for task in task_names:
        task_metrics = row_data["metrics"].get(task, {})
        for metric in metrics:
            value = task_metrics.get(metric)
            row.append("" if value is None else value)
    
    # 如果启用平均且是多任务，计算并添加平均值
    if average_tasks and len(task_names) > 1:
        for metric in metrics:
            task_values = []
            for task in task_names:
                task_metrics = row_data["metrics"].get(task, {})
                value = task_metrics.get(metric)
                if value:
                    task_values.append(value)
            avg_value = _compute_task_average(task_values)
            row.append(avg_value)
    
    return row


def _build_empty_row(num_cols: int) -> list:
    """构建空行（所有列都为空字符串）"""
    return [""] * num_cols


def _write_csv(output_path: str, task_names: list, rows: list, metrics: list, average_tasks: bool = False) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    header = _build_header(task_names, metrics, average_tasks)
    num_cols = len(header)
    try:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row_data in rows:
                if row_data is None:
                    # 空行
                    writer.writerow(_build_empty_row(num_cols))
                else:
                    writer.writerow(_build_row(task_names, row_data, metrics, average_tasks))
    except OSError as exc:
        if exc.errno == 122:  # Disk quota exceeded
            raise OSError(f"Disk quota exceeded. Cannot write to: {output_path}")
        else:
            raise


def _export_missing_models_csv(missing_models: list, output_path: str) -> str:
    """导出未运行模型的CSV文件"""
    if not missing_models:
        return ""
    
    try:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["model"])  # 只有一列：模型名称
            for model in missing_models:
                writer.writerow([model])
        print(f"[INFO] Saved missing models CSV to: {output_path}")
        return output_path
    except OSError as exc:
        if "Disk quota exceeded" in str(exc):
            print(f"[ERROR] {exc}")
        else:
            print(f"[ERROR] Failed to write missing models CSV file: {exc}")
        return ""


def _export_task_csv(task: str, metric: str = "mcc", output_path: str = None, average_tasks: bool = False) -> str:
    results_dir = os.path.join(PROJECT_ROOT, SUPPORTED_TASKS[task])
    task_names, rows, missing_models = _collect_model_rows(results_dir, metric)
    if not rows:
        print(f"[WARN] No model results found under: {results_dir}")
        # 即使没有结果，也尝试导出未运行的模型列表
        if missing_models:
            missing_path = os.path.join(results_dir, f"missing_models.csv")
            _export_missing_models_csv(missing_models, missing_path)
        return ""
    if output_path is None:
        output_path = os.path.join(results_dir, f"metrics_{metric}.csv")
    try:
        _write_csv(output_path, task_names, rows, [metric], average_tasks)
        print(f"[INFO] Saved CSV to: {output_path}")
        
        # 同时导出未运行的模型列表
        if missing_models:
            missing_path = os.path.join(results_dir, f"missing_models.csv")
            _export_missing_models_csv(missing_models, missing_path)
        
        return output_path
    except OSError as exc:
        if "Disk quota exceeded" in str(exc):
            print(f"[ERROR] {exc}")
            print(f"[INFO] Please free up disk space and try again.")
        else:
            print(f"[ERROR] Failed to write CSV file: {exc}")
        return ""


# 汇总表列组：(组名, 该组下按 genus/times 区分的 task 列表，顺序：taxon-genus, taxon-times, host-genus, host-times)
SUMMARY_GROUPS = [
    ("ALL", ["ALL-taxon-genus", "ALL-taxon-times", "ALL-host-genus", "ALL-host-times"]),
    ("DNA", ["DNA-taxon-genus", "DNA-taxon-times", "DNA-host-genus", "DNA-host-times"]),
    ("RNA", ["RNA-taxon-genus", "RNA-taxon-times", "RNA-host-genus", "RNA-host-times"]),
]


def _build_summary_table(csv_paths: dict, metric: str):
    """根据各任务 CSV 构建汇总表：列为 Name | ALL Taxon genus | ALL Taxon times | ALL Host genus | ALL Host times | ...（DNA/RNA 同理）。
    每组（ALL/DNA/RNA）下按 Taxon/Host 再按 genus、times 区分，共 12 列。
    """
    import pandas as pd

    # 按 model 汇总各 task 的指标值
    model_to_tasks = {}  # model -> {task: value_str}
    for task in TASK_ORDER:
        path = csv_paths.get(task)
        if not path or not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if "model" not in df.columns or df.empty:
            continue
        # 指标列：除 model 外的第一列（每个 task 的 CSV 只有一列指标）
        metric_col = [c for c in df.columns if c != "model"]
        if not metric_col:
            continue
        metric_col = metric_col[0]
        for _, row in df.iterrows():
            model = row["model"]
            if model not in model_to_tasks:
                model_to_tasks[model] = {}
            val = row.get(metric_col)
            if pd.isna(val):
                val = ""
            else:
                val = str(val).strip()
            model_to_tasks[model][task] = val if val else None

    if not model_to_tasks:
        return None

    # 表头：Name, ALL Taxon genus, ALL Taxon times, ALL Host genus, ALL Host times, DNA ..., RNA ...
    summary_cols = ["Name"]
    for group_name, task_list in SUMMARY_GROUPS:
        # task_list 顺序：taxon-genus, taxon-times, host-genus, host-times
        summary_cols.append(f"{group_name} Taxon genus")
        summary_cols.append(f"{group_name} Taxon times")
        summary_cols.append(f"{group_name} Host genus")
        summary_cols.append(f"{group_name} Host times")

    rows = []
    # 保持与主表一致的顺序：从第一个有数据的 task 的 CSV 取 model 顺序（含空行）
    model_order = []
    for task in TASK_ORDER:
        path = csv_paths.get(task)
        if not path or not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if "model" not in df.columns or df.empty:
            continue
        model_order = df["model"].astype(str).tolist()
        break
    if not model_order:
        model_order = sorted(model_to_tasks.keys())

    for model in model_order:
        model_str = str(model).strip() if model is not None and pd.notna(model) else ""
        if model_str == "nan":
            model_str = ""
        task_vals = model_to_tasks.get(model, {}) if model_str else {}
        name_display = model_str if model_str else ""
        row = [name_display]
        for _group_name, task_list in SUMMARY_GROUPS:
            for task in task_list:
                row.append(task_vals.get(task) or "")
        rows.append(row)

    return pd.DataFrame(rows, columns=summary_cols)


def _export_all_tasks_xlsx(xlsx_path: str, metric: str = "mcc", average_tasks: bool = False) -> int:
    try:
        import pandas as pd
    except Exception as exc:
        print(f"[ERROR] pandas is required for --task all. {exc}")
        return 1

    os.makedirs(os.path.dirname(xlsx_path), exist_ok=True)
    csv_paths = {}
    missing_models_by_task = {}  # {task: [missing_models]}

    # 收集所有任务的未运行模型
    for task in TASK_ORDER:
        results_dir = os.path.join(PROJECT_ROOT, SUPPORTED_TASKS[task])
        if os.path.exists(results_dir):
            _, _, missing_models = _collect_model_rows(results_dir, metric)
            if missing_models:
                missing_models_by_task[task] = missing_models

        csv_path = _export_task_csv(task, metric, average_tasks=average_tasks)
        if csv_path:
            csv_paths[task] = csv_path

    if not csv_paths and not missing_models_by_task:
        print("[WARN] No CSV files generated and no missing models found.")
        return 1

    writer = None
    try:
        writer = pd.ExcelWriter(xlsx_path)

        # 写入每个任务的结果
        for task in TASK_ORDER:
            csv_path = csv_paths.get(task)
            if not csv_path:
                continue
            df = pd.read_csv(csv_path)
            df.to_excel(writer, sheet_name=task, index=False)

        # 写入汇总表：ALL / DNA / RNA 下按 Taxon/Host 再按 genus、times 区分的 12 列
        summary_df = _build_summary_table(csv_paths, metric)
        if summary_df is not None and not summary_df.empty:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            print("[INFO] Saved Summary sheet (ALL/DNA/RNA × Taxon/Host × genus/times).")
        
        # 写入未运行模型的汇总表
        if missing_models_by_task:
            missing_data = []
            for task in TASK_ORDER:
                missing_models = missing_models_by_task.get(task, [])
                if missing_models:
                    for model in missing_models:
                        missing_data.append({"task": task, "model": model})
            
            if missing_data:
                missing_df = pd.DataFrame(missing_data)
                missing_df.to_excel(writer, sheet_name="missing_models", index=False)
                print(f"[INFO] Found {len(missing_data)} missing model entries across {len(missing_models_by_task)} tasks")
        
        # 手动关闭，避免在 with 语句退出时出错
        # ExcelWriter 的 close() 方法会自动保存
        try:
            writer.close()
        except OSError as close_exc:
            if close_exc.errno == 122:  # Disk quota exceeded
                raise close_exc
            # 其他错误也抛出
            raise
        
        print(f"[INFO] Saved XLSX to: {xlsx_path}")
        return 0
    except OSError as exc:
        if exc.errno == 122:  # Disk quota exceeded
            print(f"[ERROR] Disk quota exceeded. Cannot write to: {xlsx_path}")
            # 尝试清理 writer 资源
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass  # 忽略清理错误
            # 尝试清理可能创建的不完整文件
            if os.path.exists(xlsx_path):
                try:
                    os.remove(xlsx_path)
                    print(f"[INFO] Removed incomplete XLSX file: {xlsx_path}")
                except OSError:
                    pass  # 如果删除也失败，忽略
            print(f"[INFO] CSV files have been generated successfully:")
            for task, csv_path in csv_paths.items():
                print(f"  - {task}: {csv_path}")
            print(f"[INFO] You can manually combine these CSV files or free up disk space and try again.")
            return 1
        else:
            print(f"[ERROR] Failed to write XLSX file: {exc}")
            # 尝试清理 writer 资源
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass
            # 尝试清理可能创建的不完整文件
            if os.path.exists(xlsx_path):
                try:
                    os.remove(xlsx_path)
                except OSError:
                    pass
            return 1
    except Exception as exc:
        print(f"[ERROR] Unexpected error while writing XLSX file: {exc}")
        # 尝试清理 writer 资源
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        # 尝试清理可能创建的不完整文件
        if os.path.exists(xlsx_path):
            try:
                os.remove(xlsx_path)
            except OSError:
                pass
        print(f"[INFO] CSV files have been generated successfully:")
        for task, csv_path in csv_paths.items():
            print(f"  - {task}: {csv_path}")
        return 1
    finally:
        # 确保 writer 被清理，抑制清理时的错误输出
        if writer is not None:
            try:
                # 使用 contextlib 抑制 stderr 输出，避免显示清理错误
                with open(os.devnull, 'w') as devnull:
                    old_stderr = sys.stderr
                    try:
                        sys.stderr = devnull
                        if hasattr(writer, 'close'):
                            writer.close()
                    except Exception:
                        pass  # 完全忽略清理错误
                    finally:
                        sys.stderr = old_stderr
            except Exception:
                pass  # 完全忽略所有清理错误


def _suppress_cleanup_errors():
    """抑制清理时的错误输出（Exception ignored 相关）"""
    original_stderr = sys.stderr
    _suppressing = False
    _buffer = []
    
    class SuppressCleanupErrors:
        def write(self, text):
            nonlocal _suppressing
            
            # 检测到 "Exception ignored" 开始抑制
            if "Exception ignored" in text:
                _suppressing = True
                _buffer.clear()
                return  # 不输出这一行
            
            # 如果正在抑制，检查是否是清理相关的错误
            if _suppressing:
                _buffer.append(text)
                # 检查是否包含磁盘配额错误
                full_text = "".join(_buffer)
                if ("Disk quota exceeded" in full_text or "Errno 122" in full_text) and \
                   ("deallocator" in full_text or "finalizing file" in full_text):
                    # 这是清理错误，完全忽略
                    _buffer.clear()
                    _suppressing = False
                    return
                # 如果遇到空行且缓冲区足够长，可能是异常块结束
                if text.strip() == "" and len(_buffer) > 5:
                    _buffer.clear()
                    _suppressing = False
                    return
                # 继续抑制
                return
            
            # 正常输出
            original_stderr.write(text)
        
        def flush(self):
            original_stderr.flush()
        
        def __getattr__(self, name):
            return getattr(original_stderr, name)
    
    sys.stderr = SuppressCleanupErrors()


def main() -> int:
    # 抑制清理时的错误输出
    _suppress_cleanup_errors()
    
    parser = argparse.ArgumentParser(description="Load finetune results and export CSV.")
    parser.add_argument(
        "--task",
        required=False,
        default="all",
        choices=sorted(list(SUPPORTED_TASKS.keys()) + ["all"]),
        help="Task name to load (one at a time). Default: all",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="f1",
        choices=["mcc", "accuracy", "acc", "f1_macro", "f1", "auprc_macro_ovr", "precision_macro", "recall_macro"],
        help="指标名称（默认 f1_macro，可选：mcc, accuracy/acc, f1_macro/f1, auprc_macro_ovr, precision_macro, recall_macro）",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (single task) or XLSX path (all tasks).",
    )
    parser.add_argument(
        "--average-tasks",
        action="store_true",
        help="If set, compute and add average column for multi-task results.",
    )
    args = parser.parse_args()

    # 规范化指标名称
    metric = _normalize_metric_name(args.metric)

    if args.task == "all":
        if args.output is None:
            output_path = os.path.join(PROJECT_ROOT, "results", f"all_metrics_{metric}.xlsx")
        else:
            output_path = args.output
        return _export_all_tasks_xlsx(output_path, metric, average_tasks=args.average_tasks)

    _export_task_csv(args.task, metric, args.output, average_tasks=args.average_tasks)
    return 0


if __name__ == "__main__":
    sys.exit(main())

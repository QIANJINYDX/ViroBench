#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 指标名称将从 all_statistics 中动态提取（所有指标的 mean 值）
# 不预先定义 METRICS，而是从数据中动态获取
SUPPORTED_TASKS = {
    "cds-long": "results/Generate/cds-long",
    "cds-short": "results/Generate/cds-short",
    "cds-medium": "results/Generate/cds-medium",
}
TASK_ORDER = [
    "cds-long",
    "cds-short",
    "cds-medium",
]

# 模型显示顺序（None 表示空行分隔）
MODEL_ORDER = [
    "evo2_1b_base",
    "evo2_7b_base",
    "evo2_7b",
    "evo2_40b_base",
    "evo2_40b",
    None,  # 空行
    "evo-1-8k-base",
    "evo-1-131k-base",
    None,  # 空行
    "evo-1.5-8k-base",
    None,  # 空行
    "hyenadna-tiny-16k",
    "hyenadna-tiny-1k",
    "hyenadna-small-32k",
    "hyenadna-medium-160k",
    "hyenadna-medium-450k",
    "hyenadna-large-1m",
    None,  # 空行
    "Genos-1.2B",
    "Genos-10B",
    "Genos-10B-v2",
    None,  # 空行
    "OmniReg-bigbird",
    "OmniReg-base",
    "OmniReg-large",
    None,  # 空行
    "GenomeOcean-100M",
    "GenomeOcean-500M",
    "GenomeOcean-4B",
    None,  # 空行
    "GENERator-v2-eukaryote-1.2b-base",
    "GENERator-v2-eukaryote-3b-base",
    "GENERator-v2-prokaryote-1.2b-base",
    "GENERator-v2-prokaryote-3b-base",
    None,  # 空行
    "ViroHyena-436k",
    "ViroHyena-1m",
    "ViroHyena-6m",
    "ViroHyena-253m",
]


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_metrics(summary: dict) -> dict:
    """从 gen_summary.json 中提取所有指标的 mean 值"""
    all_statistics = summary.get("all_statistics")
    if all_statistics is None:
        raise KeyError("Missing all_statistics in gen_summary.json")
    
    # 提取所有指标的 mean 值（排除 count 和 valid_count）
    metrics = {}
    for metric_name, metric_data in all_statistics.items():
        if metric_name in ["count", "valid_count"]:
            continue  # 跳过 count 和 valid_count
        if isinstance(metric_data, dict) and "mean" in metric_data:
            metrics[metric_name] = metric_data["mean"]
    
    return metrics


def _is_slice_directory(dir_name: str) -> bool:
    """判断目录名是否是切片目录（纯数字）"""
    return dir_name.isdigit()


def _load_model_metrics(model_dir: str) -> dict:
    """加载模型指标，支持切片形式和非切片形式"""
    # 首先检查是否有切片目录
    slice_dirs = []
    for item in os.listdir(model_dir):
        item_path = os.path.join(model_dir, item)
        if os.path.isdir(item_path) and _is_slice_directory(item):
            slice_dirs.append(item)
    
    if slice_dirs:
        # 有切片目录，加载所有切片并合并
        slice_dirs.sort(key=int)  # 按数字排序
        all_metrics_list = []
        
        for slice_dir in slice_dirs:
            slice_path = os.path.join(model_dir, slice_dir)
            summary_path = os.path.join(slice_path, "gen_summary.json")
            
            if not os.path.exists(summary_path):
                print(f"[WARN] Missing gen_summary.json in slice: {summary_path}")
                continue
            
            try:
                summary = _load_json(summary_path)
                metrics = _extract_metrics(summary)
                all_metrics_list.append(metrics)
            except Exception as exc:
                print(f"[WARN] Failed to load slice {summary_path}: {exc}")
                continue
        
        if not all_metrics_list:
            raise ValueError(f"No valid slice results found in {model_dir}")
        
        # 合并所有切片的指标（计算平均值）
        merged_metrics = {}
        all_metric_names = set()
        for metrics in all_metrics_list:
            all_metric_names.update(metrics.keys())
        
        for metric_name in all_metric_names:
            values = []
            for metrics in all_metrics_list:
                if metric_name in metrics:
                    value = metrics[metric_name]
                    if value is not None:
                        values.append(value)
            
            if values:
                merged_metrics[metric_name] = sum(values) / len(values)
            else:
                merged_metrics[metric_name] = None
        
        return merged_metrics
    else:
        # 没有切片目录，直接加载模型目录下的 gen_summary.json
        summary_path = os.path.join(model_dir, "gen_summary.json")
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"gen_summary.json not found: {summary_path}")
        
        summary = _load_json(summary_path)
        return _extract_metrics(summary)


def _collect_model_rows(results_dir: str) -> tuple:
    """收集所有模型的结果"""
    model_results = {}  # {model_name: [rows]}
    model_dirs = [
        name for name in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, name)) and not name.startswith(".")
    ]
    
    # 收集所有指标名称（用于确定表头）
    all_metric_names = set()
    
    for model_name in model_dirs:
        model_dir = os.path.join(results_dir, model_name)
        
        try:
            metrics = _load_model_metrics(model_dir)
            
            # 收集指标名称
            all_metric_names.update(metrics.keys())
            
            # 模型名称直接使用模型名称（没有 config 和 lr 子目录）
            row = {"model": model_name, "metrics": metrics}
            
            if model_name not in model_results:
                model_results[model_name] = []
            model_results[model_name].append(row)
        except Exception as exc:
            print(f"[WARN] Failed to load metrics from {model_dir}: {exc}")
            continue
    
    # 按照 MODEL_ORDER 的顺序输出，遇到 None 时插入空行
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
    
    # 添加不在 MODEL_ORDER 中的模型（排在最后）
    for model_name in sorted(model_dirs):
        if model_name not in MODEL_ORDER and model_name in model_results:
            all_rows.extend(model_results[model_name])

    # 返回指标名称列表（排序以便保持一致性）
    metric_names = sorted(list(all_metric_names))
    return metric_names, all_rows


def _build_header(metric_names: list) -> list:
    """构建 CSV 表头"""
    header = ["model"]
    header.extend(metric_names)
    return header


def _build_row(metric_names: list, row_data: dict) -> list:
    """构建 CSV 行数据"""
    row = [row_data["model"]]
    metrics = row_data.get("metrics", {})
    for metric_name in metric_names:
        value = metrics.get(metric_name)
        row.append("" if value is None else value)
    return row


def _build_empty_row(num_cols: int) -> list:
    """构建空行（所有列都为空字符串）"""
    return [""] * num_cols


def _write_csv(output_path: str, metric_names: list, rows: list) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    header = _build_header(metric_names)
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
                    writer.writerow(_build_row(metric_names, row_data))
    except OSError as exc:
        if exc.errno == 122:  # Disk quota exceeded
            raise OSError(f"Disk quota exceeded. Cannot write to: {output_path}")
        else:
            raise


def _export_task_csv(task: str, output_path: str = None) -> str:
    results_dir = os.path.join(PROJECT_ROOT, SUPPORTED_TASKS[task])
    metric_names, rows = _collect_model_rows(results_dir)
    if not rows:
        print(f"[WARN] No model results found under: {results_dir}")
        return ""
    if output_path is None:
        output_path = os.path.join(results_dir, "gen_metrics.csv")
    try:
        _write_csv(output_path, metric_names, rows)
        print(f"[INFO] Saved CSV to: {output_path}")
        return output_path
    except OSError as exc:
        if "Disk quota exceeded" in str(exc):
            print(f"[ERROR] {exc}")
            print(f"[INFO] Please free up disk space and try again.")
        else:
            print(f"[ERROR] Failed to write CSV file: {exc}")
        return ""


def _export_all_tasks_xlsx(xlsx_path: str) -> int:
    """导出所有任务到 XLSX 文件"""
    try:
        import pandas as pd
    except Exception as exc:
        print(f"[ERROR] pandas is required for --task all. {exc}")
        return 1

    os.makedirs(os.path.dirname(xlsx_path), exist_ok=True)
    csv_paths = {}
    for task in TASK_ORDER:
        csv_path = _export_task_csv(task)
        if csv_path:
            csv_paths[task] = csv_path

    if not csv_paths:
        print("[WARN] No CSV files generated.")
        return 1

    writer = None
    try:
        writer = pd.ExcelWriter(xlsx_path)
        for task in TASK_ORDER:
            csv_path = csv_paths.get(task)
            if not csv_path:
                continue
            df = pd.read_csv(csv_path)
            df.to_excel(writer, sheet_name=task, index=False)
        
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


def _load_jsonl(path: str) -> list:
    """加载 JSONL 文件"""
    data = []
    if not os.path.exists(path):
        return data
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except Exception as exc:
        print(f"[WARN] Failed to load JSONL from {path}: {exc}")
    return data


def _load_model_per_sample_data(model_dir: str) -> list:
    """加载模型的所有 per-sample 数据，支持切片形式和非切片形式"""
    # 首先检查是否有切片目录
    slice_dirs = []
    for item in os.listdir(model_dir):
        item_path = os.path.join(model_dir, item)
        if os.path.isdir(item_path) and _is_slice_directory(item):
            slice_dirs.append(item)
    
    all_samples = []
    
    if slice_dirs:
        # 有切片目录，加载所有切片
        slice_dirs.sort(key=int)  # 按数字排序
        for slice_dir in slice_dirs:
            slice_path = os.path.join(model_dir, slice_dir)
            jsonl_path = os.path.join(slice_path, "all_gen_per_sample.jsonl")
            
            if not os.path.exists(jsonl_path):
                print(f"[WARN] Missing all_gen_per_sample.jsonl in slice: {jsonl_path}")
                continue
            
            samples = _load_jsonl(jsonl_path)
            all_samples.extend(samples)
    else:
        # 没有切片目录，直接加载模型目录下的 all_gen_per_sample.jsonl
        jsonl_path = os.path.join(model_dir, "all_gen_per_sample.jsonl")
        if os.path.exists(jsonl_path):
            all_samples = _load_jsonl(jsonl_path)
        else:
            print(f"[WARN] Missing all_gen_per_sample.jsonl: {jsonl_path}")
    
    return all_samples


def _export_all_per_sample_jsonl(output_path: str) -> int:
    """导出所有模型的所有 per-sample 数据到一个大的 JSONL 文件"""
    try:
        from tqdm import tqdm
    except ImportError:
        print("[ERROR] tqdm is required for progress bar. Please install it: pip install tqdm")
        return 1
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    task_to_group = {
        "cds-short": "short",
        "cds-medium": "medium",
        "cds-long": "long",
    }
    
    # 第一步：收集所有需要处理的模型-任务组合
    model_task_list = []  # 存储 (model_name, task, group, model_dir) 的列表
    
    for task in TASK_ORDER:
        group = task_to_group.get(task, task)
        results_dir = os.path.join(PROJECT_ROOT, SUPPORTED_TASKS[task])
        
        if not os.path.exists(results_dir):
            print(f"[WARN] Task directory not found: {results_dir}")
            continue
        
        model_dirs = [
            name for name in os.listdir(results_dir)
            if os.path.isdir(os.path.join(results_dir, name)) and not name.startswith(".")
        ]
        
        for model_name in sorted(model_dirs):
            model_dir = os.path.join(results_dir, model_name)
            model_task_list.append((model_name, task, group, model_dir))
    
    if not model_task_list:
        print("[WARN] No models found to process.")
        return 1
    
    # 第二步：加载所有数据并显示进度条
    print("[INFO] Loading all samples...")
    model_task_data = []  # 存储 (model_name, task, group, samples) 的列表
    
    with tqdm(total=len(model_task_list), desc="Loading samples", unit="models", ncols=100) as load_pbar:
        for model_name, task, group, model_dir in model_task_list:
            samples = _load_model_per_sample_data(model_dir)
            
            if samples:
                model_task_data.append((model_name, task, group, samples))
            else:
                print(f"[WARN] No samples found for model {model_name} in task {task}")
            
            load_pbar.set_postfix_str(f"{model_name} ({task})")
            load_pbar.update(1)
    
    total_samples = sum(len(samples) for _, _, _, samples in model_task_data)
    
    if total_samples == 0:
        print("[WARN] No samples to export.")
        return 1
    
    print(f"[INFO] Found {total_samples} total samples from {len(model_task_data)} model-task combinations")
    print(f"[INFO] Writing to {output_path}...")
    
    # 第三步：写入数据并显示进度条，同时检查三个字段是否相等
    length_mismatches = []  # 存储 ground_truth 和生成序列不等长的记录信息
    field_mismatches = []  # 存储三个生成字段不相等的记录信息
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            with tqdm(total=total_samples, desc="Writing samples", unit="samples", ncols=100) as pbar:
                for model_name, task, group, samples in model_task_data:
                    # 为每条记录添加 model_name 和 group 字段，并选择需要的字段
                    for sample in samples:
                        ground_truth = sample.get("ground_truth", "")
                        generated_full = sample.get("generated_full", "")
                        generated_continuation = sample.get("generated_continuation", "")
                        generated_tail = sample.get("generated_tail", "")
                        
                        # 检查三个生成字段是否相等
                        all_equal = (
                            generated_full == generated_continuation == generated_tail
                            and generated_full != ""  # 确保不是都为空
                        )
                        
                        if not all_equal:
                            # 记录不相等的情况
                            field_mismatches.append({
                                "model_name": model_name,
                                "task": task,
                                "group": group,
                                "sequence_index": sample.get("sequence_index"),
                                "taxid": sample.get("taxid"),
                            })
                        
                        # 如果三个字段相等，使用 generated_sequence；否则使用 generated_continuation
                        if all_equal:
                            generated_sequence = generated_full
                        else:
                            if model_name == "evo-1.5-8k-base":
                                generated_sequence = generated_full
                            else:
                                generated_sequence = generated_continuation  # 默认使用 generated_continuation
                        
                        # 截断逻辑：如果生成的序列长度超过 ground_truth，则截断到 ground_truth 的长度
                        gt_len = len(ground_truth) if ground_truth else 0
                        gen_len = len(generated_sequence) if generated_sequence else 0
                        
                        if gt_len > 0 and gen_len > gt_len:
                            # 截断到 ground_truth 的长度
                            generated_sequence = generated_sequence[:gt_len]
                            gen_len = gt_len  # 更新长度
                        
                        # 检查 ground_truth 和生成序列的长度是否相等（截断后）
                        if gt_len != gen_len:
                            length_mismatches.append({
                                "model_name": model_name,
                                "task": task,
                                "group": group,
                                "sequence_index": sample.get("sequence_index"),
                                "taxid": sample.get("taxid"),
                                "ground_truth_length": gt_len,
                                "generated_sequence_length": gen_len,
                                "length_diff": abs(gt_len - gen_len),
                            })
                        
                        output_sample = {
                            "sequence_index": sample.get("sequence_index"),
                            "taxid": sample.get("taxid"),
                            "model_name": model_name,
                            "prompt": sample.get("prompt"),
                            "ground_truth": ground_truth,
                            "generated_sequence": generated_sequence,
                            # "exact_match_acc": sample.get("exact_match_acc"),
                            # "edit_distance": sample.get("edit_distance"),
                            # "alignment_identity": sample.get("alignment_identity"),
                            # "kmer_KS": sample.get("kmer_KS"),
                            # "kmer_JSD": sample.get("kmer_JSD"),
                            # "kmer_EMD": sample.get("kmer_EMD"),
                            "group": group,
                        }
                        
                        f.write(json.dumps(output_sample, ensure_ascii=False) + "\n")
                        pbar.update(1)
                    
                    # 更新进度条描述，显示当前处理的模型
                    pbar.set_postfix_str(f"{model_name} ({task})")
        
        print(f"[INFO] Successfully saved {total_samples} total samples to: {output_path}")
        
        # 输出三个生成字段不相等情况的统计信息
        if field_mismatches:
            print(f"\n[WARN] Found {len(field_mismatches)} samples where generated_full, generated_continuation, and generated_tail are not all equal:")
            
            # 按模型分组统计
            field_model_stats = {}
            for mismatch in field_mismatches:
                model_key = (mismatch["model_name"], mismatch["task"], mismatch["group"])
                if model_key not in field_model_stats:
                    field_model_stats[model_key] = {"count": 0}
                field_model_stats[model_key]["count"] += 1
            
            # 按模型输出统计信息
            print("\nField mismatch statistics by model (where three fields are not equal):")
            print("-" * 100)
            print(f"{'Model Name':<40} {'Task':<15} {'Group':<10} {'Count':<10}")
            print("-" * 100)
            
            for (model_name, task, group), stats in sorted(field_model_stats.items()):
                print(
                    f"{model_name:<40} {task:<15} {group:<10} "
                    f"{stats['count']:<10}"
                )
            
            print("-" * 100)
            print(f"Total field mismatches: {len(field_mismatches)} out of {total_samples} samples "
                  f"({len(field_mismatches)/total_samples*100:.2f}%)")
        else:
            print(f"[INFO] All {total_samples} samples have equal generated_full, generated_continuation, and generated_tail fields.")
        
        # 输出长度不匹配的统计信息
        if length_mismatches:
            print(f"\n[WARN] Found {len(length_mismatches)} samples with length mismatches between ground_truth and generated_sequence:")
            
            # 按模型分组统计
            model_stats = {}
            for mismatch in length_mismatches:
                model_key = (mismatch["model_name"], mismatch["task"], mismatch["group"])
                if model_key not in model_stats:
                    model_stats[model_key] = {
                        "count": 0,
                        "total_diff": 0,
                        "max_diff": 0,
                    }
                model_stats[model_key]["count"] += 1
                model_stats[model_key]["total_diff"] += mismatch["length_diff"]
                model_stats[model_key]["max_diff"] = max(
                    model_stats[model_key]["max_diff"], 
                    mismatch["length_diff"]
                )
            
            # 按模型输出统计信息
            print("\nLength mismatch statistics by model:")
            print("-" * 100)
            print(f"{'Model Name':<40} {'Task':<15} {'Group':<10} {'Count':<10} {'Avg Diff':<12} {'Max Diff':<10}")
            print("-" * 100)
            
            for (model_name, task, group), stats in sorted(model_stats.items()):
                avg_diff = stats["total_diff"] / stats["count"] if stats["count"] > 0 else 0
                print(
                    f"{model_name:<40} {task:<15} {group:<10} "
                    f"{stats['count']:<10} {avg_diff:<12.2f} {stats['max_diff']:<10}"
                )
            
            print("-" * 100)
            print(f"Total length mismatches: {len(length_mismatches)} out of {total_samples} samples "
                  f"({len(length_mismatches)/total_samples*100:.2f}%)")
        else:
            print(f"[INFO] All {total_samples} samples have matching ground_truth and generated_sequence lengths.")
        
        return 0
    except OSError as exc:
        if exc.errno == 122:  # Disk quota exceeded
            print(f"[ERROR] Disk quota exceeded. Cannot write to: {output_path}")
            return 1
        else:
            print(f"[ERROR] Failed to write JSONL file: {exc}")
            return 1
    except Exception as exc:
        print(f"[ERROR] Unexpected error while writing JSONL file: {exc}")
        return 1


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
    
    parser = argparse.ArgumentParser(description="Load Generate results and export CSV or JSONL.")
    parser.add_argument(
        "--task",
        required=False,
        default="all",
        choices=sorted(list(SUPPORTED_TASKS.keys()) + ["all"]),
        help="Task name to load (one at a time). Default: all",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (single task) or XLSX path (all tasks).",
    )
    parser.add_argument(
        "--export-jsonl",
        action="store_true",
        help="Export all per-sample data to a single JSONL file instead of CSV/XLSX.",
    )
    parser.add_argument(
        "--jsonl-output",
        default=None,
        help="Output JSONL path (only used with --export-jsonl). Default: results/Generate/all_gen_per_sample.jsonl",
    )
    args = parser.parse_args()

    # 如果指定了 --export-jsonl，生成合并的 JSONL 文件
    if args.export_jsonl:
        if args.jsonl_output is None:
            output_path = os.path.join(PROJECT_ROOT, "results", "Generate", "all_gen_per_sample.jsonl")
        else:
            output_path = args.jsonl_output
        return _export_all_per_sample_jsonl(output_path)

    # 否则，按原来的逻辑生成 CSV/XLSX
    if args.task == "all":
        if args.output is None:
            output_path = os.path.join(PROJECT_ROOT, "results", "Generate", "all_gen_metrics.xlsx")
        else:
            output_path = args.output
        return _export_all_tasks_xlsx(output_path)

    _export_task_csv(args.task, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

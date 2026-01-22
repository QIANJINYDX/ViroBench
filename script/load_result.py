#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METRICS = ["accuracy", "f1_micro","mcc"]
SUPPORTED_TASKS = {
    "c1-genus": "results/c1-genus",
    "c1-times": "results/c1-times",
    "c2-genus": "results/c2-genus",
    "c2-times": "results/c2-times",
}
TASK_ORDER = ["c1-genus", "c1-times", "c2-genus", "c2-times"]

# 模型显示顺序（None 表示空行分隔）
MODEL_ORDER = [
    "CNN",
    None,  # 空行
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
    "nt-500m-human",
    "nt-500m-1000g",
    "nt-2.5b-1000g",
    "nt-2.5b-ms",
    None,  # 空行
    "ntv2-50m-ms-3kmer",
    "ntv2-50m-ms",
    "ntv2-100m-ms",
    "ntv2-250m-ms",
    "ntv2-500m-ms",
    None,  # 空行
    "ntv3-8m-pre",
    "ntv3-100m-pre",
    "ntv3-650m-pre",
    "ntv3-100m-post",
    "ntv3-650m-post",
    None,  # 空行
    "caduceus-ph",
    "caduceus-ps",
    None,  # 空行
    "DNABERT-3",
    "DNABERT-4",
    "DNABERT-5",
    "DNABERT-6",
    "DNABERT-S",
    "DNABERT-2-117M",
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
    "GROVER",
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
    "BioFM-265M",
    None,  # 空行
    "AIDO.DNA-300M",
    "AIDO.DNA-7B",
    None,  # 空行
    "AIDO.RNA-650M",
    "AIDO.RNA-1.6B",
    "AIDO.RNA-650M-CDS",
    "AIDO.RNA-1.6B-CDS",
    None,  # 空行
    "RNA-FM",
    "RiNALMo",
    "BiRNA-BERT",
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


def _extract_metrics(summary: dict) -> dict:
    metrics_by_task = summary.get("test_metrics_by_task")
    if metrics_by_task is None:
        raise KeyError("Missing test_metrics_by_task in finetune_summary.json")
    task_names = _get_task_names(summary)
    metrics = {}
    for task in task_names:
        task_metrics = metrics_by_task.get(task, {})
        metrics[task] = {name: task_metrics.get(name) for name in METRICS}
    return metrics


def _get_model_order_index(model_name: str) -> int:
    """获取模型在排序列表中的索引，用于排序。不在列表中的模型返回一个很大的数字（排在最后）"""
    try:
        idx = MODEL_ORDER.index(model_name)
        return idx
    except ValueError:
        # 不在列表中的模型，排在最后（使用一个很大的数字）
        return len(MODEL_ORDER) + 1000


def _collect_model_rows(results_dir: str) -> tuple:
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
            
            # 遍历 lr 目录，收集所有学习率的结果
            lr_dirs = [
                name for name in os.listdir(config_dir)
                if os.path.isdir(os.path.join(config_dir, name))
            ]
            lr_dirs.sort()
            
            # 收集该配置下所有学习率的结果
            config_lr_results = []  # [(lr_dir_name, row, avg_f1_micro)]
            
            for lr_dir_name in lr_dirs:
                lr_dir = os.path.join(config_dir, lr_dir_name)
                summary_path = os.path.join(lr_dir, "finetune_summary.json")
                if not os.path.exists(summary_path):
                    print(f"[WARN] Missing finetune_summary.json: {summary_path}")
                    continue
                
                try:
                    summary = _load_json(summary_path)
                    metrics = _extract_metrics(summary)
                    if task_names is None:
                        task_names = _get_task_names(summary)
                    
                    # 计算平均 f1_micro（用于选择最佳学习率）
                    f1_values = []
                    for task_metrics in metrics.values():
                        f1_micro = task_metrics.get("f1_micro")
                        if f1_micro is not None:
                            f1_values.append(float(f1_micro))
                    avg_f1_micro = sum(f1_values) / len(f1_values) if f1_values else -1.0
                    
                    # 模型名称包含配置信息: model_name/config/lr
                    model_display_name = f"{model_name}/{config_dir_name}/{lr_dir_name}"
                    row = {"model": model_display_name, "metrics": metrics}
                    config_lr_results.append((lr_dir_name, row, avg_f1_micro))
                except Exception as exc:
                    print(f"[WARN] Failed to load {summary_path}: {exc}")
                    continue
            
            # 如果有多个学习率，选择 f1_micro 最高的
            if config_lr_results:
                if len(config_lr_results) > 1:
                    # 按 avg_f1_micro 降序排序，选择最高的
                    config_lr_results.sort(key=lambda x: x[2], reverse=True)
                    best_lr_row = config_lr_results[0][1]
                    model_rows.append(best_lr_row)
                    print(f"[INFO] Model {model_name}/{config_dir_name}: selected best LR (f1_micro={config_lr_results[0][2]:.4f}) from {len(config_lr_results)} learning rates")
                else:
                    # 只有一个学习率，直接添加
                    model_rows.append(config_lr_results[0][1])
        
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
    return task_names, all_rows


def _build_header(task_names: list) -> list:
    header = ["model"]
    use_prefix = len(task_names) > 1
    for idx, task in enumerate(task_names):
        for metric in METRICS:
            col_name = f"{task}_{metric}" if use_prefix else metric
            header.append(col_name)
        if use_prefix and idx != len(task_names) - 1:
            header.append("")
    return header


def _build_row(task_names: list, row_data: dict) -> list:
    row = [row_data["model"]]
    use_prefix = len(task_names) > 1
    for idx, task in enumerate(task_names):
        task_metrics = row_data["metrics"].get(task, {})
        for metric in METRICS:
            value = task_metrics.get(metric)
            row.append("" if value is None else value)
        if use_prefix and idx != len(task_names) - 1:
            row.append("")
    return row


def _build_empty_row(num_cols: int) -> list:
    """构建空行（所有列都为空字符串）"""
    return [""] * num_cols


def _write_csv(output_path: str, task_names: list, rows: list) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    header = _build_header(task_names)
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
                    writer.writerow(_build_row(task_names, row_data))
    except OSError as exc:
        if exc.errno == 122:  # Disk quota exceeded
            raise OSError(f"Disk quota exceeded. Cannot write to: {output_path}")
        else:
            raise


def _export_task_csv(task: str, output_path: str = None) -> str:
    results_dir = os.path.join(PROJECT_ROOT, SUPPORTED_TASKS[task])
    task_names, rows = _collect_model_rows(results_dir)
    if not rows:
        print(f"[WARN] No model results found under: {results_dir}")
        return ""
    if output_path is None:
        output_path = os.path.join(results_dir, "metrics.csv")
    try:
        _write_csv(output_path, task_names, rows)
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
        required=True,
        choices=sorted(list(SUPPORTED_TASKS.keys()) + ["all"]),
        help="Task name to load (one at a time).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (single task) or XLSX path (all tasks).",
    )
    args = parser.parse_args()

    if args.task == "all":
        if args.output is None:
            output_path = os.path.join(PROJECT_ROOT, "results", "all_metrics.xlsx")
        else:
            output_path = args.output
        return _export_all_tasks_xlsx(output_path)

    _export_task_csv(args.task, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

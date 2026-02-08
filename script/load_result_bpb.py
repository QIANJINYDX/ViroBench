#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METRICS = ["min", "max", "median","mean"]
SUPPORTED_TASKS = {
    "genome-long": "results/Bpb/genome-long",
    "genome-short": "results/Bpb/genome-short",
    "genome-medium": "results/Bpb/genome-medium",
}
TASK_ORDER = [
    "genome-long",
    "genome-short",
    "genome-medium",
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
    """从 bpb_summary.json 中提取指标"""
    all_statistics = summary.get("all_statistics")
    if all_statistics is None:
        raise KeyError("Missing all_statistics in bpb_summary.json")
    
    # BPB 结果只有一个任务，使用 "bpb" 作为任务名称
    metrics = {
        "bpb": {name: all_statistics.get(name) for name in METRICS}
    }
    return metrics


def _collect_model_rows(results_dir: str) -> tuple:
    """收集所有模型的结果"""
    model_results = {}  # {model_name: [rows]}
    model_dirs = [
        name for name in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, name)) and not name.startswith(".")
    ]
    
    for model_name in model_dirs:
        model_dir = os.path.join(results_dir, model_name)
        summary_path = os.path.join(model_dir, "bpb_summary_new.json")
        if not os.path.exists(summary_path):
            # 回退到 bpb_summary.json（格式一致，含 all_statistics）
            summary_path = os.path.join(model_dir, "bpb_summary.json")
        if not os.path.exists(summary_path):
            print(f"[WARN] Missing bpb_summary_new.json or bpb_summary.json: {model_dir}")
            continue
        
        try:
            summary = _load_json(summary_path)
            metrics = _extract_metrics(summary)
            
            # 模型名称直接使用模型名称（没有 config 和 lr 子目录）
            row = {"model": model_name, "metrics": metrics}
            
            if model_name not in model_results:
                model_results[model_name] = []
            model_results[model_name].append(row)
        except Exception as exc:
            print(f"[WARN] Failed to load {summary_path}: {exc}")
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

    # BPB 结果只有一个任务 "bpb"
    task_names = ["bpb"]
    return task_names, all_rows


def _build_header(task_names: list) -> list:
    """构建 CSV 表头"""
    header = ["model"]
    # BPB 只有一个任务，不需要前缀
    for metric in METRICS:
        header.append(metric)
    return header


def _build_row(task_names: list, row_data: dict) -> list:
    """构建 CSV 行数据"""
    row = [row_data["model"]]
    # BPB 只有一个任务 "bpb"
    task_metrics = row_data["metrics"].get("bpb", {})
    for metric in METRICS:
        value = task_metrics.get(metric)
        row.append("" if value is None else value)
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
        output_path = os.path.join(results_dir, "bpb_metrics.csv")
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


def _build_average_sheet(task_dfs: dict):
    """根据短/中/长三个任务的 DataFrame 构建平均结果表，模型为第一列。"""
    import pandas as pd
    import numpy as np

    # 指标列名（与 METRICS 一致）
    metric_cols = METRICS  # ["min", "max", "median", "mean"]
    # 按 MODEL_ORDER 顺序收集模型，遇 None 插入空行
    rows_out = []
    for item in MODEL_ORDER:
        if item is None:
            rows_out.append(None)  # 空行
            continue
        # 从各任务中取该模型的指标
        values_by_metric = {m: [] for m in metric_cols}
        for task, df in task_dfs.items():
            if "model" not in df.columns:
                continue
            match = df.loc[df["model"] == item]
            if match.empty:
                continue
            for m in metric_cols:
                if m not in match.columns:
                    continue
                val = match[m].iloc[0]
                if val is not None and val != "" and not (isinstance(val, float) and np.isnan(val)):
                    try:
                        values_by_metric[m].append(float(val))
                    except (TypeError, ValueError):
                        pass
        # 计算平均
        row = [item]
        for m in metric_cols:
            arr = values_by_metric[m]
            if arr:
                row.append(np.mean(arr))
            else:
                row.append("")
        rows_out.append((item, row))

    # 转为 DataFrame：第一列 model，其余列为各指标平均
    data = []
    for x in rows_out:
        if x is None:
            data.append([""] * (1 + len(metric_cols)))
        else:
            _, row = x
            data.append(row)
    return pd.DataFrame(data, columns=["model"] + metric_cols)


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
        task_dfs = {}  # task -> DataFrame
        for task in TASK_ORDER:
            csv_path = csv_paths.get(task)
            if not csv_path:
                continue
            df = pd.read_csv(csv_path)
            task_dfs[task] = df
            df.to_excel(writer, sheet_name=task, index=False)

        # 添加“短中长”三任务平均结果表：模型为第一列，列为 min/max/median/mean 的平均
        if len(task_dfs) >= 1:
            df_avg = _build_average_sheet(task_dfs)
            df_avg.to_excel(writer, sheet_name="average", index=False)

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
    
    parser = argparse.ArgumentParser(description="Load BPB results and export CSV.")
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
    args = parser.parse_args()

    if args.task == "all":
        if args.output is None:
            output_path = os.path.join(PROJECT_ROOT, "results", "Bpb", "all_bpb_metrics.xlsx")
        else:
            output_path = args.output
        return _export_all_tasks_xlsx(output_path)

    _export_task_csv(args.task, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

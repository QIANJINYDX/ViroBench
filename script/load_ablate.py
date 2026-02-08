#!/usr/bin/env python3
"""
消融实验结果加载脚本
加载不同窗口大小下的测试集MCC结果；支持对所有任务、所有模型动态发现并汇总。
"""

import json
import os
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Set

# 配置
BASE_DIR = Path("/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/results/Classification")
# 若为空则自动发现 BASE_DIR 下所有任务
TASKS: List[str] = []  # 例如 ["ALL-taxon-genus", "ALL-host-genus"] 或留空表示全部
# 优先使用的学习率（按顺序尝试）
LR_PREFERENCE = ["0.001", "0.0001", "0.01"]
# 窗口配置格式：数字_数字_数字
WINDOW_PATTERN = re.compile(r"^\d+_\d+_\d+$")
# 排除的时间戳式目录（如 2026-02-05_00-24-03-771786）
TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")


def discover_tasks() -> List[str]:
    """发现 BASE_DIR 下所有任务目录（子目录名）。"""
    if not BASE_DIR.exists():
        return []
    tasks = [
        d.name for d in BASE_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]
    return sorted(tasks)


def is_timestamp_dir(name: str) -> bool:
    """判断是否为时间戳式目录名（通常为临时/单次运行）。"""
    return bool(TIMESTAMP_PATTERN.match(name))


def is_window_config(name: str) -> bool:
    """判断是否为窗口配置目录名，如 256_16_128。"""
    return bool(WINDOW_PATTERN.match(name))


def discover_models_for_task(task_name: str) -> List[str]:
    """发现指定任务下所有模型目录（排除时间戳、文件、明显非模型目录）。"""
    task_dir = BASE_DIR / task_name
    if not task_dir.exists() or not task_dir.is_dir():
        return []
    models = []
    for d in task_dir.iterdir():
        if not d.is_dir():
            continue
        if is_timestamp_dir(d.name):
            continue
        # 排除常见非模型目录
        if d.name.endswith(".csv") or d.name in ("plots", "plots_all.tar.gz"):
            continue
        models.append(d.name)
    return sorted(models)


def discover_window_sizes_for_task(task_name: str) -> List[str]:
    """发现该任务下所有出现过的窗口配置（并集），保持固定顺序。"""
    order = ["256_16_128", "512_8_64", "1024_4_32", "2048_2_16"]
    seen: Set[str] = set()
    task_dir = BASE_DIR / task_name
    if not task_dir.exists():
        return list(order)
    for d in task_dir.iterdir():
        if not d.is_dir() or is_timestamp_dir(d.name):
            continue
        for sub in d.iterdir():
            if sub.is_dir() and is_window_config(sub.name):
                seen.add(sub.name)
    # 先按标准顺序，再补其余
    result = [w for w in order if w in seen]
    for w in sorted(seen):
        if w not in result:
            result.append(w)
    return result if result else list(order)


def get_mcc_json_path(model_dir: Path, window_size: str) -> Optional[Path]:
    """在 model_dir/window_size 下按 LR 优先级查找 finetune_summary.json，返回该 json 路径。"""
    window_dir = model_dir / window_size
    if not window_dir.exists() or not window_dir.is_dir():
        return None
    for lr in LR_PREFERENCE:
        json_path = window_dir / lr / "finetune_summary.json"
        if json_path.exists():
            return json_path
    # 任意 lr 目录下的 summary 也接受
    for sub in window_dir.iterdir():
        if sub.is_dir():
            p = sub / "finetune_summary.json"
            if p.exists():
                return p
    return None


def load_mcc_from_json(json_path: Path, task_name: str) -> Optional[float]:
    """
    从JSON文件中加载测试集的MCC值
    
    Args:
        json_path: JSON文件路径
        task_name: 任务名称（用于确定提取哪个任务的MCC）
    
    Returns:
        MCC值，如果不存在则返回None
    """
    if not json_path.exists():
        return None
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        test_metrics = data.get("test_metrics_by_task", {})
        if not test_metrics:
            return None
        # ALL-host-genus 下 JSON 里任务键为 host_label
        if task_name == "ALL-host-genus":
            if "host_label" in test_metrics:
                return test_metrics["host_label"].get("mcc")
        # 多任务（如 ALL-taxon-genus 的 kingdom/phylum/...）或其它任务：取所有子任务 MCC 的平均
        mcc_values = []
        for task_metrics in test_metrics.values():
            mcc = task_metrics.get("mcc") if isinstance(task_metrics, dict) else None
            if mcc is not None:
                mcc_values.append(mcc)
        if mcc_values:
            return sum(mcc_values) / len(mcc_values)
        return None
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None


def load_ablation_results(
    task_name: str,
    models: Optional[List[str]] = None,
    window_sizes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    加载指定任务的消融实验结果（支持对所有模型、所有窗口配置）。
    
    Args:
        task_name: 任务名称
        models: 模型列表，None 则自动发现该任务下所有模型
        window_sizes: 窗口配置列表，None 则自动发现该任务下所有窗口配置
    
    Returns:
        DataFrame，第一列为 Model，其余列为各窗口配置的 MCC
    """
    if models is None:
        models = discover_models_for_task(task_name)
    if window_sizes is None:
        window_sizes = discover_window_sizes_for_task(task_name)
    if not models:
        return pd.DataFrame(columns=["Model"] + list(window_sizes))

    results = []
    task_dir = BASE_DIR / task_name
    for model in models:
        row: Dict[str, Optional[float]] = {"Model": model}
        model_dir = task_dir / model
        for window_size in window_sizes:
            json_path = get_mcc_json_path(model_dir, window_size)
            mcc = load_mcc_from_json(json_path, task_name) if json_path else None
            row[window_size] = mcc
        results.append(row)

    df = pd.DataFrame(results)
    return df


def main():
    """主函数：对所有任务、所有模型加载消融结果并保存"""
    tasks = TASKS if TASKS else discover_tasks()
    if not tasks:
        print("未发现任何任务目录，请检查 BASE_DIR:", BASE_DIR)
        return
    print("任务列表:", tasks)
    for task in tasks:
        print(f"\n正在加载任务: {task}")
        models = discover_models_for_task(task)
        window_sizes = discover_window_sizes_for_task(task)
        print(f"  发现模型数: {len(models)}, 窗口配置数: {len(window_sizes)}")
        df = load_ablation_results(task, models=models, window_sizes=window_sizes)
        if df.empty:
            print(f"  任务 {task} 无有效数据，跳过保存")
            continue
        # 显示结果
        print(f"\n{task} 消融实验结果 (共 {len(df)} 个模型):")
        print(df.to_string(index=False))
        # 保存为CSV
        safe_name = task.replace("-", "_")
        output_path = BASE_DIR.parent / f"ablation_results_{safe_name}.csv"
        df.to_csv(output_path, index=False)
        print(f"\n结果已保存到: {output_path}")
        # 保存为Excel（可选）
        excel_path = BASE_DIR.parent / f"ablation_results_{safe_name}.xlsx"
        try:
            df.to_excel(excel_path, index=False)
            print(f"结果已保存到: {excel_path}")
        except Exception as e:
            print(f"Excel 保存跳过: {e}")


if __name__ == "__main__":
    main()

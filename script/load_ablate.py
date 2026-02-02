#!/usr/bin/env python3
"""
消融实验结果加载脚本
加载不同窗口大小下的测试集MCC结果
"""

import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

# 配置
BASE_DIR = Path("/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/results/Classification")
TASKS = ["ALL-taxon-genus", "ALL-host-genus"]
MODELS = ["CNN", "evo2_1b_base", "ntv3-100m-pre", "DNABERT-6", "DNABERT-2-117M", "hyenadna-large-1m"]
WINDOW_SIZES = ["256_16_128", "512_8_64", "1024_4_32", "2048_2_16"]
LR = "0.001"  # 1e-3


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
        
        # 对于ALL-host-genus，任务名是host_label
        if task_name == "ALL-host-genus":
            if "host_label" in test_metrics:
                return test_metrics["host_label"].get("mcc")
        
        # 对于ALL-taxon-genus，有多个任务（kingdom, phylum, class, order, family）
        # 计算所有任务的平均MCC
        elif task_name == "ALL-taxon-genus":
            mcc_values = []
            for task_metrics in test_metrics.values():
                mcc = task_metrics.get("mcc")
                if mcc is not None:
                    mcc_values.append(mcc)
            
            if mcc_values:
                return sum(mcc_values) / len(mcc_values)
        
        return None
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None


def load_ablation_results(task_name: str) -> pd.DataFrame:
    """
    加载指定任务的消融实验结果
    
    Args:
        task_name: 任务名称
    
    Returns:
        DataFrame，第一列为模型，其余列为不同窗口大小的MCC结果
    """
    results = []
    
    for model in MODELS:
        row = {"Model": model}
        
        for window_size in WINDOW_SIZES:
            json_path = BASE_DIR / task_name / model / window_size / LR / "finetune_summary.json"
            mcc = load_mcc_from_json(json_path, task_name)
            row[window_size] = mcc
        
        results.append(row)
    
    df = pd.DataFrame(results)
    return df


def main():
    """主函数：加载所有任务的结果并保存"""
    for task in TASKS:
        print(f"\n正在加载任务: {task}")
        df = load_ablation_results(task)
        
        # 显示结果
        print(f"\n{task} 消融实验结果:")
        print(df.to_string(index=False))
        
        # 保存为CSV
        output_path = BASE_DIR.parent / f"ablation_results_{task.replace('-', '_')}.csv"
        df.to_csv(output_path, index=False)
        print(f"\n结果已保存到: {output_path}")
        
        # 保存为Excel（可选）
        excel_path = BASE_DIR.parent / f"ablation_results_{task.replace('-', '_')}.xlsx"
        df.to_excel(excel_path, index=False)
        print(f"结果已保存到: {excel_path}")


if __name__ == "__main__":
    main()

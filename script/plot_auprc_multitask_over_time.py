#!/usr/bin/env python3
"""
绘制多任务模型在不同时间点的平均 AUPRC 箱线图
横坐标：时间（first_release_date，按年分组）
纵坐标：平均 AUPRC（跨所有任务：kingdom, phylum, class, order, family）

对于多任务问题：
1. 计算每个任务（kingdom, phylum, class, order, family）的 AUPRC
2. 计算所有任务的平均 AUPRC
3. 按时间分组，绘制箱线图
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from datetime import datetime
import glob
from collections import defaultdict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield"
RESULTS_ROOT = os.path.join(BASE_DIR, "results", "Classification", "ALL-taxon-times")
LR = "0.001"
OUTPUT_DIR = os.path.join(BASE_DIR, "script", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 多任务列表
TASK_NAMES = ['kingdom', 'phylum', 'class', 'order', 'family']

def parse_date(date_str):
    """解析日期字符串，返回datetime对象"""
    if pd.isna(date_str) or date_str == "":
        return None
    try:
        # 处理 ISO 8601 格式：2020-02-03T00:00:00+00:00
        if isinstance(date_str, str):
            # 去掉时区信息，只保留日期和时间部分
            date_str = date_str.split('+')[0].split('T')[0]
        return pd.to_datetime(date_str)
    except:
        return None

def calculate_task_auprc(y_true, y_pred, y_conf, num_classes):
    """
    计算单个任务的 AUPRC (macro-average)
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_conf: 预测置信度
        num_classes: 类别数量
    
    Returns:
        AUPRC 值（macro-average）
    """
    # 过滤掉无效标签
    valid_mask = (y_true >= 0) & (y_pred >= 0) & (~np.isnan(y_conf))
    if valid_mask.sum() == 0:
        return None
    
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    y_conf_valid = y_conf[valid_mask]
    
    # 计算 AUPRC (macro-average)
    # 对于多分类问题，使用 one-vs-rest 方式计算每个类别的 AUPRC，然后取平均
    try:
        auprc_scores = []
        for class_id in range(num_classes):
            # 该类别为正类，其他为负类
            y_binary = (y_true_valid == class_id).astype(int)
            
            # 构建预测分数
            y_scores = np.zeros(len(y_true_valid))
            for i in range(len(y_true_valid)):
                if y_pred_valid[i] == class_id:
                    y_scores[i] = y_conf_valid[i]
                else:
                    # 如果预测不是该类别，使用一个较小的值
                    y_scores[i] = (1.0 - y_conf_valid[i]) / max(1, num_classes - 1)
            
            # 计算该类别的 AUPRC
            if y_binary.sum() > 0:  # 确保有正样本
                auprc = average_precision_score(y_binary, y_scores)
                auprc_scores.append(auprc)
        
        if len(auprc_scores) > 0:
            return np.mean(auprc_scores)
        else:
            return None
    except Exception as e:
        return None

def calculate_multitask_auprc_by_time(df, time_granularity='year'):
    """
    按时间分组计算多任务的平均 AUPRC
    
    Args:
        df: DataFrame，包含所有任务的 true_label_id, predicted_label_id, confidence, first_release_date
        time_granularity: 'year', 'month', 'quarter', 'half_year'
    
    Returns:
        DataFrame with columns: time, mean_auprc, count, task_auprcs (dict)
    """
    # 解析日期
    df['date'] = df['first_release_date'].apply(parse_date)
    
    # 过滤掉日期为空的行
    df_valid = df[df['date'].notna()].copy()
    
    if len(df_valid) == 0:
        print(f"[WARN] 没有有效的日期数据")
        return None
    
    # 按时间粒度分组
    if time_granularity == 'year':
        df_valid['time_group'] = df_valid['date'].dt.to_period('Y')
    elif time_granularity == 'month':
        df_valid['time_group'] = df_valid['date'].dt.to_period('M')
    elif time_granularity == 'quarter':
        df_valid['time_group'] = df_valid['date'].dt.to_period('Q')
    elif time_granularity == 'half_year':
        quarter = df_valid['date'].dt.to_period('Q')
        year = quarter.dt.year
        quarter_num = quarter.dt.quarter
        half_year_num = ((quarter_num - 1) // 2) + 1
        df_valid['time_group'] = year.astype(str) + '-H' + half_year_num.astype(str)
    else:
        raise ValueError(f"不支持的时间粒度: {time_granularity}")
    
    # 按时间分组计算 AUPRC
    results = []
    for time_group, group_df in df_valid.groupby('time_group'):
        # 过滤掉样本量太少的组（少于10个样本）
        min_samples = 10
        if len(group_df) < min_samples:
            continue
        
        task_auprcs = {}
        task_counts = {}
        
        # 对每个任务计算 AUPRC
        for task_name in TASK_NAMES:
            true_col = f"{task_name}_true_label_id"
            pred_col = f"{task_name}_predicted_label_id"
            conf_col = f"{task_name}_confidence"
            
            if true_col not in group_df.columns or pred_col not in group_df.columns or conf_col not in group_df.columns:
                continue
            
            y_true = group_df[true_col].values
            y_pred = group_df[pred_col].values
            y_conf = group_df[conf_col].values
            
            # 获取类别数量
            all_labels = set(y_true[~np.isnan(y_true)]) | set(y_pred[~np.isnan(y_pred)])
            all_labels = [l for l in all_labels if l >= 0]
            if len(all_labels) == 0:
                continue
            num_classes = max(all_labels) + 1
            
            # 计算该任务的 AUPRC
            auprc = calculate_task_auprc(y_true, y_pred, y_conf, num_classes)
            if auprc is not None:
                task_auprcs[task_name] = auprc
                task_counts[task_name] = len(y_true[(y_true >= 0) & (y_pred >= 0) & (~np.isnan(y_conf))])
        
        # 计算所有任务的平均 AUPRC
        if len(task_auprcs) > 0:
            mean_auprc = np.mean(list(task_auprcs.values()))
            results.append({
                'time': time_group,
                'mean_auprc': mean_auprc,
                'count': len(group_df),
                'task_auprcs': task_auprcs,
                'num_tasks': len(task_auprcs)
            })
    
    if len(results) == 0:
        return None
    
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('time')
    return result_df

def main():
    print("[INFO] 开始收集所有模型的预测结果...")
    print(f"[INFO] 结果目录: {RESULTS_ROOT}")
    print(f"[INFO] 学习率: {LR}")
    
    # 收集所有模型的预测结果
    model_results = {}
    
    # 遍历所有模型目录
    for model_dir in sorted(glob.glob(os.path.join(RESULTS_ROOT, "*"))):
        if not os.path.isdir(model_dir):
            continue
        
        model_name = os.path.basename(model_dir)
        csv_path = os.path.join(model_dir, "512_8_64", LR, "test_predictions.csv")
        
        if not os.path.exists(csv_path):
            print(f"[SKIP] {model_name}: 未找到 test_predictions.csv")
            continue
        
        print(f"[INFO] 处理模型: {model_name}")
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_path)
            
            # 计算按时间的平均 AUPRC（按年分组）
            auprc_by_time = calculate_multitask_auprc_by_time(df, time_granularity='year')
            
            if auprc_by_time is not None and len(auprc_by_time) > 0:
                model_results[model_name] = auprc_by_time
                print(f"  [OK] 找到 {len(auprc_by_time)} 个时间点，平均 AUPRC 范围: [{auprc_by_time['mean_auprc'].min():.4f}, {auprc_by_time['mean_auprc'].max():.4f}]")
            else:
                print(f"  [WARN] 无法计算 AUPRC")
        except Exception as e:
            print(f"  [ERROR] 处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    if len(model_results) == 0:
        print("[ERROR] 没有找到任何有效的模型结果")
        return
    
    print(f"\n[INFO] 共找到 {len(model_results)} 个模型的有效结果")
    
    # 准备箱线图数据
    print("[INFO] 开始绘制箱线图...")
    
    # 收集所有时间点和对应的平均 AUPRC 值
    plot_data = []
    plot_labels = []
    
    # 获取所有唯一的时间点
    all_times = set()
    for model_name, auprc_df in model_results.items():
        all_times.update(auprc_df['time'].astype(str))
    
    all_times = sorted(all_times)
    
    # 为每个时间点收集所有模型的平均 AUPRC 值
    for time_str in all_times:
        auprc_values = []
        for model_name, auprc_df in model_results.items():
            time_data = auprc_df[auprc_df['time'].astype(str) == time_str]
            if len(time_data) > 0:
                auprc_values.append(time_data['mean_auprc'].values[0])
        
        if len(auprc_values) > 0:
            plot_data.append(auprc_values)
            plot_labels.append(str(time_str))
    
    # 绘制箱线图 - 使用更美观的尺寸和样式
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 设置更美观的样式参数
    plt.rcParams.update({
        'font.size': 10,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })
    
    # 绘制箱线图
    bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True, 
                    showmeans=True, meanline=True, widths=0.5)
    
    # 美化箱线图 - 使用更现代的颜色方案
    box_colors = ['#4A90E2'] * len(bp['boxes'])  # 现代蓝色
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor('#2C5F8D')
        patch.set_linewidth(1.2)
    
    # 设置其他元素的样式
    for element in ['whiskers', 'caps']:
        plt.setp(bp[element], color='#2C5F8D', linewidth=1.2)
    
    # 中位数线 - 使用更醒目的颜色
    plt.setp(bp['medians'], color='#E74C3C', linewidth=2)
    
    # 均值线 - 使用虚线样式
    plt.setp(bp['means'], color='#F39C12', linewidth=1.8, linestyle='--')
    
    # 异常值 - 使用更柔和的颜色
    plt.setp(bp['fliers'], marker='o', markersize=4, 
             markerfacecolor='#95A5A6', markeredgecolor='#7F8C8D', 
             alpha=0.6, linewidth=0.8)
    
    # 设置标签和标题 - 使用更合适的字体大小
    ax.set_xlabel('Time', fontsize=11, labelpad=8)
    ax.set_ylabel('Mean AUPRC (across all tasks)', fontsize=11, labelpad=8)
    ax.set_title('Mean AUPRC Performance Across Time Points\n(Multi-task: kingdom, phylum, class, order, family)', 
                 fontsize=12, pad=12)
    
    # 设置y轴范围，确保从0开始或接近0
    y_min = min([min(data) for data in plot_data])
    y_max = max([max(data) for data in plot_data])
    y_range = y_max - y_min
    ax.set_ylim(max(0, y_min - y_range * 0.05), y_max + y_range * 0.05)
    
    # 网格 - 使用更柔和的样式
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, axis='y')
    ax.set_axisbelow(True)  # 将网格放在数据下方
    
    # 旋转x轴标签 - 根据标签数量调整角度
    if len(plot_labels) > 10:
        rotation_angle = 60
    else:
        rotation_angle = 45
    plt.xticks(rotation=rotation_angle, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(OUTPUT_DIR, "auprc_multitask_over_time_boxplot.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] 图片已保存到: {output_path}")
    
    # 也保存为PDF格式
    output_path_pdf = os.path.join(OUTPUT_DIR, "auprc_multitask_over_time_boxplot.pdf")
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"[INFO] PDF已保存到: {output_path_pdf}")
    
    # 保存数据到CSV
    output_csv = os.path.join(OUTPUT_DIR, "auprc_multitask_over_time_data.csv")
    all_data = []
    for model_name, auprc_df in model_results.items():
        for _, row in auprc_df.iterrows():
            data_row = {
                'model': model_name,
                'time': str(row['time']),
                'mean_auprc': row['mean_auprc'],
                'count': row['count'],
                'num_tasks': row['num_tasks']
            }
            # 添加每个任务的 AUPRC
            for task_name in TASK_NAMES:
                if task_name in row['task_auprcs']:
                    data_row[f'{task_name}_auprc'] = row['task_auprcs'][task_name]
                else:
                    data_row[f'{task_name}_auprc'] = None
            all_data.append(data_row)
    
    df_all = pd.DataFrame(all_data)
    df_all.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"[INFO] 数据已保存到: {output_csv}")
    
    # 打印统计信息
    print("\n[INFO] 统计信息:")
    print(f"  时间点数量: {len(all_times)}")
    print(f"  模型数量: {len(model_results)}")
    print(f"  每个时间点的模型数量范围: [{min(len(v) for v in plot_data)}, {max(len(v) for v in plot_data)}]")
    print(f"  平均 AUPRC 范围: [{df_all['mean_auprc'].min():.4f}, {df_all['mean_auprc'].max():.4f}]")

if __name__ == "__main__":
    main()


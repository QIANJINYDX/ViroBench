#!/usr/bin/env python3
"""
绘制所有模型在不同时间点的 AUPRC 箱线图
横坐标：时间（first_release_date，按年分组，每十二个月为一个时间节点）
纵坐标：AUPRC（按时间分组计算）
箱线图显示所有模型在每个时间点的 AUPRC 分布

注意：样本量少于10个的时间点会被过滤掉，避免指标不稳定（如2025年只有1个样本）
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
RESULTS_ROOT = os.path.join(BASE_DIR, "results", "Classification", "ALL-host-times")
LR = "0.001"
OUTPUT_DIR = os.path.join(BASE_DIR, "script", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def calculate_auprc_by_time(df, time_granularity='year'):
    """
    按时间分组计算 AUPRC (macro-average)
    
    Args:
        df: DataFrame，包含 true_label_id, predicted_label_id, confidence, first_release_date
        time_granularity: 'year', 'month', 'day', 'quarter' 或 'half_year'
    
    Returns:
        DataFrame with columns: time, auprc, count
    """
    # 获取标签列名（可能是 host_label_true_label_id 或其他任务）
    true_col = None
    pred_col = None
    conf_col = None
    
    for col in df.columns:
        if 'true_label_id' in col:
            true_col = col
        if 'predicted_label_id' in col:
            pred_col = col
        if 'confidence' in col:
            conf_col = col
    
    if true_col is None or pred_col is None or conf_col is None:
        print(f"[WARN] 未找到必要的列，跳过")
        return None
    
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
        # 半年：先转换为季度，然后根据季度编号分组为上半年/下半年
        quarter = df_valid['date'].dt.to_period('Q')
        year = quarter.dt.year
        quarter_num = quarter.dt.quarter
        # 创建半年标识：1-2季度为H1，3-4季度为H2
        half_year_num = ((quarter_num - 1) // 2) + 1
        # 使用年份和半年标识作为分组键
        df_valid['time_group'] = year.astype(str) + '-H' + half_year_num.astype(str)
    elif time_granularity == 'day':
        df_valid['time_group'] = df_valid['date'].dt.to_period('D')
    else:
        raise ValueError(f"不支持的时间粒度: {time_granularity}")
    
    # 获取所有类别数量
    all_labels = set(df_valid[true_col].unique()) | set(df_valid[pred_col].unique())
    num_classes = len([l for l in all_labels if l >= 0])
    
    # 按时间分组计算 AUPRC
    results = []
    for time_group, group_df in df_valid.groupby('time_group'):
        y_true = group_df[true_col].values
        y_pred = group_df[pred_col].values
        y_conf = group_df[conf_col].values
        
        # 过滤掉无效标签（如果有的话）
        valid_mask = (y_true >= 0) & (y_pred >= 0) & (~np.isnan(y_conf))
        if valid_mask.sum() == 0:
            continue
        
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        y_conf_valid = y_conf[valid_mask]
        
        # 过滤掉样本量太少的组（少于10个样本），避免指标不稳定
        min_samples = 10
        if len(y_true_valid) < min_samples:
            continue
        
        # 计算 AUPRC (macro-average)
        # 对于多分类问题，使用 one-vs-rest 方式计算每个类别的 AUPRC，然后取平均
        try:
            # 构建每个类别的二分类标签和分数
            auprc_scores = []
            for class_id in range(num_classes):
                # 该类别为正类，其他为负类
                y_binary = (y_true_valid == class_id).astype(int)
                
                # 构建预测分数：如果预测是该类别，使用confidence；否则使用 1-confidence 的某种分配
                # 简单方法：如果预测是该类别，使用confidence；否则使用一个小的值（如 1-confidence 除以类别数）
                y_scores = np.zeros(len(y_true_valid))
                for i in range(len(y_true_valid)):
                    if y_pred_valid[i] == class_id:
                        y_scores[i] = y_conf_valid[i]
                    else:
                        # 如果预测不是该类别，使用一个较小的值
                        # 这里使用 1-confidence 除以 (num_classes-1) 作为其他类别的概率
                        y_scores[i] = (1.0 - y_conf_valid[i]) / max(1, num_classes - 1)
                
                # 计算该类别的 AUPRC
                if y_binary.sum() > 0:  # 确保有正样本
                    auprc = average_precision_score(y_binary, y_scores)
                    auprc_scores.append(auprc)
            
            if len(auprc_scores) > 0:
                auprc_macro = np.mean(auprc_scores)
                results.append({
                    'time': time_group,
                    'auprc': auprc_macro,
                    'count': len(y_true_valid)
                })
        except Exception as e:
            print(f"[WARN] 计算 AUPRC 失败: {e}")
            continue
    
    if len(results) == 0:
        return None
    
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('time')
    return result_df

def main():
    print("[INFO] 开始收集所有模型的预测结果...")
    
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
            
            # 计算按时间的 AUPRC（按年，每十二个月为一个时间节点）
            auprc_by_time = calculate_auprc_by_time(df, time_granularity='year')
            
            if auprc_by_time is not None and len(auprc_by_time) > 0:
                model_results[model_name] = auprc_by_time
                print(f"  [OK] 找到 {len(auprc_by_time)} 个时间点")
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
    
    # 收集所有时间点和对应的 AUPRC 值
    plot_data = []
    plot_labels = []
    
    # 获取所有唯一的时间点
    all_times = set()
    for model_name, auprc_df in model_results.items():
        all_times.update(auprc_df['time'].astype(str))
    
    all_times = sorted(all_times)
    
    # 为每个时间点收集所有模型的 AUPRC 值
    for time_str in all_times:
        auprc_values = []
        for model_name, auprc_df in model_results.items():
            time_data = auprc_df[auprc_df['time'].astype(str) == time_str]
            if len(time_data) > 0:
                auprc_values.append(time_data['auprc'].values[0])
        
        if len(auprc_values) > 0:
            plot_data.append(auprc_values)
            plot_labels.append(time_str)
    
    # 绘制箱线图
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # 绘制箱线图
    bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True, 
                    showmeans=True, meanline=True)
    
    # 美化箱线图
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black', linewidth=1.5)
    
    ax.set_xlabel('时间 (年份)', fontsize=12)
    ax.set_ylabel('AUPRC (Macro-average)', fontsize=12)
    ax.set_title('不同模型在不同时间点的 AUPRC 分布（箱线图）', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 旋转x轴标签
    plt.xticks(rotation=45, ha='right')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(OUTPUT_DIR, "auprc_over_time_boxplot.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] 图片已保存到: {output_path}")
    
    # 也保存为PDF格式
    output_path_pdf = os.path.join(OUTPUT_DIR, "auprc_over_time_boxplot.pdf")
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"[INFO] PDF已保存到: {output_path_pdf}")
    
    # 保存数据到CSV
    output_csv = os.path.join(OUTPUT_DIR, "auprc_over_time_data.csv")
    all_data = []
    for model_name, auprc_df in model_results.items():
        for _, row in auprc_df.iterrows():
            all_data.append({
                'model': model_name,
                'time': str(row['time']),
                'auprc': row['auprc'],
                'count': row['count']
            })
    df_all = pd.DataFrame(all_data)
    df_all.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"[INFO] 数据已保存到: {output_csv}")

if __name__ == "__main__":
    main()



"""
清理 C1_data.csv 数据集：
1. 使用孤立森林算法去除过长的序列（离群点）
2. 去除 N 含量大于 5% 的序列
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import os

def calculate_n_content(sequence):
    """计算序列中 N 的含量（百分比）"""
    if pd.isna(sequence) or len(str(sequence)) == 0:
        return 100.0
    seq = str(sequence).upper()
    n_count = seq.count('N')
    total_length = len(seq)
    if total_length == 0:
        return 100.0
    return (n_count / total_length) * 100.0

def filter_outliers_only_long(df, contamination=0.1, random_state=42):
    """
    使用孤立森林算法，但只标记过长的序列为离群点
    不过滤过短的序列
    
    Args:
        df: DataFrame with 'sequence' column
        contamination: 预期的离群点比例
        random_state: 随机种子
    
    Returns:
        DataFrame with outliers removed (only long sequences)
    """
    # 计算序列长度
    lengths = df['sequence'].str.len().values.reshape(-1, 1)
    
    # 使用孤立森林检测离群点
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    outlier_labels = iso_forest.fit_predict(lengths)
    
    # 获取所有被标记为离群点的索引
    outlier_indices = np.where(outlier_labels == -1)[0]
    outlier_lengths = lengths[outlier_indices].flatten()
    
    # 计算中位数长度作为阈值
    median_length = np.median(lengths)
    
    # 只标记那些长度大于中位数的离群点为真正的离群点
    # 即只去除过长的序列，不去除过短的序列
    true_outlier_indices = []
    for idx in outlier_indices:
        if lengths[idx][0] > median_length:
            true_outlier_indices.append(idx)
    
    print(f"孤立森林检测到 {len(outlier_indices)} 个离群点")
    print(f"其中 {len(true_outlier_indices)} 个是过长的序列（将被去除）")
    print(f"中位数长度: {median_length:,.0f} bp")
    if len(true_outlier_indices) > 0:
        print(f"被去除的过长序列长度范围: {lengths[true_outlier_indices].min():,.0f} - {lengths[true_outlier_indices].max():,.0f} bp")
    
    # 创建掩码：保留所有非过长离群点的样本
    mask = np.ones(len(df), dtype=bool)
    mask[true_outlier_indices] = False
    
    return df[mask].copy(), len(true_outlier_indices)

def clean_c1_data(input_path, output_path, contamination=0.1, n_threshold=5.0):
    """
    清理 C1_data.csv 数据集
    
    Args:
        input_path: 输入 CSV 文件路径
        output_path: 输出 CSV 文件路径
        contamination: 孤立森林的离群点比例（默认 0.1，即 10%）
        n_threshold: N 含量阈值（百分比，默认 5.0%）
    """
    print(f"正在读取数据: {input_path}")
    df = pd.read_csv(input_path)
    print(f"原始数据: {len(df)} 条序列")
    
    # 步骤 1: 去除 N 含量大于阈值的序列
    print(f"\n步骤 1: 去除 N 含量 > {n_threshold}% 的序列...")
    df['n_content'] = df['sequence'].apply(calculate_n_content)
    n_outliers = len(df[df['n_content'] > n_threshold])
    print(f"发现 {n_outliers} 条序列的 N 含量 > {n_threshold}%")
    if n_outliers > 0:
        print(f"N 含量范围: {df[df['n_content'] > n_threshold]['n_content'].min():.2f}% - {df[df['n_content'] > n_threshold]['n_content'].max():.2f}%")
    
    df_filtered = df[df['n_content'] <= n_threshold].copy()
    print(f"去除 N 含量过高的序列后: {len(df_filtered)} 条序列")
    
    # 步骤 2: 使用孤立森林去除过长的序列（离群点）
    print(f"\n步骤 2: 使用孤立森林算法去除过长的序列（contamination={contamination})...")
    df_cleaned, long_outliers = filter_outliers_only_long(df_filtered, contamination=contamination)
    print(f"去除过长的序列后: {len(df_cleaned)} 条序列")
    
    # 删除临时列
    df_cleaned = df_cleaned.drop(columns=['n_content'], errors='ignore')
    
    # 保存清理后的数据
    print(f"\n正在保存清理后的数据到: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_cleaned.to_csv(output_path, index=False)
    
    # 统计信息
    print(f"\n清理完成！")
    print(f"原始数据: {len(df)} 条序列")
    print(f"去除 N 含量 > {n_threshold}%: {n_outliers} 条")
    print(f"去除过长的序列: {long_outliers} 条")
    print(f"最终数据: {len(df_cleaned)} 条序列")
    print(f"保留比例: {len(df_cleaned)/len(df)*100:.2f}%")
    
    # 显示最终数据的统计信息
    final_lengths = df_cleaned['sequence'].str.len()
    print(f"\n最终数据统计:")
    print(f"  序列长度范围: {final_lengths.min():,} - {final_lengths.max():,} bp")
    print(f"  平均长度: {final_lengths.mean():,.0f} bp")
    print(f"  中位数长度: {final_lengths.median():,.0f} bp")
    
    return df_cleaned

if __name__ == "__main__":
    input_path = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/data/C1_data.csv"
    output_path = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/data/C1_data_cleaned.csv"
    
    # 清理数据
    df_cleaned = clean_c1_data(
        input_path=input_path,
        output_path=output_path,
        contamination=0.1,  # 预期 10% 的离群点
        n_threshold=5.0     # N 含量阈值 5%
    )


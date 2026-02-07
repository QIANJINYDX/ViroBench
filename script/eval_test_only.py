#!/usr/bin/env python3
"""
只运行测试集评估的脚本
使用已保存的 embeddings 和训练好的模型权重
支持 CNN 模型（直接处理序列，无需预计算 embeddings）
"""
import argparse
import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, Optional, Tuple, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.mlp_head import MLPHead
from evaluators.finetune import FineTuneSeqEvaluator, ChunkedEmbeddingDataset
from datasets.virus_datasets import VirusSplitDatasets


class MultiTaskMLPHead(nn.Module):
    def __init__(self, input_dim: int, task_out_dims: dict):
        super().__init__()
        self.task_names = list(task_out_dims.keys())
        self.heads = nn.ModuleDict({
            name: MLPHead(
                input_dim=input_dim,
                task="multiclass",
                num_outputs=int(task_out_dims[name]),
            )
            for name in self.task_names
        })

    def forward(self, x: torch.Tensor) -> dict:
        return {name: head(x) for name, head in self.heads.items()}


def is_cnn_model(result_dir: str) -> bool:
    """检测是否是 CNN 模型"""
    parts = result_dir.split('/')
    for part in parts:
        if part == "CNN":
            return True
    return False


def load_model_and_config(result_dir: str):
    """加载模型配置和权重"""
    # 加载配置
    summary_path = os.path.join(result_dir, "finetune_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"找不到配置文件: {summary_path}")
    
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    # 加载 label2id
    label2id_path = os.path.join(result_dir, "label2id.json")
    if not os.path.exists(label2id_path):
        raise FileNotFoundError(f"找不到标签映射文件: {label2id_path}")
    
    with open(label2id_path, 'r', encoding='utf-8') as f:
        label2id = json.load(f)
    
    # 获取训练参数
    training_args = summary.get("training_args", {})
    task_names = training_args.get("task_names", list(label2id.keys()))
    num_classes = summary.get("num_classes", {})
    
    # 检测是否是 CNN 模型
    is_cnn = is_cnn_model(result_dir)
    
    if is_cnn:
        # CNN 模型：加载完整的 CNN 模型（包括 backbone 和 head）
        from models.cnn import CNNConfig, GenomeCNN1D
        
        cfg = CNNConfig()
        task_out_dims = {name: num_classes.get(name, len(label2id.get(name, {}))) for name in task_names}
        model = GenomeCNN1D(out_dim=task_out_dims, cfg=cfg)
        
        # 加载权重
        weight_path = os.path.join(result_dir, "best_head.pt")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"找不到模型权重文件: {weight_path}")
        
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        print(f"[INFO] 已加载 CNN 模型权重: {weight_path}")
        print(f"[INFO] task_names = {task_names}")
        print(f"[INFO] num_classes = {num_classes}")
        
        return model, summary, label2id, None  # CNN 不需要 embedding_dir
    else:
        # 其他模型：使用预计算的 embeddings
        # 推断 hidden_size（从第一个 chunk 的 embedding 维度）
        embedding_dir = os.path.dirname(result_dir)
        embedding_dir = os.path.join(embedding_dir, "embeddings")
        test_chunk_dir = os.path.join(embedding_dir, "test_chunks")
        
        if not os.path.exists(test_chunk_dir):
            raise FileNotFoundError(f"找不到 embeddings 目录: {test_chunk_dir}")
        
        # 从第一个 chunk 获取 embedding 维度
        chunk_files = sorted([f for f in os.listdir(test_chunk_dir) if f.startswith("chunk_") and f.endswith(".pt")])
        if not chunk_files:
            raise FileNotFoundError(f"找不到 chunk 文件: {test_chunk_dir}")
        
        first_chunk_path = os.path.join(test_chunk_dir, chunk_files[0])
        chunk_data = torch.load(first_chunk_path, map_location="cpu", weights_only=True)
        hidden_size = int(chunk_data["feats"].shape[1])
        
        print(f"[INFO] hidden_size = {hidden_size}")
        print(f"[INFO] task_names = {task_names}")
        print(f"[INFO] num_classes = {num_classes}")
        
        # 创建模型
        task_out_dims = {name: num_classes.get(name, len(label2id.get(name, {}))) for name in task_names}
        model = MultiTaskMLPHead(hidden_size, task_out_dims)
        
        # 加载权重
        weight_path = os.path.join(result_dir, "best_head.pt")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"找不到模型权重文件: {weight_path}")
        
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        print(f"[INFO] 已加载模型权重: {weight_path}")
        
        return model, summary, label2id, embedding_dir


def load_test_data_for_cnn(result_dir: str):
    """为 CNN 模型加载测试数据"""
    # 从 result_dir 推断数据集路径
    # result_dir 格式: .../Classification/{dataset_name}/{model_name}/{window_len}_{train_num_windows}_{eval_num_windows}/{lr}/
    parts = result_dir.split('/')
    dataset_name_idx = None
    for i, part in enumerate(parts):
        if part == "Classification":
            dataset_name_idx = i + 1
            break
    
    if not dataset_name_idx or dataset_name_idx >= len(parts):
        raise ValueError(f"无法从 result_dir 推断数据集名称: {result_dir}")
    
    dataset_name = parts[dataset_name_idx]
    dataset_parts = dataset_name.split("-")
    if len(dataset_parts) != 3:
        raise ValueError(f"无效的数据集名称格式: {dataset_name}")
    
    na_type, label, check = dataset_parts
    split_dir = f"/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/all_viral/cls_data/{na_type}/{label}/{check}"
    
    # 获取窗口参数
    # result_dir 格式: .../Classification/{dataset_name}/{model_name}/{window_len}_{train_num_windows}_{eval_num_windows}/{lr}/
    # 所以需要跳过模型名称部分（dataset_name_idx + 1），窗口参数在 dataset_name_idx + 2
    if dataset_name_idx + 2 < len(parts):
        window_params_str = parts[dataset_name_idx + 2]
        window_params = window_params_str.split("_")
        if len(window_params) >= 3:
            window_len = int(window_params[0])
            train_num_windows = int(window_params[1])
            eval_num_windows = int(window_params[2])
        else:
            raise ValueError(f"无法解析窗口参数: {window_params_str}，期望格式: window_len_train_num_windows_eval_num_windows")
    else:
        raise ValueError(f"无法从 result_dir 推断窗口参数: {result_dir}，路径格式应为: .../Classification/{{dataset_name}}/{{model_name}}/{{window_params}}/{{lr}}/")
    
    # 确定标签列
    if label == "taxon":
        labels = ["kingdom", "phylum", "class", "order", "family"]
    elif label == "host":
        labels = ["host_label"]
    else:
        raise ValueError(f"无效的 label 类型: {label}")
    
    # 加载数据集
    base = VirusSplitDatasets(
        split_dir,
        label_cols=labels,
        return_format="dict",
        attach_sequences=True,
    )
    win = base.make_windowed(
        window_len=window_len,
        train_num_windows=train_num_windows,
        eval_num_windows=eval_num_windows,
        seed=42,
        return_format="dict",
    )
    
    return win.test, base.label2id


def seqs_to_tokens_cnn(seqs: List[str], device: str = "cpu") -> torch.Tensor:
    """将序列转换为 CNN 模型所需的 token ids"""
    vocab = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5}
    pad_idx = 0
    max_len = max(len(s) for s in seqs) if seqs else 0
    tokens = torch.full((len(seqs), max_len), pad_idx, dtype=torch.long)
    for i, seq in enumerate(seqs):
        if isinstance(seq, list):
            seq = "".join(seq)
        s = str(seq).upper()
        ids = [vocab.get(ch, 5) for ch in s]
        if ids:
            tokens[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return tokens.to(device)


def eval_test_only(result_dir: str, device: str = None):
    """只运行测试集评估"""
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARN] 请求使用 {device}，但 CUDA 不可用，改用 CPU")
        device = "cpu"
    
    print(f"[INFO] 使用设备: {device}")
    print(f"[INFO] 结果目录: {result_dir}")
    
    # 加载模型和配置
    model, summary, label2id, embedding_dir = load_model_and_config(result_dir)
    device_obj = torch.device(device)
    model = model.to(device_obj)
    model.eval()
    
    # 检测是否是 CNN 模型
    is_cnn = is_cnn_model(result_dir)
    
    task_names = summary.get("training_args", {}).get("task_names", list(label2id.keys()))
    
    if is_cnn:
        # CNN 模型：加载原始序列数据
        print("[INFO] 检测到 CNN 模型，加载原始序列数据...")
        test_dataset, label2id_loaded = load_test_data_for_cnn(result_dir)
        print(f"[INFO] 测试集大小: {len(test_dataset)}")
        
        # 创建 DataLoader
        training_args = summary.get("training_args", {})
        batch_size = training_args.get("batch_size", 64)
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        print("[INFO] 开始评估测试集...")
        
        all_logits = {}
        all_labels = []
        all_groups = []
        
        with torch.no_grad():
            for batch in test_loader:
                # DataLoader 会将字典列表转换为字典的字典（每个键对应一个列表）
                if isinstance(batch, dict):
                    seqs = batch["sequence"]
                    labels_raw = batch["labels"]
                    group_ids_raw = batch.get("seq_index", None)
                    
                    # 处理标签：如果是多任务，labels_raw 可能是列表的列表
                    if isinstance(labels_raw, (list, tuple)):
                        if len(labels_raw) > 0 and isinstance(labels_raw[0], (list, tuple, np.ndarray)):
                            # 多任务：每个元素是一个数组
                            labels_array = np.array([np.asarray(lab, dtype=np.int64) for lab in labels_raw])
                        else:
                            # 单任务：每个元素是一个标量
                            labels_array = np.array(labels_raw, dtype=np.int64).reshape(-1, 1)
                    else:
                        labels_array = np.asarray(labels_raw, dtype=np.int64)
                        if labels_array.ndim == 1:
                            labels_array = labels_array.reshape(-1, 1)
                    
                    # 处理 group_ids
                    if group_ids_raw is not None:
                        group_ids = np.array(group_ids_raw, dtype=np.int64)
                    else:
                        group_ids = None
                else:
                    # 如果不是字典格式，尝试按列表处理
                    seqs = [item["sequence"] if isinstance(item, dict) else item[0] for item in batch]
                    labels_list = []
                    group_ids = []
                    for item in batch:
                        if isinstance(item, dict):
                            lab = item["labels"]
                            if isinstance(lab, (list, tuple, np.ndarray)):
                                labels_list.append(np.asarray(lab, dtype=np.int64))
                            else:
                                labels_list.append(int(lab))
                            group_ids.append(int(item.get("seq_index", -1)) if "seq_index" in item and item["seq_index"] is not None else -1)
                        else:
                            labels_list.append(item[1])
                            group_ids.append(-1)
                    
                    labels_array = np.array(labels_list)
                    if labels_array.ndim == 1:
                        labels_array = labels_array.reshape(-1, 1)
                    group_ids = np.array(group_ids) if any(g >= 0 for g in group_ids) else None
                
                if not seqs:
                    continue
                
                # 将序列转换为 tokens
                tokens = seqs_to_tokens_cnn(seqs, device_obj)
                
                # 前向传播
                logits_dict = model(tokens)
                
                # 收集结果
                for tname in task_names:
                    if tname not in all_logits:
                        all_logits[tname] = []
                    all_logits[tname].append(logits_dict[tname].detach().cpu().numpy())
                
                # 处理标签
                all_labels.append(labels_array)
                if group_ids is not None:
                    all_groups.append(group_ids)
    else:
        # 其他模型：使用预计算的 embeddings
        test_chunk_dir = os.path.join(embedding_dir, "test_chunks")
        if not os.path.exists(test_chunk_dir):
            raise FileNotFoundError(f"找不到测试集 embeddings: {test_chunk_dir}")
        
        test_dataset = ChunkedEmbeddingDataset(test_chunk_dir)
        print(f"[INFO] 测试集大小: {len(test_dataset)}")
        
        # 创建 DataLoader
        training_args = summary.get("training_args", {})
        batch_size = training_args.get("batch_size", 64)
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # 使用 0 避免多进程问题
        )
        
        print("[INFO] 开始评估测试集...")
        
        all_logits = {}
        all_labels = []
        all_groups = []
        
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 3:
                    xb, yb, gb = batch
                elif len(batch) == 2:
                    xb, yb = batch
                    gb = None
                else:
                    raise ValueError(f"意外的 batch 格式: {len(batch)}")
                
                xb = xb.to(device_obj, dtype=torch.float32)
                yb = yb.to(device_obj)
                
                # 前向传播
                logits_dict = model(xb)
                
                # 收集结果
                for tname in task_names:
                    if tname not in all_logits:
                        all_logits[tname] = []
                    all_logits[tname].append(logits_dict[tname].detach().cpu().numpy())
                
                all_labels.append(yb.detach().cpu().numpy())
                if gb is not None:
                    all_groups.append(gb.detach().cpu().numpy())
    
    # 合并结果
    logits_np = {tname: np.concatenate(all_logits[tname], axis=0) for tname in task_names}
    labels_np = np.concatenate(all_labels, axis=0)
    groups_np = np.concatenate(all_groups, axis=0) if all_groups else None
    
    print(f"[INFO] logits shape: {[logits_np[t].shape for t in task_names]}")
    print(f"[INFO] labels shape: {labels_np.shape}")
    if groups_np is not None:
        print(f"[INFO] groups shape: {groups_np.shape}")
    
    # 使用 FineTuneSeqEvaluator 的评估方法
    # 创建一个临时的评估器实例来使用其方法
    from evaluators.finetune import FineTuneSeqEvaluator
    
    # 创建虚拟数据集（不会被使用，因为我们直接传入了 logits）
    class DummyDataset:
        def __init__(self, length):
            self.length = length
        def __len__(self):
            return self.length
        def __getitem__(self, idx):
            return torch.zeros(1), torch.zeros(1)
    
    # 创建评估器（仅用于调用方法）
    evaluator = FineTuneSeqEvaluator(
        embedder=None,  # 不需要 embedder
        model=model,
        train_ds=DummyDataset(1),
        val_ds=DummyDataset(1),
        test_ds=DummyDataset(1),
        output_dir=result_dir,
        task="multiclass",
        multitask=True,
        task_names=task_names,
        save_predictions=True,  # 启用保存预测结果
    )
    
    # 手动调用评估逻辑
    # 我们需要聚合和计算指标
    from evaluators.finetune import _compute_metrics_from_logits
    from sklearn.metrics import classification_report, confusion_matrix
    
    metrics_by_task = {}
    report_by_task = {}
    cm_by_task = {}
    
    # 聚合 logits 和 labels（如果有 groups）
    logits_agg = {}
    labels_agg = None
    if groups_np is not None:
        for ti, tname in enumerate(task_names):
            logits_t = logits_np[tname]
            labels_t = labels_np[:, ti] if labels_np.ndim > 1 else labels_np
            logits_t_agg, labels_t_agg = evaluator._aggregate_by_group_mean(
                logits_t, labels_t, groups_np)
            logits_agg[tname] = logits_t_agg
            if labels_agg is None:
                labels_agg = np.zeros((len(labels_t_agg), len(task_names)), dtype=labels_np.dtype)
            labels_agg[:, ti] = labels_t_agg
    else:
        logits_agg = logits_np
        labels_agg = labels_np
    
    # 收集元数据（taxid 和 first_release_date）
    # 需要从原始数据集中获取
    taxids = None
    first_release_dates = None
    try:
        # 从 result_dir 推断数据集路径
        # result_dir 格式: .../Classification/{dataset_name}/{model_name}/{window_len}_{train_num_windows}_{eval_num_windows}/{lr}/
        parts = result_dir.split('/')
        dataset_name_idx = None
        for i, part in enumerate(parts):
            if part == "Classification":
                dataset_name_idx = i + 1
                break
        
        if dataset_name_idx and dataset_name_idx < len(parts):
            dataset_name = parts[dataset_name_idx]
            # 解析数据集名称
            dataset_parts = dataset_name.split("-")
            if len(dataset_parts) == 3:
                na_type, label, check = dataset_parts
                split_dir = f"/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/all_viral/cls_data/{na_type}/{label}/{check}"
                test_csv_path = os.path.join(split_dir, "test.csv")
                
                if os.path.exists(test_csv_path):
                    import pandas as pd
                    df = pd.read_csv(test_csv_path)
                    
                    # 如果有 groups，groups 就是 seq_index（原始序列的索引）
                    if groups_np is not None:
                        # 聚合后的 groups 对应唯一的序列索引
                        unique_groups = np.unique(groups_np)
                        # 使用与 _aggregate_by_group_mean 相同的排序逻辑
                        order = np.argsort(groups_np, kind="mergesort")
                        g = groups_np[order]
                        uniq, idx_start = np.unique(g, return_index=True)
                        
                        taxids_list = []
                        first_release_dates_list = []
                        for g_val in uniq:
                            # g_val 就是 seq_index
                            seq_idx = int(g_val)
                            if seq_idx < len(df):
                                taxids_list.append(df.iloc[seq_idx]['taxid'] if 'taxid' in df.columns else None)
                                first_release_dates_list.append(
                                    df.iloc[seq_idx].get('first_release_date', None) if 'first_release_date' in df.columns else None)
                            else:
                                taxids_list.append(None)
                                first_release_dates_list.append(None)
                        
                        taxids = np.array(taxids_list)
                        first_release_dates = np.array(first_release_dates_list)
                    else:
                        # 没有 groups，直接使用
                        if len(df) == len(labels_agg):
                            taxids = df['taxid'].values if 'taxid' in df.columns else None
                            first_release_dates = df['first_release_date'].values if 'first_release_date' in df.columns else None
                    
                    print(f"[INFO] 已加载元数据: taxids={taxids is not None}, first_release_dates={first_release_dates is not None}")
    except Exception as e:
        print(f"[WARN] 无法获取元数据: {e}")
        import traceback
        traceback.print_exc()
    
    # 保存预测结果
    evaluator._save_predictions_to_csv(
        logits_agg, labels_agg, None, "test", task_names=task_names,
        taxids=taxids, first_release_dates=first_release_dates
    )
    
    # 计算指标
    for ti, tname in enumerate(task_names):
        logits_t = logits_agg[tname]
        labels_t = labels_agg[:, ti] if labels_agg.ndim > 1 else labels_agg
        
        # 过滤无效标签
        if logits_t.ndim == 1 or logits_t.shape[1] == 1:
            valid = (labels_t >= 0) & (labels_t <= 1)
        else:
            valid = (labels_t >= 0) & (labels_t < logits_t.shape[1])
        
        logits_t = logits_t[valid]
        labels_t = labels_t[valid]
        
        if logits_t.shape[0] == 0:
            continue
        
        # 计算指标
        task_type = "binary" if (logits_t.ndim == 1 or logits_t.shape[1] == 1) else "multiclass"
        metrics_by_task[tname] = _compute_metrics_from_logits(
            logits_t, labels_t, task=task_type, debug=False)
        
        # 计算预测
        if task_type == "binary":
            preds = (1/(1+np.exp(-logits_t.reshape(-1))) >= 0.5).astype(np.int64)
        else:
            preds = logits_t.argmax(axis=-1)
        
        report_by_task[tname] = classification_report(
            labels_t, preds, digits=4, output_dict=True, zero_division=0)
        cm_by_task[tname] = confusion_matrix(labels_t, preds).tolist()
    
    # 打印结果
    print("\n" + "="*80)
    print("测试集评估结果")
    print("="*80)
    for tname in task_names:
        if tname in metrics_by_task:
            print(f"\n任务: {tname}")
            print(f"  Accuracy: {metrics_by_task[tname].get('accuracy', 'N/A'):.4f}")
            print(f"  F1-macro: {metrics_by_task[tname].get('f1_macro', 'N/A'):.4f}")
            print(f"  MCC: {metrics_by_task[tname].get('mcc', 'N/A'):.4f}")
    
    print(f"\n[INFO] 预测结果已保存到: {os.path.join(result_dir, 'test_predictions.csv')}")
    
    return metrics_by_task, report_by_task, cm_by_task

"""
用法示例：
  python script/eval_test_only.py --result_dir /inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/results/Classification/ALL-taxon-genus/AIDO.DNA-7B/512_8_64/0.001
"""
def main():
    parser = argparse.ArgumentParser(description="只运行测试集评估")
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="结果目录路径（包含 best_head.pt 和 finetune_summary.json 的目录）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备（cuda:0 或 cpu），默认自动选择",
    )
    args = parser.parse_args()
    
    eval_test_only(args.result_dir, args.device)


if __name__ == "__main__":
    main()


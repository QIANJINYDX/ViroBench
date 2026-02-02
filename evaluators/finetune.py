# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import glob
import warnings
import json
import math
from typing import Any, Dict, Optional, List, Tuple, Literal, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score,
    matthews_corrcoef, average_precision_score,
)
from sklearn.preprocessing import label_binarize
from tqdm.auto import tqdm
from collections import Counter
import numpy as np
import pandas as pd

def _debug_y(name, y, debug: bool = False):
    if not debug:
        return
    y = np.asarray(y)
    uniq = np.unique(y)
    top = Counter(y.tolist()).most_common(5)
    print(f"[DEBUG] {name}: n={len(y)} unique={len(uniq)} top5={top}")

def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _compute_metrics_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    task: Literal["binary", "multiclass", "regression"] = "multiclass",
    debug: bool = False,
) -> Dict[str, float]:
    """与之前评估器一致风格的指标聚合。"""
    metrics: Dict[str, float] = {}
    if task == "regression":
        # 只返回占位；你可以按需扩展 MAE/R2 等
        return metrics

    if logits.ndim == 1 or logits.shape[1] == 1:
        # binary
        probs = 1 / (1 + np.exp(-logits.reshape(-1)))
        preds = (probs >= 0.5).astype(np.int64)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="binary")
        mcc = matthews_corrcoef(labels, preds) if labels.size else 0.0
        prec = precision_score(
            labels, preds, average="binary", zero_division=0)
        rec = recall_score(labels, preds, average="binary", zero_division=0)
        metrics.update(dict(
            accuracy=acc, f1_macro=f1, f1_micro=f1, f1_weighted=f1,
            precision_macro=prec, recall_macro=rec, mcc=mcc
        ))
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings(
                    "ignore",
                    message="No positive class found in y_true.*",
                    category=UserWarning,
                    module="sklearn.metrics._ranking",
                )
                metrics["auc_macro_ovr"] = float(roc_auc_score(labels, probs))
        except Exception:
            pass
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings(
                    "ignore",
                    message="No positive class found in y_true.*",
                    category=UserWarning,
                    module="sklearn.metrics._ranking",
                )
                metrics["auprc"] = float(
                    average_precision_score(labels, probs))
        except Exception:
            pass
        return metrics

    # multiclass
    preds = logits.argmax(axis=-1)
    
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_micro = f1_score(labels, preds, average="micro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    prec_macro = precision_score(
        labels, preds, average="macro", zero_division=0)
    rec_macro = recall_score(labels, preds, average="macro", zero_division=0)
    
    # 计算MCC，如果所有标签相同则返回0
    mcc_val = 0.0
    if labels.size > 0:
        # 诊断：检查标签和预测的分布
        unique_labels = np.unique(labels)
        unique_preds = np.unique(preds)
        label_counts = np.bincount(labels, minlength=logits.shape[1] if logits.ndim > 1 else 2)
        pred_counts = np.bincount(preds, minlength=logits.shape[1] if logits.ndim > 1 else 2)
        
        # 检查是否所有标签都是同一个类别
        if len(unique_labels) == 1:
            if debug:
                print(f"[DEBUG] MCC=0: All labels are the same (label={unique_labels[0]}, n={len(labels)})")
            mcc_val = 0.0
        # 检查是否所有预测都是同一个类别
        elif len(unique_preds) == 1:
            if debug:
                print(f"[DEBUG] MCC=0: All predictions are the same (pred={unique_preds[0]}, n={len(preds)})")
            mcc_val = 0.0
        else:
            try:
                # 捕获警告但不转换为异常，只检查返回值
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    mcc_val = matthews_corrcoef(labels, preds)
                    
                # 检查返回值是否为NaN或Inf
                if np.isnan(mcc_val) or np.isinf(mcc_val):
                    if debug:
                        print(f"[DEBUG] MCC is NaN/Inf after calculation, setting to 0.0")
                        print(f"[DEBUG]   unique_labels={unique_labels}, unique_preds={unique_preds}")
                        print(f"[DEBUG]   label_counts={label_counts[label_counts>0]}, pred_counts={pred_counts[pred_counts>0]}")
                    mcc_val = 0.0
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Error computing MCC: {e}, setting to 0.0")
                    print(f"[DEBUG]   unique_labels={unique_labels}, unique_preds={unique_preds}")
                    print(f"[DEBUG]   label_counts={label_counts[label_counts>0]}, pred_counts={pred_counts[pred_counts>0]}")
                mcc_val = 0.0
    
    metrics.update(dict(
        accuracy=acc, f1_macro=f1_macro, f1_micro=f1_micro, f1_weighted=f1_weighted,
        precision_macro=prec_macro, recall_macro=rec_macro,
        mcc=mcc_val
    ))
    try:
        # softmax 概率用于 AUC(OVR)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        prob = e / e.sum(axis=1, keepdims=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings(
                "ignore",
                message="No positive class found in y_true.*",
                category=UserWarning,
                module="sklearn.metrics._ranking",
            )
            metrics["auc_macro_ovr"] = float(
                roc_auc_score(labels, prob, multi_class="ovr", average="macro")
            )
    except Exception:
        pass
    try:
        # AUPRC (macro, OVR) using one-vs-rest binarized labels
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        prob = e / e.sum(axis=1, keepdims=True)
        y_bin = label_binarize(labels, classes=np.arange(prob.shape[1]))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings(
                "ignore",
                message="No positive class found in y_true.*",
                category=UserWarning,
                module="sklearn.metrics._ranking",
            )
            metrics["auprc_macro_ovr"] = float(
                average_precision_score(y_bin, prob, average="macro")
            )
    except Exception:
        pass
    return metrics


class FineTuneSeqEvaluator:
    """
    冻结主干，仅微调分类头（MLPHead 或等价 nn.Module），专门用于单序列数据集：
      1) 预计算 train/val/test 的 embedding（来自 embedder.get_embedding）
      2) 训练分类头；每个 epoch 同时评估验证集与测试集
      3) 用验证集 F1(macro) 做早停；保存结果 JSON（及可选最佳权重）

    适用于单序列数据集（如 GUEDataset），数据集 __getitem__ 返回 (seq, seq, label) 或 (wt_seq, mt_seq, label)
    注意：对于单序列数据集，wt_seq 和 mt_seq 通常是相同的，所以只使用一个序列的 embedding
    """

    def __init__(
        self,
        embedder,                       # 可选：如 CaduceusModel 实例，需提供 .get_embedding()
        model: nn.Module,                # 分类头（例如 MLPHead/FlexibleMLPHead）
        # 单序列数据集（__getitem__ -> (wt_seq, mt_seq, label)）
        train_ds,
        val_ds,
        test_ds=None,
        output_dir: str = "./runs/freeze_head",
        # 优化与训练
        task: Literal["binary", "multiclass", "regression"] = "multiclass",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        num_epochs: int = 100,
        batch_size: int = 128,
        num_workers: int = 0,
        class_weights: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,  # CrossEntropy 的权重（单任务为Tensor，多任务为Dict[task_name, Tensor]）
        # 早停
        early_stopping_patience: int = 5,
        early_stopping_min_delta: float = 1e-4,
        early_stopping_metric: str = "mcc",  # 早停指标：mcc, accuracy, f1_macro
        # embedding 计算参数
        emb_pool: Literal["final", "mean"] = "final",
        emb_batch_size: int = 128,
        emb_chunk_size: int = 2048,
        emb_average_rc: bool = False,    # 预留；如需可在 embedder 内部支持
        # 保存
        save_checkpoints: bool = True,   # 保存最佳头部权重
        embedding_save_dir: Optional[str] = None,  # 预计算 embedding 缓存目录
        force_recompute_embeddings: bool = False,  # 是否强制重新计算 embedding（忽略缓存）
        seed: int = 2025,
        device: Optional[str] = None,
        # evo2 额外参数（可指定层）
        emb_layer_name: Optional[str] = None,
        # 可选：对每个向量做 L2 归一化（有助于数值稳定）
        emb_l2norm: bool = False,
        # 多任务分类
        multitask: bool = False,
        task_names: Optional[List[str]] = None,
        # 调试模式
        debug: bool = False,
        # 保存预测结果
        save_predictions: bool = False,  # 是否保存预测结文件
    ):
        super().__init__()
        self.embedder = embedder
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.debug = debug

        self.task = task
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.class_weights = class_weights
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_metric = early_stopping_metric.lower()  # 转换为小写

        self.emb_pool = emb_pool
        self.emb_batch_size = emb_batch_size
        self.emb_chunk_size = emb_chunk_size
        self.emb_average_rc = emb_average_rc
        self.emb_layer_name = emb_layer_name
        self.save_checkpoints = save_checkpoints
        self.embedding_save_dir = embedding_save_dir
        self.force_recompute_embeddings = force_recompute_embeddings
        self.seed = seed
        self.device = torch.device(device or (
            "cuda:0" if torch.cuda.is_available() else "cpu"))

        self.emb_l2norm = emb_l2norm
        self.multitask = multitask
        self.task_names = list(task_names) if task_names is not None else None
        self.save_predictions = save_predictions
        self._train_batch_sampler = None

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    # ---------- Embedding 底层统一接口 ----------
    def _normalize_embeddings_output(self, embs: Any) -> np.ndarray:
        """
        统一 embedding 输出为 (N, D) 的 numpy 数组，处理 list/torch/numpy 等情况。
        """
        if isinstance(embs, torch.Tensor):
            embs = embs.detach().cpu().numpy()
        if isinstance(embs, np.ndarray):
            arr = embs
        elif isinstance(embs, list):
            if len(embs) == 0:
                return np.zeros((0, 0), dtype=np.float32)
            arrs: List[np.ndarray] = []
            for e in embs:
                a = np.asarray(e)
                if a.ndim == 1:
                    a = a.reshape(1, -1)
                elif a.ndim == 2 and a.shape[0] == 1:
                    pass
                elif a.ndim == 3:
                    if self.emb_pool == "mean":
                        a = a.mean(axis=1)
                    elif self.emb_pool == "final":
                        a = a[:, -1, :]
                    else:
                        raise ValueError(
                            f"Unknown emb_pool={self.emb_pool} for 3D embeddings")
                elif a.ndim != 2:
                    raise ValueError(
                        f"Unexpected embedding ndim={a.ndim} with shape {a.shape}")
                arrs.append(a)
            try:
                arr = np.concatenate(arrs, axis=0)
            except ValueError as exc:
                raise ValueError(
                    "Embedding outputs have inconsistent shapes; "
                    "check pooling/layer selection and input sequences."
                ) from exc
        else:
            arr = np.asarray(embs)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim == 3:
            if self.emb_pool == "mean":
                arr = arr.mean(axis=1)
            elif self.emb_pool == "final":
                arr = arr[:, -1, :]
            else:
                raise ValueError(
                    f"Unknown emb_pool={self.emb_pool} for 3D embeddings")
        if arr.ndim != 2:
            raise ValueError(
                f"Expected 2D embeddings, got shape {arr.shape}")
        return arr

    def _embed_batch_inputs(
        self,
        seqs: List[str],
    ) -> np.ndarray:
        """
        根据当前 embedder 类型与配置，批量拿到 (N, D) 的 np.float32 embedding。
        兼容 gpn-msa / 普通序列 / layer_name。
        """
        if self.emb_layer_name is not None:
            embs = self.embedder.get_embedding(
                seqs,
                batch_size=self.emb_batch_size,
                pool=self.emb_pool,
                layer_name=self.emb_layer_name,
                return_numpy=True,
            )
        else:
            embs = self.embedder.get_embedding(
                seqs,
                batch_size=self.emb_batch_size,
                pool=self.emb_pool,
                return_numpy=True,
            )
        embs = self._normalize_embeddings_output(embs).astype(
            np.float32, copy=False)
        if self.emb_l2norm:
            denom = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
            embs = embs / denom
        return embs

    def _is_cnn_embedder(self) -> bool:
        try:
            from models.cnn import GenomeCNN1D
        except Exception:
            return False
        if self.embedder is not None:
            return isinstance(self.embedder, GenomeCNN1D)
        return isinstance(self.model, GenomeCNN1D)

    def _embed_batch_inputs_cnn(
        self,
        seqs: List[str],
    ) -> np.ndarray:
        """
        CNN 专用 embedding 提取：
        - 将序列映射为 token ids (A/C/G/T/N -> 1..5, PAD=0)
        - 使用 GenomeCNN1D.forward_features 获取 pooled 特征
        """
        if not seqs:
            return np.zeros((0, 0), dtype=np.float32)

        vocab = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5}
        cnn_model = self.embedder
        if cnn_model is None:
            raise ValueError("CNN embedder is None; cannot extract embeddings.")
        pad_idx = getattr(getattr(cnn_model, "cfg", None), "pad_idx", 0)
        device = self.device
        try:
            device = next(cnn_model.parameters()).device
        except StopIteration:
            pass

        outputs: List[np.ndarray] = []
        cnn_model.eval()
        with torch.no_grad():
            for start in range(0, len(seqs), self.emb_batch_size):
                batch = seqs[start:start + self.emb_batch_size]
                max_len = max(len(s) for s in batch)
                tokens = torch.full(
                    (len(batch), max_len), pad_idx, dtype=torch.long)
                for i, seq in enumerate(batch):
                    if isinstance(seq, list):
                        seq = "".join(seq)
                    s = str(seq).upper()
                    ids = [vocab.get(ch, 5) for ch in s]
                    if ids:
                        tokens[i, :len(ids)] = torch.tensor(
                            ids, dtype=torch.long)
                tokens = tokens.to(device)
                feats = cnn_model.forward_features(tokens)
                outputs.append(feats.detach().cpu().numpy())

        embs = np.concatenate(outputs, axis=0).astype(np.float32, copy=False)
        if self.emb_l2norm:
            denom = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
            embs = embs / denom
        return embs

    def _seqs_to_tokens(self, seqs: List[str]) -> torch.Tensor:
        vocab = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5}
        pad_idx = getattr(getattr(self.model, "cfg", None), "pad_idx", 0)
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = self.device
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

    def _parse_batch_items(self, items: List[Any]) -> Tuple[List[str], torch.Tensor, Optional[torch.Tensor]]:
        seqs: List[str] = []
        labels: List[Any] = []
        group_ids: List[Optional[int]] = []
        for item in items:
            if isinstance(item, dict):
                seqs.append(item["sequence"])
                lab = item["labels"]
                if isinstance(lab, (list, tuple, np.ndarray)):
                    if not self.multitask:
                        raise ValueError("multitask labels require multitask=True")
                    labels.append(np.asarray(lab, dtype=np.int64))
                else:
                    labels.append(int(lab))
                group_ids.append(int(item["seq_index"]) if "seq_index" in item and item["seq_index"] is not None else None)
                continue
            if not isinstance(item, (tuple, list)):
                raise ValueError("dataset item must be tuple/list/dict")
            if len(item) == 3:
                _, mt_seq, lab = item
                seqs.append(mt_seq)
                labels.append(int(lab))
                group_ids.append(None)
            elif len(item) == 2:
                seq, lab = item
                seqs.append(seq)
                if isinstance(lab, (list, tuple, np.ndarray)):
                    if not self.multitask:
                        raise ValueError("multitask labels require multitask=True")
                    labels.append(np.asarray(lab, dtype=np.int64))
                else:
                    labels.append(int(lab))
                group_ids.append(None)
            else:
                raise ValueError(
                    f"数据集 __getitem__ 返回了 {len(item)} 个值，期望 2/3 或 dict")
        if self.multitask:
            labels_arr = np.stack(
                [np.asarray(l, dtype=np.int64).reshape(-1)
                 if not np.isscalar(l) else np.asarray([l], dtype=np.int64)
                 for l in labels],
                axis=0,
            ) if labels else np.zeros((0, 0), dtype=np.int64)
            if self.task_names is None and labels_arr.size:
                self.task_names = [f"task{i}" for i in range(labels_arr.shape[1])]
            labels_t = torch.from_numpy(labels_arr.astype(np.int64))
        else:
            labels_t = torch.tensor(labels, dtype=torch.long)
        if any(g is not None for g in group_ids) and not all(g is not None for g in group_ids):
            group_t = None
        elif all(g is not None for g in group_ids):
            group_t = torch.tensor([int(g) for g in group_ids], dtype=torch.long)
        else:
            group_t = None
        return seqs, labels_t, group_t

    def _labels_list_to_tensor(self, labels: List[Any]) -> torch.Tensor:
        if self.multitask:
            if len(labels) == 0:
                return torch.zeros((0, 0), dtype=torch.long)
            labels_arr = np.stack(
                [np.asarray(l, dtype=np.int64).reshape(-1)
                 if not np.isscalar(l) else np.asarray([l], dtype=np.int64)
                 for l in labels],
                axis=0,
            )
            if self.task_names is None and labels_arr.size:
                self.task_names = [
                    f"task{i}" for i in range(labels_arr.shape[1])]
            return torch.from_numpy(labels_arr.astype(np.int64))
        return torch.tensor(labels, dtype=torch.long)

    def _normalize_multitask_logits(self, logits: Any) -> Dict[str, torch.Tensor]:
        if isinstance(logits, dict):
            if self.task_names is None:
                self.task_names = list(logits.keys())
            missing = [t for t in self.task_names if t not in logits]
            if missing:
                raise KeyError(f"multitask logits missing keys: {missing}")
            return {t: logits[t] for t in self.task_names}
        if isinstance(logits, (list, tuple)):
            if self.task_names is None:
                self.task_names = [f"task{i}" for i in range(len(logits))]
            if len(logits) != len(self.task_names):
                raise ValueError(
                    "multitask logits length does not match task_names")
            return {t: logits[i] for i, t in enumerate(self.task_names)}
        raise ValueError(
            "multitask model output must be dict/list/tuple of logits")

    @staticmethod
    def _infer_task_type_from_logits(logits: np.ndarray) -> Literal["binary", "multiclass"]:
        if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
            return "binary"
        return "multiclass"

    # ---------- Embedding 提取（单序列数据集） ----------
    def _extract_embeddings(self, dataset) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        从单序列数据集提取 (wt_seq, mt_seq, label)，
        对于单序列数据集，wt_seq 和 mt_seq 通常是相同的，所以只使用一个序列的 embedding。
        返回:
          feats: [N, D] (float32, CPU)
          labels: [N] (long, CPU)
        """
        seqs, labels = [], []
        group_ids: List[Optional[int]] = []
        for i in range(len(dataset)):
            item = dataset[i]
            # 支持 dict（例如 windowed dataset 返回 seq_index/window_index 等元信息）
            if isinstance(item, dict):
                if "sequence" not in item:
                    raise KeyError("dataset item dict missing key: sequence")
                if "labels" not in item:
                    raise KeyError("dataset item dict missing key: labels")
                seqs.append(item["sequence"])
                lab = item["labels"]
                if isinstance(lab, (list, tuple, np.ndarray)):
                    if not self.multitask:
                        raise ValueError(
                            "FineTuneSeqEvaluator expects scalar label when multitask=False.")
                    labels.append(np.asarray(lab, dtype=np.int64))
                else:
                    labels.append(int(lab))
                group_ids.append(int(
                    item["seq_index"]) if "seq_index" in item and item["seq_index"] is not None else None)
                continue

            # tuple/list：兼容旧数据集格式
            if not isinstance(item, (tuple, list)):
                raise ValueError("dataset item must be tuple/list/dict")
            if len(item) == 3:
                wt_seq, mt_seq, lab = item
                seqs.append(mt_seq)
                labels.append(int(lab))
                group_ids.append(None)
            elif len(item) == 2:
                seq, lab = item
                seqs.append(seq)
                if isinstance(lab, (list, tuple, np.ndarray)):
                    if not self.multitask:
                        raise ValueError(
                            "FineTuneSeqEvaluator expects scalar label when multitask=False.")
                    labels.append(np.asarray(lab, dtype=np.int64))
                else:
                    labels.append(int(lab))
                group_ids.append(None)
            else:
                raise ValueError(
                    f"数据集 __getitem__ 返回了 {len(item)} 个值，期望 2/3 或 dict")

        if self._is_cnn_embedder():
            embs = self._embed_batch_inputs_cnn(seqs)
        else:
            embs = self._embed_batch_inputs(seqs)
        feats_np = embs  # (N, D)

        feats = torch.from_numpy(feats_np.astype(np.float32))  # CPU
        if self.multitask:
            if len(labels) == 0:
                labels_t = torch.zeros((0, 0), dtype=torch.long)
            else:
                labels_arr = np.stack(
                    [np.asarray(l, dtype=np.int64).reshape(-1)
                     if not np.isscalar(l) else np.asarray([l], dtype=np.int64)
                     for l in labels],
                    axis=0,
                )
                if self.task_names is None:
                    self.task_names = [
                        f"task{i}" for i in range(labels_arr.shape[1])]
                labels_t = torch.from_numpy(labels_arr.astype(np.int64))
        else:
            labels_t = torch.tensor(labels, dtype=torch.long)      # CPU
        
        # 如果所有样本都有 seq_index（windowed dataset），按 seq_index 聚合 embedding
        # 这样可以大幅减少存储空间（从 N 个窗口减少到 M 个序列，M << N）
        if all(g is not None for g in group_ids):
            group_t = torch.tensor([int(g) for g in group_ids], dtype=torch.long)
            # 按 seq_index 聚合：对每个唯一的 seq_index，取该组内所有窗口 embedding 的平均值
            unique_groups, inverse_indices = torch.unique(group_t, return_inverse=True)
            num_unique = len(unique_groups)
            emb_dim = feats.shape[1]
            
            # 使用 index_add_ 和计数来聚合
            aggregated_feats = torch.zeros((num_unique, emb_dim), dtype=feats.dtype)
            counts = torch.zeros(num_unique, dtype=torch.long)
            aggregated_feats.index_add_(0, inverse_indices, feats)
            counts.index_add_(0, inverse_indices, torch.ones(len(group_ids), dtype=torch.long))
            # 取平均
            aggregated_feats = aggregated_feats / counts.unsqueeze(1).clamp(min=1).float()
            
            # 聚合后的 labels：每个 seq_index 只保留一个 label（应该都相同）
            aggregated_labels = torch.zeros(num_unique, dtype=labels_t.dtype)
            if self.multitask:
                aggregated_labels = torch.zeros((num_unique, labels_t.shape[1]), dtype=labels_t.dtype)
            for idx, gid in enumerate(unique_groups):
                mask = (group_t == gid)
                # 取第一个 label（所有窗口的 label 应该相同）
                aggregated_labels[idx] = labels_t[mask][0]
            
            feats = aggregated_feats
            labels_t = aggregated_labels
            group_t = unique_groups  # 使用唯一的 seq_index
            print(f"Embedding shape (after aggregation): {tuple(feats.shape)} (aggregated from {len(group_ids)} windows to {num_unique} sequences)")
        elif any(g is not None for g in group_ids):
            # 混合情况很危险：禁用聚合，保持 window-level 评估
            group_t = None
            print("Embedding shape:", tuple(feats.shape), "(mixed seq_index, no aggregation)")
        else:
            group_t = None
            print("Embedding shape:", tuple(feats.shape))
        
        return feats, labels_t, group_t

    def _extract_embeddings_chunked(
        self,
        dataset,
        split: Optional[str] = None,
        return_full: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        分批提取 embedding，并按 chunk 保存到磁盘（可选）。
        返回拼接后的全量 feats/labels/groups。
        """
        chunk_size = max(int(self.emb_chunk_size or 0), 1)
        seqs: List[str] = []
        labels: List[Any] = []
        group_ids: List[Optional[int]] = []
        feats_chunks: List[torch.Tensor] = []
        labels_chunks: List[torch.Tensor] = []
        groups_chunks: List[Optional[torch.Tensor]] = []
        chunk_idx = 0

        group_mode: Literal["unknown", "all", "none", "mixed"] = "unknown"
        chunk_dir = None
        if self.embedding_save_dir:
            tag = f"{split}_chunks" if split else "chunks"
            chunk_dir = os.path.join(self.embedding_save_dir, tag)
            os.makedirs(chunk_dir, exist_ok=True)

        def flush_chunk():
            nonlocal chunk_idx
            if not seqs:
                return
            if self._is_cnn_embedder():
                embs = self._embed_batch_inputs_cnn(seqs)
            else:
                embs = self._embed_batch_inputs(seqs)
            feats_t = torch.from_numpy(embs.astype(np.float32, copy=False))
            labels_t = self._labels_list_to_tensor(labels)
            
            # 如果所有样本都有 seq_index（windowed dataset），按 seq_index 聚合 embedding
            if group_mode == "all" and all(g is not None for g in group_ids):
                group_t = torch.tensor([int(g) for g in group_ids], dtype=torch.long)
                # 按 seq_index 聚合：对每个唯一的 seq_index，取该组内所有窗口 embedding 的平均值
                unique_groups, inverse_indices = torch.unique(group_t, return_inverse=True)
                num_unique = len(unique_groups)
                emb_dim = feats_t.shape[1]
                
                # 使用 index_add_ 和计数来聚合
                aggregated_feats = torch.zeros((num_unique, emb_dim), dtype=feats_t.dtype)
                counts = torch.zeros(num_unique, dtype=torch.long)
                aggregated_feats.index_add_(0, inverse_indices, feats_t)
                counts.index_add_(0, inverse_indices, torch.ones(len(group_ids), dtype=torch.long))
                # 取平均
                aggregated_feats = aggregated_feats / counts.unsqueeze(1).clamp(min=1).float()
                
                # 聚合后的 labels：每个 seq_index 只保留一个 label（应该都相同）
                aggregated_labels = torch.zeros(num_unique, dtype=labels_t.dtype)
                if self.multitask:
                    aggregated_labels = torch.zeros((num_unique, labels_t.shape[1]), dtype=labels_t.dtype)
                for idx, gid in enumerate(unique_groups):
                    mask = (group_t == gid)
                    # 取第一个 label（所有窗口的 label 应该相同）
                    aggregated_labels[idx] = labels_t[mask][0]
                
                feats_t = aggregated_feats
                labels_t = aggregated_labels
                group_t = unique_groups  # 使用唯一的 seq_index
            elif group_mode == "all":
                group_t = torch.tensor(
                    [int(g) for g in group_ids], dtype=torch.long)
            else:
                group_t = None
            
            if return_full:
                feats_chunks.append(feats_t)
                labels_chunks.append(labels_t)
                groups_chunks.append(group_t)
            if chunk_dir is not None:
                torch.save(
                    {"feats": feats_t, "labels": labels_t, "groups": group_t},
                    os.path.join(
                        chunk_dir, f"chunk_{chunk_idx:06d}.pt"),
                )
            chunk_idx += 1
            seqs.clear()
            labels.clear()
            group_ids.clear()

        total = len(dataset)
        pbar = tqdm(
            total=total,
            desc=f"{split or 'all'} embedding",
            position=1,
            leave=True,
        )
        try:
            for i in range(total):
                item = dataset[i]
                if isinstance(item, dict):
                    if "sequence" not in item or "labels" not in item:
                        raise KeyError("dataset item dict missing key: sequence/labels")
                    seqs.append(item["sequence"])
                    lab = item["labels"]
                    if isinstance(lab, (list, tuple, np.ndarray)):
                        if not self.multitask:
                            raise ValueError(
                                "FineTuneSeqEvaluator expects scalar label when multitask=False.")
                        labels.append(np.asarray(lab, dtype=np.int64))
                    else:
                        labels.append(int(lab))
                    gid = int(item["seq_index"]) if "seq_index" in item and item["seq_index"] is not None else None
                    group_ids.append(gid)
                else:
                    if not isinstance(item, (tuple, list)):
                        raise ValueError("dataset item must be tuple/list/dict")
                    if len(item) == 3:
                        _, mt_seq, lab = item
                        seqs.append(mt_seq)
                        labels.append(int(lab))
                        group_ids.append(None)
                    elif len(item) == 2:
                        seq, lab = item
                        seqs.append(seq)
                        if isinstance(lab, (list, tuple, np.ndarray)):
                            if not self.multitask:
                                raise ValueError(
                                    "FineTuneSeqEvaluator expects scalar label when multitask=False.")
                            labels.append(np.asarray(lab, dtype=np.int64))
                        else:
                            labels.append(int(lab))
                        group_ids.append(None)
                    else:
                        raise ValueError(
                            f"数据集 __getitem__ 返回了 {len(item)} 个值，期望 2/3 或 dict")

                if group_ids[-1] is None:
                    if group_mode == "unknown":
                        group_mode = "none"
                    elif group_mode == "all":
                        group_mode = "mixed"
                else:
                    if group_mode == "unknown":
                        group_mode = "all"
                    elif group_mode == "none":
                        group_mode = "mixed"

                if len(seqs) >= chunk_size:
                    flush_chunk()
                pbar.update(1)
        finally:
            pbar.close()

        flush_chunk()

        if return_full:
            feats = torch.cat(feats_chunks, dim=0) if feats_chunks else torch.zeros(
                (0, 0), dtype=torch.float32)
            if labels_chunks:
                labels_t = torch.cat(labels_chunks, dim=0)
            else:
                labels_t = torch.zeros(
                    (0, 0), dtype=torch.long) if self.multitask else torch.zeros(
                    (0,), dtype=torch.long)
            if group_mode == "all" and any(g is not None for g in groups_chunks):
                groups_t = torch.cat([g for g in groups_chunks if g is not None], dim=0)
            else:
                groups_t = None
            return feats, labels_t, groups_t
        empty_labels = torch.zeros((0, 0), dtype=torch.long) if self.multitask else torch.zeros(
            (0,), dtype=torch.long)
        return torch.zeros((0, 0), dtype=torch.float32), empty_labels, None


    def _make_loaders(self) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        if self.embedder is None:
            collate = lambda x: x
            train_loader = DataLoader(
                self.train_ds, batch_size=self.batch_size, shuffle=True,
                drop_last=False, collate_fn=collate, num_workers=self.num_workers)
            val_loader = DataLoader(
                self.val_ds, batch_size=self.batch_size, shuffle=False,
                drop_last=False, collate_fn=collate, num_workers=self.num_workers)
            if self.test_ds is not None:
                test_loader = DataLoader(
                    self.test_ds, batch_size=self.batch_size, shuffle=False,
                    drop_last=False, collate_fn=collate, num_workers=self.num_workers)
            else:
                test_loader = None
            return train_loader, val_loader, test_loader

        def _load_or_extract(split: str, dataset):
            cache_dir = self.embedding_save_dir
            use_chunked = self.emb_chunk_size is not None and self.emb_chunk_size > 0
            if use_chunked and not cache_dir:
                raise ValueError(
                    "emb_chunk_size>0 requires embedding_save_dir for low-memory mode.")
            if cache_dir and not self.force_recompute_embeddings:
                os.makedirs(cache_dir, exist_ok=True)
                cache_path = os.path.join(cache_dir, f"{split}_embeddings.pt")
                if os.path.exists(cache_path):
                    cached = torch.load(cache_path, map_location="cpu", weights_only=True)
                    return cached["feats"], cached["labels"], cached.get("groups")
                chunk_dir = os.path.join(cache_dir, f"{split}_chunks")
                if os.path.isdir(chunk_dir):
                    chunk_files = sorted(
                        glob.glob(os.path.join(chunk_dir, "chunk_*.pt")))
                    if chunk_files:
                        chunks = [
                            torch.load(cf, map_location="cpu", weights_only=True)
                            for cf in chunk_files
                        ]
                        feats = torch.cat([c["feats"] for c in chunks], dim=0)
                        labels = torch.cat([c["labels"] for c in chunks], dim=0)
                        groups_list = [c.get("groups") for c in chunks]
                        if all(g is not None for g in groups_list):
                            groups = torch.cat(groups_list, dim=0)
                        else:
                            groups = None
                        return feats, labels, groups
            if use_chunked:
                feats, labels, groups = self._extract_embeddings_chunked(
                    dataset, split=split)
            else:
                feats, labels, groups = self._extract_embeddings(dataset)
            if cache_dir:
                torch.save(
                    {"feats": feats, "labels": labels, "groups": groups},
                    cache_path,
                )
            return feats, labels, groups

        use_chunked = self.emb_chunk_size is not None and self.emb_chunk_size > 0
        if use_chunked:
            if not self.embedding_save_dir:
                raise ValueError(
                    "emb_chunk_size>0 requires embedding_save_dir for low-memory mode.")
            for split, ds in [("train", self.train_ds), ("val", self.val_ds), ("test", self.test_ds)]:
                if ds is None:
                    continue
                chunk_dir = os.path.join(self.embedding_save_dir, f"{split}_chunks")
                if self.force_recompute_embeddings or not os.path.isdir(chunk_dir) or not glob.glob(os.path.join(chunk_dir, "chunk_*.pt")):
                    if self.force_recompute_embeddings and os.path.isdir(chunk_dir):
                        # 删除旧的 chunk 文件
                        import shutil
                        shutil.rmtree(chunk_dir)
                    self._extract_embeddings_chunked(ds, split=split, return_full=False)

            train_ds = ChunkedEmbeddingDataset(
                os.path.join(self.embedding_save_dir, "train_chunks"))
            val_ds = ChunkedEmbeddingDataset(
                os.path.join(self.embedding_save_dir, "val_chunks"))
            test_ds = (ChunkedEmbeddingDataset(
                os.path.join(self.embedding_save_dir, "test_chunks"))
                if self.test_ds is not None else None)

            self._train_batch_sampler = ChunkSequentialBatchSampler(
                train_ds,
                batch_size=self.batch_size,
                drop_last=False,
                shuffle_within_chunk=True,
                seed=self.seed,
            )
            train_loader = DataLoader(
                train_ds,
                batch_sampler=self._train_batch_sampler,
                num_workers=self.num_workers,
            )
            val_loader = DataLoader(
                val_ds, batch_size=self.batch_size, shuffle=False,
                drop_last=False, num_workers=self.num_workers)
            if test_ds is None:
                test_loader = None
            else:
                test_loader = DataLoader(
                    test_ds, batch_size=self.batch_size, shuffle=False,
                    num_workers=self.num_workers)
            return train_loader, val_loader, test_loader

        trX, trY, _ = _load_or_extract("train", self.train_ds)
        vaX, vaY, vaG = _load_or_extract("val", self.val_ds)
        if self.test_ds is not None:
            teX, teY, teG = _load_or_extract("test", self.test_ds)
        else:
            teX = teY = teG = None

        # 调试：检查加载的标签分布
        if self.debug:
            print("[DEBUG] 检查数据加载器中的标签分布...")
            if self.multitask:
                if trY.ndim == 2:
                    for ti, tname in enumerate(self.task_names or []):
                        if ti < trY.shape[1]:
                            task_labels = trY[:, ti].numpy()
                            valid_task_labels = task_labels[task_labels >= 0]
                            if len(valid_task_labels) > 0:
                                _debug_y(f"loader_train_{tname}", valid_task_labels, debug=self.debug)
                if vaY.ndim == 2:
                    for ti, tname in enumerate(self.task_names or []):
                        if ti < vaY.shape[1]:
                            task_labels = vaY[:, ti].numpy()
                            valid_task_labels = task_labels[task_labels >= 0]
                            if len(valid_task_labels) > 0:
                                _debug_y(f"loader_val_{tname}", valid_task_labels, debug=self.debug)
            else:
                trY_np = trY.numpy()
                valid_trY = trY_np[trY_np >= 0] if trY_np.ndim == 1 else trY_np
                if len(valid_trY) > 0:
                    _debug_y("loader_train", valid_trY, debug=self.debug)
                vaY_np = vaY.numpy()
                valid_vaY = vaY_np[vaY_np >= 0] if vaY_np.ndim == 1 else vaY_np
                if len(valid_vaY) > 0:
                    _debug_y("loader_val", valid_vaY, debug=self.debug)

        train_loader = DataLoader(TensorDataset(
            trX, trY), batch_size=self.batch_size, shuffle=True,
            drop_last=False, num_workers=self.num_workers)
        if vaG is None:
            val_loader = DataLoader(TensorDataset(
                vaX, vaY), batch_size=self.batch_size, shuffle=False,
                drop_last=False, num_workers=self.num_workers)
        else:
            val_loader = DataLoader(TensorDataset(
                vaX, vaY, vaG), batch_size=self.batch_size, shuffle=False,
                drop_last=False, num_workers=self.num_workers)

        if teX is None:
            test_loader = None
        else:
            if teG is None:
                test_loader = DataLoader(TensorDataset(
                    teX, teY), batch_size=self.batch_size, shuffle=False,
                    num_workers=self.num_workers)
            else:
                test_loader = DataLoader(TensorDataset(
                    teX, teY, teG), batch_size=self.batch_size, shuffle=False,
                    num_workers=self.num_workers)
        return train_loader, val_loader, test_loader

    # ---------- 训练 / 评估 ----------
    def _build_criterion(self):
        if self.task == "regression":
            return nn.MSELoss()
        if self.task == "binary":
            return nn.BCEWithLogitsLoss()
        # multiclass
        cw = None
        if self.class_weights is not None:
            # 单任务：class_weights是Tensor；多任务：是Dict（但多任务不使用此方法）
            if isinstance(self.class_weights, torch.Tensor):
                cw = self.class_weights.to(self.device, dtype=torch.float32)
            elif isinstance(self.class_weights, dict) and self.task_names and len(self.task_names) == 1:
                # 单任务但传入了字典格式，取第一个任务的权重
                cw = self.class_weights[self.task_names[0]].to(self.device, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=cw)

    def _run_epoch(self, loader: DataLoader, train: bool) -> Tuple[Any, np.ndarray, Optional[np.ndarray], float]:
        self.model.train(train)
        total_loss = 0.0
        all_logits: Any = [] if not self.multitask else {}
        all_labels: List[np.ndarray] = []
        all_groups: List[np.ndarray] = []
        invalid_counts: Dict[str, int] = {} if self.multitask else {}

        pbar = tqdm(loader, desc="train" if train else "eval", leave=False)
        for batch in pbar:
            if self.embedder is None:
                seqs, yb, gb = self._parse_batch_items(batch)
                xb = self._seqs_to_tokens(seqs)
                yb = yb.to(xb.device)
            else:
                if len(batch) == 2:
                    xb, yb = batch
                    gb = None
                elif len(batch) == 3:
                    xb, yb, gb = batch
                else:
                    raise ValueError(
                        "Unexpected batch size from loader (expected 2 or 3 tensors)")
                xb = xb.to(self.device, dtype=torch.float32)
                yb = yb.to(self.device)

            if train:
                self.opt.zero_grad(set_to_none=True)

            if not self.multitask:
                logits = self.model(xb)  # [B, C] 或 [B, 1]
                if self.task == "binary":
                    # BCEWithLogits 需要浮点 targets
                    loss = self.criterion(logits.view(-1), yb.float())
                elif self.task == "multiclass":
                    loss = self.criterion(logits, yb)
                else:
                    loss = self.criterion(logits.view(-1), yb.float())
            else:
                logits = self.model(xb)
                logits_by_task = self._normalize_multitask_logits(logits)
                if not isinstance(all_logits, dict) or not all_logits:
                    all_logits = {t: [] for t in logits_by_task}
                loss = 0.0
                for ti, tname in enumerate(self.task_names or []):
                    logit_t = logits_by_task[tname]
                    yb_t = yb[:, ti]
                    task_type = self._infer_task_type_from_logits(
                        _to_numpy(logit_t.detach()))
                    if task_type == "binary":
                        valid = (yb_t >= 0) & (yb_t <= 1)
                        if valid.any():
                            loss_t = nn.BCEWithLogitsLoss()(
                                logit_t.view(-1)[valid], yb_t.float()[valid])
                        else:
                            loss_t = torch.zeros((), device=logit_t.device)
                    else:
                        valid = (yb_t >= 0) & (yb_t < logit_t.size(1))
                        if valid.any():
                            # 支持多任务的类别权重
                            task_weights = None
                            if isinstance(self.class_weights, dict) and tname in self.class_weights:
                                task_weights = self.class_weights[tname].to(self.device, dtype=torch.float32)
                            elif isinstance(self.class_weights, torch.Tensor) and not self.multitask:
                                task_weights = self.class_weights.to(self.device, dtype=torch.float32)
                            loss_t = nn.CrossEntropyLoss(weight=task_weights)(
                                logit_t[valid], yb_t[valid])
                        else:
                            loss_t = torch.zeros((), device=logit_t.device)
                    invalid_counts[tname] = invalid_counts.get(
                        tname, 0) + int((~valid).sum().item())
                    loss = loss + loss_t
                if (self.task_names or []):
                    loss = loss / max(len(self.task_names or []), 1)

            if train:
                loss.backward()
                self.opt.step()

            total_loss += float(loss.detach().cpu()) * xb.size(0)
            pbar.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")
            if not self.multitask:
                all_logits.append(_to_numpy(logits.view(logits.size(0), -1)))
            else:
                for tname, logit_t in self._normalize_multitask_logits(logits).items():
                    all_logits[tname].append(
                        _to_numpy(logit_t.view(logit_t.size(0), -1)))
            all_labels.append(_to_numpy(yb))
            if gb is not None:
                all_groups.append(_to_numpy(gb))

        N = sum(len(a) for a in all_labels)
        avg_loss = total_loss / max(N, 1)
        if not self.multitask:
            logits_np = np.concatenate(all_logits, axis=0) if all_logits else np.zeros(
                (0, 1), dtype=np.float32)
        else:
            logits_np = {t: (np.concatenate(v, axis=0) if v else np.zeros((0, 1), dtype=np.float32))
                         for t, v in all_logits.items()}
        labels_np = np.concatenate(
            all_labels, axis=0) if all_labels else np.zeros((0,), dtype=np.int64)
        groups_np = np.concatenate(all_groups, axis=0) if all_groups else None
        
        # 调试：检查标签分布
        split_name = "train" if train else "eval"
        if self.multitask:
            bad = {k: v for k, v in invalid_counts.items() if v > 0}
            if bad:
                tqdm.write(f"[WARN] {split_name} invalid labels per task: {bad}")
            # 多任务：检查每个任务的标签
            if self.debug and labels_np.ndim == 2 and labels_np.shape[1] > 0:
                for ti, tname in enumerate(self.task_names or []):
                    if ti < labels_np.shape[1]:
                        task_labels = labels_np[:, ti]
                        valid_task_labels = task_labels[task_labels >= 0]
                        if len(valid_task_labels) > 0:
                            _debug_y(f"{split_name}_{tname}", valid_task_labels, debug=self.debug)
        else:
            # 单任务：检查标签
            if self.debug and len(labels_np) > 0:
                valid_labels = labels_np[labels_np >= 0] if labels_np.ndim == 1 else labels_np
                if len(valid_labels) > 0:
                    _debug_y(split_name, valid_labels, debug=self.debug)
        
        return logits_np, labels_np, groups_np, avg_loss

    @staticmethod
    def _aggregate_by_group_mean(
        logits: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将 window-level logits 聚合到 sequence-level：
        - 同 group 的 logits 取均值
        - label 取该 group 第一个样本的 label（假设同一序列各窗口 label 相同）
        """
        if logits.ndim == 1:
            logits = logits.reshape(-1, 1)
        if logits.shape[0] != labels.shape[0] or logits.shape[0] != groups.shape[0]:
            raise ValueError("logits/labels/groups length mismatch")

        order = np.argsort(groups, kind="mergesort")
        g = groups[order]
        l = labels[order]
        z = logits[order]

        uniq, idx_start = np.unique(g, return_index=True)
        # counts per group
        idx_end = np.append(idx_start[1:], len(g))
        counts = (idx_end - idx_start).astype(np.float32)

        # mean logits using reduceat
        sums = np.add.reduceat(z, idx_start, axis=0)
        mean_logits = sums / counts.reshape(-1, 1)
        group_labels = l[idx_start]
        return mean_logits, group_labels

    def _get_label2id(self) -> Dict[str, Any]:
        """获取label2id映射"""
        if hasattr(self.train_ds, "get_label_map"):
            return self.train_ds.get_label_map()  # type: ignore
        elif hasattr(self.train_ds, "label2id"):
            return getattr(self.train_ds, "label2id")
        elif hasattr(getattr(self.train_ds, "base", None), "label2id"):
            return getattr(getattr(self.train_ds, "base"), "label2id")
        return {}

    def _collect_metadata(self, loader: DataLoader, groups: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        从数据集中收集 taxid 和 first_release_date
        
        Returns:
            (taxids, first_release_dates): 两个数组，如果无法获取则返回 None
        """
        try:
            dataset = loader.dataset
            taxids_list = []
            first_release_dates_list = []
            
            # 检查是否是 windowed dataset
            base_dataset = None
            if hasattr(dataset, 'base'):
                base_dataset = dataset.base
            elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'base'):
                base_dataset = dataset.dataset.base
            
            # 获取基础数据集（包含原始 DataFrame）
            if base_dataset is not None and hasattr(base_dataset, 'df'):
                df = base_dataset.df
                has_first_release = 'first_release_date' in df.columns
                
                # 遍历数据集收集信息
                for idx in range(len(dataset)):
                    try:
                        item = dataset[idx]
                        if isinstance(item, dict):
                            seq_index = item.get('seq_index')
                            if seq_index is not None and seq_index < len(df):
                                taxid = df.iloc[seq_index]['taxid']
                                first_release_date = df.iloc[seq_index].get('first_release_date', None) if has_first_release else None
                                taxids_list.append(taxid)
                                first_release_dates_list.append(first_release_date)
                            else:
                                taxids_list.append(None)
                                first_release_dates_list.append(None)
                        else:
                            # tuple 格式，尝试从 base dataset 获取
                            if hasattr(dataset, '_flat_index') and idx < len(dataset._flat_index):
                                seq_index, _ = dataset._flat_index[idx]
                                if seq_index < len(df):
                                    taxid = df.iloc[seq_index]['taxid']
                                    first_release_date = df.iloc[seq_index].get('first_release_date', None) if has_first_release else None
                                    taxids_list.append(taxid)
                                    first_release_dates_list.append(first_release_date)
                                else:
                                    taxids_list.append(None)
                                    first_release_dates_list.append(None)
                            else:
                                taxids_list.append(None)
                                first_release_dates_list.append(None)
                    except Exception as e:
                        tqdm.write(f"[WARN] 获取索引 {idx} 的元数据失败: {e}")
                        taxids_list.append(None)
                        first_release_dates_list.append(None)
                
                # 如果有 groups，需要聚合（取每个 group 的第一个，与 _aggregate_by_group_mean 保持一致）
                if groups is not None and len(taxids_list) > 0:
                    groups_np = np.array(groups)
                    # 使用与 _aggregate_by_group_mean 相同的排序逻辑
                    order = np.argsort(groups_np, kind="mergesort")
                    g = groups_np[order]
                    taxids_ordered = [taxids_list[i] for i in order]
                    first_release_dates_ordered = [first_release_dates_list[i] for i in order]
                    
                    uniq, idx_start = np.unique(g, return_index=True)
                    # 取每个 group 的第一个元素
                    taxids_agg = [taxids_ordered[idx_start[i]] for i in range(len(uniq))]
                    first_release_dates_agg = [first_release_dates_ordered[idx_start[i]] for i in range(len(uniq))]
                    return np.array(taxids_agg), np.array(first_release_dates_agg)
                else:
                    return np.array(taxids_list), np.array(first_release_dates_list)
        except Exception as e:
            tqdm.write(f"[WARN] 收集元数据失败: {e}")
            return None, None
        
        return None, None

    def _save_predictions_to_csv(
        self,
        logits: Any,
        labels: np.ndarray,
        preds: Optional[np.ndarray],
        split_name: str,
        task_names: Optional[List[str]] = None,
        taxids: Optional[np.ndarray] = None,
        first_release_dates: Optional[np.ndarray] = None,
    ):
        """
        保存预测结果到CSV文件
        
        Args:
            logits: 模型输出的logits（单任务为np.ndarray，多任务为Dict[str, np.ndarray]）
            labels: 真实标签（单任务为1D数组，多任务为2D数组 [N, num_tasks]）
            preds: 预测类别（单任务时提供，多任务时从logits计算）
            split_name: 数据集名称（"train", "val", "test"）
            task_names: 任务名称列表（多任务时使用）
        """
        label2id = self._get_label2id()
        rows = []
        
        if not self.multitask:
            # 单任务
            if preds is None:
                if logits.ndim == 1 or logits.shape[1] == 1:
                    preds = (1/(1+np.exp(-logits.reshape(-1))) >= 0.5).astype(np.int64)
                else:
                    preds = logits.argmax(axis=-1)
            
            # 计算概率
            if logits.ndim == 1 or logits.shape[1] == 1:
                probs = 1 / (1 + np.exp(-logits.reshape(-1)))
            else:
                e = np.exp(logits - logits.max(axis=1, keepdims=True))
                probs = e / e.sum(axis=1, keepdims=True)
                probs = probs[np.arange(len(preds)), preds]
            
            # 获取类别名称
            id2label = {}
            if isinstance(label2id, dict) and len(label2id) > 0:
                if isinstance(list(label2id.values())[0], dict):
                    # 多任务格式但单任务使用
                    task_name = list(label2id.keys())[0]
                    id2label = {v: k for k, v in label2id[task_name].items()}
                else:
                    id2label = {v: k for k, v in label2id.items()}
            
            for i in range(len(labels)):
                true_id = int(labels[i])
                pred_id = int(preds[i])
                true_label = id2label.get(true_id, f"class_{true_id}")
                pred_label = id2label.get(pred_id, f"class_{pred_id}")
                confidence = float(probs[i])
                
                row = {
                    "true_label_id": true_id,
                    "true_label": true_label,
                    "predicted_label_id": pred_id,
                    "predicted_label": pred_label,
                    "confidence": confidence,
                }
                # 添加 taxid 和 first_release_date
                if taxids is not None and i < len(taxids):
                    row["taxid"] = taxids[i] if taxids[i] is not None else ""
                else:
                    row["taxid"] = ""
                if first_release_dates is not None and i < len(first_release_dates):
                    row["first_release_date"] = first_release_dates[i] if first_release_dates[i] is not None else ""
                else:
                    row["first_release_date"] = ""
                
                rows.append(row)
        else:
            # 多任务
            if task_names is None:
                task_names = self.task_names or []
            
            # 计算每个任务的预测和概率
            preds_by_task = {}
            probs_by_task = {}
            for tname in task_names:
                if tname in logits:
                    logit_t = logits[tname]
                    if logit_t.ndim == 1 or logit_t.shape[1] == 1:
                        preds_by_task[tname] = (1/(1+np.exp(-logit_t.reshape(-1))) >= 0.5).astype(np.int64)
                        probs_by_task[tname] = 1 / (1 + np.exp(-logit_t.reshape(-1)))
                    else:
                        preds_by_task[tname] = logit_t.argmax(axis=-1)
                        e = np.exp(logit_t - logit_t.max(axis=1, keepdims=True))
                        prob_t = e / e.sum(axis=1, keepdims=True)
                        probs_by_task[tname] = prob_t[np.arange(len(preds_by_task[tname])), preds_by_task[tname]]
            
            # 获取每个任务的id2label映射
            id2label_by_task = {}
            if isinstance(label2id, dict):
                for tname in task_names:
                    if tname in label2id:
                        id2label_by_task[tname] = {v: k for k, v in label2id[tname].items()}
            
            # 构建行数据
            num_samples = len(labels)
            for i in range(num_samples):
                row = {}
                for ti, tname in enumerate(task_names):
                    if ti < labels.shape[1]:
                        true_id = int(labels[i, ti])
                        if tname in preds_by_task and i < len(preds_by_task[tname]):
                            pred_id = int(preds_by_task[tname][i])
                            confidence = float(probs_by_task[tname][i])
                            
                            # 获取类别名称
                            id2label = id2label_by_task.get(tname, {})
                            true_label = id2label.get(true_id, f"class_{true_id}")
                            pred_label = id2label.get(pred_id, f"class_{pred_id}")
                            
                            row[f"{tname}_true_label_id"] = true_id
                            row[f"{tname}_true_label"] = true_label
                            row[f"{tname}_predicted_label_id"] = pred_id
                            row[f"{tname}_predicted_label"] = pred_label
                            row[f"{tname}_confidence"] = confidence
                
                # 添加 taxid 和 first_release_date
                if taxids is not None and i < len(taxids):
                    row["taxid"] = taxids[i] if taxids[i] is not None else ""
                else:
                    row["taxid"] = ""
                if first_release_dates is not None and i < len(first_release_dates):
                    row["first_release_date"] = first_release_dates[i] if first_release_dates[i] is not None else ""
                else:
                    row["first_release_date"] = ""
                
                rows.append(row)
        
        # 保存到CSV
        if rows:
            df = pd.DataFrame(rows)
            csv_path = os.path.join(self.output_dir, f"{split_name}_predictions.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8')
            tqdm.write(f"[INFO] 预测结果已保存到: {csv_path} (共 {len(rows)} 条记录)")

    def _eval_split(self, loader: DataLoader, split_name: str = "eval") -> Tuple[Any, Any, Any]:
        """
        评估单个数据集
        
        Args:
            loader: 数据加载器
            split_name: 数据集名称（"train", "val", "test"），用于保存预测结果时的文件名
        """
        logits, labels, groups, _ = self._run_epoch(loader, train=False)
        
        # 收集 taxid 和 first_release_date（用于保存预测结果）
        taxids = None
        first_release_dates = None
        if self.save_predictions and split_name == "test":
            taxids, first_release_dates = self._collect_metadata(loader, groups)
        if not self.multitask:
            # 若是 windowed val/test：按 seq_index 聚合到 sequence-level
            if groups is not None:
                logits, labels = self._aggregate_by_group_mean(
                    logits, labels, groups)
            task = self.task
            metrics = _compute_metrics_from_logits(logits, labels, task=task, debug=self.debug)
            # 明细报告（分类任务）
            report = {}
            cm = []
            if task != "regression":
                if logits.ndim == 1 or logits.shape[1] == 1:
                    preds = (1/(1+np.exp(-logits.reshape(-1)))
                             >= 0.5).astype(np.int64)
                else:
                    preds = logits.argmax(axis=-1)
                report = classification_report(
                    labels, preds, digits=4, output_dict=True, zero_division=0)
                cm = confusion_matrix(labels, preds).tolist()
                
                # 保存预测结果到CSV（仅测试集）
                if self.save_predictions and split_name == "test":
                    self._save_predictions_to_csv(
                        logits, labels, preds, split_name, task_names=None,
                        taxids=taxids, first_release_dates=first_release_dates
                    )
            
            return metrics, report, cm

        metrics_by_task: Dict[str, Any] = {}
        report_by_task: Dict[str, Any] = {}
        cm_by_task: Dict[str, Any] = {}
        
        # 先聚合所有任务的logits和labels（用于保存预测结果）
        logits_agg = {}
        labels_agg = None
        if groups is not None:
            # 需要聚合
            for ti, tname in enumerate(self.task_names or []):
                logits_t = logits[tname]
                labels_t = labels[:, ti]
                logits_t_agg, labels_t_agg = self._aggregate_by_group_mean(
                    logits_t, labels_t, groups)
                logits_agg[tname] = logits_t_agg
                if labels_agg is None:
                    labels_agg = np.zeros((len(labels_t_agg), len(self.task_names or [])), dtype=labels.dtype)
                labels_agg[:, ti] = labels_t_agg
        else:
            # 不需要聚合，直接使用
            logits_agg = logits
            labels_agg = labels
        
        # 保存预测结果（仅测试集，在过滤之前，保存所有样本）
        if self.save_predictions and split_name == "test":
            self._save_predictions_to_csv(
                logits_agg, labels_agg, None, split_name, task_names=self.task_names,
                taxids=taxids, first_release_dates=first_release_dates
            )
        
        for ti, tname in enumerate(self.task_names or []):
            logits_t = logits[tname]
            labels_t = labels[:, ti]
            if groups is not None:
                logits_t, labels_t = self._aggregate_by_group_mean(
                    logits_t, labels_t, groups)
            # 过滤非法标签（例如 -1 或超出类别范围）
            if logits_t.ndim == 1 or logits_t.shape[1] == 1:
                valid = (labels_t >= 0) & (labels_t <= 1)
            else:
                valid = (labels_t >= 0) & (labels_t < logits_t.shape[1])
            logits_t = logits_t[valid]
            labels_t = labels_t[valid]
            if logits_t.shape[0] == 0:
                metrics_by_task[tname] = {}
                report_by_task[tname] = {}
                cm_by_task[tname] = []
                continue
            
            # 诊断：检查过滤后的标签分布
            invalid_count = (~valid).sum()
            if invalid_count > 0:
                tqdm.write(f"[WARN] {tname}: 过滤了 {invalid_count} 个无效标签 (剩余 {len(labels_t)} 个有效样本)")
            
            # 检查有效标签的分布
            if len(labels_t) > 0:
                unique_labels = np.unique(labels_t)
                label_counts = np.bincount(labels_t)
                if len(unique_labels) == 1:
                    tqdm.write(f"[WARN] {tname}: 过滤后所有标签都相同 (label={unique_labels[0]}, n={len(labels_t)}), MCC将=0")
                elif len(unique_labels) < 3:
                    tqdm.write(f"[WARN] {tname}: 过滤后只有 {len(unique_labels)} 个不同标签 (labels={unique_labels}, counts={label_counts[label_counts>0]}), MCC可能很低")
            
            # 调试：检查每个任务的标签
            if len(labels_t) > 0:
                _debug_y(f"eval_{tname}", labels_t, debug=self.debug)
            
            task_type = self._infer_task_type_from_logits(logits_t)
            metrics_by_task[tname] = _compute_metrics_from_logits(
                logits_t, labels_t, task=task_type, debug=self.debug)
            if task_type != "regression":
                if logits_t.ndim == 1 or logits_t.shape[1] == 1:
                    preds = (1/(1+np.exp(-logits_t.reshape(-1)))
                             >= 0.5).astype(np.int64)
                else:
                    preds = logits_t.argmax(axis=-1)
                report_by_task[tname] = classification_report(
                    labels_t, preds, digits=4, output_dict=True, zero_division=0)
                cm_by_task[tname] = confusion_matrix(labels_t, preds).tolist()
        
        return metrics_by_task, report_by_task, cm_by_task

    # ---------- 主流程 ----------
    def run(self) -> Dict[str, Any]:
        # 1) 数据与特征
        train_loader, val_loader, test_loader = self._make_loaders()

        # 2) 头部模型与优化器
        self.model.to(self.device)
        self.criterion = self._build_criterion()
        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_val_metric = -1.0
        best_epoch = -1
        best_state: Optional[Dict[str, torch.Tensor]] = None
        epochs_bad = 0

        per_epoch_rows: List[Dict[str, Any]] = []

        # 3) 训练循环（每个 epoch 评估 val & test）
        def _mean_metric(metrics_by_task: Dict[str, Dict[str, float]], key: str) -> float:
            vals = [v.get(key) for v in metrics_by_task.values()
                    if v.get(key) is not None]
            return float(np.mean(vals)) if vals else -1.0

        for epoch in range(1, self.num_epochs + 1):
            if self._train_batch_sampler is not None:
                self._train_batch_sampler.set_epoch(epoch)
            # train
            _, _, _, train_loss = self._run_epoch(train_loader, train=True)

            # eval (val only; test is run once after training)
            val_metrics, _, _ = self._eval_split(val_loader, split_name="val")
            test_metrics = None

            # 记录
            row: Dict[str, Any] = {
                "epoch": epoch,
                "train_loss": float(train_loss),
            }
            if not self.multitask:
                row.update({
                    "val_f1_macro": val_metrics.get("f1_macro"),
                    "val_accuracy": val_metrics.get("accuracy"),
                    "val_auc_macro_ovr": val_metrics.get("auc_macro_ovr"),
                    "val_auprc": val_metrics.get("auprc") or val_metrics.get("auprc_macro_ovr"),
                    "val_mcc": val_metrics.get("mcc"),
                })
            else:
                for tname, m in val_metrics.items():
                    row[f"val_f1_macro_{tname}"] = m.get("f1_macro")
                    row[f"val_accuracy_{tname}"] = m.get("accuracy")
                    row[f"val_auc_macro_ovr_{tname}"] = m.get("auc_macro_ovr")
                    row[f"val_auprc_{tname}"] = m.get("auprc") or m.get("auprc_macro_ovr")
                    row[f"val_mcc_{tname}"] = m.get("mcc")
            per_epoch_rows.append(row)

            # 早停（根据选择的指标）
            if not self.multitask:
                # 单任务：直接从val_metrics获取
                metric_key = self.early_stopping_metric
                if metric_key == "acc":
                    metric_key = "accuracy"
                elif metric_key == "f1":
                    metric_key = "f1_macro"
                cur = val_metrics.get(metric_key, -1.0)
            else:
                # 多任务：计算所有任务的平均值
                metric_key = self.early_stopping_metric
                if metric_key == "acc":
                    metric_key = "accuracy"
                elif metric_key == "f1":
                    metric_key = "f1_macro"
                cur = _mean_metric(val_metrics, metric_key)
            improved = (cur is not None) and (
                cur > best_val_metric + self.early_stopping_min_delta)
            if improved:
                best_val_metric = cur
                best_epoch = epoch
                epochs_bad = 0
                if self.save_checkpoints:
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in self.model.state_dict().items()}
            else:
                epochs_bad += 1

            # 简单日志
            if not self.multitask:
                tqdm.write(
                    f"[Epoch {epoch:03d}] train_loss={train_loss:.4f}  "
                    f"val_f1={val_metrics.get('f1_macro'):.4f}  "
                    f"val_acc={val_metrics.get('accuracy'):.4f}  "
                    f"val_mcc={val_metrics.get('mcc'):.4f}  "
                    f"val_auc={val_metrics.get('auc_macro_ovr'):.4f}  "
                    f"val_auprc={(val_metrics.get('auprc') or val_metrics.get('auprc_macro_ovr')):.4f}"
                )
            else:
                val_mean = _mean_metric(val_metrics, "f1_macro")
                val_acc = _mean_metric(val_metrics, "accuracy")
                val_mcc = _mean_metric(val_metrics, "mcc")
                val_auc = _mean_metric(val_metrics, "auc_macro_ovr")
                val_auprc = _mean_metric(val_metrics, "auprc")
                if val_auprc < 0:
                    val_auprc = _mean_metric(val_metrics, "auprc_macro_ovr")
                tqdm.write(
                    f"[Epoch {epoch:03d}] train_loss={train_loss:.4f}  "
                    f"val_f1_mean={val_mean:.4f}  "
                    f"val_acc_mean={val_acc:.4f}  "
                    f"val_mcc_mean={val_mcc:.4f}  "
                    f"val_auc_mean={val_auc:.4f}  "
                    f"val_auprc_mean={val_auprc:.4f}"
                )
                # 如果MCC很低，输出诊断信息
                if val_mcc is not None and val_mcc < 0.1 and epoch % 10 == 0:  # 每10个epoch输出一次
                    tqdm.write(f"[DIAG] MCC很低 (={val_mcc:.4f})，检查各任务的MCC:")
                    for tname, m in val_metrics.items():
                        task_mcc = m.get("mcc", None)
                        if task_mcc is not None:
                            tqdm.write(f"  {tname}: MCC={task_mcc:.4f}, F1={m.get('f1_macro', 0):.4f}, Acc={m.get('accuracy', 0):.4f}")

            if epochs_bad >= self.early_stopping_patience:
                tqdm.write(
                    f"[EarlyStop] No improvement in {self.early_stopping_patience} epochs (metric: {self.early_stopping_metric}).")
                break

        # 如需，把最佳权重加载回去再做一次完整版评测
        if best_state is not None:
            self.model.load_state_dict(best_state)
            torch.save(best_state, os.path.join(
                self.output_dir, "best_head.pt"))

        # 4) 结束后：完整版报告（使用最佳权重）
        val_metrics, val_report, val_cm = self._eval_split(val_loader, split_name="val")
        test_metrics = test_report = test_cm = None
        if test_loader is not None:
            test_metrics, test_report, test_cm = self._eval_split(test_loader, split_name="test")

        # 标签映射保存
        label2id_path = os.path.join(self.output_dir, "label2id.json")
        with open(label2id_path, "w", encoding="utf-8") as f:
            label_map = None
            if hasattr(self.train_ds, "get_label_map"):
                label_map = self.train_ds.get_label_map()  # type: ignore
            elif hasattr(self.train_ds, "label2id"):
                label_map = getattr(self.train_ds, "label2id")
            elif hasattr(getattr(self.train_ds, "base", None), "label2id"):
                label_map = getattr(getattr(self.train_ds, "base"), "label2id")
            json.dump(label_map or {}, f, indent=2, ensure_ascii=False)

        # 逐 epoch 指标写 JSONL
        with open(os.path.join(self.output_dir, "per_epoch_metrics.jsonl"), "w", encoding="utf-8") as f:
            for r in per_epoch_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # 汇总 JSON
        num_classes: Any = None
        if isinstance(label_map, dict) and label_map:
            if all(isinstance(v, dict) for v in label_map.values()):
                num_classes = {k: len(v) for k, v in label_map.items()}
            else:
                num_classes = int(len(label_map))
        summary: Dict[str, Any] = {
            "output_dir": self.output_dir,
            "num_classes": num_classes,
            "train_size": int(len(self.train_ds)),
            "val_size": int(len(self.val_ds)),
            "test_size": (int(len(self.test_ds)) if self.test_ds is not None else None),
            "best_epoch": best_epoch,
            "best_val_metric": best_val_metric,
            "best_val_metric_name": self.early_stopping_metric,
            "best_val_mcc": best_val_metric if self.early_stopping_metric == "mcc" else None,  # 保持兼容性
            "training_args": {
                "task": self.task,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "emb_pool": self.emb_pool,
                "emb_batch_size": self.emb_batch_size,
                "emb_average_rc": self.emb_average_rc,
                "save_checkpoints": self.save_checkpoints,
                "force_recompute_embeddings": self.force_recompute_embeddings,
                "seed": self.seed,
                "emb_l2norm": self.emb_l2norm,
                "emb_layer_name": self.emb_layer_name,
                "multitask": self.multitask,
                "task_names": self.task_names,
                "early_stopping_patience": self.early_stopping_patience,
                "early_stopping_min_delta": self.early_stopping_min_delta,
                "early_stopping_metric": self.early_stopping_metric,
            },
            "label2id_path": label2id_path,
        }
        if not self.multitask:
            summary.update({
                "val_metrics": {
                    "val_accuracy": val_metrics.get("accuracy"),
                    "val_f1_macro": val_metrics.get("f1_macro"),
                    "val_f1_micro": val_metrics.get("f1_micro"),
                    "val_f1_weighted": val_metrics.get("f1_weighted"),
                    "val_precision_macro": val_metrics.get("precision_macro"),
                    "val_recall_macro": val_metrics.get("recall_macro"),
                    "val_auc_macro_ovr": val_metrics.get("auc_macro_ovr"),
                    "val_auprc": val_metrics.get("auprc") or val_metrics.get("auprc_macro_ovr"),
                    "val_mcc": val_metrics.get("mcc"),
                },
                "val_classification_report": val_report,
                "val_confusion_matrix": val_cm,
            })
            if test_metrics is not None:
                summary.update({
                    "test_metrics": {
                        "test_accuracy": test_metrics.get("accuracy"),
                        "test_f1_macro": test_metrics.get("f1_macro"),
                        "test_f1_micro": test_metrics.get("f1_micro"),
                        "test_f1_weighted": test_metrics.get("f1_weighted"),
                        "test_precision_macro": test_metrics.get("precision_macro"),
                        "test_recall_macro": test_metrics.get("recall_macro"),
                        "test_auc_macro_ovr": test_metrics.get("auc_macro_ovr"),
                        "test_auprc": test_metrics.get("auprc") or test_metrics.get("auprc_macro_ovr"),
                        "test_mcc": test_metrics.get("mcc"),
                    },
                    "test_classification_report": test_report,
                    "test_confusion_matrix": test_cm,
                })
        else:
            # 多任务：按任务保存的同时，计算各任务指标平均值并保存
            def _avg_over_tasks(metrics_by_task: Dict[str, Dict[str, Any]], prefix: str) -> Dict[str, Any]:
                if not metrics_by_task:
                    return {}
                keys_to_avg = [
                    "accuracy", "f1_macro", "f1_micro", "f1_weighted",
                    "precision_macro", "recall_macro", "auc_macro_ovr", "mcc",
                    "auprc", "auprc_macro_ovr",
                ]
                out = {}
                for k in keys_to_avg:
                    vals = [m.get(k) for m in metrics_by_task.values() if m.get(k) is not None]
                    if vals:
                        out[f"{prefix}_{k}"] = float(np.mean(vals))
                # 与单任务一致：auprc 优先用 auprc，否则 auprc_macro_ovr
                if f"{prefix}_auprc" not in out and f"{prefix}_auprc_macro_ovr" in out:
                    out[f"{prefix}_auprc"] = out[f"{prefix}_auprc_macro_ovr"]
                return out

            val_metrics_avg = _avg_over_tasks(val_metrics, "val")
            summary.update({
                "val_metrics_by_task": val_metrics,
                "val_metrics_avg": val_metrics_avg,
                "val_classification_report_by_task": val_report,
                "val_confusion_matrix_by_task": val_cm,
            })
            if test_metrics is not None:
                test_metrics_avg = _avg_over_tasks(test_metrics, "test")
                summary.update({
                    "test_metrics_by_task": test_metrics,
                    "test_metrics_avg": test_metrics_avg,
                    "test_classification_report_by_task": test_report,
                    "test_confusion_matrix_by_task": test_cm,
                })

        # 写主 summary
        with open(os.path.join(self.output_dir, "finetune_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return summary


class ChunkedEmbeddingDataset(Dataset):
    def __init__(self, chunk_dir: str):
        self.chunk_files = sorted(glob.glob(os.path.join(chunk_dir, "chunk_*.pt")))
        if not self.chunk_files:
            raise ValueError(f"No chunk files found in {chunk_dir}")
        self.chunk_sizes: List[int] = []
        self.has_groups = True
        for cf in self.chunk_files:
            data = torch.load(cf, map_location="cpu", weights_only=True)
            self.chunk_sizes.append(int(data["feats"].shape[0]))
            if data.get("groups") is None:
                self.has_groups = False
        self.cum_sizes = np.cumsum(self.chunk_sizes).tolist()
        self._cache_idx: Optional[int] = None
        self._cache_data: Optional[Dict[str, torch.Tensor]] = None

    def __len__(self) -> int:
        return int(self.cum_sizes[-1])

    def _locate(self, idx: int) -> Tuple[int, int]:
        if idx < 0 or idx >= len(self):
            raise IndexError("index out of range")
        lo, hi = 0, len(self.cum_sizes) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < self.cum_sizes[mid]:
                hi = mid
            else:
                lo = mid + 1
        chunk_idx = lo
        prev = 0 if chunk_idx == 0 else self.cum_sizes[chunk_idx - 1]
        inner_idx = idx - prev
        return chunk_idx, inner_idx

    def _load_chunk(self, chunk_idx: int) -> Dict[str, torch.Tensor]:
        if self._cache_idx != chunk_idx:
            self._cache_data = torch.load(
                self.chunk_files[chunk_idx], map_location="cpu", weights_only=True)
            self._cache_idx = chunk_idx
        return self._cache_data  # type: ignore[return-value]

    def __getitem__(self, idx: int):
        chunk_idx, inner_idx = self._locate(idx)
        data = self._load_chunk(chunk_idx)
        feats = data["feats"][inner_idx]
        labels = data["labels"][inner_idx]
        if self.has_groups:
            groups = data["groups"][inner_idx]
            return feats, labels, groups
        return feats, labels


class ChunkSequentialBatchSampler:
    def __init__(
        self,
        dataset: ChunkedEmbeddingDataset,
        batch_size: int,
        drop_last: bool = False,
        shuffle_within_chunk: bool = True,
        seed: int = 2025,
    ):
        self.dataset = dataset
        self.batch_size = max(int(batch_size), 1)
        self.drop_last = drop_last
        self.shuffle_within_chunk = shuffle_within_chunk
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        start = 0
        for size in self.dataset.chunk_sizes:
            idxs = np.arange(size)
            if self.shuffle_within_chunk:
                rng.shuffle(idxs)
            if start > 0:
                idxs = idxs + start
            for i in range(0, size, self.batch_size):
                batch = idxs[i:i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                yield batch.tolist()
            start += size

    def __len__(self) -> int:
        total = 0
        for size in self.dataset.chunk_sizes:
            if self.drop_last:
                total += size // self.batch_size
            else:
                total += (size + self.batch_size - 1) // self.batch_size
        return total

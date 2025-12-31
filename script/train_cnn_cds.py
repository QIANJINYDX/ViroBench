# script/train_cnn_cds.py
from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# -----------------------------
# Make imports work (project root)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# -----------------------------
# Import model (MUST match your provided model/cnn.py)
# -----------------------------
from model.cnn import GenomeCNN1D, CNNConfig
from util.cds_cls_dataset import CdsClsDataset


# ============================================================
# Utilities
# ============================================================
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_run_name(prefix: str = "CdsCls_CNN") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


# ============================================================
# Window helpers
# ============================================================
def _pad_or_crop_window_uint8(
    x_np: np.ndarray,
    start: int,
    window_len: int,
    pad_idx: int = 0,
) -> torch.Tensor:
    out = torch.full((window_len,), int(pad_idx), dtype=torch.uint8)
    L = int(len(x_np))
    if L <= 0:
        return out
    end = min(start + window_len, L)
    src = torch.from_numpy(x_np[start:end])  # uint8 numpy
    out[: (end - start)] = src
    return out


def _get_window_from_seq_list(
    seq_list: List[np.ndarray],
    seq_idx: int,
    start: int,
    window_len: int,
    pad_idx: int = 0,
) -> torch.Tensor:
    """
    Extract a window from a specific sequence in the list.
    
    Args:
        seq_list: List of sequences (each is a numpy array)
        seq_idx: Index of the sequence to extract window from
        start: Start position in the selected sequence
        window_len: Window length
        pad_idx: Padding index
        
    Returns:
        Window tensor of shape (window_len,)
    """
    if not seq_list or seq_idx < 0 or seq_idx >= len(seq_list):
        return torch.full((window_len,), int(pad_idx), dtype=torch.uint8)
    
    return _pad_or_crop_window_uint8(seq_list[seq_idx], start, window_len, pad_idx)


class FixedWindowEvalDataset(Dataset):
    """
    Eval wrapper:
      Takes first window_len bases from each of the top_k CDS sequences,
      returns (x_win, y, sample_id)
    """

    def __init__(
        self,
        base_dataset: Dataset,
        window_len: int,
        pad_idx: int = 0,
    ):
        self.base = base_dataset
        self.window_len = int(window_len)
        self.pad_idx = int(pad_idx)

        self._index: List[Tuple[int, int]] = []  # (base_idx, seq_idx)
        for base_idx in range(len(self.base)):
            x_list, _y = self.base[base_idx]
            # Each sample has top_k sequences, create one window per sequence
            for seq_idx in range(len(x_list)):
                self._index.append((base_idx, seq_idx))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int):
        base_idx, seq_idx = self._index[i]
        x_list, y = self.base[base_idx]
        # Always take first window_len bases (start=0)
        x_win = _get_window_from_seq_list(x_list, seq_idx, 0, self.window_len, self.pad_idx)
        return x_win, int(y), int(base_idx)


def extract_base_labels(base_ds: Dataset) -> torch.Tensor:
    ys = []
    for i in range(len(base_ds)):
        _x_list, y = base_ds[i]  # x_list is now a list of sequences
        ys.append(int(y))
    return torch.tensor(ys, dtype=torch.long)


def collate_train(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.tensor(ys, dtype=torch.long)


def collate_eval(batch):
    xs, ys, sids = zip(*batch)
    return torch.stack(xs, 0), torch.tensor(ys, dtype=torch.long), torch.tensor(sids, dtype=torch.long)


# ============================================================
# TRAIN: fixed windows dataset
# ============================================================
class FixedWindowTrainDataset(Dataset):
    """
    Train: fixed windows (first window_len bases from each CDS sequence)
    
    Each sample has top_k sequences, so N = len(base_dataset) * top_k
    """

    def __init__(
        self,
        base_dataset: Dataset,
        window_len: int,
        top_k: int,
        pad_idx: int = 0,
    ):
        self.base = base_dataset
        self.window_len = int(window_len)
        self.top_k = int(top_k)
        self.pad_idx = int(pad_idx)

        # Each sample has top_k sequences, so total windows = len(base_dataset) * top_k
        self.N = len(self.base) * self.top_k

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, i: int):
        base_idx = i // self.top_k
        seq_idx = i % self.top_k
        x_list, y = self.base[base_idx]
        
        # Always take first window_len bases (start=0)
        x_win = _get_window_from_seq_list(x_list, seq_idx, 0, self.window_len, self.pad_idx)
        return x_win, int(y)


# ============================================================
# Metrics (NO sklearn dependency + NO np.trapz)
# ============================================================
def softmax_np(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def multiclass_mcc_from_cm(cm: np.ndarray) -> float:
    cm = cm.astype(np.float64)
    t_sum = cm.sum(axis=1)
    p_sum = cm.sum(axis=0)
    n = cm.sum()
    c = np.trace(cm)

    s = n
    sum_p_t = np.sum(p_sum * t_sum)
    sum_p2 = np.sum(p_sum ** 2)
    sum_t2 = np.sum(t_sum ** 2)

    num = (c * s) - sum_p_t
    den = np.sqrt((s**2 - sum_p2) * (s**2 - sum_t2))
    if den <= 0:
        return float("nan")
    return float(num / den)


def f1_scores_from_cm(cm: np.ndarray) -> Dict[str, Any]:
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)

    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)

    macro_f1 = float(np.nanmean(f1))
    weighted_f1 = float(np.nansum(f1 * support) / (support.sum() + 1e-12))

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": f1.astype(np.float32),
        "per_class_precision": prec.astype(np.float32),
        "per_class_recall": rec.astype(np.float32),
        "support": support.astype(np.int64),
    }


def roc_curve_binary(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = y_true.astype(np.int32)
    pos = int(y_true.sum())
    neg = int(y_true.size - pos)
    if pos == 0 or neg == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    order = np.argsort(-y_score, kind="mergesort")
    yt = y_true[order]
    ys = y_score[order]

    distinct = np.where(np.diff(ys))[0]
    thr_idx = np.r_[distinct, yt.size - 1]

    tps = np.cumsum(yt)[thr_idx]
    fps = (1 + thr_idx) - tps

    tpr = tps / (pos + 1e-12)
    fpr = fps / (neg + 1e-12)
    thr = ys[thr_idx]

    # add start point
    tpr = np.r_[0.0, tpr].astype(np.float32)
    fpr = np.r_[0.0, fpr].astype(np.float32)
    thr = np.r_[thr[0] + 1e-6, thr].astype(np.float32)
    return fpr, tpr, thr


def auc_trapz(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 2 or y.size < 2:
        return float("nan")
    if not np.all(x[1:] >= x[:-1]):
        order = np.argsort(x)
        x = x[order]
        y = y[order]
    dx = x[1:] - x[:-1]
    return float(np.sum((y[1:] + y[:-1]) * 0.5 * dx))


def average_precision_and_pr_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    y_true = y_true.astype(np.int32)
    pos = int(y_true.sum())
    if pos == 0:
        return float("nan"), np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    order = np.argsort(-y_score, kind="mergesort")
    yt = y_true[order]
    ys = y_score[order]

    tp = np.cumsum(yt)
    fp = np.cumsum(1 - yt)

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (pos + 1e-12)

    # AP = mean precision at positive ranks
    ap = float(np.sum(precision[yt == 1]) / (pos + 1e-12))

    distinct = np.where(np.diff(ys))[0]
    thr_idx = np.r_[distinct, yt.size - 1]
    pc = precision[thr_idx]
    rc = recall[thr_idx]
    thr = ys[thr_idx]

    pc = np.r_[1.0, pc].astype(np.float32)
    rc = np.r_[0.0, rc].astype(np.float32)
    thr = np.r_[thr[0] + 1e-6, thr].astype(np.float32)
    return ap, pc, rc, thr


def compute_all_metrics(
    y_true: np.ndarray,
    agg_logits: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_curves: bool = False,
    curves_out_npz: Optional[Path] = None,
) -> Dict[str, Any]:
    num_classes = int(agg_logits.shape[1])
    probs = softmax_np(agg_logits.astype(np.float64)).astype(np.float32)
    y_pred = probs.argmax(axis=1).astype(np.int64)

    cm = confusion_matrix_np(y_true, y_pred, num_classes=num_classes)
    acc = float((y_pred == y_true).mean())

    f1d = f1_scores_from_cm(cm)
    mcc = multiclass_mcc_from_cm(cm)

    per_auc = np.full((num_classes,), np.nan, dtype=np.float32)
    per_ap = np.full((num_classes,), np.nan, dtype=np.float32)

    roc_fpr = []
    roc_tpr = []
    roc_thr = []
    pr_prec = []
    pr_rec = []
    pr_thr = []

    for k in range(num_classes):
        yk = (y_true == k).astype(np.int32)
        sk = probs[:, k].astype(np.float64)

        fpr, tpr, thr = roc_curve_binary(yk, sk)
        per_auc[k] = np.float32(auc_trapz(fpr, tpr))

        ap, pc, rc, pthr = average_precision_and_pr_curve(yk, sk)
        per_ap[k] = np.float32(ap)

        if save_curves:
            roc_fpr.append(fpr.astype(np.float32))
            roc_tpr.append(tpr.astype(np.float32))
            roc_thr.append(thr.astype(np.float32))
            pr_prec.append(pc.astype(np.float32))
            pr_rec.append(rc.astype(np.float32))
            pr_thr.append(pthr.astype(np.float32))

    macro_auroc = float(np.nanmean(per_auc)) if np.any(~np.isnan(per_auc)) else float("nan")
    macro_aupr = float(np.nanmean(per_ap)) if np.any(~np.isnan(per_ap)) else float("nan")

    support = cm.sum(axis=1).astype(np.float64)
    w_auroc = float(np.nansum(per_auc * support) / (support.sum() + 1e-12))
    w_aupr = float(np.nansum(per_ap * support) / (support.sum() + 1e-12))

    metrics: Dict[str, Any] = {
        "acc": acc,
        "mcc": float(mcc),
        "macro_f1": float(f1d["macro_f1"]),
        "weighted_f1": float(f1d["weighted_f1"]),
        "macro_auroc_ovr": macro_auroc,
        "weighted_auroc_ovr": w_auroc,
        "macro_aupr_ovr": macro_aupr,
        "weighted_aupr_ovr": w_aupr,
        "num_classes": num_classes,
        "confusion_matrix": cm.tolist(),
        "per_class": {
            "auroc_ovr": per_auc.tolist(),
            "aupr_ovr": per_ap.tolist(),
            "f1": f1d["per_class_f1"].tolist(),
            "precision": f1d["per_class_precision"].tolist(),
            "recall": f1d["per_class_recall"].tolist(),
            "support": f1d["support"].astype(int).tolist(),
            "class_names": class_names if class_names is not None else None,
        },
    }

    if save_curves and curves_out_npz is not None:
        ensure_dir(curves_out_npz.parent)
        np.savez_compressed(
            curves_out_npz,
            roc_fpr=np.array(roc_fpr, dtype=object),
            roc_tpr=np.array(roc_tpr, dtype=object),
            roc_thr=np.array(roc_thr, dtype=object),
            pr_prec=np.array(pr_prec, dtype=object),
            pr_rec=np.array(pr_rec, dtype=object),
            pr_thr=np.array(pr_thr, dtype=object),
            class_names=np.array(class_names if class_names is not None else [str(i) for i in range(num_classes)], dtype=object),
        )
        metrics["curves_npz"] = str(curves_out_npz)

    return metrics


# ============================================================
# Train / Eval
# ============================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    epoch: int,
    amp: bool,
    scaler: torch.amp.GradScaler,
) -> Tuple[float, float]:
    model.train()
    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Train E{epoch:03d}", leave=False)
    for x_u8, y in pbar:
        x_u8 = x_u8.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if amp and device.type == "cuda":
            with torch.amp.autocast("cuda", enabled=True):
                logits = model(x_u8.long())
                loss = ce(logits, y)
            scaler.scale(loss).backward()
            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x_u8.long())
            loss = ce(logits, y)
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        pred = logits.argmax(dim=-1)
        bs = y.size(0)
        total_correct += (pred == y).sum().item()
        total += bs
        total_loss += loss.item() * bs

        pbar.set_postfix(loss=f"{total_loss/max(total,1):.4f}", acc=f"{total_correct/max(total,1):.4f}")

    return total_loss / max(total, 1), total_correct / max(total, 1)


@torch.no_grad()
def aggregate_logits_over_windows(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_base: int,
    epoch: int,
    amp: bool,
) -> Tuple[float, np.ndarray]:
    """
    Aggregate window logits -> per-sample logits (mean over windows)
    """
    model.eval()
    ce_sum = nn.CrossEntropyLoss(reduction="sum")

    sum_logits: Optional[torch.Tensor] = None
    cnt = torch.zeros(num_base, dtype=torch.long, device=device)

    total_loss = 0.0
    total_windows = 0

    pbar = tqdm(loader, desc=f"Agg  E{epoch:03d}", leave=False)
    for x_u8, y, sid in pbar:
        x_u8 = x_u8.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        sid = sid.to(device, non_blocking=True)

        if amp and device.type == "cuda":
            with torch.amp.autocast("cuda", enabled=True):
                logits = model(x_u8.long())
        else:
            logits = model(x_u8.long())

        logits_fp32 = logits.float()

        if sum_logits is None:
            sum_logits = torch.zeros((num_base, logits_fp32.size(-1)), device=device, dtype=torch.float32)

        total_loss += ce_sum(logits_fp32, y).item()
        total_windows += y.numel()

        sum_logits.index_add_(0, sid, logits_fp32)
        cnt.index_add_(0, sid, torch.ones_like(sid, dtype=torch.long))

        pbar.set_postfix(windows=total_windows)

    if sum_logits is None:
        return 0.0, np.zeros((num_base, 1), dtype=np.float32)

    agg_logits = (sum_logits / cnt.clamp(min=1).unsqueeze(-1)).detach().cpu().numpy().astype(np.float32)
    avg_window_loss = total_loss / max(total_windows, 1)
    
    # Debug: check window count distribution
    cnt_np = cnt.cpu().numpy()
    unique_counts = np.unique(cnt_np, return_counts=True)
    if epoch <= 3 or epoch % 10 == 0:  # Print debug info for first few epochs and every 10 epochs
        print(f"[DEBUG] Window counts per sample: min={cnt_np.min()}, max={cnt_np.max()}, mean={cnt_np.mean():.2f}")
        if len(unique_counts[0]) <= 10:
            print(f"[DEBUG] Window count distribution: {dict(zip(unique_counts[0].tolist(), unique_counts[1].tolist()))}")
    
    return avg_window_loss, agg_logits


# ============================================================
# Main
# ============================================================
def _auto_find_cds_json(splits_dir: Path) -> Path:
    # prefer *cds*.json excluding label2id.json
    jss = [p for p in splits_dir.glob("*.json") if p.name != "label2id.json"]
    if not jss:
        raise FileNotFoundError(f"No CDS json found under splits_dir={splits_dir} (excluding label2id.json)")
    # prefer name contains 'cds'
    cds = [p for p in jss if "cds" in p.name.lower()]
    return sorted(cds)[0] if cds else sorted(jss)[0]


def main():
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()

    # fixed splits directory (train.csv/val.csv/test.csv/label2id.json + cds json)
    parser.add_argument("--splits_dir", type=str, required=True)
    parser.add_argument("--cds_json", type=str, default="", help="CDS json path. If empty, auto-find in splits_dir.")

    # columns
    parser.add_argument("--label_column", type=str, default="host_group")

    # CDS concat
    parser.add_argument("--top_k_cds", type=int, default=1, help="Concatenate top-k longest CDS per sample.")

    # windowing
    parser.add_argument("--window_len", type=int, default=2048, help="Length of window to extract from each CDS sequence (takes first window_len bases)")

    # training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    # workers
    parser.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 8) - 1))

    # device
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--amp", action="store_true")

    # results
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--run_name", type=str, default="")

    # model hparams (MATCH your CNNConfig)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--head_hidden", type=int, default=256)
    parser.add_argument("--head_dropout", type=float, default=0.3)

    parser.add_argument("--channels", type=str, default="64,128,256", help="Comma-separated stage channels")
    parser.add_argument("--blocks_per_stage", type=str, default="2,2,2", help="Comma-separated blocks per stage")
    parser.add_argument("--norm", type=str, default="bn", choices=["bn", "gn"])
    parser.add_argument("--gn_groups", type=int, default=8)
    parser.add_argument("--global_pool", type=str, default="avg", choices=["avg", "max"])
    parser.add_argument("--use_se", action="store_true")
    parser.add_argument("--se_ratio", type=float, default=0.25)

    # metrics / curves
    parser.add_argument("--save_curves", action="store_true", help="Save per-class ROC/PR curve points to NPZ.")
    parser.add_argument("--save_logits", action="store_true", help="Save aggregated logits/probs/preds for val/test.")

    args = parser.parse_args()
    set_seed(args.seed)

    # device
    if args.gpu is None or args.gpu < 0 or (not torch.cuda.is_available()):
        device = torch.device("cpu")
    else:
        if args.gpu >= torch.cuda.device_count():
            raise ValueError(f"--gpu {args.gpu} out of range. visible cuda devices={torch.cuda.device_count()}")
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")

    # parse channels / blocks
    channels = tuple(int(x.strip()) for x in args.channels.split(",") if x.strip())
    blocks_per_stage = tuple(int(x.strip()) for x in args.blocks_per_stage.split(",") if x.strip())
    if len(channels) == 0 or len(channels) != len(blocks_per_stage):
        raise ValueError("--channels and --blocks_per_stage must be same-length non-empty lists")

    # results
    rn = args.run_name.strip() or make_run_name("CdsCls_CNN")
    out_dir = ensure_dir(Path(args.results_dir) / rn)
    ckpt_dir = ensure_dir(out_dir / "checkpoints")

    # load fixed splits
    splits_dir = Path(args.splits_dir)
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"
    if args.label_column == "family":
        label2id_path = splits_dir / "label2id_family.json"
    else:
        label2id_path = splits_dir / "label2id.json"

    for p in [train_csv, val_csv, test_csv, label2id_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    cds_json_path = Path(args.cds_json) if args.cds_json else _auto_find_cds_json(splits_dir)
    if not cds_json_path.exists():
        raise FileNotFoundError(f"CDS json not found: {cds_json_path}")

    label2id = load_json(label2id_path)
    num_classes = len(label2id)
    id2label = {int(v): str(k) for k, v in label2id.items()}
    class_names = [id2label[i] for i in range(num_classes)]

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # base datasets
    train_base = CdsClsDataset(
        csv_path=train_df,
        json_path=str(cds_json_path),
        label2id=label2id,
        label_column=args.label_column,
        cache_dir="",
        top_k=args.top_k_cds,
    )
    val_base = CdsClsDataset(
        csv_path=val_df,
        json_path=str(cds_json_path),
        label2id=label2id,
        label_column=args.label_column,
        cache_dir="",
        top_k=args.top_k_cds,
    )
    test_base = CdsClsDataset(
        csv_path=test_df,
        json_path=str(cds_json_path),
        label2id=label2id,
        label_column=args.label_column,
        cache_dir="",
        top_k=args.top_k_cds,
    )

    print(f"[INFO] device={device} cuda={torch.cuda.is_available()} workers={args.num_workers}")
    print(f"[INFO] base sizes train/val/test = {len(train_base)}/{len(val_base)}/{len(test_base)}")
    print(f"[INFO] classes={num_classes}")
    print(f"[INFO] cds_json={cds_json_path}")
    print(f"[INFO] top_k_cds={args.top_k_cds} window_len={args.window_len}")
    print(f"[INFO] splits_dir={splits_dir.resolve()} label2id={label2id_path}")
    
    # Extract and save sequences used in training/validation/testing
    print("[INFO] Extracting sequences used in training/validation/testing...")
    
    # Load CDS JSON data once
    with open(cds_json_path, 'r') as f:
        cds_data = json.load(f)
    
    def extract_sequences_info(df: pd.DataFrame, split_name: str, top_k: int, window_len: int) -> pd.DataFrame:
        """
        Extract information about sequences used in the dataset.
        Returns a DataFrame with one row per sequence (after top_k selection).
        """
        rows = []
        for dataset_idx, (_, row) in enumerate(df.iterrows()):
            row_data = row.to_dict()
            csv_idx = int(row_data["idx"])
            idx_str = str(csv_idx)
            
            if idx_str not in cds_data:
                continue
            
            cds_list = cds_data[idx_str]
            if not cds_list:
                continue
            
            # Extract sequences and sort by length (descending)
            seq_with_info = []
            for item in cds_list:
                seq = item.get("sequence", "")
                if seq and len(seq) > 0:
                    header_info = item.get("header_info", {})
                    seq_with_info.append({
                        "sequence": seq,
                        "length": len(seq),
                        "header_info": header_info
                    })
            
            if not seq_with_info:
                continue
            
            # Sort by length (descending) and take top K
            seq_with_info.sort(key=lambda x: x["length"], reverse=True)
            top_k_seqs = seq_with_info[:top_k]
            
            # Create one row per sequence
            for seq_idx, seq_info in enumerate(top_k_seqs):
                seq_row = row_data.copy()
                seq_row["dataset_idx"] = dataset_idx  # Index in the dataset (0-based)
                seq_row["cds_seq_idx"] = seq_idx  # Index within top_k (0 to top_k-1)
                seq_row["cds_seq_length"] = seq_info["length"]
                seq_row["cds_sequence"] = seq_info["sequence"]  # Full sequence
                seq_row["window_len"] = window_len
                seq_row["window_start"] = 0  # Always starts from 0
                seq_row["window_end"] = min(window_len, seq_info["length"])
                seq_row["window_used_length"] = min(window_len, seq_info["length"])
                seq_row["is_padded"] = seq_info["length"] < window_len
                # Add header_info fields if available
                header_info = seq_info.get("header_info", {})
                if isinstance(header_info, dict):
                    for key, value in header_info.items():
                        seq_row[f"cds_header_{key}"] = value
                rows.append(seq_row)
        
        return pd.DataFrame(rows)
    
    train_seqs_df = extract_sequences_info(train_df, "train", args.top_k_cds, args.window_len)
    val_seqs_df = extract_sequences_info(val_df, "val", args.top_k_cds, args.window_len)
    test_seqs_df = extract_sequences_info(test_df, "test", args.top_k_cds, args.window_len)
    
    # Save sequences CSV
    train_seqs_df.to_csv(out_dir / "train_sequences.csv", index=False)
    val_seqs_df.to_csv(out_dir / "val_sequences.csv", index=False)
    test_seqs_df.to_csv(out_dir / "test_sequences.csv", index=False)
    print(f"[INFO] Saved sequences to:")
    print(f"  - {out_dir / 'train_sequences.csv'} ({len(train_seqs_df)} rows, {len(train_df)} samples)")
    print(f"  - {out_dir / 'val_sequences.csv'} ({len(val_seqs_df)} rows, {len(val_df)} samples)")
    print(f"  - {out_dir / 'test_sequences.csv'} ({len(test_seqs_df)} rows, {len(test_df)} samples)")

    # save config
    cfg: Dict[str, Any] = {
        "run_name": rn,
        "cmd": " ".join(sys.argv),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "splits_dir": str(splits_dir.resolve()),
        "cds_json": str(cds_json_path.resolve()),
        "files": {
            "train": str(train_csv),
            "val": str(val_csv),
            "test": str(test_csv),
            "label2id": str(label2id_path),
        },
        "data": {
            "label_column": args.label_column,
            "top_k_cds": int(args.top_k_cds),
        },
        "windowing": {
            "window_len": int(args.window_len),
            "top_k_cds": int(args.top_k_cds),
        },
        "train": {
            "epochs": int(args.epochs),
            "eval_every": int(args.eval_every),
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "num_workers": int(args.num_workers),
            "amp": bool(args.amp),
            "gpu": int(args.gpu),
        },
        "model": {
            "type": "ResCNN1D",
            "vocab_size": 5,
            "embed_dim": int(args.embed_dim),
            "kernel_size": int(args.kernel_size),
            "channels": channels,
            "blocks_per_stage": blocks_per_stage,
            "norm": args.norm,
            "gn_groups": int(args.gn_groups),
            "dropout": float(args.dropout),
            "global_pool": args.global_pool,
            "head_hidden": int(args.head_hidden),
            "head_dropout": float(args.head_dropout),
            "use_se": bool(args.use_se),
            "se_ratio": float(args.se_ratio),
            "out_dim": int(num_classes),
        },
        "metrics": {
            "compute": ["ACC", "MCC", "MacroF1", "WeightedF1", "OVR-AUROC", "OVR-AUPR"],
            "save_curves": bool(args.save_curves),
            "save_logits": bool(args.save_logits),
        },
    }
    save_json(cfg, out_dir / "config.json")
    save_json(label2id, out_dir / "label2id.json")

    # y_true for base samples
    y_val_base = extract_base_labels(val_base).numpy().astype(np.int64)
    y_test_base = extract_base_labels(test_base).numpy().astype(np.int64)

    # eval datasets and loaders
    val_eval_ds = FixedWindowEvalDataset(
        val_base,
        window_len=args.window_len,
        pad_idx=0,
    )
    test_eval_ds = FixedWindowEvalDataset(
        test_base,
        window_len=args.window_len,
        pad_idx=0,
    )

    val_loader = DataLoader(
        val_eval_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_eval,
    )
    test_loader = DataLoader(
        test_eval_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_eval,
    )

    # --------- model (MATCH your model/cnn.py) ----------
    cnn_cfg = CNNConfig(
        vocab_size=5,
        pad_idx=0,
        embed_dim=args.embed_dim,
        channels=channels,
        blocks_per_stage=blocks_per_stage,
        kernel_size=args.kernel_size,
        norm=args.norm,
        gn_groups=args.gn_groups,
        dropout=args.dropout,
        global_pool=args.global_pool,
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
        use_se=bool(args.use_se),
        se_ratio=args.se_ratio,
    )
    model = GenomeCNN1D(out_dim=num_classes, cfg=cnn_cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Add learning rate scheduler for more stable training
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    # log
    log_path = out_dir / "train_log.csv"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,val_window_loss,val_acc,val_mcc,val_macro_f1,val_macro_auroc,val_macro_aupr\n")

    best_val_acc = -1.0
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"

    epoch_pbar = tqdm(range(1, args.epochs + 1), desc="Epochs", leave=True)
    for epoch in epoch_pbar:
        train_ds = FixedWindowTrainDataset(
            base_dataset=train_base,
            window_len=args.window_len,
            top_k=args.top_k_cds,
            pad_idx=0,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=False,
            prefetch_factor=4 if args.num_workers > 0 else None,
            collate_fn=collate_train,
        )

        tr_loss, tr_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            amp=args.amp,
            scaler=scaler,
        )

        do_eval = (epoch % max(1, args.eval_every) == 0) or (epoch == args.epochs)
        if do_eval:
            val_win_loss, val_agg_logits = aggregate_logits_over_windows(
                model=model,
                loader=val_loader,
                device=device,
                num_base=len(val_base),
                epoch=epoch,
                amp=args.amp,
            )
            val_metrics = compute_all_metrics(
                y_true=y_val_base,
                agg_logits=val_agg_logits,
                class_names=class_names,
                save_curves=False,
            )
            val_acc = val_metrics["acc"]
            val_mcc = val_metrics["mcc"]
            val_macro_f1 = val_metrics["macro_f1"]
            val_macro_auroc = val_metrics["macro_auroc_ovr"]
            val_macro_aupr = val_metrics["macro_aupr_ovr"]
            
            # Update learning rate scheduler based on validation accuracy
            scheduler.step(val_acc)
        else:
            val_win_loss = float("nan")
            val_acc = float("nan")
            val_mcc = float("nan")
            val_macro_f1 = float("nan")
            val_macro_auroc = float("nan")
            val_macro_aupr = float("nan")

        epoch_pbar.set_postfix(train_acc=f"{tr_acc:.4f}", val_acc=f"{val_acc:.4f}")

        print(
            f"[E{epoch:03d}] train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
            f"val acc={val_acc:.4f} mcc={val_mcc:.4f} macroF1={val_macro_f1:.4f} "
            f"macroAUROC={val_macro_auroc:.4f} macroAUPR={val_macro_aupr:.4f}"
        )

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{tr_loss:.6f},{tr_acc:.6f},"
                f"{val_win_loss:.6f},{val_acc:.6f},{val_mcc:.6f},{val_macro_f1:.6f},{val_macro_auroc:.6f},{val_macro_aupr:.6f}\n"
            )

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_val_acc": best_val_acc,
            "label2id": label2id,
            "config": cfg,
        }
        torch.save(state, last_path)

        if do_eval and (val_acc > best_val_acc):
            best_val_acc = val_acc
            state["best_val_acc"] = best_val_acc
            torch.save(state, best_path)

    # ============================================================
    # FINAL: Evaluate best checkpoint + Save curves/logits
    # ============================================================
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    # VAL final (with curves optional)
    val_win_loss, val_agg_logits = aggregate_logits_over_windows(
        model=model,
        loader=val_loader,
        device=device,
        num_base=len(val_base),
        epoch=args.epochs,
        amp=args.amp,
    )
    val_metrics = compute_all_metrics(
        y_true=y_val_base,
        agg_logits=val_agg_logits,
        class_names=class_names,
        save_curves=bool(args.save_curves),
        curves_out_npz=(out_dir / "val_curves.npz") if args.save_curves else None,
    )
    save_json({"val_window_loss": val_win_loss, **val_metrics}, out_dir / "val_metrics.json")

    # TEST final
    test_win_loss, test_agg_logits = aggregate_logits_over_windows(
        model=model,
        loader=test_loader,
        device=device,
        num_base=len(test_base),
        epoch=args.epochs,
        amp=args.amp,
    )
    test_metrics = compute_all_metrics(
        y_true=y_test_base,
        agg_logits=test_agg_logits,
        class_names=class_names,
        save_curves=bool(args.save_curves),
        curves_out_npz=(out_dir / "test_curves.npz") if args.save_curves else None,
    )
    save_json({"test_window_loss": test_win_loss, **test_metrics}, out_dir / "test_metrics.json")

    if args.save_logits:
        val_probs = softmax_np(val_agg_logits.astype(np.float64)).astype(np.float32)
        test_probs = softmax_np(test_agg_logits.astype(np.float64)).astype(np.float32)
        np.savez_compressed(
            out_dir / "agg_outputs.npz",
            val_logits=val_agg_logits.astype(np.float32),
            val_probs=val_probs,
            val_y_true=y_val_base.astype(np.int64),
            val_y_pred=val_probs.argmax(axis=1).astype(np.int64),
            test_logits=test_agg_logits.astype(np.float32),
            test_probs=test_probs,
            test_y_true=y_test_base.astype(np.int64),
            test_y_pred=test_probs.argmax(axis=1).astype(np.int64),
            class_names=np.array(class_names, dtype=object),
        )

    summary = {
        "best_val_acc": float(best_val_acc),
        "val": {
            "acc": val_metrics["acc"],
            "mcc": val_metrics["mcc"],
            "macro_f1": val_metrics["macro_f1"],
            "macro_auroc_ovr": val_metrics["macro_auroc_ovr"],
            "macro_aupr_ovr": val_metrics["macro_aupr_ovr"],
        },
        "test": {
            "acc": test_metrics["acc"],
            "mcc": test_metrics["mcc"],
            "macro_f1": test_metrics["macro_f1"],
            "macro_auroc_ovr": test_metrics["macro_auroc_ovr"],
            "macro_aupr_ovr": test_metrics["macro_aupr_ovr"],
        },
        "paths": {
            "results_dir": str(out_dir),
            "val_metrics": str(out_dir / "val_metrics.json"),
            "test_metrics": str(out_dir / "test_metrics.json"),
            "val_curves_npz": str(out_dir / "val_curves.npz") if args.save_curves else None,
            "test_curves_npz": str(out_dir / "test_curves.npz") if args.save_curves else None,
            "agg_outputs_npz": str(out_dir / "agg_outputs.npz") if args.save_logits else None,
        },
    }
    save_json(summary, out_dir / "summary.json")

    print(
        f"[VAL ] loss(win)={val_win_loss:.4f} acc={val_metrics['acc']:.4f} mcc={val_metrics['mcc']:.4f} "
        f"macroF1={val_metrics['macro_f1']:.4f} macroAUROC={val_metrics['macro_auroc_ovr']:.4f} macroAUPR={val_metrics['macro_aupr_ovr']:.4f}"
    )
    print(
        f"[TEST] loss(win)={test_win_loss:.4f} acc={test_metrics['acc']:.4f} mcc={test_metrics['mcc']:.4f} "
        f"macroF1={test_metrics['macro_f1']:.4f} macroAUROC={test_metrics['macro_auroc_ovr']:.4f} macroAUPR={test_metrics['macro_aupr_ovr']:.4f}"
    )
    if args.save_curves:
        print(f"[CURVES] saved: {out_dir/'val_curves.npz'} and {out_dir/'test_curves.npz'}")
    if args.save_logits:
        print(f"[OUTPUTS] saved: {out_dir/'agg_outputs.npz'}")
    print(f"[DONE] results saved to: {out_dir}")


if __name__ == "__main__":
    main()

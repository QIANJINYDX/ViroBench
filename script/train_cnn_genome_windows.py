# script/train_cnn.py
from __future__ import annotations

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

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
# Import model / dataset
# -----------------------------
from model.cnn import GenomeCNN1D, CNNConfig
from util.genome_cls_dataset import GenomeClsDataset


# ============================================================
# Helpers
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


def make_run_name(prefix: str = "GenomeCls_CNN") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def short_hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


def file_fingerprint(p: Path, head_bytes: int = 2_048_576, tail_bytes: int = 2_048_576) -> Dict[str, Any]:
    """
    Stable fingerprint (content-based) to avoid rebuilding cache when only mtime changes.
    Uses md5 of head+tail samples.
    """
    st = p.stat()
    h = hashlib.md5()
    with open(p, "rb") as f:
        head = f.read(head_bytes)
        h.update(head)
        if st.st_size > head_bytes + tail_bytes:
            f.seek(max(0, st.st_size - tail_bytes))
            h.update(f.read(tail_bytes))
        else:
            h.update(f.read())
    return {"size": int(st.st_size), "md5_sample": h.hexdigest()}


# ============================================================
# Window slicing
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


def _make_starts(L: int, window_len: int, stride: int) -> List[int]:
    """
    Sliding windows starts.
    Ensures tail coverage by appending (L - window_len) when needed.
    """
    L = int(L)
    window_len = int(window_len)
    stride = int(stride)

    if L <= window_len:
        return [0]

    starts = list(range(0, L - window_len + 1, stride))
    if starts and starts[-1] != (L - window_len):
        starts.append(L - window_len)
    if not starts:
        starts = [0]
    return [int(s) for s in starts]


def _cap_starts(starts: List[int], max_windows_per_seq: int) -> List[int]:
    """
    If max_windows_per_seq > 0 and len(starts) > max_windows_per_seq,
    subsample uniformly (deterministic).
    If max_windows_per_seq <= 0 => keep all.
    """
    if max_windows_per_seq is None or int(max_windows_per_seq) <= 0:
        return starts
    K = int(max_windows_per_seq)
    if len(starts) <= K:
        return starts
    idxs = np.linspace(0, len(starts) - 1, K).round().astype(int)
    return [starts[j] for j in idxs]


class SlidingWindowDataset(Dataset):
    """
    Build windows from base dataset.

    - Train: max_windows_per_seq = train_windows_per_seq
        * -1 => all windows to sequence end
        * >0 => keep at most K windows per sequence (uniform subsample)

    - Val/Test: use max_windows_per_seq = -1 (all windows)

    Returns:
      (x_win:uint8[window_len], y:int, sid:int(base_idx))
    """

    def __init__(
        self,
        base_dataset: Dataset,
        window_len: int,
        stride: int,
        max_windows_per_seq: int,
        pad_idx: int = 0,
    ):
        self.base = base_dataset
        self.window_len = int(window_len)
        self.stride = int(stride)
        self.max_windows_per_seq = int(max_windows_per_seq)
        self.pad_idx = int(pad_idx)

        self._index: List[Tuple[int, int]] = []  # [(base_idx, start)]
        for base_idx in range(len(self.base)):
            item = self.base[base_idx]
            if len(item) == 2:
                x_full, _y = item
            else:
                x_full, _seq, _y = item

            L = int(len(x_full))
            starts = _make_starts(L, self.window_len, self.stride)
            starts = _cap_starts(starts, self.max_windows_per_seq)

            for st in starts:
                self._index.append((int(base_idx), int(st)))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int):
        base_idx, start = self._index[i]
        item = self.base[base_idx]
        if len(item) == 2:
            x_full, y = item
        else:
            x_full, _seq, y = item

        x_win = _pad_or_crop_window_uint8(x_full, start, self.window_len, self.pad_idx)
        return x_win, torch.tensor(int(y), dtype=torch.long), torch.tensor(int(base_idx), dtype=torch.long)


def extract_base_labels(base_ds: Dataset) -> torch.Tensor:
    ys = []
    for i in range(len(base_ds)):
        item = base_ds[i]
        if len(item) == 2:
            _x, y = item
        else:
            _x, _seq, y = item
        ys.append(int(y))
    return torch.tensor(ys, dtype=torch.long)


# ============================================================
# Cached tensor dataset (train/val/test)
# ============================================================
class TensorWindowDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, sid: Optional[torch.Tensor] = None):
        assert x.dtype == torch.uint8 and x.dim() == 2
        assert y.dtype == torch.long and y.dim() == 1
        assert x.size(0) == y.size(0)
        if sid is not None:
            assert sid.dtype == torch.long and sid.dim() == 1
            assert sid.size(0) == x.size(0)
        self.x = x
        self.y = y
        self.sid = sid

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, i: int):
        if self.sid is None:
            return self.x[i], self.y[i]
        return self.x[i], self.y[i], self.sid[i]


# ============================================================
# Cache dirs
# ============================================================
def make_cache_dir(cache_root: Path, splits_dir: Path, window_len: int, stride: int) -> Path:
    cache_root = ensure_dir(cache_root)
    group = ensure_dir(cache_root / f"genome_cls_w{int(window_len)}")
    dtag = short_hash(str(splits_dir.resolve()))
    return ensure_dir(group / f"stride{int(stride)}_d{dtag}")


def load_or_build_pt(
    pt_path: Path,
    meta_path: Path,
    expected_meta: Dict[str, Any],
    build_fn,
    rebuild: bool = False,
) -> Dict[str, torch.Tensor]:
    if (not rebuild) and pt_path.exists() and meta_path.exists():
        try:
            saved = load_json(meta_path)
            if saved == expected_meta:
                print(f"[CACHE HIT] {pt_path}")
                return torch.load(pt_path, map_location="cpu")
            else:
                print(f"[CACHE MISS] {pt_path.name} meta mismatch -> rebuild")
        except Exception:
            print(f"[CACHE MISS] {pt_path.name} meta load failed -> rebuild")

    obj = build_fn()
    ensure_dir(pt_path.parent)
    torch.save(obj, pt_path)
    save_json(expected_meta, meta_path)
    return obj


def build_windows_cache(
    base_ds: Dataset,
    split_name: str,
    window_len: int,
    stride: int,
    max_windows_per_seq: int,
    num_workers: int,
) -> Dict[str, torch.Tensor]:
    """
    Build and return:
      x: uint8 [N, window_len]
      y: long  [N]
      sid: long [N]
    """
    win_ds = SlidingWindowDataset(
        base_ds,
        window_len=window_len,
        stride=stride,
        max_windows_per_seq=max_windows_per_seq,
        pad_idx=0,
    )
    N = len(win_ds)
    x = torch.empty((N, window_len), dtype=torch.uint8)
    y = torch.empty((N,), dtype=torch.long)
    sid = torch.empty((N,), dtype=torch.long)

    loader = DataLoader(
        win_ds,
        batch_size=4096,
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    offset = 0
    pbar = tqdm(loader, desc=f"Caching {split_name.upper()} windows", leave=True)
    for xb, yb, sb in pbar:
        bsz = xb.size(0)
        x[offset : offset + bsz].copy_(xb)
        y[offset : offset + bsz].copy_(yb)
        sid[offset : offset + bsz].copy_(sb)
        offset += bsz

    assert offset == N
    return {"x": x, "y": y, "sid": sid}


# ============================================================
# Metrics (NO sklearn dependency)
# ============================================================
def softmax_np(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)  # rows=true, cols=pred
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def multiclass_mcc_from_cm(cm: np.ndarray) -> float:
    cm = cm.astype(np.float64)
    t_sum = cm.sum(axis=1)  # true per class
    p_sum = cm.sum(axis=0)  # pred per class
    n = cm.sum()
    c = np.trace(cm)

    s = n
    sum_p_t = np.sum(p_sum * t_sum)
    sum_p2 = np.sum(p_sum**2)
    sum_t2 = np.sum(t_sum**2)

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
    }


def roc_curve_binary(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = y_true.astype(np.int32)
    pos = y_true.sum()
    neg = y_true.size - pos
    if pos == 0 or neg == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    order = np.argsort(-y_score, kind="mergesort")
    y_true_s = y_true[order]
    y_score_s = y_score[order]

    distinct = np.where(np.diff(y_score_s))[0]
    thr_idx = np.r_[distinct, y_true_s.size - 1]

    tps = np.cumsum(y_true_s)[thr_idx]
    fps = (1 + thr_idx) - tps

    tpr = tps / (pos + 1e-12)
    fpr = fps / (neg + 1e-12)
    thresholds = y_score_s[thr_idx]

    tpr = np.r_[0.0, tpr].astype(np.float32)
    fpr = np.r_[0.0, fpr].astype(np.float32)
    thresholds = np.r_[thresholds[0] + 1e-6, thresholds].astype(np.float32)
    return fpr, tpr, thresholds


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
    pos = y_true.sum()
    if pos == 0:
        return float("nan"), np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    order = np.argsort(-y_score, kind="mergesort")
    y_true_s = y_true[order]
    y_score_s = y_score[order]

    tp = np.cumsum(y_true_s)
    fp = np.cumsum(1 - y_true_s)

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (pos + 1e-12)

    ap = float(np.sum(precision[y_true_s == 1]) / (pos + 1e-12))

    distinct = np.where(np.diff(y_score_s))[0]
    thr_idx = np.r_[distinct, y_true_s.size - 1]
    prec_curve = precision[thr_idx]
    rec_curve = recall[thr_idx]
    thr = y_score_s[thr_idx]

    prec_curve = np.r_[1.0, prec_curve].astype(np.float32)
    rec_curve = np.r_[0.0, rec_curve].astype(np.float32)
    thr = np.r_[thr[0] + 1e-6, thr].astype(np.float32)
    return ap, prec_curve, rec_curve, thr


def compute_all_metrics(
    y_true: np.ndarray,
    agg_logits: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_curves: bool = False,
    curves_out_npz: Optional[Path] = None,
) -> Dict[str, Any]:
    num_classes = agg_logits.shape[1]
    probs = softmax_np(agg_logits.astype(np.float64)).astype(np.float32)
    y_pred = probs.argmax(axis=1).astype(np.int64)

    cm = confusion_matrix_np(y_true, y_pred, num_classes=num_classes)
    acc = float((y_pred == y_true).mean())

    f1d = f1_scores_from_cm(cm)
    mcc = multiclass_mcc_from_cm(cm)

    per_auc = np.full((num_classes,), np.nan, dtype=np.float32)
    per_ap = np.full((num_classes,), np.nan, dtype=np.float32)

    roc_fpr, roc_tpr, roc_thr = {}, {}, {}
    pr_prec, pr_rec, pr_thr = {}, {}, {}

    for k in range(num_classes):
        yk = (y_true == k).astype(np.int32)
        sk = probs[:, k].astype(np.float64)

        fpr, tpr, thr = roc_curve_binary(yk, sk)
        auc = auc_trapz(fpr, tpr)
        per_auc[k] = np.float32(auc) if not np.isnan(auc) else np.float32(np.nan)

        ap, pc, rc, pthr = average_precision_and_pr_curve(yk, sk)
        per_ap[k] = np.float32(ap) if not np.isnan(ap) else np.float32(np.nan)

        if save_curves:
            roc_fpr[str(k)] = fpr
            roc_tpr[str(k)] = tpr
            roc_thr[str(k)] = thr
            pr_prec[str(k)] = pc
            pr_rec[str(k)] = rc
            pr_thr[str(k)] = pthr

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
        "num_classes": int(num_classes),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            "auroc_ovr": per_auc.tolist(),
            "aupr_ovr": per_ap.tolist(),
            "f1": f1d["per_class_f1"].tolist(),
            "precision": f1d["per_class_precision"].tolist(),
            "recall": f1d["per_class_recall"].tolist(),
            "support": support.astype(int).tolist(),
            "class_names": class_names if class_names is not None else None,
        },
    }

    if save_curves and curves_out_npz is not None:
        ensure_dir(curves_out_npz.parent)
        np.savez_compressed(
            curves_out_npz,
            roc_fpr=np.array([roc_fpr.get(str(k), np.array([], dtype=np.float32)) for k in range(num_classes)], dtype=object),
            roc_tpr=np.array([roc_tpr.get(str(k), np.array([], dtype=np.float32)) for k in range(num_classes)], dtype=object),
            roc_thr=np.array([roc_thr.get(str(k), np.array([], dtype=np.float32)) for k in range(num_classes)], dtype=object),
            pr_prec=np.array([pr_prec.get(str(k), np.array([], dtype=np.float32)) for k in range(num_classes)], dtype=object),
            pr_rec=np.array([pr_rec.get(str(k), np.array([], dtype=np.float32)) for k in range(num_classes)], dtype=object),
            pr_thr=np.array([pr_thr.get(str(k), np.array([], dtype=np.float32)) for k in range(num_classes)], dtype=object),
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
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x_u8.long())
            loss = ce(logits, y)
            loss.backward()
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
    Returns:
      avg_window_loss (float)
      agg_logits (np.ndarray float32) shape (num_base, C)
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
    return avg_window_loss, agg_logits


# ============================================================
# Main
# ============================================================
def main():
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()

    # fixed splits directory (train.csv/val.csv/test.csv/label2id.json)
    parser.add_argument("--splits_dir", type=str, required=True)

    # columns
    parser.add_argument("--label_column", type=str, default="family")
    parser.add_argument("--sequence_column", type=str, default="sequence")

    # windowing (统一逻辑)
    parser.add_argument("--window_len", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument(
        "--train_windows_per_seq",
        type=int,
        default=8,
        help="每条训练序列最多取多少个滑窗；-1 表示滑到序列尾部（使用所有滑窗）",
    )

    # caching (仅缓存 train/val/test 三个 split 的 windows)
    parser.add_argument(
        "--public_cache_root",
        type=str,
        default="/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/cache",
    )
    parser.add_argument("--cache_rebuild", action="store_true")

    # base full-seq cache (GenomeClsDataset 内部缓存，可选)
    parser.add_argument("--base_cache_dir", type=str, default="")

    # training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    # workers
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, (os.cpu_count() or 8) - 1),
        help="用于构建 cache 的 DataLoader workers；训练/评估会自动设置为更安全的值",
    )

    # device
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--amp", action="store_true")

    # results
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--run_name", type=str, default="")

    # model hparams
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--head_hidden", type=int, default=256)
    parser.add_argument("--head_dropout", type=float, default=0.3)

    # resnet-ish
    parser.add_argument("--channels", type=str, default="64,128,256", help="Comma-separated stage channels")
    parser.add_argument("--blocks_per_stage", type=str, default="2,2,2", help="Comma-separated blocks per stage")
    parser.add_argument("--norm", type=str, default="bn", choices=["bn", "gn"])
    parser.add_argument("--gn_groups", type=int, default=8)
    parser.add_argument("--global_pool", type=str, default="avg", choices=["avg", "max"])
    parser.add_argument("--use_se", action="store_true")
    parser.add_argument("--se_ratio", type=float, default=0.25)

    # metrics / curves
    parser.add_argument("--save_curves", action="store_true")
    parser.add_argument("--save_logits", action="store_true")

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
    rn = args.run_name.strip() or make_run_name("GenomeCls_CNN")
    out_dir = ensure_dir(Path(args.results_dir) / rn)
    ckpt_dir = ensure_dir(out_dir / "checkpoints")

    # load fixed splits
    splits_dir = Path(args.splits_dir)
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"
    label2id_path = splits_dir / "label2id.json"

    for p in [train_csv, val_csv, test_csv, label2id_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    label2id = load_json(label2id_path)
    num_classes = len(label2id)
    id2label = {int(v): str(k) for k, v in label2id.items()}
    class_names = [id2label[i] for i in range(num_classes)]

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # base datasets
    base_cache_root = Path(args.base_cache_dir) if args.base_cache_dir else None
    train_base_cache = str(base_cache_root / "train") if base_cache_root else ""
    val_base_cache = str(base_cache_root / "val") if base_cache_root else ""
    test_base_cache = str(base_cache_root / "test") if base_cache_root else ""

    train_base = GenomeClsDataset(
        csv_path=train_df,
        label2id=label2id,
        label_column=args.label_column,
        sequence_column=args.sequence_column,
        cache_dir=train_base_cache,
    )
    val_base = GenomeClsDataset(
        csv_path=val_df,
        label2id=label2id,
        label_column=args.label_column,
        sequence_column=args.sequence_column,
        cache_dir=val_base_cache,
    )
    test_base = GenomeClsDataset(
        csv_path=test_df,
        label2id=label2id,
        label_column=args.label_column,
        sequence_column=args.sequence_column,
        cache_dir=test_base_cache,
    )

    print(f"[INFO] device={device} cuda={torch.cuda.is_available()} cache_workers={args.num_workers}")
    print(f"[INFO] base sizes train/val/test = {len(train_base)}/{len(val_base)}/{len(test_base)}")
    print(f"[INFO] classes={num_classes}")
    print(f"[INFO] window_len={args.window_len} stride={args.stride} train_windows_per_seq={args.train_windows_per_seq}")
    print(f"[INFO] splits_dir={splits_dir.resolve()} label2id={label2id_path}")

    # save config
    cfg: Dict[str, Any] = {
        "run_name": rn,
        "cmd": " ".join(sys.argv),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "splits_dir": str(splits_dir.resolve()),
        "files": {
            "train": str(train_csv),
            "val": str(val_csv),
            "test": str(test_csv),
            "label2id": str(label2id_path),
        },
        "columns": {"label": args.label_column, "sequence": args.sequence_column},
        "windowing": {
            "window_len": int(args.window_len),
            "stride": int(args.stride),
            "train_windows_per_seq": int(args.train_windows_per_seq),
            "val_test_windows_per_seq": -1,  # always full
        },
        "cache": {
            "public_cache_root": args.public_cache_root,
            "cache_rebuild": bool(args.cache_rebuild),
            "base_cache_dir": args.base_cache_dir,
        },
        "train": {
            "epochs": args.epochs,
            "eval_every": args.eval_every,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "amp": bool(args.amp),
            "gpu": args.gpu,
        },
        "model": {
            "type": "ResCNN1D",
            "vocab_size": 5,
            "embed_dim": args.embed_dim,
            "kernel_size": args.kernel_size,
            "channels": channels,
            "blocks_per_stage": blocks_per_stage,
            "norm": args.norm,
            "gn_groups": args.gn_groups,
            "dropout": args.dropout,
            "global_pool": args.global_pool,
            "head_hidden": args.head_hidden,
            "head_dropout": args.head_dropout,
            "use_se": bool(args.use_se),
            "se_ratio": args.se_ratio,
            "out_dim": num_classes,
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

    # ============================================================
    # CACHE: only cache windows for train/val/test
    # ============================================================
    cache_root = Path(args.public_cache_root)
    cache_dir = make_cache_dir(cache_root, splits_dir, args.window_len, args.stride)
    print(f"[INFO] cache_dir = {cache_dir}")
    save_json({"cache_dir": str(cache_dir)}, out_dir / "cache_path.json")

    common_fp = {
        "splits_dir": str(splits_dir.resolve()),
        "window_len": int(args.window_len),
        "stride": int(args.stride),
        "num_classes": int(num_classes),
        "label2id": label2id,
        "train_fp": file_fingerprint(train_csv),
        "val_fp": file_fingerprint(val_csv),
        "test_fp": file_fingerprint(test_csv),
        "label2id_fp": file_fingerprint(label2id_path),
    }

    # TRAIN cache: cap by train_windows_per_seq (or -1 for full)
    train_expected_meta = {
        "split": "train",
        "max_windows_per_seq": int(args.train_windows_per_seq),
        **common_fp,
    }
    train_obj = load_or_build_pt(
        pt_path=cache_dir / "train_windows.pt",
        meta_path=cache_dir / "train_windows.meta.json",
        expected_meta=train_expected_meta,
        build_fn=lambda: build_windows_cache(
            train_base,
            "train",
            window_len=args.window_len,
            stride=args.stride,
            max_windows_per_seq=args.train_windows_per_seq,
            num_workers=args.num_workers,
        ),
        rebuild=args.cache_rebuild,
    )

    # VAL/TEST cache: always full (-1)
    val_expected_meta = {
        "split": "val",
        "max_windows_per_seq": -1,
        **common_fp,
    }
    val_obj = load_or_build_pt(
        pt_path=cache_dir / "val_windows.pt",
        meta_path=cache_dir / "val_windows.meta.json",
        expected_meta=val_expected_meta,
        build_fn=lambda: build_windows_cache(
            val_base,
            "val",
            window_len=args.window_len,
            stride=args.stride,
            max_windows_per_seq=-1,
            num_workers=args.num_workers,
        ),
        rebuild=args.cache_rebuild,
    )

    test_expected_meta = {
        "split": "test",
        "max_windows_per_seq": -1,
        **common_fp,
    }
    test_obj = load_or_build_pt(
        pt_path=cache_dir / "test_windows.pt",
        meta_path=cache_dir / "test_windows.meta.json",
        expected_meta=test_expected_meta,
        build_fn=lambda: build_windows_cache(
            test_base,
            "test",
            window_len=args.window_len,
            stride=args.stride,
            max_windows_per_seq=-1,
            num_workers=args.num_workers,
        ),
        rebuild=args.cache_rebuild,
    )

    # Datasets/Loaders
    train_ds = TensorWindowDataset(train_obj["x"], train_obj["y"])  # (x,y)
    val_ds = TensorWindowDataset(val_obj["x"], val_obj["y"], val_obj["sid"])  # (x,y,sid)
    test_ds = TensorWindowDataset(test_obj["x"], test_obj["y"], test_obj["sid"])

    # 注意：大 tensor 用多 worker 可能导致额外内存开销，默认更稳妥
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    print(f"[INFO] cached windows: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    # --------- model ----------
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
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    # log
    log_path = out_dir / "train_log.csv"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,val_window_loss,val_acc,val_mcc,val_macro_f1,val_macro_auroc,val_macro_aupr\n")

    best_val_acc = -1.0
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"

    # ============================================================
    # Train
    # ============================================================
    epoch_pbar = tqdm(range(1, args.epochs + 1), desc="Epochs", leave=True)
    for epoch in epoch_pbar:
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
            "cache_dir": str(cache_dir),
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

    # VAL final
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
            test_probs=test_probs.astype(np.float32),
            test_y_true=y_test_base.astype(np.int64),
            test_y_pred=test_probs.argmax(axis=1).astype(np.int64),
            class_names=np.array(class_names, dtype=object),
        )

    summary = {
        "best_val_acc": best_val_acc,
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
        "cache": {
            "cache_dir": str(cache_dir),
            "train_windows_pt": str(cache_dir / "train_windows.pt"),
            "val_windows_pt": str(cache_dir / "val_windows.pt"),
            "test_windows_pt": str(cache_dir / "test_windows.pt"),
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
    print(f"[DONE] cache_dir: {cache_dir}")


if __name__ == "__main__":
    main()

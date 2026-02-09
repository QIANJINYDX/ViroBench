"""
通用病毒数据集封装：给定一个 split 文件夹，自动构建 train/val/test 三个 Dataset。

目标场景
- split_dir 下通常包含：train.csv / val.csv / test.csv（“元数据/标签”）
- 以及可选的：train_sequences.jsonl / val_sequences.jsonl / test_sequences.jsonl
  或 train_sequences.csv / val_sequences.csv / test_sequences.csv（“序列”）

你可以指定 label（单标签或多标签）：
- label_cols="host_label"（单标签）
- label_cols=["host_label", "family"]（多标签：返回多个标签）

支持输出格式：
- return_format="tuple": 返回 (sequence, label) 或 (sequence, labels_array)
- return_format="dict" : 返回 dict，包括 taxid/sequence/labels 等字段
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


SplitName = Literal["train", "val", "test"]
ReturnFormat = Literal["tuple", "dict"]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _resolve_split_file(split_dir: Path, split: SplitName, kind: Literal["meta", "seq"]) -> Path:
    """
    自动发现 split 文件。

    meta: 优先 {split}.csv
    seq : 优先 {split}_sequences.jsonl > {split}_sequences.csv
    """
    if kind == "meta":
        p = split_dir / f"{split}.csv"
        if not p.exists():
            raise FileNotFoundError(f"missing meta file: {p}")
        return p

    # kind == "seq"
    candidates = [
        split_dir / f"{split}_sequences.jsonl",
        split_dir / f"{split}_sequences.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"missing sequence file for split={split}: tried {[str(x) for x in candidates]}")


def _load_sequences_table(seq_path: Path) -> pd.DataFrame:
    """
    读取序列表，返回 DataFrame，至少包含：
    - taxid
    - sequence (csv) 或 sequences(list) (jsonl)
    """
    if seq_path.suffix.lower() == ".csv":
        df = pd.read_csv(seq_path)
        if "sequence" not in df.columns:
            raise KeyError(f"{seq_path} missing column: sequence")
        if "taxid" not in df.columns:
            raise KeyError(f"{seq_path} missing column: taxid")
        return df[["taxid", "sequence"]].copy()

    if seq_path.suffix.lower() == ".jsonl":
        rows = _read_jsonl(seq_path)
        df = pd.DataFrame(rows)
        if "sequences" not in df.columns:
            raise KeyError(f"{seq_path} missing key: sequences")
        if "taxid" not in df.columns:
            raise KeyError(f"{seq_path} missing key: taxid")
        return df[["taxid", "sequences"]].copy()

    raise ValueError(f"unsupported sequence file: {seq_path}")


def _coerce_label_cols(label_cols: Optional[Union[str, Sequence[str]]]) -> List[str]:
    if label_cols is None:
        return []
    if isinstance(label_cols, str):
        return [label_cols]
    return list(label_cols)


def _build_label2id(series: pd.Series) -> Dict[str, int]:
    vals = series.dropna().astype(str).tolist()
    uniq = sorted(set(v.strip() for v in vals if v.strip() != ""))
    return {v: i for i, v in enumerate(uniq)}


def _encode_labels(
    df: pd.DataFrame,
    label_cols: List[str],
    label2id: Optional[Dict[str, Dict[str, int]]],
    *,
    build_from_train: bool,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """
    将 label_cols 编码成整数（若是字符串类别）。
    返回 (df_with_label_ids, label2id_by_col)。
    """
    label2id_by_col: Dict[str, Dict[str, int]] = {} if label2id is None else {k: dict(v) for k, v in label2id.items()}

    for col in label_cols:
        if col not in df.columns:
            raise KeyError(f"missing label column: {col}")

        s = df[col]
        # 若本来就是数值，直接保留
        if pd.api.types.is_numeric_dtype(s):
            df[f"{col}__id"] = pd.to_numeric(s, errors="coerce")
            continue

        # 字符串类别：需要映射
        if col not in label2id_by_col:
            if not build_from_train:
                raise ValueError(f"label2id for '{col}' not provided (and build_from_train=False)")
            label2id_by_col[col] = _build_label2id(s)

        mapping = label2id_by_col[col]
        df[f"{col}__id"] = s.astype(str).map(lambda x: mapping.get(x.strip(), -1)).astype(int)

    return df, label2id_by_col


def _attach_sequences(
    meta_df: pd.DataFrame,
    seq_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    将 seq_df 的 sequence/sequences 附加到 meta_df。
    优先按行对齐（长度相同且 taxid 完全一致）；
    否则按 taxid 合并（要求 one-to-one，否则会报错）。
    """
    # 统一 taxid 类型（尽量转成 int）
    meta_tax = pd.to_numeric(meta_df["taxid"], errors="coerce")
    seq_tax = pd.to_numeric(seq_df["taxid"], errors="coerce")
    meta_df = meta_df.copy()
    seq_df = seq_df.copy()
    meta_df["taxid"] = meta_tax
    seq_df["taxid"] = seq_tax

    if len(meta_df) == len(seq_df) and meta_df["taxid"].equals(seq_df["taxid"]):
        # 行对齐：最快且不会引入重复
        for c in ("sequence", "sequences"):
            if c in seq_df.columns:
                meta_df[c] = seq_df[c].values
        return meta_df

    # taxid join：要求唯一
    if seq_df["taxid"].duplicated().any():
        raise ValueError("sequence file has duplicated taxid; cannot safely merge by taxid")
    if meta_df["taxid"].duplicated().any():
        # 允许 meta 有重复 taxid（多条样本共享 taxid），这种情况 merge 仍然是 many-to-one
        pass

    # many-to-one: meta can have duplicated taxid, seq must be unique by taxid
    return meta_df.merge(
        seq_df,
        on="taxid",
        how="left",
        validate="many_to_one",
    )


class VirusSequenceDataset(Dataset):
    """
    单个 split 的 Dataset。

    sequence_source:
    - 若 df 有 'sequence'（字符串），直接使用
    - 若 df 有 'sequences'（list[str]），默认 join 成一个字符串；也可 return_list=True 返回 list
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        label_cols: Optional[Union[str, Sequence[str]]] = None,
        label2id: Optional[Dict[str, Dict[str, int]]] = None,
        return_format: ReturnFormat = "tuple",
        return_list: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.label_cols = _coerce_label_cols(label_cols)
        self.label2id = label2id or {}
        self.return_format = return_format
        self.return_list = bool(return_list)

        if "taxid" not in self.df.columns:
            raise KeyError("dataset df missing column: taxid")

        # label id columns（如果 label_cols 为空，则不做）
        self.label_id_cols = [f"{c}__id" for c in self.label_cols if f"{c}__id" in self.df.columns]
        if self.label_cols and not self.label_id_cols:
            # 尝试直接使用 label_cols（可能本身就是数值）
            self.label_id_cols = list(self.label_cols)

        # sequence columns
        self.has_sequence = "sequence" in self.df.columns
        self.has_sequences = "sequences" in self.df.columns
        if not self.has_sequence and not self.has_sequences:
            raise KeyError("dataset df missing sequence column: expected 'sequence' or 'sequences'")

    def __len__(self) -> int:
        return len(self.df)

    # ---- public helpers (for wrapper datasets) ----
    def get_taxid(self, idx: int):
        return self.df.iloc[idx]["taxid"]

    def get_sequence(self, idx: int) -> Union[str, List[str]]:
        return self._get_sequence(idx)

    def get_labels(self, idx: int) -> Any:
        return self._get_labels(idx)

    def _get_sequence(self, idx: int) -> Union[str, List[str]]:
        row = self.df.iloc[idx]
        if self.has_sequence:
            s = row["sequence"]
            return "" if pd.isna(s) else str(s)
        # sequences list
        seqs = row["sequences"]
        if seqs is None or (isinstance(seqs, float) and np.isnan(seqs)):
            seq_list: List[str] = []
        elif isinstance(seqs, list):
            seq_list = [str(x) for x in seqs]
        else:
            # 兼容从 CSV 读出的字符串形式的 list（极少数情况下）
            try:
                seq_list = json.loads(seqs)
                if not isinstance(seq_list, list):
                    seq_list = [str(seqs)]
            except Exception:
                seq_list = [str(seqs)]

        if self.return_list:
            return seq_list
        return "".join(seq_list)

    def _get_labels(self, idx: int) -> Any:
        if not self.label_id_cols:
            return None
        row = self.df.iloc[idx]
        if len(self.label_id_cols) == 1:
            return int(row[self.label_id_cols[0]])
        return np.array([int(row[c]) for c in self.label_id_cols], dtype=np.int64)

    def __getitem__(self, idx: int):
        seq = self._get_sequence(idx)
        y = self._get_labels(idx)
        taxid = self.df.iloc[idx]["taxid"]

        if self.return_format == "dict":
            out = {"taxid": taxid, "sequence": seq, "labels": y}
            # 额外 label 名称（原始字符串）也返回
            for c in self.label_cols:
                if c in self.df.columns:
                    out[c] = self.df.iloc[idx][c]
            return out

        # tuple
        return seq, y


def split_nonoverlap_windows(seq_len: int, window_len: int) -> List[int]:
    """
    生成窗口起点列表（start indices）。

    规则（按你的描述实现）：
    - 先按 window_len 做无重叠切分：0, w, 2w, ...
    - 若末尾有剩余且不足 window_len，则再额外加入一个“从末尾倒着取”的窗口：
        start = max(0, seq_len - window_len)
      这样最后一个窗口一定覆盖序列尾部；可能与前一个窗口有重叠，这是预期行为。
    """
    w = int(window_len)
    L = int(seq_len)
    if w <= 0:
        raise ValueError("window_len must be > 0")
    if L <= 0:
        # 空序列：不生成窗口，避免产生空窗口
        return []
    if L <= w:
        return [0]

    starts = list(range(0, L - w + 1, w))
    if not starts:
        starts = [0]
    last_start = starts[-1]
    last_end = last_start + w
    if last_end < L:
        tail_start = max(0, L - w)
        if tail_start != last_start:
            starts.append(tail_start)
    return starts


class WindowedVirusDataset(Dataset):
    """
    将 VirusSequenceDataset “展开”为 window-level 的 dataset。

    - train: 每条序列随机选择 train_num_windows 个窗口（无重叠窗口列表中采样），每个 epoch 可重采样
    - val/test: 覆盖每条序列的所有窗口（用于聚合判断一条序列）

    返回（return_format="dict"）字段：
      - taxid
      - sequence: 当前窗口序列（string；若 base 返回 list 且 return_list=True，则保持 list 形式也可以）
      - labels
      - seq_index: 原始序列在 base dataset 中的 index
      - window_index: 当前窗口在该序列窗口列表中的序号
      - num_windows: 该序列总窗口数
      - window_start: 窗口起点
    """

    def __init__(
        self,
        base: VirusSequenceDataset,
        *,
        split: SplitName,
        window_len: int,
        train_num_windows: int = 0,
        eval_num_windows: int = -1,
        seed: int = 42,
        return_format: ReturnFormat = "dict",
    ):
        self.base = base
        self.split = split
        self.window_len = int(window_len)
        self.train_num_windows = int(train_num_windows)
        self.eval_num_windows = int(eval_num_windows)
        self.seed = int(seed)
        self.return_format = return_format

        # 预计算每条序列的窗口 starts
        self._starts_by_seq: List[List[int]] = []
        for i in range(len(self.base)):
            seq = self.base.get_sequence(i)
            L = len(seq) if isinstance(seq, str) else sum(len(x) for x in seq)
            self._starts_by_seq.append(split_nonoverlap_windows(L, self.window_len))
        empty_seq_count = sum(1 for starts in self._starts_by_seq if len(starts) == 0)
        if empty_seq_count > 0:
            print(f"[WARN] WindowedVirusDataset split={self.split}: "
                  f"skipping {empty_seq_count} empty sequences")

        self._epoch = 0
        self._flat_index: List[Tuple[int, int]] = []  # (seq_index, window_index_in_seq)
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        """训练集每个 epoch 调用一次可重采样窗口。val/test 可不调用。"""
        self._epoch = int(epoch)
        rng = np.random.default_rng(self.seed + self._epoch)
        flat: List[Tuple[int, int]] = []

        if self.split == "train":
            k_req = max(0, int(self.train_num_windows))
            for si, starts in enumerate(self._starts_by_seq):
                n = len(starts)
                if n <= 0:
                    # 空序列：不产生窗口
                    continue
                if k_req <= 0 or k_req >= n:
                    # k_req<=0: 退化为全窗口（有时你可能想训练也全覆盖）
                    for wi in range(n):
                        flat.append((si, wi))
                else:
                    # 不放回采样 window indices
                    chosen = rng.choice(np.arange(n), size=k_req, replace=False)
                    for wi in chosen.tolist():
                        flat.append((si, int(wi)))
        else:
            # val/test：按指定数量随机选窗口（-1 表示全覆盖）
            k_req = int(self.eval_num_windows)
            for si, starts in enumerate(self._starts_by_seq):
                n = len(starts)
                if n <= 0:
                    # 空序列：不产生窗口
                    continue
                if k_req < 0 or k_req >= n:
                    for wi in range(n):
                        flat.append((si, wi))
                else:
                    chosen = rng.choice(np.arange(n), size=k_req, replace=False)
                    for wi in chosen.tolist():
                        flat.append((si, int(wi)))

        self._flat_index = flat

    def __len__(self) -> int:
        return len(self._flat_index)

    def _slice_window(self, seq: Union[str, List[str]], start: int) -> str:
        """
        将单条序列切成窗口字符串。
        - 若 seq 是 list[str]（多 contig），先直接拼接为字符串再切窗口
          （保持与你之前“整条序列判断”一致的视角；如需按 contig 分开，可再拓展）
        """
        if isinstance(seq, list):
            seq_str = "".join(seq)
        else:
            seq_str = seq
        L = len(seq_str)
        w = self.window_len
        if L <= w:
            return seq_str
        s = int(start)
        if s + w <= L:
            return seq_str[s : s + w]
        # 兜底：从末尾倒着取
        return seq_str[max(0, L - w) : L]

    def __getitem__(self, idx: int):
        seq_index, win_index = self._flat_index[idx]
        starts = self._starts_by_seq[seq_index]
        win_index = int(win_index)
        win_index = min(max(win_index, 0), len(starts) - 1)
        start = int(starts[win_index])

        taxid = self.base.get_taxid(seq_index)
        seq = self.base.get_sequence(seq_index)
        y = self.base.get_labels(seq_index)
        win_seq = self._slice_window(seq, start)

        if self.return_format == "tuple":
            return win_seq, y

        return {
            "taxid": taxid,
            "sequence": win_seq,
            "labels": y,
            "seq_index": int(seq_index),
            "window_index": int(win_index),
            "num_windows": int(len(starts)),
            "window_start": int(start),
        }


@dataclass
class VirusSplitDatasets:
    """
    传入 split 文件夹，自动构建 train/val/test 三个 dataset。
    """

    split_dir: Union[str, Path]
    label_cols: Optional[Union[str, Sequence[str]]] = None
    return_format: ReturnFormat = "tuple"
    return_list: bool = False
    # 是否在 split_dir 中寻找 {split}_sequences.(jsonl|csv) 并与 {split}.csv 结合
    attach_sequences: bool = True

    train: VirusSequenceDataset = None  # type: ignore
    val: VirusSequenceDataset = None  # type: ignore
    test: VirusSequenceDataset = None  # type: ignore
    label2id: Dict[str, Dict[str, int]] = None  # type: ignore

    def __post_init__(self):
        sd = Path(self.split_dir).resolve()
        self.label2id = {}
        label_cols_list = _coerce_label_cols(self.label_cols)

        # ---- load train first (build label2id if needed) ----
        train_meta = pd.read_csv(_resolve_split_file(sd, "train", "meta"))
        if self.attach_sequences:
            try:
                train_seq = _load_sequences_table(_resolve_split_file(sd, "train", "seq"))
                train_meta = _attach_sequences(train_meta, train_seq)
            except FileNotFoundError:
                pass

        if label_cols_list:
            train_meta, self.label2id = _encode_labels(
                train_meta,
                label_cols_list,
                label2id=None,
                build_from_train=True,
            )

        self.train = VirusSequenceDataset(
            train_meta,
            label_cols=label_cols_list,
            label2id=self.label2id,
            return_format=self.return_format,
            return_list=self.return_list,
        )

        # ---- val/test: reuse label2id ----
        for split in ("val", "test"):
            meta = pd.read_csv(_resolve_split_file(sd, split, "meta"))
            if self.attach_sequences:
                try:
                    seq = _load_sequences_table(_resolve_split_file(sd, split, "seq"))
                    meta = _attach_sequences(meta, seq)
                except FileNotFoundError:
                    pass

            if label_cols_list:
                meta, _ = _encode_labels(
                    meta,
                    label_cols_list,
                    label2id=self.label2id,
                    build_from_train=False,
                )

            ds = VirusSequenceDataset(
                meta,
                label_cols=label_cols_list,
                label2id=self.label2id,
                return_format=self.return_format,
                return_list=self.return_list,
            )
            setattr(self, split, ds)

    def make_windowed(
        self,
        *,
        window_len: int,
        train_num_windows: int,
        eval_num_windows: int = -1,
        seed: int = 42,
        return_format: ReturnFormat = "dict",
    ) -> "VirusWindowedSplitDatasets":
        """
        基于当前的 train/val/test base datasets 构造 window-level datasets。
        - train: 随机采样 train_num_windows
        - val/test: 全覆盖
        """
        return VirusWindowedSplitDatasets(
            base=self,
            window_len=window_len,
            train_num_windows=train_num_windows,
            eval_num_windows=eval_num_windows,
            seed=seed,
            return_format=return_format,
        )


@dataclass
class VirusWindowedSplitDatasets:
    """
    window-level 的 train/val/test datasets。
    你可以在训练时每个 epoch 调用 train.set_epoch(epoch) 来重采样窗口。
    """

    base: VirusSplitDatasets
    window_len: int
    train_num_windows: int
    eval_num_windows: int = -1
    seed: int = 42
    return_format: ReturnFormat = "dict"

    train: WindowedVirusDataset = None  # type: ignore
    val: WindowedVirusDataset = None  # type: ignore
    test: WindowedVirusDataset = None  # type: ignore

    def __post_init__(self):
        self.train = WindowedVirusDataset(
            self.base.train,
            split="train",
            window_len=self.window_len,
            train_num_windows=self.train_num_windows,
            eval_num_windows=self.eval_num_windows,
            seed=self.seed,
            return_format=self.return_format,
        )
        self.val = WindowedVirusDataset(
            self.base.val,
            split="val",
            window_len=self.window_len,
            train_num_windows=0,
            eval_num_windows=self.eval_num_windows,
            seed=self.seed,
            return_format=self.return_format,
        )
        self.test = WindowedVirusDataset(
            self.base.test,
            split="test",
            window_len=self.window_len,
            train_num_windows=0,
            eval_num_windows=self.eval_num_windows,
            seed=self.seed,
            return_format=self.return_format,
        )

if __name__ == "__main__":
    import argparse
    from pprint import pprint

    ap = argparse.ArgumentParser(description="Quick sanity checks for VirusSplitDatasets")
    ap.add_argument(
        "--split_dir",
        default="data/all_viral/cls_data/C1-times",
        help="Folder containing train/val/test.csv and optional *_sequences.(jsonl|csv)",
    )
    ap.add_argument(
        "--label_cols",
        default="host_label",
        help="Label column(s). Comma-separated for multi-label, e.g. 'host_label,family'. Empty means no labels.",
    )
    ap.add_argument(
        "--return_format",
        choices=["tuple", "dict"],
        default="dict",
        help="Sample return format",
    )
    ap.add_argument(
        "--return_list",
        action="store_true",
        help="If sequences are stored as list (jsonl), return list instead of joined string.",
    )
    ap.add_argument(
        "--no_attach_sequences",
        action="store_true",
        help="Do not attach *_sequences.* files; metadata-only. (Will fail if no sequence column in meta.)",
    )
    ap.add_argument("--show_n", type=int, default=2, help="How many samples to print per split.")
    ap.add_argument("--window_len", type=int, default=0, help="If >0, build windowed datasets with this window length.")
    ap.add_argument("--train_num_windows", type=int, default=0, help="Train: random windows per sequence (only used when window_len>0).")
    ap.add_argument(
        "--eval_num_windows",
        type=int,
        default=-1,
        help="Val/Test: windows per sequence (-1 for all windows).",
    )
    args = ap.parse_args()

    label_cols: Optional[Union[str, Sequence[str]]]
    s = (args.label_cols or "").strip()
    if s == "":
        label_cols = None
    elif "," in s:
        label_cols = [x.strip() for x in s.split(",") if x.strip()]
    else:
        label_cols = s

    print("[INFO] split_dir:", args.split_dir)
    print("[INFO] label_cols:", label_cols)
    print("[INFO] return_format:", args.return_format)
    print("[INFO] return_list:", bool(args.return_list))
    print("[INFO] attach_sequences:", not bool(args.no_attach_sequences))

    bundle = VirusSplitDatasets(
        args.split_dir,
        label_cols=label_cols,
        return_format=args.return_format,
        return_list=bool(args.return_list),
        attach_sequences=not bool(args.no_attach_sequences),
    )

    print("[OK] dataset sizes:", len(bundle.train), len(bundle.val), len(bundle.test))
    print("[OK] label2id keys:", list(bundle.label2id.keys()))

    # Optional: windowed dataset demo
    if int(args.window_len) > 0:
        wbundle = bundle.make_windowed(
            window_len=int(args.window_len),
            train_num_windows=int(args.train_num_windows),
            eval_num_windows=int(args.eval_num_windows),
            seed=42,
            return_format=args.return_format,
        )
        print("[OK] windowed sizes:", len(wbundle.train), len(wbundle.val), len(wbundle.test))

    def show_samples(name: str, ds: Dataset):
        print(f"\n--- {name} samples ---")
        for i in range(min(int(args.show_n), len(ds))):
            item = ds[i]
            if isinstance(item, dict):
                # avoid dumping huge sequences; show lengths only
                seq = item.get("sequence")
                item2 = dict(item)
                if isinstance(seq, str):
                    item2["sequence_len"] = len(seq)
                    item2["sequence"] = seq[:50] + ("..." if len(seq) > 50 else "")
                elif isinstance(seq, list):
                    item2["sequence_lens"] = [len(x) for x in seq[:3]]
                    item2["sequence"] = f"<list len={len(seq)}>"
                pprint(item2)
            else:
                seq, y = item
                if isinstance(seq, str):
                    print({"sequence_len": len(seq), "labels": y})
                else:
                    print({"sequence_type": type(seq).__name__, "labels": y})

    show_samples("train", bundle.train)
    show_samples("val", bundle.val)
    show_samples("test", bundle.test)

    if int(args.window_len) > 0:
        show_samples("train(windowed)", wbundle.train)  # type: ignore
        show_samples("val(windowed)", wbundle.val)      # type: ignore
        show_samples("test(windowed)", wbundle.test)    # type: ignore
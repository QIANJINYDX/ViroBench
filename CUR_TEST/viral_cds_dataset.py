# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import gzip
import json
import hashlib
from typing import Dict, Any, Iterator, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from Bio import SeqIO

import torch
from torch.utils.data import Dataset


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _safe_mkdir(d: str):
    os.makedirs(d, exist_ok=True)


def _open_maybe_gzip(path: str, mode: str = "rt"):
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)


def _normalize_nt(seq: str) -> str:
    seq = seq.upper().replace("U", "T")
    return re.sub(r"[^ACGTNRYKMSWBDHV]", "N", seq)


def iter_cds_from_fasta(cds_fasta_path: str) -> Iterator[Tuple[str, Dict[str, Any]]]:
    with _open_maybe_gzip(cds_fasta_path, "rt") as f:
        for rec in SeqIO.parse(f, "fasta"):
            seq = _normalize_nt(str(rec.seq))
            yield seq, {"cds_id": rec.id, "cds_desc": rec.description}


def _coerce_db_source(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ("genbank", "gb", "gbk"):
        return "genbank"
    if s in ("refseq", "rs"):
        return "refseq"
    return None


def find_cds_fasta_for_row(row: pd.Series, downloads_root: str) -> Optional[str]:
    """
    downloads_root 形如：
      /.../ncbi_viral
    其中包含：
      genbank/downloads/{taxid}/{asm}/...
      refseq/downloads/{taxid}/{asm}/...

    优先用 row['db_source'] 指定的库；找不到则 fallback 到另一个库。
    """
    asm = str(row.get("asm", "") or "").strip()
    taxid = str(row.get("taxid", "") or "").strip()
    if not asm or not taxid:
        return None

    # 允许 TSV 里直接给 cds_fasta_path / cds_path（最高优先级）
    for k in ("cds_fasta_path", "cds_path"):
        if k in row and pd.notna(row[k]) and str(row[k]).strip():
            p = str(row[k]).strip()
            if os.path.exists(p) and os.path.getsize(p) > 0:
                return p

    db_pref = _coerce_db_source(row.get("db_source", None)) or _coerce_db_source(row.get("source", None))

    # 你截图中的标准文件名
    filename = f"{asm}_cds_from_genomic.fna.gz"
    filename2 = f"{asm}_cds_from_genomic.fna"  # 有些不压缩

    def candidate_paths(db: str) -> List[str]:
        base = os.path.join(downloads_root, db, "downloads", taxid, asm)
        return [
            os.path.join(base, filename),
            os.path.join(base, filename2),
            # 容错：有些人会把 cds 放成 cds.fna(.gz)
            os.path.join(base, f"{asm}_cds.fna.gz"),
            os.path.join(base, f"{asm}_cds.fna"),
        ]

    search_order = []
    if db_pref in ("genbank", "refseq"):
        search_order.append(db_pref)
        search_order.append("refseq" if db_pref == "genbank" else "genbank")
    else:
        # 不知道来自哪个库：两个都查
        search_order = ["genbank", "refseq"]

    for db in search_order:
        for p in candidate_paths(db):
            if os.path.exists(p) and os.path.getsize(p) > 0:
                return p

    return None


def build_cds_cache_from_fasta(
    tsv_path: str,
    downloads_root: str,
    cache_dir: str,
    prefer_fixed_group: bool = True,
    min_len: int = 60,
    max_len: int = 1_000_000,
    allowed_host_groups: Optional[List[str]] = None,
) -> str:
    _safe_mkdir(cache_dir)
    df = pd.read_csv(tsv_path, sep="\t")

    group_col = "host_group_fixed" if prefer_fixed_group and "host_group_fixed" in df.columns else "host_group"
    if group_col not in df.columns:
        raise ValueError(f"Need host_group_fixed or host_group, got: {df.columns.tolist()}")

    cache_key = _sha1(os.path.abspath(tsv_path) + "||" + os.path.abspath(downloads_root))[:12]
    out_path = os.path.join(cache_dir, f"cds_cache_fasta.{cache_key}.jsonl.gz")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    with gzip.open(out_path, "wt", encoding="utf-8") as w:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building CDS cache (genbank/refseq)"):
            cds_path = find_cds_fasta_for_row(row, downloads_root)
            if not cds_path:
                continue

            host_name = row.get("host_name_clean", row.get("host_name", None))
            host_group = row.get(group_col, None)

            if allowed_host_groups is not None and host_group not in allowed_host_groups:
                continue

            base_meta = {
                "asm": row.get("asm", None),
                "taxid": row.get("taxid", None),
                "db_source": row.get("db_source", None),
                "ftp_path": row.get("ftp_path", None),
                "organism_name": row.get("organism_name", None),
                "family": row.get("family", None),
                "genus": row.get("genus", None),
                "species": row.get("species", None),
                "host_name": host_name,
                "host_group": host_group,
                "cds_fasta_path": cds_path,
            }

            try:
                for cds_nt, cds_meta in iter_cds_from_fasta(cds_path):
                    L = len(cds_nt)
                    if L < min_len or L > max_len:
                        continue
                    obj = {**base_meta, **cds_meta, "cds_nt": cds_nt, "cds_len": L}
                    w.write(json.dumps(obj, ensure_ascii=False) + "\n")
            except Exception:
                continue

    return out_path


class ViralCDSFastaDataset(Dataset):
    def __init__(
        self,
        tsv_path: str,
        downloads_root: str,   # <-- 注意：这里是根目录，包含 genbank/ 和 refseq/
        cache_dir: str,
        rebuild_cache: bool = False,
        prefer_fixed_group: bool = True,
        min_len: int = 60,
        max_len: int = 1_000_000,
        allowed_host_groups: Optional[List[str]] = None,
    ):
        self.tsv_path = tsv_path
        self.downloads_root = downloads_root
        self.cache_dir = cache_dir
        _safe_mkdir(self.cache_dir)

        cache_key = _sha1(os.path.abspath(tsv_path) + "||" + os.path.abspath(downloads_root))[:12]
        self.cache_path = os.path.join(cache_dir, f"cds_cache_fasta.{cache_key}.jsonl.gz")
        if rebuild_cache and os.path.exists(self.cache_path):
            os.remove(self.cache_path)

        self.cache_path = build_cds_cache_from_fasta(
            tsv_path=tsv_path,
            downloads_root=downloads_root,
            cache_dir=cache_dir,
            prefer_fixed_group=prefer_fixed_group,
            min_len=min_len,
            max_len=max_len,
            allowed_host_groups=allowed_host_groups,
        )

        items: List[Dict[str, Any]] = []
        with gzip.open(self.cache_path, "rt", encoding="utf-8") as r:
            for line in r:
                if line.strip():
                    items.append(json.loads(line))
        self.items = items

        groups = sorted({x.get("host_group", "UNK") for x in self.items})
        self.group2id = {g: i for i, g in enumerate(groups)}
        self.id2group = {i: g for g, i in self.group2id.items()}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x = self.items[idx]
        g = x.get("host_group", "UNK")
        y = self.group2id.get(g, -1)
        return {
            "cds_nt": x["cds_nt"],
            "label_id": y,
            "host_group": g,
            "host_name": x.get("host_name", None),
            "family": x.get("family", None),
            "organism_name": x.get("organism_name", None),
            "asm": x.get("asm", None),
            "taxid": x.get("taxid", None),
            "db_source": x.get("db_source", None),
            "cds_id": x.get("cds_id", None),
            "cds_desc": x.get("cds_desc", None),
        }


def simple_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    cds = [b["cds_nt"] for b in batch]
    labels = torch.tensor([b["label_id"] for b in batch], dtype=torch.long)
    meta = [{k: v for k, v in b.items() if k not in ("cds_nt", "label_id")} for b in batch]
    return {"cds_nt": cds, "labels": labels, "meta": meta}

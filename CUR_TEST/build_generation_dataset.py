#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build generation-evaluation dataset from a TSV that contains a 'sequence' column.

Input TSV (your new data):
/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/ncbi_viral/merged/rep.family_top100.with_header.length_leq131k.n_leq5pct.with_sequence.tsv

Output:
- JSONL file: each line is one sample with fields:
  {sample_id, family, asm, taxid, organism_name, source, db_source, ftp_path, seq_len,
   start_bp, prefix, target, prefix_len_bp, target_len_bp, n_ratio_prefix, n_ratio_target, gc_prefix, gc_target}

This script is streaming & memory-friendly.
"""

import argparse
import csv
import json
import os
import random
import sys
from typing import Dict, Iterable, Tuple

DNA_ALLOWED = set("ACGTN")  # you said N ratio already processed, still keep guard.


def calc_gc(seq: str) -> float:
    if not seq:
        return 0.0
    s = seq.upper()
    gc = sum(1 for c in s if c in ("G", "C"))
    atgc = sum(1 for c in s if c in ("A", "C", "G", "T"))
    return (gc / atgc) if atgc > 0 else 0.0


def calc_n_ratio(seq: str) -> float:
    if not seq:
        return 0.0
    s = seq.upper()
    n = s.count("N")
    return n / len(s)


def normalize_seq(seq: str, strict: bool = False) -> str:
    """
    Normalize sequence: uppercase, strip spaces, remove non-ACGTN if strict=False (replace with N).
    """
    if seq is None:
        return ""
    s = "".join(seq.strip().split()).upper()  # remove whitespace/newlines
    if not s:
        return s

    if strict:
        # If any illegal char -> empty (skip)
        if any(c not in DNA_ALLOWED for c in s):
            return ""
        return s

    # soft mode: replace illegal with N
    out = []
    for c in s:
        out.append(c if c in DNA_ALLOWED else "N")
    return "".join(out)


def iter_tsv_rows(path: str) -> Iterable[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise RuntimeError("TSV has no header. Please ensure it includes column names.")
        for row in reader:
            yield row


def choose_starts(seq_len: int, prefix_len: int, target_len: int, n_windows: int, rng: random.Random) -> Iterable[int]:
    """
    Choose window start positions. Ensures start+prefix+target <= seq_len.
    """
    total = prefix_len + target_len
    if seq_len < total:
        return []
    max_start = seq_len - total
    if max_start == 0:
        return [0]

    # Sample distinct starts when possible
    if n_windows <= 1:
        return [rng.randint(0, max_start)]
    starts = set()
    trials = 0
    while len(starts) < n_windows and trials < n_windows * 20:
        starts.add(rng.randint(0, max_start))
        trials += 1
    if not starts:
        starts.add(0)
    return sorted(starts)


def build_samples_from_row(
    row: Dict[str, str],
    prefix_len: int,
    target_len: int,
    n_windows_per_seq: int,
    strict_seq: bool,
    rng: random.Random,
) -> Iterable[Dict]:
    seq_raw = row.get("sequence", "")
    seq = normalize_seq(seq_raw, strict=strict_seq)
    if not seq:
        return

    seq_len = len(seq)
    total = prefix_len + target_len
    if seq_len < total:
        return

    starts = choose_starts(seq_len, prefix_len, target_len, n_windows_per_seq, rng)
    for start in starts:
        prefix = seq[start : start + prefix_len]
        target = seq[start + prefix_len : start + prefix_len + target_len]
        if not prefix or not target:
            continue

        yield {
            "taxid": row.get("taxid", ""),
            "ftp_path": row.get("ftp_path", ""),
            "asm": row.get("asm", ""),
            "organism_name": row.get("organism_name", ""),
            "source": row.get("source", ""),
            "family": row.get("family", ""),
            "genus": row.get("genus", ""),
            "species": row.get("species", ""),
            "db_source": row.get("db_source", ""),
            "seq_len": seq_len,
            "start_bp": start,
            "prefix": prefix,
            "target": target,
            "prefix_len_bp": len(prefix),
            "target_len_bp": len(target),
            "n_ratio_prefix": calc_n_ratio(prefix),
            "n_ratio_target": calc_n_ratio(target),
            "gc_prefix": calc_gc(prefix),
            "gc_target": calc_gc(target),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", type=str, required=True, help="Input TSV with a 'sequence' column")
    ap.add_argument("--out", type=str, required=True, help="Output JSONL path")
    ap.add_argument("--prefix_len", type=int, default=1024, help="Prefix length (bp)")
    ap.add_argument("--target_len", type=int, default=512, help="Target length (bp)")
    ap.add_argument("--windows_per_seq", type=int, default=2, help="How many windows to sample per sequence")
    ap.add_argument("--max_per_family", type=int, default=200, help="Cap samples per family (approx)")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--strict_seq", action="store_true", help="If set, skip any sequence containing illegal chars")
    ap.add_argument("--progress_every", type=int, default=2000, help="Print progress every N rows")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rng = random.Random(args.seed)
    family_counts: Dict[str, int] = {}
    n_rows = 0
    n_written = 0
    n_skipped_short = 0
    n_skipped_invalid = 0

    with open(args.out, "w", encoding="utf-8") as w:
        for row in iter_tsv_rows(args.tsv):
            n_rows += 1
            fam = row.get("family", "") or "UNKNOWN"

            # soft cap per family
            if family_counts.get(fam, 0) >= args.max_per_family:
                continue

            seq_raw = row.get("sequence", "")
            seq_norm = normalize_seq(seq_raw, strict=args.strict_seq)
            if not seq_norm:
                n_skipped_invalid += 1
                continue

            if len(seq_norm) < (args.prefix_len + args.target_len):
                n_skipped_short += 1
                continue

            # rebuild from normalized seq to ensure window extraction is consistent
            row_local = dict(row)
            row_local["sequence"] = seq_norm

            any_written = False
            for sample in build_samples_from_row(
                row_local,
                prefix_len=args.prefix_len,
                target_len=args.target_len,
                n_windows_per_seq=args.windows_per_seq,
                strict_seq=args.strict_seq,
                rng=rng,
            ):
                # final family cap check again
                if family_counts.get(fam, 0) >= args.max_per_family:
                    break

                sample_id = f"{fam}|{row.get('asm','')}|{row.get('taxid','')}|{sample['start_bp']}|{args.prefix_len}+{args.target_len}"
                sample["sample_id"] = sample_id
                w.write(json.dumps(sample, ensure_ascii=False) + "\n")
                n_written += 1
                family_counts[fam] = family_counts.get(fam, 0) + 1
                any_written = True

            if args.progress_every > 0 and n_rows % args.progress_every == 0:
                sys.stderr.write(
                    f"[build] rows={n_rows} written={n_written} "
                    f"skipped_short={n_skipped_short} skipped_invalid={n_skipped_invalid} "
                    f"families={len(family_counts)}\n"
                )

    sys.stderr.write(
        f"[done] rows={n_rows} written={n_written} skipped_short={n_skipped_short} "
        f"skipped_invalid={n_skipped_invalid} families={len(family_counts)}\n"
    )


if __name__ == "__main__":
    main()

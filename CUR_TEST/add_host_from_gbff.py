#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import gzip
import os
import re
from collections import Counter
from glob import glob
from typing import Dict, List, Tuple, Optional


# ---- qualifier regex (handles /key="value...") ----
QUAL_START_RE = re.compile(r'/(host|lab_host|isolation_source)="(.*)$')


def _read_multiline_quoted_value(first_tail: str, it) -> str:
    """
    GBFF qualifier value may span multiple lines until the closing quote.
    first_tail: text after the first quote (may already contain closing quote)
    it: iterator over remaining lines
    """
    # If closes on same line
    if first_tail.endswith('"'):
        return first_tail[:-1]

    chunks = [first_tail]
    for line in it:
        s = line.rstrip("\n")
        # GBFF continuation lines are indented; we just take the raw tail
        # Find closing quote
        if '"' in s:
            before, _after = s.split('"', 1)
            chunks.append(before.strip())
            break
        chunks.append(s.strip())
    # Join with space to avoid concatenating words
    return " ".join([c for c in chunks if c])


def parse_gbff_hosts(gbff_gz_path: str) -> Tuple[str, str, str]:
    """
    Parse /host, /lab_host, /isolation_source from a GBFF (.gz) file.
    Return (host, lab_host, isolation_source) as ';' joined unique strings.
    """
    hosts = []
    lab_hosts = []
    isol = []

    try:
        with gzip.open(gbff_gz_path, "rt", encoding="utf-8", errors="replace") as f:
            it = iter(f)
            for line in it:
                # qualifiers usually appear in FEATURES section; we can just scan all lines
                m = QUAL_START_RE.search(line)
                if not m:
                    continue
                key = m.group(1)
                tail = m.group(2).rstrip("\n")

                val = _read_multiline_quoted_value(tail, it)
                val = val.strip()

                if not val:
                    continue
                if key == "host":
                    hosts.append(val)
                elif key == "lab_host":
                    lab_hosts.append(val)
                elif key == "isolation_source":
                    isol.append(val)
    except Exception:
        # If file is corrupted/unreadable, just return empty
        return "", "", ""

    def uniq_join(vals: List[str]) -> str:
        seen = []
        sset = set()
        for v in vals:
            if v not in sset:
                sset.add(v)
                seen.append(v)
        return ";".join(seen)

    return uniq_join(hosts), uniq_join(lab_hosts), uniq_join(isol)


def guess_gbff_path(
    base_dir: str,
    db_source: str,
    taxid: str,
    asm: str,
) -> Optional[str]:
    """
    Prefer the canonical download layout first:
      {base_dir}/{db_source}/downloads/{taxid}/{asm}/{asm}_genomic.gbff.gz
    Otherwise fall back to glob search under those folders.
    """
    # canonical
    cand = os.path.join(base_dir, db_source, "downloads", taxid, asm, f"{asm}_genomic.gbff.gz")
    if os.path.exists(cand):
        return cand

    # common alternative naming patterns
    patterns = [
        os.path.join(base_dir, db_source, "downloads", taxid, asm, "*genomic.gbff.gz"),
        os.path.join(base_dir, db_source, "downloads", taxid, asm, "*.gbff.gz"),
        os.path.join(base_dir, db_source, "downloads", taxid, "**", f"{asm}*genomic.gbff.gz"),
        os.path.join(base_dir, db_source, "downloads", taxid, "**", "*.gbff.gz"),
    ]
    for pat in patterns:
        hits = glob(pat, recursive=True)
        # try prefer exact asm prefix if possible
        hits = sorted(hits, key=lambda x: (0 if os.path.basename(x).startswith(asm) else 1, len(x)))
        for h in hits:
            if asm in os.path.basename(h):
                return h
        if hits:
            return hits[0]

    # last resort: global search (can be slow if huge)
    pat = os.path.join(base_dir, "**", f"{asm}*genomic.gbff.gz")
    hits = glob(pat, recursive=True)
    if hits:
        return sorted(hits, key=len)[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_tsv", required=True)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument(
        "--base_dir",
        default="/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/ncbi_viral",
        help="ncbi_viral root dir which contains refseq/ and genbank/ downloads/",
    )
    ap.add_argument("--make_stats", action="store_true", help="Also write host stats tsv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_tsv) or ".", exist_ok=True)

    host_counter = Counter()

    with open(args.in_tsv, "r", encoding="utf-8", newline="") as r, \
         open(args.out_tsv, "w", encoding="utf-8", newline="") as w:

        reader = csv.DictReader(r, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])

        # Required columns
        for req in ("taxid", "asm", "db_source"):
            if req not in fieldnames:
                raise RuntimeError(f"Input TSV missing required column: {req}")

        out_fields = fieldnames + ["host_name", "lab_host_name", "isolation_source", "gbff_path"]
        writer = csv.DictWriter(w, delimiter="\t", fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()

        n = 0
        n_found = 0
        for row in reader:
            n += 1
            taxid = str(row.get("taxid", "")).strip()
            asm = str(row.get("asm", "")).strip()
            db_source = str(row.get("db_source", "")).strip().lower()

            # normalize db_source folder name
            if db_source not in ("refseq", "genbank"):
                # fall back to "source" column if needed
                src = str(row.get("source", "")).strip().lower()
                if src in ("refseq", "genbank"):
                    db_source = src

            gbff = None
            if taxid and asm and db_source in ("refseq", "genbank"):
                gbff = guess_gbff_path(args.base_dir, db_source, taxid, asm)

            host_name = lab_host = iso_src = ""
            if gbff and os.path.exists(gbff):
                n_found += 1
                host_name, lab_host, iso_src = parse_gbff_hosts(gbff)

                # stats
                if host_name:
                    for h in host_name.split(";"):
                        hh = h.strip()
                        if hh:
                            host_counter[hh] += 1

            row["host_name"] = host_name
            row["lab_host_name"] = lab_host
            row["isolation_source"] = iso_src
            row["gbff_path"] = gbff or ""

            writer.writerow(row)

            if n % 200 == 0:
                print(f"[host] processed={n}, gbff_found={n_found}")

    if args.make_stats:
        stats_path = args.out_tsv + ".host_stats.tsv"
        with open(stats_path, "w", encoding="utf-8", newline="") as f:
            f.write("host_name\tcount\n")
            for host, cnt in host_counter.most_common():
                f.write(f"{host}\t{cnt}\n")
        print(f"[host] wrote stats: {stats_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import gzip
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

# 你自己的 Evo2 类：确保能 import 到
from myevo2 import Evo2


DNA_RE = re.compile(r"^[ACGTN]+$")

def read_first_fasta_sequence_gz(path: str, max_bp: Optional[int] = None) -> str:
    """
    Read the first sequence from a .fna.gz fasta file.
    If max_bp is set, stop after reading that many bases (saves time for huge genomes).
    """
    seq_lines = []
    total = 0
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if seq_lines:
                    break
                continue
            s = line.strip().upper()
            if not s:
                continue
            seq_lines.append(s)
            total += len(s)
            if max_bp is not None and total >= max_bp:
                break
    return "".join(seq_lines).upper()


def build_genome_path(outdir: str, taxid: str, asm: str, db_source: str) -> str:
    base = "refseq" if db_source == "refseq" else "genbank"
    return os.path.join(outdir, base, "downloads", taxid, asm, f"{asm}_genomic.fna.gz")


def clean_seq_basic(seq: str) -> str:
    # 将 U 视为 T，其它 IUPAC 统一变 N
    seq = seq.upper().replace("U", "T")
    out = []
    for ch in seq:
        if ch in ("A", "C", "G", "T", "N"):
            out.append(ch)
        else:
            out.append("N")
    return "".join(out)


def n_fraction(seq: str) -> float:
    if not seq:
        return 1.0
    return seq.count("N") / len(seq)


def gc_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    s = seq.replace("N", "")
    if not s:
        return 0.0
    return (s.count("G") + s.count("C")) / len(s)


def max_run_fraction(seq: str) -> float:
    """最大同字符 run / len，用于粗略低复杂度过滤"""
    if not seq:
        return 1.0
    best = 1
    cur = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best / len(seq)


def kmer_set(seq: str, k: int = 6) -> set:
    s = seq.replace("N", "")
    if len(s) < k:
        return set()
    return {s[i:i+k] for i in range(len(s)-k+1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return inter / uni if uni else 0.0


def hamming_accuracy(a: str, b: str) -> float:
    """按位相等比例；长度不等时按 min_len 对齐计算"""
    if not a or not b:
        return 0.0
    L = min(len(a), len(b))
    if L == 0:
        return 0.0
    same = sum(1 for i in range(L) if a[i] == b[i])
    return same / L


@dataclass
class GenConfig:
    outdir: str
    tsv: str
    model_name: str = "evo2_1b_base"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 清洗阈值
    min_len: int = 2048
    max_len: int = 131_000
    max_read_bp: int = 200_000          # 读取上限，防止极大序列拖慢 IO（>= max_len 已会过滤）
    max_n_frac: float = 0.10            # N 比例阈值
    max_run_frac: float = 0.30          # 最大同字符 run 比例阈值（低复杂度）

    # 生成参数
    temperature: float = 1.0
    top_k: int = 4
    top_p: float = 1.0

    # 批处理
    batch_prompts: int = 4              # 一次 generate 的 prompt 数（根据显存调）
    resume: bool = True                 # 若输出存在则跳过已生成 asm
    limit: int = 0                      # 0 表示全量
    seed: int = 42

    # 输出
    out_tsv: str = ""


def load_input_tsv(tsv: str) -> pd.DataFrame:
    df = pd.read_csv(tsv, sep="\t")
    # 基本列检查
    need = ["taxid", "asm", "family", "db_source"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {tsv}. Got columns={list(df.columns)}")
    # genus 可缺失
    if "genus" not in df.columns:
        df["genus"] = "NA"
    if "species" not in df.columns:
        df["species"] = "NA"
    if "organism_name" not in df.columns:
        df["organism_name"] = "NA"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="NCBI viral OUTDIR (contains refseq/genbank downloads)")
    ap.add_argument("--tsv", required=True, help="Input TSV with header (rep.family_top100.with_header.tsv)")
    ap.add_argument("--model", default="evo2_1b_base", help="Evo2 model name")
    ap.add_argument("--device", default=None, help="cuda:0 / cuda:1 / cpu")
    ap.add_argument("--min-len", type=int, default=2048)
    ap.add_argument("--max-len", type=int, default=131000)
    ap.add_argument("--max-n-frac", type=float, default=0.10)
    ap.add_argument("--max-run-frac", type=float, default=0.30)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--batch-prompts", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--out", default="", help="Output TSV path")
    args = ap.parse_args()

    cfg = GenConfig(
        outdir=args.outdir,
        tsv=args.tsv,
        model_name=args.model,
        device=args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"),
        min_len=args.min_len,
        max_len=args.max_len,
        max_n_frac=args.max_n_frac,
        max_run_frac=args.max_run_frac,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        batch_prompts=args.batch_prompts,
        resume=(not args.no_resume),
        limit=args.limit,
        out_tsv=args.out if args.out else os.path.join(os.path.dirname(args.tsv), f"gen_eval_halfprompt.{args.model}.tsv"),
    )

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    df = load_input_tsv(cfg.tsv)
    if cfg.limit and cfg.limit > 0:
        df = df.sample(n=min(cfg.limit, len(df)), random_state=cfg.seed).reset_index(drop=True)

    # resume: 记录已完成的 asm
    done_asm = set()
    if cfg.resume and os.path.exists(cfg.out_tsv):
        try:
            prev = pd.read_csv(cfg.out_tsv, sep="\t")
            if "asm" in prev.columns:
                done_asm = set(prev["asm"].astype(str).tolist())
        except Exception:
            done_asm = set()

    # 加载模型
    print(f"[INFO] loading Evo2 model={cfg.model_name} device={cfg.device}")
    evo2 = Evo2(cfg.model_name)
    # 注意：Evo2 内部可能已做 device placement；这里不强制 .to()

    rows_out = []
    out_cols = [
        "taxid","asm","family","genus","species","db_source","organism_name",
        "genome_len","half_len",
        "prompt","truth_suffix","gen_suffix",
        "acc_hamming","exact_match",
        "n_frac_full","n_frac_gen",
        "gc_full","gc_gen",
        "kmer6_jaccard",
        "score_truth_suffix","score_gen_suffix",
        "status","reason","fna_path"
    ]

    def flush():
        if not rows_out:
            return
        out_df = pd.DataFrame(rows_out, columns=out_cols)
        header = not os.path.exists(cfg.out_tsv) or os.path.getsize(cfg.out_tsv) == 0
        out_df.to_csv(cfg.out_tsv, sep="\t", index=False, mode="a", header=header)
        rows_out.clear()

    # 批处理生成：先准备 prompts
    pending = []  # list of dict with metadata + prompt + truth suffix + half_len
    for _, r in tqdm(df.iterrows(), total=len(df), desc="prepare"):
        taxid = str(r["taxid"])
        asm = str(r["asm"])
        db = str(r["db_source"])
        fam = str(r["family"])
        genus = str(r.get("genus","NA"))
        species = str(r.get("species","NA"))
        org = str(r.get("organism_name","NA"))

        if asm in done_asm:
            continue

        fna = build_genome_path(cfg.outdir, taxid, asm, db)
        if not os.path.exists(fna):
            rows_out.append([taxid,asm,fam,genus,species,db,org,
                             0,0,"","","",0.0,0,1.0,1.0,0.0,0.0,0.0,
                             np.nan,np.nan,"drop","missing_file",fna])
            if len(rows_out) >= 200:
                flush()
            continue

        try:
            seq = read_first_fasta_sequence_gz(fna, max_bp=cfg.max_read_bp)
        except Exception as e:
            rows_out.append([taxid,asm,fam,genus,species,db,org,
                             0,0,"","","",0.0,0,1.0,1.0,0.0,0.0,0.0,
                             np.nan,np.nan,"drop",f"read_error:{type(e).__name__}",fna])
            if len(rows_out) >= 200:
                flush()
            continue

        seq = clean_seq_basic(seq)
        L = len(seq)
        if L < cfg.min_len:
            rows_out.append([taxid,asm,fam,genus,species,db,org,
                             L,0,"","","",0.0,0,n_fraction(seq),1.0,gc_fraction(seq),0.0,0.0,
                             np.nan,np.nan,"drop","too_short",fna])
            if len(rows_out) >= 200:
                flush()
            continue
        if L > cfg.max_len:
            rows_out.append([taxid,asm,fam,genus,species,db,org,
                             L,0,"","","",0.0,0,n_fraction(seq),1.0,gc_fraction(seq),0.0,0.0,
                             np.nan,np.nan,"drop","too_long",fna])
            if len(rows_out) >= 200:
                flush()
            continue

        nf = n_fraction(seq)
        if nf > cfg.max_n_frac:
            rows_out.append([taxid,asm,fam,genus,species,db,org,
                             L,0,"","","",0.0,0,nf,1.0,gc_fraction(seq),0.0,0.0,
                             np.nan,np.nan,"drop","high_N",fna])
            if len(rows_out) >= 200:
                flush()
            continue

        mr = max_run_fraction(seq)
        if mr > cfg.max_run_frac:
            rows_out.append([taxid,asm,fam,genus,species,db,org,
                             L,0,"","","",0.0,0,nf,1.0,gc_fraction(seq),0.0,0.0,
                             np.nan,np.nan,"drop","low_complexity",fna])
            if len(rows_out) >= 200:
                flush()
            continue

        half = L // 2
        prompt = seq[:half]
        truth = seq[half:half+half]  # 生成目标：和 prompt 等长（如果 L 为奇数，忽略最后 1bp）
        if len(truth) < 16:
            rows_out.append([taxid,asm,fam,genus,species,db,org,
                             L,half,prompt,truth,"",0.0,0,nf,1.0,gc_fraction(seq),0.0,0.0,
                             np.nan,np.nan,"drop","suffix_too_short",fna])
            if len(rows_out) >= 200:
                flush()
            continue

        pending.append({
            "taxid": taxid, "asm": asm, "family": fam, "genus": genus, "species": species,
            "db_source": db, "organism_name": org, "fna": fna,
            "genome_len": L, "half": half,
            "prompt": prompt, "truth": truth,
            "n_frac_full": nf, "gc_full": gc_fraction(seq),
            "seq_full": seq,
        })

        # 满一批就生成
        if len(pending) >= cfg.batch_prompts:
            prompts = [p["prompt"] for p in pending]
            n_tokens = pending[0]["half"]  # 同一批最好 half 相近；这里用第一条 half
            # 如果 half 不一致，简单做：按各自 half 生成（逐条）；更稳但慢
            same_half = all(p["half"] == n_tokens for p in pending)
            if not same_half:
                # 逐条生成（更真实）
                for p in pending:
                    _do_one(evo2, cfg, p, rows_out)
                pending.clear()
                flush()
            else:
                # 批量生成
                try:
                    gen_seqs, _ = evo2.generate(
                        prompt_seqs=prompts,
                        n_tokens=n_tokens,
                        temperature=cfg.temperature,
                        top_k=cfg.top_k,
                        top_p=cfg.top_p,
                        batched=True,
                        cached_generation=True,
                        verbose=0,
                    )
                except Exception as e:
                    # 批生成失败就降级逐条
                    for p in pending:
                        rows_out.append([
                            p["taxid"],p["asm"],p["family"],p["genus"],p["species"],p["db_source"],p["organism_name"],
                            p["genome_len"],p["half"],p["prompt"],p["truth"],"",
                            0.0,0,p["n_frac_full"],1.0,p["gc_full"],0.0,0.0,
                            np.nan,np.nan,"error",f"batch_generate_error:{type(e).__name__}",p["fna"]
                        ])
                    pending.clear()
                    flush()
                    continue

                for p, g in zip(pending, gen_seqs):
                    gen_suffix = str(g)[len(p["prompt"]):]  # 取 prompt 后面的新增部分
                    _score_and_append(evo2, p, gen_suffix, rows_out)
                pending.clear()
                flush()

    # 处理剩余 pending
    if pending:
        for p in pending:
            _do_one(evo2, cfg, p, rows_out)
        pending.clear()
        flush()

    flush()
    print(f"[INFO] wrote: {cfg.out_tsv}")


def _do_one(evo2: Evo2, cfg: GenConfig, p: Dict, rows_out: List):
    """逐条生成（half 不一致或更稳健时用）"""
    try:
        gen_seqs, _ = evo2.generate(
            prompt_seqs=[p["prompt"]],
            n_tokens=p["half"],
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            batched=True,
            cached_generation=True,
            verbose=0,
        )
        g = gen_seqs[0]
        gen_suffix = str(g)[len(p["prompt"]):]
        _score_and_append(evo2, p, gen_suffix, rows_out)
    except Exception as e:
        rows_out.append([
            p["taxid"],p["asm"],p["family"],p["genus"],p["species"],p["db_source"],p["organism_name"],
            p["genome_len"],p["half"],p["prompt"],p["truth"],"",
            0.0,0,p["n_frac_full"],1.0,p["gc_full"],0.0,0.0,
            np.nan,np.nan,"error",f"generate_error:{type(e).__name__}",p["fna"]
        ])


def _score_and_append(evo2: Evo2, p: Dict, gen_suffix: str, rows_out: List):
    # 对齐长度
    truth = p["truth"]
    half = p["half"]
    gen_suffix = clean_seq_basic(gen_suffix)[:half]

    acc = hamming_accuracy(truth, gen_suffix)
    exact = int((len(truth) == len(gen_suffix)) and (truth == gen_suffix))

    nf_gen = n_fraction(gen_suffix)
    gc_gen = gc_fraction(gen_suffix)

    # k-mer Jaccard（6-mer）
    j6 = jaccard(kmer_set(truth, 6), kmer_set(gen_suffix, 6))

    # Evo2 scoring：对“后半段本身”打分（可选，也可对整条生成序列打分）
    try:
        score_truth = float(evo2.score_sequences([truth])[0])
    except Exception:
        score_truth = float("nan")
    try:
        score_gen = float(evo2.score_sequences([gen_suffix])[0])
    except Exception:
        score_gen = float("nan")

    rows_out.append([
        p["taxid"],p["asm"],p["family"],p["genus"],p["species"],p["db_source"],p["organism_name"],
        p["genome_len"],half,
        p["prompt"],truth,gen_suffix,
        acc,exact,
        p["n_frac_full"],nf_gen,
        p["gc_full"],gc_gen,
        j6,
        score_truth,score_gen,
        "ok","",p["fna"]
    ])
"""
export OUTDIR=/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/ncbi_viral
export TSV=$OUTDIR/merged/rep.family_top100.with_header.tsv

python gen_eval_evo2_halfprompt.py \
  --outdir "$OUTDIR" \
  --tsv "$TSV" \
  --model evo2_1b_base \
  --device cuda:1 \
  --min-len 0 \
  --max-len 131000 \
  --max-n-frac 0.10 \
  --batch-prompts 2 \
  --out "$OUTDIR/merged/gen_eval_halfprompt.evo2_1b_base.tsv"
"""

if __name__ == "__main__":
    main()

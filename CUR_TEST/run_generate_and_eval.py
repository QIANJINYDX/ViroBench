#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import contextlib
import io
import json
import math
import os
import sys
import traceback
import subprocess
import tempfile
import shutil
from typing import Dict, Any, Tuple, List, Optional

import torch

from myevo2 import Evo2  # 你已改好


DNA_ALLOWED = set("ACGTN")


def normalize_seq(seq: str) -> str:
    if seq is None:
        return ""
    s = "".join(str(seq).strip().split()).upper()
    out = []
    for c in s:
        out.append(c if c in DNA_ALLOWED else "N")
    return "".join(out)


def calc_gc(seq: str) -> float:
    if not seq:
        return 0.0
    s = seq.upper()
    gc = sum(1 for c in s if c in ("G", "C"))
    atgc = sum(1 for c in s if c in ("A", "C", "G", "T"))
    return (gc / atgc) if atgc > 0 else 0.0


def kmer_freq(seq: str, k: int = 3) -> Dict[str, float]:
    seq = normalize_seq(seq)
    if len(seq) < k:
        return {}
    counts: Dict[str, int] = {}
    total = 0
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if "N" in kmer:
            continue
        counts[kmer] = counts.get(kmer, 0) + 1
        total += 1
    if total == 0:
        return {}
    return {kk: vv / total for kk, vv in counts.items()}


def js_divergence(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-12) -> float:
    keys = set(p.keys()) | set(q.keys())
    if not keys:
        return 0.0

    def kl(a: Dict[str, float], b: Dict[str, float]) -> float:
        s = 0.0
        for k in keys:
            av = a.get(k, 0.0) + eps
            bv = b.get(k, 0.0) + eps
            s += av * math.log(av / bv)
        return s

    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def simple_similarity(a: str, b: str) -> float:
    a = normalize_seq(a)
    b = normalize_seq(b)
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    m = sum(1 for i in range(n) if a[i] == b[i])
    return m / n


def parse_evo2_output(output: Any) -> Tuple[str, float]:
    if output is None:
        return "", float("nan")

    # 处理 GenerationOutput 对象（有 sequences 和 logprobs_mean 属性）
    if hasattr(output, 'sequences') and hasattr(output, 'logprobs_mean'):
        gen_seqs = output.sequences
        scores = output.logprobs_mean
        gen = gen_seqs[0] if isinstance(gen_seqs, (list, tuple)) and gen_seqs else str(gen_seqs)
        sc = scores[0] if isinstance(scores, (list, tuple)) and scores else float("nan")
        try:
            sc = float(sc)
        except Exception:
            sc = float("nan")
        return gen, sc

    # 处理元组格式 (gen_seqs, scores)
    if isinstance(output, tuple) and len(output) == 2:
        gen_seqs, scores = output
        gen = gen_seqs[0] if isinstance(gen_seqs, (list, tuple)) and gen_seqs else str(gen_seqs)
        sc = scores[0] if isinstance(scores, (list, tuple)) and scores else float("nan")
        try:
            sc = float(sc)
        except Exception:
            sc = float("nan")
        return gen, sc

    # 兜底：如果是列表，取第一个元素
    if isinstance(output, (list, tuple)) and output and isinstance(output[0], str):
        return output[0], float("nan")

    return str(output), float("nan")


def parse_evo2_output_batch(output: Any) -> List[Tuple[str, float]]:
    if output is None:
        return []

    # 处理 GenerationOutput 对象（有 sequences 和 logprobs_mean 属性）
    if hasattr(output, 'sequences') and hasattr(output, 'logprobs_mean'):
        gen_seqs = output.sequences
        scores = output.logprobs_mean
        if not isinstance(gen_seqs, (list, tuple)):
            gen_seqs = [str(gen_seqs)]
        if not isinstance(scores, (list, tuple)):
            scores = [float("nan")]

        results = []
        for i in range(len(gen_seqs)):
            gen = gen_seqs[i] if i < len(gen_seqs) else ""
            sc = scores[i] if i < len(scores) else float("nan")
            try:
                sc = float(sc)
            except Exception:
                sc = float("nan")
            results.append((gen, sc))
        return results

    # 处理元组格式 (gen_seqs, scores)
    if isinstance(output, tuple) and len(output) == 2:
        gen_seqs, scores = output
        if not isinstance(gen_seqs, (list, tuple)):
            gen_seqs = [str(gen_seqs)]
        if not isinstance(scores, (list, tuple)):
            scores = [float("nan")]

        results = []
        for i in range(len(gen_seqs)):
            gen = gen_seqs[i] if i < len(gen_seqs) else ""
            sc = scores[i] if i < len(scores) else float("nan")
            try:
                sc = float(sc)
            except Exception:
                sc = float("nan")
            results.append((gen, sc))
        return results

    # 兜底：如果是列表，每个元素作为一个序列
    if isinstance(output, (list, tuple)) and output:
        return [(str(item), float("nan")) for item in output]

    return []


@contextlib.contextmanager
def suppress_output(enabled: bool):
    if not enabled:
        yield
        return
    stdout = sys.stdout
    stderr = sys.stderr
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stdout = stdout
        sys.stderr = stderr


# -------------------- BLAST evaluation --------------------

def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def run_blastn_pair(
    query_seq: str,
    subject_seq: str,
    blastn_path: str = "blastn",
    task: str = "blastn",
    word_size: int = 11,
    evalue: float = 1e-5,
    max_target_seqs: int = 1,
    timeout_sec: int = 30,
) -> Tuple[str, str, str, str]:
    """
    Run blastn query vs subject (no DB needed) and return:
      (pident, aln_len, evalue, bitscore) as strings ('' if no hit)
    Uses outfmt 6 for easy parsing.
    """
    query_seq = normalize_seq(query_seq)
    subject_seq = normalize_seq(subject_seq)

    # 太短/空就不跑
    if len(query_seq) < 20 or len(subject_seq) < 20:
        return "", "", "", ""

    # 如果 blastn 不存在，直接返回空（你也可以改成 raise）
    if _which(blastn_path) is None:
        return "", "", "", ""

    outfmt = "6 pident length evalue bitscore"
    with tempfile.TemporaryDirectory() as td:
        q_fa = os.path.join(td, "q.fa")
        s_fa = os.path.join(td, "s.fa")
        with open(q_fa, "w", encoding="utf-8") as f:
            f.write(">q\n")
            f.write(query_seq + "\n")
        with open(s_fa, "w", encoding="utf-8") as f:
            f.write(">s\n")
            f.write(subject_seq + "\n")

        cmd = [
            blastn_path,
            "-query", q_fa,
            "-subject", s_fa,
            "-task", task,
            "-word_size", str(int(word_size)),
            "-evalue", str(float(evalue)),
            "-max_target_seqs", str(int(max_target_seqs)),
            "-outfmt", outfmt,
        ]

        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
        )

        if p.returncode != 0:
            # 这里不把 stderr 写进主 error_detail，避免污染；需要的话你可以扩展列
            return "", "", "", ""

        lines = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
        if not lines:
            return "", "", "", ""
        # 取最优 hit 的第一行
        cols = lines[0].split()
        if len(cols) < 4:
            return "", "", "", ""
        pident, aln_len, ev, bits = cols[0], cols[1], cols[2], cols[3]
        return pident, aln_len, ev, bits


# -------------------- Evo2 generation --------------------

def evo2_generate_one(
    evo2: Evo2,
    prefix: str,
    n_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    cached_generation: bool,
    force_prompt_threshold: int,
    suppress_vortex_warnings: bool,
    max_retries: int = 3,
) -> Tuple[str, str, str, int]:
    prefix = normalize_seq(prefix)
    retries = 0
    cur_n_tokens = int(n_tokens)
    cur_force = force_prompt_threshold
    cur_cached = bool(cached_generation)

    while True:
        try:
            with torch.no_grad():
                with suppress_output(suppress_vortex_warnings):
                    out = evo2.generate(
                        prompt_seqs=[prefix],
                        n_tokens=cur_n_tokens,
                        temperature=float(temperature),
                        top_k=int(top_k),
                        top_p=float(top_p),
                        batched=False,
                        cached_generation=cur_cached,
                        verbose=0,
                        force_prompt_threshold=cur_force,
                    )
            gen, _score = parse_evo2_output(out)
            return normalize_seq(gen), "", "", retries

        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            err_type = "OutOfMemoryError"
            err_detail = f"OOM at n_tokens={cur_n_tokens}, cached_generation={cur_cached}, force_prompt_threshold={cur_force}"

            if retries >= max_retries:
                return "", err_type, err_detail, retries

            retries += 1
            if cur_n_tokens > 64:
                cur_n_tokens = max(64, cur_n_tokens // 2)
                continue
            if cur_force is None:
                cur_force = 2048
                continue
            if cur_cached:
                cur_cached = False
                continue
            continue

        except Exception:
            etype = type(sys.exc_info()[1]).__name__
            tb = traceback.format_exc()
            return "", etype, tb, retries


def evo2_generate_batch(
    evo2: Evo2,
    prefixes: List[str],
    n_tokens_list: List[int],
    temperature: float,
    top_k: int,
    top_p: float,
    cached_generation: bool,
    force_prompt_threshold: int,
    suppress_vortex_warnings: bool,
    max_retries: int = 3,
) -> List[Tuple[str, str, str, int]]:
    normalized_prefixes = [normalize_seq(p) for p in prefixes]
    if len(n_tokens_list) != len(normalized_prefixes):
        raise ValueError("n_tokens_list length must match prefixes length")

    retries = 0
    cur_n_tokens = max(n_tokens_list) if n_tokens_list else 500
    cur_force = force_prompt_threshold
    cur_cached = bool(cached_generation)

    results = [("", "", "", 0) for _ in normalized_prefixes]

    while True:
        try:
            with torch.no_grad():
                with suppress_output(suppress_vortex_warnings):
                    out = evo2.generate(
                        prompt_seqs=normalized_prefixes,
                        n_tokens=cur_n_tokens,
                        temperature=float(temperature),
                        top_k=int(top_k),
                        top_p=float(top_p),
                        batched=True,
                        cached_generation=cur_cached,
                        verbose=0,
                        force_prompt_threshold=cur_force,
                    )
            batch_results = parse_evo2_output_batch(out)

            for i in range(len(normalized_prefixes)):
                if i < len(batch_results):
                    gen, _score = batch_results[i]
                    results[i] = (normalize_seq(gen), "", "", retries)
                else:
                    results[i] = ("", "IndexError", f"Output missing for index {i}", retries)

            return results

        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            err_type = "OutOfMemoryError"
            err_detail = (
                f"OOM at n_tokens={cur_n_tokens}, cached_generation={cur_cached}, "
                f"force_prompt_threshold={cur_force}, batch_size={len(normalized_prefixes)}"
            )

            if retries >= max_retries:
                for i in range(len(results)):
                    if results[i][0] == "":
                        results[i] = ("", err_type, err_detail, retries)
                return results

            retries += 1
            if cur_n_tokens > 64:
                cur_n_tokens = max(64, cur_n_tokens // 2)
                continue
            if cur_force is None:
                cur_force = 2048
                continue
            if cur_cached:
                cur_cached = False
                continue
            continue

        except Exception:
            etype = type(sys.exc_info()[1]).__name__
            tb = traceback.format_exc()
            for i in range(len(results)):
                if results[i][0] == "":
                    results[i] = ("", etype, tb, retries)
            return results


# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, required=True)
    ap.add_argument("--out_tsv", type=str, required=True)

    ap.add_argument("--model_name", type=str, default="evo2_1b_base")
    ap.add_argument("--local_path", type=str, default=None)

    ap.add_argument("--n_tokens", type=int, default=0,
                    help="0 表示用样本自身 target_len_bp；否则强制用该值作为续写长度")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--cached_generation", action="store_true")
    ap.add_argument("--force_prompt_threshold", type=int, default=None)

    ap.add_argument("--suppress_vortex_warnings", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1)

    # ---- BLAST options ----
    ap.add_argument("--do_blast", action="store_true", help="Enable blastn evaluation between target and generated continuation")
    ap.add_argument("--blastn_path", type=str, default="blastn")
    ap.add_argument("--blast_task", type=str, default="blastn")
    ap.add_argument("--blast_word_size", type=int, default=11)
    ap.add_argument("--blast_evalue", type=float, default=1e-5)
    ap.add_argument("--blast_timeout", type=int, default=30)

    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_tsv) or ".", exist_ok=True)

    evo2 = Evo2(model_name=args.model_name, local_path=args.local_path, use_local=True)

    headers = [
        "sample_id","family","asm","taxid",
        "prefix_len_bp","target_len_bp",
        "gen_text","gen_len_bp",
        "error_type","error_detail","retries",
        "gc_prefix","gc_target","gc_gen","gc_diff_target_gen",
        "jsd_3mer_target_gen","similarity_target_gen",
        # BLAST columns
        "blast_pident","blast_aln_len","blast_evalue","blast_bitscore",
    ]

    n = 0
    batch_size = max(1, int(args.batch))

    with open(args.out_tsv, "w", encoding="utf-8", newline="") as w:
        w.write("\t".join(headers) + "\n")
        w.flush()

        with open(args.jsonl, "r", encoding="utf-8") as f:
            batch_samples = []

            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                n += 1
                if args.limit > 0 and n > args.limit:
                    break

                sample = {
                    "sample_id": obj.get("sample_id",""),
                    "family": obj.get("family",""),
                    "asm": obj.get("asm",""),
                    "taxid": obj.get("taxid",""),
                    "prefix": normalize_seq(obj.get("prefix","")),
                    "target": normalize_seq(obj.get("target","")),
                }
                sample["n_tokens"] = args.n_tokens if args.n_tokens > 0 else len(sample["target"])
                batch_samples.append(sample)

                if len(batch_samples) >= batch_size:
                    write_batch(batch_samples, evo2, args, w)
                    batch_samples = []
                    if n % 200 == 0:
                        sys.stderr.write(f"[evo2] processed {n}\n")

            if batch_samples:
                write_batch(batch_samples, evo2, args, w)


def write_batch(batch_samples: List[Dict[str, str]], evo2: Evo2, args, w):
    batch_prefixes = [s["prefix"] for s in batch_samples]
    batch_n_tokens = [int(s["n_tokens"]) for s in batch_samples]

    if len(batch_samples) == 1:
        gen_full, err_type, err_detail, retries = evo2_generate_one(
            evo2=evo2,
            prefix=batch_prefixes[0],
            n_tokens=batch_n_tokens[0],
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            cached_generation=args.cached_generation,
            force_prompt_threshold=args.force_prompt_threshold,
            suppress_vortex_warnings=args.suppress_vortex_warnings,
            max_retries=3,
        )
        batch_results = [(gen_full, err_type, err_detail, retries)]
    else:
        batch_results = evo2_generate_batch(
            evo2=evo2,
            prefixes=batch_prefixes,
            n_tokens_list=batch_n_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            cached_generation=args.cached_generation,
            force_prompt_threshold=args.force_prompt_threshold,
            suppress_vortex_warnings=args.suppress_vortex_warnings,
            max_retries=3,
        )

    for i, sample in enumerate(batch_samples):
        gen_full, err_type, err_detail, retries = batch_results[i] if i < len(batch_results) else ("", "IndexError", "Missing result", 0)

        prefix = sample["prefix"]
        target = sample["target"]

        # 只取续写部分（后半段）
        if gen_full and gen_full.startswith(prefix):
            continuation = gen_full[len(prefix):]
        else:
            continuation = gen_full or ""

        # 对齐 target 长度
        cont_clip = continuation[:len(target)] if target else continuation

        # 基础指标
        gc_prefix = calc_gc(prefix)
        gc_target = calc_gc(target)
        gc_gen = calc_gc(cont_clip)
        gc_diff = abs(gc_target - gc_gen)

        jsd3 = js_divergence(kmer_freq(target, 3), kmer_freq(cont_clip, 3))
        sim = simple_similarity(target, cont_clip)

        # BLAST 指标（可选）
        blast_pident = blast_aln_len = blast_evalue = blast_bitscore = ""
        if args.do_blast and (not err_type) and target and cont_clip:
            blast_pident, blast_aln_len, blast_evalue, blast_bitscore = run_blastn_pair(
                query_seq=cont_clip,
                subject_seq=target,
                blastn_path=args.blastn_path,
                task=args.blast_task,
                word_size=args.blast_word_size,
                evalue=args.blast_evalue,
                max_target_seqs=1,
                timeout_sec=args.blast_timeout,
            )

        err_detail_s = (err_detail or "").replace("\n", "\\n")
        if len(err_detail_s) > 5000:
            err_detail_s = err_detail_s[:5000] + "...(truncated)"

        row = [
            sample["sample_id"], sample["family"], sample["asm"], sample["taxid"],
            str(len(prefix)), str(len(target)),
            cont_clip, str(len(cont_clip)),
            err_type, err_detail_s, str(retries),
            f"{gc_prefix:.6f}", f"{gc_target:.6f}", f"{gc_gen:.6f}", f"{gc_diff:.6f}",
            f"{jsd3:.6f}", f"{sim:.6f}",
            blast_pident, blast_aln_len, blast_evalue, blast_bitscore,
        ]
        w.write("\t".join(row) + "\n")

    # 实时落盘（你之前问过是否一条写一条）
    w.flush()


if __name__ == "__main__":
    main()

from Bio import pairwise2
from Bio.Align import substitution_matrices
from Bio.Seq import Seq
import pandas as pd
import sys
from tqdm import tqdm

def compare_homology(
    seq1: str,
    seq2: str,
    seq_type: str = "dna",       # "dna" or "protein"
    mode: str = "global",        # "global" or "local"
    match: float = 1.0,          # for dna scoring
    mismatch: float = 0.0,       # for dna scoring (常见也用 -1)
    gap_open: float = -10.0,
    gap_extend: float = -0.5,
    matrix_name: str = "BLOSUM62",
    return_alignment: bool = False,
):
    """
    Compare homology between two sequences via pairwise alignment.

    Returns:
        dict with:
          - score
          - identity_ungapped_pct: identity over aligned columns excluding gaps
          - identity_aln_pct: identity over full alignment length (including gaps)
          - matches, mismatches, gaps
          - aligned_len, ungapped_len
          - coverage_seq1_pct, coverage_seq2_pct
          - (optional) aligned_seq1, aligned_seq2
    """
    s1 = "".join(seq1.split()).upper()
    s2 = "".join(seq2.split()).upper()

    if len(s1) == 0 or len(s2) == 0:
        raise ValueError("Empty sequence detected.")

    # choose alignment function
    if seq_type.lower() in ("dna", "nt", "nuc", "nucleotide"):
        if mode == "global":
            aln = pairwise2.align.globalms(
                s1, s2, match, mismatch, gap_open, gap_extend, one_alignment_only=True
            )[0]
        elif mode == "local":
            aln = pairwise2.align.localms(
                s1, s2, match, mismatch, gap_open, gap_extend, one_alignment_only=True
            )[0]
        else:
            raise ValueError("mode must be 'global' or 'local'")
    elif seq_type.lower() in ("protein", "aa", "pep"):
        matrix = substitution_matrices.load(matrix_name)
        if mode == "global":
            aln = pairwise2.align.globalds(
                s1, s2, matrix, gap_open, gap_extend, one_alignment_only=True
            )[0]
        elif mode == "local":
            aln = pairwise2.align.localds(
                s1, s2, matrix, gap_open, gap_extend, one_alignment_only=True
            )[0]
        else:
            raise ValueError("mode must be 'global' or 'local'")
    else:
        raise ValueError("seq_type must be 'dna' or 'protein'")

    aligned1, aligned2, score, begin, end = aln
    aligned_len = len(aligned1)

    matches = 0
    mismatches = 0
    gaps = 0
    ungapped_cols = 0

    for a, b in zip(aligned1, aligned2):
        if a == "-" or b == "-":
            gaps += 1
            continue
        ungapped_cols += 1
        if a == b:
            matches += 1
        else:
            mismatches += 1

    identity_ungapped_pct = (matches / ungapped_cols * 100.0) if ungapped_cols else 0.0
    identity_aln_pct = (matches / aligned_len * 100.0) if aligned_len else 0.0

    # coverage: aligned non-gap columns / original length
    coverage_seq1_pct = (ungapped_cols / len(s1) * 100.0) if len(s1) else 0.0
    coverage_seq2_pct = (ungapped_cols / len(s2) * 100.0) if len(s2) else 0.0

    out = {
        "score": float(score),
        "identity_ungapped_pct": float(identity_ungapped_pct),
        "identity_aln_pct": float(identity_aln_pct),
        "matches": int(matches),
        "mismatches": int(mismatches),
        "gaps": int(gaps),
        "aligned_len": int(aligned_len),
        "ungapped_len": int(ungapped_cols),
        "coverage_seq1_pct": float(coverage_seq1_pct),
        "coverage_seq2_pct": float(coverage_seq2_pct),
        "mode": mode,
        "seq_type": seq_type,
    }

    if return_alignment:
        out["aligned_seq1"] = aligned1
        out["aligned_seq2"] = aligned2

    return out


def translate_dna_to_protein(dna_seq):
    """将DNA序列翻译为蛋白质序列"""
    if pd.isna(dna_seq) or not dna_seq or str(dna_seq).strip() == '':
        return None
    try:
        seq = Seq(str(dna_seq).upper())
        protein = seq.translate(to_stop=True)  # to_stop=False保留终止密码子
        return str(protein)
    except Exception:
        # 如果产生错误则忽略该行，返回None
        return None


def combine_and_translate(row):
    """拼接prompt_sequence和generated_sequence，然后翻译为蛋白质"""
    try:
        prompt = str(row['prompt_sequence']) if pd.notna(row['prompt_sequence']) else ''
        generated = str(row['generated_sequence']) if pd.notna(row['generated_sequence']) else ''
        combined = prompt + generated
        return translate_dna_to_protein(combined)
    except Exception:
        # 如果产生错误则忽略该行，返回None
        return None


# ========== Example ==========
"""
Usage example:
    # With filter for equal length proteins
python script/blast.py \
    /inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/results/gen_cds_results_evo2_1b_base.csv \
    --output /inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/results/gen_cds_results_evo2_1b_base_protein.csv \
    --filter-equal-length
python script/blast.py \
    /inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/results/gen_cds_results_evo2_7b_base.csv \
    --output /inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/results/gen_cds_results_evo2_7b_base_protein.csv \
    --filter-equal-length
python script/blast.py \
    /inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/results/gen_cds_results_evo2_7b.csv \
    --output /inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/results/gen_cds_results_evo2_7b_protein.csv \
    --filter-equal-length
python script/blast.py \
    /inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/results/gen_cds_results_evo2_40b_base.csv \
    --output /inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/results/gen_cds_results_evo2_40b_base_protein.csv \
    --filter-equal-length
python script/blast.py \
    /inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/results/gen_cds_results_evo2_40b.csv \
    --output /inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/results/gen_cds_results_evo2_40b_protein.csv \
    --filter-equal-length
"""

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate homology between original and generated sequences from CSV")
    parser.add_argument("input_csv", type=str, help="Input CSV file path")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output CSV file path (default: overwrite input file)")
    parser.add_argument("--seq-type", type=str, default="protein", choices=["dna", "protein"], help="Sequence type (default: protein)")
    parser.add_argument("--mode", type=str, default="global", choices=["global", "local"], help="Alignment mode (default: global)")
    parser.add_argument("--mismatch", type=float, default=-1.0, help="Mismatch penalty for DNA (default: -1.0)")
    parser.add_argument("--filter-equal-length", action="store_true", help="Filter rows where original_protein and generated_protein have equal length")
    
    args = parser.parse_args()
    
    # Read CSV file
    df = pd.read_csv(args.input_csv)
    
    # Check required columns
    required_cols = ["original_sequence", "prompt_sequence", "generated_sequence"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Translate DNA sequences to proteins
    print("Translating original_sequence to original_protein...")
    df['original_protein'] = [translate_dna_to_protein(seq) for seq in tqdm(df['original_sequence'], desc="Original sequences")]
    
    print("\nTranslating prompt_sequence+generated_sequence to generated_protein...")
    df['generated_protein'] = [combine_and_translate(row) for _, row in tqdm(df.iterrows(), total=len(df), desc="Generated sequences")]
    
    # Display translation results
    print(f"\nTotal rows: {len(df)}")
    print(f"Original protein non-null: {df['original_protein'].notna().sum()}")
    print(f"Generated protein non-null: {df['generated_protein'].notna().sum()}")
    
    # Filter equal length rows if requested
    if args.filter_equal_length:
        equal_length_mask = df['original_protein'].str.len() == df['generated_protein'].str.len()
        df = df[equal_length_mask].copy()
        print(f"\nFiltered to {len(df)} rows with equal protein lengths")
    
    # Calculate homology for each row using proteins
    identity_scores = []
    pbar = tqdm(df.iterrows(), total=len(df), desc="Processing sequences")
    for idx, row in pbar:
        original_protein = row["original_protein"]
        generated_protein = row["generated_protein"]
        
        # Skip if proteins are empty or None
        if pd.isna(original_protein) or pd.isna(generated_protein) or not original_protein or not generated_protein:
            identity_scores.append(None)
            pbar.set_postfix({"identity_ungapped_pct": "N/A"})
            continue
        
        try:
            # Calculate homology between proteins
            r = compare_homology(
                str(original_protein),
                str(generated_protein),
                seq_type="protein",  # Always use protein for comparison
                mode=args.mode,
                return_alignment=False
            )
            identity_pct = r["identity_ungapped_pct"]
            identity_scores.append(identity_pct)
            pbar.set_postfix({"identity_ungapped_pct": f"{identity_pct:.2f}%"})
        except Exception as e:
            print(f"Error processing row {idx}: {e}", file=sys.stderr)
            identity_scores.append(None)
            pbar.set_postfix({"identity_ungapped_pct": "Error"})
    
    # Add identity_ungapped_pct column
    df["identity_ungapped_pct"] = identity_scores
    
    # Remove specified columns before saving
    columns_to_remove = ["prompt_sequence", "ground_truth", "generated_sequence", "generated_full_sequence","ground_truth_length","original_sequence"]
    existing_cols_to_remove = [col for col in columns_to_remove if col in df.columns]
    if existing_cols_to_remove:
        df = df.drop(columns=existing_cols_to_remove)
    
    # Save to output file
    output_path = args.output if args.output else args.input_csv
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

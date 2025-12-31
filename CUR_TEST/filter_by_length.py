import os
import gzip
import pandas as pd
import numpy as np
from tqdm import tqdm

# Import functions from training script
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_cnn_viral_family_genus_disjoint_randcrop_multicrop import (
    read_first_fasta_sequence_gz,
    build_genome_path,
    Config
)

def get_sequence_length(outdir: str, taxid: str, asm: str, db_source: str) -> int:
    """Get the length of a sequence from the fasta file."""
    fna = build_genome_path(outdir, taxid, asm, db_source)
    if not os.path.exists(fna):
        return 0
    try:
        seq = read_first_fasta_sequence_gz(fna)
        return len(seq)
    except Exception as e:
        print(f"Warning: Failed to load {fna}: {e}")
        return 0

def filter_by_length(input_tsv: str, output_tsv: str, max_length: int, outdir: str):
    """
    Filter TSV file to keep only sequences with length <= max_length.
    
    Args:
        input_tsv: Path to input TSV file
        output_tsv: Path to output TSV file
        max_length: Maximum sequence length in bp (131000)
        outdir: Base directory for genome files
    """
    print(f"Reading TSV file: {input_tsv}")
    df = pd.read_csv(input_tsv, sep="\t")
    print(f"Total samples: {len(df)}")
    
    # Calculate sequence lengths
    print(f"\nCalculating sequence lengths (filtering for <= {max_length:,} bp)...")
    seq_lengths = []
    valid_indices = []
    
    for idx in tqdm(range(len(df)), desc="Processing sequences"):
        row = df.iloc[idx]
        taxid = str(row["taxid"])
        asm = str(row["asm"])
        db = str(row["db_source"])
        
        length = get_sequence_length(outdir, taxid, asm, db)
        seq_lengths.append(length)
        
        if length > 0 and length <= max_length:
            valid_indices.append(idx)
    
    # Filter dataframe
    filtered_df = df.iloc[valid_indices].copy()
    
    print(f"\nFiltering results:")
    print(f"  Original samples: {len(df)}")
    print(f"  Samples with length <= {max_length:,} bp: {len(filtered_df)}")
    print(f"  Samples removed: {len(df) - len(filtered_df)}")
    
    if len(filtered_df) > 0:
        filtered_lengths = [seq_lengths[i] for i in valid_indices]
        print(f"\nFiltered dataset statistics:")
        print(f"  Min length: {min(filtered_lengths):,} bp")
        print(f"  Max length: {max(filtered_lengths):,} bp")
        print(f"  Mean length: {np.mean(filtered_lengths):,.0f} bp")
        print(f"  Median length: {np.median(filtered_lengths):,.0f} bp")
    
    # Save filtered TSV
    print(f"\nSaving filtered data to: {output_tsv}")
    filtered_df.to_csv(output_tsv, sep="\t", index=False)
    print("Done!")

if __name__ == "__main__":
    # Configuration
    OUTDIR = os.environ.get("OUTDIR", "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/ncbi_viral")
    INPUT_TSV = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/ncbi_viral/merged/rep.family_top100.with_header.tsv"
    OUTPUT_TSV = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/ncbi_viral/merged/rep.family_top100.with_header.length_leq131k.tsv"
    MAX_LENGTH = 131_000  # 131K bp
    
    filter_by_length(INPUT_TSV, OUTPUT_TSV, MAX_LENGTH, OUTDIR)


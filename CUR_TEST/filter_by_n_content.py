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

def get_n_content(outdir: str, taxid: str, asm: str, db_source: str) -> tuple:
    """
    Get the N content percentage and length of a sequence.
    Returns: (n_content_percentage, length) or (None, 0) if failed
    """
    fna = build_genome_path(outdir, taxid, asm, db_source)
    if not os.path.exists(fna):
        return None, 0
    try:
        seq = read_first_fasta_sequence_gz(fna)
        length = len(seq)
        if length == 0:
            return None, 0
        n_count = seq.count('N') + seq.count('n')
        n_content = (n_count / length) * 100.0
        return n_content, length
    except Exception as e:
        print(f"Warning: Failed to load {fna}: {e}")
        return None, 0

def filter_by_n_content(input_tsv: str, output_tsv: str, max_n_content: float, outdir: str):
    """
    Filter TSV file to keep only sequences with N content <= max_n_content.
    
    Args:
        input_tsv: Path to input TSV file
        output_tsv: Path to output TSV file
        max_n_content: Maximum N content percentage (10.0)
        outdir: Base directory for genome files
    """
    print(f"Reading TSV file: {input_tsv}")
    df = pd.read_csv(input_tsv, sep="\t")
    print(f"Total samples: {len(df)}")
    
    # Calculate N content for each sequence
    print(f"\nCalculating N content (filtering for <= {max_n_content}%)...")
    n_contents = []
    valid_indices = []
    
    for idx in tqdm(range(len(df)), desc="Processing sequences"):
        row = df.iloc[idx]
        taxid = str(row["taxid"])
        asm = str(row["asm"])
        db = str(row["db_source"])
        
        n_content, length = get_n_content(outdir, taxid, asm, db)
        
        if n_content is not None:
            n_contents.append(n_content)
            if n_content <= max_n_content:
                valid_indices.append(idx)
        else:
            n_contents.append(None)
    
    # Filter dataframe
    filtered_df = df.iloc[valid_indices].copy()
    
    print(f"\nFiltering results:")
    print(f"  Original samples: {len(df)}")
    print(f"  Samples with N content <= {max_n_content}%: {len(filtered_df)}")
    print(f"  Samples removed: {len(df) - len(filtered_df)}")
    
    if len(filtered_df) > 0:
        filtered_n_contents = [n_contents[i] for i in valid_indices if n_contents[i] is not None]
        if filtered_n_contents:
            print(f"\nFiltered dataset N content statistics:")
            print(f"  Min N content: {min(filtered_n_contents):.2f}%")
            print(f"  Max N content: {max(filtered_n_contents):.2f}%")
            print(f"  Mean N content: {np.mean(filtered_n_contents):.2f}%")
            print(f"  Median N content: {np.median(filtered_n_contents):.2f}%")
    
    # Save filtered TSV
    print(f"\nSaving filtered data to: {output_tsv}")
    filtered_df.to_csv(output_tsv, sep="\t", index=False)
    print("Done!")

if __name__ == "__main__":
    # Configuration
    OUTDIR = os.environ.get("OUTDIR", "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/ncbi_viral")
    INPUT_TSV = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/ncbi_viral/merged/rep.family_top100.with_header.length_leq131k.n_leq10pct.tsv"
    OUTPUT_TSV = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/ncbi_viral/merged/rep.family_top100.with_header.length_leq131k.n_leq5pct.tsv"
    MAX_N_CONTENT = 5.0  # 5%
    
    filter_by_n_content(INPUT_TSV, OUTPUT_TSV, MAX_N_CONTENT, OUTDIR)


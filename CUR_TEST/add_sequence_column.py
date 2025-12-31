import os
import gzip
import pandas as pd
from tqdm import tqdm

# Import functions from training script
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_cnn_viral_family_genus_disjoint_randcrop_multicrop import (
    read_first_fasta_sequence_gz,
    build_genome_path
)

def add_sequence_column(input_tsv: str, output_tsv: str, outdir: str):
    """
    Add a 'sequence' column to the TSV file containing the full DNA sequence.
    
    Args:
        input_tsv: Path to input TSV file
        output_tsv: Path to output TSV file
        outdir: Base directory for genome files
    """
    print(f"Reading TSV file: {input_tsv}")
    df = pd.read_csv(input_tsv, sep="\t")
    print(f"Total samples: {len(df)}")
    
    # Add sequence column
    print(f"\nLoading sequences...")
    sequences = []
    failed_indices = []
    
    for idx in tqdm(range(len(df)), desc="Processing sequences"):
        row = df.iloc[idx]
        taxid = str(row["taxid"])
        asm = str(row["asm"])
        db = str(row["db_source"])
        
        try:
            fna = build_genome_path(outdir, taxid, asm, db)
            if not os.path.exists(fna):
                sequences.append("")
                failed_indices.append(idx)
                continue
            
            seq = read_first_fasta_sequence_gz(fna)
            sequences.append(seq)
        except Exception as e:
            print(f"Warning: Failed to load sequence at index {idx} (taxid={taxid}, asm={asm}): {e}")
            sequences.append("")
            failed_indices.append(idx)
    
    # Add sequence column to dataframe
    df["sequence"] = sequences
    
    # Report statistics
    print(f"\nProcessing results:")
    print(f"  Total samples: {len(df)}")
    print(f"  Successfully loaded: {len(df) - len(failed_indices)}")
    print(f"  Failed to load: {len(failed_indices)}")
    
    if failed_indices:
        print(f"  Failed indices (first 10): {failed_indices[:10]}")
    
    # Count non-empty sequences
    non_empty = df["sequence"].str.len() > 0
    print(f"  Non-empty sequences: {non_empty.sum()}")
    print(f"  Empty sequences: {(~non_empty).sum()}")
    
    # Save TSV with sequence column
    print(f"\nSaving TSV file with sequence column to: {output_tsv}")
    df.to_csv(output_tsv, sep="\t", index=False)
    print("Done!")

if __name__ == "__main__":
    # Configuration
    OUTDIR = os.environ.get("OUTDIR", "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/ncbi_viral")
    INPUT_TSV = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/ncbi_viral/merged/rep.family_top100.with_header.length_leq131k.n_leq5pct.tsv"
    OUTPUT_TSV = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/ncbi_viral/merged/rep.family_top100.with_header.length_leq131k.n_leq5pct.with_sequence.tsv"
    
    add_sequence_column(INPUT_TSV, OUTPUT_TSV, OUTDIR)


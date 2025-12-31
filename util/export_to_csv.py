"""
Script to export TSV data with sequences to CSV format.
"""
import os
import sys
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.c1_dataset import ClassificationDataset


def export_tsv_to_csv(
    tsv_path: str,
    outdir: str,
    output_csv_path: str,
    label_column: str = "family",
    cache_dir: str = "",
    max_cache_bp: int = 200_000
):
    """
    Export TSV data with sequences to CSV format.
    
    Args:
        tsv_path: Path to input TSV file
        outdir: Base directory where genome files are stored
        output_csv_path: Path to output CSV file
        label_column: Column name for labels (default: 'family')
        cache_dir: Directory to cache encoded sequences. Set "" to disable cache.
        max_cache_bp: Maximum sequence length to cache
    """
    # Load dataframe
    print(f"Loading TSV file: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t")
    print(f"Loaded {len(df)} samples")
    
    # Create label mapping
    labels = sorted(df[label_column].astype(str).unique())
    label2id = {label: i for i, label in enumerate(labels)}
    print(f"Number of classes: {len(label2id)}")
    
    # Initialize dataset
    print("Initializing dataset...")
    dataset = ClassificationDataset(
        df=df,
        outdir=outdir,
        label2id=label2id,
        label_column=label_column,
        cache_dir=cache_dir,
        max_cache_bp=max_cache_bp
    )
    
    # Prepare output data
    print("Loading sequences and preparing CSV data...")
    output_data = []
    
    for idx in tqdm(range(len(dataset)), desc="Processing sequences"):
        row = df.iloc[idx]
        
        # Get sequence
        sequence = dataset.get_raw_sequence(idx)
        
        # Prepare row data: idx, taxid, organism_name, family, genus, species, sequence
        output_data.append({
            "idx": idx,
            "taxid": str(row["taxid"]),
            "organism_name": str(row["organism_name"]),
            "family": str(row["family"]),
            "genus": str(row["genus"]),
            "species": str(row["species"]),
            "sequence": sequence
        })
    
    # Create output dataframe
    output_df = pd.DataFrame(output_data)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    # Save to CSV
    print(f"Saving to CSV: {output_csv_path}")
    output_df.to_csv(output_csv_path, index=False)
    print(f"Saved {len(output_df)} rows to {output_csv_path}")
    
    # Print summary
    print("\nSummary:")
    print(f"  Total samples: {len(output_df)}")
    print(f"  Samples with sequences: {(output_df['sequence'].str.len() > 0).sum()}")
    print(f"  Samples without sequences: {(output_df['sequence'].str.len() == 0).sum()}")
    print(f"  Average sequence length: {output_df['sequence'].str.len().mean():.2f}")
    print(f"  Min sequence length: {output_df['sequence'].str.len().min()}")
    print(f"  Max sequence length: {output_df['sequence'].str.len().max()}")


if __name__ == "__main__":
    # Configuration
    tsv_path = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/ncbi_viral/merged/rep.family_top100.with_header.tsv"
    outdir = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/ncbi_viral"
    output_csv_path = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/data/exported_data.csv"
    
    export_tsv_to_csv(
        tsv_path=tsv_path,
        outdir=outdir,
        output_csv_path=output_csv_path,
        label_column="family",
        cache_dir="",  # Set to "" to disable cache
        max_cache_bp=200_000
    )


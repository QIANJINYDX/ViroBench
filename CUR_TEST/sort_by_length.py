import pandas as pd

def sort_by_sequence_length(input_tsv: str, output_tsv: str):
    """
    Sort TSV file by sequence length (ascending order).
    
    Args:
        input_tsv: Path to input TSV file
        output_tsv: Path to output TSV file
    """
    print(f"Reading TSV file: {input_tsv}")
    df = pd.read_csv(input_tsv, sep="\t")
    print(f"Total samples: {len(df)}")
    
    # Check if sequence column exists
    if "sequence" not in df.columns:
        raise ValueError("Column 'sequence' not found in the TSV file!")
    
    # Calculate sequence lengths
    print("Calculating sequence lengths...")
    df["sequence_length"] = df["sequence"].str.len()
    
    # Sort by sequence length (ascending)
    print("Sorting by sequence length (ascending)...")
    df_sorted = df.sort_values("sequence_length", ascending=True).reset_index(drop=True)
    
    # Remove the temporary sequence_length column (optional, you can keep it if needed)
    df_sorted = df_sorted.drop(columns=["sequence_length"])
    
    # Report statistics
    print(f"\nSorting results:")
    print(f"  Total samples: {len(df_sorted)}")
    if len(df_sorted) > 0:
        lengths = df_sorted["sequence"].str.len()
        print(f"  Min length: {lengths.min():,} bp")
        print(f"  Max length: {lengths.max():,} bp")
        print(f"  Mean length: {lengths.mean():,.0f} bp")
        print(f"  Median length: {lengths.median():,.0f} bp")
    
    # Save sorted TSV
    print(f"\nSaving sorted TSV file to: {output_tsv}")
    df_sorted.to_csv(output_tsv, sep="\t", index=False)
    print("Done!")

if __name__ == "__main__":
    INPUT_TSV = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/ncbi_viral/merged/rep.family_top100.with_header.length_leq131k.n_leq5pct.with_sequence.tsv"
    OUTPUT_TSV = INPUT_TSV  # Overwrite the original file
    
    sort_by_sequence_length(INPUT_TSV, OUTPUT_TSV)


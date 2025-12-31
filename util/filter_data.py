"""
Script to filter CSV data:
- Remove rows with sequence length > 131K
- Remove rows with N content > 5%
"""
import pandas as pd
from tqdm import tqdm


def filter_data(
    input_csv: str,
    output_csv: str,
    max_length: int = 131_000,
    max_n_content: float = 0.05
):
    """
    Filter CSV data based on sequence length and N content.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
        max_length: Maximum sequence length (default: 131000)
        max_n_content: Maximum N content ratio (default: 0.05 = 5%)
    """
    print(f"Loading data from: {input_csv}")
    
    # Read CSV in chunks to handle large files
    chunk_size = 10000
    chunks = []
    total_rows = 0
    
    # First pass: count total rows
    print("Counting total rows...")
    for chunk in pd.read_csv(input_csv, chunksize=chunk_size):
        total_rows += len(chunk)
    
    print(f"Total rows: {total_rows}")
    print("Processing data...")
    
    # Second pass: process and filter
    filtered_chunks = []
    removed_length = 0
    removed_n_content = 0
    kept = 0
    
    for chunk in tqdm(pd.read_csv(input_csv, chunksize=chunk_size), 
                      total=(total_rows // chunk_size + 1),
                      desc="Filtering"):
        # Calculate sequence lengths
        chunk['seq_length'] = chunk['sequence'].str.len()
        
        # Filter by length
        length_mask = chunk['seq_length'] <= max_length
        removed_length += (~length_mask).sum()
        
        # Filter by N content
        n_counts = chunk['sequence'].str.count('N') + chunk['sequence'].str.count('n')
        n_content = n_counts / chunk['seq_length']
        n_mask = n_content <= max_n_content
        removed_n_content += (~n_mask).sum()
        
        # Combine filters
        final_mask = length_mask & n_mask
        filtered_chunk = chunk[final_mask].copy()
        
        # Drop temporary column
        filtered_chunk = filtered_chunk.drop(columns=['seq_length'])
        
        if len(filtered_chunk) > 0:
            filtered_chunks.append(filtered_chunk)
        
        kept += final_mask.sum()
    
    # Concatenate all filtered chunks
    print("Combining filtered data...")
    if filtered_chunks:
        filtered_df = pd.concat(filtered_chunks, ignore_index=True)
    else:
        print("Warning: No data remaining after filtering!")
        filtered_df = pd.DataFrame(columns=['idx', 'taxid', 'organism_name', 'family', 'genus', 'species', 'sequence'])
    
    # Save to CSV
    print(f"Saving filtered data to: {output_csv}")
    filtered_df.to_csv(output_csv, index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("Filtering Summary:")
    print("="*60)
    print(f"  Original rows:        {total_rows:,}")
    print(f"  Removed (length):     {removed_length:,} (length > {max_length:,})")
    print(f"  Removed (N content): {removed_n_content:,} (N > {max_n_content*100:.1f}%)")
    print(f"  Final rows:          {len(filtered_df):,}")
    print(f"  Retention rate:      {len(filtered_df)/total_rows*100:.2f}%")
    
    if len(filtered_df) > 0:
        print(f"\nSequence Statistics (after filtering):")
        seq_lengths = filtered_df['sequence'].str.len()
        print(f"  Average length:      {seq_lengths.mean():.2f}")
        print(f"  Min length:          {seq_lengths.min():,}")
        print(f"  Max length:          {seq_lengths.max():,}")
        
        # Calculate N content statistics
        n_counts = filtered_df['sequence'].str.count('N') + filtered_df['sequence'].str.count('n')
        n_content = n_counts / seq_lengths
        print(f"  Average N content:   {n_content.mean()*100:.4f}%")
        print(f"  Max N content:       {n_content.max()*100:.4f}%")
    
    print("="*60)


if __name__ == "__main__":
    input_csv = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/data/C1_data.csv"
    output_csv = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/data/G1_data.csv"
    
    filter_data(
        input_csv=input_csv,
        output_csv=output_csv,
        max_length=131_000,  # 131K
        max_n_content=0.05    # 5%
    )


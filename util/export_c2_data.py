"""
Script to export TSV data with CDS information to CSV+JSON format.
"""
import os
import sys
import gzip
import json
import re
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_cds_header(header: str) -> Dict[str, Any]:
    """
    Parse CDS header line to extract information.
    
    Example: >lcl|JX489135.1_cds_AGJ91147.1_1 [locus_tag=VACV_TT8_001] [protein=chemokine-binding protein] [protein_id=AGJ91147.1] [location=complement(72..791)] [gbkey=CDS]
    """
    info = {}
    
    # Extract sequence ID (before first space)
    parts = header.split(' ', 1)
    if parts:
        info['sequence_id'] = parts[0].lstrip('>')
    
    # Extract attributes in [key=value] format
    if len(parts) > 1:
        attr_pattern = r'\[([^=]+)=([^\]]+)\]'
        matches = re.findall(attr_pattern, parts[1])
        for key, value in matches:
            info[key.strip()] = value.strip()
    
    return info


def read_all_cds_from_gz(path: str) -> List[Dict[str, Any]]:
    """
    Read all CDS sequences from a gzipped FASTA file.
    
    Returns:
        List of dictionaries, each containing:
        - header_info: parsed header information
        - sequence: DNA sequence string
    """
    cds_list = []
    
    if not os.path.exists(path):
        return cds_list
    
    try:
        with gzip.open(path, "rt") as f:
            current_header = None
            current_seq_lines = []
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith(">"):
                    # Save previous CDS if exists
                    if current_header is not None:
                        sequence = "".join(current_seq_lines).upper()
                        header_info = parse_cds_header(current_header)
                        cds_list.append({
                            "header_info": header_info,
                            "sequence": sequence
                        })
                    
                    # Start new CDS
                    current_header = line
                    current_seq_lines = []
                else:
                    current_seq_lines.append(line)
            
            # Save last CDS
            if current_header is not None:
                sequence = "".join(current_seq_lines).upper()
                header_info = parse_cds_header(current_header)
                cds_list.append({
                    "header_info": header_info,
                    "sequence": sequence
                })
    except Exception as e:
        print(f"Warning: Error reading CDS file {path}: {e}")
        return []
    
    return cds_list


def build_cds_path(outdir: str, taxid: str, asm: str, db_source: str) -> str:
    """Build path to CDS fasta.gz file."""
    base = "refseq" if db_source == "refseq" else "genbank"
    return os.path.join(outdir, base, "downloads", taxid, asm, f"{asm}_cds_from_genomic.fna.gz")


def export_c2_data(
    tsv_path: str,
    outdir: str,
    output_csv_path: str,
    output_json_path: str
):
    """
    Export TSV data with CDS information to CSV and JSON format.
    
    Args:
        tsv_path: Path to input TSV file
        outdir: Base directory where CDS files are stored
        output_csv_path: Path to output CSV file (without CDS column)
        output_json_path: Path to output JSON file (CDS data)
    """
    print(f"Loading TSV file: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t")
    print(f"Loaded {len(df)} samples")
    
    # Prepare output data
    print("Processing data and loading CDS sequences...")
    csv_data = []
    json_data = {}  # {idx: [cds_list]}
    
    for idx in tqdm(range(len(df)), desc="Processing"):
        row = df.iloc[idx]
        
        # Extract required columns
        taxid = str(row["taxid"])
        asm = str(row["asm"])
        db_source = str(row["db_source"])
        
        # Build CDS path and load CDS
        cds_path = build_cds_path(outdir, taxid, asm, db_source)
        cds_list = read_all_cds_from_gz(cds_path)
        
        # Prepare CSV row (without CDS)
        csv_row = {
            "idx": idx,
            "taxid": taxid,
            "organism_name": str(row["organism_name"]),
            "family": str(row["family"]),
            "genus": str(row["genus"]),
            "species": str(row["species"]),
            "host_name": str(row["host_name_clean"]),
            "host_group": str(row["host_group_fixed"])
        }
        csv_data.append(csv_row)
        
        # Store CDS data in JSON format
        json_data[str(idx)] = cds_list
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    # Save CSV
    print(f"Saving CSV to: {output_csv_path}")
    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv(output_csv_path, index=False)
    
    # Save JSON
    print(f"Saving JSON to: {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*60)
    print("Export Summary:")
    print("="*60)
    print(f"  Total samples:        {len(csv_data):,}")
    
    # Count CDS statistics
    total_cds = sum(len(cds_list) for cds_list in json_data.values())
    samples_with_cds = sum(1 for cds_list in json_data.values() if len(cds_list) > 0)
    samples_without_cds = len(json_data) - samples_with_cds
    
    print(f"  Total CDS entries:    {total_cds:,}")
    print(f"  Samples with CDS:     {samples_with_cds:,}")
    print(f"  Samples without CDS:  {samples_without_cds:,}")
    
    if total_cds > 0:
        avg_cds_per_sample = total_cds / samples_with_cds if samples_with_cds > 0 else 0
        print(f"  Avg CDS per sample:    {avg_cds_per_sample:.2f}")
        
        # Calculate sequence length statistics
        all_lengths = []
        for cds_list in json_data.values():
            for cds in cds_list:
                all_lengths.append(len(cds['sequence']))
        
        if all_lengths:
            print(f"  CDS length stats:")
            print(f"    Average:            {sum(all_lengths)/len(all_lengths):.2f}")
            print(f"    Min:                {min(all_lengths):,}")
            print(f"    Max:                {max(all_lengths):,}")
    
    print("="*60)
    print(f"\nFiles saved:")
    print(f"  CSV: {output_csv_path}")
    print(f"  JSON: {output_json_path}")


if __name__ == "__main__":
    tsv_path = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/ncbi_viral/merged/rep.family_top100.with_header.with_host.with_group.corrected.cleaned_host.tsv"
    outdir = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/ncbi_viral"
    output_csv_path = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/data/C2_data.csv"
    output_json_path = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/data/C2_data_cds.json"
    
    export_c2_data(
        tsv_path=tsv_path,
        outdir=outdir,
        output_csv_path=output_csv_path,
        output_json_path=output_json_path
    )


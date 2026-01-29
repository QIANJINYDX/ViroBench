import csv
import json
import os
from pathlib import Path

CODON2AA = {
    # Phenylalanine / Leucine
    "TTT":"F","TTC":"F","TTA":"L","TTG":"L",
    "CTT":"L","CTC":"L","CTA":"L","CTG":"L",
    # Isoleucine / Methionine
    "ATT":"I","ATC":"I","ATA":"I","ATG":"M",
    # Valine
    "GTT":"V","GTC":"V","GTA":"V","GTG":"V",
    # Serine / Proline / Threonine / Alanine
    "TCT":"S","TCC":"S","TCA":"S","TCG":"S",
    "CCT":"P","CCC":"P","CCA":"P","CCG":"P",
    "ACT":"T","ACC":"T","ACA":"T","ACG":"T",
    "GCT":"A","GCC":"A","GCA":"A","GCG":"A",
    # Tyrosine / Histidine / Glutamine / Asparagine / Lysine
    "TAT":"Y","TAC":"Y",
    "CAT":"H","CAC":"H",
    "CAA":"Q","CAG":"Q",
    "AAT":"N","AAC":"N",
    "AAA":"K","AAG":"K",
    # Aspartic acid / Glutamic acid
    "GAT":"D","GAC":"D",
    "GAA":"E","GAG":"E",
    # Cysteine / Tryptophan
    "TGT":"C","TGC":"C",
    "TGG":"W",
    # Arginine / Glycine
    "CGT":"R","CGC":"R","CGA":"R","CGG":"R","AGA":"R","AGG":"R",
    "GGT":"G","GGC":"G","GGA":"G","GGG":"G",
    # Stop
    "TAA":"*","TAG":"*","TGA":"*",
}
AA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY") + ["*"]


def cds_to_protein(cds_sequence):
    """
    Convert CDS (DNA) sequence to protein sequence.
    Stops at the first stop codon.
    """
    protein = []
    # Process in groups of 3 (codons)
    for i in range(0, len(cds_sequence) - 2, 3):
        codon = cds_sequence[i:i+3].upper()
        if codon in CODON2AA:
            aa = CODON2AA[codon]
            # Stop at stop codon
            if aa == "*":
                break
            protein.append(aa)
        else:
            # Unknown codon, skip or use X
            protein.append("X")
    return "".join(protein)


def create_af3_input(sequence, name, model_seed=42):
    """
    Create AlphaFold3 input JSON structure.
    """
    return {
        "name": name,
        "modelSeeds": [model_seed],
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": sequence,
                    "unpairedMsa": "",
                    "unpairedMsaPath": "",
                    "pairedMsa": "",
                    "pairedMsaPath": "",
                    "templates": ""
                }
            }
        ],
        "dialect": "alphafold3",
        "version": 2
    }


def create_protenix_input(sequence, name, count=1):
    """
    Create Protenix input JSON structure.
    """
    return [
        {
            "sequences": [
                {
                    "proteinChain": {
                        "sequence": sequence,
                        "count": count
                    }
                }
            ],
            "name": name
        }
    ]


def build_af3_inputs():
    """
    Build AlphaFold3 input files.
    """
    # Paths
    csv_path = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/results/CDS_success/all_models_high_accuracy.csv"
    output_dir = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/results/af3/input"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read CSV and process each row
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row['model']
            taxid = row['taxid']
            ground_truth = row['ground_truth']
            generated_continuation = row['generated_continuation']
            
            # Skip empty rows
            if not ground_truth or not generated_continuation:
                continue
            
            # Convert CDS to protein sequences
            try:
                protein_truth = cds_to_protein(ground_truth)
                protein_gen = cds_to_protein(generated_continuation)
                
                # Skip if protein sequences are empty
                if not protein_truth or not protein_gen:
                    print(f"Warning: Empty protein sequence for {model}_{taxid}, skipping...")
                    continue
                
                # Create file names
                name_truth = f"{model}_{taxid}_truth"
                name_gen = f"{model}_{taxid}_gen"
                
                # Create AF3 input structures
                af3_input_truth = create_af3_input(protein_truth, name_truth)
                af3_input_gen = create_af3_input(protein_gen, name_gen)
                
                # Save JSON files
                output_file_truth = os.path.join(output_dir, f"{name_truth}.json")
                output_file_gen = os.path.join(output_dir, f"{name_gen}.json")
                
                with open(output_file_truth, 'w') as f_out:
                    json.dump(af3_input_truth, f_out, indent=2)
                
                with open(output_file_gen, 'w') as f_out:
                    json.dump(af3_input_gen, f_out, indent=2)
                
                print(f"Created AF3: {name_truth}.json and {name_gen}.json")
                
            except Exception as e:
                print(f"Error processing {model}_{taxid}: {e}")
                continue
    
    print(f"\nAll AF3 files saved to: {output_dir}")


def build_protenix_inputs():
    """
    Build Protenix input files.
    """
    # Paths
    csv_path = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/results/CDS_success/all_models_high_accuracy.csv"
    output_dir = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/results/protenix/input"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read CSV and process each row
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row['model']
            taxid = row['taxid']
            ground_truth = row['ground_truth']
            generated_continuation = row['generated_continuation']
            
            # Skip empty rows
            if not ground_truth or not generated_continuation:
                continue
            
            # Convert CDS to protein sequences
            try:
                protein_truth = cds_to_protein(ground_truth)
                protein_gen = cds_to_protein(generated_continuation)
                
                # Skip if protein sequences are empty
                if not protein_truth or not protein_gen:
                    print(f"Warning: Empty protein sequence for {model}_{taxid}, skipping...")
                    continue
                
                # Create file names
                name_truth = f"{model}_{taxid}_truth"
                name_gen = f"{model}_{taxid}_gen"
                
                # Create Protenix input structures
                protenix_input_truth = create_protenix_input(protein_truth, name_truth)
                protenix_input_gen = create_protenix_input(protein_gen, name_gen)
                
                # Save JSON files
                output_file_truth = os.path.join(output_dir, f"{name_truth}.json")
                output_file_gen = os.path.join(output_dir, f"{name_gen}.json")
                
                with open(output_file_truth, 'w') as f_out:
                    json.dump(protenix_input_truth, f_out, indent=2)
                
                with open(output_file_gen, 'w') as f_out:
                    json.dump(protenix_input_gen, f_out, indent=2)
                
                print(f"Created Protenix: {name_truth}.json and {name_gen}.json")
                
            except Exception as e:
                print(f"Error processing {model}_{taxid}: {e}")
                continue
    
    print(f"\nAll Protenix files saved to: {output_dir}")


def main():
    """
    Main function to build both AF3 and Protenix inputs.
    """
    print("Building AlphaFold3 inputs...")
    build_af3_inputs()
    
    print("\nBuilding Protenix inputs...")
    build_protenix_inputs()


if __name__ == "__main__":
    main()

import json
from torch.utils.data import Dataset


class GenDataset(Dataset):
    """
    Dataset for loading genome sequence data from JSONL files.
    
    Each JSONL file contains one JSON object per line with:
    - taxid: taxonomy ID
    - sequences: list of sequence strings
    
    If a JSON object has multiple sequences, each sequence becomes a separate sample.
    The dataset loads data from a single JSONL file (long, medium, or short).
    
    Args:
        jsonl_path: Path to JSONL file (e.g., long_sequences.jsonl, medium_sequences.jsonl, short_sequences.jsonl)
    """
    
    def __init__(
        self,
        jsonl_path: str,
    ):
        self.jsonl_path = jsonl_path
        
        # Load JSONL data and expand sequences
        print(f"[INFO] Loading data from {jsonl_path}...")
        self.data = []  # List of (sequence, taxid) tuples
        num_objects = 0
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "taxid" not in obj or "sequences" not in obj:
                        print(f"[WARN] Skipping line {line_num}: missing 'taxid' or 'sequences' field")
                        continue
                    
                    taxid = obj.get("taxid")
                    sequences = obj.get("sequences", [])
                    if not sequences:
                        print(f"[WARN] Skipping line {line_num}: empty sequences list")
                        continue
                    
                    num_objects += 1
                    # Each sequence becomes a separate sample, with taxid
                    for seq in sequences:
                        if seq:  # Only add non-empty sequences
                            self.data.append((seq, taxid))
                        
                except json.JSONDecodeError as e:
                    print(f"[WARN] Skipping line {line_num}: JSON decode error - {e}")
                    continue
        
        print(f"[INFO] Loaded {len(self.data)} sequences from {jsonl_path} (expanded from {num_objects} JSON objects)")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int):
        """
        Get a sample from the dataset.
        
        Returns:
            Tuple of (idx, sequence, taxid):
            - idx: index (int)
            - sequence: raw sequence string
            - taxid: taxonomy ID (int)
        """
        sequence, taxid = self.data[idx]
        return idx, sequence, taxid

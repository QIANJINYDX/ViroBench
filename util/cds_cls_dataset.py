import os
import json
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit

# Global cache for JSON data (shared within the same process)
_JSON_CACHE: Dict[str, Dict] = {}


class CdsClsDataset(Dataset):
    """
    Dataset for CDS sequence classification tasks.
    
    Loads data from CSV file with idx and host_group label.
    CDS sequences are loaded from JSON file where each idx corresponds to multiple CDS sequences.
    For each idx, selects the top K longest CDS sequences and returns them as a list.
    Returns list of encoded sequences (variable length) as numpy arrays.
    Supports caching to speed up data loading.
    
    Args:
        csv_path: Path to CSV file with columns: idx, taxid, organism_name, family, genus, species, host_name, host_group
                 Or a pandas DataFrame with the same structure
        json_path: Path to JSON file containing CDS sequences. Format: {idx: [{"sequence": "...", "header_info": {...}}, ...]}
        label2id: Dictionary mapping label strings to integer IDs
        label_column: Column name in CSV for labels (default: 'host_group')
        cache_dir: Directory to cache encoded sequences. Set "" to disable cache.
                  All sequences will be cached if cache_dir is provided.
        vocab: Dictionary mapping nucleotide characters to integers (default: A=1, C=2, G=3, T=4, N=5, 0=PAD)
        top_k: Number of longest CDS sequences to select (default: 3)
    """
    
    VOCAB = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5}  # 0 is PAD
    
    def __init__(
        self,
        csv_path: Union[str, pd.DataFrame],
        json_path: str,
        label2id: Dict[str, int],
        label_column: str = "host_group",
        cache_dir: str = "",
        vocab: Optional[Dict[str, int]] = None,
        top_k: int = 3
    ):
        # Load DataFrame from CSV or use provided DataFrame
        if isinstance(csv_path, str):
            self.df = pd.read_csv(csv_path)
        else:
            self.df = csv_path.copy()
        
        self.df = self.df.reset_index(drop=True)
        self.json_path = json_path
        self.label2id = label2id
        self.label_column = label_column
        self.cache_dir = cache_dir
        self.vocab = vocab if vocab is not None else self.VOCAB
        self.top_k = top_k
        
        # Validate required columns
        if self.label_column not in self.df.columns:
            raise ValueError(f"Column '{self.label_column}' not found in CSV. Available columns: {list(self.df.columns)}")
        if "idx" not in self.df.columns:
            raise ValueError(f"Column 'idx' not found in CSV. Available columns: {list(self.df.columns)}")
        
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _load_json(self):
        """Lazy load JSON data with process-level caching."""
        global _JSON_CACHE
        
        # Check if JSON is already loaded in this process
        if self.json_path not in _JSON_CACHE:
            print(f"[INFO] Loading CDS sequences from {self.json_path}...")
            with open(self.json_path, 'r') as f:
                _JSON_CACHE[self.json_path] = json.load(f)
            print(f"[INFO] Loaded CDS data for {len(_JSON_CACHE[self.json_path])} indices.")
        
        # Use cached data
        self._cds_data = _JSON_CACHE[self.json_path]
    
    def __len__(self) -> int:
        return len(self.df)
    
    def _cache_path(self, idx: int) -> str:
        """Generate cache file path for a given index."""
        return os.path.join(self.cache_dir, f"{idx}.npz")
    
    def _encode_sequence_to_uint8(self, seq: str) -> np.ndarray:
        """Encode DNA sequence to uint8 numpy array."""
        if not seq or len(seq) == 0:
            return np.zeros(0, dtype=np.uint8)
        arr = np.zeros(len(seq), dtype=np.uint8)
        for i, ch in enumerate(seq):
            arr[i] = self.vocab.get(ch.upper(), 5)  # default to 5 (N) for unknown characters
        return arr
    
    def _get_top_k_cds_sequences_list(self, idx: int) -> list:
        """
        Get the top K longest CDS sequences for a given idx as a list.
        
        Args:
            idx: Index in the dataset
            
        Returns:
            List of top K longest CDS sequence strings (may be shorter than top_k if fewer sequences available)
        """
        self._load_json()
        
        row = self.df.iloc[idx]
        csv_idx = int(row["idx"])
        idx_str = str(csv_idx)
        
        if idx_str not in self._cds_data:
            # This should not happen if data was properly filtered
            return []
        
        cds_list = self._cds_data[idx_str]
        if not cds_list or len(cds_list) == 0:
            # This should not happen if data was properly filtered
            return []
        
        # Extract sequences and sort by length (descending)
        seq_with_length = [(item.get("sequence", ""), len(item.get("sequence", ""))) for item in cds_list]
        seq_with_length = [(seq, length) for seq, length in seq_with_length if seq]  # Filter out empty sequences
        
        if not seq_with_length:
            # This should not happen if data was properly filtered
            return []
        
        # Sort by length (descending) and take top K
        seq_with_length.sort(key=lambda x: x[1], reverse=True)
        top_k_seqs = [seq for seq, _ in seq_with_length[:self.top_k]]
        
        return top_k_seqs
    
    def _get_top_k_cds_sequences(self, idx: int) -> str:
        """
        Get the top K longest CDS sequences for a given idx and concatenate them.
        
        Args:
            idx: Index in the dataset
            
        Returns:
            Concatenated sequence string of top K longest CDS sequences
        """
        top_k_seqs = self._get_top_k_cds_sequences_list(idx)
        return "".join(top_k_seqs)
    
    def get_raw_sequence(self, idx: int) -> str:
        """
        Get the raw concatenated DNA sequence string for a given index.
        
        Returns:
            Raw concatenated DNA sequence as a string (empty string if not found)
        """
        return self._get_top_k_cds_sequences(idx)
    
    def __getitem__(self, idx: int) -> Tuple[List[np.ndarray], int]:
        """
        Get a single sample from the dataset.
        
        Returns:
            Tuple of (encoded_sequences, label_id):
            - encoded_sequences: List of uint8 numpy arrays, each representing one of the top K CDS sequences
            - label_id: integer label ID
        """
        row = self.df.iloc[idx]
        label = str(row[self.label_column])
        y = self.label2id[label]
        
        # 1) Try to load from cache if exists
        if self.cache_dir:
            cache_path = self._cache_path(idx)
            if os.path.exists(cache_path):
                cached_data = np.load(cache_path, allow_pickle=False)
                # Load all sequences from cache
                x_list = []
                for i in range(self.top_k):
                    key = f"seq_{i}"
                    if key in cached_data:
                        x_list.append(cached_data[key])
                if x_list:
                    return x_list, int(y)
        
        # 2) Load CDS sequences from JSON, select top K longest, and encode each separately
        seq_list = self._get_top_k_cds_sequences_list(idx)
        x_list = [self._encode_sequence_to_uint8(seq) for seq in seq_list]
        
        # 3) Cache if cache_dir is provided
        if self.cache_dir:
            cache_path = self._cache_path(idx)
            cache_dict = {f"seq_{i}": x_list[i] for i in range(len(x_list))}
            np.savez(cache_path, **cache_dict)
        
        return x_list, int(y)


# -----------------------------
# Data Splitting Functions
# -----------------------------
def make_group_key(df: pd.DataFrame) -> pd.Series:
    """Create group key for genus-disjoint splitting."""
    g = df["genus"].astype(str).fillna("")
    bad = g.isna() | (g.str.strip() == "") | (g.str.upper() == "NA") | (g.str.upper() == "NAN") | (g == "-")
    key = g.copy()
    key[bad] = "taxid_" + df.loc[bad, "taxid"].astype(str)
    return key


def group_split_2way(df: pd.DataFrame, groups: pd.Series, test_size: float, seed: int):
    """Split dataframe into two groups using GroupShuffleSplit."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_a, idx_b = next(gss.split(df, groups=groups))
    return df.iloc[idx_a].copy(), df.iloc[idx_b].copy()


def assert_disjoint(a: pd.Series, b: pd.Series, name_a: str, name_b: str):
    """Assert that two series have disjoint values."""
    sa = set(a.unique())
    sb = set(b.unique())
    inter = sa & sb
    if inter:
        raise RuntimeError(f"[SPLIT ERROR] groups overlap between {name_a} and {name_b}: {len(inter)}")


def _norm_genus_series(s: pd.Series) -> pd.Series:
    """Normalize genus values: strip, upper NA handling, empty->NA"""
    g = s.astype(str).fillna("")
    g = g.str.strip()
    g = g.replace({"": "NA", "-": "NA"})
    g = g.replace({"nan": "NA", "NaN": "NA", "NAN": "NA"})
    return g


def validate_genus_disjoint(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                           genus_col: str = "genus", group_key_col: str = "group_key"):
    """
    Validate that genus sets are disjoint across splits.
    
    - Check 1: group_key disjoint (this MUST hold, since you used GroupShuffleSplit on group_key)
    - Check 2: REAL genus disjoint (exclude fallback keys like 'taxid_XXXXX' and genus==NA)
    """
    # ---- check group_key disjoint (includes taxid fallback) ----
    assert_disjoint(train_df[group_key_col], val_df[group_key_col], "train", "val")
    assert_disjoint(train_df[group_key_col], test_df[group_key_col], "train", "test")
    assert_disjoint(val_df[group_key_col], test_df[group_key_col], "val", "test")
    
    # ---- extract "real genus" sets (exclude NA and exclude taxid_ fallback) ----
    def real_genus_set(df: pd.DataFrame):
        g = _norm_genus_series(df[genus_col])
        # Exclude missing genus and exclude fallback-style keys
        g = g[(g != "NA") & (~g.str.startswith("taxid_"))]
        return set(g.unique())
    
    tr_g = real_genus_set(train_df)
    va_g = real_genus_set(val_df)
    te_g = real_genus_set(test_df)
    
    inter_tv = tr_g & va_g
    inter_tt = tr_g & te_g
    inter_vt = va_g & te_g
    
    # ---- print a short report ----
    def count_missing(df: pd.DataFrame):
        g = _norm_genus_series(df[genus_col])
        return int((g == "NA").sum())
    
    print("[CHECK] genus summary:")
    print(f"  train: rows={len(train_df)} genus_unique={train_df[genus_col].astype(str).nunique()} "
          f"missing_genus={count_missing(train_df)} real_genus_unique={len(tr_g)}")
    print(f"  val  : rows={len(val_df)} genus_unique={val_df[genus_col].astype(str).nunique()} "
          f"missing_genus={count_missing(val_df)} real_genus_unique={len(va_g)}")
    print(f"  test : rows={len(test_df)} genus_unique={test_df[genus_col].astype(str).nunique()} "
          f"missing_genus={count_missing(test_df)} real_genus_unique={len(te_g)}")
    
    # ---- assert "real genus" disjoint ----
    if inter_tv or inter_tt or inter_vt:
        # show a few overlaps for debugging
        def sample(s):
            return list(sorted(s))[:20]
        raise RuntimeError(
            "[SPLIT ERROR] REAL genus overlap detected!\n"
            f"  train∩val: {len(inter_tv)} (e.g. {sample(inter_tv)})\n"
            f"  train∩test: {len(inter_tt)} (e.g. {sample(inter_tt)})\n"
            f"  val∩test: {len(inter_vt)} (e.g. {sample(inter_vt)})\n"
            "Tip: This usually means your 'genus' column contains inconsistent strings, or you are not splitting by genus-derived key."
        )
    
    print("[CHECK] PASS: genus sets are disjoint across train/val/test (real genus).")
    print("[CHECK] PASS: group_key sets are disjoint across train/val/test.")


def split_dataset_genus_disjoint(
    csv_path: Union[str, pd.DataFrame],
    json_path: str,
    label_column: str = "host_group",
    label_column2: Optional[str] = None,
    genus_column: str = "genus",
    test_size: float = 0.10,
    val_size: float = 0.10,
    seed: int = 42,
    cache_dir: str = "",
    validate: bool = True,
    save_splits: Optional[str] = None,
    vocab: Optional[Dict[str, int]] = None,
    top_k: int = 3
) -> Tuple[CdsClsDataset, CdsClsDataset, CdsClsDataset, Dict[str, int], Optional[Dict[str, int]]]:
    """
    Split dataset into train/val/test with genus-disjoint splitting.
    
    Args:
        csv_path: Path to CSV file or DataFrame
        json_path: Path to JSON file containing CDS sequences
        label_column: Column name for primary labels (default: 'host_group')
        label_column2: Optional second column name for labels (e.g., 'family'). If provided, will create a second label2id.
        genus_column: Column name for genus (default: 'genus')
        test_size: Final proportion for test set in the whole dataset (default: 0.10, i.e., 10%)
        val_size: Final proportion for validation set in the whole dataset (default: 0.10, i.e., 10%)
        seed: Random seed for splitting
        cache_dir: Directory to cache encoded sequences. All sequences will be cached if provided.
        validate: Whether to validate genus disjointness (default: True)
        save_splits: Directory to save split CSVs. If None, splits are not saved.
        vocab: Dictionary mapping nucleotide characters to integers (default: A=1, C=2, G=3, T=4, N=5, 0=PAD)
        top_k: Number of longest CDS sequences to select (default: 3)
        
    Note:
        The splitting process:
        1. First split: (1-test_size-val_size) train, (test_size+val_size) temp 
           (e.g., 80% train, 20% temp when test_size=0.10, val_size=0.10)
        2. Second split: temp is split into val and test 
           (e.g., 20% temp -> 50% val, 50% test = 10% val, 10% test)
        Final result: 8:1:1 (train:val:test) when test_size=0.10, val_size=0.10
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, label2id, label2id2):
        - train_dataset: CdsClsDataset for training (uses label_column)
        - val_dataset: CdsClsDataset for validation (uses label_column)
        - test_dataset: CdsClsDataset for testing (uses label_column)
        - label2id: Dictionary mapping label_column strings to integer IDs
        - label2id2: Dictionary mapping label_column2 strings to integer IDs (None if label_column2 not provided)
    """
    # Load DataFrame
    if isinstance(csv_path, str):
        df = pd.read_csv(csv_path)
    else:
        df = csv_path.copy()
    
    # Validate required columns
    required_cols = [label_column, genus_column, "taxid", "idx"]
    if label_column2:
        required_cols.append(label_column2)
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {list(df.columns)}")
    
    # Filter out samples with no CDS sequences in JSON
    print("[INFO] Filtering samples with no CDS sequences...")
    with open(json_path, 'r') as f:
        cds_data = json.load(f)
    
    def has_valid_cds(idx_val):
        idx_str = str(int(idx_val))
        if idx_str not in cds_data:
            return False
        cds_list = cds_data[idx_str]
        if not cds_list or len(cds_list) == 0:
            return False
        # Check if there's at least one valid sequence
        for item in cds_list:
            seq = item.get("sequence", "")
            if seq and len(seq) > 0:
                return True
        return False
    
    before_count = len(df)
    df = df[df["idx"].apply(has_valid_cds)].copy()
    after_count = len(df)
    filtered_count = before_count - after_count
    if filtered_count > 0:
        print(f"[INFO] Filtered out {filtered_count} samples with no valid CDS sequences ({filtered_count/before_count*100:.2f}%)")
        print(f"[INFO] Remaining samples: {after_count}")
    
    # ---- Genus-disjoint split (8:1:1) ----
    df["group_key"] = make_group_key(df)
    
    # First split: (1-test_size-val_size) train, (test_size+val_size) temp
    temp_size = test_size + val_size
    train_df, temp_df = group_split_2way(df, df["group_key"], test_size=temp_size, seed=seed)
    
    # Second split: split temp into val and test
    # val_size / temp_size is the proportion of val within temp
    val_proportion_in_temp = val_size / temp_size
    val_df, test_df = group_split_2way(temp_df, temp_df["group_key"], test_size=1-val_proportion_in_temp, seed=seed + 1)
    
    if validate:
        validate_genus_disjoint(train_df, val_df, test_df, genus_col=genus_column, group_key_col="group_key")
    
    # Ensure classes exist in train
    labels_train = set(train_df[label_column].astype(str).unique())
    before_classes = len(df[label_column].astype(str).unique())
    train_df = train_df[train_df[label_column].astype(str).isin(labels_train)].copy()
    val_df = val_df[val_df[label_column].astype(str).isin(labels_train)].copy()
    test_df = test_df[test_df[label_column].astype(str).isin(labels_train)].copy()
    
    # Verify all samples still have valid CDS after filtering (should not happen, but check anyway)
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        invalid_samples = []
        for _, row in split_df.iterrows():
            if not has_valid_cds(row["idx"]):
                invalid_samples.append(int(row["idx"]))
        if invalid_samples:
            raise RuntimeError(
                f"[SPLIT ERROR] Found {len(invalid_samples)} samples without valid CDS in {split_name} split "
                f"after filtering. This should not happen. Sample idxs: {invalid_samples[:10]}"
            )
    
    labels = sorted(train_df[label_column].astype(str).unique())
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    
    # Create second label2id if label_column2 is provided
    label2id2 = None
    if label_column2:
        labels2 = sorted(train_df[label_column2].astype(str).unique())
        label2id2 = {lbl: i for i, lbl in enumerate(labels2)}
        print("[INFO] split=Genus-disjoint (group_key=genus else taxid fallback)")
        print("[INFO] sizes: train/val/test =", len(train_df), len(val_df), len(test_df))
        print("[INFO] classes in train:")
        print(f"  {label_column}: {len(labels)} classes (original: {before_classes})")
        print(f"  {label_column2}: {len(labels2)} classes")
    else:
        print("[INFO] split=Genus-disjoint (group_key=genus else taxid fallback)")
        print("[INFO] sizes: train/val/test =", len(train_df), len(val_df), len(test_df))
        print("[INFO] classes in train:", len(labels), "(original:", before_classes, ")")
    
    # Add CDS count column to each split
    def get_cds_count(idx_val):
        """Get the number of CDS sequences for a given idx."""
        idx_str = str(int(idx_val))
        if idx_str not in cds_data:
            return 0
        cds_list = cds_data[idx_str]
        if not cds_list:
            return 0
        # Count valid sequences (non-empty)
        count = 0
        for item in cds_list:
            seq = item.get("sequence", "")
            if seq and len(seq) > 0:
                count += 1
        return count
    
    train_df["cds_count"] = train_df["idx"].apply(get_cds_count)
    val_df["cds_count"] = val_df["idx"].apply(get_cds_count)
    test_df["cds_count"] = test_df["idx"].apply(get_cds_count)
    
    # Save splits if requested
    if save_splits:
        os.makedirs(save_splits, exist_ok=True)
        train_df.to_csv(os.path.join(save_splits, "train.csv"), index=False)
        val_df.to_csv(os.path.join(save_splits, "val.csv"), index=False)
        test_df.to_csv(os.path.join(save_splits, "test.csv"), index=False)
        # Save label2id mapping
        with open(os.path.join(save_splits, "label2id.json"), 'w') as f:
            json.dump(label2id, f, indent=2)
        print("[INFO] wrote splits under:", save_splits)
        print("[INFO] wrote label2id.json under:", save_splits)
        
        # Save second label2id if provided
        if label_column2 and label2id2 is not None:
            label2id2_filename = f"label2id_{label_column2}.json"
            with open(os.path.join(save_splits, label2id2_filename), 'w') as f:
                json.dump(label2id2, f, indent=2)
            print(f"[INFO] wrote {label2id2_filename} under:", save_splits)
    
    # Create datasets
    train_ds = CdsClsDataset(
        csv_path=train_df,
        json_path=json_path,
        label2id=label2id,
        label_column=label_column,
        cache_dir=cache_dir,
        vocab=vocab,
        top_k=top_k
    )
    
    val_ds = CdsClsDataset(
        csv_path=val_df,
        json_path=json_path,
        label2id=label2id,
        label_column=label_column,
        cache_dir=cache_dir,
        vocab=vocab,
        top_k=top_k
    )
    
    test_ds = CdsClsDataset(
        csv_path=test_df,
        json_path=json_path,
        label2id=label2id,
        label_column=label_column,
        cache_dir=cache_dir,
        vocab=vocab,
        top_k=top_k
    )
    
    return train_ds, val_ds, test_ds, label2id, label2id2


# -----------------------------
# Usage Example
# -----------------------------
if __name__ == "__main__":
    import pandas as pd
    from torch.utils.data import DataLoader
    
    # Example: Split dataset into train/val/test with genus-disjoint splitting
    csv_path = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/data/C2_data.csv"
    json_path = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/data/C2_data_cds.json"
    
    # Load dataframe to get label information
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    print(f"Columns: {list(df.columns)}")
    
    # Split dataset into train/val/test with genus-disjoint splitting (8:1:1)
    # First split: 80% train, 20% temp (test_size + val_size = 0.10 + 0.10 = 0.20)
    # Second split: 20% temp -> 10% val, 10% test (val_size=0.10, test_size=0.10)
    save_dir = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/cds_cls"
    train_ds, val_ds, test_ds, label2id, label2id_family = split_dataset_genus_disjoint(
        csv_path=csv_path,
        json_path=json_path,
        label_column="host_group",
        label_column2="family",  # Create second label2id for family
        genus_column="genus",
        test_size=0.10,  # Final test set proportion: 10%
        val_size=0.10,   # Final validation set proportion: 10%
        seed=42,
        cache_dir="",  # Set to "" to disable cache, or provide a path
        validate=True,
        save_splits=save_dir,  # Save splits to data/cds_cls folder
        top_k=3  # Select top 3 longest CDS sequences
    )
    
    print(f"\nTrain dataset size: {len(train_ds)}")
    print(f"Val dataset size: {len(val_ds)}")
    print(f"Test dataset size: {len(test_ds)}")
    print(f"Number of classes (host_group): {len(label2id)}")
    if label2id_family:
        print(f"Number of classes (family): {len(label2id_family)}")
    
    # Test getting samples
    x_train_list, y_train = train_ds[0]
    x_val_list, y_val = val_ds[0]
    x_test_list, y_test = test_ds[0]
    print(f"\nSample test:")
    print(f"  Train sample 0: {len(x_train_list)} sequences, label={y_train}, label_name={list(label2id.keys())[y_train]}")
    for i, seq in enumerate(x_train_list):
        print(f"    Sequence {i}: length={len(seq)}")
    print(f"  Val sample 0: {len(x_val_list)} sequences, label={y_val}, label_name={list(label2id.keys())[y_val]}")
    for i, seq in enumerate(x_val_list):
        print(f"    Sequence {i}: length={len(seq)}")
    print(f"  Test sample 0: {len(x_test_list)} sequences, label={y_test}, label_name={list(label2id.keys())[y_test]}")
    for i, seq in enumerate(x_test_list):
        print(f"    Sequence {i}: length={len(seq)}")
    
    # Test getting raw sequence (concatenated)
    raw_seq_train = train_ds.get_raw_sequence(0)
    print(f"  Train raw sequence (concatenated) length: {len(raw_seq_train)}")
    print(f"  Train raw sequence preview (first 100 chars): {raw_seq_train[:100]}...")
    
    # Example 2: Basic usage with CSV and JSON files (without splitting)
    """
    # Load dataframe to get label information
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    print(f"Columns: {list(df.columns)}")
    
    # Create label2id mapping
    host_groups = sorted(df["host_group"].astype(str).unique())
    label2id = {hg: i for i, hg in enumerate(host_groups)}
    print(f"Number of classes: {len(label2id)}")
    print(f"Classes: {label2id}")
    
    # Create dataset
    dataset = CdsClsDataset(
        csv_path=csv_path,
        json_path=json_path,
        label2id=label2id,
        label_column="host_group",
        cache_dir="",  # Set to "" to disable cache, or provide a path
        top_k=3  # Select top 3 longest CDS sequences
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test getting samples
    x, y = dataset[0]
    print(f"\nSample test:")
    print(f"  Sample 0: length={len(x)}, label={y}, label_name={list(label2id.keys())[y]}")
    
    # Test getting raw sequence
    raw_seq = dataset.get_raw_sequence(0)
    print(f"  Raw sequence length: {len(raw_seq)}")
    print(f"  Raw sequence preview (first 100 chars): {raw_seq[:100]}...")
    """


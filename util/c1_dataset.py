import os
import json
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit


class GenomeClsDataset(Dataset):
    """
    Dataset for sequence classification tasks.
    
    Loads data from CSV file with sequence column.
    Returns full encoded sequences (variable length) as numpy arrays.
    Supports caching to speed up data loading.
    
    Args:
        csv_path: Path to CSV file with columns: idx, taxid, organism_name, family, genus, species, sequence
                 Or a pandas DataFrame with the same structure
        label2id: Dictionary mapping label strings to integer IDs
        label_column: Column name in CSV for labels (default: 'family')
        sequence_column: Column name in CSV for sequences (default: 'sequence')
        cache_dir: Directory to cache encoded sequences. Set "" to disable cache.
                  All sequences will be cached if cache_dir is provided.
        vocab: Dictionary mapping nucleotide characters to integers (default: A=1, C=2, G=3, T=4, N=5, 0=PAD)
    """
    
    VOCAB = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5}  # 0 is PAD
    
    def __init__(
        self,
        csv_path: Union[str, pd.DataFrame],
        label2id: Dict[str, int],
        label_column: str = "family",
        sequence_column: str = "sequence",
        cache_dir: str = "",
        vocab: Optional[Dict[str, int]] = None
    ):
        # Load DataFrame from CSV or use provided DataFrame
        if isinstance(csv_path, str):
            self.df = pd.read_csv(csv_path)
        else:
            self.df = csv_path.copy()
        
        self.df = self.df.reset_index(drop=True)
        self.label2id = label2id
        self.label_column = label_column
        self.sequence_column = sequence_column
        self.cache_dir = cache_dir
        self.vocab = vocab if vocab is not None else self.VOCAB
        
        # Validate required columns
        if self.sequence_column not in self.df.columns:
            raise ValueError(f"Column '{self.sequence_column}' not found in CSV. Available columns: {list(self.df.columns)}")
        if self.label_column not in self.df.columns:
            raise ValueError(f"Column '{self.label_column}' not found in CSV. Available columns: {list(self.df.columns)}")
        
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def _cache_path(self, idx: int) -> str:
        """Generate cache file path for a given index."""
        return os.path.join(self.cache_dir, f"{idx}.npy")
    
    def _encode_sequence_to_uint8(self, seq: str) -> np.ndarray:
        """Encode DNA sequence to uint8 numpy array."""
        if not seq or len(seq) == 0:
            return np.zeros(0, dtype=np.uint8)
        arr = np.zeros(len(seq), dtype=np.uint8)
        for i, ch in enumerate(seq):
            arr[i] = self.vocab.get(ch.upper(), 5)  # default to 5 (N) for unknown characters
        return arr
    
    def get_raw_sequence(self, idx: int) -> str:
        """
        Get the raw DNA sequence string for a given index.
        
        Returns:
            Raw DNA sequence as a string (empty string if not found)
        """
        row = self.df.iloc[idx]
        seq = str(row[self.sequence_column])
        return seq if seq and seq != "nan" else ""
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """
        Get a single sample from the dataset.
        
        Returns:
            Tuple of (encoded_sequence, label_id):
            - encoded_sequence: uint8 numpy array of the full sequence
            - label_id: integer label ID
        """
        row = self.df.iloc[idx]
        label = str(row[self.label_column])
        y = self.label2id[label]
        
        # 1) Try to load from cache if exists
        if self.cache_dir:
            cache_path = self._cache_path(idx)
            if os.path.exists(cache_path):
                x_full = np.load(cache_path, mmap_mode=None)  # uint8
                return x_full, int(y)
        
        # 2) Load sequence from CSV and encode
        seq = self.get_raw_sequence(idx)
        x_full = self._encode_sequence_to_uint8(seq)
        
        # 3) Cache if cache_dir is provided (always cache full length)
        if self.cache_dir:
            np.save(self._cache_path(idx), x_full)
        
        return x_full, int(y)


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
    label_column: str = "family",
    sequence_column: str = "sequence",
    genus_column: str = "genus",
    test_size: float = 0.10,
    val_size: float = 0.10,
    seed: int = 42,
    cache_dir: str = "",
    validate: bool = True,
    save_splits: Optional[str] = None
) -> Tuple[GenomeClsDataset, GenomeClsDataset, GenomeClsDataset, Dict[str, int]]:
    """
    Split dataset into train/val/test with genus-disjoint splitting.
    
    Args:
        csv_path: Path to CSV file or DataFrame
        label_column: Column name for labels (default: 'family')
        sequence_column: Column name for sequences (default: 'sequence')
        genus_column: Column name for genus (default: 'genus')
        test_size: Final proportion for test set in the whole dataset (default: 0.10, i.e., 10%)
        val_size: Final proportion for validation set in the whole dataset (default: 0.10, i.e., 10%)
        seed: Random seed for splitting
        cache_dir: Directory to cache encoded sequences. All sequences will be cached if provided.
        validate: Whether to validate genus disjointness (default: True)
        save_splits: Directory to save split CSVs. If None, splits are not saved.
        
    Note:
        The splitting process:
        1. First split: (1-test_size-val_size) train, (test_size+val_size) temp 
           (e.g., 80% train, 20% temp when test_size=0.10, val_size=0.10)
        2. Second split: temp is split into val and test 
           (e.g., 20% temp -> 50% val, 50% test = 10% val, 10% test)
        Final result: 8:1:1 (train:val:test) when test_size=0.10, val_size=0.10
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, label2id):
        - train_dataset: GenomeClsDataset for training
        - val_dataset: GenomeClsDataset for validation
        - test_dataset: GenomeClsDataset for testing
        - label2id: Dictionary mapping label strings to integer IDs
    """
    # Load DataFrame
    if isinstance(csv_path, str):
        df = pd.read_csv(csv_path)
    else:
        df = csv_path.copy()
    
    # Validate required columns
    required_cols = [label_column, sequence_column, genus_column, "taxid"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {list(df.columns)}")
    
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
    fam_train = set(train_df[label_column].astype(str).unique())
    before_classes = len(df[label_column].astype(str).unique())
    train_df = train_df[train_df[label_column].astype(str).isin(fam_train)].copy()
    val_df = val_df[val_df[label_column].astype(str).isin(fam_train)].copy()
    test_df = test_df[test_df[label_column].astype(str).isin(fam_train)].copy()
    
    families = sorted(train_df[label_column].astype(str).unique())
    label2id = {f: i for i, f in enumerate(families)}
    
    print("[INFO] split=Genus-disjoint (group_key=genus else taxid fallback)")
    print("[INFO] sizes: train/val/test =", len(train_df), len(val_df), len(test_df))
    print("[INFO] classes in train:", len(families), "(original:", before_classes, ")")
    
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
    
    # Create datasets
    train_ds = GenomeClsDataset(
        csv_path=train_df,
        label2id=label2id,
        label_column=label_column,
        sequence_column=sequence_column,
        cache_dir=cache_dir
    )
    
    val_ds = GenomeClsDataset(
        csv_path=val_df,
        label2id=label2id,
        label_column=label_column,
        sequence_column=sequence_column,
        cache_dir=cache_dir
    )
    
    test_ds = GenomeClsDataset(
        csv_path=test_df,
        label2id=label2id,
        label_column=label_column,
        sequence_column=sequence_column,
        cache_dir=cache_dir
    )
    
    return train_ds, val_ds, test_ds, label2id


# -----------------------------
# Usage Example
# -----------------------------
if __name__ == "__main__":
    import pandas as pd
    from torch.utils.data import DataLoader
    
    # Example 1: Basic usage with CSV file
    csv_path = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/data/C1_data.csv"
    
    # Load dataframe to get label information
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    print(f"Columns: {list(df.columns)}")
    
    # Split dataset into train/val/test with genus-disjoint splitting (8:1:1)
    # First split: 80% train, 20% temp (test_size + val_size = 0.10 + 0.10 = 0.20)
    # Second split: 20% temp -> 10% val, 10% test (val_size=0.10, test_size=0.10)
    save_dir = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/genome_cls"
    train_ds, val_ds, test_ds, label2id = split_dataset_genus_disjoint(
        csv_path=csv_path,
        label_column="family",
        sequence_column="sequence",
        genus_column="genus",
        test_size=0.10,  # Final test set proportion: 10%
        val_size=0.10,   # Final validation set proportion: 10%
        seed=42,
        cache_dir="",  # Set to "" to disable cache, or provide a path
        validate=True,
        save_splits=save_dir  # Save splits to data/genome_cls folder
    )
    
    print(f"\nTrain dataset size: {len(train_ds)}")
    print(f"Val dataset size: {len(val_ds)}")
    print(f"Test dataset size: {len(test_ds)}")
    print(f"Number of classes: {len(label2id)}")
    
    # Test getting samples
    x_train, y_train = train_ds[0]
    x_val, y_val = val_ds[0]
    x_test, y_test = test_ds[0]
    print(f"\nSample test:")
    print(f"  Train sample 0: length={len(x_train)}, label={y_train}, label_name={list(label2id.keys())[y_train]}")
    print(f"  Val sample 0: length={len(x_val)}, label={y_val}, label_name={list(label2id.keys())[y_val]}")
    print(f"  Test sample 0: length={len(x_test)}, label={y_test}, label_name={list(label2id.keys())[y_test]}")
    
    # Example 2: Using with DataFrame directly
    """
    df = pd.read_csv(csv_path)
    families = sorted(df["family"].astype(str).unique())
    label2id = {f: i for i, f in enumerate(families)}
    
    dataset = GenomeClsDataset(
        csv_path=df,  # Can pass DataFrame directly
        label2id=label2id,
        label_column="family",
        sequence_column="sequence"
    )
    """
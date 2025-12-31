import os, gzip
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    outdir: str
    tsv: str

    seq_len: int = 4096
    batch_size: int = 64
    num_workers: int = 4

    lr: float = 2e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    seed: int = 42

    device: str = "cuda:1" if torch.cuda.is_available() else "cpu"

    # cache encoded full sequences (uint8). Set "" to disable cache.
    cache_dir: str = ""

    # multi-crop settings
    val_crops: int = 5     # usually 1 (faster)
    test_crops: int = 5    # 5/7/9 are common
    train_random_crop: bool = True

    # to avoid extreme huge viruses exploding cache/disk:
    # if genome length > max_cache_bp, we still load full sequence to crop,
    # but we will NOT cache it. (set 0 to always cache)
    max_cache_bp: int = 200_000


VOCAB = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5}  # 0 PAD


# -----------------------------
# IO & encoding
# -----------------------------
def read_first_fasta_sequence_gz(path: str) -> str:
    seq_lines = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if seq_lines:
                    break
                continue
            seq_lines.append(line.strip())
    return "".join(seq_lines).upper()


def encode_full_seq_to_uint8(seq: str) -> np.ndarray:
    # uint8 saves disk/memory (values 0..5)
    arr = np.zeros(len(seq), dtype=np.uint8)
    for i, ch in enumerate(seq):
        arr[i] = VOCAB.get(ch, 5)
    return arr


def build_genome_path(outdir: str, taxid: str, asm: str, db_source: str) -> str:
    base = "refseq" if db_source == "refseq" else "genbank"
    return os.path.join(outdir, base, "downloads", taxid, asm, f"{asm}_genomic.fna.gz")


# -----------------------------
# Genus-disjoint split
# -----------------------------
def make_group_key(df: pd.DataFrame) -> pd.Series:
    g = df["genus"].astype(str).fillna("")
    bad = g.isna() | (g.str.strip() == "") | (g.str.upper() == "NA") | (g.str.upper() == "NAN") | (g == "-")
    key = g.copy()
    key[bad] = "taxid_" + df.loc[bad, "taxid"].astype(str)
    return key


def group_split_2way(df: pd.DataFrame, groups: pd.Series, test_size: float, seed: int):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_a, idx_b = next(gss.split(df, groups=groups))
    return df.iloc[idx_a].copy(), df.iloc[idx_b].copy()


def assert_disjoint(a: pd.Series, b: pd.Series, name_a: str, name_b: str):
    sa = set(a.unique())
    sb = set(b.unique())
    inter = sa & sb
    if inter:
        raise RuntimeError(f"[SPLIT ERROR] groups overlap between {name_a} and {name_b}: {len(inter)}")


# -----------------------------
# Dataset: returns FULL encoded genome (variable length)
# -----------------------------
class ViralFamilyFullSeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: Config, label2id: dict):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.label2id = label2id
        self.cache_dir = cfg.cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def _cache_path(self, asm: str) -> str:
        return os.path.join(self.cache_dir, f"{asm}.npy")

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        taxid = str(r["taxid"])
        asm = str(r["asm"])
        db = str(r["db_source"])
        y = self.label2id[str(r["family"])]

        # 1) load from cache if exists
        if self.cache_dir:
            cp = self._cache_path(asm)
            if os.path.exists(cp):
                x_full = np.load(cp, mmap_mode=None)  # uint8
                return x_full, int(y)

        # 2) load fasta.gz and encode full
        fna = build_genome_path(self.cfg.outdir, taxid, asm, db)
        if not os.path.exists(fna):
            x_full = np.zeros(0, dtype=np.uint8)
        else:
            seq = read_first_fasta_sequence_gz(fna)
            x_full = encode_full_seq_to_uint8(seq)

        # 3) optionally cache (skip extreme huge genomes if max_cache_bp > 0)
        if self.cache_dir:
            if self.cfg.max_cache_bp <= 0 or len(x_full) <= self.cfg.max_cache_bp:
                np.save(self._cache_path(asm), x_full)

        return x_full, int(y)

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


# -----------------------------
# Collate: crop + pad to fixed L
# -----------------------------
def crop_pad_one(x_full: np.ndarray, L: int, start: int) -> np.ndarray:
    out = np.zeros(L, dtype=np.int64)
    if x_full is None or len(x_full) == 0:
        return out
    n = len(x_full)
    if n >= L:
        out[:] = x_full[start:start + L].astype(np.int64)
    else:
        out[:n] = x_full.astype(np.int64)
    return out


def make_collate_train(L: int, seed: int):
    # numpy RNG per worker is tricky; we do torch RNG for deterministic-ish behavior
    g = torch.Generator()
    g.manual_seed(seed)

    def collate(batch):
        xs, ys = [], []
        for x_full, y in batch:
            n = len(x_full)
            if n >= L:
                # random start
                max_start = n - L
                # use torch.randint for reproducibility with generator
                s = int(torch.randint(low=0, high=max_start + 1, size=(1,), generator=g).item())
            else:
                s = 0
            xs.append(crop_pad_one(x_full, L, s))
            ys.append(y)
        x = torch.tensor(np.stack(xs, axis=0), dtype=torch.long)
        y = torch.tensor(ys, dtype=torch.long)
        return x, y

    return collate


def make_collate_eval_single(L: int, mode: str = "center"):
    # deterministic 1-crop for val (fast)
    def collate(batch):
        xs, ys = [], []
        for x_full, y in batch:
            n = len(x_full)
            if n >= L:
                max_start = n - L
                if mode == "start":
                    s = 0
                elif mode == "end":
                    s = max_start
                else:
                    s = max_start // 2
            else:
                s = 0
            xs.append(crop_pad_one(x_full, L, s))
            ys.append(y)
        x = torch.tensor(np.stack(xs, axis=0), dtype=torch.long)
        y = torch.tensor(ys, dtype=torch.long)
        return x, y

    return collate


# -----------------------------
# Model
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, vocab_size: int = 6, emb_dim: int = 32):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(emb_dim, 128, 7, padding=3), nn.ReLU(),
            nn.Conv1d(128, 128, 7, padding=3), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.cls = nn.Sequential(nn.Dropout(0.2), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.emb(x).transpose(1, 2)      # (B,C,L)
        x = self.conv(x)                     # (B,256,L')
        x = torch.amax(x, dim=-1)            # (B,256)
        return self.cls(x)


# -----------------------------
# Train / Eval loops
# -----------------------------
@torch.no_grad()
def eval_singlecrop(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    ys, ps = [], []
    total_loss = 0.0

    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = loss_fn(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = torch.argmax(logits, dim=-1)
        ys.append(y.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())

    ys = np.concatenate(ys)
    ps = np.concatenate(ps)
    avg_loss = total_loss / len(ys)
    return avg_loss, accuracy_score(ys, ps), f1_score(ys, ps, average="macro")


def train_one_epoch(model, loader, optim, device):
    model.train(True)
    loss_fn = nn.CrossEntropyLoss()
    ys, ps = [], []
    total_loss = 0.0

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = loss_fn(logits, y)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        total_loss += loss.item() * x.size(0)
        pred = torch.argmax(logits, dim=-1)
        ys.append(y.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())

    ys = np.concatenate(ys)
    ps = np.concatenate(ps)
    avg_loss = total_loss / len(ys)
    return avg_loss, accuracy_score(ys, ps), f1_score(ys, ps, average="macro")


@torch.no_grad()
def evaluate_multicrop(model, dataset, device, L: int, K: int, batch_size: int, num_workers: int):
    """
    Multi-crop test:
      for each sample, take K evenly-spaced crops (if genome >= L), average logits across crops.
    dataset should return (x_full:uint8 array, y:int)
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda b: b  # keep as list of (x_full,y)
    )

    all_y, all_p = [], []
    total_loss = 0.0
    total_n = 0

    for batch in tqdm(loader, desc=f"test-multicrop(K={K})", leave=False):
        x_full_list = [bf[0] for bf in batch]
        y_list = [bf[1] for bf in batch]
        B = len(y_list)

        # build B*K crops
        crops = []
        for x_full in x_full_list:
            n = len(x_full)
            if n >= L:
                max_start = n - L
                if K == 1:
                    starts = [max_start // 2]
                else:
                    # evenly spaced in [0, max_start]
                    starts = np.linspace(0, max_start, K)
                    starts = np.round(starts).astype(int).tolist()
            else:
                starts = [0] * K

            for s in starts:
                crops.append(crop_pad_one(x_full, L, s))

        x = torch.tensor(np.stack(crops, axis=0), dtype=torch.long).to(device, non_blocking=True)
        y = torch.tensor(y_list, dtype=torch.long).to(device, non_blocking=True)

        logits = model(x)  # (B*K, C)
        C = logits.size(-1)
        logits = logits.view(B, K, C).mean(dim=1)  # (B, C)

        total_loss += loss_fn(logits, y).item()
        total_n += B

        pred = torch.argmax(logits, dim=-1)
        all_y.append(y.detach().cpu().numpy())
        all_p.append(pred.detach().cpu().numpy())

    ys = np.concatenate(all_y)
    ps = np.concatenate(all_p)
    avg_loss = total_loss / total_n
    return avg_loss, accuracy_score(ys, ps), f1_score(ys, ps, average="macro")


# -----------------------------
# Main
# -----------------------------
def main():
    OUTDIR = os.environ["OUTDIR"]
    TSV = os.environ["TSV"]

    cfg = Config(outdir=OUTDIR, tsv=TSV)

    # cache (optional but recommended)
    cfg.cache_dir = os.path.join(OUTDIR, "merged", "cache_fullseq_uint8_top100")

    df = pd.read_csv(cfg.tsv, sep="\t")

    # ---- Genus-disjoint split (8:1:1) ----
    df["group_key"] = make_group_key(df)

    # First split: 80% train, 20% temp (will be split into val and test)
    train_df, temp_df = group_split_2way(df, df["group_key"], test_size=0.20, seed=cfg.seed)
    # Second split: split temp (20%) into val (10%) and test (10%)
    val_df, test_df = group_split_2way(temp_df, temp_df["group_key"], test_size=0.50, seed=cfg.seed + 1)

    validate_genus_disjoint(train_df, val_df, test_df, genus_col="genus", group_key_col="group_key")


    # ensure classes exist in train
    fam_train = set(train_df["family"].astype(str).unique())
    before_classes = len(df["family"].astype(str).unique())
    train_df = train_df[train_df["family"].astype(str).isin(fam_train)].copy()
    val_df   = val_df[val_df["family"].astype(str).isin(fam_train)].copy()
    test_df  = test_df[test_df["family"].astype(str).isin(fam_train)].copy()

    families = sorted(train_df["family"].astype(str).unique())
    label2id = {f: i for i, f in enumerate(families)}

    print("[INFO] split=Genus-disjoint (group_key=genus else taxid fallback)")
    print("[INFO] sizes: train/val/test =", len(train_df), len(val_df), len(test_df))
    print("[INFO] classes in train:", len(families), "(original:", before_classes, ")")
    print("[INFO] device:", cfg.device)
    print("[INFO] train_random_crop:", cfg.train_random_crop, "| val_crops:", cfg.val_crops, "| test_crops:", cfg.test_crops)
    print("[INFO] cache_dir:", cfg.cache_dir, "| max_cache_bp:", cfg.max_cache_bp)

    # write split manifests
    split_dir = os.path.join(os.path.dirname(cfg.tsv), "splits_genus_disjoint")
    os.makedirs(split_dir, exist_ok=True)
    train_df.to_csv(os.path.join(split_dir, "train.tsv"), sep="\t", index=False)
    val_df.to_csv(os.path.join(split_dir, "val.tsv"), sep="\t", index=False)
    test_df.to_csv(os.path.join(split_dir, "test.tsv"), sep="\t", index=False)
    print("[INFO] wrote splits under:", split_dir)

    # datasets
    train_ds = ViralFamilyFullSeqDataset(train_df, cfg, label2id)
    val_ds   = ViralFamilyFullSeqDataset(val_df, cfg, label2id)
    test_ds  = ViralFamilyFullSeqDataset(test_df, cfg, label2id)

    # loaders
    if cfg.train_random_crop:
        train_collate = make_collate_train(cfg.seq_len, cfg.seed)
    else:
        train_collate = make_collate_eval_single(cfg.seq_len, mode="center")

    val_collate = make_collate_eval_single(cfg.seq_len, mode="center")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=train_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=val_collate
    )

    # model
    model = SimpleCNN(num_classes=len(families)).to(cfg.device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best = -1.0
    best_path = os.path.join(os.path.dirname(cfg.tsv), "cnn_top100_best.genus_disjoint.randcrop.pt")

    for ep in range(1, cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, optim, cfg.device)

        # val: single crop (fast) or multi-crop if you set cfg.val_crops > 1
        if cfg.val_crops <= 1:
            va = eval_singlecrop(model, val_loader, cfg.device)
        else:
            va = evaluate_multicrop(
                model, val_ds, cfg.device, cfg.seq_len, cfg.val_crops,
                batch_size=cfg.batch_size, num_workers=cfg.num_workers
            )

        print(f"[E{ep:02d}] train loss={tr[0]:.4f} acc={tr[1]:.4f} f1m={tr[2]:.4f} | "
              f"val loss={va[0]:.4f} acc={va[1]:.4f} f1m={va[2]:.4f}")

        if va[2] > best:
            best = va[2]
            torch.save({"model": model.state_dict(), "label2id": label2id, "cfg": cfg.__dict__}, best_path)

    # test: multi-crop
    ckpt = torch.load(best_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model"])

    te = evaluate_multicrop(
        model, test_ds, cfg.device, cfg.seq_len, cfg.test_crops,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers
    )
    print(f"[TEST-multicrop] loss={te[0]:.4f} acc={te[1]:.4f} f1m={te[2]:.4f}")
    print("[INFO] best ckpt:", best_path)


if __name__ == "__main__":
    main()

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
    cache_dir: str = ""   # e.g. "/path/cache_seq4096"


VOCAB = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5}  # 0 PAD


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


def encode_seq(seq: str, L: int) -> np.ndarray:
    arr = np.zeros(L, dtype=np.int64)
    seq = seq[:L]
    for i, ch in enumerate(seq):
        arr[i] = VOCAB.get(ch, 5)
    return arr


def build_genome_path(outdir: str, taxid: str, asm: str, db_source: str) -> str:
    base = "refseq" if db_source == "refseq" else "genbank"
    return os.path.join(outdir, base, "downloads", taxid, asm, f"{asm}_genomic.fna.gz")


def make_group_key(df: pd.DataFrame) -> pd.Series:
    """
    Genus-disjoint grouping key.
    - If genus is valid: use genus
    - Else fallback to taxid-based pseudo-genus to avoid leakage
    """
    g = df["genus"].astype(str).fillna("")
    bad = g.isna() | (g.str.strip() == "") | (g.str.upper() == "NA") | (g.str.upper() == "NAN") | (g == "-")
    key = g.copy()
    key[bad] = "taxid_" + df.loc[bad, "taxid"].astype(str)
    return key


def group_split_2way(df: pd.DataFrame, groups: pd.Series, test_size: float, seed: int):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_train, idx_test = next(gss.split(df, groups=groups))
    return df.iloc[idx_train].copy(), df.iloc[idx_test].copy()


def assert_disjoint(a: pd.Series, b: pd.Series, name_a: str, name_b: str):
    sa = set(a.unique())
    sb = set(b.unique())
    inter = sa & sb
    if inter:
        raise RuntimeError(f"[SPLIT ERROR] groups overlap between {name_a} and {name_b}: {len(inter)}")


class ViralFamilyDataset(Dataset):
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

        if self.cache_dir:
            cp = self._cache_path(asm)
            if os.path.exists(cp):
                x = np.load(cp)
                return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

        fna = build_genome_path(self.cfg.outdir, taxid, asm, db)
        if not os.path.exists(fna):
            x = np.zeros(self.cfg.seq_len, dtype=np.int64)
        else:
            seq = read_first_fasta_sequence_gz(fna)
            x = encode_seq(seq, self.cfg.seq_len)

        if self.cache_dir:
            np.save(self._cache_path(asm), x)

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


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


def run_epoch(model, loader, optim, device, train: bool):
    model.train(train)
    loss_fn = nn.CrossEntropyLoss()
    ys, ps = [], []
    total_loss = 0.0

    for x, y in tqdm(loader, desc=("train" if train else "eval"), leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = loss_fn(logits, y)

        if train:
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


def main():
    OUTDIR = os.environ["OUTDIR"]
    TSV = os.environ["TSV"]

    cfg = Config(outdir=OUTDIR, tsv=TSV)
    cfg.cache_dir = os.path.join(OUTDIR, "merged", "cache_seq4096_top100")

    df = pd.read_csv(cfg.tsv, sep="\t")

    # ---- Genus-disjoint split ----
    df["group_key"] = make_group_key(df)

    # 1) split out test by group
    train_all, test_df = group_split_2way(df, df["group_key"], test_size=0.10, seed=cfg.seed)
    # 2) split train_all into train/val by group
    train_df, val_df = group_split_2way(train_all, train_all["group_key"], test_size=0.10, seed=cfg.seed + 1)

    # safety check: no group overlap
    assert_disjoint(train_df["group_key"], val_df["group_key"], "train", "val")
    assert_disjoint(train_df["group_key"], test_df["group_key"], "train", "test")
    assert_disjoint(val_df["group_key"], test_df["group_key"], "val", "test")

    # optional: drop families that are absent from train (otherwise the class exists but model never sees it)
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

    # write split manifests for reproducibility
    split_dir = os.path.join(os.path.dirname(cfg.tsv), "splits_genus_disjoint")
    os.makedirs(split_dir, exist_ok=True)
    train_df.to_csv(os.path.join(split_dir, "train.tsv"), sep="\t", index=False)
    val_df.to_csv(os.path.join(split_dir, "val.tsv"), sep="\t", index=False)
    test_df.to_csv(os.path.join(split_dir, "test.tsv"), sep="\t", index=False)
    print("[INFO] wrote splits under:", split_dir)

    train_ds = ViralFamilyDataset(train_df, cfg, label2id)
    val_ds   = ViralFamilyDataset(val_df, cfg, label2id)
    test_ds  = ViralFamilyDataset(test_df, cfg, label2id)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)

    model = SimpleCNN(num_classes=len(families)).to(cfg.device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best = -1.0
    best_path = os.path.join(os.path.dirname(cfg.tsv), "cnn_top100_best.genus_disjoint.pt")

    for ep in range(1, cfg.epochs + 1):
        tr = run_epoch(model, train_loader, optim, cfg.device, True)
        va = run_epoch(model, val_loader, optim, cfg.device, False)
        print(f"[E{ep:02d}] train loss={tr[0]:.4f} acc={tr[1]:.4f} f1m={tr[2]:.4f} | "
              f"val loss={va[0]:.4f} acc={va[1]:.4f} f1m={va[2]:.4f}")

        if va[2] > best:
            best = va[2]
            torch.save({"model": model.state_dict(), "label2id": label2id, "cfg": cfg.__dict__}, best_path)

    ckpt = torch.load(best_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model"])
    te = run_epoch(model, test_loader, optim, cfg.device, False)
    print(f"[TEST] loss={te[0]:.4f} acc={te[1]:.4f} f1m={te[2]:.4f}")
    print("[INFO] best ckpt:", best_path)


if __name__ == "__main__":
    main()

from torch.utils.data import DataLoader
from viral_cds_dataset import ViralCDSFastaDataset, simple_collate_fn

tsv_path = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/ncbi_viral/merged/rep.family_top100.with_header.with_host.with_group.corrected.cleaned_host.tsv"

# 这个目录下面要有 genbank/ 和 refseq/ 两个子目录（与你截图一致）
downloads_root = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/ncbi_viral"

cache_dir = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/cache_cds_fasta"

ds = ViralCDSFastaDataset(
    tsv_path=tsv_path,
    downloads_root=downloads_root,
    cache_dir=cache_dir,
    rebuild_cache=False,
    prefer_fixed_group=True,
    min_len=60,
)

print("N CDS:", len(ds), "groups:", ds.group2id)

loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=2, collate_fn=simple_collate_fn)
batch = next(iter(loader))
print(batch["labels"][:5], batch["meta"][0])

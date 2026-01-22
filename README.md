1、首先从NCBI上下载所有的taxid,共计273974个
# 下载taxonomy
mkdir -p taxonomy && cd taxonomy
wget -c https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz
tar -xzf taxdump.tar.gz
# 生成 Viruses(10239) 全部后代 taxid
taxonkit list --data-dir . --ids 10239 \
  | awk 'NF{print $1}' \
  | sort -n > taxids.txt
# 获取病毒的family/genus
# taxids.txt: 一行一个 taxid
# 确认文件在当前目录
ls -lh names.dmp nodes.dmp delnodes.dmp merged.dmp
cat taxids.txt \
  | taxonkit lineage --data-dir . -i 1 -n -r \
  | taxonkit reformat2 --data-dir . -I 1 -f "{f}|{g}" \
  > taxid_to_family_genus.tsv
# 添加时间
python3 export_taxid_info.py \
  --taxids taxids.txt \
  --taxdir taxonomy \
  --out taxid_info.csv
python3 export_taxid_dates_from_data_report.py \
  --taxids taxids.txt \
  --data_report /inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/v10239_meta/ncbi_dataset/data/data_report.jsonl \
  --out taxid_release_update.csv \
  --match_mode any_lineage
# 统计缺失的情况
python3 stats_missing_taxid_info.py --csv taxid_info.csv --out missing_stats.csv
2、进行初步过滤：1、保留包含 登记时间、登记科、登记属 这三个信息的 taxid 204603个
# 筛选脚本
python3 filter_taxid_info.py --in taxid_info.csv --out taxid_info.filtered.csv
3、通过taxid进行下载
# 未过滤直接下载
bash download_from_taxids.sh
# 从过滤后的进行下载（√）
bash download_from_filtered_csv.sh taxid_info.filtered.csv downloads 16
# 检查哪些没有被下载
python3 check_missing_downloads.py \
  --csv taxid_info.filtered.csv \
  --downloads downloads \
  --refseq_asm assembly_summary.refseq_viral.txt \
  --genbank_asm assembly_summary.genbank_viral.txt
# 对确实文件直接重新下载
python3 check_and_redownload_missing.py \
  --csv taxid_info.filtered.csv \
  --downloads downloads \
  --refseq_asm assembly_summary.refseq_viral.txt \
  --genbank_asm assembly_summary.genbank_viral.txt \
  --jobs 16 \
  --run
# 统计所有的taxid是否包含至少一个的assembly_summary
python3 stat_taxid_has_assembly.py \
  --csv taxid_info.filtered.csv \
  --downloads downloads \
  --refseq_asm assembly_summary.refseq_viral.txt \
  --genbank_asm assembly_summary.genbank_viral.txt \
  --out_csv taxid_download_presence.csv \
  --require_file "{assembly}_genomic.fna.gz"

===== SUMMARY (taxid-level) =====
Total taxids: 204,603
Has any assembly in summary: 67,749 (33.11%)
Downloaded any assembly locally: 67,749 (33.11%)
Has assembly but NOT downloaded: 0 (0.00%)
Downloaded but NO assembly in summary: 0 (0.00%)

----- By DB (taxid-level) -----
Taxids with RefSeq assemblies in summary: 9,830 (4.80%)
Taxids with GenBank assemblies in summary: 67,396 (32.94%)
Taxids downloaded from RefSeq locally: 9,830 (4.80%)
Taxids downloaded from GenBank locally: 67,396 (32.94%)

# 导出一个初级预处理好的文档

python3 export_taxids_with_local_assembly_csv.py \
  --in_csv taxid_info.filtered.csv \
  --downloads downloads \
  --out_csv taxid_info.assembly_downloaded.csv \
  --require_file "{assembly}_genomic.fna.gz"

# 每个组装的内容介绍

1) _genomic.fna.gz

内容：该组装的基因组核酸序列（DNA/RNA 的 A/C/G/T(U) 序列）

格式：FASTA（.fna = nucleic acid FASTA），gzip 压缩

用途：做基因组序列分析、建库、比对、k-mer、模型训练等最常用的主文件

2) _cds_from_genomic.fna.gz

内容：从基因组注释中提取的CDS（Coding Sequence，编码区）核酸序列
也就是每个基因的编码区序列，通常不包含内含子（病毒一般也少）

格式：FASTA（核酸），gzip 压缩

用途：只用编码区做下游分析（基因预测、CDS 分类、蛋白翻译前的核酸层面任务等）

注意：是否存在取决于该组装是否有结构化注释（有些 assembly 会缺这个文件）

3) _protein.faa.gz

内容：由注释得到的蛋白质序列（把 CDS 翻译后的 amino acids）

格式：FASTA（.faa = amino acid FASTA），gzip 压缩

用途：蛋白比对、同源搜索、功能注释、蛋白家族分析、做蛋白模型输入等

注意：同样依赖注释质量，部分 assembly 可能没有

4) _genomic.gbff.gz

内容：该组装的 GenBank Flat File（“带注释的序列文件”）

格式：GenBank flatfile（.gbff），gzip 压缩

包含信息：

序列本身

feature 注释：CDS、gene、tRNA、rRNA、misc_feature 等

每个 feature 的坐标、product、protein_id、dbxref、note…

用途：想要“序列 + 注释”一起用时最方便；很多解析工具（BioPython）直接读 gbff

5) _genomic.gff.gz

内容：该组装注释的 GFF3 格式（General Feature Format）

格式：GFF（常见是 GFF3），gzip 压缩

包含信息：以“表格行”的形式列出 feature（gene/CDS 等）的坐标与属性

用途：基因组浏览器（IGV/JBrowse）、注释坐标处理、和 BED/gtf/gff 工具链对接

注意：很多病毒组装可能不提供 gff（所以你之前会遇到 404）

# 进行组装的选择
一、总体策略（两层选择）

先库选择：如果该 taxid 在 RefSeq 有 assembly → 只在 RefSeq 候选里选 1 个；否则去 GenBank 候选里选 1 个。

库内排序：对候选 assembly 做打分/排序，取第一。

二、库内如何选（推荐的排序规则）

你可以按下面优先级从上到下比较，直到分出胜负。

规则 1：优先“参考/代表性”组装（如果能拿到）

在 assembly_summary.txt 里有一些字段可以标识（不同版本列名略有差异）：

refseq_category：reference genome / representative genome / na

assembly_level：Complete Genome > Chromosome > Scaffold > Contig

如果能拿到：

reference genome > representative genome > 其他

同类再比 assembly_level

病毒这里很多是 Complete Genome，所以这条很有效。

规则 2：优先“Complete Genome”

同上：assembly_level
推荐排序：

Complete Genome

Chromosome

Scaffold

Contig

规则 3：优先“最新版本/最新更新”

看 assembly_summary 里的日期字段（常见有）：

seq_rel_date（序列发布日期）

asm_update_date（组装更新日期）

有时还有 submission_date

一般：

先比 asm_update_date（越新越好）

再比 seq_rel_date

规则 4：优先“更高质量/更规范的注释可用性”

你现在关心的下游任务是 genome/CDS/protein，所以可以用“文件是否齐全”作为 proxy：

有 *_genomic.gbff.gz（有注释）

有 *_cds_from_genomic.fna.gz

有 *_protein.faa.gz

推荐打分（示例）：

gbff +2

cds +1

protein +1

gff +0.5（可选）

规则 5：优先“RefSeq accession（GCF）高版本号”

当以上都相同，你可以用 accession 作为最终 tie-break：

GCF_XXXXXX.2 > GCF_XXXXXX.1

或同一个 base accession，取版本更高的
# 筛选
python3 select_and_collate_best_assembly.py \
  --taxid_csv taxid_info.assembly_downloaded.csv \
  --downloads_root downloads \
  --refseq_asm assembly_summary.refseq_viral.txt \
  --genbank_asm assembly_summary.genbank_viral.txt \
  --out_collate /inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/all_viral/downloads/collate
# 添加宿主列

add_host_to_manifest.py
给 taxid_best_assembly_manifest.csv 新增一列 host（危害宿主）。它会从你的 data_report.jsonl(.gz) 里提取“宿主/host”信息，并按 taxid 聚合（默认取出现频次最高的宿主；也可输出所有宿主去重拼接）。
python3 add_host_to_manifest.py \
  --manifest taxid_best_assembly_manifest.csv \
  --data_report /inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/GeneShield/data/v10239_meta/ncbi_dataset/data/data_report.jsonl \
  --out taxid_best_assembly_manifest.with_host.csv \
  --mode top1 \
  --match_mode any_lineage

python3 add_host_from_gbff.py \
  --manifest taxid_best_assembly_manifest.csv \
  --out taxid_best_assembly_manifest.with_host.csv

# 去除Host为空、去除order为空的行
# 去除重复的行

共计61410个样本

col	label	nunique	note
0	host	宿主 (Host)	8170

1	kingdom	界 (Kingdom)	34
2	phylum	门 (Phylum)	61
3	class	纲 (Class)	312
4	order	目 (Order)	493
5	family	科 (Family)	344

6	genus	属 (Genus)	2558	
7	species	种 (Species)	16460	

# 划分数据集
C1：分类 界门纲目科
按属不相交划分
total rows: 61410
train rows= 49128 val rows= 6141 test rows= 6141
train ratio= 0.8 val ratio= 0.1 test ratio= 0.1
unique_genus train/val/test = 651 655 656

按时间进行划分

base: data rows= 61410
cutoff 80% month: 2017-09 cutoff 90% month: 2019-12
rows train/val/test = 49216 6154 6040
ratio train/val/test = 0.8014 0.1002 0.0984

移除
val 未见类别：kingdom 3 / phylum 3 / class 23 / order 49 / family 24
test 未见类别：kingdom 2 / phylum 6 / class 18 / order 39 / family 67
对应的行数分别是：
val 行数：7 / 7 / 24 / 74 / 117
test 行数：9 / 33 / 62 / 102 / 342

C2：危害host进行划分
属划分：
total rows: 53103
train rows= 42483 val rows= 5310 test rows= 5310
train ratio= 0.8 val ratio= 0.1 test ratio= 0.1
unique_genus train/val/test = 655 650 652
时间划分：
total rows: 53103
cutoff 80% month: 2017-11 cutoff 90% month: 2020-06
rows train/val/test = 42926 4921 5256
ratio train/val/test = 0.8084 0.0927 0.099
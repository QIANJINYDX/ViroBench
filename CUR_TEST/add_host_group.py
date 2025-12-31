#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import pandas as pd


def norm(s: str) -> str:
    """Normalize host string for matching."""
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def pick_host_column(df: pd.DataFrame, user_col: str | None = None) -> str:
    if user_col:
        if user_col not in df.columns:
            raise ValueError(f"--host_col '{user_col}' not found. Available cols: {list(df.columns)}")
        return user_col

    # common candidates
    cand = []
    for c in df.columns:
        cl = c.lower()
        if "host" in cl:
            cand.append(c)
    if cand:
        # prefer exact "host" then others
        for preferred in ["host", "host_name", "host_taxon", "host_organism", "host organism", "isolate_host"]:
            for c in cand:
                if c.lower() == preferred:
                    return c
        return cand[0]

    raise ValueError(
        "No obvious host column found. Please specify --host_col.\n"
        f"Available cols: {list(df.columns)}"
    )


def build_patterns():
    # ---- A: bacteria (species + common clinical/food/env bacteria names)
    bacteria_species = [
        "escherichia coli", "klebsiella pneumoniae", "pseudomonas aeruginosa",
        "salmonella enterica", "staphylococcus aureus", "acinetobacter baumannii",
        "enterobacter cloacae", "shigella flexneri", "shigella sonnei",
        "listeria monocytogenes", "vibrio cholerae", "vibrio parahaemolyticus",
        "vibrio harveyi", "yersinia pestis", "yersinia enterocolitica",
        "campylobacter jejuni", "aeromonas hydrophila", "bacillus subtilis",
        "bacillus cereus", "enterococcus faecalis", "enterococcus faecium",
    ]
    # allow strain notations, e.g., "E. coli K-12 MG1655"
    bacteria_regex = [
        r"\b(e\.?\s*coli)\b",
        r"\bescherichia\s+coli\b",
        r"\bklebsiella\s+pneumoniae\b",
        r"\bpseudomonas\s+aeruginosa\b",
        r"\bsalmonella\s+enterica\b",
        r"\bstaphylococcus\s+aureus\b",
        r"\bacinetobacter\s+baumannii\b",
        r"\benterobacter\s+cloacae\b",
        r"\bshigella\s+(flexneri|sonnei)\b",
        r"\blisteria\s+monocytogenes\b",
        r"\bvibrio\s+(cholerae|parahaemolyticus|harveyi)\b",
        r"\byersinia\s+(pestis|enterocolitica)\b",
        r"\bcampylobacter\s+jejuni\b",
        r"\baeromonas\s+hydrophila\b",
        r"\bbacillus\s+(subtilis|cereus)\b",
        r"\benterococcus\s+(faecalis|faecium)\b",
        # generic signals
        r"\bbacteri(a|um|al)\b",
        r"\bclinical isolate\b",
    ]

    # ---- B: fungi/oomycetes (mycovirus / plant pathogen fungi)
    fungi_oomycetes_regex = [
        r"\berysiphe\s+necator\b",
        r"\bbotrytis\s+cinerea\b",
        r"\bsclerotinia\s+sclerotiorum\b",
        r"\bfusarium\b",
        r"\brhizoctonia\s+solani\b",
        r"\bmagnaporthe\s+oryzae\b",
        r"\bpyricularia\s+oryzae\b",
        r"\bustilaginoidea\s+virens\b",
        # oomycetes
        r"\bplasmopara\s+viticola\b",
        r"\bphytophthora\b",
        r"\boomicete\b",
        # generic fungus signals
        r"\bfung(i|us|al)\b",
        r"\bmycovirus\b",
        r"\byeast\b",
        r"\bmold\b",
    ]

    # ---- C: plants (common crops + generic plant signals)
    plant_regex = [
        r"\bsolanum\s+lycopersicum\b",
        r"\blycopersicon\s+esculentum\b",
        r"\btomato\b",
        r"\bvitis\s+vinifera\b",
        r"\bgrape(vine)?\b",
        r"\bzea\s+mays\b",
        r"\bmaize\b",
        r"\boryza\s+sativa\b",
        r"\brice\b",
        r"\bglycine\s+max\b",
        r"\bsoybean\b",
        r"\btriticum\s+aestivum\b",
        r"\bwheat\b",
        r"\bcotton\b",
        r"\bpapaya\b",
        r"\bwatermelon\b",
        r"\beggplant\b",
        r"\bpepper\b",
        r"\bchilli\b",
        r"\bokra\b",
        # generic plant signals
        r"\bplant\b",
        r"\bleaf\b",
        r"\broot\b",
        r"\bstem\b",
        r"\bseed\b",
        r"\bcultivar\b",
    ]

    # ---- D1: human & primates
    d1_regex = [
        r"\bhomo\s+sapiens\b",
        r"\bhuman\b",
        r"\bpatient\b",
        r"\bwoman\b",
        r"\bman\b",
        r"\bchild\b",
        r"\bclinical\b",
        r"\bmacaca\s+(mulatta|fascicularis)\b",
        r"\bpan\s+troglodytes\b",
        r"\bgorilla\s+gorilla\b",
        r"\bprimate\b",
    ]

    # ---- D2: livestock & companion animals
    d2_regex = [
        r"\bsus\s+scrofa\b",
        r"\bpig\b|\bswine\b|\bporcine\b",
        r"\bbos\s+taurus\b",
        r"\bcattle\b|\bbovine\b",
        r"\bovis\s+aries\b|\bsheep\b",
        r"\bcapra\s+hircus\b|\bgoat\b",
        r"\bcanis\s+lupus\s+familiaris\b|\bdog\b|\bcanine\b",
        r"\bfelis\s+catus\b|\bcat\b|\bfeline\b",
        r"\bgallus\s+gallus\s+domesticus\b|\bchicken\b",
        r"\bduck\b|\banas\b",
        r"\bpoultry\b",
    ]

    # ---- E: vectors & invertebrates (mosquito/tick/sandfly + other invertebrates)
    e_regex = [
        r"\baedes\b", r"\bculex\b", r"\banopheles\b", r"\bmosquito\b",
        r"\bixodes\b", r"\brhipicephalus\b", r"\bhaemaphysalis\b", r"\btick(s)?\b",
        r"\bphlebotomus\b", r"\blutzomyia\b", r"\bsand\s*fly\b|\bsandfly\b",
        r"\bbemisia\s+tabaci\b",
        r"\bnilaparvata\s+lugens\b",
        r"\bsogatella\s+furcifera\b",
        r"\bbactrocera\b",
        r"\bapis\b|\bhoney\s*bee\b|\bbee\b",
        # aquatic invertebrates (coarse)
        r"\bshrimp\b|\bprawn\b|\bcrab\b|\boyster\b|\bclam\b|\bmussel\b|\bshellfish\b",
        r"\bsea\s*urchin\b|\bstarfish\b|\bsea\s*star\b",
        # generic invertebrate signals
        r"\binvertebrate\b",
        r"\barthropod\b",
        r"\binsect\b",
    ]

    # ---- D3: wild vertebrates (bat/rodent/carnivores/marine mammals etc.)
    # Note: bats/rodents are high-value; keep broad patterns.
    d3_regex = [
        # bats
        r"\brhinolophus\b", r"\bminiopterus\b", r"\bpteropus\b",
        r"\btadarida\b", r"\bmolossus\b",
        r"\bchiroptera\b", r"\bfruit\s*bat\b|\bmicrobat\b|\bbat\b",
        # rodents
        r"\brattus\b", r"\bmus\s+musculus\b|\bmouse\b",
        r"\bmarmota\b", r"\bperomyscus\b",
        r"\brodent\b",
        # marine mammals (a few)
        r"\borcinus\s+orca\b", r"\bphoca\s+vitulina\b", r"\btursiops\b",
        r"\bdolphin\b|\bwhale\b|\bseal\b",
        # generic wild animal signals
        r"\bwild\b",
        r"\bferal\b",
    ]

    compiled = {
        "A": [re.compile(p, re.I) for p in bacteria_regex] + [re.compile(re.escape(x), re.I) for x in bacteria_species],
        "B": [re.compile(p, re.I) for p in fungi_oomycetes_regex],
        "C": [re.compile(p, re.I) for p in plant_regex],
        "D1": [re.compile(p, re.I) for p in d1_regex],
        "D2": [re.compile(p, re.I) for p in d2_regex],
        "D3": [re.compile(p, re.I) for p in d3_regex],
        "E": [re.compile(p, re.I) for p in e_regex],
    }

    # priority: D1 > D2 > D3 > E > A > B > C (you can change)
    priority = ["D1", "D2", "D3", "E", "A", "B", "C"]
    return compiled, priority


def assign_group(host: str, compiled, priority):
    h = host or ""
    # quick normalize (keeps original too)
    hn = norm(h)
    if hn == "" or hn in {"na", "n/a", "none", "unknown", "unidentified"}:
        return "UNK"

    # match by priority
    for g in priority:
        for pat in compiled[g]:
            if pat.search(h) or pat.search(hn):
                return g
    return "OTHER"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv", help="Input TSV path")
    ap.add_argument("--host_col", default=None, help="Host column name (optional)")
    ap.add_argument("--out", default=None, help="Output TSV path (optional)")
    ap.add_argument("--new_col", default="host_group", help="New group column name")
    args = ap.parse_args()

    in_path = args.tsv
    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)

    df = pd.read_csv(in_path, sep="\t", dtype=str, keep_default_na=False, na_values=[])
    host_col = pick_host_column(df, args.host_col)

    compiled, priority = build_patterns()
    df[args.new_col] = df[host_col].apply(lambda x: assign_group(x, compiled, priority))

    out_path = args.out
    if out_path is None:
        base, ext = os.path.splitext(in_path)
        out_path = f"{base}.with_group.tsv"

    df.to_csv(out_path, sep="\t", index=False)
    print(f"[OK] host_col = {host_col}")
    print(df[args.new_col].value_counts(dropna=False).to_string())
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()

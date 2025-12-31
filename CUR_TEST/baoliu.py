# =========================
# A. 细菌宿主（Bacteria）
# =========================

A_bacteria_canonical = [
    # Enterobacteriaceae / 肠杆菌相关
    "Escherichia coli",
    "Klebsiella pneumoniae",
    "Enterobacter cloacae",
    "Shigella flexneri",
    "Shigella sonnei",
    "Salmonella enterica",
    "Yersinia pestis",
    "Yersinia enterocolitica",

    # 非发酵菌
    "Pseudomonas aeruginosa",
    "Acinetobacter baumannii",

    # 革兰阳性常见临床/环境
    "Staphylococcus aureus",
    "Enterococcus faecalis",
    "Enterococcus faecium",
    "Bacillus subtilis",
    "Bacillus cereus",
    "Listeria monocytogenes",

    # 食源性/水环境常见
    "Vibrio cholerae",
    "Vibrio parahaemolyticus",
    "Vibrio harveyi",
    "Campylobacter jejuni",
    "Aeromonas hydrophila",
]

# 建议：菌株信息作为“子字段/metadata”保留，不让它把物种拆散
A_bacteria_strain_keywords = [
    # E. coli 常见
    "K-12", "MG1655", "BL21", "DH5", "O157:H7", "CFT073",
    # P. aeruginosa 常见
    "PAO1", "PA14", "ATCC 27853",
    # K. pneumoniae 常见
    "ATCC", "B5055", "clinical isolate", "strain",
]

# 常见拼写/变体（用于 normalize）
A_bacteria_common_misspellings = {
    "Pseduomonas aeruginosa": "Pseudomonas aeruginosa",
    "Pseudomona aeruginosa": "Pseudomonas aeruginosa",
}

# =========================
# B. 真菌/卵菌/植物病原（Fungi / Oomycetes）
# =========================

B_fungi_oomycetes_canonical = [
    # 真菌（Fungi）
    "Erysiphe necator",
    "Botrytis cinerea",
    "Sclerotinia sclerotiorum",
    "Fusarium spp.",              # 你数据里 Fusarium 很多，先按属/复合条目保留
    "Rhizoctonia solani",
    "Magnaporthe oryzae",
    "Pyricularia oryzae",         # 与 Magnaporthe 常互写
    "Ustilaginoidea virens",

    # 卵菌（Oomycetes，常被误写 fungi）
    "Plasmopara viticola",
    "Phytophthora infestans",
    "Phytophthora spp.",
]

B_fungi_oomycetes_alias_patterns = [
    "powdery mildew", "downy mildew", "gray mold", "sclerotinia",
    "Phytophthora sp.", "Fusarium sp.", "fungal", "mycovirus",
]

# =========================
# C. 植物宿主（Plants）
# =========================

C_plants_canonical = [
    "Solanum lycopersicum",   # tomato
    "Vitis vinifera",         # grapevine
    "Zea mays",               # maize
    "Oryza sativa",           # rice
    "Glycine max",            # soybean
    "Triticum aestivum",      # wheat
]

# 常见作物/俗名（用于映射到学名；你可按需要继续扩展）
C_plants_commonname_to_scientific = {
    "tomato": "Solanum lycopersicum",
    "grapevine": "Vitis vinifera",
    "grape": "Vitis vinifera",
    "maize": "Zea mays",
    "corn": "Zea mays",
    "rice": "Oryza sativa",
    "soybean": "Glycine max",
    "wheat": "Triticum aestivum",

    # 你提到的高频俗名（先保留为“植物”类别关键词也行）
    "chilli": "Capsicum spp.",
    "pepper": "Capsicum spp.",
    "okra": "Abelmoschus esculentus",
    "papaya": "Carica papaya",
    "cotton": "Gossypium spp.",
    "eggplant": "Solanum melongena",
    "watermelon": "Citrullus lanatus",
}

# 你数据里常见“旧学名/同义名”映射
C_plants_synonyms = {
    "Lycopersicon esculentum": "Solanum lycopersicum",
    "Solanum lycopersicum (tomato)": "Solanum lycopersicum",
}

# =========================
# D. 脊椎动物宿主（Vertebrates）
# =========================

# D1. 人类与灵长类
D1_primates_canonical = [
    "Homo sapiens",
    "Macaca mulatta",
    "Macaca fascicularis",
    "Pan troglodytes",
    "Gorilla gorilla",
]

# 用于把 “Homo sapiens 42 year old woman” 拆解
D1_human_metadata_keywords = ["year old", "man", "woman", "patient", "male", "female"]

# D2. 家畜/伴侣动物
D2_domestic_canonical = [
    "Sus scrofa",                 # pig/swine/porcine
    "Bos taurus",                 # cattle/bovine
    "Ovis aries",                 # sheep
    "Capra hircus",               # goat
    "Canis lupus familiaris",     # dog
    "Felis catus",                # cat
    "Gallus gallus domesticus",   # chicken
    "Anas platyrhynchos domesticus",  # duck（常见家鸭写法）
]

D2_commonname_to_scientific = {
    "pig": "Sus scrofa", "swine": "Sus scrofa", "porcine": "Sus scrofa",
    "cattle": "Bos taurus", "bovine": "Bos taurus",
    "sheep": "Ovis aries",
    "goat": "Capra hircus",
    "dog": "Canis lupus familiaris",
    "cat": "Felis catus",
    "chicken": "Gallus gallus domesticus",
    "duck": "Anas platyrhynchos domesticus",
}

# D3. 野生动物（重点：蝙蝠、啮齿类、食肉目、海洋哺乳类）
D3_wildlife_key_taxa = [
    # 蝙蝠：先按属/目保留，再细分到种
    "Chiroptera", "Rhinolophus spp.", "Miniopterus spp.", "Pteropus spp.",
    "Tadarida spp.", "Molossus spp.",

    # 啮齿类
    "Rattus norvegicus", "Mus musculus", "Marmota spp.", "Peromyscus spp.",

    # 海洋哺乳类（示例）
    "Orcinus orca", "Phoca vitulina", "Tursiops spp.",
]

D3_wildlife_alias_patterns = [
    "bat", "microbat", "fruit bat",
    "rodent", "rat", "mouse",
    "seal", "dolphin", "orca", "whale",
]

# =========================
# E. 节肢动物媒介/无脊椎宿主（Vectors & Invertebrates）
# =========================

E_vectors_arthropods_canonical = [
    # 蚊
    "Culex spp.", "Aedes spp.",

    # 蜱
    "Ixodes spp.", "Rhipicephalus spp.", "Haemaphysalis spp.",

    # 白蛉/沙蝇
    "Phlebotomus spp.", "Lutzomyia spp.",

    # 其他重要昆虫媒介/农业害虫
    "Bemisia tabaci",
    "Nilaparvata lugens",
    "Sogatella furcifera",
    "Bactrocera spp.",
    "Apis spp.",
]

E_vectors_alias_patterns = [
    "mosquito", "tick", "ticks",
    "sandfly", "sand fly",
    "whitefly", "planthopper",
    "fruit fly", "bee",
]

# 水生/海洋无脊椎（如你数据里很多，可作为“非节肢动物宿主/环境样本关联”保留）
E_aquatic_invertebrates_keywords = [
    "crab", "shrimp", "prawn", "lobster",
    "clam", "oyster", "mussel", "scallop",
    "sea star", "starfish", "sea urchin",
]

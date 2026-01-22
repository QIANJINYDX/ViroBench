import math

def ppl_details_to_bpb(details):
    """
    details: Evo2Model.get_ppl(..., return_details=True) 的返回
             每个元素至少包含: avg_nll_token, token_count, char_count
    返回: 每条样本的 bpb（float），无法计算则为 nan
    """
    out = []
    ln2 = math.log(2.0)

    for d in details:
        avg = d.get("avg_nll_token", float("nan"))
        tok = int(d.get("token_count", 0) or 0)
        ch  = int(d.get("char_count", 0) or 0)

        if ch <= 0 or tok <= 0 or not math.isfinite(avg):
            out.append(float("nan"))
            continue

        total_nll = avg * tok                 # nat
        nll_per_base = total_nll / ch         # nat/base
        bpb = nll_per_base / ln2              # bits/base
        out.append(float(bpb))

    return out


# ===== 示例 =====
details = [
    {"avg_nll_token": 1.1522243023, "token_count": 512, "char_count": 512},
    {"avg_nll_token": 0.0508479675, "token_count": 480, "char_count": 512},
]
print(ppl_details_to_bpb(details))
# 解释：
# 第1条：token_count==char_count -> bpb ≈ avg_nll_token / ln2
# 第2条：token_count < char_count -> 每个 base 的平均 NLL 更小（被“摊薄”）

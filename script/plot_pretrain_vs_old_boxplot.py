#!/usr/bin/env python3
"""
可视化预训练前后性能对比，以箱线图形式展示。
数据格式：每个指标为 mean(std)，共 12 个指标/任务。
输出可编辑 PDF 到 补充材料结果/pretrian_vs_old/。
"""
import os
import re

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "补充材料结果", "pretrian_vs_old")


def parse_mean_std(s: str):
    """解析 '25.40(8.03)' 为 (mean, std)。"""
    m = re.match(r"([\d.]+)\s*\(\s*([\d.]+)\s*\)", s.strip())
    if m:
        return float(m.group(1)), float(m.group(2))
    raise ValueError(f"Cannot parse: {s!r}")


def load_data():
    """预训练前 (hyenadna-large-1m) vs 预训练后 (ViroHyena 各规模) 的 12 个指标。"""
    # 每行：模型名 + 12 个 mean(std)
    # 基线 + ViroHyena 按模型规模从小到大：436k < 1m < 6m < 253m
    raw = [
        "HyenaDNA-Large-1M	25.40(8.03)	18.76(1.70)	53.64(7.84)	5.65(8.99)	26.42(7.40)	16.53(1.53)	35.62(6.77)	16.11(0.00)	47.61(8.91)	33.04(0.00)	41.55(4.36)	3.91(0.00)",
        "ViroHyena-436k	46.66(1.96)	31.95(0.50)	66.39(6.63)	19.85(3.12)	52.97(4.14)	26.64(4.44)	49.12(1.97)	33.60(3.43)	48.09(7.72)	43.03(0.91)	44.80(5.87)	16.73(11.74)",
        "ViroHyena-1m	41.08(2.24)	25.76(1.80)	60.66(3.93)	21.64(0.73)	44.60(4.38)	26.70(2.86)	48.15(0.93)	36.03(2.05)	51.10(0.89)	41.54(1.05)	46.07(2.83)	26.39(1.64)",
        "ViroHyena-6m	49.93(2.13)	31.57(3.32)	72.28(3.94)	24.73(3.22)	61.83(9.32)	33.31(4.13)	50.12(3.48)	42.59(3.47)	63.68(5.78)	50.33(5.10)	46.38(0.90)	25.28(3.37)",
        "ViroHyena-253m	52.35(3.36)	36.15(3.45)	60.56(6.22)	30.13(2.90)	55.85(4.38)	32.74(3.94)	44.43(0.89)	40.42(1.82)	68.17(3.94)	56.66(3.52)	40.69(1.88)	31.19(0.65)",
    ]
    names = []
    means_list = []
    stds_list = []
    for line in raw:
        parts = line.strip().split("\t")
        name = parts[0]
        names.append(name)
        means, stds = [], []
        for cell in parts[1:]:
            m, s = parse_mean_std(cell)
            means.append(m)
            stds.append(s)
        means_list.append(means)
        stds_list.append(stds)
    return names, means_list, stds_list


def plot_boxplot(names, means_list, output_dir):
    """绘制两组性能的箱线图并保存为可编辑 PDF。"""
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = "sans-serif"
    n_models = len(names)
    # 画布统一 6x6，与 parallel_coords 一致；不设 data aspect，避免纵轴被压扁
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    data = [np.array(m) for m in means_list]
    bp = ax.boxplot(
        data,
        tick_labels=names,
        patch_artist=True,
        widths=0.5,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=5),
        medianprops=dict(color="black", linewidth=2),
    )
    # 基线灰蓝，ViroHyena 系列由浅到深橙
    colors = ["#7eb8da", "#f7a35c", "#e85d04", "#dc2f02", "#9d0208"][:n_models]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    # Legend: median (line in box), mean (diamond)
    median_line = mlines.Line2D([], [], color="black", linewidth=2, label="Median")
    mean_marker = mlines.Line2D(
        [], [], color="black", marker="D", linestyle="None",
        markerfacecolor="white", markeredgecolor="black", markersize=6, label="Mean"
    )
    ax.legend(handles=[median_line, mean_marker], loc="upper left", framealpha=0.9)
    ax.set_ylabel("Performance", fontsize=11)
    ax.set_title("F1 Performance Comparison: Hyena vs. ViroHyena", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_ylim(0, None)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    # 使绘图区为正方形：x、y 轴物理长度一致
    pos = ax.get_position()
    size = min(pos.width, pos.height)
    ax.set_position([pos.x0 + (pos.width - size) / 2, pos.y0 + (pos.height - size) / 2, size, size])

    out_dir = output_dir
    out_pdf = os.path.join(out_dir, "pretrain_vs_old_boxplot.pdf")
    out_png = os.path.join(out_dir, "pretrain_vs_old_boxplot.png")
    fig.savefig(out_pdf, bbox_inches="tight", transparent=True)
    fig.savefig(out_png, bbox_inches="tight", transparent=True, dpi=330)
    plt.close()
    print(f"[OK] Saved: {out_pdf}, {out_png} (330 dpi)")
    return out_pdf


def main():
    names, means_list, stds_list = load_data()
    print(f"Models: {names}")
    for i, name in enumerate(names):
        print(f"  {name}: mean={np.mean(means_list[i]):.2f}")
    plot_boxplot(names, means_list, OUTPUT_DIR)


if __name__ == "__main__":
    main()

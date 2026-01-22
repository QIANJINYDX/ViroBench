#!/usr/bin/env python3
"""
可视化CNN模型结构
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.cnn import GenomeCNN1D, CNNConfig

try:
    from torchinfo import summary
    HAS_TORCHINFO = True
except ImportError:
    HAS_TORCHINFO = False
    print("Warning: torchinfo not installed. Install with: pip install torchinfo")
    print("Will use basic torch summary instead.\n")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")
    print("Will skip architecture diagram.\n")


def print_model_summary(model, input_size=(4, 512)):
    """打印模型摘要"""
    print("=" * 80)
    print("CNN Model Structure Summary")
    print("=" * 80)
    
    if HAS_TORCHINFO:
        summary(model, input_size=input_size, col_names=["input_size", "output_size", "num_params", "kernel_size"])
    else:
        # 基本统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"\nModel Structure:")
        print(model)
        
        # 打印各层信息
        print("\n" + "-" * 80)
        print("Layer Details:")
        print("-" * 80)
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 叶子节点
                num_params = sum(p.numel() for p in module.parameters())
                if num_params > 0:
                    print(f"{name:50s} | Params: {num_params:>10,}")


def visualize_architecture(cfg, out_dim=100):
    """可视化模型架构图"""
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    y_pos = 11
    box_height = 0.6
    spacing = 0.8
    
    # 标题
    ax.text(5, y_pos + 0.5, 'GenomeCNN1D Architecture', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Input
    y_pos -= spacing
    input_box = mpatches.FancyBboxPatch((3.5, y_pos - box_height/2), 3, box_height,
                                        boxstyle="round,pad=0.1", 
                                        facecolor='lightblue', edgecolor='black', linewidth=1.5)
    ax.add_patch(input_box)
    ax.text(5, y_pos, f'Input\n(B, L)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Embedding
    y_pos -= spacing
    embed_box = mpatches.FancyBboxPatch((3.5, y_pos - box_height/2), 3, box_height,
                                        boxstyle="round,pad=0.1",
                                        facecolor='lightgreen', edgecolor='black', linewidth=1.5)
    ax.add_patch(embed_box)
    ax.text(5, y_pos, f'Embedding\n(vocab={cfg.vocab_size+1}, dim={cfg.embed_dim})', 
            ha='center', va='center', fontsize=9)
    ax.arrow(5, y_pos + box_height/2 + 0.1, 0, -0.2, head_width=0.15, head_length=0.1, fc='black')
    
    # Stem
    y_pos -= spacing
    stem_box = mpatches.FancyBboxPatch((3.5, y_pos - box_height/2), 3, box_height,
                                       boxstyle="round,pad=0.1",
                                       facecolor='lightyellow', edgecolor='black', linewidth=1.5)
    ax.add_patch(stem_box)
    ax.text(5, y_pos, f'Stem\nConv1d(k={cfg.kernel_size}, s=2) + BN + ReLU', 
            ha='center', va='center', fontsize=8)
    ax.arrow(5, y_pos + box_height/2 + 0.1, 0, -0.2, head_width=0.15, head_length=0.1, fc='black')
    
    # Stages
    stage_colors = ['lightcoral', 'lightpink', 'plum']
    for si, (out_ch, nblk) in enumerate(zip(cfg.channels, cfg.blocks_per_stage)):
        y_pos -= spacing
        stride = 1 if si == 0 else 2
        stage_box = mpatches.FancyBboxPatch((3.5, y_pos - box_height/2), 3, box_height,
                                           boxstyle="round,pad=0.1",
                                           facecolor=stage_colors[si % len(stage_colors)], 
                                           edgecolor='black', linewidth=1.5)
        ax.add_patch(stage_box)
        ax.text(5, y_pos, f'Stage {si}\n{out_ch} channels, {nblk} blocks\n(stride={stride})', 
                ha='center', va='center', fontsize=8)
        ax.arrow(5, y_pos + box_height/2 + 0.1, 0, -0.2, head_width=0.15, head_length=0.1, fc='black')
        
        # 显示Residual Block结构
        if si == 0:  # 只在第一个stage显示详细结构
            x_detail = 0.5
            y_detail = y_pos
            detail_box = mpatches.FancyBboxPatch((x_detail - 0.3, y_detail - 0.4), 1.5, 0.8,
                                                boxstyle="round,pad=0.05",
                                                facecolor='white', edgecolor='gray', linewidth=1, linestyle='--')
            ax.add_patch(detail_box)
            ax.text(x_detail, y_detail + 0.2, 'ResidualBlock:', ha='left', va='center', fontsize=7, fontweight='bold')
            ax.text(x_detail, y_detail, 'Norm → ReLU → Conv1d', ha='left', va='center', fontsize=6)
            ax.text(x_detail, y_detail - 0.15, '→ Norm → ReLU → Conv1d', ha='left', va='center', fontsize=6)
            ax.text(x_detail, y_detail - 0.3, '+ Skip Connection', ha='left', va='center', fontsize=6, style='italic')
    
    # Global Pooling
    y_pos -= spacing
    pool_box = mpatches.FancyBboxPatch((3.5, y_pos - box_height/2), 3, box_height,
                                      boxstyle="round,pad=0.1",
                                      facecolor='lightcyan', edgecolor='black', linewidth=1.5)
    ax.add_patch(pool_box)
    ax.text(5, y_pos, f'Global Pooling\n{cfg.global_pool.upper()}', 
            ha='center', va='center', fontsize=9)
    ax.arrow(5, y_pos + box_height/2 + 0.1, 0, -0.2, head_width=0.15, head_length=0.1, fc='black')
    
    # Head
    y_pos -= spacing
    head_box = mpatches.FancyBboxPatch((3.5, y_pos - box_height/2), 3, box_height,
                                      boxstyle="round,pad=0.1",
                                      facecolor='lightsalmon', edgecolor='black', linewidth=1.5)
    ax.add_patch(head_box)
    ax.text(5, y_pos, f'Head\nLinear({cfg.head_hidden}) → ReLU → Dropout → Linear({out_dim})', 
            ha='center', va='center', fontsize=8)
    ax.arrow(5, y_pos + box_height/2 + 0.1, 0, -0.2, head_width=0.15, head_length=0.1, fc='black')
    
    # Output
    y_pos -= spacing
    output_box = mpatches.FancyBboxPatch((3.5, y_pos - box_height/2), 3, box_height,
                                        boxstyle="round,pad=0.1",
                                        facecolor='lightblue', edgecolor='black', linewidth=1.5)
    ax.add_patch(output_box)
    ax.text(5, y_pos, f'Output\n(B, {out_dim})', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 配置信息框
    config_text = f"Config:\n"
    config_text += f"vocab_size: {cfg.vocab_size}\n"
    config_text += f"embed_dim: {cfg.embed_dim}\n"
    config_text += f"channels: {cfg.channels}\n"
    config_text += f"blocks_per_stage: {cfg.blocks_per_stage}\n"
    config_text += f"kernel_size: {cfg.kernel_size}\n"
    config_text += f"norm: {cfg.norm}\n"
    config_text += f"dropout: {cfg.dropout}\n"
    config_text += f"global_pool: {cfg.global_pool}\n"
    config_text += f"head_hidden: {cfg.head_hidden}\n"
    config_text += f"use_se: {cfg.use_se}"
    
    config_box = mpatches.FancyBboxPatch((7.5, 1), 2.3, 4,
                                        boxstyle="round,pad=0.1",
                                        facecolor='wheat', edgecolor='black', linewidth=1.5)
    ax.add_patch(config_box)
    ax.text(8.65, 3, config_text, ha='left', va='center', fontsize=7, family='monospace')
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cnn_architecture.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nArchitecture diagram saved to: {output_path}")
    plt.close()


def print_data_flow(model, input_size=(4, 512)):
    """打印数据流（各层的输出形状）"""
    print("\n" + "=" * 80)
    print("Data Flow Analysis (Input Shape: {})".format(input_size))
    print("=" * 80)
    
    model.eval()
    x = torch.randint(0, 6, input_size)
    
    print(f"\nInput: {tuple(x.shape)}")
    
    # Embedding
    x_emb = model.embedding(x).transpose(1, 2)
    print(f"After Embedding: {tuple(x_emb.shape)}")
    
    # Stem
    x_stem = model.stem(x_emb)
    print(f"After Stem: {tuple(x_stem.shape)}")
    
    # Stages
    x_stages = x_stem
    for si, stage in enumerate(model.stages):
        x_stages = stage(x_stages)
        print(f"After Stage {si}: {tuple(x_stages.shape)}")
    
    # Global Pooling
    x_pool = model.global_pool(x_stages).squeeze(-1)
    print(f"After Global Pooling: {tuple(x_pool.shape)}")
    
    # Head
    x_head = model.head(x_pool)
    print(f"After Head (Output): {tuple(x_head.shape)}")


def main():
    # 使用默认配置
    cfg = CNNConfig(
        vocab_size=5,
        pad_idx=0,
        embed_dim=64,
        channels=(64, 128, 256),
        blocks_per_stage=(2, 2, 2),
        kernel_size=7,
        norm="bn",
        dropout=0.1,
        global_pool="avg",
        head_hidden=256,
        head_dropout=0.3,
        use_se=False,
    )
    
    model = GenomeCNN1D(out_dim=100, cfg=cfg)
    
    # 打印模型摘要
    print_model_summary(model, input_size=(4, 512))
    
    # 打印数据流
    print_data_flow(model, input_size=(4, 512))
    
    # 可视化架构
    if HAS_MATPLOTLIB:
        print("\n" + "=" * 80)
        print("Generating Architecture Diagram...")
        print("=" * 80)
        visualize_architecture(cfg, out_dim=100)
    
    print("\n" + "=" * 80)
    print("Visualization Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()


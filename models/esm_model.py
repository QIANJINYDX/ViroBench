# models/esm_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import sys
from typing import List, Optional, Union
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import numpy as np
from tqdm import tqdm

from .base_model import BaseModel


Pooling = Literal["mean", "max", "cls"]


class ESMModel(BaseModel):
    """
    ESM 适配器（DNA → 蛋白 → ESM）

    - 外部接口：仍然接受 DNA 序列（ATCG）
    - 内部流程：DNA -> 氨基酸序列 -> ESM -> 序列向量

    使用 facebookresearch/esm 提供的预训练蛋白语言模型：
        https://github.com/facebookresearch/esm

    默认通过 esm.pretrained.load_model_and_alphabet_local(model_path) 加载本地 .pt 权重，
    其中 model_path 一般为 "esm2_t33_650M_UR50D.pt" 这类文件。
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        device: Optional[str] = None,
        translation_mode: Literal["first_orf", "fixed_frame"] = "first_orf",
        translation_frame: int = 0,
        stop_at_stop: bool = True,
    ):
        """
        Args:
            model_name: 逻辑名（比如 "ESM2-650M"），仅用于日志
            model_path: 本地 ESM 权重路径，传给 load_model_and_alphabet_local
                       （例如 "/mnt/.../esm2_t33_650M_UR50D.pt"）
            device:     "cuda:0" / "cpu" / None(自动)
            translation_mode:
                - "first_orf": 从第一个 ATG 作为起始密码子开始翻译，遇到终止密码子停止
                - "fixed_frame": 按 translation_frame 指定的阅读框（0/1/2）直接按 3bp 一段翻译
            translation_frame: 在 "fixed_frame" 模式下使用，0/1/2
            stop_at_stop: 翻译时遇到终止密码子是否立刻停止（True）；
                          否则用 "X" 填充该位继续翻译
        """
        super().__init__(model_name, model_path)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        self.translation_mode = translation_mode
        self.translation_frame = int(translation_frame) % 3
        self.stop_at_stop = stop_at_stop

        # ESM 相关对象
        self.model = None
        self.alphabet = None
        self.batch_converter = None

        # 一些辅助属性
        self.model_max_len: Optional[int] = None   # 模型允许的最大长度（含/不含特殊 token 视 ESM 实现而定）

        # 标准遗传密码表（DNA 三联体 → 氨基酸）
        # 只处理 A/C/G/T，其余碱基统一翻译为 "X"
        self._codon_table = {
            # Phe
            "TTT": "F", "TTC": "F",
            # Leu
            "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
            # Ile / Met
            "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
            # Val
            "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
            # Ser
            "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S",
            # Pro
            "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
            # Thr
            "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
            # Ala
            "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
            # Tyr
            "TAT": "Y", "TAC": "Y",
            # His
            "CAT": "H", "CAC": "H",
            # Gln
            "CAA": "Q", "CAG": "Q",
            # Asn
            "AAT": "N", "AAC": "N",
            # Lys
            "AAA": "K", "AAG": "K",
            # Asp
            "GAT": "D", "GAC": "D",
            # Glu
            "GAA": "E", "GAG": "E",
            # Cys
            "TGT": "C", "TGC": "C",
            # Trp
            "TGG": "W",
            # Arg
            "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R", "AGG": "R",
            # Gly
            "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
            # Stop
            "TAA": "*", "TAG": "*", "TGA": "*",
        }

        self._load_model()

    # ---------- 加载 ESM ----------
    def _load_model(self):
        # 尝试添加本地 esm 路径到 sys.path
        # models/esm_model.py 所在目录是 models/
        # 本地 esm 仓库位于 models/esm/
        current_dir = os.path.dirname(os.path.abspath(__file__))
        esm_repo_path = os.path.join(current_dir, "esm")
        if os.path.exists(esm_repo_path) and esm_repo_path not in sys.path:
            sys.path.insert(0, esm_repo_path)

        try:
            import esm  # type: ignore
        except ImportError as e:
            raise ImportError(
                "[ESMModel] 请先安装 ESM 库：pip install fair-esm\n"
                "参考：https://github.com/facebookresearch/esm"
            ) from e

        # 处理 model_path 为目录的情况
        if os.path.isdir(self.model_path):
            # 尝试拼接 model_name + .pt
            candidate = os.path.join(self.model_path, f"{self.model_name}.pt")
            if os.path.exists(candidate):
                self.model_path = candidate

        # 优先认为 model_path 是本地 .pt 文件
        if os.path.exists(self.model_path) and os.path.isfile(self.model_path):
            model, alphabet = esm.pretrained.load_model_and_alphabet_local(self.model_path)
        else:
            # 如果本地找不到，就尝试按 model_name 从 hub 下载（需要网络）
            # 例如 model_name="esm2_t33_650M_UR50D"
            if hasattr(esm.pretrained, self.model_name):
                fn = getattr(esm.pretrained, self.model_name)
                model, alphabet = fn()
            else:
                # 兜底：直接当成 hub 名称
                model, alphabet = esm.pretrained.load_model_and_alphabet_hub(self.model_name)

        self.model = model.to(self.device).eval()
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()

        # ESM 的最大长度信息
        max_positions = getattr(self.model, "max_positions", None)
        if isinstance(max_positions, (tuple, list)):
            max_positions = max_positions[-1]
        self.model_max_len = max_positions or 1022  # 常见 ESM-1b/ESM-2 默认 1022

        print(
            f"[ESMModel] loaded '{self.model_name}' on {self.device}, "
            f"max_len={self.model_max_len}, num_layers={getattr(self.model, 'num_layers', '?')}"
        )

    # ---------- DNA → AA 翻译相关 ----------
    @staticmethod
    def _clean_dna(seq: str) -> str:
        """统一大小写并只保留 A/C/G/T/N，U 替换为 T。"""
        s = seq.strip().upper().replace("U", "T")
        out = []
        for ch in s:
            if ch in ("A", "C", "G", "T", "N"):
                out.append(ch)
        return "".join(out)

    @staticmethod
    def _revcomp(seq: str) -> str:
        """DNA 反向互补。"""
        tbl = str.maketrans(
            "ACGTRYMKBDHVNacgtrymkbdhvn",
            "TGCAYRKMVHDBNtgcayrkmvhdbn",
        )
        return seq.translate(tbl)[::-1]

    def _translate_dna_to_protein(self, dna_seq: str) -> str:
        """
        将 DNA 序列翻译为氨基酸序列（单条序列版）。

        - first_orf:
            从第一个 ATG 起，按 3bp 一段翻译，遇到终止密码子停止；
            若找不到 ATG，则从 0 帧开始。
        - fixed_frame:
            从 translation_frame (0/1/2) 指定的阅读框开始，按 3bp 翻译到末尾。
        """
        dna = self._clean_dna(dna_seq)
        if len(dna) < 3:
            return "X"  # 太短，直接返回一个占位氨基酸

        if self.translation_mode == "first_orf":
            start = dna.find("ATG")
            if start == -1:
                start = 0
            frame = start % 3
            dna = dna[start:]
        else:  # "fixed_frame"
            frame = self.translation_frame
            dna = dna[frame:]

        aa_list: List[str] = []
        for i in range(0, len(dna) - 2, 3):
            codon = dna[i: i + 3]
            if len(codon) < 3:
                break
            aa = self._codon_table.get(codon, "X")
            if aa == "*":  # 终止密码子
                if self.stop_at_stop:
                    break
                else:
                    aa = "X"
            aa_list.append(aa)

        if not aa_list:
            return "X"
        return "".join(aa_list)

    # ---------- PLL 评分 ----------
    def score_sequences(self, sequences: List[str], batch_size: int = 256) -> List[float]:
        """
        对序列进行 PLL (Pseudo-Log-Likelihood) 评分。
        注意：
            1. 这里的 sequences 是 DNA 序列。
            2. 内部先翻译成 Protein (AA)，再对 AA 序列做 masked LM 评分。
            3. 得分表示该氨基酸序列在 ESM 模型下的 likelihood。

        Args:
            sequences: 原始 DNA 序列列表（ATCG）
            batch_size: 每一次前向中同时遮罩的 token 数（不是样本 batch）

        Returns:
            每个序列的平均 log-likelihood 得分（按 AA token 平均）
        """
        if self.model is None or self.alphabet is None or self.batch_converter is None:
            raise RuntimeError("[ESMModel] model not loaded.")

        all_scores: List[float] = []
        # batch_size 用作 mask_batch_size
        mask_batch_size = batch_size

        with torch.no_grad():
            for seq in tqdm(sequences, desc="Scoring sequences (ESM)"):
                # 先翻译成 AA
                aa_seq = self._translate_dna_to_protein(seq)
                score = self._score_single_sequence(aa_seq, mask_batch_size)
                all_scores.append(score)

        return all_scores

    def _score_single_sequence(self, aa_sequence: str, mask_batch_size: int = 256) -> float:
        """
        对单条氨基酸序列进行 PLL 评分。
        """
        if len(aa_sequence) == 0:
            return 0.0

        device = self.device
        model = self.model
        alphabet = self.alphabet
        batch_converter = self.batch_converter

        # 1. 转换为 tokens
        # batch_converter 接受 list of (label, seq)
        data = [("seq1", aa_sequence)]
        _, _, tokens = batch_converter(data)  # [1, T]
        tokens = tokens.to(device)

        # 截断
        if self.model_max_len and tokens.size(1) > self.model_max_len:
            tokens = tokens[:, :self.model_max_len]

        T = tokens.size(1)
        token_ids = tokens[0]  # [T]

        # 2. 找出有效位置（排除 PAD / BOS / EOS）
        # ESM 中 BOS=cls_idx, EOS=eos_idx
        pad_idx = alphabet.padding_idx
        cls_idx = alphabet.cls_idx
        eos_idx = alphabet.eos_idx
        mask_idx = alphabet.mask_idx

        valid_positions: List[int] = []
        token_ids_list = token_ids.tolist()

        for i in range(T):
            tid = token_ids_list[i]
            if tid in (pad_idx, cls_idx, eos_idx):
                continue
            valid_positions.append(i)

        if len(valid_positions) == 0:
            return 0.0

        # 3. 分块遮罩并计算 log-probability
        total_logprob = 0.0
        total_count = 0

        for start_idx in range(0, len(valid_positions), mask_batch_size):
            chunk_positions = valid_positions[start_idx : start_idx + mask_batch_size]
            chunk_size = len(chunk_positions)

            # 复制 tokens 用于遮罩
            masked_tokens = tokens.repeat(chunk_size, 1)  # [B, T]
            true_tokens: List[int] = []

            # 遮罩对应位置
            for b, pos in enumerate(chunk_positions):
                true_token = int(token_ids[pos].item())
                true_tokens.append(true_token)
                masked_tokens[b, pos] = mask_idx

            # 前向
            # ESM 的 forward 返回 dict，其中 "logits" 是 [B, T, V]
            # 注意：对于某些旧版 esm，可能只有 "representations"；但标准 ESM-1b/ESM-2 都有 logits
            out = model(masked_tokens, return_contacts=False)
            
            if "logits" not in out:
                # 如果模型输出不包含 logits，说明可能是不带 LM head 的版本
                # 尝试用 representations 映射（通常只有 MLM 训练过的才有）
                # 这里只能报错或者返回 None
                raise RuntimeError("[ESMModel] Model output does not contain 'logits'. Cannot compute PLL.")
            
            logits = out["logits"]  # [B, T, V]

            # 获取遮罩位置的 logits
            batch_indices = torch.arange(chunk_size, device=device)
            pos_indices = torch.tensor(chunk_positions, device=device, dtype=torch.long)
            
            # [B, V]
            pos_logits = logits[batch_indices, pos_indices, :]

            # log_softmax
            log_probs = torch.log_softmax(pos_logits, dim=-1) # [B, V]

            true_token_ids = torch.tensor(true_tokens, device=device, dtype=torch.long)
            token_log_probs = log_probs[batch_indices, true_token_ids]  # [B]

            total_logprob += token_log_probs.sum().item()
            total_count += chunk_size
            
            del out, logits, pos_logits, log_probs, token_log_probs, true_token_ids, masked_tokens
            if "cuda" in str(device):
                torch.cuda.empty_cache()

        avg_logprob = total_logprob / max(1, total_count)
        return float(avg_logprob)

    # ---------- 对外的简易封装 ----------
    @torch.no_grad()
    def embed_sequences(
        self,
        sequences: List[str],
        batch_size: int = 16,
        pooling: Pooling = "mean",
        average_reverse_complement: bool = False,
        truncation: bool = True,
        return_numpy: bool = True,
    ):
        """
        直接输入 DNA 序列列表，返回每条序列的 ESM 向量。

        Args:
            sequences: DNA 序列列表（ATCG...）
            batch_size: batch 大小
            pooling: 'mean' | 'max' | 'cls'
            average_reverse_complement:
                若为 True，则对正向与反向互补分别翻译+前向，再对两个向量取平均
            truncation:
                序列过长时是否截断到模型最大长度
        """
        return self.get_embedding(
            sequences=sequences,
            layer_name=None,
            batch_size=batch_size,
            pool=pooling,
            layer_index=-1,
            average_reverse_complement=average_reverse_complement,
            truncation=truncation,
            max_length=None,
            return_numpy=return_numpy,
        )

    # ---------- 通用 embedding 接口 ----------
    @torch.no_grad()
    def get_embedding(
        self,
        sequences: Union[str, List[str]],
        layer_name: str = None,              # 为兼容 BaseModel，实际不用
        batch_size: int = 16,
        pool: Pooling = "mean",              # "cls" | "mean" | "max"
        layer_index: int = -1,               # -1 表示最后一层；其它为 0-based：0 -> 第 1 层
        average_reverse_complement: bool = False,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        对外统一接口：输入 DNA 序列，输出每条序列一个向量 (N, H)。

        注意：
        - 内部会先 DNA→氨基酸，再调用 ESM
        - ESM token[0] 是 BOS，真正的第一个残基是 token[1]
        """
        if self.model is None or self.alphabet is None or self.batch_converter is None:
            raise RuntimeError("[ESMModel] model not loaded, please check _load_model.")

        if isinstance(sequences, str):
            seq_list = [sequences]
        else:
            seq_list = list(sequences)

        device = self.device
        model = self.model
        alphabet = self.alphabet
        batch_converter = self.batch_converter

        # 选定要抽取的隐藏层（ESM 用 1~num_layers 表示中间层；0 是 embedding）
        num_layers = getattr(model, "num_layers", None)
        if num_layers is None:
            raise RuntimeError("[ESMModel] model.num_layers not found, cannot select layer.")

        if layer_index is None or layer_index == -1:
            repr_layer = num_layers
        else:
            if not (0 <= layer_index < num_layers):
                raise ValueError(
                    f"[ESMModel] layer_index={layer_index} 超出范围 [0, {num_layers-1}]"
                )
            repr_layer = layer_index + 1  # 0-based -> 1-based

        # 最大长度（token 维度），通常包含 BOS/EOS
        if max_length is None:
            max_len = self.model_max_len or 1022
        else:
            max_len = max_length

        def _aa_forward(aa_batch: List[str]) -> torch.Tensor:
            """
            输入氨基酸序列列表，返回 [B, H] 的序列级向量。
            """
            # 构造 batch_data: (name, seq)
            data = [(f"seq{idx}", seq if len(seq) > 0 else "X")
                    for idx, seq in enumerate(aa_batch)]
            _, _, tokens = batch_converter(data)  # tokens: [B, T]
            tokens = tokens.to(device)

            # 截断过长序列
            if truncation and tokens.size(1) > max_len:
                tokens = tokens[:, :max_len]

            with torch.no_grad():
                out = model(
                    tokens,
                    repr_layers=[repr_layer],
                    return_contacts=False,
                )
            token_reps = out["representations"][repr_layer]  # [B, T, H]

            # 与 tokens 对齐（防止 ESM 内部修改长度）
            if token_reps.size(1) != tokens.size(1):
                T = min(token_reps.size(1), tokens.size(1))
                token_reps = token_reps[:, :T, :]
                tokens = tokens[:, :T]

            # pooling，注意 token 0 是 BOS；通常不参与 mean/max
            pad_idx = alphabet.padding_idx
            cls_idx = alphabet.cls_idx
            eos_idx = alphabet.eos_idx

            if pool == "cls":
                # 直接取 BOS（index=0）对应的 hidden
                seq_emb = token_reps[:, 0, :]      # [B, H]
                return seq_emb

            # 构造有效位点 mask：排除 padding / BOS / EOS
            mask = tokens != pad_idx
            special = (tokens == cls_idx) | (tokens == eos_idx)
            mask = mask & (~special)

            # 防止某些序列掩码全 False 的极端情况
            if mask.sum().item() == 0:
                mask = tokens != pad_idx

            if pool == "mean":
                m = mask.unsqueeze(-1).to(token_reps.dtype)  # [B, T, 1]
                summed = (token_reps * m).sum(dim=1)         # [B, H]
                denom = m.sum(dim=1).clamp_min(1.0)          # [B, 1]
                return summed / denom
            elif pool == "max":
                masked = token_reps.masked_fill(
                    ~mask.unsqueeze(-1),
                    float("-inf"),
                )
                seq_emb, _ = masked.max(dim=1)  # [B, H]
                # 如果整行都是 -inf，则退回到未 mask 的 max
                inf_mask = torch.isinf(seq_emb).any(dim=1)
                if inf_mask.any():
                    seq_emb[inf_mask] = token_reps[inf_mask].max(dim=1).values
                return seq_emb
            else:
                raise ValueError(f"[ESMModel] unknown pool='{pool}'")

        outputs: List[torch.Tensor] = []
        for st in tqdm(range(0, len(seq_list), batch_size), desc="Getting ESM embedding"):
            batch_dna = seq_list[st: st + batch_size]

            # 正向翻译
            aa_f = [self._translate_dna_to_protein(s) for s in batch_dna]
            vec_f = _aa_forward(aa_f)

            if average_reverse_complement:
                # 反向互补再翻译
                rc_dna = [self._revcomp(s) for s in batch_dna]
                aa_r = [self._translate_dna_to_protein(s) for s in rc_dna]
                vec_r = _aa_forward(aa_r)
                vec = 0.5 * (vec_f + vec_r)
            else:
                vec = vec_f

            outputs.append(vec.detach().to(torch.float32).cpu())

        if outputs:
            out = torch.cat(outputs, dim=0)  # [N, H]
        else:
            out = torch.empty(0, 0)

        return out.numpy() if return_numpy else out

if __name__ == "__main__":
    # 用户指定的路径（可能是目录）
    MODEL_PATH = "/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/esm1b_t33_650M_UR50S/esm1b_t33_650M_UR50S.pt"
    # 对应 ESM-1b 模型名
    MODEL_NAME = "esm1b_t33_650M_UR50S"

    m = ESMModel(
        model_name=MODEL_NAME,
        model_path=MODEL_PATH,
        device=None,                 # 自动选 GPU / CPU
        translation_mode="first_orf",
        translation_frame=0,
        stop_at_stop=True,
    )

    dna_list = [
        "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG",
        "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATTCGTTAGC",
        "AACCC"
    ]

    embs = m.embed_sequences(
        sequences=dna_list,
        batch_size=8,
        pooling="mean",
        average_reverse_complement=False,
    )
    print("Embedding shape:", embs.shape)   # (2, hidden_size)

    # 测试 PLL 评分
    print("\nTesting PLL scoring...")
    try:
        scores = m.score_sequences(dna_list, batch_size=16)
        print("Sequence scores:", scores)
    except Exception as e:
        print(f"Scoring failed: {e}")
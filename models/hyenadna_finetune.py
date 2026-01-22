# models/hyenadna_finetune.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

Pooling = Literal["mean", "max", "cls"]

def _revcomp(seq: str) -> str:
    tbl = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(tbl)[::-1]

from transformers import EarlyStoppingCallback

class ValMetricsSaver(transformers.TrainerCallback):
    """
    - 每次验证后（on_evaluate）将 eval_* 指标写入 out_dir/val_metrics.jsonl
    - 如果当前 eval_auroc 为最佳，额外写出 out_dir/best_threshold.json
    """
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.best_auroc = -1.0

    def on_evaluate(self, args, state, control, **kwargs):
        metrics = kwargs.get("metrics", {}) or {}
        path = os.path.join(self.out_dir, "val_metrics.jsonl")
        # 追加记录
        rec = {"epoch": state.epoch}
        rec.update({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # 维护最佳 AUROC，并保存对应阈值
        auroc = float(metrics.get("eval_auroc", -1.0))
        if auroc > self.best_auroc:
            self.best_auroc = auroc
            keep = {k: metrics[k] for k in metrics.keys() if k.startswith("eval_best_")}
            keep["eval_auroc"] = auroc
            keep["epoch"] = state.epoch
            with open(os.path.join(self.out_dir, "best_threshold.json"), "w", encoding="utf-8") as f:
                json.dump(keep, f, indent=2, ensure_ascii=False)

# ========= 适配器：把 PromoterAIDataset -> 单序列监督数据 =========
class PromoterAIForFinetune(Dataset):
    """
    将你的 PromoterAIDataset 适配为“单序列 + 标签”的监督训练集合。
    - 使用变异窗口 mt_seq 作为训练输入
    - 返回 {seq, label, weight?, id?}
    - 可选在线 RC 数据增强
    """
    def __init__(
        self,
        promoter_ds,                          # 你的 PromoterAIDataset 实例
        rc_prob: float = 0.0,                 # 训练期随机反向互补概率
        weight_col: Optional[str] = None,     # 如有样本权重列（在 promoter_ds.df 中）
        id_col: Optional[str] = None,
    ):
        self.ds = promoter_ds
        self.rc_prob = rc_prob
        self.weight_col = weight_col if (weight_col and weight_col in self.ds.df.columns) else None
        self.id_col = id_col if (id_col and id_col in self.ds.df.columns) else None

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        wt, mt, label = self.ds[idx]   # (wt_seq, mt_seq, label)
        seq = mt
        if self.rc_prob > 0 and np.random.rand() < self.rc_prob:
            seq = _revcomp(seq)
        item = {"seq": seq, "labels": int(label)}
        if self.weight_col:
            item["weight"] = float(self.ds.df.iloc[idx][self.weight_col])
        if self.id_col:
            item["id"] = self.ds.df.iloc[idx][self.id_col]
        return item

# ========= Collator：tokenize + 透传样本权重与ID =========
@dataclass
class CollatorWithTokenizer:
    tokenizer: Any
    padding: str = "longest"
    truncation: bool = True
    max_length: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        seqs = [f["seq"] for f in features]
        enc = self.tokenizer(
            seqs, padding=self.padding, truncation=self.truncation,
            max_length=self.max_length, return_tensors=self.return_tensors
        )
        # labels
        if "labels" in features[0]:
            labels = [f["labels"] for f in features]
            enc["labels"] = torch.tensor(labels, dtype=torch.long)
        # sample weight (optional)
        weights = [float(f.get("weight", 1.0)) for f in features]
        enc["sample_weight"] = torch.tensor(weights, dtype=torch.float)
        # ids (optional,仅预测保存时用)
        enc["example_ids"] = [f.get("id", None) for f in features]
        return enc

# ========= 自定义 Trainer：支持样本权重 & 二/多分类 =========
class WeightedTrainer(Trainer):
    def __init__(self, pos_weight: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        sample_weight = inputs.pop("sample_weight", None)
        outputs = model(**{k: v for k, v in inputs.items() if k != "example_ids"})
        logits = outputs.get("logits")

        # 自动适配 head：如果 num_labels == 1 -> BCE；否则 CE
        num_labels = model.config.num_labels
        if num_labels == 1:
            # BCE with logits for binary
            loss_fct = nn.BCEWithLogitsLoss(
                reduction="none",
                pos_weight=(torch.tensor(self.pos_weight, device=logits.device) if self.pos_weight else None)
            )
            loss_vec = loss_fct(logits.view(-1), labels.float().view(-1))
        else:
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            loss_vec = loss_fct(logits.view(-1, num_labels), labels.view(-1).long())

        if sample_weight is not None:
            sample_weight = sample_weight.to(loss_vec.device)
            loss = (loss_vec * sample_weight).mean()
        else:
            loss = loss_vec.mean()

        return (loss, outputs) if return_outputs else loss

# ========= FineTuner：统一加载/Trainer 构建/长序列推理 =========
class HyenaDNAFineTuner:
    """
    - 加载 tokenizer/model（支持 bf16/fp16，device_map="auto"）
    - 冻结骨干/只训分类头（可选）
    - get_trainer(...) 返回自定义 WeightedTrainer
    - predict_logits_longseq(...) 用滑窗聚合超长序列 logits（可选 RC-TTA）
    """
    def __init__(
        self,
        model_path: str,
        hf_home: Optional[str] = None,
        trust_remote_code: bool = True,
        torch_dtype: Optional[torch.dtype] = torch.bfloat16,
        device_map: Optional[str] = "auto",
        num_labels: Optional[int] = None,
        freeze_backbone: bool = False,
        gradient_checkpointing: bool = True,
    ):
        if hf_home: os.environ["HF_HOME"] = hf_home
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)

        common = dict(trust_remote_code=trust_remote_code)
        if torch_dtype is not None: common["torch_dtype"] = torch_dtype
        if device_map is not None: common["device_map"] = device_map

        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, **common)
        if num_labels is not None:
            self.model.config.num_labels = int(num_labels)
        # 自动设定 problem_type
        if self.model.config.num_labels == 1:
            self.model.config.problem_type = "single_label_classification"
        else:
            self.model.config.problem_type = "single_label_classification"

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if freeze_backbone:
            for n, p in self.model.named_parameters():
                if "classifier" not in n and "score" not in n and "lm_head" not in n:
                    p.requires_grad = False

        self.max_position = getattr(self.model.config, "max_position_embeddings", 0) or \
                            getattr(self.tokenizer, "model_max_length", 0)

    def get_trainer(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        output_dir: str,
        lr: float = 2e-5,
        epochs: int = 3,
        per_device_train_batch_size: int = 4,
        per_device_eval_batch_size: int = 8,
        weight_decay: float = 0.05,
        warmup_ratio: float = 0.06,
        gradient_accumulation_steps: int = 1,
        bf16: bool = True,
        fp16: bool = False,
        logging_steps: int = 50,
        eval_strategy: str = "steps",
        eval_steps: int = 200,
        save_strategy: str = "steps",
        save_steps: int = 200,
        load_best_model_at_end: bool = True,
        metric_for_best_model: Optional[str] = "auroc",
        greater_is_better: Optional[bool] = True,
        pos_weight: Optional[float] = None,
        max_length: Optional[int] = None,
        compute_metrics_fn: Optional[Any] = None,
        seed: int = 42,
    ) -> Trainer:
        collator = CollatorWithTokenizer(
            tokenizer=self.tokenizer,
            padding="longest",
            truncation=True,
            max_length=max_length,
        )
        args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=lr,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            num_train_epochs=epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            bf16=bf16,
            fp16=fp16,
            logging_steps=logging_steps,
            evaluation_strategy=eval_strategy if eval_dataset is not None else "no",
            eval_steps=eval_steps if eval_dataset is not None else None,
            save_strategy=save_strategy,
            save_steps=save_steps,
            load_best_model_at_end=load_best_model_at_end if eval_dataset is not None else False,
            metric_for_best_model="eval_auroc",
            greater_is_better=True,
            report_to=["none"],
            seed=seed,
        )
        trainer = WeightedTrainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics_fn,
            pos_weight=pos_weight,
        )
        # 注册回调：保存每个 epoch 的验证结果 + 早停
        
        callbacks = []
        callbacks.append(ValMetricsSaver(output_dir))
        if eval_dataset is not None and early_stopping_patience is not None:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_min_delta
            ))
        trainer.add_callback(callbacks[0])
        if len(callbacks) > 1:
            trainer.add_callback(callbacks[1])

        return trainer

    # 供评估/推理：对超长序列滑窗聚合 logits（可选RC-TTA）
    @torch.no_grad()
    def predict_logits_longseq(
        self,
        seq: str,
        stride: int = 4096,
        max_length: Optional[int] = None,
        rc_tta: bool = False,
        reduction: Literal["mean", "max"] = "mean",
        batch_size: int = 8,
    ) -> np.ndarray:
        device = next(self.model.parameters()).device
        ml = max_length or getattr(self.tokenizer, "model_max_length", 4096)
        chunks = []
        for start in range(0, len(seq), stride):
            frag = seq[start:start+ml]
            if not frag: break
            chunks.append(frag)
            if len(frag) < ml: break

        def _pass(batch_seqs: List[str]) -> torch.Tensor:
            enc = self.tokenizer(batch_seqs, padding=True, truncation=True, max_length=ml, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = self.model(**enc)
            return out.logits

        logits_list = []
        for i in range(0, len(chunks), batch_size):
            logits_list.append(_pass(chunks[i:i+batch_size]))
        logits = torch.cat(logits_list, dim=0)

        if rc_tta:
            rc_chunks = [_revcomp(c) for c in chunks]
            rc_logits_list = []
            for i in range(0, len(rc_chunks), batch_size):
                rc_logits_list.append(_pass(rc_chunks[i:i+batch_size]))
            rc_logits = torch.cat(rc_logits_list, dim=0)
            logits = 0.5 * (logits + rc_logits)

        agg = logits.mean(dim=0) if reduction == "mean" else logits.max(dim=0).values
        return agg.float().cpu().numpy()

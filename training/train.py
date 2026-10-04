"""Finetune a 3-class NLI checkpoint (DeBERTa-v3) on unified pair JSONL.

Starting from an NLI checkpoint and continuing on domain + public pairs.
Output dir holds config.json + model.safetensors + tokenizer.json and loads
directly in raz:  raz ask ... --scorer nli --nli-model <out>
"""

import argparse
import json

import numpy as np
import torch
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

ID2LABEL = {0: "contradiction", 1: "entailment", 2: "neutral"}


def load_pairs(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ece(probs, labels, bins=10):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    err, n = 0.0, len(labels)
    for b in range(bins):
        lo = b / bins
        m = (conf >= lo) & ((conf < lo + 1 / bins) | (lo >= (bins - 1) / bins))
        if m.sum():
            err += abs((pred[m] == labels[m]).mean() - conf[m].mean()) * m.sum() / n
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="append", required=True, help="train pair JSONL (repeatable)")
    ap.add_argument("--eval", required=True, help="eval pair JSONL (disjoint rows!)")
    ap.add_argument("--base", default="cross-encoder/nli-deberta-v3-xsmall")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--resume", action="store_true",
                    help="resume from latest checkpoint in --out (survives kills)")
    ap.add_argument("--save-steps", type=int, default=1000)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base,
        num_labels=3,
        id2label={i: n for i, n in ID2LABEL.items()},
        label2id={n: i for i, n in ID2LABEL.items()},
    )

    def encode(path):
        rows = load_pairs(path)
        ds = Dataset.from_list(rows)
        return ds.map(
            lambda b: tok(b["premise"], b["hypothesis"], truncation=True, max_length=args.max_len),
            batched=True,
            remove_columns=["premise", "hypothesis"],
        ).rename_column("label", "labels")

    train_ds, eval_ds = encode(args.train[0]), encode(args.eval)
    for extra in args.train[1:]:
        from datasets import concatenate_datasets
        train_ds = concatenate_datasets([train_ds, encode(extra)])

    def metrics(p):
        logits, labels = p.predictions, p.label_ids
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        return {
            "accuracy": float((probs.argmax(1) == labels).mean()),
            "ece": float(ece(probs, labels)),
        }

    targs = TrainingArguments(
        output_dir=args.out,
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=100,
        train_sampling_strategy="group_by_length",
        bf16=torch.cuda.is_available(),
        seed=args.seed,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
    )
    collator = DataCollatorWithPadding(tok)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      eval_dataset=eval_ds, compute_metrics=metrics,
                      data_collator=collator)
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    print("final eval:", trainer.evaluate())
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print(f"raz-ready: --scorer nli --nli-model {args.out}")


if __name__ == "__main__":
    main()

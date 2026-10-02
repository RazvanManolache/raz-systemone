"""Listwise soft-CE finetuning (Qyvos-style objective on our 3-class head).

Each row: premise + K options + soft target. Forward all (premise, option)
pairs, take the entail logit per option, log_softmax, soft cross-entropy.
A strictly proper scoring rule: honest probabilities minimize the loss.

Output is a raz-ready dir (safetensors + config + tokenizer), loadable via
--scorer nli --nli-model <dir>, same as train.py.

Example:
  training/.venv/Scripts/python.exe training/train_soft.py \
      --train training/runs/openjev-soft.jsonl --train training/runs/tickets-soft.jsonl \
      --eval training/runs/openjev-val-soft.jsonl --out training/runs/v10 --epochs 1
"""

import argparse
import json
import random

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

ID2LABEL = {0: "contradiction", 1: "entailment", 2: "neutral"}
ENTAIL = 1


class Rows(Dataset):
    def __init__(self, paths):
        self.rows = []
        for p in paths:
            for line in open(p):
                r = json.loads(line)
                k = len(r["options"])
                assert k >= 2 and abs(sum(r["target"][:k]) - 1.0) < 0.05, f"bad row: {line[:120]}"
                self.rows.append(r)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(rows, tok, max_len):
    prems, opts, tgts, kinds = [], [], [], []
    for r in rows:
        for o in r["options"]:
            prems.append(r["premise"])
            opts.append(o)
        tgts.append(r["target"])
        kinds.append(len(r["options"]))
    enc = tok(prems, opts, truncation=True, max_length=max_len, padding=True,
              return_tensors="pt")
    kmax = max(kinds)
    pad = torch.zeros(len(tgts), kmax)
    for j, t in enumerate(tgts):
        pad[j, : len(t)] = torch.tensor(t)
    return enc, pad, kinds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="append", required=True)
    ap.add_argument("--eval", required=True, help="listwise rows (acc + softCE)")
    ap.add_argument("--base", default="cross-encoder/nli-deberta-v3-xsmall")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-rows", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--eval-rows", type=int, default=2000)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=3, id2label=ID2LABEL,
        label2id={v: k for k, v in ID2LABEL.items()}).to(dev)
    train = Rows(args.train)
    evrows = Rows([args.eval]).rows
    random.Random(999).shuffle(evrows)
    evrows = evrows[: args.eval_rows]
    print(f"train rows: {len(train)}, eval rows: {len(evrows)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sch = None
    model.train()
    step = 0
    for ep in range(args.epochs):
        idx = list(range(len(train)))
        random.Random(args.seed + ep).shuffle(idx)
        tot = n = 0.0
        for i in range(0, len(idx), args.batch_rows):
            rows = [train[j] for j in idx[i:i + args.batch_rows]]
            enc, tgt, kinds = collate(rows, tok, args.max_len)
            enc = {k: v.to(dev) for k, v in enc.items()}
            tgt = tgt.to(dev)
            logits = model(**enc).logits[:, ENTAIL]
            off, losses = 0, []
            for j, k in enumerate(kinds):
                s = logits[off:off + k]
                losses.append(-(tgt[j, :k] * torch.log_softmax(s, -1)).sum())
                off += k
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sch is not None:
                sch.step()
            opt.zero_grad()
            tot += loss.item()
            n += 1
            step += 1
            if step == 1:
                nsteps = args.epochs * ((len(train) + args.batch_rows - 1) // args.batch_rows)
                sch = get_linear_schedule_with_warmup(opt, 100, nsteps)
        print(f"epoch {ep} mean_loss={tot / max(1, n):.4f}", flush=True)

    model.eval()
    ok = ce = nt = 0
    with torch.no_grad():
        for i in range(0, len(evrows), args.batch_rows):
            rows = evrows[i:i + args.batch_rows]
            enc, tgt, kinds = collate(rows, tok, args.max_len)
            enc = {k: v.to(dev) for k, v in enc.items()}
            logits = model(**enc).logits[:, ENTAIL].cpu()
            off = 0
            for j, k in enumerate(kinds):
                s = logits[off:off + k]
                off += k
                p = s.softmax(-1)
                t = tgt[j, :k]
                ok += int(p.argmax()) == int(t.argmax())
                ce += float(-(t * p.log()).sum())
                nt += 1
    print(f"eval: acc={ok / nt:.4f} softCE={ce / nt:.4f} n={nt}")

    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()

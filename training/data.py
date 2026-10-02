"""Converters to unified NLI-pair JSONL for finetuning.

Row format: {"premise": str, "hypothesis": str, "label": 0|1|2}
Label ids: 0 = contradiction, 1 = entailment, 2 = neutral.
(train.py writes this mapping into the model config; systemone reads it back
dynamically, so the order only has to be self-consistent.)
"""

import argparse
import json

LABELS = {"contradiction": 0, "entailment": 1, "neutral": 2}

# Must stay identical to the Rust zero-shot template in src/nli.rs.
CHOICE_TEMPLATE = "This text is about {}."


def tickets_to_pairs(tickets_path):
    """Our labeled tickets -> NLI pairs. Choice options become entail/contra
    hypotheses; score levels map by distance (exact=entail, adjacent=neutral,
    far=contradiction); noul maps directly."""
    rows = []
    with open(tickets_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            state = c["state"]
            if c.get("department"):
                for key, desc in [
                    ("billing", "Payment or subscription issues"),
                    ("technical", "Bugs or integration problems"),
                    ("sales", "Pricing or account questions"),
                ]:
                    hyp = CHOICE_TEMPLATE.format(desc)
                    label = "entailment" if key == c["department"] else "contradiction"
                    rows.append({"premise": state, "hypothesis": hyp, "label": LABELS[label]})
            if c.get("frustration") is not None:
                levels = [
                    "Calm, just stating facts",
                    "Frustrated but civil",
                    "Very angry, strong language",
                ]
                for i, level in enumerate(levels):
                    d = abs(i - c["frustration"])
                    label = "entailment" if d == 0 else ("neutral" if d == 1 else "contradiction")
                    rows.append({"premise": state, "hypothesis": level, "label": LABELS[label]})
            if c.get("is_urgent") is not None:
                label = "entailment" if c["is_urgent"] else "contradiction"
                rows.append(
                    {
                        "premise": state,
                        "hypothesis": "The message conveys urgency or time-sensitivity",
                        "label": LABELS[label],
                    }
                )
    return rows


def hf_nli_to_pairs(dataset_name, split="train", limit=0, offset=0):
    """Public NLI datasets (mnli/snli/anli) -> pairs. Label mapping is by
    NAME (robust to id-order differences); unknown labels are skipped."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    names = [n.lower() for n in ds.features["label"].names]
    rows = []
    skipped = 0
    for ex in ds:
        if skipped < offset:
            skipped += 1
            continue
        if limit and len(rows) >= limit:
            break
        name = names[ex["label"]] if ex["label"] >= 0 else "skip"
        if name not in LABELS:
            continue
        hyp_key = "hypothesis" if "hypothesis" in ex else "hypo"
        rows.append(
            {
                "premise": ex["premise"],
                "hypothesis": ex[hyp_key],
                "label": LABELS[name],
            }
        )
    return rows


def write_jsonl(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} pairs to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickets", help="tickets.jsonl path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf", action="append", default=[],
                    help="HF dataset slice, e.g. --hf nyu-mll/multi_nli:train:50000[:offset]")
    args = ap.parse_args()
    rows = []
    if args.tickets:
        rows += tickets_to_pairs(args.tickets)
    for spec in args.hf:
        parts = spec.split(":")
        name, split, limit = parts[0], parts[1], int(parts[2])
        offset = int(parts[3]) if len(parts) > 3 else 0
        rows += hf_nli_to_pairs(name, split, limit, offset)
    write_jsonl(rows, args.out)


if __name__ == "__main__":
    main()

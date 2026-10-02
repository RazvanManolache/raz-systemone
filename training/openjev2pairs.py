"""Convert Open-Jev parquet rows to raz training formats.

Hard pairs (for train.py pointwise CE):
  premise = "Q: <question>\\nS: <compact state_json>"
  hypothesis = option text; label 1 (entail) for target argmax else 0 (contra).

Soft listwise rows (for train_soft.py):
  {"premise": ..., "options": [...], "target": [...]}

Rows with <2 options or a non-normalized target are skipped, mirroring
Qyvos's own validation.

Example:
  training/.venv/Scripts/python.exe training/openjev2pairs.py \
      --parquet I:/AI/Data/HF-systemone/datasets/ZefanCai-Open-Jev/data/release-v2-redistributable/train-00000-of-00001.parquet \
      --out-hard training/runs/openjev-hard.jsonl --out-soft training/runs/openjev-soft.jsonl
"""

import argparse
import glob
import json
from collections import Counter

import pyarrow.parquet as pq


def premise_of(question: str, state_json: str, state_chars: int) -> str:
    try:
        state = json.dumps(json.loads(state_json), separators=(",", ":"))
    except Exception:
        state = state_json
    return f"Q: {question}\nS: {state[:state_chars]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="file or glob")
    ap.add_argument("--out-hard", required=True)
    ap.add_argument("--out-soft", required=True)
    ap.add_argument("--state-chars", type=int, default=1000)
    args = ap.parse_args()

    files = sorted(glob.glob(args.parquet))
    assert files, f"no files match {args.parquet}"
    kinds: Counter = Counter()
    n_rows = n_pairs = n_skip = 0
    with open(args.out_hard, "w") as fh, open(args.out_soft, "w") as fs:
        for f in files:
            t = pq.read_table(f, columns=["kind", "question", "options", "target", "state_json"])
            for row in t.to_pylist():
                opts = list(row["options"])
                tgt = [float(x) for x in row["target"][: len(opts)]]
                if len(opts) < 2 or abs(sum(tgt) - 1.0) > 0.05:
                    n_skip += 1
                    continue
                kinds[row["kind"]] += 1
                n_rows += 1
                prem = premise_of(row["question"], row["state_json"], args.state_chars)
                gold = max(range(len(opts)), key=lambda i: tgt[i])
                for i, o in enumerate(opts):
                    fh.write(json.dumps({"premise": prem, "hypothesis": o,
                                         "label": 1 if i == gold else 0}) + "\n")
                    n_pairs += 1
                fs.write(json.dumps({"premise": prem, "options": opts, "target": tgt}) + "\n")
    print(f"rows={n_rows} pairs={n_pairs} skipped={n_skip} kinds={dict(kinds)}")


if __name__ == "__main__":
    main()

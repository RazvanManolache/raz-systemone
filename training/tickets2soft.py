"""Tickets fit file -> soft listwise rows (for train_soft.py).

Each state becomes up to 3 rows (choice/fru/urg) with one-hot targets,
using the same option texts as data.py so hard- and soft-training agree:
  choice: ["This text is about <criterion>" x3], fru: 3 level texts,
  urg: ["no", "yes"] with the statement as the question.
Premise format matches openjev2pairs.py: "Q: <question>\\nS: <state>".
"""

import argparse
import json

CHOICE_LABELS = {"billing": "Payment or subscription issues",
                 "technical": "Bugs or integration problems",
                 "sales": "Pricing or account questions"}
FRU_LEVELS = ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"]
CHOICE_Q = "Which team should handle this?"
FRU_Q = "How frustrated is the customer?"
URG_Q = "The message conveys urgency or time-sensitivity."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()

    n = 0
    with open(args.out, "w") as f:
        for line in open(args.tickets):
            r = json.loads(line)
            s = r["state"]
            rows = []
            if "department" in r:
                opts = [f"This text is about {c}." for c in CHOICE_LABELS.values()]
                tgt = [1.0 if k == r["department"] else 0.0 for k in CHOICE_LABELS]
                rows.append((CHOICE_Q, s, opts, tgt))
            opts = list(FRU_LEVELS)
            tgt = [1.0 if i == r["frustration"] else 0.0 for i in range(3)]
            rows.append((FRU_Q, s, opts, tgt))
            rows.append((URG_Q, s, ["no", "yes"],
                         [0.0, 1.0] if r["is_urgent"] else [1.0, 0.0]))
            for _ in range(args.repeat):
                for q, st, opts, tgt in rows:
                    f.write(json.dumps({"premise": f"Q: {q}\nS: {st}",
                                        "options": opts, "target": tgt}) + "\n")
                    n += 1
    print(f"wrote {n} listwise rows to {args.out}")


if __name__ == "__main__":
    main()

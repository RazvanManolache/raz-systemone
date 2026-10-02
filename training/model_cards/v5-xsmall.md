---
license: mit
base_model: cross-encoder/nli-deberta-v3-xsmall
tags: [nli, cross-encoder, entailment, calibration]
library_name: transformers
---

<!-- Eval rows mirror the master table in training/README.md of
     https://github.com/RazvanManolache/raz-raz -->

# raz-raz-nli-xsmall (v5)

3-class NLI cross-encoder finetuned for **typed decisions with probabilities**:
given a state + a hypothesis per answer, entailment scores become a calibrated
distribution. Powers the `nli` scorer in
[raz](https://github.com/RazvanManolache/raz-raz) (`choice` / `score` /
`noul` questions) via `--nli-model <dir>`, CPU-only, 283MB.

## Training

- Base: `cross-encoder/nli-deberta-v3-xsmall` (continued finetuning)
- Data: 725 NLI pairs from 110 labeled support-ticket states (repeat 8x,
  frustration pairs 16x) + 200,000 MNLI rows; disjoint 10k MNLI eval
- Recipe: 2 epochs, batch 64 Ã¢â‚¬â€ see
  [training/](https://github.com/RazvanManolache/raz-raz/tree/master/training)

## Eval (all on states never seen in training)

| split | judgments | accuracy |
|---|---|---|
| holdout C | 60 | .900 |
| holdout D | 30 | .833 |
| holdout E | 30 | .900 |
| holdout F | 28 | .893 |
| holdout G | 29 | .897 |
| MNLI (disjoint 10k) | 10000 | .941 |

For reference, a 14B LLM judge and the Jev API score .894 on the same
70-state core set.

## Use

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("RazvanManolache/raz-raz-nli-xsmall")
m = AutoModelForSequenceClassification.from_pretrained("RazvanManolache/raz-raz-nli-xsmall")
premise = "The integration keeps failing, please help ASAP."
hyps = ["This text is about Payment or subscription issues.",
        "This text is about Bugs or integration problems.",
        "This text is about Pricing or account questions."]
with torch.no_grad():
    entail = [m(**tok(premise, h, return_tensors="pt")).logits.softmax(-1)[0, 1].item()
            for h in hyps]  # label 1 = entailment
print(entail)  # -> technical wins
```

Or in raz: `hf download RazvanManolache/raz-raz-nli-xsmall --local-dir nli-xsmall`
then `--scorer nli --nli-model nli-xsmall`.

## Limits

Frustration tone (`score`) is the weakest axis (~.70Ã¢â‚¬â€œ.90 per split);
billing-vs-sales phrasing and sarcasm still miss. Labels are 190
hand-written English support tickets Ã¢â‚¬â€ narrow domain by design.

---
license: mit
base_model: cross-encoder/nli-deberta-v3-xsmall
tags: [nli, cross-encoder, entailment, calibration]
library_name: transformers
---

<!-- Eval rows mirror the master table in training/README.md of
     https://github.com/RazvanManolache/raz-systemone -->

# raz-systemone-nli-xsmall-openjev (v9)

The bilingual sibling of `raz-systemone-nli-xsmall`: same 71M
DeBERTa-v3-xsmall cross-encoder, but trained on our 725 ticket pairs
(repeat 8x) **plus 267k Open-Jev pairs** plus 200k MNLI, 2 epochs. It
holds our support-ticket distribution while also speaking Open-Jev —
the only model here fluent in both. Powers the `nli` scorer in
[raz](https://github.com/RazvanManolache/raz-systemone) via
`--nli-model <dir>`. CPU-only, 283MB.

## Eval

Our splits (states never seen in training) and a fixed 900-row Open-Jev
test sample (OJ-900; TypeSafe's Qyvos scores .831 there, the Jev API .811):

| split | judgments | accuracy |
|---|---|---|
| holdout C | 60 | .883 |
| holdout D | 30 | .80 |
| holdout E | 30 | .90 |
| holdout F | 28 | .964 |
| holdout G | 29 | .862 |
| Open-Jev-900 | 900 | .846 |
| MNLI (disjoint) | 10000 | .939 |

## Use

Same as xsmall: `AutoModelForSequenceClassification`, label 1 =
entailment, premise + hypothesis per answer. Or in raz:
`hf download RazvanManolache/raz-systemone-nli-xsmall-openjev --local-dir nli-v9`
then `--scorer nli --nli-model nli-v9`.

## Limits

Frustration tone is still the weakest axis on our labels; Open-Jev
coverage is synthetic control tasks, not real tickets.

---
license: mit
base_model: cross-encoder/nli-deberta-v3-base
tags: [nli, cross-encoder, entailment, calibration]
library_name: transformers
---

<!-- Eval rows mirror the master table in training/README.md of
     https://github.com/RazvanManolache/raz-systemone -->

# raz-systemone-nli-base (v7)

184M-param sibling of `raz-systemone-nli-xsmall`: same 725 NLI pairs from 110
labeled support-ticket states (repeat 8x) + 200,000 MNLI rows, 2 epochs,
from `cross-encoder/nli-deberta-v3-base`. Powers the `nli` scorer in
[raz](https://github.com/RazvanManolache/raz-systemone) via
`--nli-model <dir>`. 738MB.

## Eval (all on states never seen in training)

| split | judgments | xsmall (v5) | base (v7) |
|---|---|---|---|
| holdout C | 60 | .900 | .983 |
| holdout D | 30 | .833 | .867 |
| holdout E | 30 | .900 | .833 |
| holdout F | 28 | .893 | .857 |
| holdout G | 29 | .897 | .897 |
| MNLI (disjoint) | 10000 | .941 | .980 |

Split decision: base nearly perfects the broad 60-state split (1 miss)
but loses the small targeted splits to xsmall. Pick base for broad
queries, xsmall for the best balance (and 10-min retrains vs ~2h).

## Use

Same as xsmall: `AutoModelForSequenceClassification`, label 1 =
entailment, premise + hypothesis per answer. Or in raz:
`hf download RazvanManolache/raz-systemone-nli-base --local-dir nli-base`
then `--scorer nli --nli-model nli-base`.

## Limits

Same narrow-domain caveats as xsmall: 190 hand-written English support
tickets; frustration tone is the weakest axis.

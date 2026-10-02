# Training: finetune our own NLI checkpoint

Every question type in systemone routes through entailment, so one finetuned
3-class NLI model upgrades choice, score, and noul at once â€” and loads with
zero Rust changes via `--nli-model <dir>`.

## Setup

The env already exists at `training/.venv` (Python 3.14, CUDA torch 2.9 from
the system interpreter via `--system-site-packages`, transformers/datasets/
accelerate/sklearn inside the venv). From a fresh shell:

```powershell
training/.venv/Scripts/Activate.ps1
```

From scratch on another machine: `python -m venv --system-site-packages
training/.venv` (needs a CUDA torch on the system python first, see
[pytorch.org](https://pytorch.org)), then
`uv pip install --python training/.venv/Scripts/python.exe -r training/requirements.txt`.

## Data

`data.py` converts everything to unified pair JSONL
(`premise`/`hypothesis`/`label`, 0=contradiction, 1=entailment, 2=neutral):

```powershell
# tickets -> pairs (choice options and score levels become NLI hypotheses)
training/.venv/Scripts/python.exe training/data.py --tickets tests/data/tickets.jsonl --out training/runs/tickets_pairs.jsonl
# + public NLI (name:split:limit[:offset]); offsets give disjoint eval slices
training/.venv/Scripts/python.exe training/data.py --out training/runs/mnli_train.jsonl --hf nyu-mll/multi_nli:train:200000
training/.venv/Scripts/python.exe training/data.py --out training/runs/mnli_eval.jsonl --hf nyu-mll/multi_nli:train:10000:200000
```

FEVER is deliberately excluded (claims need retrieved evidence, not direct
pairs). SNLI/ANLI work the same way (`snli`, `facebook/anli`).

## Train

```powershell
# smoke (minutes): tickets + 2k MNLI, proves the loop end to end
training/.venv/Scripts/python.exe training/train.py --train training/runs/smoke_train.jsonl --eval training/runs/smoke_eval.jsonl --out training/runs/smoke --epochs 1
# full: continue-finetune the xsmall NLI checkpoint on the mix
training/.venv/Scripts/python.exe training/train.py --train training/runs/tickets_pairs.jsonl --train training/runs/mnli_train.jsonl --eval training/runs/mnli_eval.jsonl --out training/runs/v1 --epochs 2
```

Best checkpoint is picked by eval accuracy; the script also reports ECE.
Output is systemone-ready:

```powershell
cargo run -q --release -- eval --data tests/data/tickets.jsonl --scorer nli --nli-model training/runs/v1
```

## Notes

- Starting point is the xsmall NLI checkpoint (already NLI-trained), not a
  raw base model: continued finetuning, faster and better.
- Label ids live in the saved config (`id2label`); the Rust loader reads
  them dynamically, so any consistent order works.
- More `tests/data/tickets.jsonl` rows flow straight into `--train` via
  `data.py` â€” but see Recipes: past a point, sampling beats labeling.

## Holdout results (master table)

The single home for checkpoint numbers. Overall accuracy on states
never seen in training â€” each holdout was quarantined *before* the
run that first reports it. The MNLI row is the disjoint 10k slice
(offset 200k). Full-`tickets.jsonl` numbers are train-contaminated
for every finetuned model and are not compared anywhere.

| split | n | base | smoke2 | v1 | v2 | v3 | v4 | v5 | v6 | v7 |
|---|---|---|---|---|---|---|---|---|---|---|
| C | 60 | .40 | .55 | .80 | .867 | .867 | .883 | .90 | .883 | **.983** |
| D | 30 | .433 | .60 | .667 | .80 | .733 | .767 | .833 | .833 | **.867** |
| E | 30 | .333 | .367 | .633 | .80 | .733 | .80 | .90 | **.933** | .833 |
| F | 28 | .679 | .607 | .786 | .857 | .857 | **.893** | **.893** | .857 | .857 |
| G | 29 | .483 | â€” | .793 | .862 | .793 | .862 | **.897** | .828 | **.897** |
| MNLI | 10000 | â€” | â€” | .942 | .9416 | .9424 | .9418 | .9411 | .941 | **.9802** |

Model cards in `model_cards/` mirror their model's column; the root
README links here instead of repeating numbers.

## Recipes

All runs: 2 epochs, batch 64, + 200k MNLI, on a 5080 (~10 min
xsmall, ~2h base). Each run changes one variable vs its parent.

- **smoke2**: 300 pairs from t01â€“t50 + 2k MNLI, 1 epoch. C .40â†’.55
  (choice .80, score .55, noul .30). A same-states .76 proved only
  that contamination must be quarantined out.
- **v1**: 454 pairs / 70 states (t01â€“50 + t71â€“90), 1x. MNLI .942 /
  ECE .006. Weakness: billing-vs-sales/tech, fru-0/1.
- **v2**: 594 pairs / 90 states (+t101â€“120, targeted billing/sales +
  fru-0/1). MNLI .9416 / .018. E choice .70â†’.90: targeted labels work.
- **v3**: 725 pairs / 110 states (+t131â€“150, fru pairs / fru-2 /
  urgency). MNLI .9424 / .018. Plateau: â‰ˆv2 everywhere. Diagnosis:
  725/200725 â‰ˆ 0.36% of training â€” the domain signal drowns in MNLI.
- **v4**: same 725 pairs, `--train pack` 8x (â‰ˆ2.8% of mix). MNLI
  .9418 / .012. Plateau broken (C .883, F .893 with perfect choice).
- **v5** (best balanced, 283MB): pack 8x + 330 fru pairs 8x extra
  (fru 16x). MNLI .9411 / .018. Bests on C/D/E/G, F tied; E has
  perfect choice and noul. LLM-judge territory.
- **v6**: whole pack 16x (â‰ˆ5.5%). MNLI .941 / .017. Best-ever E
  (.933, 2 misses) but drops F/G: blanket repetition â‰  targeted.
- **v7** (broad-query best, 738MB): v4's data/sampling on
  `cross-encoder/nli-deberta-v3-base`. MNLI .9802 / .011. C .983
  (1 miss; perfect choice and score), takes D, ties G, loses E/F.

Open: v5's fru-targeted sampling on the base backbone; the banked
t161â€“t180 fru labels (pack `full-pack4`, 850 pairs) unused by any run.

## Data splits

| file | states | role |
|---|---|---|
| `tests/data/tickets.jsonl` | 190 (t01â€“t190) | all labels; full-set eval contaminated, never compared |
| `tests/data/fit100.jsonl` | 70 | v1 train states |
| `tests/data/fit120.jsonl` | 90 | v2 train states |
| `tests/data/fit150.jsonl` | 110 | v3â€“v7 train states |
| `tests/data/fit180.jsonl` | 130 | banked (v8), unused |
| `tests/data/holdout{C,D,E,F,G}.jsonl` | 20/10/10/10/10 | quarantined; in no train mix |

Packs under `training/runs/` (git-ignored, reproducible via `data.py`):
`full-pack` 454, `full-pack2` 594, `full-pack3` 725, `full-pack4` 850,
`fru-only` 330, `mnli200k`, `mnli-eval10k`.

## Releasing a new checkpoint

1. Labels: append to `tests/data/tickets.jsonl`, extend the fit file,
   quarantine the new holdout **before** training on anything.
2. `data.py` â†’ new pack; `train.py` â†’ `training/runs/vN` (one
   variable vs the previous run).
3. `eval` the checkpoint on **all** holdouts (CPU, minutes).
4. Add `model_cards/vN-*.md` (mirror the new master-table column).
5. `upload.py` â†’ Hub; update the master table + Recipes above.

## Publishing models

```powershell
# auth once: hf auth login  (or set HF_TOKEN)
training/.venv/Scripts/python.exe training/upload.py --checkpoint training/runs/v5 --card training/model_cards/v5-xsmall.md --repo RazvanManolache/systemone-nli-xsmall
training/.venv/Scripts/python.exe training/upload.py --checkpoint training/runs/v7 --card training/model_cards/v7-base.md --repo RazvanManolache/systemone-nli-base
```

Consumers then: `hf download RazvanManolache/systemone-nli-xsmall --local-dir nli-xsmall`
and `--scorer nli --nli-model nli-xsmall` â€” zero Rust changes.

# Training: finetune our own NLI checkpoint

Every question type in raz routes through entailment, so one finetuned
3-class NLI model upgrades choice, score, and noul at once — and loads with
zero Rust changes via `--nli-model <dir>`.

Pipeline: `tests/data/*.jsonl` labels → `data.py` (unified NLI pairs) →
`train.py` (continued finetuning on tickets + MNLI) → `raz eval` on
quarantined holdouts → `model_cards/` + `upload.py` to the Hub.

## Setup

The env already exists at `training/.venv` (Python 3.14, CUDA torch 2.9 from
the system interpreter via `--system-site-packages`, transformers/datasets/
accelerate/sklearn/huggingface_hub inside the venv). From a fresh shell:

```powershell
training/.venv/Scripts/Activate.ps1
```

From scratch on another machine: install a CUDA torch on the system python
first (see [pytorch.org](https://pytorch.org)), then:

```powershell
python -m venv --system-site-packages training/.venv
uv pip install --python training/.venv/Scripts/python.exe -r training/requirements.txt
```

(`pip` works in place of `uv pip`. Keep the env inside the project —
never in a scratch dir or the system interpreter.)

## Data

`data.py` converts everything to unified pair JSONL
(`premise`/`hypothesis`/`label`, 0=contradiction, 1=entailment, 2=neutral;
`train.py` writes the mapping into the model config and raz reads it back
dynamically, so the order only has to be self-consistent):

```powershell
# tickets -> pairs (choice options and score levels become NLI hypotheses)
training/.venv/Scripts/python.exe training/data.py --tickets tests/data/fit150.jsonl --out training/runs/full-pack3
# + public NLI; --hf is repeatable (name:split:limit[:offset]); offsets give disjoint eval slices
training/.venv/Scripts/python.exe training/data.py --out training/runs/mnli200k.jsonl --hf nyu-mll/multi_nli:train:200000
training/.venv/Scripts/python.exe training/data.py --out training/runs/mnli-eval10k.jsonl --hf nyu-mll/multi_nli:train:10000:200000
# --tickets and --hf can also combine into one --out file
```

Ticket → pair mapping (templates must stay identical to the zero-shot ones
in `src/nli.rs`):

| label | pairs per state | hypotheses |
|---|---|---|
| `department` | 3 (one per option) | `"This text is about {Payment or subscription issues / Bugs or integration problems / Pricing or account questions}."` — match=entail, others=contra |
| `frustration` | 3 (one per level) | `"Calm, just stating facts"` / `"Frustrated but civil"` / `"Very angry, strong language"` — exact=entail, adjacent=neutral, far=contra |
| `is_urgent` | 1 | `"The message conveys urgency or time-sensitivity"` — true=entail, false=contra |

FEVER is deliberately excluded (claims need retrieved evidence, not direct
pairs). SNLI/ANLI work the same way (`snli`, `facebook/anli`). Public-NLI
label mapping is by NAME (robust to id-order differences); unknown labels
are skipped.

## Train

```powershell
# smoke (minutes): tickets + 2k MNLI, proves the loop end to end
training/.venv/Scripts/python.exe training/train.py --train training/runs/smoke_train.jsonl --eval training/runs/smoke_eval.jsonl --out training/runs/smoke --epochs 1
# full: continue-finetune the xsmall NLI checkpoint on the mix (--train is repeatable: repeat a file to upsample it)
training/.venv/Scripts/python.exe training/train.py --train training/runs/full-pack3 --train training/runs/full-pack3 --train training/runs/mnli200k.jsonl --eval training/runs/mnli-eval10k.jsonl --out training/runs/vN --epochs 2
```

| flag | meaning |
|---|---|
| `--train` | train pair JSONL; repeatable — repeating a file upsamples it |
| `--eval` | eval pair JSONL (must be disjoint rows!) |
| `--base` | starting checkpoint id (default xsmall; v7 used `cross-encoder/nli-deberta-v3-base`) |
| `--out` | output dir (raz-ready: `--scorer nli --nli-model <dir>`) |
| `--epochs` / `--batch` / `--lr` / `--max-len` / `--seed` | 2 / 64 / default / default / fixed-seed |

Best checkpoint is picked by eval accuracy; the script also reports ECE.
Starting point is an already-NLI-trained checkpoint, not a raw base model:
continued finetuning is faster and better (~10 min xsmall, ~2h base on a
5080 at batch 64).

Sampling guide (earned the hard way): tickets at 0.36% of the mix drown in
MNLI (v3 plateaued); 8x repetition (≈2.8%) broke it (v4); extra repetition
of just the 330 frustration pairs won overall (v5); 16x of everything
helped one split and hurt others (v6). Change one variable per run.

## Eval protocol

Two gates, in order:

1. **MNLI health check** — the disjoint 10k slice must stay ≈.94 (xsmall)
   with low ECE. A drop means the domain mix is collapsing general NLI.
2. **Holdout sweep** — `raz eval` on **every** `tests/data/holdout*.jsonl`
   (CPU, minutes). Each holdout was quarantined before the run that first
   reports it, so these are the only honest numbers. Full-file numbers are
   train-contaminated — never compare on them.

```powershell
cargo run -q --release -- eval --data tests/data/holdoutC.jsonl --scorer nli --nli-model training/runs/vN
```

## Holdout results (master table)

The single home for checkpoint numbers. Overall accuracy on states never
seen in training. Model cards in `model_cards/` mirror their model's
column; the root README links here instead of repeating numbers.

| split | n | base | smoke2 | v1 | v2 | v3 | v4 | v5 | v6 | v7 | v9 | v10 | v11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C | 60 | .40 | .55 | .80 | .867 | .867 | .883 | .90 | .883 | **.983** | .883 | .717 | .917 |
| D | 30 | .433 | .60 | .667 | .80 | .733 | .767 | .833 | .833 | **.867** | .80 | .633 | .833 |
| E | 30 | .333 | .367 | .633 | .80 | .733 | .80 | .90 | **.933** | .833 | .90 | .533 | .867 |
| F | 28 | .679 | .607 | .786 | .857 | .857 | .893 | .893 | .857 | .857 | **.964** | .714 | .857 |
| G | 29 | .483 | — | .793 | .862 | .793 | .862 | **.897** | .828 | **.897** | .862 | .552 | .828 |
| MNLI | 10000 | — | — | .942 | .9416 | .9424 | .9418 | .9411 | .941 | **.9802** | .9385 | — | .9378 |
| OJ-900 | 900 | .370 | — | — | — | — | — | .432 | — | .474 | **.846** | .782 | .829 |

## Recipes

All runs: 2 epochs, batch 64, + 200k MNLI, on a 5080 (~10 min xsmall,
~2h base). Each run changes one variable vs its parent.

- **smoke2**: 300 pairs from t01–t50 + 2k MNLI, 1 epoch. C .40→.55
  (choice .80, score .55, noul .30). A same-states .76 proved only
  that contamination must be quarantined out.
- **v1**: 454 pairs / 70 states (t01–50 + t71–90), 1x. MNLI .942 /
  ECE .006. Weakness: billing-vs-sales/tech, fru-0/1.
- **v2**: 594 pairs / 90 states (+t101–120, targeted billing/sales +
  fru-0/1). MNLI .9416 / .018. E choice .70→.90: targeted labels work.
- **v3**: 725 pairs / 110 states (+t131–150, fru pairs / fru-2 /
  urgency). MNLI .9424 / .018. Plateau: ≈v2 everywhere. Diagnosis:
  725/200725 ≈ 0.36% of training — the domain signal drowns in MNLI.
- **v4**: same 725 pairs, `--train pack` 8x (≈2.8% of mix). MNLI
  .9418 / .012. Plateau broken (C .883, F .893 with perfect choice).
- **v5** (best balanced, 283MB): pack 8x + 330 fru pairs 8x extra
  (fru 16x). MNLI .9411 / .018. Bests on C/D/E/G, F tied; E has
  perfect choice and noul. LLM-judge territory.
  Hub: `RazvanManolache/raz-systemone-nli-xsmall`.
- **v6**: whole pack 16x (≈5.5%). MNLI .941 / .017. Best-ever E
  (.933, 2 misses) but drops F/G: blanket repetition ≠ targeted.
- **v7** (broad-query best, 738MB): v4's data/sampling on
  `cross-encoder/nli-deberta-v3-base`. MNLI .9802 / .011. C .983
  (1 miss; perfect choice and score), takes D, ties G, loses E/F.
  Hub: `RazvanManolache/raz-systemone-nli-base`.

Open: v5's fru-targeted sampling on the base backbone; the banked
t161–t180 fru labels (pack `full-pack4`, 850 pairs) unused by any run.

## Outside-data era (v9–v11)

Public Jev-ecosystem artifacts (TypeSafe's Open-Jev, shreyanbr's pairs —
see Data sources) unlocked cross-distribution training. New tools:
`openjev2pairs.py` (parquet → hard pairs + soft listwise rows),
`tickets2soft.py` (tickets → listwise rows), `train_soft.py` (listwise
soft-CE, Qyvos-style objective on our 3-class head), `calibrate.py`
(offline per-question temperature + Platt on `--dump` output).

- **v9** (bilingual champ): v4 mix + 267k Open-Jev hard pairs (1x).
  MNLI .9385 / .021. Holds our splits (F .964, new best), jumps
  Open-Jev-900 .43 → **.846** — past Qyvos (.831) and the Jev API (.811).
  Hub: `RazvanManolache/raz-systemone-nli-xsmall-openjev`.
- **v11** (+99k shreyanbr pairs mapped 1:1): MNLI .9378 / .023. C .917
  (new best, perfect choice) but F/G drop: banking-intent data helps
  choice and dilutes tone. More data ≠ better; domain match matters.
- **v10** (soft-CE, 1 epoch, `train_soft.py`): Open-Jev val .77 / .61;
  ours C–G .55–.72, OJ-900 .782. Loses to hard pointwise CE on both
  turfs — confounded by 1-vs-2 epochs and inference-template mismatch
  (raw options at train vs templated hypotheses + contrast-noul at
  inference). Needs a 2-epoch rematch before judging the objective.

## Calibration (post-hoc, offline only)

`calibrate.py` fits per-question temperature (choice/score) + Platt
(noul) on `--dump` output. Fit on the quarantined `calib.jsonl` (29
judgments), tested for transfer on C–G: helps C consistently (v5 NLL
.37→.23, ECE .12→.07) but is mixed elsewhere and blows up NLL on G
(v5 .59→1.23) — 29 judgments fit noise, same lesson as the early
temperature attempt. Not wired into the scorer. The honest next try is
fitting on Open-Jev's 4672-row calibration split.

## Data sources (outside `tests/data`)

| source | what | used in |
|---|---|---|
| `ZefanCai/Open-Jev` train (CC0) | 79k rows → 267k hard pairs + 79k soft rows | v9, v10, v11 |
| `ZefanCai/Open-Jev` validation | 3723 soft rows (train_soft eval) | v10 |
| `ZefanCai/Open-Jev` test | fixed 900-row round-robin sample (OJ-900) | cross-bench only, never train |
| `shreyanbr/system-one-training-pairs` | 99k pairs, labels map 1:1 to ours | v11 |
| `TypeSafeAI/Qyvos`, `shreyanbr-gold` | external reference models | cross-bench only |
| `typesafe/evalsafe-*` | checked; action-generation eval, doesn't map to typed decisions | — |

## Data splits

| file | states | role |
|---|---|---|
| `tests/data/tickets.jsonl` | 190 (t01–t190) | all labels; full-set eval contaminated, never compared |
| `tests/data/fit100.jsonl` | 70 | v1 train states |
| `tests/data/fit120.jsonl` | 90 | v2 train states |
| `tests/data/fit150.jsonl` | 110 | v3–v7 train states |
| `tests/data/fit180.jsonl` | 130 | banked (v8), unused |
| `tests/data/holdout{C,D,E,F,G}.jsonl` | 20/10/10/10/10 | quarantined; in no train mix |
| `tests/data/calib.jsonl` | 10 (t191–200) | quarantined; fit scalers only — never train, never test-report |

Packs under `training/runs/` (git-ignored, reproducible via `data.py`):
`full-pack` 454, `full-pack2` 594, `full-pack3` 725, `full-pack4` 850,
`fru-only` 330, `mnli200k`, `mnli-eval10k`.

## Releasing a new checkpoint

1. Labels: append to `tests/data/tickets.jsonl`, extend the fit file,
   quarantine the new holdout **before** training on anything.
2. `data.py` → new pack; `train.py` → `training/runs/vN` (one
   variable vs the previous run).
3. `eval` the checkpoint on **all** holdouts (CPU, minutes).
4. Add `model_cards/vN-*.md` (mirror the new master-table column).
5. `upload.py` → Hub; update the master table + Recipes above.

## Publishing models

```powershell
# auth once: hf auth login  (or set HF_TOKEN)
training/.venv/Scripts/python.exe training/upload.py --checkpoint training/runs/v5 --card training/model_cards/v5-xsmall.md --repo RazvanManolache/raz-systemone-nli-xsmall
training/.venv/Scripts/python.exe training/upload.py --checkpoint training/runs/v7 --card training/model_cards/v7-base.md --repo RazvanManolache/raz-systemone-nli-base
```

Consumers then: `hf download RazvanManolache/raz-systemone-nli-xsmall --local-dir nli-xsmall`
and `--scorer nli --nli-model nli-xsmall` — zero Rust changes.

## Troubleshooting

- `torch` without CUDA: reinstall from pytorch.org for your CUDA version
  *before* the rest of requirements (the file's extra index only helps
  when torch itself comes from a CUDA wheel).
- CUDA out of memory: drop `--batch` to 32 (base model) or 16; shorten
  `--max-len`. xsmall at batch 64 uses ~13GB with Ollama idle.
- Windows symlink warnings from `huggingface_hub`: harmless (degraded
  cache layout); silence with `HF_HUB_DISABLE_SYMLINKS_WARNING=1` or
  enable Developer Mode.
- Run-to-run ±1 judgment noise on near-tie argmaxes (GPU): compare on
  the full holdout suite, not single judgments.

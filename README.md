# Raz SystemOne

Raz (`raz`) gives **probabilities for answers, not text**. Send a state plus
typed questions — `choice` (pick one), `score` (pick a rubric level), `noul`
(probability a statement holds) — and get back a calibrated distribution per
question. A Jev-compatible mini-clone that runs on local models, with its own
finetuned NLI checkpoints that match LLM-judge accuracy at a fraction of the
cost.

```powershell
cargo run -q --release -- ask --state-file examples/state.txt --questions examples/questions.json --scorer nli
# {"answers": {"department": {"type": "choice", "choice": "technical", ...}}}
```

## Install

Prereqs: a Rust stable toolchain (`rustup`), and optionally
[Ollama](https://ollama.com) (only the `embed`/`llm` scorers need it) and the
`hf` CLI (only for downloading our checkpoints).

```powershell
git clone https://github.com/RazvanManolache/raz-systemone.git
cd raz-systemone
cargo build --release        # binary: target/release/raz(.exe)
cargo test                   # 12 hermetic unit tests
```

Copy `.env.example` to `.env` (git-ignored) only if you use the `jev`
reference scorer, which needs `TYPESAFE_API_KEY`.

## Quickstart

One-shot CLI (default scorer: `embed`, needs Ollama + `nomic-embed-text`):

```powershell
raz ask --state-file examples/state.txt --questions examples/questions.json --scorer nli
raz ask --state "The integration keeps failing, please help ASAP." --questions examples/questions.json --scorer llm
Get-Content ticket.txt | raz ask --state-file - --questions examples/questions.json
```

HTTP server (default scorer: `nli`, CPU-only, no Ollama needed):

```powershell
raz serve --port 18080 --scorer nli
curl -s -X POST http://127.0.0.1:18080/v1/systemone -H "Content-Type: application/json" `
  -d '{"state": "...", "model": "nli", "questions": {"department": {"type": "choice", "instructions": "Which team", "criteria": {"billing": "Payment issues", "technical": "Bugs"}}}}'
```

Calibration eval on the labeled set (200 states, 576 judgments):

```powershell
cargo run -q --release -- eval --data tests/data/tickets.jsonl --scorer nli --dump out.jsonl
```

## CLI reference

`raz ask` — answer questions, print `{"answers": {...}}` as JSON.

| flag | meaning |
|---|---|
| `--state` / `--state-file` | state inline, from file, or `-` for stdin (exactly one) |
| `--questions` | JSON file `{"name": {"type": ...}}`, or `-` for stdin |
| `--scorer` | `embed` (default for ask), `llm`, `nli`, `jev` |

`raz serve` — HTTP server on `127.0.0.1`.

| flag | default | meaning |
|---|---|---|
| `--port` | 8080 | bind port (examples use 18080; 8080 is blocked on some machines) |
| `--scorer` | `nli` | default when a request omits `"model"` |

`raz eval` — accuracy + calibration (ECE) on a labeled JSONL set.

| flag | default | meaning |
|---|---|---|
| `--data` | — | labeled cases, one JSON object per line (see `tests/data`) |
| `--scorer` | `nli` | which scorer to measure |
| `--limit` | 0 (all) | stop after N cases |
| `--dump` | — | write one JSON line per judgment (offline analysis/fitting) |

Backend flags (all subcommands):

| flag | default | used by |
|---|---|---|
| `--ollama-url` | `http://localhost:11434` | `embed`, `llm` |
| `--embed-model` | `nomic-embed-text` | `embed` |
| `--llm-model` | `huihui_ai/phi4-abliterated` | `llm` |
| `--nli-model` | `cross-encoder/nli-deberta-v3-xsmall` | `nli` (HF id or local dir) |
| `--jev-model` | `jev-latest` | `jev` (key from `TYPESAFE_API_KEY` or `.env`) |
| `--temperature` | 0.1 | `embed` (softmax over cosine scores) |
| `--route` | — (required for `route`) | `route` spec `choice=<s>,score=<s>,noul=<s>` |

## HTTP API

`POST /v1/systemone` (path kept Jev-compatible). Body: `state`, optional
`model` (one of `embed|llm|nli|jev|route`, else the server default; `route`
uses the server's `--route` spec), and
`questions`. Response: `{"model": ..., "answers": {...}}`.

`GET /v1/models` → `{"default": ..., "scorers": [...], "backends": {...}}`.
`GET /healthz` → `ok`. Errors are `{"error": msg}` (400 unknown scorer or
unconfigured `jev`; 500 scorer failure).

## Question types and answer shapes

```json
"department": {"type": "choice", "instructions": "Which team should handle this",
  "criteria": {"billing": "Payment or subscription issues", "technical": "Bugs or integration problems", "sales": "Pricing or account questions"}}
"frustration": {"type": "score", "instructions": "How frustrated is the customer",
  "criteria": ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"]}
"is_urgent": {"type": "noul", "instructions": "The message conveys urgency or time-sensitivity"}
```

Answers (untagged `type` field tells them apart):

```json
"department": {"type": "choice", "choice": "technical",
  "probabilities": {"billing": 0.08, "technical": 0.85, "sales": 0.07}, "confidence": 0.85},
"frustration": {"type": "score", "score": 1.2,
  "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}, "confidence": 0.6,
  "legend": {"0": "Calm, just stating facts", "1": "Frustrated but civil", "2": "Very angry, strong language"}},
"is_urgent": {"type": "noul", "noul": 0.91}
```

## Scorers

- `embed`: cosine similarity between state and option embeddings + softmax.
  Fast, local, uncalibrated; **cannot handle negation** (negated phrases
  still score ~1.0). Embed model via `--embed-model`.
- `llm`: a chat model answers a single-token constrained question;
  first-token logprobs become the distribution. Needs a model that emits
  the bare label first — chatty/thinking models fail loudly. Default
  `huihui_ai/phi4-abliterated` (measured best, holdout-validated);
  `judge-3b-4k` (a local 4k-context llama3.2:3b: 23GB → 7GB, 3x faster)
  trades accuracy for speed.
- `nli`: a DeBERTa-v3 cross-encoder reads state + hypothesis together in
  one batched forward pass; entailment (choice) or entail-vs-contra
  contrast (score/noul) becomes the score. No Ollama, CPU-only, model
  loads once per process (CLI ~1.7s cold) or once per server (~220ms
  served, scales near-linearly to 4 concurrent). `--nli-model` takes a
  HF id or a local dir — stock or our finetuned checkpoints.
- `jev`: the real TypeSafe API as a reference scorer. Key from
  `TYPESAFE_API_KEY` or `.env`.
- `route`: per-question-type router — each of choice/score/noul answered
  by its own scorer (e.g. `--route choice=nli:./v7,score=nli:./v7,noul=nli:./v9`).
  Best measured Jev-free combo beats every single model including the Jev API;
  see Scoreboard. Each leg is `name[:model]`; a bare name reuses the backend
  defaults.

## Checkpoints

Our finetuned NLI checkpoints ([full results table](training/README.md#holdout-results-master-table)):

- [raz-systemone-nli-xsmall](https://huggingface.co/RazvanManolache/raz-systemone-nli-xsmall)
  (v5, 283MB) — best balanced on our labels; at/near LLM-judge accuracy, CPU-only.
- [raz-systemone-nli-xsmall-openjev](https://huggingface.co/RazvanManolache/raz-systemone-nli-xsmall-openjev)
  (v9, 283MB) — bilingual: holds our labels, beats the Jev API on Open-Jev's own test.
- [raz-systemone-nli-base](https://huggingface.co/RazvanManolache/raz-systemone-nli-base)
  (v7, 738MB) — best on broad queries; near-perfect on the 60-state split.

```powershell
hf download RazvanManolache/raz-systemone-nli-xsmall --local-dir nli-xsmall
raz ask --state-file examples/state.txt --questions examples/questions.json --scorer nli --nli-model nli-xsmall
```

## Scoreboard (every model on the same unseen splits; macro = mean of C–G)

OJ-900 = accuracy on a fixed 900-row Open-Jev test sample (LLM judges via
the same letter-label logprob harness as `llm.rs`, verified bit-identical
on cross-check rows; only `route` can't run it — its legs are our-3-question
specialists). All holdouts were quarantined before any training run.

| model | size | C 60 | D 30 | E 30 | F 28 | G 29 | macro | OJ-900 |
|---|---|---|---|---|---|---|---|---|
| route (choice→v7, score→v7, noul→v9) | ours | .950 | .900 | .867 | .893 | .897 | **.901** | — |
| jev (API reference) | cloud | .933 | .933 | .867 | .857 | .897 | .897 | .811 |
| v7 (ours, base) | 184M | .983 | .867 | .833 | .857 | .897 | .887 | .474 |
| v5 (ours, xsmall) | 71M | .900 | .833 | .900 | .893 | .897 | .885 | .432 |
| v9 (ours, xsmall + Open-Jev) | 71M | .883 | .800 | .900 | .964 | .862 | .882 | **.846** |
| v11 (ours, v9 + shreyanbr pairs) | 71M | .917 | .833 | .867 | .857 | .828 | .860 | .829 |
| phi4-abliterated (LLM judge) | 14B | .883 | .833 | .867 | .821 | .862 | .853 | .527 |
| mistral-nemo (LLM judge) | 12B | .683 | .700 | .633 | .679 | .759 | .687 | .473 |
| v10 (ours, soft-CE, 1 epoch) | 71M | .717 | .633 | .533 | .714 | .552 | .630 | .782 |
| judge-3b-4k (LLM judge) | 3B | .533 | .667 | .500 | .679 | .690 | .614 | .397 |
| shreyanbr-gold (external xsmall) | 71M | .600 | .567 | .600 | .571 | .586 | .585 | .343 |
| embed (nomic) | — | .500 | .467 | .533 | .500 | .517 | .503 | .263 |
| moritz (DeBERTa-mnli-fever-anli) | 184M | .417 | .467 | .333 | .714 | .483 | .483 | .563 |
| base xsmall (zero-shot) | 71M | .400 | .433 | .333 | .679 | .483 | .466 | .370 |
| qyvos (official open Jev) | 144M | .467 | .400 | .600 | .464 | .241 | .434 | .831 |
| base-base (zero-shot) | 184M | .383 | .400 | .167 | .679 | .517 | .429 | .490 |

The per-type router leads (.901 macro, 161/177 pooled) using only our own
checkpoints — choice→v7 (54/57), score→v7 (52/60), noul→v9 (55/60), each
the best Jev-free model at its job. (A Jev leg would add +.014 — score→jev
is 54/60 — but Jev is who we're beating.) The Jev-free per-split oracle is
.938 pooled, so most headroom is captured — but the routing was picked on
these same splits, so the true edge is likely smaller until a fresh holdout
confirms it. Below the router, Jev leads single models on our labels while
our 71M v5/v9 beat it on E and F (v9 takes F to .964) and beat the 14B phi4
judge on every split. Choice is essentially solved; frustration tone is the gap
(Jev still leads score .89 vs next-best .87). On Open-Jev's own turf, our
v9 beats both the Jev API (.846 vs .811) and the official Qyvos (.831) —
the only model bilingual in both distributions. v10's soft-CE (1 epoch)
loses to hard pointwise CE on both turfs; needs a 2-epoch rematch with
matched inference templates before judging the objective. A calm-No prompt
example, an NLI blend, and temperature scaling were all tried early: the
example shipped (it fixed small-model Yes-bias); the blend and temperatures
fit noise and were removed. ±1 judgment of run-to-run GPU noise on
near-tie argmaxes.

## Performance (pure processing; RTX 5080 + Ryzen 9 9950X3D)

NLI medians are batched forward passes on CPU (one-time model load
excluded); judges are med-of-3 warm `ask` runs over localhost (network
≈ 0, so this is pure processing); Jev is med-of-3 client-side (its
server time is unobservable — the API returns no timing fields).

| model | ms/question | runs on |
|---|---|---|
| ours xsmall (v5/v9/v10/v11) | 13 | CPU, batched |
| ours base (v7) | 31 | CPU, batched |
| qyvos | 15 | CPU |
| shreyanbr-gold | 14 | CPU |
| judge-3b-4k | ~256 | GPU, 7GB VRAM |
| phi4-abliterated | ~282 | GPU, 12GB VRAM |
| mistral-nemo | ~388 | 26% GPU + CPU offload (262k ctx) |
| nomic-embed | ~248 | GPU, 323MB VRAM |
| jev (API) | ~351 | cloud (client-side, incl. internet) |

The 71M NLI answers ~20x faster than the 14B judge and ~25x faster than
the Jev API call. Server NLI scales near-linearly to 4 concurrent
(lock-free) and costs ~400MB RAM; phi4 cold-loads in ~3.5s.

## Labels and eval data

`tests/data/tickets.jsonl` holds 200 hand-written support states with
`department` / `frustration` / `is_urgent` labels (576 judgments). Fit
files (`fit100/120/150/180.jsonl`) are train-state lists; `holdout{C,D,E,F,G}.jsonl`
are quarantined states no checkpoint trained on — the only honest
comparison (full-file numbers are train-contaminated for finetuned
models); `calib.jsonl` (t191–200) is a quarantined calibration-only split
(fit post-hoc scalers on it, never train on it, never report accuracy on it).
See [training/README.md](training/README.md) for the splits registry and
the finetuning results.

## Tests

- `cargo test` — hermetic unit tests (math, logprob parsing, contrast, ECE).
- `cargo test -- --ignored` — live tests (Ollama scorers, NLI, HTTP server,
  Jev reference; need Ollama and, for Jev, `TYPESAFE_API_KEY`).

## Layout

- `src/` — the Rust runner: CLI (`main.rs`), library (`lib.rs`), scorers
  (`embed.rs`, `llm.rs`, `nli.rs`, `jev.rs`), server (`server.rs`), eval
  harness (`eval.rs`), shared Ollama client + math.
- `tests/` — live (ignored) integration tests.
- `tests/data/` — labels and train/holdout splits.
- `examples/` — sample state + questions for `ask`.
- `training/` — trainer (`data.py`, `train.py`, `upload.py`), model cards,
  and the results log. `training/runs/` (checkpoints, MNLI) is git-ignored.

## Troubleshooting

- `serve` fails to bind: port 8080 is blocked on some machines — use
  `--port 18080` (or anything free).
- `llm` scorer errors about missing labels: the model chatters instead of
  emitting the bare label — use phi4-abliterated or `judge-3b-4k`.
- CUDA out of memory with Ollama: keep one resident judge (phi4 ~12GB);
  `ollama stop <model>` frees the rest. The `nli` scorer needs no GPU.
- Slow first NLI call: one-time model download + load (~90MB stock);
  the server loads once and reuses it.

## Training your own checkpoint

`training/` holds the env setup, NLI-pair converters, the finetuning
script, and the full results log: every question type routes through
entailment, so one finetuned checkpoint upgrades all three scorer paths
and loads via `--nli-model <dir>` with no Rust changes. See
[training/README.md](training/README.md) (includes the release checklist
and Hub-publish commands).

## License

MIT — see [LICENSE](LICENSE).

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

## HTTP API

`POST /v1/systemone` (path kept Jev-compatible). Body: `state`, optional
`model` (one of `embed|llm|nli|jev`, else the server default), and
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

## Checkpoints

Our finetuned NLI checkpoints ([full results table](training/README.md#holdout-results-master-table)):

- [raz-systemone-nli-xsmall](https://huggingface.co/RazvanManolache/raz-systemone-nli-xsmall)
  (v5, 283MB) — best balanced; at/near LLM-judge accuracy, CPU-only.
- [raz-systemone-nli-base](https://huggingface.co/RazvanManolache/raz-systemone-nli-base)
  (v7, 738MB) — best on broad queries; near-perfect on the 60-state split.

```powershell
hf download RazvanManolache/raz-systemone-nli-xsmall --local-dir nli-xsmall
raz ask --state-file examples/state.txt --questions examples/questions.json --scorer nli --nli-model nli-xsmall
```

## Scoreboard (70-state core eval, 199 judgments, labels fixed before any run)

| scorer | overall | choice | score | noul | ECE |
|---|---|---|---|---|---|
| jev (jev-latest, reference) | .89 | .97 | .89 | .84 | .07 |
| llm (phi4-abliterated 14B) | .89 | .97 | .81 | .91 | .09 |
| llm (mistral-nemo 12B) | .76 | .90 | .50 | .90 | .12 |
| llm (llama3.2:3b, 4k ctx) | .63 | .85 | .41 | .67 | .13 |
| nli (deberta-v3-xsmall) | .57 | .61 | .57 | .54 | .18 |
| nli (deberta-v3-base) | .51 | .76 | .44 | .36 | .29 |
| nli (DeBERTa-v3-base-mnli-fever-anli) | .51 | .58 | .60 | .36 | .25 |
| embed (nomic) | .49 | .70 | .57 | .23 | .36 |

Local phi4 ties Jev overall (.894 = 178/199 each) with opposite strengths:
phi4 wins urgency (.91 vs .84), Jev wins tone (.89 vs .81) and calibration.
phi4 was selected on 50 states and confirmed on a 20-state holdout (.883)
before becoming the default judge. A calm-No prompt example, an NLI blend,
and temperature scaling were all tried: the example shipped (it fixed
small-model Yes-bias), while the blend and temperatures fit noise on split
A and lost on split B, so they were removed rather than shipped.
Embed-noul sits exactly on the always-true rate: similarity saturates, it
never says no. ±1 judgment of run-to-run GPU noise on near-tie argmaxes.

Warm 3-question latency, CLI med-of-5: embed ~150ms, llm-phi4 ~150ms,
llm-3B ~160ms, jev-api ~370ms, NLI ~1.7s (per-process model load) /
~220ms served. Server NLI scales near-linearly to 4 concurrent
(lock-free) and costs ~400MB RAM; phi4 holds ~12GB VRAM (cold load ~3.5s).

## Showdown (all models on the same unseen holdouts C–G, macro = mean)

| model | size | C 60 | D 30 | E 30 | F 28 | G 29 | macro |
|---|---|---|---|---|---|---|---|
| jev (API reference) | cloud | .933 | .933 | .867 | .857 | .897 | **.897** |
| v7 (ours, base) | 184M | .983 | .867 | .833 | .857 | .897 | .887 |
| v5 (ours, xsmall) | 71M | .900 | .833 | .900 | .893 | .897 | .885 |
| v9 (ours, xsmall + Open-Jev) | 71M | .883 | .800 | .900 | .964 | .862 | .882 |
| v11 (ours, v9 + shreyanbr pairs) | 71M | .917 | .833 | .867 | .857 | .828 | .860 |
| phi4-abliterated (LLM judge) | 14B | .883 | .833 | .867 | .821 | .862 | .853 |
| mistral-nemo (LLM judge) | 12B | .683 | .700 | .633 | .679 | .759 | .687 |
| judge-3b-4k (LLM judge) | 3B | .533 | .667 | .500 | .679 | .690 | .614 |
| shreyanbr-gold (external xsmall) | 71M | .600 | .567 | .600 | .571 | .586 | .585 |
| embed (nomic) | — | .500 | .467 | .533 | .500 | .517 | .503 |
| base xsmall (zero-shot) | 71M | .400 | .433 | .333 | .679 | .483 | .466 |
| qyvos (official open Jev) | 144M | .467 | .400 | .600 | .464 | .241 | .434 |

Jev still leads overall, but our 71M v5/v9 beat it on E and F (v9 takes
F to .964) and trail by ~.01 macro — while beating the 14B phi4 judge on
every split. Choice is essentially solved (several perfect 1.00s);
frustration tone is the remaining gap. External references (qyvos,
shreyanbr-gold) confirm the pattern: everyone wins at home, nobody
travels — except v9, which also beats Qyvos on Open-Jev's own test
(.846 vs .831; see [training/README.md](training/README.md)).

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

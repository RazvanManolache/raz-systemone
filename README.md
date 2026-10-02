# systemone

Jev-like typed decisions over local models: send a state + typed
questions (`choice` / `score` / `noul`), get back probabilities, not text.

## Usage

```powershell
# one-shot CLI
cargo run -q -- ask --state-file examples/state.txt --questions examples/questions.json --scorer nli

# HTTP server (Jev-compatible POST /v1/systemone, + /v1/models, /healthz)
cargo run -q -- serve --port 18080 --scorer nli

# calibration eval on the labeled set (190 states, 547 judgments)
cargo run -q --release -- eval --data tests/data/tickets.jsonl --scorer llm
cargo run -q --release -- eval --data tests/data/tickets.jsonl --scorer llm --llm-model judge-3b-4k
# per-judgment JSONL dump for offline analysis/fitting
cargo run -q --release -- eval --data tests/data/tickets.jsonl --scorer llm --dump out.jsonl
```

Request shape (`model` picks the scorer, defaults to the server default):

```json
{
  "state": "The integration keeps failing, please help ASAP.",
  "model": "nli",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this",
      "criteria": {
        "billing": "Payment or subscription issues",
        "technical": "Bugs or integration problems",
        "sales": "Pricing or account questions"
      }
    }
  }
}
```

## Scorers

- `embed` (default for `ask`): cosine similarity + softmax. Fast, local,
  uncalibrated; cannot handle negation. `--embed-model` (default
  `nomic-embed-text`), `--temperature` (default 0.1).
- `llm`: a chat model answers a single-token constrained question; first-token
  logprobs become the distribution. Needs a model that emits the bare label
  first â€” chatty/thinking models (qwen3) fail loudly. `--llm-model`.
- `nli` (default for `serve`/`eval`): DeBERTa-v3 cross-encoder (MNLI/SNLI)
  reads state + hypothesis together in one batched forward pass; entailment
  (choice) or entail-vs-contra contrast (score/noul) becomes the score.
  No Ollama needed, CPU-only. `--nli-model` loads a local dir: the stock
  checkpoint (~90MB from HF Hub) or our finetuned ones (see Checkpoints).
- `jev`: the real TypeSafe API as a reference scorer. Key from
  `TYPESAFE_API_KEY` or a (git-ignored) `.env` file. `--jev-model`.

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
never says no. Â±1 judgment of run-to-run GPU noise on near-tie argmaxes.

Warm 3-question latency, CLI med-of-5: embed ~150ms, llm-phi4 ~150ms,
llm-3B ~160ms, jev-api ~370ms, NLI ~1.7s (per-process model load) /
~220ms served. Server NLI scales near-linearly to 4 concurrent
(lock-free candle) and costs ~400MB RAM; phi4 holds ~12GB VRAM
(cold load ~3.5s).

## Checkpoints

Our finetuned NLI checkpoints ([results table](training/README.md#holdout-results-master-table)):

- [systemone-nli-xsmall](https://huggingface.co/RazvanManolache/systemone-nli-xsmall)
  (v5, 283MB) â€” best balanced; at/near LLM-judge accuracy, CPU-only.
- [systemone-nli-base](https://huggingface.co/RazvanManolache/systemone-nli-base)
  (v7, 738MB) â€” best on broad queries; near-perfect on the 60-state split.

```powershell
hf download RazvanManolache/systemone-nli-xsmall --local-dir nli-xsmall
cargo run -q --release -- ask --state-file examples/state.txt --questions examples/questions.json --scorer nli --nli-model nli-xsmall
```

## Training your own checkpoint

`training/` holds the env setup, NLI-pair converters, the finetuning
script, and the full results log: every question type routes through
entailment, so one finetuned checkpoint upgrades all three scorer paths
and loads via `--nli-model <dir>` with no Rust changes. See
[training/README.md](training/README.md) (includes the release checklist
and Hub-publish commands).

## Tests

- `cargo test` â€” hermetic unit tests (math, logprob parsing, contrast, ECE).
- `cargo test -- --ignored` â€” live tests (Ollama scorers, NLI, HTTP server).

## Notes

- `judge-3b-4k` is a local Ollama variant (`num_ctx 4096`) of
  llama3.2:3b-instruct-fp16: 23GB â†’ 7GB, 3x faster, identical verdicts.
- Port 8080 is blocked on this machine; examples use 18080.

## Layout

- `src/` + `tests/` + `examples/` â€” the Rust runner (CLI + server + eval).
- `tests/data/` â€” labels (`tickets.jsonl`) and train/holdout splits.
- `training/` â€” trainer (`data.py`, `train.py`, `upload.py`), model cards,
  and the results log. `training/runs/` (checkpoints, MNLI) is git-ignored.

## License

MIT â€” see [LICENSE](LICENSE).

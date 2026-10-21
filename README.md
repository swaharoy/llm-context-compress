# LLM Context Compression via Bloom Filters

COMP 480/580 Assignment 1. A single hand-written Bloom filter stores a fixed,
context-independent vocabulary of low-meaning words (stopwords, optionally the most
common English words) and decorative markdown tokens. Each context is compressed in
one pass: split on whitespace and drop every token the filter reports as present.
The compressed context and the question go to `gemini-3.5-flash-lite`. The design
is in [`specs.md`](specs.md).

> Status: all stages are done. The report is `report/report.pdf` (source `report/report.tex`; build with `tectonic report/report.tex`).
> The Bloom-filter LLM runs and the tokens-vs-accuracy plot (stage 5) are next.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env             # then set GEMINI_API_KEY (needed for token counting and LLM runs)
```

## Commands

Every data file is passed explicitly; nothing is discovered from folders. Files pair by
position: the i-th context, questions file and answers file must be the same document.
`experiments.py` checks that each questions/answers pair has the same `doc_id` and fails
before any API call if the order is wrong. Held-out data uses the same flags.

```bash
# Reproduce every experiment and plot (the TA command)
.venv/bin/python experiments.py --mode all \
    --contexts  assignment/contexts/antikythera_dossier_v1.txt \
                assignment/contexts/dwh_dossier_v1.txt \
                assignment/contexts/harbour_puzzle_v1.txt \
                assignment/contexts/lunarleaper_2609_11453.txt \
    --questions assignment/questions/antikythera_dossier_v1_questions.json \
                assignment/questions/dwh_dossier_v1_questions.json \
                assignment/questions/harbour_puzzle_v1_questions.json \
                assignment/questions/lunarleaper_2609_11453_questions.json \
    --answers   assignment/answers/antikythera_dossier_v1_answers.json \
                assignment/answers/dwh_dossier_v1_answers.json \
                assignment/answers/harbour_puzzle_v1_answers.json \
                assignment/answers/lunarleaper_2609_11453_answers.json

# Same flags, other modes:
#   --mode stats_only   Bloom-filter grid: memory, FPR, retained context, token counts (no answers generated)
#   --mode llm_only     LLM runs: baseline, each vocabulary tier, and the FPR sweep, graded

# One configuration on one document
.venv/bin/python ask_llm.py \
    --contexts assignment/contexts/dwh_dossier_v1.txt \
    --questions assignment/questions/dwh_dossier_v1_questions.json \
    --size 4096 --hash-count 6 --tier curated --calc-fpr --out response.json

# Full-context baseline (no compression, same prompt and model)
.venv/bin/python ask_llm.py --contexts ... --questions ... --baseline
```

`ask_llm.py` flags:

| Flag | Meaning |
|---|---|
| `--contexts`, `--questions` | paired lists of files, matched by position |
| `--size`, `--hash-count` | Bloom filter bits R and hash count k |
| `--tier` | vocabulary: `curated`, `nltk_full`, `common500`, `common1000`, `common2000`, or `common5000` |
| `--stats-only` | skip answer generation; count tokens with `count_tokens` |
| `--calc-fpr` | also build an exact `set` to measure empirical FPR and memory vs. the set |
| `--baseline` | no compression |

`ask_llm.run(...)` takes the same parameters when imported.

## Outputs

| File | Contents |
|---|---|
| `response.json` | one `ask_llm` run: config, filter stats, per-document compression stats, per-question tokens, latency, and answers |
| `results/stats_grid.json` | stats-only grid over tier × R × k |
| `results/llm_runs.json` | every LLM run with grades |
| `results/summary.csv` | one row per LLM run: memory, FPR, retained %, tokens, latency, accuracy |
| `results/plots/*.png` | BF memory vs FPR, BF memory vs input tokens, input tokens vs accuracy, and diagnostics |

## Layout

```
ask_llm.py       vocabulary tiers, BloomFilter, compress(), prompt, Gemini client + cache, run()
experiments.py   data-file checks, evaluator (the only reader of answer files), experiment grids, summary, plots
specs.md         design spec
assignment/      contexts, questions, answers, assignment PDF
```

## Development

Unit tests are a development aid and are not part of the submission:

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

Tests never call the network: the LLM and token counter are replaced by a fake client.

## Reproducibility

- Hashing uses keyed `blake2b` (seed 580) with double hashing, never Python's
  per-process salted `hash()`, so filter bits are identical across runs and machines.
- Vocabulary lists are hard-coded or come from a pinned `wordfreq` version.
- Gemini runs use temperature 0 and seed 580. Responses and token counts are
  cached under `cache/`, so reruns are free and identical. Cached replays are
  flagged `cached: true`, because their latency is not a fresh measurement.

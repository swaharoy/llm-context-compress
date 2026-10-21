# LLM Context Compression via Bloom Filters

COMP 480/580 Assignment 1. Each document is cut into chunks, and every chunk gets
its own hand-written Bloom filter of its terms. For each question, the chunks whose
filters best match the question's terms are sent to `gemini-3.5-flash-lite` instead
of the whole document. The design is in [`specs.md`](specs.md).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export GEMINI_API_KEY=...        # only needed for the LLM stages
```

## Commands

```bash
# Tests (no network)
.venv/bin/python -m pytest -q

# Every experiment that needs no LLM: R×k grid, measured vs theoretical FPR, memory,
# retrieval-only sweep/ablations/budgets, bit-sliced equivalence + benchmark,
# a FakeLLM dry run of the full pipeline, and plots (a), (b-estimated), (d), (e).
.venv/bin/python src/main.py experiments --offline-only

# The paid study: stage 1 smoke test, stage 2 baseline + R sweep + exact control,
# stage 3 ablations + budgets. Reruns are free (disk cache).
.venv/bin/python src/main.py experiments --stage all

# Fresh calls for reported latency (bypasses the cache)
.venv/bin/python src/main.py experiments --stage 2 --no-cache --run-name llm_fresh

# One document, ad hoc
.venv/bin/python src/main.py ask --context assignment/contexts/dwh_dossier_v1.txt \
    --question "How many sorties were flown to release the dispersant?" --print
```

Results go to `results/<run-name>/` (`offline` or `llm` by default):

| File | Contents |
|---|---|
| `config.json` | configs, seed, git commit, package versions |
| `index_stats.jsonl` | per (doc, config): memory, fill, measured / theoretical FPR, build time |
| `retrieval_grid.jsonl`, `retrieval_configs.jsonl` | per (doc, question, config) offline retrieval: kept chunks, estimated tokens, gold-span recall |
| `records.jsonl`, `graded.jsonl` | per (doc, question, config) LLM run: real tokens, latency, answer, grade |
| `summary*.csv` | means overall, per document, tier and lexical overlap, with bootstrap 95% CIs |
| `bitsliced_check.json` | bit-sliced vs row-major equivalence and query benchmark |
| `plots/*.png` (+ `.csv` of plotted points) | required plots (a)–(c) and diagnostics |

## Layout

```
src/
  main.py         CLI
  bloom.py        hashing, BloomFilter, RowMajor / BitSliced / ExactSet chunk indexes, FPR measurement
  text.py         structure-aware chunking and term extraction
  retriever.py    scoring through the index, budget-aware selection, context assembly, full-context baseline
  llm.py          fixed prompt, Gemini client, fake client, disk cache
  driver.py       per-document pipeline: prepare -> retrieve -> answer -> records (with resume)
  evaluate.py     grading against the answer files, gold-span recall, summaries
  experiments.py  presets and the staged experiment plan
  plots.py        charts
tests/            one test file per module, plus a check that the pipeline never reads gold answers
```

**Why 9 files instead of the suggested 2.** Each file is one concern that can be
tested on its own. The graded data structure (`bloom.py`) and the compression
method (`text.py`, `retriever.py`) are independent of the experiment harness
(`llm.py`, `driver.py`, `experiments.py`). Only `evaluate.py` reads the answer
files, and `tests/test_no_gold_leak.py` enforces that.

## Reproducibility notes

- Hashing uses keyed `blake2b` (seed 580), never Python's per-process salted
  `hash()`. A test checks that positions are identical across processes.
- All random sampling (FPR probes, bootstrap) is seeded.
- LLM responses are cached by `sha256(model | generation config | prompt)`.
  Cached replays are flagged `cached=true` in the records, and their latency
  is not a fresh measurement.
- Offline token counts are the estimate `ceil(chars / 4)`. Real counts come
  from the API's `usage_metadata` in the LLM stages.

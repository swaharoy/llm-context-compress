# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

COMP 480/580 Assignment 1 (due 2026-10-21): use **Bloom filters** to compress long LLM contexts before question answering, and study the tradeoff between Bloom filter memory, false-positive rate, retained context, LLM input tokens, latency, and QA accuracy. The full spec is `assignment/assignment_description.pdf`; `assignment/README.md` is a shorter summary of it.

**Method:** one hand-written Bloom filter holds a fixed, context-independent vocabulary of low-meaning words, plus decorative markdown tokens. Compression is one pass: split the context on whitespace and drop every token the filter matches. Vocabulary **tiers** (`curated`, `nltk_full`, `common500`–`common5000`) are a tuning knob beside R and k.

`specs.md` is the source of truth for the design. `specs-v1.md` is the user's original rough plan, so keep it unchanged. `README.md` has the commands.

## Code layout: two Python files only (professor's requirement)

- `ask_llm.py`: vocabulary lists and tiers, `BloomFilter`, `compress()`, the prompt, the Gemini client with its disk cache, stats, `run()`, and a CLI with the same parameters.
- `experiments.py`: the evaluator (the **only** code that reads `assignment/answers/`), the stats-only and LLM grids, the summary, plots, and a CLI (`--mode stats_only|llm_only|all`).

Commands (see README):
- everything: `.venv/bin/python experiments.py --mode all --contexts ... --questions ... --answers ...` (the full command with the four v1 files is in README). The data files are **always passed explicitly** on the command line and paired by position; never discover them by scanning folders (TAs run it on held-out files).
- one config: `.venv/bin/python ask_llm.py --contexts C.txt --questions Q.json --size 4096 --hash-count 6 --tier curated [--stats-only] [--calc-fpr] [--baseline]`

**Tests:** unit tests live in `tests/` (`test_ask_llm.py`, `test_experiments.py`, plus a `conftest.py` that puts the repo root on `sys.path`). They are a development aid and are **not submitted**, so the two-file limit applies only to the submitted code.
- Install with `.venv/bin/pip install -r requirements-dev.txt`.
- Run all: `.venv/bin/python -m pytest -q`. Run one: `.venv/bin/python -m pytest tests/test_ask_llm.py::test_no_false_negatives`.
- Every new function gets tests (the plan is in `specs.md` §12).
- Tests never use the network: pass a fake client to `run(client=...)`.
- Each development stage's tests must pass before moving on.

**Prompt:** a single template built by `build_prompt()`, with no system instruction, so stats-only `count_tokens` equals the real `prompt_token_count`. It uses `question`, `answer_type` (with a generic hint), `unit` (only when present) and `max_answer_tokens`. It never uses `qid`. See `specs.md` §6.

## Hard requirements from the spec

- **Implement the Bloom filter yourself** (insert + membership query). Existing hash functions (`hashlib`) are allowed, but Bloom filter libraries are not.
- **Model:** `gemini-3.5-flash-lite` for every experiment. Keep the model, prompt template, and generation config identical between the full-context baseline and every compressed run.
- **Generality:** the code is also graded on held-out contexts and questions. Build the vocabulary only from fixed general-English lists, with no rules specific to these documents. Never read `answers/` at compression time.
- **Sweep R (bits) and k (hash count)**, and report measured FPR against the theoretical value `(1 - e^{-kN/R})^k`. The optimal k is `(R/N)·ln 2`.
- **Metrics:** accuracy, total input tokens (context + question + instructions), latency (compression + generation), and memory (the Bloom filter, compared with an exact `set` via `pympler`).
- **Required plots:** BF memory vs FPR, BF memory vs LLM input tokens, and LLM input tokens vs accuracy.
- **Submission:** 2 Python files, a pinned `requirements.txt`, a README with the exact run command, and a short report that answers the six questions in §4 of the PDF.
- **Reproducibility:** TAs re-run everything. SEED = 580 is used for keyed `blake2b` hashing and the LLM, and Python's salted `hash()` is never used. LLM responses and token counts are cached under `cache/`. Everything runs end to end with no manual steps.

## Data layout (`assignment/`)

Use only the four v1 contexts. Each has exactly 20 questions, for 80 in total. The contexts are about 90–206 KB of text, roughly 15K–50K tokens each:
`antikythera_dossier_v1`, `dwh_dossier_v1`, `harbour_puzzle_v1`, `lunarleaper_2609_11453`.

- `contexts/<doc_id>.txt` holds the raw document. Tables are written as cells joined by ` | `, and headings as `## Title`.
- `questions/<doc_id>_questions.json` has top-level `{doc_id, count, questions: [...]}`. Each question has `qid`, `question`, `answer_type` (`number|entity|phrase|date`), `unit`, and `max_answer_tokens`.
- `answers/<doc_id>_answers.json` has top-level `{answers: [...]}`, keyed by `qid`. Each answer has:
  - `expected` and `aliases`
  - `grading.normaliser`, either `numeric` (relative tolerance `grading.numeric_tolerance_rel`) or `exact_ci` (case-insensitive exact match)
  - `tier`, one of `T1_lookup` (44), `T2_multihop` (24), or `T3_aggregate` (12)

Note: some `source_document` and `generated_from` fields inside the JSON point to stale paths such as `corpus/...`. Resolve files by `doc_id` instead.

## Repo state

- The old 9-file `src/` design (per-chunk Bloom indexes plus retrieval) was deleted. It remains in git history, and `git show HEAD:src/evaluate.py` and `HEAD:src/llm.py` are worth porting from.
- The data was moved from the repo root into `assignment/`, and that move is uncommitted.

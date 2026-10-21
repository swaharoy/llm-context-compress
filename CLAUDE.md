# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

COMP 480/580 Assignment 1 (due 2026-10-21): use **Bloom filters** to compress long LLM contexts before question answering, and study the tradeoff between Bloom filter memory, false-positive rate, retained context, LLM input tokens, latency, and QA accuracy. The full spec is `assignment/assignment_description.pdf`; `assignment/README.md` is a shorter summary of it.

The design is in `specs.md` (the source of truth), and the code is in `src/`, following its module layout. `README.md` has the commands:

- tests: `.venv/bin/python -m pytest -q`; a single test: `.venv/bin/python -m pytest tests/test_bloom.py::test_no_false_negatives`
- offline experiments, no LLM: `.venv/bin/python src/main.py experiments --offline-only`
- paid Gemini runs: `.venv/bin/python src/main.py experiments --stage all`

Modules import each other flat from `src/` (tests add it to `sys.path` in `tests/conftest.py`). Only `evaluate.py` may read `assignment/answers/`.

## Hard requirements from the spec

- **Implement the Bloom filter yourself** (insert + membership query). Existing hash functions are allowed, but Bloom filter libraries are not. The spec rewards creative, efficient designs.
- **Model:** `gemini-3.5-flash-lite` for every experiment. Keep the model and prompt template identical between the full-context baseline and every compressed run, so differences come only from the compression.
- **Baseline:** `LLM(full context + question)`. **Method:** `full context → Bloom-filter selection → compressed context + question → LLM`.
- **Generality:** the code is also graded on held-out contexts and questions. Don't add rules specific to these documents, and never read `answers/` fields (`gold_spans`, `retrieval_keys`, etc.) at compression time. The answer files are only for scoring.
- **Sweep R (bits) and k (hash count)**, and report measured FPR against the theoretical value `(1 - e^{-kN/R})^k`. The optimal k is `(R/N)·ln 2`.
- **Metrics:** accuracy, total input tokens (context + question + instructions), latency (retrieval + generation), and memory (the Bloom filter(s) plus any other data structures).
- **Required plots:** BF memory vs FPR, BF memory vs LLM input tokens, and LLM input tokens vs accuracy.
- **Submission:** ideally ≤2 Python files, `requirements.txt`, a README with the exact run command, and a short report that answers the six questions in §4 of the PDF.
- **Reproducibility:** TAs re-run everything. Fix all seeds, and make everything run end to end with no manual steps.

## Data layout (`assignment/`)

Use only the four v1 contexts. Each has exactly 20 questions, for 80 in total. The contexts are about 90–206 KB of text, roughly 15K–50K tokens each:
`antikythera_dossier_v1`, `dwh_dossier_v1`, `harbour_puzzle_v1`, `lunarleaper_2609_11453`.

- `contexts/<doc_id>.txt` holds the raw document.
- `questions/<doc_id>_questions.json` has top-level `{doc_id, count, questions: [...]}`. Each question has `qid`, `question`, `answer_type` (`number|entity|phrase|date`), `unit`, and `max_answer_tokens`.
- `answers/<doc_id>_answers.json` has top-level `{answers: [...]}`, keyed by `qid`. Each answer has:
  - `expected` and `aliases`, which also appear as `gold_answer` and `accepted_aliases`
  - `grading.normaliser`, either `numeric` (relative tolerance `grading.numeric_tolerance_rel`, e.g. 0.005) or `exact_ci` (case-insensitive exact match against expected or aliases)
  - `tier`, one of `T1_lookup` (44), `T2_multihop` (24), or `T3_aggregate` (12). Multi-hop and aggregate questions need several separated spans, so naive top-1 chunk selection will hurt them.
  - `known_distractor_spans`, which describe near-miss values (e.g. sibling table rows). Aggressive compression that keeps the wrong neighbouring rows causes errors here.

Note: some `source_document` and `generated_from` fields inside the JSON point to stale paths such as `corpus/...`. Resolve files by `doc_id` instead.

## Repo state

The data was moved from the repo root into `assignment/`, and that move is uncommitted. Git still tracks the old root-level `contexts/`, `questions/`, and `answers/` paths, which now show as deleted.

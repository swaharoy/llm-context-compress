"""Command-line entry point (specs.md §11).

Usage::

    python src/main.py ask --context assignment/contexts/dwh_dossier_v1.txt \\
        --question "How many sorties were flown?" --print
    python src/main.py experiments --offline-only     # no LLM calls
    python src/main.py experiments --stage all        # paid Gemini runs

``ask`` runs one document and its question(s) through the driver. ``experiments``
reproduces the study in :mod:`experiments`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from bloom import BloomParams
from driver import Document, Question, RunConfig, prepare, run_document
from experiments import CACHE_DIR, RESULTS_DIR, SEED, run_llm_stages, run_offline
from llm import CachedLLM, FakeLLM, GeminiClient
from retriever import RetrievalParams


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bloom-filter context compression for LLM question answering.")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="answer question(s) about one document")
    ask.add_argument("--context", type=Path, required=True, help="path to the context .txt file")
    source = ask.add_mutually_exclusive_group(required=True)
    source.add_argument("--question", help="a single question")
    source.add_argument("--questions", type=Path, help="a questions JSON file (same format as assignment/questions)")
    ask.add_argument("--full", action="store_true", help="send the full context (baseline) instead of compressing")
    ask.add_argument("--index", default="rowmajor", choices=["rowmajor", "bitsliced", "exact"])
    ask.add_argument("--R", type=int, default=1024, help="bits per chunk filter (power of two)")
    ask.add_argument("--k", default="auto", help="hash count, or 'auto' for the optimal k")
    ask.add_argument("--budget", type=int, default=1500, help="token budget for the compressed context")
    ask.add_argument("--print", dest="print_answers", action="store_true", help="print each answer")
    ask.add_argument("--out", type=Path, default=RESULTS_DIR / "ask", help="directory for records.jsonl")
    ask.add_argument("--fake", action="store_true", help="use the offline FakeLLM instead of Gemini")
    ask.add_argument("--no-cache", action="store_true", help="always call the model")

    exp = sub.add_parser("experiments", help="run the experiment plan")
    exp.add_argument("--stage", default="all", choices=["1", "2", "3", "all"])
    exp.add_argument("--offline-only", action="store_true", help="run only the stages that need no LLM")
    exp.add_argument("--no-cache", action="store_true", help="fresh LLM calls, for latency measurements")
    exp.add_argument("--run-name", default=None, help="results/<run-name>/ (default: 'offline' or 'llm')")
    return parser.parse_args(argv)


def load_questions(args: argparse.Namespace) -> tuple[Question, ...]:
    if args.question:
        # Derive the id from the text, so a new question is never mistaken for an already recorded one.
        qid = "q-" + hashlib.sha1(args.question.encode("utf-8")).hexdigest()[:10]
        return (Question(qid, args.question, "unknown", None),)
    raw = json.loads(args.questions.read_text(encoding="utf-8"))
    return tuple(Question(q["qid"], q["question"], q.get("answer_type", ""), q.get("unit")) for q in raw["questions"])


def ask(args: argparse.Namespace) -> None:
    document = Document(args.context.stem, args.context.read_text(encoding="utf-8"), load_questions(args))
    k = None if args.k == "auto" else int(args.k)
    config = RunConfig(
        name="ask-full" if args.full else "ask",
        mode="full" if args.full else "bloom",
        index=args.index,
        bloom=BloomParams(R=args.R, k=k, seed=SEED),
        retrieval=RetrievalParams(budget_tokens=args.budget),
    )
    inner = FakeLLM() if args.fake else GeminiClient(seed=SEED)
    llm = CachedLLM(inner, CACHE_DIR, enabled=not args.no_cache)
    records = run_document(prepare(document, config), llm, args.out / "records.jsonl")
    if args.print_answers:
        for r in records:
            print(f"{r.qid}: {r.answer_raw}   ({r.prompt_tokens} input tokens, {r.retained_frac:.1%} of context)")
    print(f"{len(records)} new record(s) written to {args.out / 'records.jsonl'}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "ask":
        ask(args)
    elif args.offline_only:
        result = run_offline(RESULTS_DIR / (args.run_name or "offline"))
        print(json.dumps(result, indent=2))
    else:
        run_llm_stages(args.stage, RESULTS_DIR / (args.run_name or "llm"), no_cache=args.no_cache)


if __name__ == "__main__":
    main()

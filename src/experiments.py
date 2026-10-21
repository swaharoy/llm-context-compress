"""The experiment plan of specs.md §10: presets, staged runs and result files.

There are two kinds of runs:

* **Offline** (:func:`run_offline`, no LLM and no network):
  - the R×k grid, with measured FPR, memory, retained tokens and gold-span recall;
  - the bit-sliced equivalence check and benchmark;
  - retrieval-only versions of the sweep, ablations and budgets;
  - a FakeLLM dry run of the whole pipeline;
  - the plots that need no accuracy.
* **LLM stages** (:func:`run_llm_stages`, Gemini): the baseline, the R sweep,
  ablations and budgets, then grading, summaries and the accuracy plot.

All presets are fixed here before any LLM run. Ablations are reported comparisons
and never select the final configuration (§6.3).

Depends on: every other module in ``src/``.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import time
from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

import plots
from bloom import BitSlicedBloomIndex, BloomFilter, BloomParams, make_fpr_probes, measure_fpr, resolve_k
from driver import (DATA_DIR, Document, PreparedDocument, RunConfig, list_documents, load_document,
                    prepare, retrieve_all, run_document)
from evaluate import Gold, gold_span_recall, grade, load_gold, summarise
from llm import (MAX_CONTEXT_TOKENS, SYSTEM_INSTRUCTION, CachedLLM, FakeLLM, GeminiClient, LLMClient,
                 build_prompt)
from retriever import BloomRetriever, RetrievalParams, estimate_tokens
from text import ChunkParams, TermParams, extract_terms

SEED = 580
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / "cache" / "llm"

# --------------------------------------------------------------------------- #
# Presets (frozen before any LLM run)
# --------------------------------------------------------------------------- #

BASELINE = RunConfig("full", mode="full")
CORE = RunConfig("core")  # dataclass defaults are the core method: R=1024, optimal k, budget 1500

ABLATIONS = [
    replace(CORE, name="abl_fixed_chunks", chunk=ChunkParams(strategy="fixed")),
    replace(CORE, name="abl_words_only", terms=TermParams(bigrams=False, numbers=False)),
    replace(CORE, name="abl_uniform_weights", retrieval=RetrievalParams(idf_weighting=False)),
    replace(CORE, name="abl_char_grams", terms=TermParams(char_grams=True)),
]
BUDGETS = [replace(CORE, name=f"budget_{b}", retrieval=RetrievalParams(budget_tokens=b)) for b in (500, 3000)]

SWEEP_R = (64, 128, 256, 512, 1024, 4096)  # LLM sweep points, each with optimal k
SWEEP = [CORE if R == CORE.bloom.R else replace(CORE, name=f"R{R}", bloom=BloomParams(R=R)) for R in SWEEP_R]
EXACT = replace(CORE, name="exact", index="exact")

GRID_R = (64, 128, 256, 512, 1024, 2048, 4096)
GRID_K = (1, 2, 3, 4, 6, 8, None)  # None means optimal k
N_FPR_PROBES = 1000


def k_label(k: int | None) -> str:
    return "opt" if k is None else str(k)


# --------------------------------------------------------------------------- #
# Small I/O helpers
# --------------------------------------------------------------------------- #


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_run_config(run_dir: Path, configs: list[RunConfig], extra: dict) -> None:
    """Record exactly what ran: configs, seed, git commit and package versions."""
    def package(name: str) -> str:
        try:
            return version(name)
        except PackageNotFoundError:
            return "not installed"

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=False).stdout.strip()
    except OSError:
        commit = ""
    info = {
        "seed": SEED,
        "git_commit": commit,
        "python": platform.python_version(),
        "packages": {p: package(p) for p in ("numpy", "matplotlib", "google-genai")},
        "configs": [c.to_dict() for c in configs],
        **extra,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(info, indent=2))


def load_documents(data_dir: Path) -> list[Document]:
    return [load_document(doc_id, data_dir) for doc_id in list_documents(data_dir)]


def estimated_prompt_tokens(context: str, question) -> int:
    """Return the offline estimate of the full prompt size, with the same template and system instruction as real runs."""
    prompt = build_prompt(context, question.question, question.answer_type, question.unit)
    return estimate_tokens(SYSTEM_INSTRUCTION + "\n" + prompt)


def retrieval_rows(prepared: PreparedDocument, gold: dict[str, Gold]) -> list[dict]:
    """Run retrieval for every question and describe each result, with no LLM involved."""
    stats = prepared.index_stats
    rows = []
    for r in retrieve_all(prepared):
        context = r.selection.text
        rows.append({
            "config": prepared.config.name,
            "doc_id": prepared.document.doc_id,
            "qid": r.question.qid,
            "index": stats.get("index"),
            "R": stats.get("R"),
            "k": stats.get("k"),
            "k_label": k_label(prepared.config.bloom.k),
            "selection_status": r.selection.status,
            "retained_chunk_ids": list(r.selection.chunk_ids),
            "retained_frac": r.selection.retained_frac,
            "context_tokens": r.selection.est_tokens,
            "prompt_tokens": estimated_prompt_tokens(context, r.question),
            "query_fp_hits": r.query_fp_hits,
            "retrieval_s": r.retrieval_s,
            "gold_span_recall": gold_span_recall(context, gold[r.question.qid]),
        })
    return rows


# --------------------------------------------------------------------------- #
# Offline stages (no LLM)
# --------------------------------------------------------------------------- #


def offline_grid(docs: list[Document], gold: dict[str, Gold], out: Path) -> tuple[list[dict], list[dict]]:
    """Stage 2 step 2: every (R, k) plus the exact control, measuring the filter and retrieval on all questions."""
    index_rows: list[dict] = []
    grid_rows: list[dict] = []
    configs = [
        replace(CORE, name=f"grid_R{R}_k{k_label(k)}", bloom=BloomParams(R=R, k=k, seed=SEED))
        for R in GRID_R for k in GRID_K
    ] + [EXACT]
    for doc in docs:
        probes = None
        for config in configs:
            prepared = prepare(doc, config)
            if probes is None:  # same chunks and terms for every grid config of this document
                probes = make_fpr_probes(prepared.chunk_terms, N_FPR_PROBES, SEED)
            fpr = measure_fpr(prepared.index, probes)
            index_rows.append({"config": config.name, "doc_id": doc.doc_id,
                               "k_label": k_label(config.bloom.k) if config.index != "exact" else "exact",
                               **prepared.index_stats, "fpr_measured": fpr})
            grid_rows += retrieval_rows(prepared, gold)
        print(f"  grid: {doc.doc_id} done ({len(configs)} configs)")
    write_jsonl(out / "index_stats.jsonl", index_rows)
    write_jsonl(out / "retrieval_grid.jsonl", grid_rows)
    return index_rows, grid_rows


def offline_configs(docs: list[Document], gold: dict[str, Gold], out: Path) -> list[dict]:
    """Retrieval-only runs of the baseline, sweep, exact control, ablations and budgets, summarised by recall and tokens."""
    rows: list[dict] = []
    for config in [BASELINE, *SWEEP, EXACT, *ABLATIONS, *BUDGETS]:
        for doc in docs:
            rows += retrieval_rows(prepare(doc, config), gold)
    write_jsonl(out / "retrieval_configs.jsonl", rows)
    summary = summarise(rows, ["gold_span_recall", "prompt_tokens", "retained_frac", "query_fp_hits"], gold)
    write_csv(out / "summary_retrieval.csv", summary)
    return summary


def bitsliced_check(docs: list[Document], out: Path) -> dict:
    """Stage 2 step 5: the bit-sliced index must reproduce row-major scores and selections exactly.

    Also benchmarks per-term query time for three implementations: a pure-Python
    loop over separate filters, the vectorised row-major index, and the bit-sliced index.
    """
    mismatches = 0
    checked = 0
    timings = {"rowmajor": 0.0, "bitsliced": 0.0}
    for config in SWEEP:
        for doc in docs:
            row = prepare(doc, config)
            sliced = prepare(doc, replace(config, index="bitsliced"))
            for q in doc.questions:
                a, b = row.selector.score(q.question), sliced.selector.score(q.question)
                same_scores = np.array_equal(a.values, b.values)
                same_ids = row.selector.select(q.question).chunk_ids == sliced.selector.select(q.question).chunk_ids
                mismatches += not (same_scores and same_ids)
                checked += 1
            timings["rowmajor"] += row.build_s
            timings["bitsliced"] += sliced.build_s
    result = {"questions_checked": checked, "mismatches": mismatches,
              "build_s_total": timings, "query_benchmark": _query_benchmark(docs)}
    (out / "bitsliced_check.json").write_text(json.dumps(result, indent=2))
    return result


def _query_benchmark(docs: list[Document], repeats: int = 3) -> dict:
    """Return the mean microseconds per term query at the core R, over every question term of every document."""
    totals = {"python_loop": 0.0, "rowmajor": 0.0, "bitsliced": 0.0}
    n_queries = 0
    for doc in docs:
        prepared = prepare(doc, CORE)
        params = replace(CORE.bloom, k=resolve_k(CORE.bloom, prepared.chunk_terms))
        filters = []
        for terms in prepared.chunk_terms:
            f = BloomFilter(params.R, params.k, params.seed)
            f.add_many(terms)
            filters.append(f)
        sliced = BitSlicedBloomIndex(prepared.chunk_terms, params)
        terms = [t for q in doc.questions for t in extract_terms(q.question, CORE.terms).all_terms()]
        n_queries += len(terms)
        for name, query in (
            ("python_loop", lambda t: [t in f for f in filters]),
            ("rowmajor", prepared.index.query),
            ("bitsliced", sliced.query),
        ):
            start = time.perf_counter()
            for _ in range(repeats):
                for t in terms:
                    query(t)
            totals[name] += (time.perf_counter() - start) / repeats
    per_query_us = {name: 1e6 * total / max(n_queries, 1) for name, total in totals.items()}
    return {"n_term_queries": n_queries, "us_per_query": per_query_us,
            "speedup_bitsliced_vs_rowmajor": per_query_us["rowmajor"] / per_query_us["bitsliced"],
            "speedup_bitsliced_vs_python": per_query_us["python_loop"] / per_query_us["bitsliced"]}


def dry_run(docs: list[Document], gold: dict[str, Gold], out: Path) -> dict:
    """Run the whole LLM pipeline with FakeLLM, to check records, grading and resume. The accuracy means nothing."""
    records_path = out / "records.jsonl"
    if records_path.exists():
        records_path.unlink()
    fake = FakeLLM()
    llm = CachedLLM(fake, out / "cache")
    prepared = run_configs([BASELINE, CORE], docs, llm, records_path, workers=1)
    first_calls = fake.calls
    run_configs([BASELINE, CORE], docs, llm, records_path, workers=1)  # resume: nothing is left to do
    graded = grade_records(read_jsonl(records_path), prepared, gold)
    write_jsonl(out / "graded.jsonl", graded)
    summary = summarise(graded, ["correct_strict", "correct_lenient", "prompt_tokens", "gold_span_recall"], gold)
    write_csv(out / "summary.csv", summary)
    return {"records": len(graded), "llm_calls_first_run": first_calls,
            "llm_calls_on_resume": fake.calls - first_calls}


def run_offline(run_dir: Path, data_dir: Path = DATA_DIR) -> dict:
    """Run every stage that needs no LLM, then draw the plots that need no accuracy."""
    docs, gold = load_documents(data_dir), load_gold(data_dir)
    write_run_config(run_dir, [BASELINE, CORE, *SWEEP, EXACT, *ABLATIONS, *BUDGETS],
                     {"mode": "offline", "token_counts": "estimated as ceil(chars/4)"})
    print("Offline grid (Stage 2, step 2)...")
    index_rows, grid_rows = offline_grid(docs, gold, run_dir)
    print("Retrieval-only configs (sweep, exact, ablations, budgets)...")
    summary = offline_configs(docs, gold, run_dir)
    print("Bit-sliced equivalence and benchmark (Stage 2, step 5)...")
    sliced = bitsliced_check(docs, run_dir)
    print("FakeLLM dry run of the full pipeline...")
    dry = dry_run(docs, gold, run_dir / "dry_run")

    baseline = next(r for r in summary if r["config"] == "full" and r["group_by"] == "all")
    plot_dir = run_dir / "plots"
    plots.plot_memory_vs_fpr(index_rows, plot_dir / "a_memory_vs_fpr.png")
    plots.plot_memory_vs_tokens(index_rows, grid_rows, plot_dir / "b_memory_vs_tokens_estimated.png",
                                baseline["prompt_tokens"], estimated=True)
    plots.plot_recall_vs_fpr(index_rows, grid_rows, plot_dir / "d_recall_vs_fpr.png")
    plots.plot_config_bars(summary, plot_dir / "e_configs_recall_tokens.png",
                           ("gold_span_recall", "prompt_tokens"),
                           ("Mean gold-span recall", "Mean estimated input tokens"),
                           order=["full", "exact", *[c.name for c in reversed(SWEEP)],
                                  *[c.name for c in BUDGETS], *[c.name for c in ABLATIONS]])
    return {"bitsliced": sliced, "dry_run": dry}


# --------------------------------------------------------------------------- #
# LLM stages
# --------------------------------------------------------------------------- #


def run_configs(configs: list[RunConfig], docs: list[Document], llm: LLMClient, records_path: Path,
                workers: int = 4) -> dict[tuple[str, str], PreparedDocument]:
    """Run each config on each document. Returns the prepared documents, which grading needs."""
    prepared_by_key = {}
    for config in configs:
        for doc in docs:
            prepared = prepare(doc, config)
            run_document(prepared, llm, records_path, workers)
            prepared_by_key[(config.name, doc.doc_id)] = prepared
    return prepared_by_key


def grade_records(records: list[dict], prepared_by_key: dict, gold: dict[str, Gold]) -> list[dict]:
    """Add grades and gold-span recall to each record.

    The context a record was answered with is rebuilt from its ``retained_chunk_ids``.
    Assembly is deterministic, so this matches the prompt exactly without storing contexts.
    """
    graded = []
    for record in records:
        prepared = prepared_by_key[(record["config"], record["doc_id"])]
        if isinstance(prepared.selector, BloomRetriever):
            context = prepared.selector.assemble(record["retained_chunk_ids"])
        else:
            context = prepared.document.text
        g = gold[record["qid"]]
        result = grade(record["answer_raw"], g)
        graded.append({
            **record, **asdict(result),
            "k_label": k_label(prepared.config.bloom.k) if record["index"] == "rowmajor" else record["index"],
            "gold_span_recall": gold_span_recall(context, g),
        })
    return graded


def preflight(docs: list[Document], client: GeminiClient) -> None:
    """Fail loudly if any full-context prompt would exceed the model's context window (§10.1)."""
    for doc in docs:
        q = doc.questions[0]
        n = client.count_tokens(SYSTEM_INSTRUCTION + "\n" + build_prompt(doc.text, q.question, q.answer_type, q.unit))
        if n > MAX_CONTEXT_TOKENS:
            raise RuntimeError(f"{doc.doc_id}: full prompt has {n} tokens, above {MAX_CONTEXT_TOKENS}")
        print(f"  preflight {doc.doc_id}: {n} tokens")


def check_model_versions(records: list[dict]) -> None:
    """Stop if records were served by different model versions, which would mix models in one comparison."""
    versions = {r["model_version"] for r in records}
    if len(versions) > 1:
        raise RuntimeError(f"records were served by several model versions: {sorted(versions)}")


def run_llm_stages(stage: str, run_dir: Path, no_cache: bool = False, data_dir: Path = DATA_DIR) -> None:
    """Run the paid stages: ``1`` (smoke test), ``2`` (baseline and sweep), ``3`` (ablations and budgets), or ``all``."""
    docs, gold = load_documents(data_dir), load_gold(data_dir)
    client = GeminiClient(seed=SEED)
    llm = CachedLLM(client, CACHE_DIR, enabled=not no_cache)
    stages = ["1", "2", "3"] if stage == "all" else [stage]
    configs: list[RunConfig] = []
    if "1" in stages:
        smoke_doc = replace(docs[0], questions=docs[0].questions[:2])
        run_configs([BASELINE, CORE], [smoke_doc], llm, run_dir / "smoke" / "records.jsonl")
        print("  smoke test passed: baseline and core answered 2 questions")
    if "2" in stages:
        preflight(docs, client)
        configs += [BASELINE, *SWEEP, EXACT]
    if "3" in stages:
        configs += [*ABLATIONS, *BUDGETS]
    if not configs:
        return
    if "2" not in stages:
        configs.insert(0, BASELINE)  # every summary needs the baseline beside it
    write_run_config(run_dir, configs, {"mode": "llm", "stages": stages, "no_cache": no_cache})
    records_path = run_dir / "records.jsonl"
    prepared = run_configs(configs, docs, llm, records_path)
    records = read_jsonl(records_path)
    check_model_versions(records)
    graded = grade_records(records, prepared, gold)
    write_jsonl(run_dir / "graded.jsonl", graded)
    summary = summarise(graded, ["correct_strict", "correct_lenient", "prompt_tokens", "gold_span_recall",
                                 "retrieval_s", "llm_s", "total_s"], gold)
    write_csv(run_dir / "summary.csv", summary)

    plot_dir = run_dir / "plots"
    plots.plot_tokens_vs_accuracy(summary, plot_dir / "c_tokens_vs_accuracy.png")
    sweep_names = {c.name for c in SWEEP}
    sweep_index = [{"index": "rowmajor", "k_label": "opt", **p.index_stats}
                   for (name, _), p in prepared.items() if name in sweep_names]
    sweep_graded = [r for r in graded if r["config"] in sweep_names]
    baseline_tokens = next(r["prompt_tokens"] for r in summary if r["config"] == "full" and r["group_by"] == "all")
    plots.plot_memory_vs_tokens(sweep_index, sweep_graded, plot_dir / "b_memory_vs_tokens.png",
                                baseline_tokens, estimated=False)

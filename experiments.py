"""Run the experiments, grade the answers and summarize the results (see specs.md §8–§9).

This is the only module that reads the gold answer files (the ``--answers`` paths).

Usage (files pair by position; README.md has the full command for the four v1 documents)::

    python experiments.py --mode all|stats_only|llm_only \
        --contexts C1.txt C2.txt --questions Q1.json Q2.json --answers A1.json A2.json

* ``stats_only``: the tier x R x k grid and its plots; no answers are generated.
* ``llm_only``: the graded LLM runs (baseline, tiers, FPR sweep).
* ``all``: both.

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import ask_llm

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

# --------------------------------------------------------------------------- #
# Data files
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Dataset:
    """Paired file paths. Entry ``i`` of each list belongs to the same document."""

    contexts: list[str]
    questions: list[str]
    answers: list[str]


def load_dataset(contexts: list[str], questions: list[str], answers: list[str]) -> Dataset:
    """Check the paired files given on the command line and return them as a ``Dataset``.

    Entry ``i`` of each list must belong to the same document. Context files carry no
    ``doc_id``, so they are paired purely by position; questions and answers files are
    cross-checked by ``doc_id`` and ``qid``, so files passed in a different order fail here,
    before any API call.

    Raises:
        ValueError: The lists differ in length, or a questions/answers pair does not match.
        FileNotFoundError: A path does not exist.
    """
    if not len(contexts) == len(questions) == len(answers):
        raise ValueError(f"Got {len(contexts)} contexts, {len(questions)} questions and {len(answers)} "
                         "answers files; they pair by position, so the counts must be equal")
    for path in (*contexts, *questions, *answers):
        if not Path(path).is_file():
            raise FileNotFoundError(f"No such file: {path}")
    for questions_path, answers_path in zip(questions, answers):
        doc_id, doc_questions = ask_llm.load_questions(questions_path)
        raw = json.loads(Path(answers_path).read_text(encoding="utf-8"))
        answers_doc_id = raw.get("doc_id") or Path(answers_path).stem.removesuffix("_answers")
        if answers_doc_id != doc_id:
            raise ValueError(f"{questions_path} (doc_id {doc_id!r}) is paired with {answers_path} "
                             f"(doc_id {answers_doc_id!r}); pass the files in the same order")
        missing = {q["qid"] for q in doc_questions} - {a["qid"] for a in raw["answers"]}
        if missing:
            raise ValueError(f"{answers_path} has no answers for {sorted(missing)}")
    return Dataset(list(contexts), list(questions), list(answers))


# --------------------------------------------------------------------------- #
# Gold answers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Gold:
    """The parts of an answer-file entry that grading and reporting use."""

    qid: str
    doc_id: str
    expected: str
    aliases: tuple[str, ...]
    normaliser: str  # "numeric" or "exact_ci"
    tolerance: float  # relative tolerance for "numeric"
    tier: str  # "T1_lookup", "T2_multihop" or "T3_aggregate"


def load_gold(answer_paths: list[str]) -> dict[str, Gold]:
    """Load answer files into a ``qid -> Gold`` map."""
    gold: dict[str, Gold] = {}
    for path in answer_paths:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        doc_id = raw.get("doc_id") or Path(path).stem.removesuffix("_answers")
        for entry in raw["answers"]:
            grading = entry["grading"]
            gold[entry["qid"]] = Gold(
                qid=entry["qid"],
                doc_id=doc_id,
                expected=entry["expected"],
                aliases=tuple(entry.get("aliases", ())),
                normaliser=grading["normaliser"],
                tolerance=grading.get("numeric_tolerance_rel") or 0.0,
                tier=entry.get("tier", ""),
            )
    return gold


# --------------------------------------------------------------------------- #
# Number parsing
# --------------------------------------------------------------------------- #

_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate(
    "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_SCALES = {"hundred": 100, "thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
# "m" and "bn" mean million and billion only straight after a currency amount ("$250m").
# Otherwise "20 m" is metres.
_CURRENCY_SCALES = {"k": 1e3, "m": 1e6, "mn": 1e6, "bn": 1e9}

# A sign counts only at the start of a token, so "1,400-psi" is 1400, not -psi.
_DIGITS = re.compile(
    r"(?<![\w.])(?P<currency>[$€£])?(?P<sign>-)?(?P<num>\d[\d,]*(?:\.\d+)?(?:e[+-]?\d+)?)"
    r"(?:\s*(?P<scale>thousand|million|billion|trillion)\b|(?P<cscale>k|mn|m|bn)\b)?"
)
_WORD_TOKEN = re.compile(r"[a-z]+")


def parse_number(text: str) -> float | None:
    """Return the first number in ``text``, written in digits or in words, or ``None``.

    Handles thousands commas, decimals, signs, ``%``, scientific notation, scale words
    ("13.3 million"), currency scales ("$250m") and number words ("sixty-three",
    "one hundred and twenty").
    """
    lowered = unicodedata.normalize("NFKC", text).casefold().replace("−", "-")
    found: list[tuple[int, float]] = []
    for m in _DIGITS.finditer(lowered):
        value = float(m.group("num").replace(",", ""))
        if m.group("sign"):
            value = -value
        if m.group("scale"):
            value *= _SCALES[m.group("scale")]
        elif m.group("cscale") and m.group("currency"):
            value *= _CURRENCY_SCALES[m.group("cscale")]
        found.append((m.start(), value))
    found.extend(_number_words(lowered))
    return min(found)[1] if found else None


def _number_words(text: str) -> list[tuple[int, float]]:
    """Return ``(position, value)`` for each run of number words, e.g. "one hundred and twenty"."""
    results: list[tuple[int, float]] = []
    words = [(m.start(), m.group(0)) for m in _WORD_TOKEN.finditer(text)]
    i = 0
    while i < len(words):
        if words[i][1] not in _UNITS and words[i][1] not in _TENS:
            i += 1
            continue
        start, total, current = words[i][0], 0.0, 0.0
        while i < len(words):
            word = words[i][1]
            if word in _UNITS:
                current += _UNITS[word]
            elif word in _TENS:
                current += _TENS[word]
            elif word == "hundred":
                current *= 100
            elif word in _SCALES:
                total += current * _SCALES[word]
                current = 0.0
            elif word == "and" and words[i - 1][1] in _SCALES and i + 1 < len(words) \
                    and _is_number_word(words[i + 1][1]):
                pass  # "one hundred and twenty"; "five and six" stays two numbers
            else:
                break
            i += 1
        results.append((start, total + current))
    return results


def _is_number_word(word: str) -> bool:
    return word in _UNITS or word in _TENS or word in _SCALES


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #

_QUOTE_DASH = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-", "−": "-",
})
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


def normalize_answer(text: str) -> str:
    """Normalize text for exact matching.

    Applies NFKC and casefolding, unifies quotes and dashes, removes thousands commas,
    collapses whitespace, strips edge punctuation and drops one leading article.
    Inner symbols such as ``±`` or ``-`` are kept.
    """
    text = unicodedata.normalize("NFKC", text).casefold().translate(_QUOTE_DASH)
    text = _strip_punctuation(" ".join(_THOUSANDS.sub("", text).split()))
    for article in ("the ", "an ", "a "):
        if text.startswith(article):
            text = text[len(article):]
            break
    return _strip_punctuation(text)


def _strip_punctuation(text: str) -> str:
    """Strip punctuation and whitespace from both ends of ``text``."""
    def is_edge(ch: str) -> bool:
        return unicodedata.category(ch).startswith("P") or ch.isspace()

    start, end = 0, len(text)
    while start < end and is_edge(text[start]):
        start += 1
    while end > start and is_edge(text[end - 1]):
        end -= 1
    return text[start:end]


def grade(answer: str | None, gold: Gold) -> bool:
    """Return whether ``answer`` is correct under the entry's ``grading.normaliser``.

    * ``numeric``: the first number in the answer is within the relative tolerance of the
      number in the expected answer or in any alias. Units are not enforced, because the
      gold aliases themselves include bare numbers.
    * ``exact_ci``: the normalized answer equals the normalized expected answer or an alias.
    """
    if not answer:
        return False
    targets = (gold.expected, *gold.aliases)
    if gold.normaliser == "numeric":
        value = parse_number(answer)
        if value is None:
            return False
        gold_values = [v for v in map(parse_number, targets) if v is not None]
        return any(abs(value - g) <= gold.tolerance * abs(g) for g in gold_values)
    return normalize_answer(answer) in {normalize_answer(t) for t in targets}


def grade_run(result: dict, gold: dict[str, Gold]) -> dict:
    """Add a ``correct`` field to every question record of an ``ask_llm.run`` result, in place,
    and return the run's summary (see :func:`summarize`)."""
    for doc in result["docs"]:
        for record in doc["questions"]:
            record["correct"] = grade(record["answer"], gold[record["qid"]])
    return summarize(result, gold)


def summarize(result: dict, gold: dict[str, Gold]) -> dict:
    """Summarize one graded run: accuracy (overall, per document and per question tier), filter
    memory and FPR, context retained, input tokens and latency. Filter fields are ``None`` for the
    baseline, and FPR fields are ``None`` unless the run used ``calc_fpr``."""
    records = [r for doc in result["docs"] for r in doc["questions"]]
    groups: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        g = gold[record["qid"]]
        groups[f"accuracy_{g.doc_id}"].append(record["correct"])
        groups[f"accuracy_{g.tier}"].append(record["correct"])
    latencies = [r["latency_s"] for r in records if r["latency_s"] is not None]
    docs = result["docs"]
    fpr_blocks = [doc["fpr"] for doc in docs if "fpr" in doc]
    nonmembers = sum(b["distinct_nonmembers"] for b in fpr_blocks)
    words_in = sum(doc["words_in"] for doc in docs)
    filter_stats = result.get("filter") or {}
    summary = {
        "n_questions": len(records),
        "accuracy": _mean([r["correct"] for r in records]),
        "n_words": filter_stats.get("n_words"),
        "memory_bits": filter_stats.get("memory_bits"),
        "bf_asizeof_bytes": filter_stats.get("bf_asizeof_bytes"),
        "set_asizeof_bytes": filter_stats.get("set_asizeof_bytes"),
        "theoretical_fpr": filter_stats.get("theoretical_fpr"),
        # Pooled over documents: false positives / distinct non-member keys queried.
        "empirical_fpr": sum(b["false_positives"] for b in fpr_blocks) / nonmembers if nonmembers else None,
        "fp_tokens_removed": sum(b["fp_tokens_removed"] for b in fpr_blocks) if fpr_blocks else None,
        "retained_frac": sum(doc["words_out"] for doc in docs) / words_in if words_in else None,
        "context_tokens": sum(doc["context_tokens"] for doc in docs),
        "mean_compress_s": _mean([doc["compress_s"] for doc in docs]),
        "total_prompt_tokens": sum(r["prompt_tokens"] for r in records),
        "mean_prompt_tokens": _mean([r["prompt_tokens"] for r in records]),
        "mean_latency_s": _mean(latencies),
        "cached_answers": sum(bool(r["cached"]) for r in records),
        "over_limit_answers": count_over_limit(records),
    }
    summary.update({name: _mean(values) for name, values in sorted(groups.items())})
    return summary


def count_over_limit(records: list[dict]) -> int:
    """How many answers used more output tokens than the question's ``max_answer_tokens``.

    A diagnostic only: grading does not penalize long answers. Records without an answer
    (stats-only runs) or without a limit are skipped.
    """
    return sum(
        1 for r in records
        if r.get("output_tokens") is not None and r.get("max_answer_tokens") is not None
        and r["output_tokens"] > r["max_answer_tokens"]
    )


def _mean(values: list) -> float | None:
    return sum(values) / len(values) if values else None


# --------------------------------------------------------------------------- #
# Stats-only grid (specs.md §9.1)
# --------------------------------------------------------------------------- #

GRID_SIZES = tuple(2 ** p for p in range(8, 18))  # R = 256 bits (32 B) ... 131072 bits (16 KB)
GRID_HASH_COUNTS = (1, 2, 3, 4, 6, 8)
MAX_HASH_COUNT = 16  # k_opt is capped here: past it the FPR is already ~0 and lookups only get slower
TOKEN_WORKERS = 4  # concurrent count_tokens calls (one per document)


def grid_hash_counts(n: int, size: int) -> list[int]:
    """The k values swept for a filter of ``size`` bits holding ``n`` words: the fixed list plus k_opt."""
    return sorted({*GRID_HASH_COUNTS, min(ask_llm.BloomFilter.optimal_k(n, size), MAX_HASH_COUNT)})


def run_stats_grid(data: Dataset, client: ask_llm.LLMClient, out_dir: Path = RESULTS) -> dict:
    """Sweep tier x R x k without generating answers, and save ``stats_grid.json`` / ``.csv``.

    For each configuration the filter is built once and every document is compressed once.
    Input tokens are estimated as ``context_tokens(doc) + overhead(question)`` summed over all
    questions, where ``overhead`` is the question's prompt with an empty context. Measured on the
    80 baseline prompts, this is exactly 1 token above the true ``prompt_token_count``, and it
    needs one ``count_tokens`` call per compressed context instead of one per prompt.

    The empirical FPR is measured on the union of distinct lookup keys of all documents that are
    not in the tier's vocabulary, so each non-member is queried exactly once, as in the theory.
    """
    from concurrent.futures import ThreadPoolExecutor

    from pympler.asizeof import asizeof

    texts, overhead = [], {}
    for context_path, questions_path in zip(data.contexts, data.questions):
        texts.append(Path(context_path).read_text(encoding="utf-8"))
        for question in ask_llm.load_questions(questions_path)[1]:
            overhead[question["qid"]] = client.count_tokens(ask_llm.build_prompt("", question))
    n_questions = [len(ask_llm.load_questions(q)[1]) for q in data.questions]
    question_overhead = sum(overhead.values())
    all_keys = {ask_llm.lookup_key(token) for text in texts for token in text.split()}
    words_in = sum(len(text.split()) for text in texts)

    with ThreadPoolExecutor(TOKEN_WORKERS) as pool:
        baseline_tokens = list(pool.map(client.count_tokens, texts))
        baseline = {
            "context_tokens": sum(baseline_tokens),
            "total_prompt_tokens": _prompt_tokens(baseline_tokens, n_questions, question_overhead),
            "n_questions": sum(n_questions),
        }
        rows = []
        for tier in ask_llm.TIERS:
            vocabulary = ask_llm.build_vocabulary(tier)
            exact = frozenset(vocabulary)
            nonmembers = sorted(all_keys - exact)
            set_bytes = asizeof(exact)
            for size in GRID_SIZES:
                k_opt = min(ask_llm.BloomFilter.optimal_k(len(vocabulary), size), MAX_HASH_COUNT)
                for k in grid_hash_counts(len(vocabulary), size):
                    bf = ask_llm.BloomFilter.from_words(vocabulary, size, k)
                    compressed, fp_tokens, compress_s = [], 0, 0.0
                    for text in texts:
                        start = time.perf_counter()
                        out, stats = ask_llm.compress(text, bf, exact)
                        compress_s += time.perf_counter() - start
                        compressed.append(out)
                        fp_tokens += stats["fp_tokens_removed"]
                    context_tokens = list(pool.map(client.count_tokens, compressed))
                    false_positives = sum(key in bf for key in nonmembers)
                    rows.append({
                        "tier": tier,
                        "n_words": len(vocabulary),
                        "size": size,
                        "hash_count": k,
                        "is_k_opt": k == k_opt,
                        "memory_bits": bf.memory_bits,
                        "bf_asizeof_bytes": asizeof(bf),
                        "set_asizeof_bytes": set_bytes,
                        "theoretical_fpr": ask_llm.BloomFilter.theoretical_fpr(len(vocabulary), size, k),
                        "empirical_fpr": false_positives / len(nonmembers),
                        "false_positives": false_positives,
                        "distinct_nonmembers": len(nonmembers),
                        "fp_tokens_removed": fp_tokens,
                        "retained_frac": sum(len(c.split()) for c in compressed) / words_in,
                        "context_tokens": sum(context_tokens),
                        "token_retained_frac": sum(context_tokens) / baseline["context_tokens"],
                        "total_prompt_tokens": _prompt_tokens(context_tokens, n_questions, question_overhead),
                        "mean_compress_s": compress_s / len(texts),
                    })
            print(f"[grid] {tier}: done", file=sys.stderr)

    grid = {"token_estimate": "context_tokens + empty-context prompt; 1 token above exact on the baseline",
            "baseline": baseline, "rows": rows}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stats_grid.json").write_text(json.dumps(grid, indent=2), encoding="utf-8")
    _write_csv(out_dir / "stats_grid.csv", rows)
    return grid


def _prompt_tokens(context_tokens: list[int], n_questions: list[int], question_overhead: int) -> int:
    """Estimated input tokens of all questions: each document's context once per question, plus overheads."""
    return sum(t * n for t, n in zip(context_tokens, n_questions)) + question_overhead


def _write_csv(path: Path, rows: list[dict]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# Plots (specs.md §9.3)
# --------------------------------------------------------------------------- #

# Categorical slots in fixed order (one per tier, never cycled); text stays in neutral ink.
TIER_COLORS = dict(zip(ask_llm.TIERS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def plot_grid(grid: dict, out_dir: Path = RESULTS) -> list[Path]:
    """Draw the stats-only plots into ``out_dir/plots`` and return their paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_style(plt)
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    rows = grid["rows"]
    at_k_opt = defaultdict(list)
    for row in rows:
        if row["is_k_opt"]:
            at_k_opt[row["tier"]].append(row)
    paths = []

    # 1. Memory vs FPR (required)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for tier, tier_rows in at_k_opt.items():
        color = TIER_COLORS[tier]
        ax.plot([r["memory_bits"] for r in tier_rows], [r["theoretical_fpr"] for r in tier_rows],
                color=color, label=f"{tier} (N={tier_rows[0]['n_words']})")
        measured = [r for r in tier_rows if r["empirical_fpr"] > 0]
        ax.scatter([r["memory_bits"] for r in measured], [r["empirical_fpr"] for r in measured],
                   color=color, s=36, edgecolor="white", linewidth=1.5, zorder=3)
    ax.set(xscale="log", yscale="log", ylim=(1e-6, 2), xlabel="Bloom filter memory R (bits)",
           ylabel="False-positive rate", title="Bloom filter memory vs. false-positive rate (k = k_opt)")
    ax.set_xscale("log", base=2)
    ax.text(0.01, 0.01, "lines: theoretical (1 - e^(-kN/R))^k, clipped at 1e-6   dots: measured (zero FPR not drawn)",
            transform=ax.transAxes, color=MUTED, fontsize=8)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    paths.append(_save(fig, plots / "memory_vs_fpr.png"))

    # 2. Memory vs LLM input tokens (required)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tier, tier_rows in at_k_opt.items():
        ax.plot([r["memory_bits"] for r in tier_rows], [r["total_prompt_tokens"] / 1e6 for r in tier_rows],
                color=TIER_COLORS[tier], marker="o", markersize=5, label=tier)
    ax.axhline(grid["baseline"]["total_prompt_tokens"] / 1e6, color=MUTED, linestyle="--", linewidth=1.2)
    ax.text(GRID_SIZES[0], grid["baseline"]["total_prompt_tokens"] / 1e6, " full-context baseline",
            color=MUTED, va="bottom", fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set(ylim=(0, None), xlabel="Bloom filter memory R (bits)",
           ylabel=f"LLM input tokens, all {grid['baseline']['n_questions']} questions (millions)",
           title="Bloom filter memory vs. LLM input tokens (k = k_opt, estimated)")
    ax.legend(fontsize=8)
    paths.append(_save(fig, plots / "memory_vs_tokens.png"))

    # 3. Measured vs theoretical FPR, every configuration
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    floor = 1e-5
    for tier in ask_llm.TIERS:
        tier_rows = [r for r in rows if r["tier"] == tier]
        ax.scatter([max(r["theoretical_fpr"], floor) for r in tier_rows],
                   [max(r["empirical_fpr"], floor) for r in tier_rows],
                   color=TIER_COLORS[tier], s=20, alpha=0.8, label=tier)
    ax.plot([floor, 1], [floor, 1], color=MUTED, linewidth=1, linestyle="--")
    ax.set(xscale="log", yscale="log", xlabel="Theoretical FPR", ylabel="Measured FPR",
           title="Measured vs. theoretical FPR (all R, k, tiers)")
    ax.text(0.02, 0.97, f"values below {floor:g} drawn at {floor:g}", transform=ax.transAxes,
            color=MUTED, fontsize=8, va="top")
    ax.legend(fontsize=8, loc="lower right")
    paths.append(_save(fig, plots / "fpr_measured_vs_theory.png"))

    # 4. FPR vs k at fixed R (curated tier)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    curated = [r for r in rows if r["tier"] == "curated"]
    for color, size in zip(["#2a78d6", "#eb6834", "#1baf7a"], (256, 512, 1024)):
        sized = sorted((r for r in curated if r["size"] == size), key=lambda r: r["hash_count"])
        k_opt = next(r["hash_count"] for r in sized if r["is_k_opt"])
        ax.plot([r["hash_count"] for r in sized], [r["theoretical_fpr"] for r in sized], color=color,
                label=f"R = {size} bits (k_opt = {k_opt})")
        ax.scatter([r["hash_count"] for r in sized], [r["empirical_fpr"] for r in sized],
                   color=color, s=36, edgecolor="white", linewidth=1.5, zorder=3)
    ax.set(yscale="log", xlabel="Hash functions k", ylabel="False-positive rate",
           title=f"FPR vs. number of hash functions (curated, N = {curated[0]['n_words']})")
    ax.text(0.01, 0.01, "lines: theoretical   dots: measured", transform=ax.transAxes, color=MUTED, fontsize=8)
    ax.legend(fontsize=8)
    paths.append(_save(fig, plots / "fpr_vs_k.png"))

    # 5. Context retained per tier, at the LLM-run configuration of each tier
    fig, ax = plt.subplots(figsize=(7, 3.8))
    picked = [_row_for(rows, c["tier"], c["size"], c["hash_count"])
              for name, c in llm_configs().items() if name.startswith("tier_")]
    labels = [f"{r['tier']}\nR={r['size']}, k={r['hash_count']}" for r in picked]
    for i, (row, label) in enumerate(zip(picked, labels)):
        ax.barh(i, row["token_retained_frac"] * 100, color=TIER_COLORS[row["tier"]], height=0.6)
        ax.text(row["token_retained_frac"] * 100 + 1, i, f"{row['token_retained_frac']:.0%} tokens, "
                f"{row['retained_frac']:.0%} words", va="center", color=INK, fontsize=8)
    ax.set_yticks(range(len(picked)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set(xlim=(0, 115), xlabel="Context retained (% of full-context tokens)",
           title="Context retained per vocabulary tier (theoretical FPR ≤ 1%)")
    ax.grid(axis="y", visible=False)
    paths.append(_save(fig, plots / "retained_by_tier.png"))
    return paths


def _apply_style(plt) -> None:
    """Shared look for every plot: recessive axes and grid, neutral ink, no frames."""
    plt.rcParams.update({
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 10, "legend.frameon": False, "lines.linewidth": 2,
    })


def _row_for(rows: list[dict], tier: str, size: int, hash_count: int) -> dict:
    """The grid row for one configuration, or a clear error if the grid does not contain it."""
    for row in rows:
        if (row["tier"], row["size"], row["hash_count"]) == (tier, size, hash_count):
            return row
    raise KeyError(f"No grid row for tier={tier}, R={size}, k={hash_count}")


def _save(fig, path: Path) -> Path:
    import matplotlib.pyplot as plt

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


QUESTION_TIERS = ("T1_lookup", "T2_multihop", "T3_aggregate")
SWEEP_COLORS = {"curated": "#eb6834", "common1000": "#1baf7a"}  # categorical slots 2 and 3


def plot_llm(runs: dict[str, dict], out_dir: Path = RESULTS) -> list[Path]:
    """Draw the LLM-run plots into ``out_dir/plots`` and return their paths.

    Expects the run names from :func:`llm_configs` (``baseline``, ``tier_*``, ``fpr<pct>_*``).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_style(plt)
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    summary = {name: run["summary"] for name, run in runs.items()}
    tokens = {name: s["total_prompt_tokens"] / 1e6 for name, s in summary.items()}
    accuracy = {name: s["accuracy"] * 100 for name, s in summary.items()}
    ladder = [f"tier_{t}" for t in ask_llm.TIERS if t != "nltk_full" and f"tier_{t}" in runs]
    paths = []

    # 1. LLM input tokens vs accuracy (required)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot([tokens[n] for n in ladder], [accuracy[n] for n in ladder], color="#2a78d6", marker="o",
            markersize=7, label="vocabulary tiers (FPR ≤ 1%)", zorder=3)
    for name in ladder:  # short labels: "curated", "500", "1000", ... keep the cluster readable
        label = name.removeprefix("tier_").removeprefix("common")
        ax.annotate(label, (tokens[name], accuracy[name]), xytext=(0, 9), textcoords="offset points",
                    fontsize=8, color="#2a78d6", ha="center", fontweight="bold")
    for tier, color in SWEEP_COLORS.items():
        sweep = sorted((n for n in runs if n.startswith("fpr") and n.endswith(f"_{tier}")),
                       key=lambda n: summary[n]["theoretical_fpr"])
        if not sweep:
            continue
        line = [f"tier_{tier}", *sweep] if f"tier_{tier}" in runs else sweep
        ax.plot([tokens[n] for n in line], [accuracy[n] for n in line], color=color, marker="s",
                markersize=6, linestyle="--", label=f"FPR sweep on {tier}", zorder=2)
        for name in sweep:
            ax.annotate(f"{summary[name]['theoretical_fpr']:.0%}", (tokens[name], accuracy[name]),
                        xytext=(-7, -3), textcoords="offset points", fontsize=8, color=color, ha="right")
    if "tier_nltk_full" in runs:
        ax.scatter(tokens["tier_nltk_full"], accuracy["tier_nltk_full"], color="#e87ba4", marker="D", s=50,
                   label="nltk_full (ablation, FPR ≤ 1%)", zorder=4, edgecolor="white", linewidth=1.5)
    if "baseline" in runs:
        ax.scatter(tokens["baseline"], accuracy["baseline"], color=INK, marker="*", s=180,
                   label="full-context baseline", zorder=4)
    ax.text(0.01, 0.01, "blue labels: vocabulary tier (500 = common500, ...)   "
            "colored %: theoretical FPR of a sweep run", transform=ax.transAxes, color=MUTED, fontsize=8)
    ax.set(xlim=(0.9 * min(tokens.values()), 1.05 * max(tokens.values())), ylim=(0, 100), xlabel=f"LLM input tokens, all {summary['baseline']['n_questions']} "
           "questions (millions)" if "baseline" in summary else "LLM input tokens (millions)",
           ylabel="Accuracy (%)", title="LLM input tokens vs. question-answering accuracy")
    ax.legend(fontsize=8, loc="lower right", bbox_to_anchor=(1, 0.05))
    paths.append(_save(fig, plots / "tokens_vs_accuracy.png"))

    # 2. FPR vs accuracy, per swept tier (the ~0% point is the tier's own run)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tier, color in SWEEP_COLORS.items():
        line = [n for n in (f"tier_{tier}", *(n for n in runs if n.startswith("fpr") and n.endswith(f"_{tier}")))
                if n in runs and summary[n]["empirical_fpr"] is not None]
        line.sort(key=lambda n: summary[n]["empirical_fpr"])
        if not line:
            continue
        ax.plot([summary[n]["empirical_fpr"] * 100 for n in line], [accuracy[n] for n in line], color=color,
                marker="o", markersize=7, label=tier)
    if "baseline" in runs:
        ax.axhline(accuracy["baseline"], color=MUTED, linestyle="--", linewidth=1.2)
        ax.text(ax.get_xlim()[1], accuracy["baseline"], "full-context baseline ", color=MUTED, va="bottom",
                ha="right", fontsize=8)
    ax.set(ylim=(0, 100), xlabel="Measured false-positive rate (%)", ylabel="Accuracy (%)",
           title="False positives vs. accuracy (vocabulary held fixed)")
    ax.legend(fontsize=8, title="FPR sweep on", title_fontsize=8, loc="lower left")
    paths.append(_save(fig, plots / "fpr_vs_accuracy.png"))

    # 3. Accuracy by question tier, every run (a table-like heatmap with the values written in)
    names = list(runs)
    values = [[(summary[n].get(f"accuracy_{t}") or 0) * 100 for t in QUESTION_TIERS] for n in names]
    fig, ax = plt.subplots(figsize=(6.5, 0.38 * len(names) + 1.4))
    ax.imshow(values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:.0f}%", ha="center", va="center", fontsize=8,
                    color="white" if value > 60 else INK)
    ax.set_xticks(range(len(QUESTION_TIERS)), ["T1 lookup", "T2 multi-hop", "T3 aggregate"])
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Accuracy by question tier")
    paths.append(_save(fig, plots / "accuracy_by_question_tier.png"))
    return paths


# --------------------------------------------------------------------------- #
# LLM experiments (specs.md §9.2)
# --------------------------------------------------------------------------- #

TARGET_FPR = 0.01  # each tier's LLM run uses the smallest power-of-two R reaching this at k_opt
FPR_SWEEP_TIERS = ("curated", "common1000")  # the core method, and a mid-ladder tier
FPR_SWEEP_TARGETS = (0.5, 0.2, 0.05)


def smallest_size_for_fpr(n: int, target: float) -> int:
    """The smallest power-of-two R whose theoretical FPR at k_opt is at most ``target``."""
    size = 1
    while ask_llm.BloomFilter.theoretical_fpr(n, size, ask_llm.BloomFilter.optimal_k(n, size)) > target:
        size *= 2
    return size


def size_for_fpr(n: int, target: float) -> int:
    """R giving about ``target`` FPR at the optimal k: ``R = -N ln p / (ln 2)^2``."""
    return math.ceil(-n * math.log(target) / math.log(2) ** 2)


def llm_configs() -> dict[str, dict]:
    """Named ``ask_llm.run`` keyword arguments for each LLM run (specs.md §9.2).

    * ``baseline``: the full context.
    * ``tier_<tier>``: each tier at the smallest power-of-two R with theoretical FPR <= 1%, k_opt.
    * ``fpr<pct>_<tier>``: each tier in ``FPR_SWEEP_TIERS`` at R chosen for ~50%, 20% and 5% FPR, k_opt.

    The tier runs vary what is removed on purpose (the vocabulary) with false positives held
    near zero; the FPR sweep varies what is removed by accident with the vocabulary held fixed.
    """
    configs: dict[str, dict] = {"baseline": {"baseline": True}}
    for tier in ask_llm.TIERS:
        n = len(ask_llm.build_vocabulary(tier))
        size = smallest_size_for_fpr(n, TARGET_FPR)
        configs[f"tier_{tier}"] = {"tier": tier, "size": size,
                                   "hash_count": ask_llm.BloomFilter.optimal_k(n, size)}
    for tier in FPR_SWEEP_TIERS:
        n = len(ask_llm.build_vocabulary(tier))
        for target in FPR_SWEEP_TARGETS:
            size = size_for_fpr(n, target)
            configs[f"fpr{round(target * 100)}_{tier}"] = {
                "tier": tier, "size": size, "hash_count": ask_llm.BloomFilter.optimal_k(n, size)}
    return configs


def run_llm_experiments(data: Dataset, out_dir: Path = RESULTS) -> dict[str, dict]:
    """Run, grade and save every LLM configuration in :func:`llm_configs`.

    Answers are cached on disk, so rerunning this re-grades without new paid calls."""
    gold = load_gold(data.answers)
    runs: dict[str, dict] = {}
    for name, params in llm_configs().items():
        print(f"== {name}", file=sys.stderr)
        calc_fpr = not params.get("baseline", False)  # FPR stats need a filter
        result = ask_llm.run(data.contexts, data.questions, out=None, calc_fpr=calc_fpr, **params)
        result["summary"] = grade_run(result, gold)
        runs[name] = result
    add_token_retention(runs)
    save_runs(runs, out_dir)
    return runs


def add_token_retention(runs: dict[str, dict]) -> None:
    """Set each summary's ``token_retained_frac``: its context tokens over the baseline's, in place."""
    if "baseline" not in runs:
        return
    full = runs["baseline"]["summary"]["context_tokens"]
    for run in runs.values():
        run["summary"]["token_retained_frac"] = run["summary"]["context_tokens"] / full


def save_runs(runs: dict[str, dict], out_dir: Path) -> None:
    """Write every graded run to ``llm_runs.json`` and one summary row per run to ``summary.csv``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "llm_runs.json").write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(out_dir / "summary.csv",
               [{"run": name, **_config_columns(run["config"]), **run["summary"]} for name, run in runs.items()])


def _config_columns(config: dict) -> dict:
    return {key: config[key] for key in ("baseline", "tier", "size", "hash_count")}


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["stats_only", "llm_only", "all"], default="all")
    parser.add_argument("--contexts", nargs="+", required=True, help="context .txt files")
    parser.add_argument("--questions", nargs="+", required=True,
                        help="questions .json files, paired with --contexts by position")
    parser.add_argument("--answers", nargs="+", required=True,
                        help="answers .json files, paired with --contexts by position")
    parser.add_argument("--out", type=Path, default=RESULTS, help="output directory")
    args = parser.parse_args(argv)
    if not len(args.contexts) == len(args.questions) == len(args.answers):
        parser.error("--contexts, --questions and --answers must list the same number of files")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data = load_dataset(args.contexts, args.questions, args.answers)
    if args.mode in ("stats_only", "all"):
        grid = run_stats_grid(data, ask_llm.CachedClient(ask_llm.GeminiClient()), args.out)
        for path in plot_grid(grid, args.out):
            print(f"wrote {path}", file=sys.stderr)
        print("LLM configurations:", file=sys.stderr)
        for name, config in llm_configs().items():
            print(f"  {name}: {config}", file=sys.stderr)
    if args.mode in ("llm_only", "all"):
        runs = run_llm_experiments(data, args.out)
        for path in plot_llm(runs, args.out):
            print(f"wrote {path}", file=sys.stderr)
        for name, run in runs.items():
            s = run["summary"]
            print(f"{name}: accuracy {s['accuracy']:.3f} over {s['n_questions']} questions, "
                  f"mean prompt tokens {s['mean_prompt_tokens']:.0f}", file=sys.stderr)


if __name__ == "__main__":
    main()

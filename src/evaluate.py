"""Grade answers against the gold files and summarise results (specs.md §9).

This is the only module that reads ``assignment/answers/``. Its main parts:

* :func:`grade`: strict grading (the headline number) plus a lenient diagnostic,
  following each entry's ``grading.normaliser``:

  - ``numeric``: the first number in the answer must be within the relative
    tolerance of the expected value or of any alias. Units are not enforced,
    because the gold aliases themselves include bare numbers.
  - ``exact_ci``: equality with the expected value or an alias, after normalisation.

* :func:`gold_span_recall`: does a compressed context still contain the
  annotated answer passages? This is a retrieval diagnostic that needs no LLM.
* :func:`bootstrap_ci` and :func:`summarise`: accuracy with uncertainty, broken
  down by config, document, tier and lexical overlap.

Depends on: :mod:`text` (shared normalisation), numpy and the standard library.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from text import normalize, tokens

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
    normaliser: str
    tolerance: float
    tier: str
    lexical_overlap: str
    gold_quotes: tuple[str, ...]


def load_gold(data_dir: Path) -> dict[str, Gold]:
    """Load every answer file in ``data_dir / "answers"``, keyed by qid."""
    gold: dict[str, Gold] = {}
    for path in sorted((data_dir / "answers").glob("*_answers.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        for entry in raw["answers"]:
            grading = entry["grading"]
            gold[entry["qid"]] = Gold(
                qid=entry["qid"],
                doc_id=raw["doc_id"],
                expected=entry["expected"],
                aliases=tuple(entry.get("aliases", ())),
                normaliser=grading["normaliser"],
                tolerance=grading.get("numeric_tolerance_rel") or 0.0,
                tier=entry.get("tier", ""),
                lexical_overlap=entry.get("lexical_overlap", ""),
                gold_quotes=tuple(s["quote"] for s in entry.get("gold_spans", ()) if s.get("quote")),
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


@dataclass(frozen=True)
class ParsedNumber:
    """The first number found in a text, and how many numbers the text contains."""

    value: float | None
    count: int


def parse_number(text: str) -> ParsedNumber:
    """Find the first number in ``text``, written in digits or in words.

    Handles thousands commas, decimals, signs, ``%``, scientific notation, scale
    words ("13.3 million"), currency scales ("$250m"), and number words
    ("sixty-three", "one hundred and twenty").
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
    if not found:
        return ParsedNumber(None, 0)
    found.sort()
    return ParsedNumber(found[0][1], len(found))


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


@dataclass(frozen=True)
class Grade:
    correct_strict: bool
    correct_lenient: bool
    answer_norm: str
    parse_status: str  # "ok", "no_number" or "multiple_numbers" (numeric); "n/a" for exact_ci


def normalize_answer(text: str) -> str:
    """Normalisation for exact-match grading.

    Applies the shared normalisation, collapses whitespace, strips leading and
    trailing punctuation, and removes one leading article. Inner symbols such as
    ``±`` or ``-`` are kept.
    """
    text = " ".join(normalize(text).split())
    text = _strip_punctuation(text)
    for article in ("the ", "an ", "a "):
        if text.startswith(article):
            text = text[len(article):]
            break
    return _strip_punctuation(text)


def _strip_punctuation(text: str) -> str:
    def is_punct(ch: str) -> bool:
        return unicodedata.category(ch).startswith("P")

    start, end = 0, len(text)
    while start < end and (is_punct(text[start]) or text[start].isspace()):
        start += 1
    while end > start and (is_punct(text[end - 1]) or text[end - 1].isspace()):
        end -= 1
    return text[start:end]


def grade(answer: str, gold: Gold) -> Grade:
    """Grade one answer strictly, and leniently for diagnostics (§9.2)."""
    answer_norm = normalize_answer(answer)
    targets = [normalize_answer(t) for t in (gold.expected, *gold.aliases)]
    lenient = any(_contains_phrase(answer_norm, t) for t in targets if t)

    if gold.normaliser == "numeric":
        parsed = parse_number(answer)
        if parsed.value is None:
            return Grade(False, lenient, answer_norm, "no_number")
        status = "multiple_numbers" if parsed.count > 1 else "ok"
        gold_values = [v for v in (parse_number(t).value for t in (gold.expected, *gold.aliases)) if v is not None]
        correct = any(_within_tolerance(parsed.value, g, gold.tolerance) for g in gold_values)
        return Grade(correct, lenient or correct, answer_norm, status)

    correct = answer_norm in targets
    return Grade(correct, lenient or correct, answer_norm, "n/a")


def _within_tolerance(value: float, gold_value: float, tolerance: float) -> bool:
    """Relative tolerance. When the gold value is 0 this becomes exact equality (it never occurs in this data)."""
    return abs(value - gold_value) <= tolerance * abs(gold_value)


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None


# --------------------------------------------------------------------------- #
# Retrieval diagnostic
# --------------------------------------------------------------------------- #


def gold_span_recall(context: str, gold: Gold, min_overlap: float = 0.9) -> float:
    """Return the fraction of the gold quotes contained in ``context``.

    A quote counts as contained when at least ``min_overlap`` of its distinct
    tokens appear in the context. Token sets make this robust to the separators
    and reordering that assembly introduces.
    """
    if not gold.gold_quotes:
        return float("nan")
    context_tokens = set(tokens(context))
    contained = 0
    for quote in gold.gold_quotes:
        quote_tokens = set(tokens(quote))
        if quote_tokens and len(quote_tokens & context_tokens) / len(quote_tokens) >= min_overlap:
            contained += 1
    return contained / len(gold.gold_quotes)


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #


def bootstrap_ci(values: list[float], n_resamples: int = 10_000, seed: int = 580) -> tuple[float, float]:
    """Return a percentile bootstrap 95% CI for the mean of ``values``."""
    data = np.asarray(values, dtype=float)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(data, size=(n_resamples, len(data)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise(rows: list[dict], metrics: list[str], gold: dict[str, Gold]) -> list[dict]:
    """Average ``metrics`` per config, overall and per document, tier and lexical overlap.

    Each row needs ``config`` and ``qid``. Groupings come from the gold entries.
    Every group gets the mean of each metric, and a bootstrap CI for the first metric.
    """
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        g = gold[row["qid"]]
        for by, value in (("all", "all"), ("doc", g.doc_id), ("tier", g.tier), ("lexical", g.lexical_overlap)):
            groups[(row["config"], by, value)].append(row)
    summary = []
    for (config, by, value), members in sorted(groups.items()):
        entry: dict = {"config": config, "group_by": by, "group": value, "n": len(members)}
        for metric in metrics:
            entry[metric] = float(np.nanmean([float(m[metric]) for m in members]))
        low, high = bootstrap_ci([float(m[metrics[0]]) for m in members])
        entry[f"{metrics[0]}_ci_low"], entry[f"{metrics[0]}_ci_high"] = low, high
        summary.append(entry)
    return summary

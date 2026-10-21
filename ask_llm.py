"""Answer questions about long contexts with Gemini, optionally after Bloom-filter compression.

Pipeline (see specs.md):

    context file --compress (Bloom filter drop pass)--> compressed context
    compressed context + one question --build_prompt--> prompt --LLM--> answer + stats

One Bloom filter holds a fixed, context-independent vocabulary of low-meaning words; every
token it reports as present is dropped. ``baseline=True`` sends the full context instead.

Use from the command line::

    python ask_llm.py --contexts C.txt --questions Q.json --size 4096 --hash-count 6 \\
        --tier curated [--stats-only] [--calc-fpr] [--out response.json]
    python ask_llm.py --contexts C.txt --questions Q.json --baseline

or from Python::

    import ask_llm
    result = ask_llm.run([...contexts], [...questions], size=4096, hash_count=6, tier="curated")

This module never reads the gold answer files; only ``experiments.py`` does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import string
import sys
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

MODEL = "gemini-3.5-flash-lite"
SEED = 580
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 64  # fixed for all runs; deliberately not the per-question max_answer_tokens
RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}
MAX_RETRIES = 10
MAX_BACKOFF_S = 60.0

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache"

# --------------------------------------------------------------------------- #
# Vocabulary (specs.md §3)
# --------------------------------------------------------------------------- #

# NLTK's English stopword list (179 words), copied verbatim so no download is needed.
NLTK_STOPWORDS = frozenset("""
i me my myself we our ours ourselves you you're you've you'll you'd your yours yourself yourselves
he him his himself she she's her hers herself it it's its itself they them their theirs themselves
what which who whom this that that'll these those am is are was were be been being have has had
having do does did doing a an the and but if or because as until while of at by for with about
against between into through during before after above below to from up down in out on off over
under again further then once here there when where why how all any both each few more most other
some such no nor not only own same so than too very s t can will just don don't should should've
now d ll m o re ve y ain aren aren't couldn couldn't didn didn't doesn doesn't hadn hadn't hasn
hasn't haven haven't isn isn't ma mightn mightn't mustn mustn't needn needn't shan shan't shouldn
shouldn't wasn wasn't weren weren't won won't wouldn wouldn't
""".split())

# Words never inserted into the filter, because deleting them can change an answer.
# Generic English rules, not tuned to any document.
PROTECTED = frozenset("""
no nor not never without don don't ain aren aren't couldn couldn't didn didn't doesn doesn't hadn
hadn't hasn hasn't haven haven't isn isn't mightn mightn't mustn mustn't needn needn't shan shan't
shouldn shouldn't wasn wasn't weren weren't won won't wouldn wouldn't
more most less least few only same very than all any each both some
over under above below before after during until again
s t d m o y ll re ve ma
zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen
sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety
hundred thousand million billion trillion half dozen
first second third fourth fifth sixth seventh eighth ninth tenth
january february march april may june july august september october november december
monday tuesday wednesday thursday friday saturday sunday
""".split())

# Markdown decoration. The table separator "|" is deliberately absent, so rows keep their columns.
DECORATIVE = frozenset(["#", "##", "###", "####", "*", "**", "-", "--", "---", ">", "•", "_"])

CURATED = NLTK_STOPWORDS - PROTECTED

# Tier name -> how many of wordfreq's most frequent English words to add to CURATED.
COMMON_TIERS = {"common500": 500, "common1000": 1000, "common2000": 2000, "common5000": 5000}
TIERS = ("curated", "nltk_full", *COMMON_TIERS)


def build_vocabulary(tier: str) -> list[str]:
    """Return the sorted words inserted into the filter for ``tier`` (specs.md §3.2).

    * ``curated``: NLTK stopwords minus ``PROTECTED``, plus ``DECORATIVE``.
    * ``nltk_full``: all NLTK stopwords plus ``DECORATIVE`` (ablation).
    * ``commonN``: ``curated`` plus wordfreq's top-N English words that are ASCII letters
      only and not ``PROTECTED``.
    """
    if tier not in TIERS:
        raise ValueError(f"Unknown tier {tier!r}; choose from {', '.join(TIERS)}")
    if tier == "nltk_full":
        return sorted(NLTK_STOPWORDS | DECORATIVE)
    words = set(CURATED | DECORATIVE)
    if tier in COMMON_TIERS:
        from wordfreq import top_n_list  # loaded only when a common-word tier is used

        words.update(w for w in top_n_list("en", COMMON_TIERS[tier])
                     if w.isascii() and w.isalpha() and w not in PROTECTED)
    return sorted(words)


# --------------------------------------------------------------------------- #
# Bloom filter (specs.md §4)
# --------------------------------------------------------------------------- #

HASH_KEY = SEED.to_bytes(8, "little")


@lru_cache(maxsize=1 << 17)
def _hash_pair(word: str) -> tuple[int, int]:
    """Two 64-bit hashes of ``word`` from one keyed blake2b call; the second is forced odd.

    They do not depend on the filter's size or hash count, so they are cached and shared by
    every filter in a parameter sweep.
    """
    digest = hashlib.blake2b(word.encode("utf-8"), digest_size=16, key=HASH_KEY).digest()
    return int.from_bytes(digest[:8], "little"), int.from_bytes(digest[8:], "little") | 1


class BloomFilter:
    """A Bloom filter: ``size`` bits (R) in a ``bytearray``, probed at ``hash_count`` (k) positions.

    The positions of a word come from Kirsch–Mitzenmacher double hashing,
    ``(h1 + i * h2) mod R`` for ``i = 0 .. k-1``, where ``h1`` and ``h2`` come from a single
    keyed blake2b digest. That costs one hash call per word whatever k is, and the positions are
    identical across processes and machines (Python's salted ``hash()`` is never used).
    """

    def __init__(self, size: int, hash_count: int):
        if size < 1 or hash_count < 1:
            raise ValueError(f"size and hash_count must be positive, got {size} and {hash_count}")
        self.size = size
        self.hash_count = hash_count
        self.bits = bytearray((size + 7) // 8)

    @classmethod
    def from_words(cls, words, size: int, hash_count: int) -> BloomFilter:
        """Build a filter of ``size`` bits and ``hash_count`` hashes containing ``words``."""
        bf = cls(size, hash_count)
        for word in words:
            bf.add(word)
        return bf

    def positions(self, word: str) -> list[int]:
        """The ``hash_count`` bit positions of ``word``."""
        h1, h2 = _hash_pair(word)
        return [(h1 + i * h2) % self.size for i in range(self.hash_count)]

    def add(self, word: str) -> None:
        for p in self.positions(word):
            self.bits[p >> 3] |= 1 << (p & 7)

    def __contains__(self, word: str) -> bool:
        return all(self.bits[p >> 3] >> (p & 7) & 1 for p in self.positions(word))

    @property
    def memory_bits(self) -> int:
        """R, the number of bits in the filter."""
        return self.size

    @staticmethod
    def theoretical_fpr(n: int, size: int, hash_count: int) -> float:
        """The standard approximation ``(1 - e^{-kN/R})^k`` for N inserted words."""
        return (1.0 - math.exp(-hash_count * n / size)) ** hash_count

    @staticmethod
    def optimal_k(n: int, size: int) -> int:
        """The hash count minimizing the FPR, ``(R/N) ln 2``, rounded and at least 1."""
        return max(1, round(size / n * math.log(2)))


# --------------------------------------------------------------------------- #
# Compression (specs.md §5)
# --------------------------------------------------------------------------- #

_EDGE_PUNCTUATION = string.punctuation + "“”‘’—–…"


def lookup_key(token: str) -> str:
    """The form of a whitespace token that is looked up in the filter.

    Lowercased, curly apostrophes made straight, and punctuation stripped from both ends, so
    "The", "the," and "(the" share the key "the". A token that is only punctuation ("##", "---")
    keeps its raw form, so decorative markdown can still match.
    """
    key = token.lower().replace("’", "'").strip(_EDGE_PUNCTUATION)
    return key or token


def compress(text: str, bf: BloomFilter, exact: set[str] | frozenset[str] | None = None) -> tuple[str, dict]:
    """Drop every whitespace token whose lookup key the filter reports as present.

    Lines are kept; the remaining tokens of each line are joined with single spaces, and runs
    of blank lines collapse into one. Each distinct key is looked up once per call.

    Args:
        text: The context.
        bf: The filter of words to drop.
        exact: The exact set of inserted words. When given, false positives are counted.

    Returns:
        ``(compressed_text, stats)``. ``stats`` is empty without ``exact``; with it, it holds
        ``distinct_nonmembers``, ``false_positives``, ``empirical_fpr`` and ``fp_tokens_removed``
        (token occurrences deleted only because of a false positive).
    """
    drop: dict[str, bool] = {}  # memo: key -> filter answer
    fp_tokens_removed = 0
    lines: list[str] = []
    previous_blank = False
    for line in text.split("\n"):
        kept = []
        for token in line.split():
            key = lookup_key(token)
            if key not in drop:
                drop[key] = key in bf
            if not drop[key]:
                kept.append(token)
            elif exact is not None and key not in exact:
                fp_tokens_removed += 1
        new_line = " ".join(kept)
        if not new_line and previous_blank:
            continue
        previous_blank = not new_line
        lines.append(new_line)

    stats: dict = {}
    if exact is not None:
        nonmembers = [key for key in drop if key not in exact]
        false_positives = sum(drop[key] for key in nonmembers)
        stats = {
            "distinct_nonmembers": len(nonmembers),
            "false_positives": false_positives,
            "empirical_fpr": false_positives / len(nonmembers) if nonmembers else 0.0,
            "fp_tokens_removed": fp_tokens_removed,
        }
    return "\n".join(lines), stats


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

# One template for every run: the baseline and all compressed runs differ only in {context}.
# There is no system instruction, so count_tokens(prompt) equals the real prompt_token_count.
PROMPT_HEADER = (
    "You answer a question about a document using only the context below.\n"
    "Some common words may have been removed from the context to shorten it.\n"
    "Reply with only the answer: no explanation and no full sentence."
)

TYPE_HINTS = {
    "number": "Give a single number with its unit and any scale word (such as million), "
              "as written in the context.",
    "entity": "Give the name of the person, place, organisation or thing, as written in the context.",
    "phrase": "Give a short phrase, worded as in the context.",
    "date": "Give the date or year, as written in the context.",
}


def build_prompt(context: str, question: dict) -> str:
    """Build the prompt for one question.

    Uses the question's ``question``, ``answer_type``, ``unit`` (the line is left out when
    the question has none) and ``max_answer_tokens``. ``qid`` and ``number`` are never used:
    they carry no meaning, and ``qid`` reveals the question's difficulty tier.
    """
    answer_type = question["answer_type"]
    type_line = f"Answer type: {answer_type}."
    if answer_type in TYPE_HINTS:
        type_line += " " + TYPE_HINTS[answer_type]

    lines = [
        PROMPT_HEADER,
        "",
        "Context:",
        context,
        "End of context.",
        "",
        f"Question: {question['question']}",
        type_line,
    ]
    if question.get("unit"):
        lines.append(f"Unit: {question['unit']}")
    lines.append(f"Answer in at most {question['max_answer_tokens']} tokens.")
    lines.append("Answer:")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# LLM clients
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LLMResponse:
    """One model answer.

    Attributes:
        text: The raw answer text (``run`` strips surrounding whitespace).
        prompt_tokens: Input tokens reported by the API (instructions + context + question).
        output_tokens: Answer tokens reported by the API.
        thoughts_tokens: Thinking tokens reported by the API (0 if the model did not think).
        latency_s: Wall time of the successful request.
        cached: True when replayed from the disk cache, so ``latency_s`` is not fresh.
    """

    text: str
    prompt_tokens: int
    output_tokens: int
    thoughts_tokens: int
    latency_s: float
    cached: bool = False


class LLMClient(Protocol):
    """What ``run`` needs from a model. Tests pass a fake implementation."""

    model: str

    def generation_config(self) -> dict:
        """Every setting that affects the answer. Part of the cache key."""
        ...

    def generate(self, prompt: str) -> LLMResponse: ...

    def count_tokens(self, text: str) -> int: ...


def load_env(path: Path = ROOT / ".env") -> None:
    """Copy ``KEY=value`` lines from a ``.env`` file into ``os.environ``.

    Variables that are already set win. Blank lines and ``#`` comments are skipped.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


class GeminiClient:
    """Gemini with fixed generation settings, retrying transient API errors with backoff.

    The API key is read from ``GEMINI_API_KEY`` (environment or ``.env``).
    """

    def __init__(self, model: str = MODEL):
        from google import genai  # imported lazily so tests never need the SDK or network

        load_env()
        self.model = model
        self._client = genai.Client()

    def generation_config(self) -> dict:
        return {"temperature": TEMPERATURE, "seed": SEED, "max_output_tokens": MAX_OUTPUT_TOKENS}

    def generate(self, prompt: str) -> LLMResponse:
        from google.genai import errors, types

        config = types.GenerateContentConfig(**self.generation_config())
        for attempt in range(MAX_RETRIES + 1):
            start = time.perf_counter()
            try:
                response = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
            except errors.APIError as error:
                if error.code not in RETRYABLE_CODES or attempt == MAX_RETRIES:
                    raise
                time.sleep(retry_delay(error.details, attempt))
                continue
            latency = time.perf_counter() - start
            usage = response.usage_metadata
            return LLMResponse(
                text=response.text or "",
                prompt_tokens=usage.prompt_token_count or 0,
                output_tokens=usage.candidates_token_count or 0,
                thoughts_tokens=usage.thoughts_token_count or 0,
                latency_s=latency,
            )
        raise RuntimeError("unreachable")

    def count_tokens(self, text: str) -> int:
        return self._client.models.count_tokens(model=self.model, contents=text).total_tokens


def retry_delay(error_details: object, attempt: int) -> float:
    """Seconds to wait before retrying a failed call.

    Uses the server's ``RetryInfo.retryDelay`` (e.g. ``"12.3s"``) plus one second when the
    error carries one, as rate-limit (429) errors do. Otherwise backs off exponentially.
    Both are capped at ``MAX_BACKOFF_S``.
    """
    details = error_details.get("error", {}).get("details", []) if isinstance(error_details, dict) else []
    for detail in details:
        if isinstance(detail, dict) and str(detail.get("@type", "")).endswith("RetryInfo"):
            try:
                return min(float(str(detail["retryDelay"]).rstrip("s")) + 1.0, MAX_BACKOFF_S)
            except (KeyError, ValueError):
                break
    return min(2.0 ** attempt, MAX_BACKOFF_S)


class CachedClient:
    """Wraps any ``LLMClient`` and stores every answer and token count on disk.

    Keys are ``sha256`` hashes of the model, generation config and text, so reruns with the
    same prompt are free and identical. Replayed answers are flagged ``cached=True``.
    """

    def __init__(self, inner: LLMClient, cache_dir: Path = CACHE_DIR):
        self.inner = inner
        self.model = inner.model
        self.cache_dir = cache_dir

    def generation_config(self) -> dict:
        return self.inner.generation_config()

    def _path(self, kind: str, *parts: str) -> Path:
        digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
        return self.cache_dir / kind / f"{digest}.json"

    def generate(self, prompt: str) -> LLMResponse:
        config = json.dumps(self.generation_config(), sort_keys=True)
        path = self._path("llm", self.model, config, prompt)
        if path.exists():
            stored = json.loads(path.read_text(encoding="utf-8"))
            return LLMResponse(**{**stored, "cached": True})
        response = self.inner.generate(prompt)
        _write_json(path, asdict(response))
        return response

    def count_tokens(self, text: str) -> int:
        path = self._path("tokens", self.model, text)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["tokens"]
        tokens = self.inner.count_tokens(text)
        _write_json(path, {"tokens": tokens})
        return tokens


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def load_questions(path: str | Path) -> tuple[str, list[dict]]:
    """Read a questions file and return ``(doc_id, questions)``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("doc_id") or Path(path).stem.removesuffix("_questions"), data["questions"]


def run(
    contexts: list[str],
    questions: list[str],
    size: int | None = None,
    hash_count: int | None = None,
    tier: str = "curated",
    stats_only: bool = False,
    calc_fpr: bool = False,
    baseline: bool = False,
    out: str | Path | None = "response.json",
    client: LLMClient | None = None,
) -> dict:
    """Answer every question of every context and record stats (schema: specs.md §7.1).

    Args:
        contexts: Context file paths. ``contexts[i]`` pairs with ``questions[i]``.
        questions: Questions file paths.
        size: Bloom filter bits R (ignored when ``baseline``).
        hash_count: Bloom filter hash functions k (ignored when ``baseline``).
        tier: Vocabulary tier (ignored when ``baseline``).
        stats_only: Count prompt tokens with ``count_tokens`` instead of calling the LLM.
        calc_fpr: Also build the exact set of the vocabulary, to measure the empirical FPR and
            compare the filter's memory with the set's.
        baseline: Send the full, uncompressed context.
        out: Where to write the result JSON; ``None`` skips writing.
        client: The model. Defaults to a cached ``GeminiClient``.

    Returns:
        The result dict that is also written to ``out``.
    """
    if len(contexts) != len(questions):
        raise ValueError(
            f"Got {len(contexts)} contexts but {len(questions)} questions files; they pair by position"
        )
    if client is None:
        client = CachedClient(GeminiClient())

    bf, exact, filter_stats = (None, None, None) if baseline else _build_filter(size, hash_count, tier, calc_fpr)
    result = {
        "config": {
            "baseline": baseline,
            "tier": None if baseline else tier,
            "size": None if baseline else size,
            "hash_count": None if baseline else hash_count,
            "model": client.model,
            "seed": SEED,
            "generation_config": client.generation_config(),
            "stats_only": stats_only,
        },
        "filter": filter_stats,
        "docs": [],
    }

    for context_path, questions_path in zip(contexts, questions):
        doc_id, doc_questions = load_questions(questions_path)
        text = Path(context_path).read_text(encoding="utf-8")
        print(f"[{doc_id}] {len(doc_questions)} questions", file=sys.stderr)

        start = time.perf_counter()
        compressed, fpr_stats = (text, {}) if bf is None else compress(text, bf, exact)
        compress_s = time.perf_counter() - start

        words_in, words_out = len(text.split()), len(compressed.split())
        doc = {
            "doc_id": doc_id,
            "words_in": words_in,
            "words_out": words_out,
            "chars_in": len(text),
            "chars_out": len(compressed),
            "retained_frac": words_out / words_in if words_in else 1.0,
            "compress_s": compress_s,
            "context_tokens": client.count_tokens(compressed),
        }
        if calc_fpr and not baseline:
            doc["fpr"] = fpr_stats
        doc["questions"] = [_answer(client, compressed, q, compress_s, stats_only) for q in doc_questions]
        result["docs"].append(doc)

    if out is not None:
        _write_json(Path(out), result)
    return result


def _build_filter(size: int | None, hash_count: int | None, tier: str,
                  calc_fpr: bool) -> tuple[BloomFilter, frozenset[str] | None, dict]:
    """Build the tier's filter; return it, the exact set (only with ``calc_fpr``) and its stats."""
    if size is None or hash_count is None:
        raise ValueError("size and hash_count are required unless baseline=True")
    vocabulary = build_vocabulary(tier)
    start = time.perf_counter()
    bf = BloomFilter.from_words(vocabulary, size, hash_count)
    stats = {
        "n_words": len(vocabulary),
        "memory_bits": bf.memory_bits,
        "build_s": time.perf_counter() - start,
        "theoretical_fpr": BloomFilter.theoretical_fpr(len(vocabulary), size, hash_count),
    }
    if not calc_fpr:
        return bf, None, stats
    from pympler.asizeof import asizeof  # only needed for the memory comparison

    exact = frozenset(vocabulary)
    stats["bf_asizeof_bytes"] = asizeof(bf)
    stats["set_asizeof_bytes"] = asizeof(exact)
    return bf, exact, stats


def _answer(client: LLMClient, context: str, question: dict, compress_s: float, stats_only: bool) -> dict:
    """Ask (or, in stats-only mode, just measure) one question and return its record."""
    prompt = build_prompt(context, question)
    if stats_only:
        return {
            "qid": question["qid"],
            "prompt_tokens": client.count_tokens(prompt),
            "token_source": "count_tokens",
            "max_answer_tokens": question.get("max_answer_tokens"),
            "answer": None,
            "output_tokens": None,
            "thoughts_tokens": None,
            "llm_s": None,
            "latency_s": compress_s,
            "cached": None,
        }
    response = client.generate(prompt)
    return {
        "qid": question["qid"],
        "prompt_tokens": response.prompt_tokens,
        "token_source": "usage",
        "max_answer_tokens": question.get("max_answer_tokens"),
        "answer": response.text.strip(),
        "output_tokens": response.output_tokens,
        "thoughts_tokens": response.thoughts_tokens,
        "llm_s": response.latency_s,
        "latency_s": compress_s + response.latency_s,
        "cached": response.cached,
    }


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--contexts", nargs="+", required=True, help="context .txt files")
    parser.add_argument("--questions", nargs="+", required=True, help="questions .json files, paired by position")
    parser.add_argument("--size", type=int, help="Bloom filter bits R")
    parser.add_argument("--hash-count", type=int, help="Bloom filter hash functions k")
    parser.add_argument("--tier", default="curated", help="vocabulary tier")
    parser.add_argument("--stats-only", action="store_true", help="count tokens instead of calling the LLM")
    parser.add_argument("--calc-fpr", action="store_true", help="measure empirical FPR against an exact set")
    parser.add_argument("--baseline", action="store_true", help="send the full, uncompressed context")
    parser.add_argument("--out", default="response.json", help="output JSON path")
    args = parser.parse_args(argv)
    if not args.baseline and (args.size is None or args.hash_count is None):
        parser.error("--size and --hash-count are required unless --baseline is given")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run(
        args.contexts,
        args.questions,
        size=args.size,
        hash_count=args.hash_count,
        tier=args.tier,
        stats_only=args.stats_only,
        calc_fpr=args.calc_fpr,
        baseline=args.baseline,
        out=args.out,
    )
    print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

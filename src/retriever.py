"""Choose which chunks of a document to send to the LLM for one question (specs.md §6).

:class:`BloomRetriever` scores every chunk for a question, using only
:meth:`bloom.ChunkIndex.query`. It then fills a token budget greedily and
assembles the chosen chunks into a compressed context. :class:`FullContextSelector`
is the baseline: it implements the same ``select`` interface and returns the
whole document unchanged.

The retriever never sees chunk term sets, only the index. Bloom false positives
therefore feed straight into scores, weights and the final selection, which is
exactly what the experiments measure.

Depends on: :mod:`text` (chunks, question terms), :mod:`bloom` (the index interface) and numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from bloom import ChunkIndex
from text import Chunk, TermParams, extract_terms

CHARS_PER_TOKEN = 4  # budget estimate: est(text) = ceil(len(text) / 4), see §6.2


@dataclass(frozen=True)
class RetrievalParams:
    """Scoring and selection settings. The defaults are the frozen core method (§6.3).

    Attributes:
        idf_weighting: Weight terms by ``log((n+1)/(df_hat+1))``; if False, every weight is 1.
        alpha: Keep chunks that score at least ``alpha × top score``.
        neighbours: Add up to this many chunks on each side of a kept chunk, within its section.
        budget_tokens: Hard cap on the estimated tokens of the assembled context.
        lambda_bigram: Multiplier on bigram weights.
        lambda_number: Multiplier on number weights.
        char_coverage_min: Minimum fraction of a word's char n-grams that must hit
            for a fuzzy match. Only used when char n-grams are enabled.
    """

    idf_weighting: bool = True
    alpha: float = 0.5
    neighbours: int = 1
    budget_tokens: int = 1500
    lambda_bigram: float = 1.5
    lambda_number: float = 2.0
    char_coverage_min: float = 0.75


@dataclass(frozen=True)
class Selection:
    """The compressed context chosen for one question.

    Attributes:
        text: What is sent to the LLM as context.
        chunk_ids: The chunks included, in document order (empty for the full context).
        status: ``"ok"``, ``"no_match"`` (no usable terms, or nothing scored) or ``"full"`` (baseline).
        match_score: Top score divided by the highest score possible. This is a
            heuristic diagnostic, not a probability, and is 0 when undefined.
        est_tokens: ``ceil(len(text) / 4)``. Real token counts come from the LLM API.
        retained_frac: ``len(text) / len(document)``.
    """

    text: str
    chunk_ids: tuple[int, ...]
    status: str
    match_score: float
    est_tokens: int
    retained_frac: float


def estimate_tokens(text: str) -> int:
    """Return the deterministic token estimate used for budgets and offline reporting."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


class Selector(Protocol):
    """Anything that turns a question into a :class:`Selection`."""

    def select(self, question: str) -> Selection: ...


class FullContextSelector:
    """Baseline selector: the whole document, unchanged (§10.1)."""

    def __init__(self, document: str):
        self.document = document

    def select(self, question: str) -> Selection:
        return Selection(self.document, (), "full", 1.0, estimate_tokens(self.document), 1.0)


@dataclass(frozen=True)
class Scores:
    """Per-chunk scores for one question, plus the highest score any chunk could reach."""

    values: np.ndarray
    max_possible: float


class BloomRetriever:
    """Score chunks with a chunk index, then select and assemble a context within budget."""

    def __init__(
        self,
        chunks: list[Chunk],
        index: ChunkIndex,
        document_length: int,
        term_params: TermParams = TermParams(),
        params: RetrievalParams = RetrievalParams(),
    ):
        if index.n_chunks != len(chunks):
            raise ValueError("index and chunk list have different lengths")
        self.chunks = chunks
        self.index = index
        self.document_length = max(document_length, 1)
        self.term_params = term_params
        self.params = params

    # ----------------------------------------------------------------- scoring

    def score(self, question: str) -> Scores:
        """Score every chunk for ``question`` (§6.1). Every membership test goes through the index."""
        terms = extract_terms(question, self.term_params)
        n = self.index.n_chunks
        hits_cache: dict[str, np.ndarray] = {}

        def hits(term: str) -> np.ndarray:
            if term not in hits_cache:
                hits_cache[term] = self.index.query(term)
            return hits_cache[term]

        def weight(term: str) -> float:
            if not self.params.idf_weighting:
                return 1.0
            df_hat = int(hits(term).sum())  # equals index.df_hat(term)
            return math.log((n + 1) / (df_hat + 1))

        values = np.zeros(n)
        max_possible = 0.0
        for word in terms.words:
            term = f"w:{word}"
            w = weight(term)
            match = hits(term).astype(float)
            if word in terms.char_grams:
                match = np.maximum(match, self._fuzzy_match(terms.char_grams[word], hits))
            values += w * match
            max_possible += w
        for bigram in terms.bigrams:
            w = self.params.lambda_bigram * weight(f"b:{bigram}")
            values += w * hits(f"b:{bigram}")
            max_possible += w
        for number in terms.numbers:
            w = self.params.lambda_number * weight(f"n:{number}")
            values += w * hits(f"n:{number}")
            max_possible += w
        return Scores(values, max_possible)

    def _fuzzy_match(self, grams: tuple[str, ...], hits) -> np.ndarray:
        """Return per-chunk coverage of a word's char n-grams, zeroed below the threshold."""
        coverage = np.mean([hits(f"c:{g}") for g in grams], axis=0)
        return np.where(coverage >= self.params.char_coverage_min, coverage, 0.0)

    # --------------------------------------------------------------- selection

    def select(self, question: str) -> Selection:
        """Choose chunks for ``question`` and assemble them within the budget (§6.2)."""
        scores = self.score(question)
        values = scores.values
        top = float(values.max()) if len(values) else 0.0
        if scores.max_possible <= 0 or top <= 0:
            return Selection("", (), "no_match", 0.0, 0, 0.0)

        threshold = self.params.alpha * top
        candidates = [int(c) for c in np.flatnonzero((values >= threshold) & (values > 0))]
        ranked = sorted(candidates, key=lambda c: (-values[c], c))  # ties: lower id first
        chosen = self._fill_budget(ranked)
        text = self.assemble(chosen)
        # Safety net: the cost model counts each breadcrumb once, but assembly
        # repeats one when sections interleave, so trim from the lowest-priority end.
        while chosen and estimate_tokens(text) > self.params.budget_tokens:
            chosen.pop()
            text = self.assemble(chosen)
        return Selection(
            text=text,
            chunk_ids=tuple(sorted(chosen)),
            status="ok",
            match_score=top / scores.max_possible,
            est_tokens=estimate_tokens(text),
            retained_frac=len(text) / self.document_length,
        )

    def _fill_budget(self, ranked: list[int]) -> list[int]:
        """Greedily add ranked chunks, then their neighbours, while the budget allows.

        Returns chunk ids in the order they were added, which is their priority order.
        """
        budget_chars = self.params.budget_tokens * CHARS_PER_TOKEN
        chosen: list[int] = []
        chosen_set: set[int] = set()
        breadcrumbs: set[str] = set()
        headers: set[int] = set()
        used = 0

        def cost(c: int) -> int:
            chunk = self.chunks[c]
            chars = len(chunk.text) + 1
            if chunk.breadcrumb and chunk.breadcrumb not in breadcrumbs:
                chars += len(section_marker(chunk.breadcrumb)) + 1
            header = chunk.header_ref
            if header is not None and header not in headers and header not in chosen_set:
                chars += len(self.chunks[header].text) + 1
            return chars

        def try_add(c: int) -> None:
            nonlocal used
            needed = cost(c)
            if c in chosen_set or used + needed > budget_chars:
                return
            chosen.append(c)
            chosen_set.add(c)
            breadcrumbs.add(self.chunks[c].breadcrumb)
            if self.chunks[c].header_ref is not None:
                headers.add(self.chunks[c].header_ref)
            used += needed

        for c in ranked:
            try_add(c)
        core = list(chosen)
        for c in core:
            for offset in range(1, self.params.neighbours + 1):
                for neighbour in (c - offset, c + offset):
                    if 0 <= neighbour < len(self.chunks) and self._same_section(c, neighbour):
                        try_add(neighbour)
        return chosen

    def _same_section(self, a: int, b: int) -> bool:
        return self.chunks[a].section_id == self.chunks[b].section_id

    def assemble(self, chunk_ids: list[int]) -> str:
        """Join chunks in document order under ``[§ breadcrumb]`` markers.

        Each table row's header appears once, above the first row of that table
        that was selected.
        """
        lines: list[str] = []
        current_breadcrumb: str | None = None
        printed_headers = set(chunk_ids)  # a selected header row prints itself
        for c in sorted(chunk_ids):
            chunk = self.chunks[c]
            if chunk.breadcrumb != current_breadcrumb:
                if chunk.breadcrumb:
                    lines.append(section_marker(chunk.breadcrumb))
                current_breadcrumb = chunk.breadcrumb
            header = chunk.header_ref
            if header is not None and header not in printed_headers:
                lines.append(self.chunks[header].text)
                printed_headers.add(header)
            lines.append(chunk.text)
        return "\n".join(lines)


def section_marker(breadcrumb: str) -> str:
    return f"[§ {breadcrumb}]"

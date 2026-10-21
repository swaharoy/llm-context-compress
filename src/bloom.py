"""Bloom filters and per-chunk Bloom indexes (specs.md §5, §5.1).

This module is the graded data structure. It hashes strings into bits and answers
"is this string probably in the set?". It knows nothing about text: terms arrive
from :mod:`text` as plain strings.

Contents:

* :func:`positions`: the k bit positions of a term, by seeded double hashing.
* :class:`BloomFilter`: one standalone filter, the textbook structure.
* Three interchangeable *chunk indexes*, one filter per document chunk, all
  implementing :class:`ChunkIndex`:

  - :class:`RowMajorBloomIndex` (core): one packed row of R bits per chunk.
  - :class:`BitSlicedBloomIndex` (Stage 2 add-on): the same bits stored
    transposed, so one query tests every chunk with k word-wide ANDs.
  - :class:`ExactSetIndex` (control): Python sets, with no false positives.

* :func:`make_fpr_probes` and :func:`measure_fpr`: the measured-FPR protocol (§8).

Depends on: numpy and the standard library.
"""

from __future__ import annotations

import hashlib
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Parameters and hashing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BloomParams:
    """Bloom filter size and hashing.

    Attributes:
        R: Bits per filter. Must be a power of two, at least 8.
        k: Hash positions per term. ``None`` means :func:`optimal_k` for the
            mean number of terms per chunk.
        seed: Key for the hash function. Different seeds give independent filters.
    """

    R: int = 1024
    k: int | None = None
    seed: int = 580


def is_power_of_two(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def optimal_k(R: int, N: float) -> int:
    """Return the number of hashes that minimises FPR for N items in R bits: ``round(R/N · ln 2)``, at least 1."""
    return max(1, round(R / N * math.log(2))) if N > 0 else 1


def theoretical_fpr(R: int, k: int, N: float) -> float:
    """Return the standard approximation ``(1 − e^{−kN/R})^k`` of a filter's false-positive rate."""
    return (1.0 - math.exp(-k * N / R)) ** k


@lru_cache(maxsize=1 << 18)
def positions(term: str, R: int, k: int, seed: int) -> tuple[int, ...]:
    """Return the k bit positions of ``term`` in a filter of R bits.

    One keyed blake2b digest is split into two 64-bit halves h1 and h2. Standard
    double hashing then gives ``g_i = (h1 + i·h2) mod R``.

    * blake2b with an explicit key is used instead of Python's ``hash()``, which is
      salted per process. That would make every run set different bits.
    * h2 is forced odd. Because R is a power of two, h2 is then coprime with R, so
      the k positions are distinct for any k ≤ R.

    The result is cached, and the seed is part of the key, so a term is hashed
    only once per (R, k, seed) however many filters it is inserted into.
    """
    if not is_power_of_two(R) or R < 8:
        raise ValueError(f"R must be a power of two >= 8, got {R}")
    if not 1 <= k <= R:
        raise ValueError(f"k must be in [1, R], got {k}")
    digest = hashlib.blake2b(
        term.encode("utf-8"), digest_size=16, key=seed.to_bytes(8, "little")
    ).digest()
    h1 = int.from_bytes(digest[:8], "little")
    h2 = int.from_bytes(digest[8:], "little") | 1
    return tuple((h1 + i * h2) % R for i in range(k))


# --------------------------------------------------------------------------- #
# A single Bloom filter
# --------------------------------------------------------------------------- #


class BloomFilter:
    """A standalone Bloom filter over strings: R bits, k hash positions per item.

    Adding an item sets its k bits. Membership is "all k bits are set", so there
    are never false negatives, and false positives occur when other items happen
    to have set all k bits.
    """

    def __init__(self, R: int, k: int, seed: int = BloomParams.seed):
        positions("", R, k, seed)  # validates R and k
        self.R, self.k, self.seed = R, k, seed
        self.bits = np.zeros(R // 8, dtype=np.uint8)
        self.n_inserted = 0

    def add(self, item: str) -> None:
        """Insert one item. Callers insert distinct items, so ``n_inserted`` is N."""
        for p in positions(item, self.R, self.k, self.seed):
            self.bits[p >> 3] |= np.uint8(1 << (p & 7))
        self.n_inserted += 1

    def add_many(self, items: Sequence[str] | frozenset[str]) -> None:
        for item in items:
            self.add(item)

    def __contains__(self, item: str) -> bool:
        return all(
            (self.bits[p >> 3] >> (p & 7)) & 1
            for p in positions(item, self.R, self.k, self.seed)
        )

    @property
    def memory_bits(self) -> int:
        """Size of the bit array as actually allocated."""
        return self.bits.nbytes * 8

    @property
    def fill_ratio(self) -> float:
        """Fraction of bits set. Values near 0.5 are ideal; near 1 everything matches."""
        return int(np.bitwise_count(self.bits).sum()) / self.R

    def theoretical_fpr(self) -> float:
        return theoretical_fpr(self.R, self.k, self.n_inserted)


# --------------------------------------------------------------------------- #
# Chunk indexes: one filter per chunk
# --------------------------------------------------------------------------- #


class ChunkIndex(Protocol):
    """What the retriever needs from an index. Term sets are deliberately not exposed."""

    n_chunks: int

    def query(self, term: str) -> np.ndarray:
        """Return a boolean vector of length ``n_chunks``: which chunks might contain ``term``."""
        ...

    def df_hat(self, term: str) -> int:
        """Return the number of chunks that claim ``term``: true df plus false positives."""
        ...

    def memory_bits(self) -> int:
        """Return the number of Bloom-filter bits allocated (0 for the exact control)."""
        ...

    def stats(self) -> dict:
        """Return size, fill and theoretical-FPR statistics, for the results files."""
        ...


def resolve_k(params: BloomParams, chunk_terms: Sequence[frozenset[str]]) -> int:
    """Return ``params.k``, or the optimal k for the mean chunk size when it is ``None``."""
    if params.k is not None:
        return params.k
    sizes = [len(terms) for terms in chunk_terms]
    return optimal_k(params.R, sum(sizes) / max(len(sizes), 1))


def _insertion_coords(
    chunk_terms: Sequence[frozenset[str]], R: int, k: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return parallel arrays ``(chunk_id, bit_position)``, one entry per bit to set."""
    chunk_ids: list[int] = []
    bit_positions: list[int] = []
    for chunk_id, terms in enumerate(chunk_terms):
        for term in terms:
            p = positions(term, R, k, seed)
            bit_positions.extend(p)
            chunk_ids.extend([chunk_id] * k)
    return np.array(chunk_ids, dtype=np.int64), np.array(bit_positions, dtype=np.int64)


def _filter_stats(sizes: np.ndarray, R: int, k: int, set_bits_per_chunk: np.ndarray) -> dict:
    """Summarise the per-chunk theoretical FPR, using each chunk's own N.

    ``fpr_from_fill`` is the mean of ``fill_c ** k``. That is the false-positive
    probability given the bits each filter actually ended up with, and it shows
    how much of any measured-vs-formula gap comes from fill varying between
    chunks, which a high k magnifies.
    """
    fprs = [theoretical_fpr(R, k, n) for n in sizes] or [0.0]
    fills = set_bits_per_chunk / R
    return {
        "fpr_from_fill": float(np.mean(fills ** k)) if len(fills) else 0.0,
        "N_mean": float(sizes.mean()) if len(sizes) else 0.0,
        "N_max": int(sizes.max()) if len(sizes) else 0,
        "fpr_theoretical": float(np.mean(fprs)),
        "fpr_theoretical_min": float(np.min(fprs)),
        "fpr_theoretical_max": float(np.max(fprs)),
    }


class RowMajorBloomIndex:
    """Core index: chunk c's filter is row c of an ``n × R/8`` packed byte matrix.

    A query gathers the term's k bit columns for all chunks at once with numpy,
    then ANDs them across the k columns.
    """

    def __init__(self, chunk_terms: Sequence[frozenset[str]], params: BloomParams):
        self.R, self.k, self.seed = params.R, resolve_k(params, chunk_terms), params.seed
        self.n_chunks = len(chunk_terms)
        self.sizes = np.array([len(terms) for terms in chunk_terms])
        self.rows = np.zeros((self.n_chunks, self.R // 8), dtype=np.uint8)
        chunk_ids, bit_positions = _insertion_coords(chunk_terms, self.R, self.k, self.seed)
        masks = np.left_shift(1, bit_positions & 7).astype(np.uint8)
        np.bitwise_or.at(self.rows, (chunk_ids, bit_positions >> 3), masks)

    def query(self, term: str) -> np.ndarray:
        p = np.array(positions(term, self.R, self.k, self.seed))
        selected_bytes = self.rows[:, p >> 3]  # n × k
        bits = (selected_bytes >> (p & 7)) & 1
        return bits.all(axis=1)

    def df_hat(self, term: str) -> int:
        return int(self.query(term).sum())

    def memory_bits(self) -> int:
        return self.rows.nbytes * 8

    def stats(self) -> dict:
        set_bits = int(np.bitwise_count(self.rows).sum())
        return {
            "index": "rowmajor", "R": self.R, "k": self.k, "n_chunks": self.n_chunks,
            "bloom_bits": self.memory_bits(), "padding_bits": 0,
            "fill_ratio": set_bits / max(self.n_chunks * self.R, 1),
            **_filter_stats(self.sizes, self.R, self.k, np.bitwise_count(self.rows).sum(axis=1)),
        }


class BitSlicedBloomIndex:
    """The same per-chunk filters as :class:`RowMajorBloomIndex`, stored transposed (§5.1).

    Row j of ``slices`` (shape ``R × ⌈n/64⌉``, ``uint64``) is a bitmap over chunks:
    chunk c's bit is set exactly when chunk c's filter has bit j set. Chunk c is
    bit ``c % 64`` of word ``c // 64`` (LSB-first). Querying a term ANDs its k
    slices, which tests 64 chunks per machine-word operation, and stops early
    once nothing survives.
    """

    def __init__(self, chunk_terms: Sequence[frozenset[str]], params: BloomParams):
        self.R, self.k, self.seed = params.R, resolve_k(params, chunk_terms), params.seed
        self.n_chunks = len(chunk_terms)
        self.sizes = np.array([len(terms) for terms in chunk_terms])
        self.n_words = math.ceil(self.n_chunks / 64)
        # Explicitly little-endian, so the byte view used in `query` is LSB-first on any host.
        self.slices = np.zeros((self.R, self.n_words), dtype="<u8")
        chunk_ids, bit_positions = _insertion_coords(chunk_terms, self.R, self.k, self.seed)
        masks = np.left_shift(np.uint64(1), (chunk_ids & 63).astype(np.uint64))
        np.bitwise_or.at(self.slices, (bit_positions, chunk_ids >> 6), masks)

    def _bitmap(self, term: str) -> np.ndarray:
        """Return the packed chunk bitmap for ``term``: the AND of its k slices."""
        first, *rest = positions(term, self.R, self.k, self.seed)
        acc = self.slices[first].copy()
        for p in rest:
            acc &= self.slices[p]
            if not acc.any():
                break
        return acc

    def query(self, term: str) -> np.ndarray:
        bits = np.unpackbits(self._bitmap(term).view(np.uint8), bitorder="little")
        return bits[: self.n_chunks].astype(bool)

    def df_hat(self, term: str) -> int:
        return int(np.bitwise_count(self._bitmap(term)).sum())

    def memory_bits(self) -> int:
        return self.slices.nbytes * 8

    def stats(self) -> dict:
        set_bits = int(np.bitwise_count(self.slices).sum())
        return {
            "index": "bitsliced", "R": self.R, "k": self.k, "n_chunks": self.n_chunks,
            "bloom_bits": self.memory_bits(),
            "padding_bits": (self.n_words * 64 - self.n_chunks) * self.R,
            "fill_ratio": set_bits / max(self.n_chunks * self.R, 1),
            **_filter_stats(self.sizes, self.R, self.k, self._set_bits_per_chunk()),
        }

    def _set_bits_per_chunk(self) -> np.ndarray:
        """Return the number of set bits in each chunk's filter (a column count over the slices)."""
        bits = np.unpackbits(self.slices.view(np.uint8), axis=1, bitorder="little")
        return bits[:, : self.n_chunks].sum(axis=0)


class ExactSetIndex:
    """Control index: exact Python sets, so FPR = 0.

    It runs the same retrieval algorithm with the Bloom filter's approximation
    removed. The accuracy gap between this and a Bloom index is therefore the
    cost of false positives (report question 3). Its memory is reported as
    ``exact_set_bytes`` and never counted as Bloom memory.
    """

    def __init__(self, chunk_terms: Sequence[frozenset[str]], params: BloomParams | None = None):
        self.sets = [frozenset(terms) for terms in chunk_terms]
        self.n_chunks = len(self.sets)

    def query(self, term: str) -> np.ndarray:
        return np.fromiter((term in s for s in self.sets), dtype=bool, count=self.n_chunks)

    def df_hat(self, term: str) -> int:
        return sum(term in s for s in self.sets)

    def memory_bits(self) -> int:
        return 0

    def stats(self) -> dict:
        exact_bytes = sum(sys.getsizeof(s) + sum(sys.getsizeof(t) for t in s) for s in self.sets)
        sizes = np.array([len(s) for s in self.sets])
        return {
            "index": "exact", "R": None, "k": None, "n_chunks": self.n_chunks,
            "bloom_bits": 0, "padding_bits": 0, "fill_ratio": None,
            "exact_set_bytes": exact_bytes,
            "N_mean": float(sizes.mean()) if len(sizes) else 0.0,
            "N_max": int(sizes.max()) if len(sizes) else 0,
            "fpr_theoretical": 0.0, "fpr_theoretical_min": 0.0, "fpr_theoretical_max": 0.0,
            "fpr_from_fill": 0.0,
        }


INDEX_KINDS = {
    "rowmajor": RowMajorBloomIndex,
    "bitsliced": BitSlicedBloomIndex,
    "exact": ExactSetIndex,
}


def build_index(kind: str, chunk_terms: Sequence[frozenset[str]], params: BloomParams) -> ChunkIndex:
    """Build the index named ``kind`` (``rowmajor``, ``bitsliced`` or ``exact``)."""
    if kind not in INDEX_KINDS:
        raise ValueError(f"unknown index kind {kind!r}; choose from {sorted(INDEX_KINDS)}")
    return INDEX_KINDS[kind](chunk_terms, params)


# --------------------------------------------------------------------------- #
# Measured false-positive rate (§8)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FprProbes:
    """Probe terms plus their ground truth: ``truth[i, c]`` says whether chunk c really contains probe i."""

    terms: tuple[str, ...]
    truth: np.ndarray


def make_fpr_probes(chunk_terms: Sequence[frozenset[str]], n_probes: int = 1000, seed: int = 580) -> FprProbes:
    """Build the probe set for :func:`measure_fpr`.

    Half the probes are terms from the document, so they are real-looking and
    absent from most chunks. The other half are random strings that are absent
    from every chunk. The ground truth comes from the exact term sets. This is
    diagnostic code; it is not on any retrieval path.
    """
    rng = np.random.default_rng(seed)
    vocabulary = sorted(set().union(*chunk_terms)) if chunk_terms else []
    n_doc = min(n_probes // 2, len(vocabulary))
    doc_terms = [str(t) for t in rng.choice(vocabulary, size=n_doc, replace=False)] if n_doc else []
    letters = np.array(list("abcdefghijklmnopqrstuvwxyz"))
    random_terms = ["x:" + "".join(rng.choice(letters, size=12)) for _ in range(n_probes - n_doc)]
    terms = tuple(doc_terms + random_terms)
    truth = np.array([[t in chunk for chunk in chunk_terms] for t in terms], dtype=bool)
    return FprProbes(terms, truth.reshape(len(terms), len(chunk_terms)))


def measure_fpr(index: ChunkIndex, probes: FprProbes) -> float:
    """Return the fraction of (probe, chunk) pairs where the chunk lacks the probe but its filter claims it.

    Every chunk's filter is probed with every probe term. Pairs where the chunk
    really contains the term are excluded, so this is a pure false-positive rate.
    """
    false_positives = 0
    negatives = 0
    for term, truly_present in zip(probes.terms, probes.truth):
        claimed = index.query(term)
        absent = ~truly_present
        negatives += int(absent.sum())
        false_positives += int((claimed & absent).sum())
    return false_positives / negatives if negatives else 0.0

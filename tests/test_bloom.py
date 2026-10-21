"""Tests for bloom.py: correctness, FPR against theory, hashing stability, memory and index equivalence."""

import math
import os
import subprocess
import sys

import numpy as np
import pytest

from bloom import (BitSlicedBloomIndex, BloomFilter, BloomParams, ExactSetIndex, RowMajorBloomIndex,
                   make_fpr_probes, measure_fpr, optimal_k, positions, theoretical_fpr)
from retriever import BloomRetriever
from text import Chunk, TermParams


def random_terms(n: int, seed: int, prefix: str = "w:") -> list[str]:
    rng = np.random.default_rng(seed)
    letters = np.array(list("abcdefghijklmnopqrstuvwxyz"))
    return [prefix + "".join(rng.choice(letters, size=10)) for _ in range(n)]


def random_chunk_terms(n_chunks: int, seed: int) -> list[frozenset[str]]:
    """Chunks drawing from a shared vocabulary, so terms recur across chunks the way real words do."""
    rng = np.random.default_rng(seed)
    vocabulary = random_terms(400, seed)
    return [frozenset(rng.choice(vocabulary, size=int(rng.integers(5, 60)), replace=False))
            for _ in range(n_chunks)]


def test_no_false_negatives():
    bf = BloomFilter(R=256, k=4)
    items = random_terms(200, seed=1)
    bf.add_many(items)
    assert all(item in bf for item in items)


@pytest.mark.parametrize("R,k,N", [(1024, 1, 100), (1024, 3, 100), (1024, 7, 100), (512, 4, 120), (2048, 2, 300)])
def test_measured_fpr_within_3_sigma_of_expectation(R, k, N):
    bf = BloomFilter(R, k)
    bf.add_many(random_terms(N, seed=2))
    probes = random_terms(20_000, seed=3, prefix="x:")
    measured = sum(p in bf for p in probes) / len(probes)
    # Given the bits actually set, the expected FPR is fill^k. The 3-sigma band is for 20k probes.
    expected = bf.fill_ratio ** k
    sigma = math.sqrt(expected * (1 - expected) / len(probes))
    assert abs(measured - expected) <= 3 * sigma + 1e-4
    # The textbook formula is an approximation of the same quantity; it should be close.
    assert measured == pytest.approx(bf.theoretical_fpr(), rel=0.35, abs=2e-3)


def test_positions_are_distinct_for_any_k_up_to_R():
    for term in random_terms(200, seed=4):
        assert len(set(positions(term, 16, 16, 580))) == 16


def test_positions_reject_non_power_of_two_R():
    with pytest.raises(ValueError):
        positions("w:x", 1000, 3, 580)
    with pytest.raises(ValueError):
        positions("w:x", 64, 0, 580)


def test_seed_changes_positions():
    assert positions("w:berth", 1024, 4, 1) != positions("w:berth", 1024, 4, 2)


def test_hashing_is_stable_across_processes():
    """Python's own hash() differs per process. Ours must not."""
    code = "import sys; sys.path.insert(0, 'src'); from bloom import positions; print(positions('w:berth', 1024, 5, 580))"
    outputs = set()
    for hash_seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                                cwd=os.path.dirname(os.path.dirname(__file__)), check=True)
        outputs.add(result.stdout.strip())
    assert outputs == {str(positions("w:berth", 1024, 5, 580))}


def test_optimal_k_and_theory():
    assert optimal_k(1024, 100) == round(1024 / 100 * math.log(2))
    assert optimal_k(64, 1000) == 1
    assert theoretical_fpr(1024, 7, 0) == 0.0


def test_memory_bits_match_allocation():
    chunk_terms = random_chunk_terms(70, seed=5)  # 70 chunks: not a multiple of 64
    params = BloomParams(R=256, k=3)
    assert BloomFilter(256, 3).memory_bits == 256
    assert RowMajorBloomIndex(chunk_terms, params).memory_bits() == 70 * 256
    sliced = BitSlicedBloomIndex(chunk_terms, params)
    assert sliced.memory_bits() == 256 * 2 * 64  # two 64-bit words per slice
    assert sliced.stats()["padding_bits"] == (128 - 70) * 256


def test_indexes_agree_with_independent_filters():
    chunk_terms = random_chunk_terms(90, seed=6)
    params = BloomParams(R=512, k=4)
    filters = []
    for terms in chunk_terms:
        bf = BloomFilter(params.R, params.k, params.seed)
        bf.add_many(terms)
        filters.append(bf)
    row, sliced = RowMajorBloomIndex(chunk_terms, params), BitSlicedBloomIndex(chunk_terms, params)
    for term in random_terms(50, seed=7) + sorted(chunk_terms[0]):
        expected = np.array([term in f for f in filters])
        assert np.array_equal(row.query(term), expected)
        assert np.array_equal(sliced.query(term), expected)
        assert row.df_hat(term) == sliced.df_hat(term) == expected.sum()


def test_bitsliced_and_rowmajor_give_identical_score_vectors():
    """Comparing whole score vectors catches bit-order bugs that only show up in scoring."""
    chunk_terms = random_chunk_terms(130, seed=8)
    vocabulary = sorted(set().union(*chunk_terms))
    chunks = [Chunk(i, "x", 0, "", "para", (0, 1)) for i in range(len(chunk_terms))]
    params = BloomParams(R=128, k=3)  # small R, so false positives occur and must match too
    retrievers = [BloomRetriever(chunks, cls(chunk_terms, params), 1000, TermParams())
                  for cls in (RowMajorBloomIndex, BitSlicedBloomIndex)]
    rng = np.random.default_rng(9)
    for _ in range(30):
        question = " ".join(t[2:] for t in rng.choice(vocabulary, size=6))
        a, b = (r.score(question).values for r in retrievers)
        assert np.array_equal(a, b)


def test_df_hat_is_never_below_true_df():
    chunk_terms = random_chunk_terms(100, seed=10)
    index = RowMajorBloomIndex(chunk_terms, BloomParams(R=64, k=2))
    for term in sorted(set().union(*chunk_terms))[:100]:
        assert index.df_hat(term) >= sum(term in t for t in chunk_terms)


def test_exact_index_has_no_false_positives():
    chunk_terms = random_chunk_terms(40, seed=11)
    probes = make_fpr_probes(chunk_terms, n_probes=200, seed=12)
    assert measure_fpr(ExactSetIndex(chunk_terms), probes) == 0.0
    assert measure_fpr(RowMajorBloomIndex(chunk_terms, BloomParams(R=64, k=1)), probes) > 0.0

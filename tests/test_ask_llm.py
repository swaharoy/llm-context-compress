"""Unit tests for ask_llm.py. No network: the LLM is replaced by FakeClient."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

import ask_llm
from ask_llm import (BloomFilter, CachedClient, LLMResponse, build_prompt, build_vocabulary, compress,
                     load_env, lookup_key, parse_args, run)

ROOT = Path(__file__).resolve().parent.parent

QUESTION = {
    "number": 1,
    "qid": "doc-T2-007",
    "question": "How long was the boom?",
    "answer_type": "number",
    "unit": "ft",
    "max_answer_tokens": 8,
}


class FakeClient:
    """Deterministic stand-in for Gemini that records how it was called."""

    model = "fake-model"

    def __init__(self):
        self.generate_calls = 0
        self.count_calls = 0

    def generation_config(self) -> dict:
        return {"temperature": 0.0}

    def generate(self, prompt: str) -> LLMResponse:
        self.generate_calls += 1
        return LLMResponse(text=" 42 ft ", prompt_tokens=len(prompt.split()), output_tokens=2,
                           thoughts_tokens=0, latency_s=0.5)

    def count_tokens(self, text: str) -> int:
        self.count_calls += 1
        return len(text.split())


@pytest.fixture
def files(tmp_path: Path) -> tuple[list[str], list[str]]:
    """One tiny context and a questions file with two questions."""
    context = tmp_path / "doc.txt"
    context.write_text("The boom was 42 ft long.\n## Notes\nNothing else.", encoding="utf-8")
    second = {**QUESTION, "qid": "doc-T1-001", "answer_type": "entity", "question": "Who?"}
    del second["unit"]
    questions = tmp_path / "doc_questions.json"
    questions.write_text(json.dumps({"doc_id": "doc", "count": 2, "questions": [QUESTION, second]}))
    return [str(context)], [str(questions)]


# --------------------------------------------------------------------------- #
# BloomFilter
# --------------------------------------------------------------------------- #


def test_no_false_negatives():
    words = [f"word{i}" for i in range(500)]
    for size, k in [(64, 1), (1024, 3), (8192, 7)]:
        bf = BloomFilter.from_words(words, size, k)
        assert all(w in bf for w in words)


def test_empty_filter_contains_nothing():
    bf = BloomFilter(256, 3)
    assert "the" not in bf and not any(bf.bits)


def test_positions_are_in_range_and_count_k():
    bf = BloomFilter(100, 5)
    positions = bf.positions("example")
    assert len(positions) == 5 and all(0 <= p < 100 for p in positions)


def test_positions_are_identical_across_processes():
    """Keyed blake2b, not Python's per-process salted hash(), decides the bits."""
    code = "import ask_llm; print(ask_llm.BloomFilter(1 << 20, 7).positions('deterministic'))"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                         check=True, env={**os.environ, "PYTHONHASHSEED": "123"})
    assert out.stdout.strip() == str(BloomFilter(1 << 20, 7).positions("deterministic"))


def test_empirical_fpr_matches_theory():
    members = [f"member{i}" for i in range(1000)]
    probes = [f"probe{i}" for i in range(20000)]
    for size, k in [(4096, 3), (8192, 6), (2048, 1)]:
        bf = BloomFilter.from_words(members, size, k)
        measured = sum(p in bf for p in probes) / len(probes)
        expected = BloomFilter.theoretical_fpr(len(members), size, k)
        assert measured == pytest.approx(expected, rel=0.15, abs=0.003)


def test_theoretical_fpr_and_optimal_k_values():
    assert BloomFilter.theoretical_fpr(100, 1000, 7) == pytest.approx((1 - math.exp(-0.7)) ** 7)
    assert BloomFilter.optimal_k(100, 1000) == 7  # 10 bits per word * ln 2 = 6.93
    assert BloomFilter.optimal_k(100, 50) == 1  # never below one hash


def test_memory_is_r_bits():
    bf = BloomFilter(1001, 2)
    assert bf.memory_bits == 1001 and len(bf.bits) == 126


@pytest.mark.parametrize("size, k", [(0, 1), (8, 0), (-1, 3)])
def test_invalid_parameters_raise(size, k):
    with pytest.raises(ValueError):
        BloomFilter(size, k)


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #


def test_base_list_sizes():
    assert len(ask_llm.NLTK_STOPWORDS) == 179
    assert len(ask_llm.CURATED) == 108
    assert "|" not in ask_llm.DECORATIVE


def test_no_protected_word_in_any_filtered_tier():
    for tier in ask_llm.TIERS:
        if tier != "nltk_full":
            assert not ask_llm.PROTECTED & set(build_vocabulary(tier)), tier


def test_tiers_are_nested():
    tiers = [set(build_vocabulary(t)) for t in ("curated", "common500", "common1000", "common2000", "common5000")]
    for smaller, larger in zip(tiers, tiers[1:]):
        assert smaller < larger
    assert set(build_vocabulary("curated")) < set(build_vocabulary("nltk_full"))


def test_vocabulary_is_sorted_and_deterministic():
    words = build_vocabulary("common1000")
    assert words == sorted(set(words)) == build_vocabulary("common1000")
    assert all(w.isascii() for w in set(words) - ask_llm.DECORATIVE)


def test_unknown_tier_raises():
    with pytest.raises(ValueError, match="Unknown tier"):
        build_vocabulary("everything")


# --------------------------------------------------------------------------- #
# compress
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("token, key", [
    ("The", "the"), ("the,", "the"), ("(the)", "the"), ("“The”", "the"),
    ("It’s", "it's"), ("##", "##"), ("---", "---"), ("|", "|"), ("U.S.", "u.s"), ("42ft.", "42ft"),
])
def test_lookup_key(token, key):
    assert lookup_key(token) == key


def test_compress_drops_members_and_keeps_structure():
    bf = BloomFilter.from_words(["the", "of", "##"], 4096, 5)
    text = "## History\nThe length of the boom | 42 ft\n\n\n\nEnd of story."
    compressed, stats = compress(text, bf)
    assert compressed == "History\nlength boom | 42 ft\n\nEnd story."
    assert stats == {}


def test_compress_is_identity_when_nothing_matches():
    bf = BloomFilter.from_words(["zzz"], 4096, 5)
    text = "Line one has words.\nTable | row | 7\n\nLast line."
    assert compress(text, bf)[0] == text


def test_compress_collapses_lines_emptied_by_dropping():
    bf = BloomFilter.from_words(["---", "the"], 4096, 5)
    assert compress("a\n---\n\nthe\nb", bf)[0] == "a\n\nb"


def test_compress_counts_false_positives():
    """A one-bit filter reports every key as present, so every non-member is a false positive."""
    bf = BloomFilter.from_words(["the"], 1, 1)
    compressed, stats = compress("the cat saw the dog dog", bf, exact={"the"})
    assert compressed == ""
    assert stats == {"distinct_nonmembers": 3, "false_positives": 3, "empirical_fpr": 1.0, "fp_tokens_removed": 4}


def test_compress_reports_zero_fpr_without_false_positives():
    bf = BloomFilter.from_words(["the"], 1 << 16, 7)
    compressed, stats = compress("the cat", bf, exact={"the"})
    assert compressed == "cat"
    assert stats == {"distinct_nonmembers": 1, "false_positives": 0, "empirical_fpr": 0.0, "fp_tokens_removed": 0}


# --------------------------------------------------------------------------- #
# build_prompt
# --------------------------------------------------------------------------- #


def test_prompt_includes_every_relevant_field():
    prompt = build_prompt("CONTEXT TEXT", QUESTION)
    assert "CONTEXT TEXT" in prompt
    assert "Question: How long was the boom?" in prompt
    assert "Answer type: number. " + ask_llm.TYPE_HINTS["number"] in prompt
    assert "Unit: ft" in prompt
    assert "at most 8 tokens" in prompt
    assert prompt.endswith("Answer:")


def test_prompt_omits_unit_line_when_absent():
    question = {k: v for k, v in QUESTION.items() if k != "unit"}
    assert "Unit:" not in build_prompt("ctx", question)
    assert "Unit:" not in build_prompt("ctx", {**question, "unit": None})


def test_prompt_never_leaks_qid_or_number():
    prompt = build_prompt("ctx", QUESTION)
    assert QUESTION["qid"] not in prompt
    assert "T2" not in prompt


def test_unknown_answer_type_gets_no_hint():
    prompt = build_prompt("ctx", {**QUESTION, "answer_type": "colour"})
    assert "Answer type: colour.\n" in prompt


def test_only_context_differs_between_runs():
    full, compressed = build_prompt("full context here", QUESTION), build_prompt("short", QUESTION)
    assert full.replace("full context here", "short") == compressed


# --------------------------------------------------------------------------- #
# CachedClient
# --------------------------------------------------------------------------- #


def test_cache_replays_answers_and_flags_them(tmp_path):
    inner = FakeClient()
    client = CachedClient(inner, cache_dir=tmp_path)
    first, second = client.generate("prompt"), client.generate("prompt")
    assert inner.generate_calls == 1
    assert first.cached is False and second.cached is True
    assert (second.text, second.prompt_tokens, second.latency_s) == (first.text, first.prompt_tokens, first.latency_s)


def test_cache_key_depends_on_prompt(tmp_path):
    inner = FakeClient()
    client = CachedClient(inner, cache_dir=tmp_path)
    client.generate("a")
    client.generate("b")
    assert inner.generate_calls == 2


def test_cache_stores_token_counts(tmp_path):
    inner = FakeClient()
    client = CachedClient(inner, cache_dir=tmp_path)
    assert client.count_tokens("one two three") == client.count_tokens("one two three") == 3
    assert inner.count_calls == 1


# --------------------------------------------------------------------------- #
# retry_delay
# --------------------------------------------------------------------------- #


def test_retry_delay_uses_server_hint():
    details = {"error": {"code": 429, "details": [
        {"@type": "type.googleapis.com/google.rpc.Help"},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "12.5s"},
    ]}}
    assert ask_llm.retry_delay(details, attempt=0) == 13.5


def test_retry_delay_falls_back_to_capped_exponential_backoff():
    assert ask_llm.retry_delay({"error": {"details": []}}, attempt=3) == 8.0
    assert ask_llm.retry_delay(None, attempt=20) == ask_llm.MAX_BACKOFF_S
    bad_hint = {"error": {"details": [{"@type": "x.RetryInfo", "retryDelay": "soon"}]}}
    assert ask_llm.retry_delay(bad_hint, attempt=1) == 2.0


def test_retry_delay_caps_server_hint():
    details = {"error": {"details": [{"@type": "x.RetryInfo", "retryDelay": "3600s"}]}}
    assert ask_llm.retry_delay(details, attempt=0) == ask_llm.MAX_BACKOFF_S


# --------------------------------------------------------------------------- #
# load_env
# --------------------------------------------------------------------------- #


def test_load_env_reads_values_without_overriding(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\n\nNEW_VAR='abc'\nKEPT_VAR=from_file\n")
    monkeypatch.delenv("NEW_VAR", raising=False)
    monkeypatch.setenv("KEPT_VAR", "from_shell")
    load_env(env)
    assert ask_llm.os.environ["NEW_VAR"] == "abc"
    assert ask_llm.os.environ["KEPT_VAR"] == "from_shell"
    monkeypatch.delenv("NEW_VAR")


def test_load_env_missing_file_is_fine(tmp_path):
    load_env(tmp_path / "missing.env")


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #


def test_baseline_run_schema_and_answers(files, tmp_path):
    contexts, questions = files
    out = tmp_path / "response.json"
    client = FakeClient()
    result = run(contexts, questions, baseline=True, out=out, client=client)

    assert json.loads(out.read_text()) == result
    assert result["config"]["baseline"] is True and result["config"]["model"] == "fake-model"
    assert result["filter"] is None
    (doc,) = result["docs"]
    assert doc["doc_id"] == "doc"
    assert doc["words_in"] == doc["words_out"] and doc["retained_frac"] == 1.0
    assert [q["qid"] for q in doc["questions"]] == ["doc-T2-007", "doc-T1-001"]
    record = doc["questions"][0]
    assert record["answer"] == "42 ft"
    assert record["token_source"] == "usage"
    assert record["latency_s"] == pytest.approx(doc["compress_s"] + record["llm_s"])
    assert client.generate_calls == 2


def test_stats_only_never_calls_the_llm(files):
    contexts, questions = files
    client = FakeClient()
    result = run(contexts, questions, baseline=True, stats_only=True, out=None, client=client)
    assert client.generate_calls == 0
    for record in result["docs"][0]["questions"]:
        assert record["answer"] is None and record["llm_s"] is None
        assert record["token_source"] == "count_tokens" and record["prompt_tokens"] > 0


def test_stats_only_counts_match_real_prompt_tokens(files):
    """The fake counts words in the exact prompt either way, so both modes must agree."""
    contexts, questions = files
    real = run(contexts, questions, baseline=True, out=None, client=FakeClient())
    stats = run(contexts, questions, baseline=True, stats_only=True, out=None, client=FakeClient())
    tokens = lambda r: [q["prompt_tokens"] for q in r["docs"][0]["questions"]]
    assert tokens(real) == tokens(stats)


def test_mismatched_lists_raise(files):
    contexts, questions = files
    with pytest.raises(ValueError, match="pair by position"):
        run(contexts * 2, questions, baseline=True, out=None, client=FakeClient())


def test_compressed_run_records_filter_and_fpr(files):
    contexts, questions = files
    result = run(contexts, questions, size=4096, hash_count=6, tier="curated", calc_fpr=True,
                 stats_only=True, out=None, client=FakeClient())
    assert result["config"]["tier"] == "curated" and result["config"]["size"] == 4096
    f = result["filter"]
    assert f["n_words"] == len(build_vocabulary("curated")) and f["memory_bits"] == 4096
    assert f["theoretical_fpr"] == pytest.approx(BloomFilter.theoretical_fpr(f["n_words"], 4096, 6))
    assert f["bf_asizeof_bytes"] > 0 and f["set_asizeof_bytes"] > 0
    (doc,) = result["docs"]
    assert doc["words_out"] < doc["words_in"]  # "The", "was", "Nothing", "##" ... are dropped
    assert set(doc["fpr"]) == {"distinct_nonmembers", "false_positives", "empirical_fpr", "fp_tokens_removed"}


def test_compressed_run_without_calc_fpr_has_no_fpr_block(files):
    contexts, questions = files
    result = run(contexts, questions, size=4096, hash_count=6, out=None, client=FakeClient())
    assert "fpr" not in result["docs"][0]
    assert "bf_asizeof_bytes" not in result["filter"]


def test_compressed_run_requires_filter_params(files):
    contexts, questions = files
    with pytest.raises(ValueError, match="size and hash_count"):
        run(contexts, questions, out=None, client=FakeClient())


def test_records_carry_max_answer_tokens(files):
    contexts, questions = files
    for stats_only in (False, True):
        result = run(contexts, questions, baseline=True, stats_only=stats_only, out=None, client=FakeClient())
        assert [q["max_answer_tokens"] for q in result["docs"][0]["questions"]] == [8, 8]


def test_run_never_reads_answer_files():
    source = Path(ask_llm.__file__).read_text(encoding="utf-8")
    assert "answers/" not in source and "_answers" not in source


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def test_cli_requires_filter_params_unless_baseline():
    with pytest.raises(SystemExit):
        parse_args(["--contexts", "c.txt", "--questions", "q.json"])
    args = parse_args(["--contexts", "c.txt", "--questions", "q.json", "--baseline"])
    assert args.baseline and args.out == "response.json"


def test_cli_parses_lists_and_flags():
    args = parse_args(["--contexts", "a.txt", "b.txt", "--questions", "a.json", "b.json",
                       "--size", "4096", "--hash-count", "6", "--tier", "common500",
                       "--stats-only", "--calc-fpr"])
    assert args.contexts == ["a.txt", "b.txt"] and args.questions == ["a.json", "b.json"]
    assert (args.size, args.hash_count, args.tier) == (4096, 6, "common500")
    assert args.stats_only and args.calc_fpr and not args.baseline

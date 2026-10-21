"""Tests for driver.py and llm.py: end-to-end records with FakeLLM, resume and the disk cache."""

import json

from driver import RunConfig, load_document, prepare, run_document
from llm import CachedLLM, FakeLLM, build_prompt


def test_end_to_end_records_are_well_formed(tmp_path, data_dir):
    doc = load_document("harbour_puzzle_v1", data_dir)
    records = run_document(prepare(doc, RunConfig("core")), FakeLLM(), tmp_path / "records.jsonl", workers=2)
    assert len(records) == 20
    rows = [json.loads(line) for line in (tmp_path / "records.jsonl").read_text().splitlines()]
    assert {r["qid"] for r in rows} == {q.qid for q in doc.questions}
    for r in rows:
        assert r["selection_status"] in ("ok", "no_match")
        assert 0 < r["retained_frac"] < 1
        assert r["prompt_tokens"] > 0 and r["build_s"] > 0
        assert r["total_s"] >= r["retrieval_s"]


def test_full_mode_sends_the_whole_document(tmp_path, data_dir):
    doc = load_document("lunarleaper_2609_11453", data_dir)
    records = run_document(prepare(doc, RunConfig("full", mode="full")), FakeLLM(), tmp_path / "r.jsonl")
    assert all(r.retained_frac == 1.0 and r.selection_status == "full" for r in records)


def test_rerun_resumes_without_new_llm_calls(tmp_path, data_dir):
    doc = load_document("dwh_dossier_v1", data_dir)
    prepared = prepare(doc, RunConfig("core"))
    first, second = FakeLLM(), FakeLLM()
    run_document(prepared, first, tmp_path / "r.jsonl")
    assert run_document(prepared, second, tmp_path / "r.jsonl") == []
    assert first.calls == 20 and second.calls == 0


def test_cache_returns_identical_answers_and_marks_them_cached(tmp_path):
    fake = FakeLLM()
    llm = CachedLLM(fake, tmp_path)
    prompt = build_prompt("Berth 7 holds the cargo.", "Which berth?", "number", None)
    first, second = llm.generate(prompt), llm.generate(prompt)
    assert fake.calls == 1
    assert second.cached and not first.cached and second.text == first.text


def test_disabled_cache_always_calls_the_model(tmp_path):
    fake = FakeLLM()
    llm = CachedLLM(fake, tmp_path, enabled=False)
    llm.generate("p")
    llm.generate("p")
    assert fake.calls == 2

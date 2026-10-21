"""Unit tests for experiments.py. No network: ask_llm.run is given a fake client or monkeypatched."""

import csv
import json
from pathlib import Path

import pytest

import ask_llm
import experiments
from experiments import (Dataset, Gold, grade, grade_run, load_dataset, load_gold, normalize_answer, parse_args,
                         parse_number)


ROOT = Path(__file__).resolve().parent.parent


def gold(expected, aliases=(), normaliser="numeric", tolerance=0.005, qid="d-T1-001", tier="T1_lookup"):
    return Gold(qid=qid, doc_id="d", expected=expected, aliases=tuple(aliases),
                normaliser=normaliser, tolerance=tolerance, tier=tier)


# --------------------------------------------------------------------------- #
# parse_number
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("text, value", [
    ("13,300,000 ft", 13_300_000),
    ("3.14", 3.14),
    ("-40 degrees", -40),
    ("25%", 25),
    ("13.3 million years", 13.3e6),
    ("$250m", 250e6),
    ("$45 million", 45e6),
    ("20 m", 20),  # metres, not million: no currency sign
    ("sixty-three", 63),
    ("one hundred and twenty", 120),
    ("Over 400 sorties", 400),
    ("1,400-psi", 1400),
    ("2.5e3", 2500),
    ("−7", -7),  # Unicode minus
])
def test_parse_number(text, value):
    assert parse_number(text) == pytest.approx(value)


def test_parse_number_takes_the_first_number():
    assert parse_number("between 150 and 1200 MHz") == 150
    assert parse_number("five and six") == 5


def test_parse_number_none_without_numbers():
    assert parse_number("Antioch") is None
    assert parse_number("") is None


# --------------------------------------------------------------------------- #
# normalize_answer / grade
# --------------------------------------------------------------------------- #


def test_normalize_answer():
    assert normalize_answer("  The   British Museum. ") == "british museum"
    assert normalize_answer("“Ecolab Inc.”") == "ecolab inc"
    assert normalize_answer("1–13 meters") == "1-13 meters"
    assert normalize_answer("26,764,083") == "26764083"
    assert normalize_answer("352.3±1.5") == "352.3±1.5"


def test_numeric_tolerance_edges():
    g = gold("1000 ft", tolerance=0.005)
    assert grade("1005", g)
    assert grade("995 ft", g)
    assert not grade("1006", g)
    assert not grade("no idea", g)


def test_numeric_matches_any_alias():
    g = gold("$250 million", aliases=["250 million", "$250m"])
    assert grade("$250 million", g)
    assert grade("250 million dollars", g)
    assert not grade("250", g)  # no bare-number alias: the scale word matters


def test_numeric_with_bare_alias_ignores_units():
    g = gold("13,300,000 ft", aliases=["13.3 million ft", "4,100 km"])
    assert grade("13.3 million feet", g)
    assert grade("4100 km", g)


def test_exact_ci_matches_expected_and_aliases():
    g = gold("Jupiter Insurance Ltd", aliases=["Jupiter Insurance"], normaliser="exact_ci")
    assert grade("jupiter insurance ltd.", g)
    assert grade("Jupiter  Insurance", g)
    assert not grade("Jupiter", g)
    assert not grade("", g)
    assert not grade(None, g)


def test_exact_ci_drops_leading_article():
    g = gold("the British Museum", normaliser="exact_ci")
    assert grade("British Museum", g)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


def _write_doc(root: Path, doc_id: str, answers: list[dict]) -> None:
    for sub in ("contexts", "questions", "answers"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "contexts" / f"{doc_id}.txt").write_text("The answer is 42 ft.\n", encoding="utf-8")
    questions = [{"qid": a["qid"], "question": "How long?", "answer_type": "number",
                  "unit": "ft", "max_answer_tokens": 8} for a in answers]
    (root / "questions" / f"{doc_id}_questions.json").write_text(
        json.dumps({"doc_id": doc_id, "count": len(questions), "questions": questions}))
    (root / "answers" / f"{doc_id}_answers.json").write_text(json.dumps({"doc_id": doc_id, "answers": answers}))


ANSWER = {"qid": "b-T1-001", "expected": "42 ft", "aliases": ["42"], "tier": "T1_lookup",
          "grading": {"normaliser": "numeric", "numeric_tolerance_rel": 0.005}}


def _paths(root: Path, *doc_ids: str) -> Dataset:
    """The paired file paths that ``_write_doc`` created, in the given order."""
    return Dataset([str(root / "contexts" / f"{d}.txt") for d in doc_ids],
                   [str(root / "questions" / f"{d}_questions.json") for d in doc_ids],
                   [str(root / "answers" / f"{d}_answers.json") for d in doc_ids])


def test_load_dataset_accepts_matching_files(tmp_path):
    _write_doc(tmp_path, "a_doc", [{**ANSWER, "qid": "a-T1-001"}])
    _write_doc(tmp_path, "b_doc", [ANSWER])
    paths = _paths(tmp_path, "b_doc", "a_doc")  # any order works as long as the lists agree
    assert load_dataset(paths.contexts, paths.questions, paths.answers) == paths


def test_load_dataset_rejects_swapped_answers(tmp_path):
    _write_doc(tmp_path, "a_doc", [{**ANSWER, "qid": "a-T1-001"}])
    _write_doc(tmp_path, "b_doc", [ANSWER])
    paths = _paths(tmp_path, "a_doc", "b_doc")
    with pytest.raises(ValueError, match="same order"):
        load_dataset(paths.contexts, paths.questions, paths.answers[::-1])


def test_load_dataset_rejects_answers_missing_a_question(tmp_path):
    _write_doc(tmp_path, "doc", [ANSWER])
    answers = tmp_path / "answers" / "doc_answers.json"
    answers.write_text(json.dumps({"doc_id": "doc", "answers": []}))
    paths = _paths(tmp_path, "doc")
    with pytest.raises(ValueError, match="no answers for"):
        load_dataset(paths.contexts, paths.questions, paths.answers)


def test_load_dataset_rejects_missing_files_and_unequal_lists(tmp_path):
    _write_doc(tmp_path, "doc", [ANSWER])
    paths = _paths(tmp_path, "doc")
    with pytest.raises(FileNotFoundError, match="nope.txt"):
        load_dataset([str(tmp_path / "nope.txt")], paths.questions, paths.answers)
    with pytest.raises(ValueError, match="counts must be equal"):
        load_dataset(paths.contexts * 2, paths.questions, paths.answers)


def test_load_dataset_on_the_assignment_data():
    docs = ["antikythera_dossier_v1", "dwh_dossier_v1", "harbour_puzzle_v1", "lunarleaper_2609_11453"]
    paths = _paths(ROOT / "assignment", *docs)
    data = load_dataset(paths.contexts, paths.questions, paths.answers)
    assert sum(len(ask_llm.load_questions(q)[1]) for q in data.questions) == 80


def test_load_gold(tmp_path):
    _write_doc(tmp_path, "doc", [ANSWER])
    loaded = load_gold([str(tmp_path / "answers" / "doc_answers.json")])
    assert loaded["b-T1-001"] == Gold("b-T1-001", "doc", "42 ft", ("42",), "numeric", 0.005, "T1_lookup")


def test_load_gold_missing_tolerance_defaults_to_zero(tmp_path):
    _write_doc(tmp_path, "doc", [{**ANSWER, "grading": {"normaliser": "exact_ci"}}])
    assert load_gold([str(tmp_path / "answers" / "doc_answers.json")])["b-T1-001"].tolerance == 0.0


# --------------------------------------------------------------------------- #
# grade_run / save_runs / run_llm_experiments
# --------------------------------------------------------------------------- #


def _result(answers: dict[str, str], context_tokens: int = 90, fpr: dict | None = None,
            filter_stats: dict | None = None) -> dict:
    """A minimal ``ask_llm.run`` result with one document."""
    doc = {"doc_id": "d", "words_in": 100, "words_out": 60, "compress_s": 0.01, "context_tokens": context_tokens,
           "questions": [{"qid": qid, "answer": a, "prompt_tokens": 100, "latency_s": 1.0, "cached": False}
                         for qid, a in answers.items()]}
    if fpr is not None:
        doc["fpr"] = fpr
    return {"config": {"baseline": filter_stats is None, "tier": None, "size": None, "hash_count": None},
            "filter": filter_stats, "docs": [doc]}


def test_grade_run_marks_records_and_summarizes():
    golds = {"d-T1-001": gold("42", qid="d-T1-001"),
             "d-T2-002": gold("Paris", normaliser="exact_ci", qid="d-T2-002", tier="T2_multihop")}
    result = _result({"d-T1-001": "42 ft", "d-T2-002": "London"})
    summary = grade_run(result, golds)
    assert [r["correct"] for r in result["docs"][0]["questions"]] == [True, False]
    assert summary["accuracy"] == 0.5
    assert summary["accuracy_T1_lookup"] == 1.0 and summary["accuracy_T2_multihop"] == 0.0
    assert summary["accuracy_d"] == 0.5
    assert summary["total_prompt_tokens"] == 200 and summary["mean_latency_s"] == 1.0


def test_summarize_reports_filter_and_fpr_columns():
    golds = {"d-T1-001": gold("42", qid="d-T1-001")}
    fpr = {"distinct_nonmembers": 200, "false_positives": 10, "empirical_fpr": 0.05, "fp_tokens_removed": 30}
    filt = {"n_words": 120, "memory_bits": 2048, "theoretical_fpr": 0.04,
            "bf_asizeof_bytes": 688, "set_asizeof_bytes": 14224}
    summary = grade_run(_result({"d-T1-001": "42"}, fpr=fpr, filter_stats=filt), golds)
    assert summary["empirical_fpr"] == 0.05 and summary["fp_tokens_removed"] == 30
    assert (summary["n_words"], summary["memory_bits"], summary["theoretical_fpr"]) == (120, 2048, 0.04)
    assert summary["retained_frac"] == 0.6 and summary["context_tokens"] == 90


def test_summarize_baseline_has_no_filter_columns():
    golds = {"d-T1-001": gold("42", qid="d-T1-001")}
    summary = grade_run(_result({"d-T1-001": "42"}), golds)
    assert summary["memory_bits"] is None and summary["empirical_fpr"] is None


def test_add_token_retention_is_relative_to_baseline():
    runs = {"baseline": {"summary": {"context_tokens": 200}}, "tier_x": {"summary": {"context_tokens": 50}}}
    experiments.add_token_retention(runs)
    assert runs["baseline"]["summary"]["token_retained_frac"] == 1.0
    assert runs["tier_x"]["summary"]["token_retained_frac"] == 0.25


def test_plot_llm_writes_pngs(tmp_path):
    golds = {"d-T1-001": gold("42", qid="d-T1-001")}
    runs = {}
    specs = {"baseline": (None, 100), "tier_curated": (0.0003, 70), "tier_nltk_full": (0.005, 68),
             "tier_common1000": (0.0002, 60), "fpr50_curated": (0.5, 40), "fpr5_common1000": (0.05, 55)}
    for name, (fpr, tokens) in specs.items():
        filt = None if fpr is None else {"n_words": 120, "memory_bits": 2048, "theoretical_fpr": fpr}
        block = None if fpr is None else {"distinct_nonmembers": 100, "false_positives": round(fpr * 100),
                                          "empirical_fpr": fpr, "fp_tokens_removed": 1}
        run = _result({"d-T1-001": "42"}, context_tokens=tokens, fpr=block, filter_stats=filt)
        run["summary"] = grade_run(run, golds)
        runs[name] = run
    experiments.add_token_retention(runs)
    paths = experiments.plot_llm(runs, tmp_path)
    assert [p.name for p in paths] == ["tokens_vs_accuracy.png", "fpr_vs_accuracy.png",
                                       "accuracy_by_question_tier.png"]
    assert all(p.stat().st_size > 0 for p in paths)


def test_count_over_limit():
    records = [
        {"output_tokens": 9, "max_answer_tokens": 8},   # over
        {"output_tokens": 8, "max_answer_tokens": 8},   # at the limit
        {"output_tokens": None, "max_answer_tokens": 8},  # stats-only: no answer
        {"output_tokens": 50},                            # no limit given
    ]
    assert experiments.count_over_limit(records) == 1


def test_run_llm_experiments_writes_graded_baseline(tmp_path, monkeypatch):
    _write_doc(tmp_path / "data", "doc", [ANSWER])
    data = _paths(tmp_path / "data", "doc")
    calls = []

    def fake_run(contexts, questions, **kwargs):
        calls.append(kwargs)
        return _result({"b-T1-001": "42 feet"})

    monkeypatch.setattr(experiments.ask_llm, "run", fake_run)
    monkeypatch.setattr(experiments, "llm_configs", lambda: {"baseline": {"baseline": True}})
    runs = experiments.run_llm_experiments(data, tmp_path / "out")

    assert calls == [{"out": None, "calc_fpr": False, "baseline": True}]
    assert runs["baseline"]["summary"]["accuracy"] == 1.0
    saved = json.loads((tmp_path / "out" / "llm_runs.json").read_text())
    assert saved["baseline"]["docs"][0]["questions"][0]["correct"] is True
    with open(tmp_path / "out" / "summary.csv") as f:
        (row,) = list(csv.DictReader(f))
    assert row["run"] == "baseline" and float(row["accuracy"]) == 1.0


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def test_cli_requires_all_data_flags():
    with pytest.raises(SystemExit):
        parse_args([])
    with pytest.raises(SystemExit):
        parse_args(["--contexts", "a.txt", "--questions", "a.json"])


def test_cli_requires_equal_lengths():
    with pytest.raises(SystemExit):
        parse_args(["--contexts", "a.txt", "--questions", "a.json", "b.json", "--answers", "a.json"])


def test_cli_parses_paired_files():
    args = parse_args(["--mode", "stats_only", "--contexts", "a.txt", "b.txt", "--questions", "a.json", "b.json",
                       "--answers", "x.json", "y.json"])
    assert args.mode == "stats_only" and args.answers == ["x.json", "y.json"]


# --------------------------------------------------------------------------- #
# Stats-only grid and LLM configuration choice
# --------------------------------------------------------------------------- #


class CountingClient:
    """Fake client for the grid: one token per whitespace word, and no generation."""

    model = "fake-model"

    def __init__(self):
        self.count_calls = 0

    def generation_config(self) -> dict:
        return {}

    def count_tokens(self, text: str) -> int:
        self.count_calls += 1
        return len(text.split())

    def generate(self, prompt):
        raise AssertionError("the stats-only grid must never generate answers")


def test_grid_hash_counts_adds_capped_k_opt():
    assert experiments.grid_hash_counts(100, 1000) == [1, 2, 3, 4, 6, 7, 8]  # k_opt = 7
    assert max(experiments.grid_hash_counts(10, 1 << 17)) == experiments.MAX_HASH_COUNT


def test_smallest_size_for_fpr_is_the_smallest_power_of_two():
    n = 120
    size = experiments.smallest_size_for_fpr(n, 0.01)
    fpr = lambda r: ask_llm.BloomFilter.theoretical_fpr(n, r, ask_llm.BloomFilter.optimal_k(n, r))
    assert size == 2048 and fpr(size) <= 0.01 < fpr(size // 2)


@pytest.mark.parametrize("target", [0.5, 0.2, 0.05, 0.01])
def test_size_for_fpr_hits_the_target(target):
    n = 930
    size = experiments.size_for_fpr(n, target)
    k = ask_llm.BloomFilter.optimal_k(n, size)
    assert ask_llm.BloomFilter.theoretical_fpr(n, size, k) == pytest.approx(target, rel=0.1)


def test_llm_configs_cover_baseline_tiers_and_fpr_sweep():
    configs = experiments.llm_configs()
    assert configs["baseline"] == {"baseline": True}
    assert {c["tier"] for name, c in configs.items() if name.startswith("tier_")} == set(ask_llm.TIERS)
    sweep = [c for name, c in configs.items() if name.startswith("fpr")]
    assert len(sweep) == len(experiments.FPR_SWEEP_TIERS) * len(experiments.FPR_SWEEP_TARGETS)
    for tier in experiments.FPR_SWEEP_TIERS:
        sizes = [c["size"] for name, c in configs.items() if name.startswith("fpr") and c["tier"] == tier]
        assert sizes == sorted(sizes)  # targets run from high to low FPR, so R grows


def test_prompt_tokens_counts_context_once_per_question():
    assert experiments._prompt_tokens([100, 10], [2, 3], question_overhead=7) == 100 * 2 + 10 * 3 + 7


@pytest.fixture
def small_grid(tmp_path, monkeypatch):
    """The grid over one tiny document, with fewer sizes so it runs fast."""
    _write_doc(tmp_path / "data", "doc", [ANSWER])
    (tmp_path / "data" / "contexts" / "doc.txt").write_text(
        "## The history of the harbour\nThe boom was 42 ft long and it was in the water.\n", encoding="utf-8")
    monkeypatch.setattr(experiments, "GRID_SIZES", (256, 2048))
    client = CountingClient()
    grid = experiments.run_stats_grid(_paths(tmp_path / "data", "doc"), client, tmp_path / "out")
    return grid, client, tmp_path / "out"


def test_stats_grid_rows(small_grid):
    grid, client, out = small_grid
    rows = grid["rows"]
    expected = sum(len(experiments.grid_hash_counts(len(ask_llm.build_vocabulary(t)), r))
                   for t in ask_llm.TIERS for r in (256, 2048))
    assert len(rows) == expected
    assert all(sum(r["is_k_opt"] for r in rows if (r["tier"], r["size"]) == (t, s)) == 1
               for t in ask_llm.TIERS for s in (256, 2048))
    for row in rows:
        assert 0 <= row["empirical_fpr"] <= 1 and 0 <= row["retained_frac"] <= 1  # an all-ones filter drops everything
        assert row["token_retained_frac"] == pytest.approx(row["context_tokens"] / grid["baseline"]["context_tokens"])
    assert (out / "stats_grid.json").exists() and (out / "stats_grid.csv").exists()


def test_stats_grid_drops_more_with_bigger_tiers(small_grid):
    grid, _, _ = small_grid
    retained = {r["tier"]: r["retained_frac"] for r in grid["rows"] if r["size"] == 2048 and r["is_k_opt"]}
    assert retained["common5000"] <= retained["common500"] <= retained["curated"] < 1


def test_tiny_filter_has_high_fpr(small_grid):
    grid, _, _ = small_grid
    row = experiments._row_for(grid["rows"], "common5000", 256, 1)
    assert row["theoretical_fpr"] > 0.9 and row["empirical_fpr"] > 0.5


def test_row_for_missing_config_raises(small_grid):
    grid, _, _ = small_grid
    with pytest.raises(KeyError, match="No grid row"):
        experiments._row_for(grid["rows"], "curated", 999, 1)


def test_plot_grid_writes_pngs(small_grid, monkeypatch):
    grid, _, out = small_grid
    # The retained-by-tier plot looks up each tier's LLM configuration, so point it at grid rows.
    monkeypatch.setattr(experiments, "llm_configs", lambda: {
        f"tier_{t}": {"tier": t, "size": 2048,
                      "hash_count": next(r["hash_count"] for r in grid["rows"]
                                         if (r["tier"], r["size"]) == (t, 2048) and r["is_k_opt"])}
        for t in ask_llm.TIERS})
    monkeypatch.setattr(experiments, "GRID_SIZES", (256, 2048))
    grid["rows"] += [{**r, "size": s} for r in grid["rows"] if r["tier"] == "curated" and r["size"] == 256
                     for s in (512, 1024)]  # the FPR-vs-k plot draws R = 256, 512 and 1024
    paths = experiments.plot_grid(grid, out)
    assert [p.name for p in paths] == ["memory_vs_fpr.png", "memory_vs_tokens.png", "fpr_measured_vs_theory.png",
                                       "fpr_vs_k.png", "retained_by_tier.png"]
    assert all(p.stat().st_size > 0 for p in paths)

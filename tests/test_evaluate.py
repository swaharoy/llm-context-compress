"""Tests for evaluate.py: grading rules against the real answer schema, gold-span recall and the CI."""

import pytest

from evaluate import Gold, bootstrap_ci, gold_span_recall, grade, load_gold, normalize_answer, parse_number


def numeric_gold(expected: str, aliases=(), tolerance=0.005) -> Gold:
    return Gold("q", "doc", expected, tuple(aliases), "numeric", tolerance, "T1_lookup", "verbatim", ())


def exact_gold(expected: str, aliases=()) -> Gold:
    return Gold("q", "doc", expected, tuple(aliases), "exact_ci", 0.0, "T1_lookup", "verbatim", ())


def test_every_gold_entry_grades_correct_against_itself_and_its_aliases(data_dir):
    gold = load_gold(data_dir)
    assert len(gold) == 80
    for g in gold.values():
        for answer in (g.expected, *g.aliases):
            assert grade(answer, g).correct_strict, (g.qid, answer)


def test_gold_entries_have_quotes_for_span_recall(data_dir):
    assert all(g.gold_quotes for g in load_gold(data_dir).values())


def test_numeric_tolerance_edges():
    g = numeric_gold("100")
    assert grade("100.5", g).correct_strict
    assert grade("99.5", g).correct_strict
    assert not grade("100.6", g).correct_strict


def test_numeric_accepts_any_alias_value():
    g = numeric_gold("13,300,000 ft", ["4,100 km"])
    assert grade("about 4100 km", g).correct_strict
    assert grade("13.3 million feet", g).correct_strict


@pytest.mark.parametrize("text,value", [
    ("$250m", 250e6), ("$250 million", 250e6), ("20 m", 20), ("one hundred and twenty", 120),
    ("sixty-three", 63), ("1,400-psi", 1400), ("-5 degrees", -5), ("25%", 25), ("3.2e3", 3200),
    ("over 400 sorties", 400),
])
def test_number_parsing(text, value):
    assert parse_number(text).value == pytest.approx(value)


def test_parse_status_flags():
    g = numeric_gold("7")
    assert grade("I don't know", g).parse_status == "no_number"
    assert not grade("I don't know", g).correct_strict
    multiple = grade("7 or 8", g)
    assert multiple.parse_status == "multiple_numbers" and multiple.correct_strict  # graded on the first


def test_exact_match_strips_articles_and_outer_punctuation_only():
    g = exact_gold("British Museum")
    assert grade("The British Museum.", g).correct_strict
    assert normalize_answer("352.3±1.5") == "352.3±1.5"
    assert not grade("British", g).correct_strict


def test_lenient_grading_finds_answer_inside_a_sentence():
    result = grade("It was the Malitis ailment.", exact_gold("Malitis"))
    assert not result.correct_strict and result.correct_lenient


def test_gold_span_recall():
    g = Gold("q", "doc", "31.7", (), "numeric", 0.005, "T1", "verbatim",
             ("Fragment G | 125 × 110 | 31.7", "Fragment C | 120 × 110 | 63.8"))
    assert gold_span_recall("[§ Design]\nFragment G | 125 × 110 | 31.7 | Yes", g) == 0.5
    assert gold_span_recall("nothing relevant", g) == 0.0


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean():
    values = [1, 0, 1, 1, 0, 1, 0, 1, 1, 1]
    low, high = bootstrap_ci(values)
    assert (low, high) == bootstrap_ci(values)
    assert low <= sum(values) / len(values) <= high

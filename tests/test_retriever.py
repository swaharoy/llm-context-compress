"""Tests for retriever.py: ranking, the hard budget, edge cases, neighbours, and index-only scoring."""

import dataclasses

import numpy as np
import pytest

from bloom import BloomParams, ExactSetIndex, RowMajorBloomIndex
from retriever import BloomRetriever, FullContextSelector, RetrievalParams, estimate_tokens
from text import TermParams, chunk_document, chunk_terms

DOC = """\
== Harbour ==

The harbour has twenty berths arranged from west to east along the reach.

== Cargo ==

The Skoukweed cargo stands at berth 7 beside the old crane on the quay.

The Letallow cargo stands at berth 12 next to the weighbridge office.

== Weather ==

The tide was high in the morning and the weather was calm all day long.
"""


def make_retriever(doc: str = DOC, params: RetrievalParams = RetrievalParams(), index_cls=RowMajorBloomIndex):
    chunks = chunk_document(doc)
    index = index_cls(chunk_terms(chunks, TermParams()), BloomParams(R=1024))
    return BloomRetriever(chunks, index, len(doc), TermParams(), params), chunks


def test_gold_chunk_ranks_first():
    retriever, chunks = make_retriever()
    scores = retriever.score("At which berth does the Skoukweed cargo stand?").values
    assert "Skoukweed" in chunks[int(np.argmax(scores))].text


def test_selection_includes_gold_chunk_and_breadcrumb():
    retriever, _ = make_retriever(params=RetrievalParams(neighbours=0))
    selection = retriever.select("Where is the Skoukweed cargo?")
    assert selection.status == "ok"
    assert "Skoukweed cargo stands at berth 7" in selection.text
    assert "[§ Cargo]" in selection.text
    assert "weather" not in selection.text.lower()


@pytest.mark.parametrize("budget", [20, 60, 200, 1500])
def test_assembled_context_never_exceeds_budget(data_dir, budget):
    doc = (data_dir / "contexts" / "antikythera_dossier_v1.txt").read_text()
    retriever, _ = make_retriever(doc, RetrievalParams(budget_tokens=budget))
    for question in ("What weight in grams is recorded for Fragment G?", "How many teeth does gear b1 have?",
                     "Who led the 2005 Cardiff University team?"):
        selection = retriever.select(question)
        assert estimate_tokens(selection.text) <= budget
        assert selection.est_tokens <= budget


def test_no_usable_terms_gives_no_match():
    retriever, _ = make_retriever()
    for question in ("what is the", "Zyxwv qwerty?"):
        selection = retriever.select(question)
        assert selection.status == "no_match"
        assert selection.text == "" and selection.chunk_ids == ()


def test_ties_are_broken_by_document_order():
    # The third section holds neither term, so both terms get a non-zero idf weight.
    doc = ("== A ==\n\nThe Mitranbark cargo was logged today.\n\n== B ==\n\nThe Mitranbark cargo was logged again.\n\n"
           "== C ==\n\nThe tide was high and calm.\n")
    retriever, chunks = make_retriever(doc, RetrievalParams(neighbours=0, budget_tokens=12))
    selection = retriever.select("Mitranbark cargo")
    assert selection.chunk_ids == (0,)  # both score the same; only the first fits the budget


def test_neighbours_stay_inside_their_section():
    retriever, chunks = make_retriever(params=RetrievalParams(neighbours=1))
    selection = retriever.select("Where is the Skoukweed cargo?")
    sections = {chunks[c].section_id for c in selection.chunk_ids}
    gold_section = next(c.section_id for c in chunks if "Skoukweed" in c.text)
    assert sections == {gold_section}
    assert any("Letallow" in chunks[c].text for c in selection.chunk_ids)  # its in-section neighbour


def test_scoring_uses_only_the_index_not_chunk_text():
    """Replacing every chunk's text after indexing must not change scores: retrieval reads only the index."""
    retriever, chunks = make_retriever()
    question = "Which cargo stands at berth 12?"
    before = retriever.score(question).values
    retriever.chunks = [dataclasses.replace(c, text="unrelated") for c in chunks]
    assert np.array_equal(before, retriever.score(question).values)


def test_retriever_sees_bloom_false_positives():
    """With a tiny filter the Bloom index claims more chunks than the exact control does."""
    chunks = chunk_document(DOC)
    terms = chunk_terms(chunks, TermParams())
    tiny = BloomRetriever(chunks, RowMajorBloomIndex(terms, BloomParams(R=8, k=1)), len(DOC))
    exact = BloomRetriever(chunks, ExactSetIndex(terms), len(DOC))
    question = "Where is the Skoukweed cargo?"
    assert (tiny.score(question).values > 0).sum() >= (exact.score(question).values > 0).sum()


def test_full_context_selector_returns_document_unchanged():
    selection = FullContextSelector(DOC).select("anything")
    assert selection.text == DOC
    assert selection.status == "full" and selection.retained_frac == 1.0

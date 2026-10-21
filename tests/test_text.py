"""Tests for text.py: chunking on small synthetic snippets of each layout, and term extraction."""

from text import (ChunkParams, TermParams, canonical_number, char_ngrams, chunk_document, chunk_terms,
                  extract_terms, normalize, stem)

WIKI = """\
== History ==

The mechanism was found in 1901. It was studied by Dr. Price for many years afterwards.

=== Discovery ===

Divers recovered many bronze fragments from the wreck site near the island of Antikythera.
"""

TABLE = """\
== Design ==

Fragment | Size [mm] | Weight [g]

Fragment A | 180 × 150 | 369.1

Fragment G | 125 × 110 | 31.7

The fragments are kept in the National Archaeological Museum in Athens, Greece.
"""

LEDGER = """\
SECTION 1 -- The western berths

   1.  The berth number of the Shamymor keeper is greater than the berth
        number of the Klaegryloc keeper. The clerk saw it.
   2.  The berth number of the Jirmekresin cargo is odd and was read out.
"""


def test_wiki_headings_build_nested_breadcrumbs():
    chunks = chunk_document(WIKI)
    assert [c.breadcrumb for c in chunks] == ["History", "History > Discovery"]


def test_sentences_do_not_split_after_abbreviations():
    chunks = chunk_document(WIKI)
    assert "Dr. Price" in chunks[0].text


def test_table_rows_are_chunks_linked_to_their_header():
    chunks = chunk_document(TABLE)
    rows = [c for c in chunks if c.kind == "row"]
    assert len(rows) == 3
    header = rows[0]
    assert header.header_ref is None
    assert all(r.header_ref == header.id for r in rows[1:])
    assert chunks[-1].kind == "para"


def test_row_terms_include_header_terms_without_cross_join_bigrams():
    chunks = chunk_document(TABLE)
    row_g = next(c for c in chunks if "Fragment G" in c.text)
    terms = chunk_terms(chunks, TermParams())[row_g.id]
    assert {"w:weight", "n:31.7", "b:fragment g"} <= terms
    assert "b:weight fragment" not in terms  # would only come from concatenating header and row


def test_numbered_entries_keep_their_continuation_lines():
    chunks = chunk_document(LEDGER)
    entries = [c for c in chunks if c.kind == "entry"]
    assert len(entries) == 2
    assert "Klaegryloc keeper" in entries[0].text and "clerk saw it" in entries[0].text
    assert entries[0].breadcrumb == "SECTION 1 -- The western berths"


def test_markdown_numbered_headings_nest_by_section_number():
    doc = "## 2 Lunar Pits\n\nPits are common on the Moon and reveal layered walls of basalt.\n\n" \
          "## 2.1 Pit Wall Stratigraphy\n\nThe walls show many distinct layers of old lava flows.\n"
    chunks = chunk_document(doc)
    assert chunks[1].breadcrumb == "2 Lunar Pits > 2.1 Pit Wall Stratigraphy"


def test_short_fragment_merges_into_following_paragraph():
    doc = "== Roll ==\n\nKeepers on the roll:\n\nToushiathan Gleigrun Sagoungrun Hillemwyn Flaigrokell Truidriorvas Dormuiric\n"
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("Keepers on the roll:")


def test_long_paragraphs_are_packed_under_the_token_limit():
    sentence = "The keeper walked along the quay and counted every berth twice. "
    chunks = chunk_document(sentence * 30, ChunkParams(max_chunk_tokens=40))
    assert len(chunks) > 1
    assert all(c.n_tokens <= 40 for c in chunks)


def test_fixed_chunker_uses_plain_word_windows():
    doc = " ".join(f"w{i}" for i in range(200))
    chunks = chunk_document(doc, ChunkParams(strategy="fixed", fixed_window=80))
    assert [c.n_tokens for c in chunks] == [80, 80, 40]
    assert {c.breadcrumb for c in chunks} == {""}


def test_chunk_ids_are_document_order_indexes():
    chunks = chunk_document(TABLE + LEDGER)
    assert [c.id for c in chunks] == list(range(len(chunks)))
    assert [c.char_span[0] for c in chunks] == sorted(c.char_span[0] for c in chunks)


def test_normalisation_unifies_numbers_quotes_and_dashes():
    assert normalize("27,758 “Berths” – b1, b2") == '27758 "berths" - b1, b2'


def test_canonical_numbers():
    assert canonical_number("031.70") == "31.7"
    assert canonical_number("100") == "100"
    assert canonical_number("0.250") == "0.25"


def test_terms_drop_stopwords_and_stopwords_break_bigrams():
    terms = extract_terms("the weight of Fragment G in grams")
    assert terms.words == ("weight", "fragment", "g", "gram")
    assert terms.bigrams == ("fragment g",)


def test_numbers_are_their_own_term_type_and_join_bigrams():
    terms = extract_terms("berth 16 holds 1,204 barrels")
    assert terms.numbers == ("16", "1204")
    assert "berth 16" in terms.bigrams
    assert "n:1204" in terms.all_terms()


def test_disabled_term_types_are_not_produced():
    terms = extract_terms("berth 16 keeper", TermParams(bigrams=False, numbers=False))
    assert terms.bigrams == () and terms.numbers == ()
    assert terms.all_terms() == {"w:berth", "w:keeper"}


def test_char_ngrams_have_boundary_marks():
    assert char_ngrams("east", 4) == ("#eas", "east", "ast#")
    assert char_ngrams("g", 4) == ("#g#",)
    terms = extract_terms("eastward", TermParams(char_grams=True))
    assert "c:#eas" in terms.all_terms()


def test_stemming_is_light():
    assert [stem(w) for w in ("berths", "cities", "boxes", "glass", "status")] == ["berth", "city", "box", "glass", "status"]

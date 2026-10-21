"""Turn raw documents into retrieval chunks and Bloom-filter terms (specs.md §3, §4).

This module does two jobs, neither of which involves Bloom filters or retrieval:

* :func:`chunk_document` cuts a document into :class:`Chunk` objects: table rows,
  numbered entries, key-value lines and packed paragraph sentences. Each chunk is
  tagged with the heading path (its "breadcrumb") that it sits under.
* :func:`extract_terms` turns any text into prefixed terms that are inserted into,
  or looked up in, a Bloom filter:
  ``w:`` words, ``b:`` bigrams, ``n:`` numbers and ``c:`` character 4-grams.

Chunks and questions go through the same normalisation. A question term can
therefore match a chunk term only if both were produced here.

Depends on: the standard library only.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Parameters and data types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChunkParams:
    """How to cut a document into chunks.

    Attributes:
        strategy: ``"structure"`` (headings, tables, entries, sentences) or
            ``"fixed"`` (plain word windows, the ablation baseline).
        max_chunk_tokens: Upper bound, in words, when packing paragraph sentences.
        min_chunk_tokens: Paragraph or key-value chunks shorter than this are merged
            into a neighbour.
        fixed_window: Window size in words for the ``"fixed"`` strategy.
    """

    strategy: str = "structure"
    max_chunk_tokens: int = 80
    min_chunk_tokens: int = 8
    fixed_window: int = 80


@dataclass(frozen=True)
class Chunk:
    """One retrieval unit of a document.

    Attributes:
        id: Position in document order, starting at 0. It is also the chunk's
            index in the list that :func:`chunk_document` returns.
        text: Original wording, which is what the LLM sees.
        section_id: Id of the innermost heading section that contains the chunk.
        breadcrumb: Heading path, such as ``"Design > Major fragments"``.
        kind: ``"row"``, ``"entry"``, ``"kv"`` or ``"para"``.
        char_span: ``(start, end)`` offsets of the source lines that contain the
            chunk. Chunks packed from the same paragraph share their paragraph's span.
        header_ref: For a table row, the id of the table's header row; otherwise None.
    """

    id: int
    text: str
    section_id: int
    breadcrumb: str
    kind: str
    char_span: tuple[int, int]
    header_ref: int | None = None

    @property
    def n_tokens(self) -> int:
        """Length in whitespace-separated words."""
        return len(self.text.split())


@dataclass(frozen=True)
class TermParams:
    """Which term types :func:`extract_terms` produces. Words are always on."""

    bigrams: bool = True
    numbers: bool = True
    char_grams: bool = False
    char_n: int = 4


@dataclass(frozen=True)
class Terms:
    """The terms extracted from one piece of text, grouped by type.

    Attributes:
        words: Stemmed content words, without the prefix, in first-seen order.
        bigrams: Adjacent content-token pairs such as ``"fragment g"``.
        numbers: Canonical numbers such as ``"27758"`` or ``"31.7"``.
        char_grams: Maps each word to the character n-grams of its surface form.
    """

    words: tuple[str, ...] = ()
    bigrams: tuple[str, ...] = ()
    numbers: tuple[str, ...] = ()
    char_grams: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def all_terms(self) -> frozenset[str]:
        """Every term with its type prefix: the set that goes into a chunk's filter."""
        terms = {f"w:{w}" for w in self.words}
        terms |= {f"b:{b}" for b in self.bigrams}
        terms |= {f"n:{n}" for n in self.numbers}
        for grams in self.char_grams.values():
            terms |= {f"c:{g}" for g in grams}
        return frozenset(terms)


# --------------------------------------------------------------------------- #
# Normalisation and term extraction (§4)
# --------------------------------------------------------------------------- #

_QUOTE_DASH = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
})
# A comma between digits that is followed by exactly three digits is a thousands
# separator ("27,758"), not a list separator ("b1, b2").
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
# Letters and digits, with dotted decimals kept whole ("31.7", "b1", "mgal").
_TOKEN = re.compile(r"[^\W_]+(?:\.\d+)*")
_NUMBER = re.compile(r"^\d+(?:\.\d+)?$")

STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been
before being below between both but by can could did do does doing down during each
few for from further had has have having he her here hers him his how i if in into is
it its itself just many me might more most much must my no nor not of off on once only
or other our ours out over own same she should so some such than that the their them
then there these they this those through to too under until up very was we were what
when where which while who whom whose why will with within would you your s t
""".split())


def normalize(text: str) -> str:
    """Apply the shared normalisation used for every chunk, question and grader.

    The steps are NFKC, casefold, unified quotes and dashes, and removal of
    thousands separators.
    """
    text = unicodedata.normalize("NFKC", text).casefold().translate(_QUOTE_DASH)
    return _THOUSANDS.sub("", text)


def tokens(text: str) -> list[str]:
    """Normalise ``text`` and split it into word and number tokens."""
    return _TOKEN.findall(normalize(text))


def stem(word: str) -> str:
    """Strip a plural suffix lightly: berths→berth, cities→city, boxes→box.

    The rule is kept deliberately weak. Char n-grams (ablation (d)) cover the
    remaining morphology.
    """
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("sses", "shes", "ches", "xes", "zes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def canonical_number(token: str) -> str:
    """Give each number one spelling: ``"031.70"`` → ``"31.7"``."""
    integer, _, fraction = token.partition(".")
    integer = integer.lstrip("0") or "0"
    fraction = fraction.rstrip("0")
    return f"{integer}.{fraction}" if fraction else integer


def char_ngrams(word: str, n: int) -> tuple[str, ...]:
    """Return the n-grams of ``#word#``. ``#`` marks the word boundaries."""
    padded = f"#{word}#"
    if len(padded) <= n:
        return (padded,)
    return tuple(padded[i:i + n] for i in range(len(padded) - n + 1))


def extract_terms(text: str, params: TermParams = TermParams()) -> Terms:
    """Extract the words, bigrams, numbers and (optionally) char n-grams of ``text``.

    Stopwords are dropped and break bigram adjacency. Numbers count as content
    tokens for bigrams, which gives ``"berth 16"``. When ``params.numbers`` is off,
    numeric tokens are dropped entirely.
    """
    words: dict[str, None] = {}
    numbers: dict[str, None] = {}
    bigrams: dict[str, None] = {}
    grams: dict[str, tuple[str, ...]] = {}
    previous: str | None = None
    for token in tokens(text):
        if _NUMBER.match(token):
            if not params.numbers:
                previous = None
                continue
            key = canonical_number(token)
            numbers[key] = None
        elif token in STOPWORDS:
            previous = None
            continue
        else:
            key = stem(token)
            words[key] = None
            if params.char_grams and key not in grams:
                grams[key] = char_ngrams(token, params.char_n)
        if previous is not None and params.bigrams:
            bigrams[f"{previous} {key}"] = None
        previous = key
    return Terms(tuple(words), tuple(bigrams), tuple(numbers), grams)


def chunk_terms(chunks: list[Chunk], params: TermParams) -> list[frozenset[str]]:
    """Return the term set to insert into each chunk's filter.

    A table row is indexed together with its header row. The header names what
    the row's values mean ("Weight [g]"), and without it a row like
    ``"Fragment G | 125 × 110 | 31.7"`` cannot be found by a question about weight.
    """
    result = []
    for chunk in chunks:
        terms = extract_terms(chunk.text, params).all_terms()
        if chunk.header_ref is not None:
            # Union rather than concatenation, so no bigram spans the header/row join.
            terms |= extract_terms(chunks[chunk.header_ref].text, params).all_terms()
        result.append(terms)
    return result


# --------------------------------------------------------------------------- #
# Chunking (§3)
# --------------------------------------------------------------------------- #

_RULE = re.compile(r"^\s*([=\-~*_])\1{4,}\s*$")
_MD_HEADING = re.compile(r"^(#{1,6})\s+(\S.*)$")
_WIKI_HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1$")
_SECTION_NUMBER = re.compile(r"^(\d+(?:\.\d+)*)\s")
_ENTRY = re.compile(r"^(\s*)(\d+[.)]|[-*•]|[A-Z]\d+:)\s+\S")
_KV = re.compile(r"^[A-Z][A-Z ]+:\s")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_ABBREVIATIONS = frozenset(
    "dr mr mrs ms st no nos fig figs vol approx e.g i.e etc vs al inc ltd co jr sr".split()
)
_MAX_HEADING_WORDS = 12


@dataclass
class _Line:
    text: str
    start: int

    @property
    def end(self) -> int:
        return self.start + len(self.text)


@dataclass
class _Block:
    """A run of non-blank lines. Blank lines and rule lines (``=====``) separate blocks."""

    lines: list[_Line]
    rule_after: str | None = None


@dataclass
class _Draft:
    """A chunk before merging and final id assignment."""

    text: str
    section_id: int
    breadcrumb: str
    kind: str
    span: tuple[int, int]
    header: _Draft | None = None


def chunk_document(text: str, params: ChunkParams = ChunkParams()) -> list[Chunk]:
    """Cut ``text`` into chunks, following ``params.strategy``."""
    if params.strategy == "fixed":
        return _fixed_chunks(text, params.fixed_window)
    if params.strategy != "structure":
        raise ValueError(f"unknown chunk strategy {params.strategy!r}")
    drafts = _structure_drafts(_blocks(text), params.max_chunk_tokens)
    return _finalise(_merge_short(drafts, params.min_chunk_tokens))


def _fixed_chunks(text: str, window: int) -> list[Chunk]:
    """Cut the text into consecutive windows of ``window`` words, ignoring structure."""
    words = list(re.finditer(r"\S+", text))
    chunks = []
    for i in range(0, len(words), window):
        start, end = words[i].start(), words[min(i + window, len(words)) - 1].end()
        chunks.append(Chunk(len(chunks), text[start:end], 0, "", "para", (start, end)))
    return chunks


def _blocks(text: str) -> list[_Block]:
    """Group lines into blocks, noting which rule character (if any) follows each one."""
    blocks: list[_Block] = []
    current: list[_Line] = []
    offset = 0
    for raw in text.split("\n"):
        line = _Line(raw.rstrip(), offset)
        offset += len(raw) + 1
        rule = _RULE.match(line.text)
        if rule or not line.text.strip():
            if current:
                blocks.append(_Block(current))
                current = []
            if rule and blocks and blocks[-1].rule_after is None:
                blocks[-1].rule_after = rule.group(1)
            continue
        current.append(line)
    if current:
        blocks.append(_Block(current))
    return blocks


def _is_caps_heading(line: str) -> bool:
    """Return whether the line is a short ``SECTION 5 -- ...`` / ``PART ONE -- ...`` style title.

    The first word must be all capitals, and the second must be capitalised too
    or be a number. This keeps captions such as "NASA image of the slick" from
    being taken as headings.
    """
    words = line.split()
    if len(words) < 2 or len(words) > _MAX_HEADING_WORDS or _KV.match(line):
        return False
    first, second = words[0], words[1]
    return (
        first.isalpha() and first.isupper() and len(first) >= 3
        and (second.isupper() or second[0].isdigit())
        and not line.endswith((".", ",", ";", ":"))
    )


def _heading_level(title: str, base: float) -> float:
    """Add one level per dot in a leading section number: ``2.1 Pit Wall`` is deeper than ``2 ...``."""
    number = _SECTION_NUMBER.match(title)
    return base + (number.group(1).count(".") if number else 0)


def _block_heading(block: _Block) -> tuple[float, str, list[_Line]] | None:
    """Return ``(level, title, remaining_lines)`` if the block starts with a heading.

    Heading styles from different documents get levels that nest sensibly:
    all-caps titles (0), other caps titles such as ``PAGE 1 of 7: ...`` (0.5),
    ``# h1``/``== h ==`` (1), and deeper levels below them. A heading must be a
    block of its own, except that a caps title may be followed by key-value
    lines (``SOURCE: ...``).
    """
    lines = block.lines
    first = lines[0].text.strip()
    single = len(lines) == 1
    if single and (m := _MD_HEADING.match(first)):
        return _heading_level(m.group(2), len(m.group(1)) - 1), m.group(2), []
    if single and (m := _WIKI_HEADING.match(first)):
        return len(m.group(1)) - 1, m.group(2), []
    rest = lines[1:]
    if _is_caps_heading(first) and all(_KV.match(line.text) for line in rest):
        return (0 if first.upper() == first else 0.5), first, rest
    # A short title block underlined by a rule (setext style).
    if block.rule_after and len(lines) <= 2:
        title = " -- ".join(line.text.strip() for line in lines)
        if len(title.split()) <= 2 * _MAX_HEADING_WORDS and not title.endswith("."):
            return 0, title, []
    return None


def _structure_drafts(blocks: list[_Block], max_tokens: int) -> list[_Draft]:
    """Walk the blocks, keeping a heading stack, and emit one draft per unit."""
    drafts: list[_Draft] = []
    stack: list[tuple[float, str]] = []
    section_id = 0
    table_header: _Draft | None = None

    def emit(text: str, kind: str, span: tuple[int, int], header: _Draft | None = None) -> _Draft:
        breadcrumb = " > ".join(title for _, title in stack)
        draft = _Draft(text, section_id, breadcrumb, kind, span, header)
        drafts.append(draft)
        return draft

    for block in blocks:
        heading = _block_heading(block)
        lines = block.lines
        if heading:
            level, title, lines = heading
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            section_id += 1
            table_header = None
        paragraph: list[_Line] = []

        def flush() -> None:
            if paragraph:
                span = (paragraph[0].start, paragraph[-1].end)
                joined = " ".join(line.text.strip() for line in paragraph)
                for piece in _pack(_sentences(joined), max_tokens):
                    emit(piece, "para", span)
                paragraph.clear()

        i = 0
        while i < len(lines):
            line = lines[i]
            entry = _ENTRY.match(line.text)
            if line.text.count("|") >= 2:
                flush()
                row = emit(line.text.strip(), "row", (line.start, line.end), table_header)
                table_header = table_header or row
                i += 1
                continue
            table_header = None
            if entry:
                flush()
                indent = len(entry.group(1))
                j = i + 1
                while j < len(lines) and _is_continuation(lines[j].text, indent):
                    j += 1
                text = " ".join(part.text.strip() for part in lines[i:j])
                emit(text, "entry", (line.start, lines[j - 1].end))
                i = j
            elif _KV.match(line.text):
                flush()
                emit(line.text.strip(), "kv", (line.start, line.end))
                i += 1
            else:
                paragraph.append(line)
                i += 1
        flush()
    return drafts


def _is_continuation(line: str, entry_indent: int) -> bool:
    """Return whether the line is more indented than the entry above it and is not a new entry or row."""
    indent = len(line) - len(line.lstrip())
    return indent > entry_indent and not _ENTRY.match(line) and line.count("|") < 2


def _sentences(text: str) -> list[str]:
    """Split a paragraph into sentences, avoiding splits after abbreviations and initials."""
    pieces = _SENTENCE_END.split(text)
    sentences: list[str] = []
    for piece in pieces:
        if sentences and _ends_with_abbreviation(sentences[-1]):
            sentences[-1] += " " + piece
        else:
            sentences.append(piece)
    return sentences


def _ends_with_abbreviation(sentence: str) -> bool:
    last = sentence.split()[-1].rstrip(".").casefold() if sentence.split() else ""
    # Single letters ("J."), dotted acronyms ("U.S."), or known abbreviations ("Dr.").
    is_initial = len(last) == 1 and last.isalpha()
    return is_initial or bool(re.fullmatch(r"(?:[a-z]\.)+[a-z]", last)) or last in _ABBREVIATIONS


def _pack(sentences: list[str], max_tokens: int) -> list[str]:
    """Greedily join consecutive sentences into pieces of at most ``max_tokens`` words.

    A single sentence longer than the limit stays whole rather than being cut mid-sentence.
    """
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for sentence in sentences:
        n = len(sentence.split())
        if current and size + n > max_tokens:
            pieces.append(" ".join(current))
            current, size = [], 0
        current.append(sentence)
        size += n
    if current:
        pieces.append(" ".join(current))
    return pieces


def _merge_short(drafts: list[_Draft], min_tokens: int) -> list[_Draft]:
    """Merge short paragraph or key-value drafts into a neighbour in the same section.

    The next paragraph is preferred, so a lead-in like "Keepers on the roll:" joins
    the list that follows it. Rows and entries are never merged: each one is a
    separate fact.
    """
    mergeable = ("para", "kv")
    result: list[_Draft] = []
    carry: _Draft | None = None  # a short draft waiting to join the next one
    for draft in drafts:
        if carry is not None:
            if draft.kind == "para" and draft.section_id == carry.section_id:
                draft.text = carry.text + " " + draft.text
                draft.span = (carry.span[0], draft.span[1])
            else:
                _append_or_merge_back(result, carry, mergeable)
            carry = None
        if draft.kind in mergeable and len(draft.text.split()) < min_tokens:
            carry = draft
        else:
            result.append(draft)
    if carry is not None:
        _append_or_merge_back(result, carry, mergeable)
    return result


def _append_or_merge_back(result: list[_Draft], draft: _Draft, mergeable: tuple[str, ...]) -> None:
    """Merge ``draft`` into the previous draft if that is allowed; otherwise keep it as is."""
    previous = result[-1] if result else None
    if previous and previous.kind in mergeable and previous.section_id == draft.section_id:
        previous.text += " " + draft.text
        previous.span = (previous.span[0], draft.span[1])
    else:
        result.append(draft)


def _finalise(drafts: list[_Draft]) -> list[Chunk]:
    """Assign ids in document order and resolve header references to ids."""
    ids = {id(draft): i for i, draft in enumerate(drafts)}
    return [
        Chunk(
            id=i,
            text=d.text,
            section_id=d.section_id,
            breadcrumb=d.breadcrumb,
            kind=d.kind,
            char_span=d.span,
            header_ref=ids[id(d.header)] if d.header is not None else None,
        )
        for i, d in enumerate(drafts)
    ]

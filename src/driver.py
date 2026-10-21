"""Run one document's questions through selection and the LLM, recording everything (specs.md §8).

The flow for one document under one :class:`RunConfig`:

1. :func:`prepare` chunks the document, extracts terms and builds the index.
   Together these are ``build_s``.
2. :func:`retrieve` selects a context for each question, timed as ``retrieval_s``.
3. :func:`run_document` sends each prompt to the LLM and appends one
   :class:`Record` per question to a JSONL file as each answer arrives. A rerun
   skips questions that are already recorded.

The baseline uses this same path with ``mode="full"``, so everything after
selection is shared. This module reads contexts and questions only; gold
answers belong to :mod:`evaluate`.

Depends on: :mod:`text`, :mod:`bloom`, :mod:`retriever` and :mod:`llm`.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bloom import BloomParams, ChunkIndex, build_index
from llm import LLMClient, build_prompt
from retriever import BloomRetriever, FullContextSelector, RetrievalParams, Selection, Selector
from text import Chunk, ChunkParams, TermParams, chunk_document, chunk_terms, extract_terms

DATA_DIR = Path(__file__).resolve().parent.parent / "assignment"


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Question:
    qid: str
    question: str
    answer_type: str
    unit: str | None


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    questions: tuple[Question, ...]


def list_documents(data_dir: Path = DATA_DIR) -> list[str]:
    """Return the ids of documents that have both a context file and a questions file."""
    contexts = {p.stem for p in (data_dir / "contexts").glob("*.txt")}
    questions = {p.name.removesuffix("_questions.json") for p in (data_dir / "questions").glob("*_questions.json")}
    return sorted(contexts & questions)


def load_document(doc_id: str, data_dir: Path = DATA_DIR) -> Document:
    """Load a context and its questions (from ``questions/``, never from the answer files)."""
    text = (data_dir / "contexts" / f"{doc_id}.txt").read_text(encoding="utf-8")
    raw = json.loads((data_dir / "questions" / f"{doc_id}_questions.json").read_text(encoding="utf-8"))
    questions = tuple(
        Question(q["qid"], q["question"], q.get("answer_type", ""), q.get("unit"))
        for q in raw["questions"]
    )
    return Document(doc_id, text, questions)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RunConfig:
    """Everything that defines one experimental condition.

    Attributes:
        name: Label written into every record, for example ``"core"`` or ``"R256"``.
        mode: ``"bloom"`` (compressed context) or ``"full"`` (baseline).
        index: ``"rowmajor"``, ``"bitsliced"`` or ``"exact"``. Ignored when ``mode="full"``.
    """

    name: str
    mode: str = "bloom"
    index: str = "rowmajor"
    chunk: ChunkParams = field(default_factory=ChunkParams)
    terms: TermParams = field(default_factory=TermParams)
    retrieval: RetrievalParams = field(default_factory=RetrievalParams)
    bloom: BloomParams = field(default_factory=BloomParams)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Preparing a document: chunk, extract terms, build the index
# --------------------------------------------------------------------------- #


@dataclass
class PreparedDocument:
    """A document ready to answer questions under one config."""

    document: Document
    config: RunConfig
    selector: Selector
    chunks: list[Chunk]
    chunk_terms: list[frozenset[str]]  # diagnostics only (false-positive counts); never used for selection
    index: ChunkIndex | None
    build_s: float
    index_stats: dict


def prepare(document: Document, config: RunConfig) -> PreparedDocument:
    """Chunk the document and build its index. The whole preparation is timed as ``build_s``."""
    start = time.perf_counter()
    if config.mode == "full":
        selector: Selector = FullContextSelector(document.text)
        build_s = time.perf_counter() - start
        stats = {"index": "full", "bloom_bits": 0, "padding_bits": 0, "aux_bytes": 0}
        return PreparedDocument(document, config, selector, [], [], None, build_s, stats)
    if config.mode != "bloom":
        raise ValueError(f"unknown mode {config.mode!r}")

    chunks = chunk_document(document.text, config.chunk)
    terms = chunk_terms(chunks, config.terms)
    index = build_index(config.index, terms, config.bloom)
    selector = BloomRetriever(chunks, index, len(document.text), config.terms, config.retrieval)
    build_s = time.perf_counter() - start

    stats = {
        **index.stats(),
        "n_sections": len({c.section_id for c in chunks}),
        "aux_bytes": aux_bytes(chunks),
        "build_s": build_s,
    }
    return PreparedDocument(document, config, selector, chunks, terms, index, build_s, stats)


def aux_bytes(chunks: list[Chunk]) -> int:
    """Return the persistent per-chunk metadata outside the Bloom bits (§8).

    That is four 64-bit integers per chunk (span start and end, section id, header
    reference) plus the UTF-8 bytes of each distinct breadcrumb. Chunk text is not
    counted, because it is the input document.
    """
    breadcrumbs = {c.breadcrumb for c in chunks}
    return 4 * 8 * len(chunks) + sum(len(b.encode("utf-8")) for b in breadcrumbs)


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Retrieval:
    """One question's selection, with timing and the false-positive diagnostic."""

    question: Question
    selection: Selection
    retrieval_s: float
    query_fp_hits: int


def retrieve(prepared: PreparedDocument, question: Question) -> Retrieval:
    """Select a context for one question. Only the selection is timed."""
    start = time.perf_counter()
    selection = prepared.selector.select(question.question)
    retrieval_s = time.perf_counter() - start
    return Retrieval(question, selection, retrieval_s, query_fp_hits(prepared, question.question))


def query_fp_hits(prepared: PreparedDocument, question: str) -> int:
    """Count (question term, chunk) pairs that the index claims but the chunk does not contain.

    This is a diagnostic that runs outside the timed path, and is the only place
    where exact term sets meet the index.
    """
    if prepared.index is None:
        return 0
    false_hits = 0
    for term in extract_terms(question, prepared.config.terms).all_terms():
        claimed = prepared.index.query(term)
        false_hits += sum(
            1 for c in claimed.nonzero()[0] if term not in prepared.chunk_terms[c]
        )
    return false_hits


def retrieve_all(prepared: PreparedDocument) -> list[Retrieval]:
    """Run retrieval for every question of the document. Used by the offline experiments."""
    return [retrieve(prepared, q) for q in prepared.document.questions]


# --------------------------------------------------------------------------- #
# Full run: retrieval plus the LLM
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Record:
    """One answered question under one config: a row of ``records.jsonl`` (§8)."""

    config: str
    doc_id: str
    qid: str
    index: str
    R: int | None
    k: int | None
    selection_status: str
    retained_chunk_ids: list[int]
    retained_chars: int
    retained_frac: float
    context_tokens: int
    prompt_tokens: int
    output_tokens: int
    build_s: float
    retrieval_s: float
    llm_s: float
    total_s: float
    retries: int
    cached: bool
    match_score: float
    query_fp_hits: int
    model_version: str
    answer_raw: str


def make_record(prepared: PreparedDocument, retrieval: Retrieval, response) -> Record:
    """Combine a retrieval and an LLM response into a :class:`Record`."""
    selection = retrieval.selection
    n_questions = max(len(prepared.document.questions), 1)
    return Record(
        config=prepared.config.name,
        doc_id=prepared.document.doc_id,
        qid=retrieval.question.qid,
        index=prepared.index_stats.get("index", ""),
        R=prepared.index_stats.get("R"),
        k=prepared.index_stats.get("k"),
        selection_status=selection.status,
        retained_chunk_ids=list(selection.chunk_ids),
        retained_chars=len(selection.text),
        retained_frac=selection.retained_frac,
        context_tokens=selection.est_tokens,
        prompt_tokens=response.prompt_tokens,
        output_tokens=response.output_tokens,
        build_s=prepared.build_s,
        retrieval_s=retrieval.retrieval_s,
        llm_s=response.latency_s,
        # Amortised end-to-end time: each question carries 1/n of the index build.
        total_s=prepared.build_s / n_questions + retrieval.retrieval_s + response.latency_s,
        retries=response.retries,
        cached=response.cached,
        match_score=selection.match_score,
        query_fp_hits=retrieval.query_fp_hits,
        model_version=response.model_version,
        answer_raw=response.text,
    )


def recorded_qids(records_path: Path, config_name: str, doc_id: str) -> set[str]:
    """Return the qids already recorded for this config and document, so a rerun can resume."""
    if not records_path.exists():
        return set()
    done = set()
    for line in records_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["config"] == config_name and row["doc_id"] == doc_id:
            done.add(row["qid"])
    return done


def run_document(
    prepared: PreparedDocument, llm: LLMClient, records_path: Path, workers: int = 4
) -> list[Record]:
    """Answer every not-yet-recorded question and append its record to ``records_path``.

    Retrieval runs sequentially, so its timings are not distorted by contention.
    LLM calls then run on a small thread pool, and each record is written as soon
    as its answer arrives.
    """
    records_path.parent.mkdir(parents=True, exist_ok=True)
    done = recorded_qids(records_path, prepared.config.name, prepared.document.doc_id)
    pending = [retrieve(prepared, q) for q in prepared.document.questions if q.qid not in done]
    lock = threading.Lock()

    def answer(retrieval: Retrieval) -> Record:
        q = retrieval.question
        prompt = build_prompt(retrieval.selection.text, q.question, q.answer_type, q.unit)
        record = make_record(prepared, retrieval, llm.generate(prompt))
        with lock, records_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record)) + "\n")
        return record

    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        return list(pool.map(answer, pending))

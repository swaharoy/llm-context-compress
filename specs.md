# LLM Context Compression via Bloom Filters — Design Spec

## 1. Goal and constraints
- Answer each (context, question) pair with `gemini-3.5-flash-lite`. The baseline gets the full context; the method gets a context compressed by a Bloom-filter index. The model, prompt template and generation config are identical for both.
- Report the following as R and k vary:
  - accuracy;
  - LLM input tokens;
  - latency (index build, retrieval and generation);
  - Bloom-filter memory;
  - membership FPR, measured vs. theoretical.
- **Generality rule:** the pipeline never reads `answers/`. Only `evaluate.py` loads gold data, and a test enforces this. There are no document-specific rules or vocabulary.
- **Reproducibility:** fixed seeds and deterministic hashing; Python's salted `hash()` is never used. One command reproduces everything.
- **Scope rule:** a simple, frozen core method that is fully evaluated comes first. Extensions are added only in the order given in §10, and only after the previous stage is complete.

## 2. Architecture

```
src/
  main.py         CLI: `ask` (one doc + question(s)) and `experiments` (full study)
  bloom.py        hashing, BloomFilter, RowMajorBloomIndex, BitSlicedBloomIndex (Stage 2), ExactSetIndex, BloomParams
  text.py         structure-aware chunker + term extraction / normalisation, ChunkParams
  retriever.py    scoring, selection under budget, context assembly, FullContextSelector, RetrievalParams -> Selection
  llm.py          LLMClient protocol, GeminiClient, FakeLLM, PROMPT_TEMPLATE, disk cache, retries
  driver.py       run_document(context, questions, RunConfig, llm) -> list[Record]
  evaluate.py     answer grading, gold-span recall, aggregation, bootstrap CI
  experiments.py  core / ablation / sweep presets, staged runs, writes results/, calls evaluate + plots
  plots.py        required plots (+ a few diagnostics)
tests/            test_bloom, test_text, test_retriever, test_evaluate, test_driver (FakeLLM), test_no_gold_leak
```

**Why 9 files instead of 2:** each file is one testable concern. The graded data structure (`bloom.py`) and the method (`text.py` + `retriever.py`) stay independent of the harness (`llm`, `driver`, `experiments`). Parameter dataclasses live in the module that owns them, and run presets live in `experiments.py`. The README states this reasoning.

**Why `text.py` and `retriever.py` are separate:**
- `text.py` defines what the document is made of (chunks and terms). `evaluate.py` and the FPR probes reuse it.
- `retriever.py` decides what to keep for a given question.
- Chunking and retrieval are separate ablation axes.

**Data flow:**
1. `text.py` turns the context into `Chunk`s.
2. `bloom.py` builds an index from the chunks' terms.
3. `retriever.py` turns a question into a `Selection`.
4. `llm.py` returns an `LLMResponse`.
5. `driver.py` builds a `Record`.
6. `evaluate.py` produces a `GradedRecord`.

Every record is a plain dataclass serialised to JSONL.

## 3. Chunking (`text.py`)
The goal is retrieval units that are small enough to be precise yet self-contained enough to be understood.

1. **Line classification, generic, in priority order:**
   - heading: markdown `#…`, wiki `=+ x =+`, a line followed by a `===`/`---` rule, or a short line that is entirely upper-case or title-case and sits between blank lines;
   - table row: contains ≥2 `|` separators;
   - list/numbered entry: `^\s*(\d+[.)]|[-*•]|[A-Z]\d*:)`, and following lines with deeper indentation are continuation lines that belong to the entry;
   - key-value line: `^[A-Z ]{2,}:\s`;
   - rule or blank (separator);
   - paragraph text.
2. **Section tree:** heading levels give each chunk a breadcrumb such as `Design > Major fragments`.
3. **Units:**
   - one table row is one chunk, carrying its table's header row;
   - one list entry plus its continuation lines is one chunk;
   - paragraphs are split into sentences (regex splitter that is aware of abbreviations and decimals), then greedily packed up to `max_chunk_tokens` (default 80) without crossing a section boundary;
   - very short fragments (under `min_chunk_tokens`, default 8) are merged with a neighbour.
4. **Chunk fields:** each `Chunk` has `id`, `char_span`, `text`, `section_id`, `breadcrumb`, `kind` (row, entry, para or kv), and `header_ref` (for table rows).
5. **Ablation chunker `fixed`:** windows of 80 whitespace tokens with no structure (ablation (a), §6.3).

## 4. Terms (`text.py`)
**Normalisation,** applied identically to chunks and questions:
- NFKC;
- casefold;
- unify quote and dash characters;
- strip thousands separators from numbers (`27,758` → `27758`).

| Type | Prefix | Definition | Core? |
|---|---|---|---|
| word | `w:` | non-stopword tokens, light suffix stripping (`berths`→`berth`) | yes |
| bigram | `b:` | adjacent content-word pairs | yes |
| number | `n:` | canonical numeric form | yes |
| char n-gram | `c:` | 4-grams of each content word with boundary marks (`#ber`, `bert`, …) | ablation (d) |

All enabled term types go into the **same** filter, namespaced by prefix. N per chunk therefore depends on which types are enabled, which changes FPR and memory. This is reported, not hidden.

## 5. Bloom filter implementation (`bloom.py`)
**Hashing.**
- One `blake2b(term, digest_size=16, key=seed)` per term, split into two 64-bit integers `h1, h2`.
- Standard double hashing (Kirsch–Mitzenmacher): `g_i = (h1 + i·h2) mod R` for `i < k`.
- **R is restricted to powers of two** and `h2` is forced odd (`h2 |= 1`). h2 is then coprime with R, so the k positions are distinct for any k ≤ R.
- Positions are memoised per `(term, R, k, seed)` within one index build.
- The theoretical FPR `(1 − e^{−kN/R})^k` is an approximation for this hash family; measured FPR is compared against it, not assumed to equal it.

**`BloomFilter(R, k, seed)`:** a standalone filter backed by a packed `numpy.uint8` array of R/8 bytes. It provides:
- `add`, `add_many` and `__contains__`;
- `n_inserted`, `fill_ratio` and `memory_bits` (from the actual allocation);
- `theoretical_fpr()` and `optimal_k(R, N) = max(1, round(R/N · ln 2))`.

It is the reference implementation for tests.

**Chunk index interface.** All index classes are interchangeable in `retriever.py`, and none exposes term sets:
- `build(chunk_terms, params)`;
- `query(term) -> chunk bitmap`;
- `df_hat(term)`;
- `memory_bits()` and `stats()`.

Implementations:
- **`RowMajorBloomIndex` (core):** one packed row of R bits per chunk, an `n × R/8` `uint8` matrix. `query(term)` gathers the k bit columns for all chunks with numpy and ANDs them.
- **`BitSlicedBloomIndex` (Stage 2 add-on):** the same filters stored transposed (§5.1). It must produce identical score vectors to `RowMajorBloomIndex`.
- **`ExactSetIndex` (control):** Python sets per chunk, so FPR = 0. It isolates the effect of approximate membership under an otherwise identical retrieval algorithm (report question 3). Its memory is reported separately and never counted as Bloom memory.

Exact term sets exist **only** inside `ExactSetIndex` and in diagnostic code (FPR probes, `query_fp_hits`). They are never on the retrieval path of a Bloom index.

### 5.1 Bit-sliced Bloom index (Stage 2 add-on)

**Problem.** Each document has n chunks (hundreds to low thousands), each with its own Bloom filter. Scoring one question tests every question term against every chunk filter. That is |terms| × n × k bit probes per question, repeated for 80 questions and every (R, k) in the sweep.

**Layout.** In row-major form, filter c is a row of R bits. Because all chunk filters share R and the same hash positions, the bits can be stored transposed. Each bit position j gets one row of n bits, called a *bit slice*. Bit c of slice j is 1 exactly when chunk c's filter has bit j set.

```
row-major (n filters × R bits)            bit-sliced (R slices × n bits)
          bit: 0 1 2 3 4 5 6 7                       chunk: c0 c1 c2 c3
chunk c0:      1 0 0 1 0 1 0 0            slice 0:          1  0  1  0
chunk c1:      0 1 0 1 0 0 1 0            slice 3:          1  1  0  1
chunk c2:      1 0 1 0 0 1 0 0            slice 5:          1  0  1  0
chunk c3:      0 0 0 1 1 0 0 1            ...
```

The slices are stored as a numpy `uint64` matrix of shape `(R, ⌈n/64⌉)`, so 64 chunks share each machine word. **Bit order is LSB-first:** chunk c is bit `c % 64` of word `c // 64`. Unpacking always uses `bitorder="little"` on the little-endian byte view.

**Operations.**
- **Build:** for each chunk c and each position g of each of its terms, set `S[g, c // 64] |= 1 << (c % 64)`. This is vectorised by grouping (g, c) pairs.
- **Query one term:** `S[g_0] & … & S[g_{k−1}]` gives a bitmap of the chunks whose filter claims the term. That is O(k·n/64) word operations, and the query can stop early once the running AND is all zeros.
- **Document frequency:** `df_hat(t) = popcount(bitmap)`. It equals the true df plus false-positive chunks, so df_hat ≥ df always.
- **Scoring:** term bitmaps are unpacked to 0/1 vectors and combined as a weighted sum, giving a score for every chunk in one pass.

**Why bit-sliced instead of row-major.** The retriever asks, for each question term t, "which of the n chunks might contain t?". All filters share R and the hash family, so t maps to the *same* k positions in every filter, and the query reads the same k bit columns from all n filters. Example: n = 2,000, R = 1,024 bits (128 B per filter), k = 4, one term:

| | Row-major (filter per row) | Bit-sliced (bit position per row) |
|---|---|---|
| Bits needed | n·k = 8,000 | 8,000 |
| Access pattern | 8,000 single-bit reads, 128 B apart | k contiguous slices of n bits (250 B each) |
| Cache lines touched | ~8,000 (1 useful bit per 512 loaded) | ~16 (every loaded bit is used) |
| Operations | 8,000 bit tests / gathers | k·⌈n/64⌉ = 128 word ANDs (64 chunks per op) |
| Result | n separate booleans | one n-bit bitmap |

The bitmap result also suits bitmap algebra: `df_hat` is one popcount, and section masks or excluding chosen chunks are a single AND / AND-NOT.

**Where row-major is better:**
- point queries ("t in chunk c?");
- filters of different sizes;
- simplicity.

That simplicity is why row-major is the core and bit slicing is an add-on.

**Caveat:** at our scale the whole index is a few hundred KB and fits in cache, so the cache-line gap overstates the wall-clock gap. The speed-up is **measured, not assumed**, against both a pure-Python row-major loop and the vectorised `RowMajorBloomIndex`.

**Memory.** The actual allocation is `8·R·⌈n/64⌉` bytes, versus `n·R/8` bytes for the row-major index, which has no padding because R is a power of two ≥ 8. The difference is padding to a multiple of 64 chunks per slice (at most 63 bits per slice). Both indexes report `memory_bits()` from their real `nbytes`, with padding listed separately. FPR is unchanged: each chunk's false-positive probability is still about `(1 − e^{−k·N_c/R})^k`.

**Constraints:**
- all filters share R and the hash family;
- chunks with more terms have a higher FPR at a shared R, so the per-chunk FPR spread is reported;
- the chunk set is fixed at build time, and the index is rebuilt per document.

**Origin.** The idea predates Bloom filters and comes from information retrieval:
- **Superimposed coding (1940s).** Calvin Mooers' *Zatocoding* punched-card system encoded each descriptor as a few random notches superimposed on one card. A query matched when all its notches were present. This is the same "OR in, AND out" rule, with the same false drops.
- **Bloom filter (1970).** B. H. Bloom, "Space/Time Trade-offs in Hash Coding with Allowable Errors", *CACM* 13(7).
- **Bit-sliced signature files (1979).** C. S. Roberts, "Partial-Match Retrieval via the Method of Superimposed Codes", *Proc. IEEE* 67(12). It stored per-document signatures transposed into bit slices, so that only the k slices a query touches are read. That is the layout used here.
- **Analysis and comparison.**
  - C. Faloutsos and S. Christodoulakis, "Signature Files: An Access Method for Documents and Its Analytical Performance Evaluation", *ACM TOIS* (1984), analysed false-drop rates.
  - J. Zobel, A. Moffat and K. Ramamohanarao, "Inverted Files Versus Signature Files for Text Indexing", *ACM TODS* 23(4) (1998), argued inverted indexes usually win for general text search. This is relevant to report question 6.
- **Revival (2017).** B. Goodwin et al., "BitFunnel: Revisiting Signatures for Search", *SIGIR 2017*. It brought bit-sliced signatures back on modern CPUs (used in Bing), adding frequency-conscious signatures and higher-rank rows. Neither addition is in scope.

(Citation volume and issue numbers are to be verified before they are used in the report.)

**Testing.**
- The **full per-chunk score vector** for random documents and queries is identical to `RowMajorBloomIndex`. This catches bit-order bugs in the scoring path, not just in membership.
- `df_hat ≥ df` always holds.
- `memory_bits()` equals the allocation.
- The speed-up is benchmarked and reported; no test asserts it.

## 6. Retrieval method (`retriever.py`)

### 6.1 Scoring (frozen core)
Question terms are extracted exactly as in §4. For each chunk c, every score component comes from `index.query`:

- **Word match:** `m(q, c) = 1[index.query("w:"+q) has bit c]` for each question content word q. The match contributes `w(q)·m(q, c)`, where `w(q) = idf_hat(q) = log((n+1)/(df_hat(q)+1))`. False positives inflate df_hat and visibly reduce weights.
- **Bigram and number matches:** each hit adds `λ_b·idf_hat` (λ_b = 1.5) or `λ_n·idf_hat` (λ_n = 2.0).
- **Char 4-grams (ablation (d) only):** if exact word match fails, `m = coverage` when the fraction of q's 4-grams whose `index.query` hits c is at least τ_c = 0.75, and 0 below that. Coverage is computed from index queries only.
- **`match_score`** = `top_score / Σ_q max_weight(q)` is a *heuristic diagnostic*, not a probability. It is 0 when the denominator is 0, and it does not affect selection in the core method.

### 6.2 Selection under budget
`budget_tokens` (default 1500) is a hard cap on the **final assembled context**, including breadcrumbs, table headers and separators. Budget checks use the deterministic estimate `est(text) = ceil(len(text)/4)`. The tokens reported come from the API (§7).

1. **Edge cases:** if the question has no usable terms, or the top score is 0, the result is an empty context with `selection_status = "no_match"`. The prompt is still sent, and the case is counted in the results.
2. **Rank:** sort chunks by score descending, breaking ties by chunk id ascending. Candidates are the chunks with score ≥ α·top (α = 0.5) and score > 0. If every chunk passes the threshold, the budget caps the result.
3. **Greedy core pass:** take candidates in rank order. The cost of a candidate is its text plus any breadcrumb or table header not already included. Add it if the cost fits the remaining budget; otherwise skip it and try the next.
4. **Neighbour pass:** for core chunks in rank order, add the ±1 neighbours within the same section, if they fit.
5. **Assemble:** order the chosen chunks by document position, group them under `[§ breadcrumb]` separators, and put each table row's header once above its rows.

### 6.3 Core method and ablations
The **core method** is frozen before any LLM run. It consists of:
- structure-aware chunks;
- words + bigrams + numbers;
- idf_hat weighting;
- α = 0.5, ±1 neighbour, budget 1500;
- R = 1024 and k = `optimal_k(R, mean N)`.

**Ablations** each change exactly one thing from the core. They are **reported comparisons only**: they never decide the final configuration, which is the core regardless of how they turn out.

| Ablation | Change from core |
|---|---|
| (a) | fixed 80-word chunker instead of structure-aware chunks |
| (b) | words only (no bigrams, no numbers) |
| (c) | uniform weights instead of idf_hat |
| (d) | + char 4-grams |

**`FullContextSelector`** (baseline, §10.1) implements the same `select(question) -> Selection` interface and returns the document unchanged.

## 7. LLM layer (`llm.py`)
- `google-genai` SDK, model `gemini-3.5-flash-lite`, `temperature=0`, fixed `seed`, and `max_output_tokens = 64`.
- One `PROMPT_TEMPLATE` for every run:
  - system instruction: answer only from the context, give only the answer, no explanation;
  - then the context block and the question;
  - then `Answer type: {answer_type}; unit: {unit}`. These fields come from the questions file.
- Input tokens come from `response.usage_metadata.prompt_token_count`. Context-only tokens come from `count_tokens(context)`, which is cached.
- Disk cache: `cache/llm/{sha256(model|gen_config|prompt)}.json` stores the response text, usage, original latency and retry count. A cached replay is flagged `cached=true`; its latency is *not* a fresh observation. `--no-cache` forces fresh calls for reported latency.
- Bounded thread pool with exponential-backoff retries. Latency is measured around the successful request, and the number of retries is recorded.
- `FakeLLM` for tests is deterministic (for example, it echoes the first number in the context).

## 8. Statistics and storage
Metrics are kept at three separate levels, never conflated:
1. **Membership FPR:** Bloom false positives on absent terms, measured vs. theoretical.
2. **Retrieval effects:** false-positive (term, chunk) matches on real questions, retained context, and the difference from `ExactSetIndex` under the same algorithm.
3. **End-to-end outcomes:** input tokens, latency and accuracy.

Each run writes `results/<run_name>/`:

- **`config.json`:** the resolved configs, seed, git commit, package versions and `--no-cache` flag.
- **`index_stats.jsonl`:** one row per (doc, config), containing:
  - `n_chunks`, `n_sections`, `N_mean/max`, `R`, `k`;
  - `bloom_bits`, the actual allocation, with `padding_bits` listed separately;
  - `aux_bytes`, the persistent chunk metadata: `nbytes` of offset and section-id arrays, plus the UTF-8 bytes of breadcrumbs;
  - `exact_set_bytes`, for `ExactSetIndex` only: `sys.getsizeof` of the sets plus their term strings;
  - `fill_ratio`;
  - `fpr_theoretical`, the mean over chunks using each N_c, plus its min and max;
  - `fpr_measured` and `build_s`.
- **`retrieval_grid.jsonl`:** the Stage 2 step 2 offline grid, with no LLM. One row per (doc, qid, R, k or exact), containing `retained_chunk_ids`, `retained_frac`, `est_context_tokens`, `selection_status`, `query_fp_hits`, `retrieval_s` and `gold_span_recall`.
- **Not counted as memory:** the context text (it is the input), the hash-position cache and scoring temporaries, which are transient.
- **`records.jsonl`:** one row per (doc, qid, config), written as soon as each answer arrives, so an interrupted run resumes. It contains:
  - `selection_status`, `retained_chunk_ids`, `retained_chars`, `retained_frac`, `context_tokens`;
  - `prompt_tokens` and `output_tokens` from the API;
  - `build_s` (shared per doc), `retrieval_s`, `llm_s`, `retries`, `cached`;
  - `total_s = build_s / n_questions + retrieval_s + llm_s`, the amortised end-to-end time;
  - `match_score` and `query_fp_hits` (diagnostic, computed off the timed path);
  - `answer_raw`.
- **`graded.jsonl`:** each record plus `answer_norm`, `parse_status`, `correct_strict`, `correct_lenient` and `gold_span_recall`.
- **`summary.csv`:** per config, totals and means overall, per document, per tier (T1/T2/T3) and per lexical overlap (verbatim, paraphrase or mismatch).
- **`plots/*.png`.**

**Measured FPR protocol:** for each chunk filter, probe with M = 1000 terms:
- half are document terms absent from that chunk according to the exact set;
- half are random strings seeded from `seed`.

`fpr_measured = positives / probes`. It is computed per (R, k) with no LLM involved.

## 9. Evaluation (`evaluate.py`)

### 9.1 Answer schema (verified from `assignment/answers/*.json`)
Each file has `{doc_id, count, answers: [...]}`. Every one of the 80 entries has the following fields:
- `qid`, `question`, `answer_type` (`number | entity | phrase | date`);
- `expected` and `aliases`, duplicated as `gold_answer` and `accepted_aliases`;
- `grading{normaliser, numeric_tolerance_rel, unit, max_answer_tokens}`;
- `tier`, `lexical_overlap`, `min_chunks_required`, `retrieval_keys`;
- `gold_spans[{source, section_path, char_start, char_end, quote}]`, where `char_start` and `char_end` are always `null`;
- `known_distractor_spans`, `provenance`, `contamination`, `status`.

The harbour entries also have `clue_ids`.

- **`numeric` (50 entries):** the tolerance is always relative, 0.005, and no expected value is 0.
  - Values and aliases carry units and scale words: `$250 million`, `13.3 million years`, `13,300,000 ft`, `$250m`, `over 400`.
  - Some aliases are equivalents in other units (`4,100 km`, `0.305 kg`), and many are bare numbers (`305`, `7`, `20`).
  - So **the data does not enforce units, and neither does the grader.** There is no unit conversion.
- **`exact_ci` (30 entries):** `numeric_tolerance_rel` is sometimes present and is ignored.
- **`unit`** is a string, `''` or `null`. It is used in the prompt only.

### 9.2 Grading rules
- **`numeric` (strict, headline):**
  1. Parse the **first** number in the answer. The parser handles commas, decimals, `$`, `%`, scale words (`thousand`, `million`, `billion`, `m`/`bn` after a number) and number words (`sixty-three`, `one hundred and twenty`).
  2. A `-` counts as a sign only at the start of a token, so `1,400-psi` parses as 1400.
  3. Parse `expected` and every alias the same way.
  4. The answer is correct if `|a − g| ≤ 0.005·|g|` for any parsed gold value. The `g = 0` case would require exact equality; it does not occur in this data.
  5. Record `parse_status ∈ {ok, no_number, multiple_numbers}`. `no_number` is graded incorrect. `multiple_numbers` is graded on the first number and flagged.
- **`exact_ci` (strict):** normalise both sides:
  - NFKC, casefold, unify dashes and quotes, collapse whitespace;
  - strip *leading and trailing* punctuation and one leading article (`a`, `an`, `the`);
  - keep inner symbols (`352.3±1.5`, `1-13 meters`).

  Correct on equality with `expected` or any alias.
- **Lenient (diagnostic only):** the normalised gold answer or an alias appears as a whole-word substring of the answer. It separates formatting failures from retrieval failures.
- **Gold-span recall (diagnostic, no LLM needed):** the fraction of a question's `gold_spans[].quote` contained in the compressed context, using token-set overlap ≥ 0.9 after normalisation.
- **Uncertainty:** bootstrap 95% CI for each config's accuracy (10,000 resamples, fixed seed), plus per-document accuracy. Questions within a document may be correlated, so config-vs-baseline differences are reported as **exploratory**.
- **Anti-overfitting:**
  - core constants are frozen in §6.3 before any LLM run;
  - ablations never select the final config;
  - results are always broken down by document, tier and lexical overlap;
  - `test_no_gold_leak` asserts that `text`, `retriever`, `bloom` and `driver` never import or open answer files.

## 10. Experiment plan (`experiments.py`)
The stages run in order. Each stage must be complete before the next begins.

**Stage 1: correctness, no paid runs.** Tests (§12) pass: Bloom correctness, cross-process hashing, FPR vs. theory, the evaluator on all 80 gold entries, a FakeLLM end-to-end run, and retrieval using only the index interface. Then a 2-question smoke test of the baseline and the core config against the real API.

**Stage 2: required evaluation.** The Bloom filter meets the 80 questions twice:
- **Step 2** is wide and free: every (R, k), retrieval only, no LLM.
- **Step 3** is narrow and paid: a few (R, k), retrieval plus LLM answers plus grading.

| Step | Bloom index used | Runs the 80 questions | LLM calls | Main outputs |
|---|---|---|---|---|
| 1 Baseline | none (full context) | yes, full context → LLM | 80 | reference accuracy, tokens, latency |
| 2 Offline grid | 49 (R, k) + `ExactSetIndex` | yes, retrieval only | 0 | memory, FPR, retained tokens, gold-span recall |
| 3 LLM sweep | 6 (R, k) + `ExactSetIndex` | yes, retrieval → LLM → grade | 560 | accuracy, real input tokens, latency |
| 4 Plots | — | — | 0 | the three required plots |
| 5 Bit-sliced add-on | the step 3 configs, bit-sliced | yes, retrieval only | 0 (cache hits) | build and retrieval time, equivalence check |

1. **Baseline (80 calls).** For each document and each of its 20 questions, send the full document plus the question to the LLM with `FullContextSelector` (§10.1), then grade. There is no Bloom filter here; this is the reference point for every comparison.

2. **Offline grid (0 calls): the Bloom filter on all 80 questions, without the LLM.** For each R ∈ {64, 128, 256, 512, 1024, 2048, 4096} × k ∈ {1, 2, 3, 4, 6, 8, optimal} (49 settings), plus `ExactSetIndex`, and for each document:
   1. **Build** the core-config index over the document's chunks, and record `build_s`, `bloom_bits`, `padding_bits`, `aux_bytes` and `fill_ratio`.
   2. **Measure the filter alone:** `fpr_measured` using the probe protocol (§8), next to `fpr_theoretical`. This is metric level 1.
   3. **Retrieve for each of the document's 20 questions:** run the core retriever (§6) to get the `Selection`. Record `retained_chunk_ids`, `retained_frac`, the estimated context tokens, `selection_status`, `query_fp_hits` and `retrieval_s`. This is metric level 2.
   4. **Score retrieval quality:** compute `gold_span_recall`, i.e. whether the kept context still contains the answer passage. This is an upper-bound signal on accuracy; it is not accuracy.

   Rows go to `index_stats.jsonl` and `retrieval_grid.jsonl`. This step answers report questions 1 and 2 and supplies plot (a). Accuracy is *not* produced here, because only the LLM produces answers.

3. **LLM sweep (7 × 80 = 560 calls): the Bloom filter on all 80 questions, end to end.** The configurations are fixed now, not chosen from step 2 results:
   - R ∈ {64, 128, 256, 512, 1024, 4096}, each with optimal k, spanning high to low FPR. R = 1024 is the core point.
   - `ExactSetIndex` (FPR = 0), using the same retriever.

   For each configuration × 80 questions:
   1. Build the index and retrieve, exactly as in step 2. The selections are identical to the matching step 2 rows.
   2. Send the compressed context plus the question with the same `PROMPT_TEMPLATE` and settings as the baseline.
   3. Record `prompt_tokens`, `output_tokens`, `llm_s` and `total_s` from the API, and write the record immediately.
   4. Grade with §9.2.

   This step answers report questions 3, 4 and 5. Comparing `ExactSetIndex` with the Bloom points separates the method's error from error caused by false positives.

4. **Required plots (0 calls), drawn from saved results:**
   - (a) Bloom memory vs. FPR, measured and theoretical per k, from step 2;
   - (b) Bloom memory vs. LLM input tokens: real API tokens at the step 3 points, with step 2's estimated tokens as the full curve;
   - (c) LLM input tokens vs. accuracy: the baseline (step 1) and the 7 step 3 configurations.

5. **Bit-sliced add-on (0 new calls).** Rebuild each step 3 configuration with `BitSlicedBloomIndex` and run all 80 questions through retrieval. Pass criteria:
   - identical `retained_chunk_ids` for every question;
   - every LLM call is a cache hit, because the prompts are byte-identical.

   Report `build_s` and `retrieval_s` against `RowMajorBloomIndex`, and the pure-Python benchmark (§5.1). Accuracy, tokens and LLM latency are identical by construction (§5.1).

**Stage 3: ablations and budget.**
- Ablations (a)–(d): 4 × 80 = 320 calls.
- Budgets {500, 3000} on the core config: 2 × 80 = 160 calls. The 1500 budget is the core point already run.

The total for Stages 1–3 is about 1,120 calls, all cached.

**Stage 4: optional, only if Stage 3 is complete.** These are described in §13.

**Plots:**
- **required:**
  - (a) Bloom memory vs. FPR, measured and theoretical curves per k;
  - (b) Bloom memory vs. LLM input tokens;
  - (c) LLM input tokens vs. accuracy, with the baseline marked and the sweep, ablations and budgets as series;
- **diagnostics:** accuracy per document, tier and lexical overlap; and gold-span recall vs. FPR.

### 10.1 Baseline
- **Same pipeline:** the baseline is a selector, not a separate script. `config.retrieval = "full"` selects `FullContextSelector`, which returns the document unchanged. Everything after selection is shared with every Bloom config: driver, `PROMPT_TEMPLATE`, model and generation settings, token counting, latency timing, cache, `Record` format and grading.
- **Recorded values:** `retained_frac = 1.0`, `bloom_bits = 0` and `aux_bytes = 0`. `build_s` and `retrieval_s` are still measured, and are about 0.
- **Run:** 4 documents × 20 questions gives 80 calls, with `config = "full"`.
- **Pre-flight check:** every full prompt is counted with `count_tokens` before the run. The run fails loudly if any prompt exceeds the model's context window.
- **Uses of the result:** the reference line on the tokens vs. accuracy plot, the denominator for tokens saved and latency saved, and the paired reference for exploratory comparisons.
- **Latency:** the reported baseline and core runs use `--no-cache`, so latency is compared on fresh calls.

## 11. CLI (`main.py`)
- `python src/main.py ask --context PATH (--question "..." | --questions JSON) [--index rowmajor|bitsliced|exact] [--R 1024] [--k auto] [--budget 1500] [--print] [--out DIR]`
- `python src/main.py experiments [--stage 1|2|3|all] [--offline-only] [--no-cache] [--run-name NAME]` runs §10, including the plots.

## 12. Testing
- **`test_bloom`:**
  - no false negatives;
  - measured FPR within 3σ of theory on random strings across several (R, k);
  - distinct positions for k ≤ R;
  - hashing stable across processes (subprocess check);
  - `memory_bits()` equals the allocation;
  - bit-sliced and row-major indexes give identical full score vectors.
- **`test_text`:** line classification and chunking on small synthetic snippets of each layout (headings, pipe table, numbered entries with continuation lines, paragraphs), number normalisation, and n-gram generation.
- **`test_retriever`:**
  - a synthetic document where the gold chunk ranks first;
  - the final assembled context never exceeds the budget estimate;
  - deterministic tie-breaking;
  - the `no_match` path;
  - neighbours stay inside their section;
  - with a Bloom index, retrieval never touches exact term sets (a spy index raises if they are accessed).
- **`test_evaluate`:**
  - every one of the 80 gold entries grades correct against its own `expected` and each alias;
  - tolerance edges;
  - scale and number words (`$250m`, `one hundred and twenty`);
  - `1,400-psi`;
  - `no_number` and `multiple_numbers`;
  - article and punctuation stripping.
- **`test_driver`:** an end-to-end run with FakeLLM produces well-formed records, and resume skips records that are already written.
- **`test_no_gold_leak`:** see §9.2.

## 13. Deferred extensions (Stage 4, optional)
Each extension is added only after Stage 3 is complete, as a separate reported comparison against the frozen core:
- **Section hierarchy:** section-level `BloomFilter`s route the query to the top S sections, and a strong match on heading terms includes the whole section. Targets heading-scoped counting questions.
- **Feedback hop:** rare terms (df_hat ≤ δ) from the top 3 chunks that are not in the question are re-queried with weight 0.5. Targets two-step questions.
- **Adaptive budget / fallback:** a low `match_score` expands the budget or sends the full context. The fallback rate is reported.
- **`bits_per_element` sizing:** R_c = b·N_c, so every chunk has equal FPR. It cannot be bit-sliced.
- **Extra statistics and plots:** paired tests with multiple-comparison correction, and further diagnostics.

If they are not implemented, the expected weaknesses on two-step and differently-worded questions are reported as limitations (report question 6).

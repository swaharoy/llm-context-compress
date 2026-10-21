# LLM Context Compression via Bloom Filters — Design Spec (v2: common-word filtering)

This spec replaces the earlier per-chunk retrieval design. It refines the rough plan in `specs-v1.md`.

## 1. Idea and constraints

**Method.** Insert a fixed, context-independent vocabulary of low-meaning words and decorative markdown tokens into **one Bloom filter**. To compress a context, make one pass over it: split on whitespace and drop every token the filter reports as present. The same filter works for every context and every question.

```
Baseline:  LLM(full context + question)
Method:    full context --Bloom filter drop pass--> compressed context + question --> LLM
```

**Why this design.**
- The Bloom filter is the whole method, not a side index.
- Every parameter has a visible effect on the output.
- A false positive deletes a real content word, such as a number or a name. That links FPR directly to tokens and accuracy (report Q3).

**Constraints.**
- **Two Python files:** `ask_llm.py` (filter, compression, LLM, stats) and `experiments.py` (sweeps, evaluator, plots).
- **Model:** `gemini-3.5-flash-lite`. The model, prompt template and generation config are identical for the baseline and every compressed run.
- **Generality:**
  - No document-specific rules. The vocabulary is built only from fixed general-English lists.
  - Only the evaluator in `experiments.py` reads `assignment/answers/`. `ask_llm.py` never does.
- **Reproducibility:**
  - SEED = 580 for hashing and for the LLM.
  - Hashing is deterministic; Python's salted `hash()` is never used.
  - The LLM responses are cached on disk.
  - One command reproduces everything.
- **Bloom filter is our own.** Hash functions come from `hashlib`; no Bloom filter library is used.

## 2. Files

| File | Contents |
|---|---|
| `ask_llm.py` | vocabulary lists and tiers, `BloomFilter`, `compress`, prompt, Gemini client and cache, stats, `run()` and its CLI |
| `experiments.py` | data-file validation, evaluator (the only reader of `answers/`), stats-only grid, LLM grid, summary, plots, CLI |
| `requirements.txt` | pinned: `google-genai`, `matplotlib`, `numpy`, `pympler`, `wordfreq` |
| `results/` | committed outputs: `stats_grid.json`, `llm_runs.json`, `summary.csv`, `plots/*.png` |
| `cache/` | git-ignored LLM and `count_tokens` cache |
| `tests/` (not submitted) | `conftest.py` (puts the repo root on `sys.path`), `test_ask_llm.py`, `test_experiments.py`. See §12 |
| `requirements-dev.txt` (not submitted) | `-r requirements.txt` plus `pytest` |

## 3. Vocabulary (`ask_llm.py`)

All lists are hard-coded constants, so there is no download step. `build_vocabulary(tier) -> list[str]` returns a sorted, de-duplicated list. **N** is its length.

### 3.1 Base lists

**`NLTK_STOPWORDS`**: NLTK's 179 English stopwords, copied verbatim.

**`PROTECTED`**: words that are never inserted, because deleting them can change an answer. These rules are generic, not tuned to the provided documents.

| Group | Words |
|---|---|
| Negations, with their NLTK contraction stubs | `no, nor, not, never, without, don, don't, ain, aren, aren't, couldn, couldn't, didn, didn't, doesn, doesn't, hadn, hadn't, hasn, hasn't, haven, haven't, isn, isn't, mightn, mightn't, mustn, mustn't, needn, needn't, shan, shan't, shouldn, shouldn't, wasn, wasn't, weren, weren't, won, won't, wouldn, wouldn't` |
| Comparatives and quantities | `more, most, less, least, few, only, same, very, than, all, any, each, both, some` |
| Order and position | `over, under, above, below, before, after, during, until, again` |
| Single-letter and contraction fragments (these appear as units, as in `20 m` or `5 s`) | `s, t, d, m, o, y, ll, re, ve, ma` |
| Number and scale words | `zero … twenty, thirty, forty, fifty, sixty, seventy, eighty, ninety, hundred, thousand, million, billion, trillion, half, dozen, first … tenth` |
| Months and weekdays | `january … december, monday … sunday` (this includes `may` and `march`) |

**`DECORATIVE`**: `#, ##, ###, ####, *, **, -, --, ---, >, •, _`. The table separator `|` is **kept**, so table rows keep their columns.

**`CURATED`** = `NLTK_STOPWORDS − PROTECTED`, which is **108 words**:
```
i me my myself we our ours ourselves you you're you've you'll you'd your yours yourself
yourselves he him his himself she she's her hers herself it it's its itself they them their
theirs themselves what which who whom this that that'll these those am is are was were be
been being have has had having do does did doing a an the and but if or because as while of
at by for with about against between into through to from up down in out on off further
then once here there when where why how other such own so too can will just should
should've now
```

### 3.2 Tiers (`--tier`)

| Tier | Contents | Purpose |
|---|---|---|
| `curated` | `CURATED ∪ DECORATIVE` (N ≈ 120) | the core method |
| `nltk_full` | `NLTK_STOPWORDS ∪ DECORATIVE` (N ≈ 191) | ablation: at almost the same memory as `curated`, measures what deleting negations, comparatives and units costs |
| `common500`, `common1000`, `common2000`, `common5000` | `curated ∪` the top-n words from `wordfreq.top_n_list('en', n)`, keeping only ASCII-alphabetic words not in `PROTECTED` | more aggressive compression, and larger filters |

- The tiers are the second tuning knob beside R and k.
- **Tier controls removal on purpose:** a larger vocabulary drops more tokens by design.
- **R and k control removal by accident:** false positives drop content words.
- **Bits needed grow with N.** For a fixed FPR p, `R ≈ N·log₂(1/p)/ln 2`. So a bigger filter "buys" a bigger vocabulary and fewer tokens, possibly at the cost of accuracy (report Q5).
- The `wordfreq` version is pinned, so the lists stay stable.

## 4. Bloom filter (`ask_llm.py`)

```python
class BloomFilter:
    def __init__(self, size: int, hash_count: int)      # size = R bits, hash_count = k
    def add(self, word: str) -> None
    def __contains__(self, word: str) -> bool
    @classmethod
    def from_words(cls, words, size, hash_count) -> "BloomFilter"
    @property
    def memory_bits(self) -> int                          # R
    @staticmethod
    def theoretical_fpr(n, size, hash_count) -> float     # (1 - e^{-kN/R})^k
    @staticmethod
    def optimal_k(n, size) -> int                         # max(1, round((R/N) ln 2))
```

- **Storage:** `bytearray((R + 7) // 8)`. Bit `p` is `byte p >> 3`, mask `1 << (p & 7)`.
- **Hashing (Kirsch–Mitzenmacher double hashing):**
  - One call per word: `blake2b(word.encode(), digest_size=16, key=SEED_BYTES)`.
  - Split the digest into two 64-bit integers h1 and h2, and force h2 odd.
  - The i-th position (i = 0 … k−1) is `(h1 + i·h2) mod R`.
  - So the cost is one hash per word whatever k is, and positions are identical across processes and machines.
- **No false negatives:** every inserted word must test as present. This is a sanity check.

## 5. Compression (`ask_llm.py`)

`compress(text, bf, exact=None) -> (compressed_text, stats)`

1. Process the text line by line, and split each line on whitespace.
2. **Lookup key:** the token lowercased, with leading and trailing punctuation stripped (`string.punctuation` plus `“”‘’—–…`). If stripping leaves nothing, the key is the raw token, so `##` and `---` still match.
3. **Drop** the token if its key is in `bf`. The whole token goes, including any attached punctuation.
4. Rejoin the kept tokens of each line with single spaces, keep the newlines, and collapse runs of blank lines into one.
5. **Memoization:** remember the membership result per distinct key during a pass, so each distinct word is hashed once.
6. **FPR accounting:** if `exact` (a Python `set` of the vocabulary) is given, also record:
   - `distinct_nonmembers`: distinct keys queried that are not in `exact`
   - `false_positives`: how many of those the filter reported as present
   - `empirical_fpr` = `false_positives / distinct_nonmembers`. This is comparable to the theoretical value, since each is one query per non-member.
   - `fp_tokens_removed`: token occurrences deleted only because of a false positive, i.e. the damage to the context
7. **Baseline mode** skips compression (the identity), but runs through the same pipeline and prompt.

Compression does not depend on the question, so each context is compressed **once per filter config** and reused for all of its questions.

## 6. Prompt and LLM (`ask_llm.py`)

### 6.1 One prompt, no system instruction

Gemini accepts two kinds of input text:
- A **system instruction** (`config.system_instruction`) holds standing rules, sent separately from the conversation. The model gives it higher priority than the user turn.
- The **prompt** (`contents`) is the user turn for one call.

Both count as input tokens. However, `count_tokens(contents=…)` sees only the prompt, so a system instruction would make stats-only token counts differ from the real `prompt_token_count`.

We therefore put **everything in one prompt** and use no system instruction. Stats-only counts are then exact, because `count_tokens` sees the same text that is sent.

### 6.2 Template

`build_prompt(context: str, question: dict) -> str` fills a single template. The template is identical for the baseline and every compressed run, and only `{context}` differs between them.

```
You answer a question about a document using only the context below.
Some common words may have been removed from the context to shorten it.
Reply with only the answer: no explanation and no full sentence.

Context:
{context}
End of context.

Question: {question}
Answer type: {answer_type}. {type_hint}
Unit: {unit}
Answer in at most {max_answer_tokens} tokens.
Answer:
```

**Fields from the questions file:**

| Field | Present in | Use |
|---|---|---|
| `question` | 80 / 80 | the `Question:` line |
| `answer_type` | 80 / 80 (number 50, entity 19, phrase 8, date 3) | the `Answer type:` line, plus a generic hint (below) |
| `unit` | 47 / 80 | the `Unit:` line, omitted entirely when absent |
| `max_answer_tokens` | 80 / 80 | a stated length limit, not the generation cap (see below) |
| `qid`, `number` | 80 / 80 | **not used**: they carry no meaning, and `qid` reveals the difficulty tier |

**`TYPE_HINTS`** (generic; an unknown type gets no hint):

| Type | Hint |
|---|---|
| `number` | Give a single number with its unit and any scale word (such as million), as written in the context. |
| `entity` | Give the name of the person, place, organisation or thing, as written in the context. |
| `phrase` | Give a short phrase, worded as in the context. |
| `date` | Give the date or year, as written in the context. |

**Why ask for unit and scale words:** grading parses the answer's number. For example, `$250 million` has no bare `250` alias, so the answer `250` would be graded wrong.

**The "common words may have been removed" line** appears in every run, including the baseline, so the template never changes. It warns the model that compressed text reads telegraphically.

### 6.3 Generation

**Generation config:**
- model `gemini-3.5-flash-lite`, temperature 0, seed 580, no system instruction
- `max_output_tokens = 64`, fixed for all runs. This is deliberately not `max_answer_tokens`:
  - Some gold answers are longer than 8 tokens, e.g. "2 hours, 4 minutes and 58 seconds".
  - Thinking tokens may count against the cap.
- **Check in the stage-1 smoke test:** whether thinking tokens consume the output budget. If they do, set the thinking config to its minimum and record it in the config.

**Retries:** exponential backoff on 408, 429, 500, 502, 503 and 504.

**Cache:**
- A JSON file per call under `cache/llm/`, keyed by `sha256(model | generation config | prompt)`.
- Cached results are flagged `cached: true`, because their latency is not a fresh measurement.
- `count_tokens` results are cached under `cache/tokens/`, keyed by `sha256(model | text)`.

**Token counts:**
- In `stats_only` mode: `client.models.count_tokens(model, contents=prompt)` (free), on the exact prompt that would be sent.
- Otherwise: `usage_metadata.prompt_token_count` from the real call.
- Both cover the full prompt: instructions, context and question.

**Client injection:** `run()` takes a `client` argument (default: the Gemini client). Tests pass a fake client, so they never touch the network.

**API key:** `GEMINI_API_KEY` from the environment or `.env`. `.env.example` documents it.

## 7. `ask_llm.py` interface

```python
def run(contexts: list[str], questions: list[str], size: int, hash_count: int,
        tier: str = "curated", stats_only: bool = False, calc_fpr: bool = False,
        baseline: bool = False, out: str | None = "response.json",
        client: LLMClient | None = None) -> dict          # None -> Gemini client
```

```bash
python ask_llm.py --contexts C1.txt C2.txt --questions Q1.json Q2.json \
    --size 4096 --hash-count 6 --tier curated [--stats-only] [--calc-fpr] [--baseline] [--out response.json]
```

- `contexts[i]` pairs with `questions[i]`; mismatched lengths are an error.
- `--baseline` ignores `--size`, `--hash-count` and `--tier`.
- The spec-v1 flag `calc_frp` is spelled `calc_fpr`.

### 7.1 Output (`response.json`)

```json
{
  "config": {"baseline": false, "tier": "curated", "size": 4096, "hash_count": 6,
             "model": "gemini-3.5-flash-lite", "seed": 580, "stats_only": false,
             "generation_config": {"temperature": 0.0, "seed": 580, "max_output_tokens": 64}},
  "filter": {"n_words": 120, "memory_bits": 4096, "bf_asizeof_bytes": 0, "set_asizeof_bytes": 0,
             "build_s": 0.0, "theoretical_fpr": 0.0},
  "docs": [{
    "doc_id": "dwh_dossier_v1",
    "words_in": 0, "words_out": 0, "chars_in": 0, "chars_out": 0, "retained_frac": 0.0,
    "compress_s": 0.0, "context_tokens": 0,
    "fpr": {"distinct_nonmembers": 0, "false_positives": 0, "empirical_fpr": 0.0, "fp_tokens_removed": 0},
    "questions": [{"qid": "dwh-T1-001", "prompt_tokens": 0, "token_source": "usage|count_tokens",
                   "max_answer_tokens": 8,
                   "answer": "…", "output_tokens": 0, "thoughts_tokens": 0,
                   "llm_s": 0.0, "latency_s": 0.0, "cached": false}]
  }]
}
```

- `fpr` and the `asizeof` fields are present only when `calc_fpr` is on.
- `answer`, `output_tokens`, `thoughts_tokens`, `llm_s` and `cached` are null when `stats_only` is on.
- In baseline mode, `filter` is null and `tier`, `size` and `hash_count` are null.
- `retained_frac` is `words_out / words_in`.

### 7.2 Metrics

| Metric | Definition |
|---|---|
| Memory | `memory_bits` (R), and the measured `pympler.asizeof` of the filter vs. the exact `set` of the same words |
| FPR | empirical (§5.6) vs. theoretical `(1 − e^{−kN/R})^k` |
| Context retained | `retained_frac`, and `context_tokens` vs. the baseline's |
| Input tokens | `prompt_tokens` per question, as a mean and a total over the 80 questions |
| Latency | per question: `latency_s = compress_s + llm_s`. The filter build time `build_s` is reported separately, because it is a one-time cost shared by all contexts |
| Accuracy | fraction of the 80 questions graded correct (§8) |

## 8. Evaluator (`experiments.py`)

This is the only code that reads `assignment/answers/<doc_id>_answers.json`. The answers are matched to questions by `qid`.

- **`numeric`:**
  - Parse the first number in the answer. Handle thousands commas, decimals, signs, `%`, scale words ("13.3 million"), currency scales ("$250m"), and number words.
  - Correct if it is within `numeric_tolerance_rel` of the number parsed from `expected` or from any alias.
  - Units are not enforced, because the aliases include bare numbers.
- **`exact_ci`:** case-insensitive equality with `expected` or any alias, after normalization (NFKC, casefold, trimming punctuation and whitespace, collapsing spaces).
- **Breakdowns:** accuracy is reported overall, per document, and per `tier` (`T1_lookup`, `T2_multihop`, `T3_aggregate`).
- **Answer-length diagnostic:** `over_limit_answers` counts answers whose `output_tokens` exceed the question's `max_answer_tokens`. It is reported, not penalized.
  - In the baseline, 10 of 80 answers ran over, and 9 of those were correct gold-format answers (e.g. `13,300,000 ft` is 11 tokens). That confirms the decision not to cap generation at `max_answer_tokens`.
- **Reuse:** port `parse_number` and the normalizers from the old `src/evaluate.py` (`git show HEAD:src/evaluate.py`), simplified.

## 9. Experiments (`experiments.py`)

```bash
python experiments.py --mode stats_only|llm_only|all --contexts C… --questions Q… --answers A… [--out results/]
```

- **All data files are required flags, paired by position.** Nothing is discovered by scanning folders, so the TAs run exactly the files they pass, including held-out ones.
- **`load_dataset` validates the pairing before any API call:**
  - every file exists, and the three lists have equal lengths
  - each answers file has the same `doc_id` as its questions file and covers every `qid`
  - Context files have no `doc_id`, so they are paired purely by position.
- **The TA command** is `python experiments.py --mode all` with the four v1 files listed explicitly. The README has the full command.
- The experiments call `ask_llm.run()` directly.

### 9.1 Stats-only grid (no LLM calls, only `count_tokens`)

- **Grid:** tier ∈ {6 tiers} × R ∈ {2^8, 2^9, …, 2^17} × k ∈ {1, 2, 3, 4, 6, 8, k_opt}, with `calc_fpr` on.
- **Records:** memory, theoretical vs. empirical FPR, `fp_tokens_removed`, `retained_frac`, and context tokens.
- **k cap:** k_opt is capped at 16 (`MAX_HASH_COUNT`). Beyond that the FPR is already about 0, and lookups only get slower.
- **No `run()` here:** the grid calls `ask_llm.compress` directly, so each filter is built once and each document is compressed once per configuration.
- **Token counting:** input tokens are estimated as `context_tokens(doc) + overhead(question)`, summed over all questions.
  - `overhead` is the question's prompt with an empty context.
  - This needs one `count_tokens` call per compressed context instead of one per prompt.
  - Checked on the 80 baseline prompts, the estimate is exactly **1 token above** the true `prompt_token_count` every time.
- **Empirical FPR:** measured on the union of distinct lookup keys of all documents that are not in the tier's vocabulary (about 10,000 keys). Each non-member is queried once, as in the theory, so the resolution is about 10⁻⁴.
- **Output:** `results/stats_grid.json`.

### 9.2 LLM grid (13 runs × 80 questions)

The filter removes tokens for two separate reasons, and each group isolates one of them:

| | Removal on purpose | Removal by accident |
|---|---|---|
| What is removed | the words we chose (the vocabulary tier) | content words the filter wrongly matches (false positives) |
| What controls it | the tier, i.e. N | R and k, through the FPR |
| Which words get hit | predictable ("the", "of", "said") | random ("13,300,000", "Glaevek") |
| Measured by | Group 2, with FPR held near 0 | Group 3, with the vocabulary held fixed |

Changing both at once would make an accuracy drop impossible to attribute.

1. **Group 1, the full-context baseline (1 run).** It is run first and cached.
2. **Group 2, the vocabulary tiers (6 runs).**
   - Each tier runs at the smallest power-of-two R with k = k_opt whose theoretical FPR is ≤ 1%. Bits needed grow with N, so R grows with the tier.
   - **Question:** with a near-perfect filter, how much can be deleted on purpose before accuracy drops? This answers report Q5 (memory buys vocabulary, vocabulary buys fewer tokens).
   - `nltk_full` is an ablation, not a ladder step: it has the same R as `curated` and differs only by the protected words.
3. **Group 3, the FPR sweep (6 runs).**
   - Tiers `curated` (the core method) and `common1000` (mid-ladder), each with R chosen for theoretical FPRs of about 50%, 20% and 5% at k_opt: `R = ⌈−N ln p / (ln 2)²⌉`.
   - Each tier's Group 2 run is the ≈0% end of its curve.
   - **Question:** what does saving filter memory cost? This answers report Q3.
   - Sweeping two tiers shows whether false-positive damage depends on how much was already removed on purpose.

**Final configs (decided before stage 5):**

| Run | Group | Tier | N | R (bits) | k | Theoretical FPR | Tokens kept (grid) |
|---|---|---|---|---|---|---|---|
| `baseline` | 1 | — | — | — | — | — | 100% |
| `tier_curated` | 2 | curated | 120 | 2,048 | 12 | 0.03% | 73% |
| `tier_nltk_full` | 2 | nltk_full | 191 | 2,048 | 7 | 0.58% | 70% |
| `tier_common500` | 2 | common500 | 464 | 8,192 | 12 | 0.02% | 66% |
| `tier_common1000` | 2 | common1000 | 930 | 16,384 | 12 | 0.02% | 61% |
| `tier_common2000` | 2 | common2000 | 1,905 | 32,768 | 12 | 0.03% | 55% |
| `tier_common5000` | 2 | common5000 | 4,848 | 65,536 | 9 | 0.15% | 47% |
| `fpr50_curated` | 3 | curated | 120 | 174 | 1 | 50% | — |
| `fpr20_curated` | 3 | curated | 120 | 402 | 2 | 20% | — |
| `fpr5_curated` | 3 | curated | 120 | 749 | 4 | 5% | — |
| `fpr50_common1000` | 3 | common1000 | 930 | 1,342 | 1 | 50% | — |
| `fpr20_common1000` | 3 | common1000 | 930 | 3,116 | 2 | 20% | — |
| `fpr5_common1000` | 3 | common1000 | 930 | 5,799 | 4 | 5% | — |

The exact configs are derived from the formulas above, not hand-picked. The stats-only results may justify adjusting them, and any change is recorded here.

**Output:** `results/llm_runs.json` (every run's `response.json` plus grades) and `results/summary.csv`. The summary has one row per run: tier, R, k, N, memory, theoretical and empirical FPR, retained %, mean and total input tokens, mean latency, and accuracy (overall and per tier).

### 9.3 Plots (`results/plots/*.png`)

**Required:**
1. **BF memory vs. FPR:** log-log, empirical points and theoretical curves, one color per tier, at k_opt.
2. **BF memory vs. LLM input tokens:** one line per tier. LLM runs use their real tokens; the grid uses `count_tokens` estimates.
3. **LLM input tokens vs. accuracy:** the LLM runs, with the baseline marked.

**Extras:**
- empirical vs. theoretical FPR for every (R, k, tier), with the y = x line
- FPR vs. k at fixed R, which shows the optimum near k_opt
- retained % per tier
- accuracy by question tier (T1, T2, T3) for each run

## 10. Development stages

Every stage includes unit tests for its code (§12). `pytest` must pass before the next stage starts.

1. **`ask_llm.py` in baseline mode:**
   - load the files and build the prompt
   - add the Gemini client, retries and cache, and the token, latency and stats recording
   - write `response.json`
   - smoke-test it on 2 questions
2. **Evaluator, then baseline:** build the evaluator, run the full baseline over all 80 questions, cache it, and grade it. This checks the pipeline end to end.
3. **Bloom filter:**
   - `BloomFilter`, the vocabulary tiers, `compress`, and the FPR accounting
   - sanity checks: no false negatives; the same positions across processes; empirical FPR matches the theory; compression is the identity when nothing matches
4. **Stats-only grid** and its plots. Use them to confirm the LLM configs in §9.2.
5. **LLM grid**, then the final summary and plots.
6. **Write-up:** README (setup and exact commands) and the report.

## 11. How the results answer the report questions (§4 of the assignment)

1. **Measured vs. theoretical FPR:** plot 1 and the FPR scatter, from the stats-only grid.
2. **Effect of R and k on FPR and memory:** the grid, plus FPR vs. k at fixed R, plus `asizeof` of the filter vs. the exact set.
3. **Effect of false positives on context, tokens and accuracy:** `fp_tokens_removed` and the FPR sweep (§9.2 item 3).
4. **Token reduction at near-baseline accuracy:** plot 3. Find the cheapest run within a few points of the baseline's accuracy.
5. **Is a larger filter always better?** The tier runs: more memory holds a larger vocabulary, which means fewer tokens but eventually lower accuracy. The `nltk_full` vs. `curated` ablation adds to this.
6. **Limitations:**
   - The filter is blind to the question.
   - Only single words can be removed.
   - Prose compresses, but tables and numbers don't.
   - False positives fall on random words.
   - The vocabulary is English-only.

## 12. Testing (development aid, not submitted)

Unit tests live in `tests/` and run with `.venv/bin/python -m pytest -q`.

**Rules:**
- Every function in the two scripts gets tests.
- Tests never use the network: the LLM is a fake client passed to `run(client=…)`, and `count_tokens` is faked the same way.
- Tests use tiny inline texts and filters. The real `assignment/` files are used only where a test says so.

**`tests/test_ask_llm.py`:**

| Area | Tests |
|---|---|
| `BloomFilter` | no false negatives; positions are deterministic, including across a subprocess; empirical FPR is close to the theoretical value on a large probe set; `optimal_k` and `theoretical_fpr` known values; `memory_bits` and the byte-array size |
| Vocabulary | `CURATED` has 108 words; no `PROTECTED` word appears in any tier; tiers are nested (`curated ⊂ common500 ⊂ common1000 ⊂ …`); `nltk_full` ⊇ `curated`; the output is sorted and deterministic |
| `compress` | lookup-key normalization (case, edge punctuation, `##`); newlines are kept and `\|` survives; the identity when nothing matches; FPR accounting on a hand-built filter with a known false positive |
| `build_prompt` | every relevant field appears; the `Unit:` line is omitted when there is no unit; `qid` never appears; an unknown `answer_type` gets no hint; only `{context}` differs between the baseline and compressed runs |
| `run()` | stats-only with a fake client gives the §7.1 schema and makes 0 generate calls; mismatched context/question lists raise an error; cache hits are flagged `cached: true` |

**`tests/test_experiments.py`:**

| Area | Tests |
|---|---|
| `parse_number` | thousands commas, decimals, `%`, scale words ("13.3 million"), currency scales ("$250m"), number words |
| Grading | numeric tolerance at and beyond the edge; `exact_ci` against aliases; case and whitespace normalization |
| Gold isolation | `ask_llm.py` never references `answers` |
| Grid selection | the §9.2 rule picks the smallest R with theoretical FPR ≤ 1% at k_opt |

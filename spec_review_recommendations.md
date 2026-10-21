1# Recommendations for Revising the Bloom-Filter Context Compression Specification

## Overall assessment

The specification has several strong ideas: a custom Bloom filter with deterministic hashing, comparison of empirical and theoretical false-positive rates (FPRs), a full-context baseline using the same LLM setup, an exact-set control, per-question result records, and a reproducible experiment pipeline. The main concern is scope: the current plan describes a fairly elaborate retrieval research prototype, which could be difficult to implement, debug, and evaluate reliably before the assignment deadline.

I recommend retaining the core experimental design while simplifying the retrieval method and resolving the correctness issues below before implementation.

## 1. Critical correctness issues

### 1.1 Make sure retrieval genuinely uses Bloom-filter membership

**Specification location:** Section 6, especially the definition of `m(q, c)`.

The current scoring rule appears to give a main word match when `w:q` is in the chunk's exact term set. If so, the retrieval path is consulting the source text's exact membership rather than the Bloom filter. A Bloom false positive would not then contribute to this primary score, weakening the experiment's ability to measure the effect of approximate membership.

**Recommendation:** Define the primary match as the result of querying the relevant chunk's Bloom filter:

\[
m(q,c)=\mathbf{1}[\operatorname{BloomQuery}(q,c)=\text{positive}].
\]

Keep exact term sets out of the retrieval path. Use them only for evaluation and diagnostics. The `ExactSetIndex` should expose the same interface but replace Bloom queries with exact membership; this makes the comparison between approximate and exact retrieval interpretable. Specify similarly whether character n-gram coverage is computed from Bloom queries or exact chunk terms.

This is the most important algorithmic change.

### 1.2 Make configuration selection consistent with the anti-overfitting policy

**Specification location:** Sections 9 and 10.

The spec says constants are fixed a priori and that tuning should use leave-one-document-out gold-span recall rather than test accuracy. However, the ablation procedure says to keep each layer only when it improves accuracy or recall. If accuracy on the same 80 evaluation questions determines which layers survive, this conflicts with the stated anti-overfitting policy.

**Recommendation:** Choose one clear protocol and state it consistently. A simple option is to freeze the core method and its constants before the final evaluation, run ablations as reported comparisons, and avoid selecting the final configuration based on the same test accuracy that is later presented as an unbiased result. If any tuning is done, state what data or diagnostic it uses and separate that from the final reported evaluation.

The ablation ladder also bundles several changes into each layer. For example, one layer adds structural chunking, breadcrumbs, table headers, and neighbours; another adds bigrams, numeric features, IDF weighting, and a new threshold. If a result changes, it will be difficult to identify which addition caused it. Keep the first ablation ladder small and make each comparison interpretable.

### 1.3 Encode the exact answer schema and grading semantics in the specification

The agent that generated `specs.md` has access to the exact answer-file format. Therefore, the schema should not be treated as unknown or left as an assumption. The specification should explicitly document the actual fields and their semantics, including any `answer_type`, `unit`, `aliases`, numerical tolerance, and `gold_spans` fields that are present.

The proposed evaluator is a useful starting point, but it needs to match the supplied grading rules precisely:

- **Units:** An answer with the same numeric value but an incompatible unit must not automatically count as correct. Specify any permitted unit normalization or conversion.
- **Tolerance:** Confirm whether the data uses relative tolerance, absolute tolerance, or another rule. The proposed condition \(|a-g| \leq \mathrm{tol}|g|\) gives zero tolerance when \(g=0\), so it is only suitable if that matches the provided semantics.
- **Parsing:** Define what happens when an answer cannot be parsed or contains multiple numbers. Do not silently grade it using whichever number happens to appear first unless that is the intended rule.
- **Aliases and exact answers:** Test the implementation against actual answer records and their documented aliases.
- **Gold spans:** If the exact schema supplies `gold_spans`, retain gold-span recall as a retrieval diagnostic and document how it is computed. If those annotations are absent, do not assume they exist; use only fields actually supplied.

The key recommendation is not to remove these features pre-emptively, but to use the exact schema available to the spec-generating agent and state it explicitly so the implementation and report cannot diverge from it.

### 1.4 Simplify and accurately describe the hash-position construction

**Specification location:** Section 5, hashing bullet.

The spec calls the following construction Kirsch–Mitzenmacher double hashing:

\[
g_i=(h_1+i h_2+i^2)\bmod R.
\]

The usual double-hashing form is \(g_i=(h_1+i h_2)\bmod R\). The quadratic term is a departure from that standard formula, and it does not by itself guarantee that all \(k\) positions are distinct.

**Recommendation:** Prefer standard double hashing and explain any implementation constraints directly. Since the proposed sweep uses power-of-two values of \(R\), making the step \(h_2\) odd ensures it is relatively prime to \(R\), so the sequence cycles through all \(R\) positions before repeating. This gives a straightforward construction. Compare empirical and theoretical FPRs as an approximation; do not present the formula as an exact prediction for every concrete hash realization.

### 1.5 Correct bit-sliced memory accounting and test bit order

**Specification location:** Section 5.1.

For a `uint64` matrix of shape \((R,\lceil n/64\rceil)\), storage is

\[
8R\lceil n/64\rceil \text{ bytes},
\]

whereas \(n\) independent filters with \(R\) bits each use \(nR/8\) bytes before other overhead. These differ when the number of chunks is not divisible by 64 because the bit-sliced representation pads each slice group.

The build rule places chunk \(c\) in bit position \(c\bmod64\). When unpacking bitmap words to produce per-chunk scores, the implementation must preserve that bit order; a default `np.unpackbits` order may not align with the least-significant-bit-first convention.

**Recommendation:** Compute memory from the actual array allocation, account for padding explicitly, and add a test that compares the complete per-chunk scoring vector from bit-sliced retrieval with the scores obtained from independent Bloom filters. Testing only membership bitmaps is not enough to rule out an ordering bug in the scoring path.

### 1.6 Enforce the token budget after context assembly

**Specification location:** Section 6, selection and context assembly.

The procedure ranks chunks and caps the selection at `budget_tokens`, but then adds neighbours, breadcrumbs, and table headers. Consequently, the final assembled context can exceed the stated budget.

**Recommendation:** Define whether `budget_tokens` applies to the core ranked chunks or the final assembled context. Prefer enforcing it on the final context, including metadata and headers, if it is intended to be a hard budget. Define deterministic behavior for:

- A question with no usable terms.
- A top score of zero.
- Every chunk passing the relative threshold.
- Ties at the selection boundary.
- A neighbour or table header that would cause the selection to exceed the budget.

The confidence ratio `top_score / max_possible_score` is a heuristic, not a calibrated probability. Specify what happens if its denominator is zero and describe the quantity as a heuristic retrieval-confidence score.

## 2. Important experimental-design concerns

### 2.1 Separate Bloom-filter FPR from retrieval and QA errors

The measured fraction of absent terms that return positive membership queries is a valid Bloom-filter FPR measurement. It is not the same as the fraction of irrelevant chunks retained or the probability of answering a question incorrectly.

Report these as separate metrics:

1. **Membership FPR:** measured against the theoretical approximation.
2. **Retrieval diagnostics:** such as false-positive term/chunk matches and the effect of the Bloom index relative to the exact-set control.
3. **End-to-end outcomes:** retained-context tokens, LLM input tokens, latency, and answer accuracy.

The exact-set control is especially useful because it isolates the impact of approximate membership under an otherwise identical retrieval algorithm.

### 2.2 Make the memory metric precise

The specification usefully distinguishes Bloom bits from auxiliary index storage, but it should state consistently whether measurements include section filters, bit-sliced padding, chunk metadata, cached hash positions, and temporary scoring arrays. `ndarray.nbytes` measures array-buffer storage, not the total memory of every Python object reachable from an array.

At minimum, report **raw Bloom-filter storage** and **persistent auxiliary index storage** separately. Report total index memory only when it is measured using a clearly defined and reproducible method. Report `ExactSetIndex` memory separately; do not count it as Bloom-filter memory.

### 2.3 Make latency measurements comparable

Report index-build time, per-question retrieval time, LLM generation time, and end-to-end latency separately. Also provide an amortized end-to-end measurement that allocates one index build across the 20 questions associated with a context.

For fresh API timing, use uncached calls and record retry/backoff behavior consistently. A cached result's original latency is useful for record-keeping, but it is not a new latency observation. API latency varies across requests even with identical inputs.

### 2.4 Keep statistical claims appropriately qualified

Paired comparisons are valuable because each configuration is evaluated on the same questions. McNemar's test can be appropriate for paired correctness outcomes. Confidence intervals help show uncertainty, but a fixed statement that the standard error is approximately five percentage points is only a rule of thumb and depends on the observed accuracy. Questions belonging to the same document may also be correlated.

Prioritize per-question records, overall and per-document accuracy, and a confidence interval. Additional significance tests are useful only if time remains. If testing many configurations, label the comparisons as exploratory or address the multiple-comparison issue.

## 3. Scope and implementation strategy

The proposed system includes structure-aware chunking, four term types, a bit-sliced index, exact-set controls, seven retrieval layers, hierarchical filtering, a feedback hop, adaptive fallback, several parameter sweeps, statistical tests, and numerous diagnostic plots. Each component is plausible, but implementing and validating all of them under the deadline introduces substantial risk.

### Keep in the core implementation

- A custom Bloom filter with deterministic hashing and tests for no false negatives.
- Theoretical and measured FPR analysis.
- One general-purpose chunking method.
- Bloom-query-based retrieval and a clearly enforced context budget.
- The same model, prompt template, and generation settings for full-context and compressed-context runs.
- The exact-set control, if feasible, to diagnose Bloom-induced retrieval effects.
- Per-question saved results, measured token counts, latency, memory accounting, and all required plots.
- A reproducible command and a README with setup and execution instructions.

### Defer until the core evaluation works

- The feedback hop, section-level hierarchy, and adaptive fallback.
- The bit-sliced index, unless profiling shows that retrieval speed is a meaningful bottleneck. If retained, implement a simple row-major reference first and measure rather than assume a speed-up.
- A large, bundled seven-layer ablation ladder.
- Extra diagnostic plots and extensive significance tests beyond those needed to support the report.

A simple implementation that is correct and fully evaluated is preferable to a more elaborate implementation whose behavior is difficult to debug or whose experiments cannot be completed.

## 4. Recommended experiment plan

### Stage 1: Correctness first

1. Implement and test Bloom insertion and membership queries.
2. Verify no false negatives, deterministic hashing across processes, and memory accounting.
3. Compare measured and theoretical FPR using a reproducible set of absent-term probes.
4. Validate the full-context baseline, prompt construction, token accounting, and evaluator on a small number of examples.
5. Confirm that retrieval decisions really use Bloom membership, not the exact term set.

### Stage 2: Complete the required evaluation

1. Run the full-context baseline for all 80 required question/context pairs.
2. Select a manageable grid of \((R,k)\) settings spanning a meaningful range of empirical FPRs.
3. Run the compressed method on the same 80 pairs for each chosen configuration.
4. Include the exact-set control if the remaining time and API budget allow it.
5. Save raw answers and per-question metrics as the runs complete, so interrupted experiments do not lose results.

A nine-point parameter grid can be reasonable. However, avoid committing simultaneously to every ablation layer, a broad budget sweep, and all optional diagnostics before the main evaluation is complete.

### Stage 3: Analyze and submit

1. Generate the three required plots from saved results: Bloom memory versus FPR, Bloom memory versus LLM input tokens, and input tokens versus answer accuracy.
2. Report token savings and accuracy difference relative to the full-context baseline.
3. Answer the six report questions using observed measurements, not assumed monotonic trends.
4. Verify that the documented command reproduces the analyses and plots, and check that every one of the 80 question records is accounted for.

Use the same question set for every configuration so comparisons are paired. If resources become constrained, evaluate fewer configurations fully rather than leaving the principal experiment incomplete.

## 5. Code structure and reproducibility

Separating Bloom filters, retrieval, LLM calls, evaluation, and plotting is good engineering practice. However, the assignment says the submission should ideally contain no more than two Python files. Ten modules are not necessarily prohibited, but they add integration work and depart from the requested preference.

A practical compromise is:

- `bloom.py`: Bloom filter, hashing, and optionally the bit-sliced or exact-set index.
- `main.py`: chunking, retrieval, LLM calls, evaluation, experiments, result writing, and plots.

Keep concerns separated with classes and functions even if they live in two files. Preserve these reproducibility features:

- Deterministic hashing and fixed seeds where applicable.
- Saved configuration and version information for every run.
- Saved raw LLM responses and usage metadata, so aggregation and plotting do not require new model calls.
- A fresh-call mode for latency measurements, separate from cached development runs.
- Tests for Bloom correctness, evaluator behavior, and end-to-end record generation.
- One exact command in the README to reproduce the experiments.

One additional implementation detail: if the memoized hash-position cache is keyed by `(term, R, k)`, include the seed too, or ensure the cache is scoped to one seed. Otherwise, positions generated under one seed could be reused incorrectly under another.

## 6. Prioritized revision checklist

| Priority | Recommendation |
|---|---|
| **P0** | Make the retrieval score use Bloom-filter membership queries; keep exact membership for diagnostics only. |
| **P0** | Make the ablation/configuration-selection protocol consistent with the anti-overfitting policy. |
| **P0** | Use the exact answer schema already available to the spec-generating agent; document fields and grading rules explicitly, including units, tolerances, aliases, and gold spans if present. |
| **P1** | Simplify and correctly describe hash-position generation. |
| **P1** | Fix bit-sliced memory accounting and test bit order through the full scoring path. |
| **P1** | Specify how the final token budget, zero scores, empty queries, and ties are handled. |
| **P1** | Separate membership FPR, retrieval effects, and end-to-end accuracy in the reported metrics. |
| **P1** | Reduce the number of layers and experiments to a feasible, interpretable core study. |
| **P2** | Add hierarchy, feedback retrieval, adaptive fallback, more plots, and expanded statistical testing only after the core evaluation is complete. |

## Final recommendation

Freeze a simple, correct retrieval algorithm before adding features. The assignment's central question is whether Bloom-filter-based context compression can reduce LLM input usage while preserving answer quality. A smaller, fully reproducible system with interpretable experiments will make a stronger submission than an elaborate system whose components or results are difficult to validate.

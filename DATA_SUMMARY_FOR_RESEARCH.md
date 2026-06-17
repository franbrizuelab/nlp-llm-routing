# LLM Routing — Dataset Brief for Deep Research

**Purpose of this document:** a self-contained, highly detailed description of the
competition data so an external research assistant (e.g., Gemini Deep Research) can
find the best-performing strategies from papers and Kaggle/LLM-routing competitions.
Everything below is computed directly from the provided `train.csv` / `test.csv` /
`sample_submission.csv` (NVIDIA-sourced data, 2026 Spring NLP final project).

---

## 1. The task in one paragraph

Given a single user **query** (free text), choose exactly one of **11 candidate LLMs**
(`Model_A` … `Model_K`) to answer it. We never run the models — we only predict *which
model to route to*. Each candidate has a known **performance** (quality of its answer,
in `{0.0, 0.5, 1.0}`) and a known **cost** (continuous, ≈ token/$ proxy) on training
queries. The router must generalize this to unseen test queries. Scoring is a
cost-penalized reward, **Reward_{0.85}** (higher = better). The trade-off: capable models
cost more; cheap models fail more. The art is routing each query to the cheapest model
that will still answer it well.

---

## 2. Files & schema

| File | Rows | Columns | Contents |
|------|------|---------|----------|
| `train.csv` | **10,182** | 24 | `ID`, `query`, then `Model_X_performance` + `Model_X_cost` for X ∈ A…K (22 numeric cols) |
| `test.csv` | **2,550** | 2 | `ID`, `query` only — labels hidden |
| `sample_submission.csv` | 2,550 | 2 | `ID`, `pred_model` (a string like `Model_A`) |

- **No missing values** anywhere in train or test. No duplicate queries in train.
- `ID` is a contiguous integer index (train 1–10182, test 1–2550); the two ID spaces are
  independent (test is not a subset of train).
- **Submission format:** exactly 2,551 lines = 1 header (`ID,pred_model`) + 2,550 rows.
  `pred_model` must be one of the 11 strings `Model_A`…`Model_K`. Header and ID order must
  match `sample_submission.csv` exactly.
- **Performance** is **ternary**, not continuous: only the values `{0.0, 0.5, 1.0}` appear
  (each model column has exactly 3 unique values). 0.0 ≈ wrong/failed, 1.0 ≈ correct,
  0.5 ≈ partial (rare, ~1.3% of all cells). This means routing is effectively a
  *"which model will get this right"* classification problem, not regression.
- **Cost** is continuous and right-skewed (token-count / price proxy), strongly correlated
  with query length (Spearman ≈ 0.55 for the priciest model). A small fraction of cells
  have `cost == 0` (notably Model_K 1.7%, Model_F 1.5%) — likely empty/refused outputs;
  these almost never coincide with a correct answer, so treat cost==0 as "no real answer."

---

## 3. Scoring metric (Reward_{0.85}) — and the single most important strategic fact

**Official formula (from the Kaggle page):**

> **Reward_{0.85} = 0.85 · P̄ − 0.15 · (C̄ / C_max)**

where, over the *whole submission*: **P̄** = mean performance of the chosen models across
all routed queries, **C̄** = mean cost of the chosen models, and **C_max** is a fixed
normalizing constant (a maximum cost). It is a submission-level weighted average — **not** a
per-query `perf − λ·cost`. (The `0.85` subscript is the performance weight; cost gets the
complementary `0.15`.) The metric is **linear and therefore separable per query**: each
query's optimal choice can be decided independently by maximizing
`0.85·perf_m − (0.15/C_max)·cost_m`.

**The two facts that should drive the entire strategy:**

1. **Accuracy dominates; cost is a weak secondary term.** Performance carries weight 0.85
   and ranges over the full `{0,0.5,1}`, while the entire cost term is capped at 0.15 and,
   because realized costs are tiny relative to C_max, contributes only ~0.001–0.05 in
   practice. **Roughly ~90%+ of the achievable reward comes from getting the *quality*
   prediction right.** The router is first and foremost an *"which model answers correctly"*
   predictor; cost shaves the margins.

2. **The per-query optimal rule is simple and robust:** *pick the cheapest model among the
   top performers* (highest perf, ties broken by lowest cost). We verified this against the
   true formula under three different plausible definitions of `C_max` (global max cell cost,
   max model mean-cost, mean of per-query maxima): the rule agrees with the exact metric on
   **99.96%–100%** of training rows in every case. So **the routing target is invariant to
   the exact value of C_max** — only the *magnitude* of the cost penalty (and hence which
   constant baseline scores best, see §4) depends on it.

**Implication for modeling:** treat this as a per-query multi-label *"who can solve this
query"* problem, then apply the fixed cost ordering as a tie-breaker. The one remaining
unknown is the exact `C_max` value — it rescales the cost weight but does **not** change the
optimal routing decision, so it can be left for final-submission tuning.

**Reference reward levels on train** (computed with C_max = global max cell cost = 1.376;
the ranking shifts with C_max but the headroom story is the same):

- Best single-model constant policy ≈ **0.49** (always-F under weak penalty; always-K ≈ 0.45
  and becomes the best constant under a strong cost penalty).
- **Oracle (per-query optimal) ≈ 0.686** (P̄ = 0.808, C̄ = 0.007).
- → There is ~**0.19 of reward headroom** between the best constant baseline and the oracle.
  That gap *is* the routing opportunity; closing it is what beats the strong baseline.

---

## 4. The 11 models: cost tiers and capability

Median cost ascending (the de-facto cost ordering used for tie-breaking) with mean
performance across all training queries:

Median cost ascending, with mean performance and the reward of an "always this model"
constant policy (C_max = global max cell cost = 1.376):

| Model | Median cost | Mean perf | Always-this Reward | Notes |
|-------|------------|-----------|--------------------|-------|
| **Model_K** | 0.00024 | 0.532 | **0.4524** | **Cheapest AND 3rd-best** accuracy — best value |
| Model_D | 0.00038 | 0.435 | 0.3695 | cheap |
| Model_C | 0.00041 | 0.408 | 0.3469 | cheap |
| Model_J | 0.00060 | 0.436 | 0.3700 | cheap |
| Model_I | 0.00120 | 0.366 | 0.3110 | cheap, **lowest accuracy** |
| Model_E | 0.00125 | 0.425 | 0.3603 | |
| Model_G | 0.00257 | 0.423 | 0.3583 | |
| Model_B | 0.00480 | 0.472 | 0.3996 | mid |
| Model_A | 0.00653 | 0.383 | 0.3240 | mid cost, low accuracy (poor value) |
| **Model_H** | 0.01541 | 0.583 | 0.4917 | expensive, 2nd-best accuracy |
| **Model_F** | 0.02479 | **0.589** | **0.4936** | **most expensive AND most accurate** |

Key points for research framing:
- **The best *constant* policy is to always pick the most-accurate model (F ≈ 0.494, H ≈
  0.492), narrowly above always-K (≈ 0.452)** — *not* the cheapest. Under the weak cost
  penalty, accuracy wins. (If C_max turns out small enough to make the cost penalty strong,
  always-K becomes the best constant instead. Either way the headroom to the oracle ≈ 0.69
  is large.) The "simple vs. strong" Kaggle baselines almost certainly sit between the best
  constant (~0.49) and the oracle (~0.69).
- **Model_K is the standout value pick** (cheapest *and* 3rd-best accuracy): it is the
  correct tie-break choice for the ~51% of queries where it ties the top performers. The win
  comes from routing the queries K *fails* (~47%) up to F/H and keeping the rest on K.
- Capability is **not monotonic in cost**: A and I cost more than K but are less accurate
  (avoid them — strictly dominated).
- Mean pairwise correlation of performance across models is ≈ **0.56** (range 0.47–0.67):
  models largely succeed/fail together (shared query difficulty), but there is meaningful
  per-model specialization to exploit (~40% independent variance).

### Best-model distribution (reward-optimal label per training query)
This is the implicit target distribution a classifier would learn — **highly imbalanced**:

| Model | K | C | D | E | J | H | I | G | B | F | A |
|-------|---|---|---|---|---|---|---|---|---|---|---|
| #queries | **5172** | 1151 | 1087 | 680 | 600 | 405 | 224 | 211 | 208 | 365 | 79 |
| share | **50.8%** | 11.3% | 10.7% | 6.7% | 5.9% | 4.0% | 2.2% | 2.1% | 2.0% | 3.6% | 0.8% |

→ Class imbalance is severe (majority class ≈ 51%). Macro-F1 / class weighting / the cost
structure all matter. Caveat: this label is the cost-aware tie-break, so K is over-
represented (it wins most ties as cheapest). Predicting this exact label is *not* the same
as maximizing reward — because of ties, several different predictions can be reward-optimal
for the same query. A reward-weighted / multi-label objective is more faithful than plain
multiclass cross-entropy on this argmax label.

---

## 5. Query-difficulty structure (the routing signal)

Distribution of **how many of the 11 models solve a query** (perf == 1.0):

| # models solving | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| # queries | 1960 | 1186 | 907 | 676 | 617 | 477 | 460 | 447 | 433 | 488 | 752 | **1779** |

- **~19.1%** of queries are solved by **no model** (all-zero rows) — unanswerable/very hard.
  For these, the reward-optimal move is the **cheapest** model (minimize wasted cost), since
  no model earns the performance reward.
- **~17.5%** are solved by **all 11** (all-perfect, trivial) — here always pick the cheapest
  (Model_K) → free reward.
- Mean number of solving models ≈ **4.98**. The interesting/decisive ~63% of queries sit in
  between, where model choice actually changes the reward.
- At least one model solves **80.8%** of queries.

This trimodal difficulty (trivial / hard-but-solvable / impossible) is the core thing a
router must detect from text alone.

---

## 6. Query text statistics

### Length (characters), extremely right-skewed

| | mean | min | 10% | 25% | **median** | 75% | 90% | 95% | 99% | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **train** | 3,229 | 9 | 58 | 83 | **156** | 564 | 1,518 | 3,302 | 91,178 | 962,265 |
| **test** | 2,901 | 12 | 56 | 82 | **150** | 564 | 1,560 | 3,522 | 86,170 | 388,555 |

### Length (words)

| | median | 90% | 95% | 99% | max |
|---|---|---|---|---|---|
| **train** | 26 | 270 | 535 | 10,403 | 118,430 |
| **test** | 25 | 283 | 566 | 9,748 | 46,329 |

- Median query is short (~150 chars / ~26 words) but the top 1% are enormous
  (hundreds of KB — pasted documents, long code, huge problem statements). **A handful of
  giant queries dominate token/cost budgets.** Truncation / length-aware features matter.
- **Train and test length distributions are nearly identical** → no train/test distribution
  shift in length; models tuned on train should transfer.
- Query length correlates with cost (Spearman ≈ 0.55) but **not** with difficulty
  (length vs mean-performance Spearman ≈ −0.03) — long ≠ hard.

### Language / script mix (train)
Predominantly English, with a long multilingual tail:

| Latin/English | CJK (zh/ja) | Cyrillic | Korean | other |
|---|---|---|---|---|
| 98.4% | 0.8% | 0.6% | 0.1% | <0.1% |

### Content-type signals (regex heuristics over train; categories overlap)

| Signal | Share |
|---|---|
| Contains a `?` (interrogative) | 65.5% |
| Starts with "what" | 16.6% |
| Starts with "how" | 3.2% |
| Math-heavy (>10 digits) | 26.4% |
| Contains LaTeX (`$`, `\frac`, `\sum`, …) | 15.8% |
| Mentions code tokens (`def `, `function`, `class`, `import`, `#include`) | 7.5% |
| Fenced code block (```` ``` ````) | 2.2% |
| Multiple-choice (`A)`/`(B)` patterns) | 1.7% |
| "summarize" | 1.1% |
| "translate" | 0.7% |
| Imperative gen ("write/create/generate/…") | 0.8% |

The corpus is a broad mix of **academic Q&A, math (heavy LaTeX), competitive programming,
science, multiple-choice exam items, creative writing, and everyday factual questions** —
consistent with an aggregation of public instruction/benchmark datasets. There is no single
column labeling the domain; domain/task type must be inferred from text if used as a feature.

---

## 7. Representative example queries (showing the variance)

Format: query (truncated), then per-model `performance` and `cost`. Models A…K.

**(a) Short, multilingual, creative — most models succeed (trivial → route cheapest):**
> `写一首关于春天的诗` ("Write a poem about spring", 9 chars)
> PERF: A0 B0 C1 D1 E1 F0.5 G1 H1 I0 J1 K1 | COST: K=0.0001 (cheapest solver) … F=0.0298
> → reward-optimal: **Model_K** (cheap and correct).

**(b) Short math, everyone solves (fully trivial):**
> `(1+i)^10 =`
> PERF: all = 1.0 | COST: K=0.0002 … F=0.0226 → **Model_K**.

**(c) Short proof request — partial credit & disagreement (hard, specialized):**
> `如何证明拉格朗日中值定理` ("How to prove the Lagrange Mean Value Theorem")
> PERF: A0.5 B1 C0 D0.5 E0.5 F0.5 G0 H0 I1 J1 K0.5 | only B, I, J fully solve
> → reward-optimal among solvers: cheapest of {B,I,J} = **Model_J** (0.0014).

**(d) Median-length creative writing — nearly all succeed:**
> "A long script of Whitney, Pokémon's Goldenrod City gym leader, getting frustrated after
> losing… then quits Pokémon battling out of frustration" (156 chars)
> PERF: A0.5 then all others 1.0 → cheapest solver **Model_C/K**.

**(e) Median-length hard math — only 3 of 11 solve (decisive routing case):**
> "The sequence {a_n} satisfies a_1=1, and a_{n+1}=10^n·a_n^2 … general term formula?"
> PERF: only D, E, I = 1.0; all others 0 | → cheapest solver = **Model_D** (0.0008).
> Note: the expensive models F (0.058) and H here **fail** — paying more is actively wrong.

**(f) Word problem — everyone solves:**
> "Mr. Jackson borrowed \$150 … 90 days … 6% interest. Find amount due." → all 1.0 → **K**.

**(g) Long competitive-programming problem — nobody solves (impossible):**
> "You are given an integer sequence of length N … print the maximum possible value …"
> (1,518 chars) PERF: all = 0.0 → route **cheapest** (minimize wasted cost); note Model_F
> cost shows 0.0 here (empty output).

**(h) Long graduate physics (magnetized sphere boundary-value problem, 3,303 chars):**
> PERF: all = 0.0 → impossible; cheapest model minimizes loss; expensive F costs 0.14 for nothing.

These illustrate the four regimes a router faces: **trivial-all-solve** (pick cheapest),
**impossible-none-solve** (pick cheapest to cut losses), **specialized** (only a specific
subset solves — the real signal), and **partial-credit** edge cases.

---

## 8. What to ask Deep Research / what to look for

Frame these when prompting the research assistant:

1. **Task framing.** Best-performing formulations for LLM routing with *known per-model
   labels on train*: (a) multi-label "win prediction" per (query, model) + cost-aware
   argmax; (b) direct multiclass classification of the reward-optimal model; (c)
   pairwise/preference (RouteLLM-style) routing; (d) regression to reward. Which wins
   empirically given ternary performance + tiny costs?
2. **Key papers/systems to mine:** RouteLLM (LMSYS), Hybrid LLM routing (Ding et al.),
   FrugalGPT / LLM cascades (Chen et al.), AutoMix, ZOOTER, Routoo, OptLLM, MetaLLM,
   "Routing to the Expert," Martian/Unify-style routers, and NVIDIA's own routing work
   (data is NVIDIA-sourced). Extract their feature pipelines and loss functions.
3. **Features that work for query→difficulty:** embedding models (e.g.
   sentence-transformers, BGE, E5, OpenAI/Cohere embeddings) + KNN over train; vs.
   fine-tuned encoder (DeBERTa/RoBERTa) classifiers; vs. LLM-as-judge. Which gives the best
   accuracy/effort trade-off for ~10k training rows?
4. **Cost-aware decision rule:** given the metric `0.85·P̄ − 0.15·(C̄/C_max)` is dominated
   by accuracy (cost weight ≤ 0.15, realized penalty ~0.001–0.05), and the optimal rule is
   "cheapest among top performers," does a cascade ("predict P(model solves) for cheap models,
   escalate to F/H only if all cheap-tier probabilities are low") beat a flat classifier? How
   to set the escalation probability threshold to trade the 0.85 perf gain against the small
   cost penalty?
5. **Class imbalance** (51% majority): calibration, class weighting, focal loss, threshold
   tuning, and macro vs. reward-weighted objectives.
6. **Handling the extreme length tail** (queries up to ~1 MB): truncation strategy, length
   as a feature, chunking for embeddings.
7. **Robustness/generalization:** train and test length distributions match, but confirm
   strategies that avoid overfitting the imbalanced label and the giant-query outliers.
8. **Baselines to characterize:** "always Model_K," "always cheapest that the global prior
   suggests," KNN-on-embeddings router, and a learned classifier — to locate the simple vs.
   strong Kaggle baselines.

---

## 9. Reproducibility note

All numbers above were computed from the provided CSVs with pandas 2.3 / numpy 2.2
(`data/train.csv`, `data/test.csv`, `data/sample_submission.csv`). The scoring metric is the
official `Reward_{0.85} = 0.85·P̄ − 0.15·(C̄/C_max)`. The one value not given in the spec is
**C_max** (the cost normalizer); we verified the optimal routing decision is invariant to it
(99.96–100% agreement across three plausible definitions), so it only needs to be pinned down
for final reward-magnitude tuning, not for the routing logic.

# Improvement Plan — LLM Routing

Current scores (train CV):
- Oracle ceiling:      0.686
- Always Model_F:      0.494  ← safe baseline
- Deployed router:     0.420  ← worse than constant

The gap to close is 0.686 − 0.494 = 0.192. Realistically, a good classifier
gets us to ~0.51–0.54. The bottleneck is how much signal query text carries.

---

## Phase 1 — classifier reframe (in progress)
`router_compare.py` is running 5-fold CV comparing:
- old regression router (22 regressors → argmax)
- LightGBM classifier (direct oracle-label prediction)
- KNN router

Expected gain: classifier > constant baseline. Results pending (~45 min from start).

**Next action:** pick winner, write submission.csv, submit to Kaggle.
Safety net: if classifier CV < 0.494, submit always-Model_F first.

---

## Phase 2 — error analysis on training data

Training data is well-suited for error analysis: we have oracle labels for all
10,182 rows, and 5-fold CV gives honest out-of-fold (OOF) predictions.

### What to build: `notebooks/error_analysis.py`

1. **OOF predictions from router_compare** — save OOF choices alongside oracle
   choices so every training query has: `(oracle_model, predicted_model, reward_oracle,
   reward_predicted)`.

2. **Reward regret per query** — `regret = reward(oracle) − reward(predicted)`.
   Most misroutes are cheap (models are close); a few are catastrophic. Sort by
   regret descending to find the high-value failure cases.

3. **Confusion matrix** — which model pairs get confused most? If K↔C is the
   dominant error, that's where to focus. If the classifier always falls back to K,
   the issue is different.

4. **Per-cluster analysis** — UMAP/PCA on the bge-large embeddings, colored by
   oracle model. Regions with mixed colors = genuinely hard queries; pure regions =
   easy wins we may already be getting right.

5. **Error characterisation** — for the top-regret failures, look at the actual
   query text. Are they a specific domain (code, math, creative)? Short vs long?
   This informs whether hand-crafted features would help.

### Limitation
Training-data analysis cannot detect distribution shift between train and test.
Use Kaggle submissions (max 3/day) to ground-truth any major strategy changes.

---

## Phase 3 — confidence-weighted fallback

**Idea:** when the classifier is uncertain, route to Model_F (safe constant)
instead of trusting a noisy prediction. Only route away when confident.

**Why this works:** the current router's problem is noise compounding. A
threshold on classifier confidence (e.g. max softmax prob > 0.6 → route,
else → Model_F) turns a noisy router into a conservative one that still beats
the constant on easy cases.

**To implement:**
- Get class probabilities from LightGBM classifier (`predict_proba`)
- CV-sweep the confidence threshold against reward
- Expected shape: reward rises then plateaus as threshold increases
  (at threshold=1.0 it degrades to always-Model_F)

---

## Phase 4 — regret-weighted training

**Idea:** not all classification errors are equal. Misrouting a query where
all models score similarly costs almost nothing; misrouting a query with one
dominant model costs a lot.

Weight each training sample by how much the oracle model beats the second-best:
`weight = reward(oracle) − reward(2nd_best)`. High-weight samples = high-stakes
queries where accuracy matters.

Pass `sample_weight` to LightGBM. Should reduce catastrophic errors even if
overall accuracy changes little.

---

## Phase 5 — hierarchical routing

**Idea:** the oracle distribution is very skewed (K=51%, then C=11%, D=11%...).
A flat 11-class classifier treats all splits equally. A hierarchy focuses
capacity where it matters:

1. Stage 1: K vs not-K (strong signal, high impact — half the dataset)
2. Stage 2: C vs D vs other (next biggest split)
3. Stage 3: fine routing for the tail

Each stage can be a separate binary/small classifier, trained only on the
relevant subset. Simpler, more interpretable, less prone to class imbalance.

---

## Phase 6 — feature augmentation

Embeddings capture semantics but miss surface-level signals that may correlate
with model routing:

- **Query length** (token count / char count) — longer queries may need more
  capable models
- **Vocabulary complexity** (type-token ratio, rare word count)
- **Domain markers** — presence of code blocks, math notation, named entities
- **Question type** — factual lookup vs reasoning vs creative

Add these as additional columns alongside the 1024-d bge-large embedding.
Cost: cheap to compute, may give the classifier easy wins.

---

## Submission strategy

| CV reward    | Action                                      |
|-------------|---------------------------------------------|
| < 0.494     | Submit always-Model_F immediately           |
| 0.494–0.52  | Submit classifier; also bank Model_F        |
| > 0.52      | Submit classifier; try confidence threshold |

Max 3 submissions/day. Always keep one slot free for a safety fallback.

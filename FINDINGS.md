# FINDINGS — LLM Routing Competition

Running record of what we've measured and concluded. Cross-machine record (synced via
git). Numbers are 5-fold OOF reward under the **real** per-query-normalized metric.

---

## 0. The metric (critical — get this right)

`Reward_0.85 = 0.85 * mean(perf) - 0.15 * mean(cost / C_max_per_query)`

C_max is the **max cost across models for THAT query** (per-query), NOT a global constant.
`src/metric.py` (global CMAX) is WRONG and does not match the leaderboard. Use:

```python
def perquery_cost_norm(cost_mat):
    q = cost_mat.max(1, keepdims=True); q[q == 0] = 1.0
    return cost_mat / q
```

Performance is discrete {0.0, 0.5, 1.0}. Cost is continuous, ~length-driven.

---

## 1. Confirmed leaderboard calibration

| submission | our CV | actual LB | CV optimism |
|---|---|---|---|
| always-F | 0.3725 | **0.36** | +0.012 |
| always-K | 0.4509 | **0.44** | +0.011 |
| bge K/H router | 0.4747 | **0.43** | **+0.045** |
| e5 conservative router (margin .03) | 0.4808 | **0.44** | **+0.041** |

**The e5 router (CV 0.4808) scored exactly always-K (0.44).** Even conservative routing
adds ZERO net LB value — the CV edge is entirely overfitting. Confirms: the routing signal
in our features does not transfer. Reshuffling features/models won't fix this; only NEW,
transferable information can (see §7).

**Two constants agree: our CV is ~0.011 optimistic (a small, stable metric offset).**
The router's gap is 0.045 → the extra ~0.034 is **overfitting**. And always-K (0.44)
**beats** our first router (0.43): early cleverness was worse than the dumbest constant.

Rule of thumb: **to beat 0.44 LB we need CV ≈ 0.46+** (add the ~0.011 offset back).

Model facts: **K** = cheap + decent, best constant, oracle pick ~51% of queries → the
safe default. **F** = most expensive → worst constant, never a fallback. **H** = high
perf but expensive → an easy "dumping ground" trap for naive routers.

---

## 2. The CV→LB gap is NOT distribution shift

Adversarial validation (classify train vs test on bge embeddings): **AUC = 0.497**.
Train and test are statistically identical; query lengths match. So CV *should* track LB.
The gap is **overfitting in the routing**, not a train/test mismatch. → Fix = regularize
hard + route conservatively, not "collect different data."

---

## 3. Where reward leaks (error analysis)

Tools: `notebooks/analyze_failures.py`, `notebooks/error_profile.py` → `notebooks/figs/`,
`notebooks/export_errors.py` → `errors_for_review.csv` (browsable, sorted by regret).

- **91% of reward loss is PERF SHORTFALL** (we route to a model that *fails* the query),
  only 9% is overpaying. The H-"dumping ground" costs little because H usually works.
- Mis-routed on ~35% of queries (regret>0.05). Loss spread across all oracle models — no
  single bug to fix.
- Error fingerprint is **weak**: mis-routed queries are only 1.1–1.4× on length / code /
  digits. Failures don't live in an obvious bucket.
- Directional split: **overpay** errors = shorter, math-y queries; **under-serve** errors
  = longer, code-heavy queries. Error rate peaks (55%) on 2k–5k char queries.

### 3a. The factual-recall ceiling (why routing is fundamentally capped)

A large share of errors are **obscure factual-recall questions** ("Who were Antonio
Negri's daughters?", "In which season does Gus confront a sniper?"). For these:

- Success depends on whether *that specific fact* sits in *that specific model's* weights
  — a **hidden per-model memorization property**, not a property of the query text.
- The winner set looks random ({K}, {FK}, {EFJK}, …) and is uncorrelated with anything
  extractable from the words. To a text-based predictor the label is **noise**.
- **Statistical crux:** when the outcome is ~independent of the features, the optimal
  decision collapses to the **constant** (route to best base-rate-per-cost = K). *No*
  router can beat the constant on a subset where the label is unpredictable from inputs.

Contrast: **difficulty-driven** queries (reasoning/code/math) carry signal in the text
(length, structure, complexity) → "harder → stronger model" is learnable. **That** is the
only subset where routing can win. Strategy implication: stay on K by default; only route
away when the text shows a real difficulty signal.

---

## 4. Feature experiment — INFORMATION-CAPPED, not algorithm-capped

`notebooks/feature_experiment.py`. Mean per-model success AUC + routing reward:

| setup | mean AUC | CV reward |
|---|---|---|
| bge only (LR) | 0.7523 | 0.4776 |
| bge + surf6 (LR) | 0.7546 | 0.4792 |
| bge + rich ~25 (LR) | 0.7668 | 0.4813 |
| rich only, no embedding (LR) | 0.7042 | 0.4743 |
| bge + rich, **LightGBM** | 0.7734 | 0.4677 |
| e5 only (LR) | 0.7655 | 0.4820 |
| **e5 + surf6 (LR)** | 0.7649 | **0.4834** ← best |
| e5 + rich (LR) | 0.7690 | 0.4808 |
| e5 + rich, **LightGBM** | 0.7770 | 0.4657 |

- AUC moves only **0.752 → 0.777** across stronger embedding + 25 extra features + a
  bigger model. Per-query success is **information-capped** (~0.77), consistent with §3a.
- **e5 > bge** slightly (free +0.004 reward). Adopt e5.
- **Rich features help AUC, hurt reward** → keep it simple (surf6).
- **LightGBM = overfitting smoking gun**: highest AUC, *lowest* reward. More capacity =
  worse on the metric. Use regularized LR.

Embeddings: bge `notebooks/Xtr.npy`/`Xte.npy`; e5 `notebooks/Xtr_e5.npy`/`Xte_e5.npy`
(`notebooks/embed_strong.py intfloat/e5-large-v2 e5`).

---

## 5. Regret cranking & TTA (`notebooks/tune_regret_tta.py`)

### Methodology (from the cross-team discussion — the mental model)
- **Regret weighting** = scale each query's training loss by how much the oracle beats
  always-K (`w = base + (adv/mean(adv))**power`). "Cranking" = lower base / raise power to
  focus harder on the gap. "Duplicating rows" = the discrete version (repeat decisive rows).
  Pick one, don't stack. Risk: overfit the small decisive set, route easy queries off K.
- **TTA (test-time augmentation)** = average predictions over multiple views of a query
  (paraphrases or embedders). Reduces *variance*, adds no *information*. Guards: keep the
  original weighted highest; drop variants with low cosine to original; gate on CV.
- **The instrument:** corrected per-query CV is ground truth and unlimited. Every knob
  (regret strength, TTA, paraphraser temp) is tuned offline on CV; nothing ships on faith.

### Results — both knobs do not help here
**PART A — regret cranking (sample_weight on e5+surf6 experts):** baseline 0.4834.

| base | power | reward | Δ |
|---|---|---|---|
| 0.15 | 1.0 | 0.4396 | −0.044 |
| 0.05 | 1.5 | 0.4394 | −0.044 |
| 0.00 | 2.0 | 0.4385 | −0.045 |

All 9 configs ≈ 0.439, **below even always-K**. Regret weighting **actively destroys**
routing: experts over-focus on decisive (partly-noise) queries, lose calibration on the
easy K-optimal majority, and route them off K wrongly. → **Keep regret OFF.** (The DeBERTa
script had `regret=True` by default — likely why it never beat a constant.)

**PART B — Stage-1 TTA (embedder ensemble, free):** baseline 0.4834.

| variant | reward |
|---|---|
| ENSEMBLE avg(bge, e5) | 0.4835 (+0.0001) |
| CONCAT [bge, e5, surf6] | 0.4800 (worse) |

Clean **no-op**: bge/e5 are correlated, averaging removes no meaningful variance; variance
isn't the bottleneck. **Decision: SKIP Stage-2 LLM-paraphrase TTA** — it also only reduces
variance, so expected gain ≈ 0 for real compute/$ (paraphrasing a factual question and
averaging cannot reveal which model memorized the fact). Not worth a paraphraser setup.

---

## 6. Current best recipe & next steps

**Best so far: `e5 + surf6 + regularized LR`, per-model P(perf=1) experts + per-query cost
regressors, routed by argmax(0.85·E[perf] − 0.15·predicted_cost). CV = 0.4834 (~0.47 LB).**
Regret OFF. TTA off. Simplicity wins; overfitting is the enemy.

Untested lever with upside:
- [ ] **Conservative routing**: K-margin / confidence threshold so we only leave K when the
      predicted gain is large. Should both lift reward and *shrink the CV→LB gap* (less
      overfit). Tune the margin on CV.
- [ ] **Candidate restriction**: routing essentially never needs A/G/I — restricting the
      set removes failure modes for free.
- [ ] Build + submit the e5 recipe to confirm it clears the 0.44 floor on the real LB.

Realistic ceiling if metric holds: ~0.46–0.50 LB. The rest of the gap to oracle (0.68) is
the factual-recall lottery (§3a) and is not reachable from query text alone.

---

## 7. LLM-feature idea (DeepSeek) — adds INFORMATION, the real bottleneck

Since routing on our features adds zero LB value (§1), the only thing that can help is
**new transferable signal**. An external LLM CANNOT predict per-model success (models are
anonymous A–K). What it CAN provide is signal about the QUERY:
- difficulty / "would a small model answer this?" (robust low-dim → should transfer better
  than overfit-prone embeddings)
- category (factual-recall-obscure vs reasoning/code/math) → identify lottery queries (§3a)
  and deliberately keep them on K.

Pilot (fail fast, cheap) before committing to ~15k calls:
- `notebooks/deepseek_features.py` (reads DEEPSEEK_API_KEY env; OpenAI-compatible client):
  per query -> JSON {category, difficulty 1-5, obscure_knowledge, small_model_can_answer}.
- Run ~2500 train queries, `notebooks/deepseek_eval.py` measures OOF reward with/without.
- Ship to full set only if the pilot lifts held-out reward meaningfully.
Expectation: helps only the difficulty-driven subset; cannot crack the factual lottery.
Models: deepseek-chat (V3, cheap) or deepseek-reasoner (R1, better difficulty, pricier).

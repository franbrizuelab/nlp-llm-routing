# HANDOFF — LLM Routing project (continue here)

Context for a fresh Claude session **on the 3080 lab machine**. Read top-to-bottom,
then run "Next steps". Deadline: **2026-06-26 23:59** (E3 + Kaggle).

## The competition
- Route each query to 1 of **11 models** (Model_A..Model_K) to maximize **Reward_{0.85}**.
- `data/train.csv` (10,182 rows): ID, query, perf+cost for all 11 models.
- `data/test.csv` (2,550 rows): ID + query only. Predict `pred_model` per row.
- Kaggle: kernel `franbrizuelab/nlp-llm-routing-tier1`, dataset
  `franbrizuelab/nlp-llm-routing-data`. **Max 3 submissions/day; pick 2 for private LB.**
- Grading: Kaggle 70% (over strong baseline=55pts, over simple=40), report 30% (needs a
  table comparing all methods tried).

## ⚠️ THE CRITICAL FACT (metric was misread for weeks)
The real Kaggle metric **penalizes cost PER-QUERY**, not by a giant global constant.
Proof: a CONSTANT `always-Model_F` submission scored **0.36** on Kaggle. A constant
can't overfit, so our local metric was wrong. The metric that reproduces the LB is:

    Reward = 0.85*perf - 0.15*(cost / max_cost_over_models_for_that_query)

Consequences (computed on train, all confirmed in `notebooks/train_deberta_router.py`):
- **always-F = 0.3725** (F is the MOST EXPENSIVE model -> WORST safe choice). Matches the 0.36 LB.
- **always-K = 0.4509** (cheapest-decent model -> the TRUE floor / simple baseline to beat).
- **oracle (cheapest-among-perf-best) = 0.6774** (ceiling).
- A perfect perf-predictor + our routing rule = **0.6531** (realistic policy ceiling).

So: **never fall back to Model_F. Fall back to Model_K.** Don't model cost for routing
(test has no cost cols and you don't need it). The oracle LABEL is metric-invariant, so
predicting performance well + "cheapest among predicted-best" is correct under true scoring.

## What's been tried (Kaggle LB)
| Submission                         | LB score | Note |
|------------------------------------|----------|------|
| 22-target regression router        | 0.42     | compounding regressor noise |
| always-Model_F                     | 0.36     | F is the cost trap |
| classifier w/ 0.99 threshold->F    | 0.37     | stayed on F ~every row = always-F in disguise |
| **always-Model_K (DO THIS NEXT)**  | ~0.45?   | new floor; confirms the metric theory |

## The plan: Part B = fine-tuned DeBERTa-v3 perf predictor
File: **`notebooks/train_deberta_router.py`** (ready, logic smoke-tested).
- Predicts each model's **ternary** perf {0,0.5,1.0} via an `11x3` head (perf is discrete!).
- Routes by **cheapest-among-predicted-best** (median-cost lookup), **K fallback** when weak.
- Validation reward uses the **per-query cost normalization** above (trustworthy CV).
- Smart bits baked in: focal loss, **regret weighting vs always-K** (focuses the decisive
  ~28% of queries where K isn't best), token-packing + `<LARGE_DATA_BLOCK>` for 962k-char
  outliers, optional surface features (math/latex/code/lang).

## Next steps (run on the 3080)
```bash
pip install -q sentencepiece transformers torch scikit-learn pandas numpy
# data: kaggle datasets download -d franbrizuelab/nlp-llm-routing-data -p data --unzip  (if missing)

# 0. SAFETY SUBMIT FIRST: always-K (~0.45). Confirms metric + banks a score >> 0.37.
python3 -c "import pandas as pd; te=pd.read_csv('data/test.csv'); \
pd.DataFrame({'ID':te.ID,'pred_model':'Model_K'}).to_csv('submission.csv',index=False)"
#   -> submit submission.csv to Kaggle, verify it lands ~0.45.

# 1. Train Part B (base first; writes submission.csv + prints val_reward & route dist):
python notebooks/train_deberta_router.py --epochs 3

# 2. Trustworthy 5-fold CV under the corrected metric:
python notebooks/train_deberta_router.py --cv --epochs 3

# 3. If base CV beats ~0.45 clearly, try large:
python notebooks/train_deberta_router.py --large --epochs 4 --batch 8
```
Tune on CV (top of the script): `EPS_TIE`, `TAU_WEAK`, `--gamma`, `--regret`, `--no_surf`.

## Submission strategy (50/50 public/private, choose 2)
Hedge: pick **always-K (~0.45 safe)** as one private submission and the **best Part-B
router** as the other. Trust CV over the public LB (public is only 1,275 queries).

## Key facts / gotchas
- This file may be read on a 2GB laptop GPU (MX550) by mistake — **train on the 3080**.
  DeBERTa-v3-base fits ~easily; large needs the 10GB + batch 8 + fp16 (script does fp16).
- DeBERTa-v3 tokenizer **requires `sentencepiece`** (not in the base Kaggle/torch image).
- Kaggle accelerator (T4 vs P100) is a UI-only setting; P100=sm_60 can't run the prebuilt
  torch kernels (see git memory). For Kaggle submission, prefer T4 or precompute locally.
- Kaggle account = **franbrizuelab** (ignore old `shincleriapr` slugs).

## Files
- `notebooks/train_deberta_router.py` — Part B (the plan above). Start here.
- `src/metric.py` — note: its default Cmax is the OLD global-max convention; the LB-correct
  metric is per-query (implemented inside train_deberta_router.py as `perquery_cost_norm`).
- `notebooks/embed.py`, `router_compare.py` — Part A (embedding classifier/kNN), still valid.
- `DATA_SUMMARY_FOR_RESEARCH.md` — dataset analysis (note: its "cost is negligible" claim is
  WRONG for the LB — see THE CRITICAL FACT above).
- `LLM Routing Strategies for Weak Signals.txt` — research report (mechanistic/lookahead
  routing in it are NOT applicable: the 11 models are anonymous, no activations available).

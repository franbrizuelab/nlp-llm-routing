# HANDOFF — LLM Routing project (continue here)

Context for a fresh Claude session on the lab machine (has a **3080**). Read this
top-to-bottom, then run the commands in "Next steps".

## The competition
- Route each query to 1 of **11 models** (Model_A..Model_K) to maximize
  **Reward_{0.85} = 0.85 · mean(perf_chosen) − 0.15 · (mean(cost_chosen) / C_max)**.
- `data/train.csv` (10,182 rows): ID, query, and perf+cost for all 11 models.
- `data/test.csv` (2,550 rows): ID + query only. Predict `pred_model` per row.
- Metric details + helpers in `src/metric.py`. Spec: `2026INLPFinalProject_LLMRouting.txt`.
- Kaggle: kernel `franbrizuelab/nlp-llm-routing-tier1`, dataset
  `franbrizuelab/nlp-llm-routing-data`. Max 3 submissions/day.

## Where we are (the problem)
Current Kaggle score = **0.42**, which is **WORSE than trivial baselines**.
Measured on train (no embeddings needed):

| Strategy                         | Reward |
|----------------------------------|--------|
| Oracle (per-query best, ceiling) | 0.686  |
| **Always Model_F** (best const)  | **0.494** |
| Always Model_H                   | 0.492  |
| Always Model_K                   | 0.452  |
| **Deployed router**              | **0.42** |
| Random                           | 0.388  |

**Root cause:** the deployed notebook (`kaggle_stage/kernel/tier1.ipynb`) regresses
**22 targets** (11 perf + 11 cost) from query embeddings, then routes by
`argmax(p̂ − α·ĉ)`. Errors across 22 noisy regressors compound → per-query choice
is near-random → worse than a smart constant. **The routing is adding noise.**

Oracle pick distribution (why a classifier should work): K=5172, C=1151, D=1087,
E=680, J=600, H=405, F=365, I=224, G=211, B=208, A=79. Very learnable IF the query
text carries signal. `C_max` (global max cost cell) = 1.376.

## The plan (fix it with evidence)
1. **Safety submission first:** always `Model_F` → ~0.49, banks a score above 0.42.
2. **Reframe as classification:** predict the oracle's best-model label directly from
   the embedding (one classifier), instead of 22 regressions. This is the main fix.
3. **Also try KNN** (route by nearest train queries' oracle labels).
4. Use a **stronger embedding model** now that we have GPU (see embed.py).
5. CV-compare all of them with `notebooks/router_compare.py` (5-fold, reports reward).
   Pick the winner, regenerate `submission.csv`, submit.

## Next steps (run these)
```bash
# 0. data: already in data/ if pulled; else: kaggle datasets download -d franbrizuelab/nlp-llm-routing-data -p data --unzip
pip install -q sentence-transformers lightgbm scikit-learn

# 1. Embed on the 3080 (~1-2 min). Try a strong model:
python notebooks/embed.py BAAI/bge-large-en-v1.5      # or bge-base-en-v1.5 to start

# 2. Head-to-head CV of routing strategies (reads notebooks/Xtr.npy):
python notebooks/router_compare.py
#    -> table of mean reward per strategy. Anything beating 0.494 is real progress.

# 3. Build submission from the winning strategy, write submission.csv, submit to Kaggle.
```

## Key facts / gotchas
- 3080 is great here: embedding/inference only needs ~1–2 GB VRAM (NOT LLM fine-tuning).
  Its 10 GB easily fits bge-large / e5-large / gte-large / even bge-m3.
- GPU helps **embedding only**. The classifier/GBM is CPU and fast at this scale.
- Kaggle accelerator type (T4 vs P100 vs TPU) is a **UI-only** session setting; the CLI
  push can't lock it. Pick "GPU T4 x2" in the editor before running. The notebook's
  `_pick_emb_device()` already falls back to CPU if torch can't run the GPU (P100=sm_60).
- Embeddings (*.npy) are gitignored — regenerate per machine via embed.py.
- Kaggle account here is **franbrizuelab** (an earlier run used `shincleriapr`; ignore it).

## Files
- `kaggle_stage/kernel/tier1.ipynb` — deployed notebook (the regression-router; what scored 0.42).
- `notebooks/embed.py` — produce cached embeddings (run first).
- `notebooks/router_compare.py` — CV comparison of strategies.
- `src/metric.py`, `src/route.py` — metric + local smoke router.
- `DATA_SUMMARY_FOR_RESEARCH.md` — dataset analysis.

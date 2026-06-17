# %% [markdown]
# # Tier-1 LLM Router — Embeddings + GBM + cost-aware routing
#
# Run on a free Colab/Kaggle **GPU** runtime. Upload `train.csv`, `test.csv`,
# `sample_submission.csv` (or mount them), then run top-to-bottom. Produces
# `submission.csv`. This mirrors `src/route.py` but uses real sentence-embeddings.
#
# **Backbone swap is the only real change vs. the local smoke test.** Try a few
# embedding models and keep the best validation reward.

# %% Install (Colab/Kaggle) — only installs what's missing; needs internet enabled
import importlib.util, subprocess, sys
for _pkg, _pip in [("sentence_transformers", "sentence-transformers"), ("lightgbm", "lightgbm")]:
    if importlib.util.find_spec(_pkg) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", _pip], check=False)

# %% Imports + config
import time, glob, os, numpy as np, pandas as pd
import torch

# Auto-detect data dir: Kaggle attaches data under /kaggle/input/<slug>/ (any depth).
if os.path.isdir("/kaggle/input"):
    print("DEBUG /kaggle/input contents:")
    for root, _dirs, files in os.walk("/kaggle/input"):
        for f in files:
            print("  ", os.path.join(root, f))
_hits = (glob.glob("/kaggle/input/**/train.csv", recursive=True)
         + glob.glob("./train.csv") + glob.glob("./data/train.csv"))
assert _hits, "train.csv not found under /kaggle/input — is the dataset attached?"
DATA = os.path.dirname(_hits[0])
print("DATA =", DATA)
EMB_MODEL = "BAAI/bge-base-en-v1.5" # fast + strong; CPU-friendly (corpus is ~98% English)
MAX_TOKENS = 512                    # truncate long queries (length != difficulty)
# Kaggle's default GPU is often a P100 (sm_60) that the installed PyTorch can't run,
# so we embed on CPU for reliability. RunPod (where we control the GPU/torch) handles
# heavier models / encoder fine-tunes later.
EMB_DEVICE = "cpu"
SEED, VAL_FRAC = 42, 0.15
PERF_WEIGHT, COST_WEIGHT = 0.85, 0.15

MODELS = list("ABCDEFGHIJK")
PERF_COLS = [f"Model_{m}_performance" for m in MODELS]
COST_COLS = [f"Model_{m}_cost" for m in MODELS]
print("cuda:", torch.cuda.is_available())

# %% Load
train = pd.read_csv(f"{DATA}/train.csv"); test = pd.read_csv(f"{DATA}/test.csv")
perf = train[PERF_COLS].to_numpy("float32"); cost = train[COST_COLS].to_numpy("float32")
CMAX = float(cost.max())
print("train", train.shape, "test", test.shape, "C_max", CMAX)

# %% Embed (the GPU step — ~minutes for 12.7k queries)
from sentence_transformers import SentenceTransformer
st = SentenceTransformer(EMB_MODEL, device=EMB_DEVICE)
st.max_seq_length = MAX_TOKENS
def embed(texts):
    return st.encode(list(texts.astype(str)), batch_size=64, show_progress_bar=True,
                     convert_to_numpy=True, normalize_embeddings=True).astype("float32")
t0 = time.time()
Xall = embed(train["query"]); Xtest = embed(test["query"])
print("embedded", Xall.shape, Xtest.shape, f"{time.time()-t0:.1f}s")
# Optional: np.save("emb_train.npy", Xall); np.save("emb_test.npy", Xtest)

# %% Reduce dims (big speedup for the 44 tree fits; ~all variance kept at 256)
from sklearn.decomposition import PCA
pca = PCA(n_components=256, random_state=SEED).fit(Xall)
Xall, Xtest = pca.transform(Xall).astype("float32"), pca.transform(Xtest).astype("float32")
print("after PCA:", Xall.shape, "explained var:", round(float(pca.explained_variance_ratio_.sum()), 3))

# %% Metric + routing helpers
def reward_from_choice(P, C, ch, cmax):
    r = np.arange(len(ch))
    return PERF_WEIGHT*P[r, ch].mean() - COST_WEIGHT*(C[r, ch].mean()/cmax)
def oracle_choice(P, C, cmax):
    return (PERF_WEIGHT*P - (COST_WEIGHT/cmax)*C).argmax(1)
def route(ph, ch_, a):
    return (ph - a*ch_).argmax(1)

# %% Regressors (LightGBM; falls back to sklearn HistGB)
try:
    import lightgbm as lgb
    def mk(): return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63,
                                       subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                                       n_jobs=-1, verbosity=-1)
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    def mk(): return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                                   max_leaf_nodes=63, early_stopping=True, random_state=SEED)
def fit_pred(Xtr, Ytr, Xp):
    out = np.zeros((Xp.shape[0], Ytr.shape[1]), "float32")
    for j in range(Ytr.shape[1]):
        m = mk(); m.fit(Xtr, Ytr[:, j]); out[:, j] = m.predict(Xp)
    return out

# %% Validation split + fit
rng = np.random.default_rng(SEED); idx = rng.permutation(len(train)); nv = int(len(train)*VAL_FRAC)
vi, ti = idx[:nv], idx[nv:]
ph_val = fit_pred(Xall[ti], perf[ti], Xall[vi])
ch_val = fit_pred(Xall[ti], cost[ti], Xall[vi])

# %% Tune cost trade-off alpha on val (maximize TRUE reward)
base = COST_WEIGHT/(PERF_WEIGHT*CMAX)
grid = np.r_[0.0, np.geomspace(base*0.01, base*100, 40)]
alpha = max(grid, key=lambda a: reward_from_choice(perf[vi], cost[vi], route(ph_val, ch_val, a), CMAX))
r_val = reward_from_choice(perf[vi], cost[vi], route(ph_val, ch_val, alpha), CMAX)
oracle = reward_from_choice(perf[vi], cost[vi], oracle_choice(perf[vi], cost[vi], CMAX), CMAX)
const = max(reward_from_choice(perf[vi], cost[vi], np.full(nv, j), CMAX) for j in range(11))
print(f"VAL reward: const={const:.4f}  router={r_val:.4f}  oracle={oracle:.4f}  "
      f"(headroom captured {(r_val-const)/(oracle-const):.1%}, alpha={alpha:.4g})")

# %% Refit on full train, predict test, write submission
ph_te = fit_pred(Xall, perf, Xtest); ch_te = fit_pred(Xall, cost, Xtest)
choice = route(ph_te, ch_te, alpha)
sub = pd.DataFrame({"ID": test["ID"], "pred_model": [f"Model_{MODELS[i]}" for i in choice]})
sub.to_csv("submission.csv", index=False)
print(sub["pred_model"].value_counts()); print("wrote submission.csv")

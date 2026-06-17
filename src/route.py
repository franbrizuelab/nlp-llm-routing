"""LLM-routing baseline: query features -> per-model perf/cost regressors ->
cost-aware argmax routing, with the routing trade-off tuned against Reward_{0.85}
on a held-out validation split.

Backbones (``--backbone``):
  * ``tfidf``         : TF-IDF + TruncatedSVD. CPU-only, no downloads. For local
                        smoke-testing the full pipeline / metric / submission.
  * ``st:<model>``    : sentence-transformers model (e.g. ``st:BAAI/bge-m3``,
                        ``st:intfloat/e5-large-v2``). The real Colab/Kaggle run --
                        just swap the backbone, everything else is identical.

Regressors: sklearn HistGradientBoosting (a LightGBM stand-in available without
extra installs). Pass ``--gbm lightgbm`` to use LightGBM if installed.

Outputs a Kaggle ``submission.csv`` and prints validation reward vs. the oracle
ceiling and best-constant baseline.
"""
from __future__ import annotations
import argparse, time, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metric import reward_from_choice, oracle_choice, default_cmax, PERF_WEIGHT, COST_WEIGHT

MODELS = list("ABCDEFGHIJK")
PERF_COLS = [f"Model_{m}_performance" for m in MODELS]
COST_COLS = [f"Model_{m}_cost" for m in MODELS]


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def build_features(train_q, test_q, backbone: str, svd_dim: int = 256, seed: int = 0):
    """Return dense (X_train, X_test) feature matrices for the query text."""
    if backbone == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Normalizer
        vec = make_pipeline(
            TfidfVectorizer(max_features=50_000, ngram_range=(1, 2),
                            sublinear_tf=True, strip_accents="unicode"),
            TruncatedSVD(n_components=svd_dim, random_state=seed),
            Normalizer(copy=False),
        )
        Xtr = vec.fit_transform(train_q).astype(np.float32)
        Xte = vec.transform(test_q).astype(np.float32)
        return Xtr, Xte
    if backbone.startswith("st:"):
        from sentence_transformers import SentenceTransformer
        model_name = backbone[3:]
        st = SentenceTransformer(model_name)
        enc = lambda texts: st.encode(
            list(texts), batch_size=64, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32)
        return enc(train_q), enc(test_q)
    raise ValueError(f"unknown backbone {backbone!r}")


# --------------------------------------------------------------------------- #
# Regressor factory
# --------------------------------------------------------------------------- #
def make_regressor(kind: str, seed: int):
    if kind == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                 num_leaves=63, subsample=0.8,
                                 colsample_bytree=0.8, random_state=seed, n_jobs=-1)
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                         max_leaf_nodes=63, l2_regularization=1.0,
                                         early_stopping=True, random_state=seed)


def fit_predict_targets(Xtr, Ytr, Xpred, gbm: str, seed: int, label: str):
    """Fit one regressor per column of Ytr; return predictions on Xpred."""
    preds = np.zeros((Xpred.shape[0], Ytr.shape[1]), dtype=np.float32)
    for j in range(Ytr.shape[1]):
        reg = make_regressor(gbm, seed)
        reg.fit(Xtr, Ytr[:, j])
        preds[:, j] = reg.predict(Xpred)
        print(f"  [{label}] fitted model {MODELS[j]}", flush=True)
    return preds


# --------------------------------------------------------------------------- #
# Routing decision + alpha sweep
# --------------------------------------------------------------------------- #
def route(perf_hat, cost_hat, alpha):
    """Choose argmax of predicted reward: perf_hat - alpha * cost_hat."""
    return (perf_hat - alpha * cost_hat).argmax(axis=1)


def tune_alpha(perf_hat_val, cost_hat_val, perf_val, cost_val, cmax):
    """Grid-search the cost-trade-off alpha to maximize TRUE Reward_{0.85} on val."""
    base = COST_WEIGHT / (PERF_WEIGHT * cmax)  # alpha implied by the metric
    grid = np.concatenate([[0.0], np.geomspace(base * 0.01, base * 100, 40)])
    best_a, best_r = 0.0, -1e9
    for a in grid:
        ch = route(perf_hat_val, cost_hat_val, a)
        r = reward_from_choice(perf_val, cost_val, ch, cmax)
        if r > best_r:
            best_r, best_a = r, a
    return best_a, best_r, base


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--backbone", default="tfidf")
    ap.add_argument("--gbm", default="hist", choices=["hist", "lightgbm"])
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()
    t0 = time.time()

    data = Path(args.data)
    train = pd.read_csv(data / "train.csv")
    test = pd.read_csv(data / "test.csv")
    perf = train[PERF_COLS].to_numpy(np.float32)
    cost = train[COST_COLS].to_numpy(np.float32)
    cmax = default_cmax(cost)
    print(f"train={train.shape} test={test.shape} C_max={cmax:.4f}")

    # ----- features -----
    print(f"[features] backbone={args.backbone}")
    Xall, Xtest = build_features(train["query"].astype(str), test["query"].astype(str),
                                 args.backbone, seed=args.seed)
    print(f"[features] X_train={Xall.shape} X_test={Xtest.shape}  ({time.time()-t0:.1f}s)")

    # ----- validation split -----
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(train))
    n_val = int(len(train) * args.val_frac)
    val_i, tr_i = idx[:n_val], idx[n_val:]
    Xtr, Xval = Xall[tr_i], Xall[val_i]
    perf_tr, perf_val = perf[tr_i], perf[val_i]
    cost_tr, cost_val = cost[tr_i], cost[val_i]

    # ----- fit perf & cost regressors on the train split, predict on val -----
    print("[fit] performance regressors (train split)")
    perf_hat_val = fit_predict_targets(Xtr, perf_tr, Xval, args.gbm, args.seed, "perf")
    print("[fit] cost regressors (train split)")
    cost_hat_val = fit_predict_targets(Xtr, cost_tr, Xval, args.gbm, args.seed, "cost")

    # ----- tune alpha on val + report references -----
    alpha, r_val, base = tune_alpha(perf_hat_val, cost_hat_val, perf_val, cost_val, cmax)
    oracle = reward_from_choice(perf_val, cost_val, oracle_choice(perf_val, cost_val, cmax), cmax)
    const = max(reward_from_choice(perf_val, cost_val, np.full(len(val_i), j), cmax)
                for j in range(len(MODELS)))
    print("\n=== VALIDATION (Reward_0.85) ===")
    print(f"  best constant baseline : {const:.4f}")
    print(f"  our router (alpha={alpha:.4g}, metric-implied={base:.4g}) : {r_val:.4f}")
    print(f"  oracle ceiling         : {oracle:.4f}")
    print(f"  fraction of headroom captured: "
          f"{(r_val-const)/(oracle-const):.1%}\n")

    # ----- refit on ALL train, predict test, write submission -----
    print("[refit] on full train -> predict test")
    perf_hat_te = fit_predict_targets(Xall, perf, Xtest, args.gbm, args.seed, "perf-full")
    cost_hat_te = fit_predict_targets(Xall, cost, Xtest, args.gbm, args.seed, "cost-full")
    choice = route(perf_hat_te, cost_hat_te, alpha)
    sub = pd.DataFrame({"ID": test["ID"], "pred_model": [f"Model_{MODELS[i]}" for i in choice]})
    sub.to_csv(args.out, index=False)
    print(f"[done] wrote {args.out} ({len(sub)} rows) in {time.time()-t0:.1f}s")
    print("  prediction distribution:")
    print(sub["pred_model"].value_counts().to_string())


if __name__ == "__main__":
    main()

"""Build submission using bge-large-en-v1.5 embeddings + surface features.

Two-stage routing:
  Stage 1: LR classifier (bge+surf) -> P(K-optimal)
  Stage 2: If P(K-opt) > threshold -> K, else argmax of LR multi-class proba
            adjusted by reward score (PERF_W * proba - COST_W * mean_norm_cost)

CV reward (bge+surf, thr=0.50, K-vs-H binary): ~0.4744
Compare to always-K: 0.4509, oracle: 0.6774
"""
from __future__ import annotations
import argparse, re
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

MODELS = list("ABCDEFGHIJK")
PERF_W, COST_W = 0.85, 0.15


def perquery_cost_norm(cost_mat):
    qmax = cost_mat.max(1, keepdims=True); qmax[qmax == 0] = 1.0
    return cost_mat / qmax


def surface_feats(texts):
    rows = []
    for t in texts:
        t = str(t); n = max(len(t), 1)
        rows.append([
            min(len(t) / 4000.0, 4.0),
            t.count(" ") / n,
            float(bool(re.search(r"[\x60]{3}|def |class |import |;\s*$", t))),
            float(len(t) > 2000),
            float(len(t) > 5000),
            min(t.count(".") / 20.0, 2.0),
        ])
    return np.array(rows, dtype=np.float32)


def reward_of_choice(perf_mat, cost_mat, choice_idx):
    cn = perquery_cost_norm(cost_mat)
    rows = np.arange(len(choice_idx))
    return PERF_W * perf_mat[rows, choice_idx].mean() - COST_W * cn[rows, choice_idx].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="lgbm", choices=["lr_binary", "lgbm"],
                    help="lr_binary: K-vs-H LR; lgbm: 11-class LightGBM reward routing")
    ap.add_argument("--threshold", type=float, default=0.50,
                    help="P(K-optimal) threshold for lr_binary mode")
    ap.add_argument("--cv", action="store_true", help="run 5-fold CV instead of submission")
    ap.add_argument("--out", default="submission_bge.csv")
    args = ap.parse_args()

    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    perf = tr[[f"Model_{m}_performance" for m in MODELS]].values
    cost = tr[[f"Model_{m}_cost" for m in MODELS]].values
    cn = perquery_cost_norm(cost)
    mean_nc = cn.mean(0)  # (11,) mean norm cost per model

    Xtr = np.load("notebooks/Xtr.npy")
    Xte = np.load("notebooks/Xte.npy")
    surf_tr = surface_feats(tr["query"].tolist())
    surf_te = surface_feats(te["query"].tolist())
    Xtr_full = np.hstack([Xtr, surf_tr])
    Xte_full = np.hstack([Xte, surf_te])

    oracle = (PERF_W * perf - COST_W * cn).argmax(1)
    kidx = MODELS.index("K")
    hidx = MODELS.index("H")

    print(f"always-K: {reward_of_choice(perf, cost, np.full(len(tr), kidx)):.4f}")
    print(f"oracle  : {(PERF_W*perf - COST_W*cn).max(1).mean():.4f}")

    if args.cv:
        from sklearn.model_selection import StratifiedKFold
        kf = StratifiedKFold(5, shuffle=True, random_state=42)
        fold_rewards = []
        for tri, vai in kf.split(Xtr_full, oracle):
            clf = _fit(args.mode, Xtr_full[tri], oracle[tri])
            choice = _predict(args.mode, clf, Xtr_full[vai], mean_nc, kidx, hidx, args.threshold)
            r = reward_of_choice(perf[vai], cost[vai], choice)
            fold_rewards.append(r)
        print(f"CV reward ({args.mode}): {np.mean(fold_rewards):.4f} +/- {np.std(fold_rewards):.4f}")
        return

    clf = _fit(args.mode, Xtr_full, oracle)
    choice = _predict(args.mode, clf, Xte_full, mean_nc, kidx, hidx, args.threshold)
    sub = pd.DataFrame({"ID": te["ID"], "pred_model": [f"Model_{MODELS[c]}" for c in choice]})
    sub.to_csv(args.out, index=False)
    print(f"wrote {args.out}  dist: {sub['pred_model'].value_counts().to_dict()}")


def _fit(mode, X, y_oracle):
    if mode == "lr_binary":
        y_bin = (y_oracle == MODELS.index("K")).astype(int)
        clf = LogisticRegression(C=0.1, max_iter=500, solver="saga", n_jobs=4)
        clf.fit(X, y_bin)
    else:
        clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.05,
                                  n_jobs=4, random_state=42, verbose=-1)
        clf.fit(pd.DataFrame(X), y_oracle)
    return clf


def _predict(mode, clf, X, mean_nc, kidx, hidx, threshold):
    MODELS_local = list("ABCDEFGHIJK")
    if mode == "lr_binary":
        p_kopt = clf.predict_proba(X)[:, 1]
        return np.where(p_kopt > threshold, kidx, hidx)
    else:
        proba = clf.predict_proba(np.array(X) if not isinstance(X, pd.DataFrame) else X)
        score = PERF_W * proba - COST_W * mean_nc[np.newaxis, :]
        return score.argmax(1)


if __name__ == "__main__":
    main()

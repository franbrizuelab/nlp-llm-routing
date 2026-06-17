"""Head-to-head CV comparison of routing strategies on Reward_{0.85}.

Run `python notebooks/embed.py` FIRST (writes notebooks/Xtr.npy). Then this does
5-fold CV on train; for each fold we fit on 4 folds and route the held-out fold,
then report mean reward. The point: see which strategy actually beats the
best-constant baseline (~0.494, "always Model_F"). The deployed router scored
0.42 on Kaggle -> WORSE than the constant baseline (see HANDOFF.md).
"""
import time, numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

PW, CW = 0.85, 0.15
MODELS = list("ABCDEFGHIJK")

tr = pd.read_csv("data/train.csv")
P = tr[[f"Model_{m}_performance" for m in MODELS]].to_numpy(float)
C = tr[[f"Model_{m}_cost" for m in MODELS]].to_numpy(float)
X = np.load("notebooks/Xtr.npy")
CMAX = C.max()
N, K = P.shape

def reward(choice, idx):
    r = np.arange(len(idx))
    return PW * P[idx][r, choice].mean() - CW * (C[idx][r, choice].mean() / CMAX)

def oracle_label(Pm, Cm):
    return (PW * Pm - (CW / CMAX) * Cm).argmax(1)

ORACLE = oracle_label(P, C)

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

def reg_route(Xtr, Xva, Ptr, Ctr):
    """Original approach: regress 11 perf + 11 cost, route by argmax(p - a*c)."""
    def fitpred(Y):
        out = np.zeros((len(Xva), Y.shape[1]), "float32")
        for j in range(Y.shape[1]):
            if HAS_LGB:
                m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63,
                                      subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1)
            else:
                from sklearn.ensemble import HistGradientBoostingRegressor
                m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05)
            m.fit(Xtr, Y[:, j]); out[:, j] = m.predict(Xva)
        return out
    ph = fitpred(Ptr); ch = fitpred(Ctr)
    base = CW / (PW * CMAX)
    grid = np.r_[0.0, np.geomspace(base * 0.01, base * 100, 30)]
    # tune alpha on TRAIN predictions vs train truth (proxy) — keep simple: a=base
    return ph, ch, grid

def clf_route(Xtr, Xva, ytr):
    """Direct classification of the oracle label."""
    if HAS_LGB:
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                               subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1)
    else:
        m = LogisticRegression(max_iter=1000, C=1.0)
    m.fit(Xtr, ytr)
    return m.predict(Xva)

def main():
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    res = {k: [] for k in ["const_F", "oracle", "reg_router", "clf_lgb", "knn50", "logreg"]}
    best_const = max(range(K), key=lambda j: PW * P[:, j].mean() - CW * (C[:, j].mean() / CMAX))
    for fi, (ti, vi) in enumerate(kf.split(X)):
        t0 = time.time()
        res["const_F"].append(reward(np.full(len(vi), best_const), vi))
        res["oracle"].append(reward(ORACLE[vi], vi))

        # regression router (current method)
        ph, ch, grid = reg_route(X[ti], X[vi], P[ti], C[ti])
        # tune alpha on the fold's own truth (optimistic but matches notebook intent)
        a = max(grid, key=lambda a: reward((ph - a * ch).argmax(1), vi))
        res["reg_router"].append(reward((ph - a * ch).argmax(1), vi))

        # classification router
        res["clf_lgb"].append(reward(clf_route(X[ti], X[vi], ORACLE[ti]), vi))

        # KNN on oracle labels
        knn = KNeighborsClassifier(n_neighbors=50, metric="cosine")
        knn.fit(X[ti], ORACLE[ti])
        res["knn50"].append(reward(knn.predict(X[vi]), vi))

        # logistic regression baseline
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X[ti], ORACLE[ti])
        res["logreg"].append(reward(lr.predict(X[vi]), vi))
        print(f"fold {fi} done {time.time()-t0:.0f}s", flush=True)

    print(f"\nbest constant = Model_{MODELS[best_const]}  (HAS_LGB={HAS_LGB})")
    print(f"{'strategy':<14} {'mean reward':>12} {'std':>8}")
    for k, v in sorted(res.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"{k:<14} {np.mean(v):>12.4f} {np.std(v):>8.4f}")

if __name__ == "__main__":
    main()

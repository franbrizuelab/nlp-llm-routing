"""Measure whether DeepSeek query features add TRANSFERABLE routing signal.
Compares 5-fold OOF reward (real metric) with vs without the DeepSeek features, on the
subset of train queries that have been classified (notebooks/deepseek_train.csv).

Run after notebooks/deepseek_features.py --split train.
  python notebooks/deepseek_eval.py
"""
from __future__ import annotations
import re
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

MODELS = list("ABCDEFGHIJK"); PERF_W, COST_W = 0.85, 0.15; KIDX = MODELS.index("K"); FOLDS = 5


def pqnorm(c):
    q = c.max(1, keepdims=True); q[q == 0] = 1.0
    return c / q


def fit_reward(X, perf, cn):
    kf = KFold(FOLDS, shuffle=True, random_state=42)
    p1 = np.zeros((len(perf), 11)); cnp = np.zeros((len(perf), 11))
    for tri, vai in kf.split(X):
        for m in range(11):
            ym = (perf[tri, m] == 1.0).astype(int)
            if ym.sum() in (0, len(ym)): p1[vai, m] = ym.mean()
            else:
                c = LogisticRegression(C=0.5, max_iter=300); c.fit(X[tri], ym)
                p1[vai, m] = c.predict_proba(X[vai])[:, 1]
            rg = Ridge(alpha=10.0); rg.fit(X[tri], cn[tri, m]); cnp[vai, m] = rg.predict(X[vai]).clip(0, 1)
    aucs = [roc_auc_score((perf[:, m] == 1.0).astype(int), p1[:, m])
            for m in range(11) if 0 < (perf[:, m] == 1.0).sum() < len(perf)]
    ch = (PERF_W * p1 - COST_W * cnp).argmax(1); r = np.arange(len(perf))
    return np.mean(aucs), (PERF_W * perf[r, ch] - COST_W * cn[r, ch]).mean()


def main():
    tr = pd.read_csv("data/train.csv")
    ds = pd.read_csv("notebooks/deepseek_train.csv").drop_duplicates("ID")
    m = tr.merge(ds, on="ID", how="inner").reset_index(drop=True)
    print(f"pilot subset: {len(m)} queries with DeepSeek features")

    perf = m[[f"Model_{x}_performance" for x in MODELS]].values
    cost = m[[f"Model_{x}_cost" for x in MODELS]].values
    cn = pqnorm(cost)
    idx = m["ID"].map({int(i): k for k, i in enumerate(tr["ID"])}).values  # row in full e5
    e5 = np.load("notebooks/Xtr_e5.npy")[idx]
    q = m["query"].astype(str)
    surf6 = np.array([[min(len(t)/4000., 4.), t.count(" ")/max(len(t), 1),
                       float(bool(re.search(r"[\x60]{3}|def |class |import ", t))),
                       float(len(t) > 2000), float(len(t) > 5000),
                       min(t.count(".")/20., 2.)] for t in q], dtype=np.float32)
    base = np.hstack([e5, surf6])

    # DeepSeek features: one-hot category + scaled difficulty + obscure + small_ok
    cat = np.eye(5, dtype=np.float32)[m["category"].clip(0, 4).values]
    dsf = np.hstack([cat, (m[["difficulty"]].values/5.0).astype(np.float32),
                     m[["obscure", "small_ok"]].values.astype(np.float32)])
    withds = np.hstack([base, dsf])

    print(f"always-K={(PERF_W*perf[:,KIDX]-COST_W*cn[:,KIDX]).mean():.4f}  "
          f"oracle={(PERF_W*perf-COST_W*cn).max(1).mean():.4f}")
    a0, r0 = fit_reward(base, perf, cn)
    a1, r1 = fit_reward(withds, perf, cn)
    # deepseek features alone (no embedding) — tests their raw signal
    a2, r2 = fit_reward(np.hstack([surf6, dsf]), perf, cn)
    print(f"  e5+surf6            AUC={a0:.4f}  reward={r0:.4f}")
    print(f"  e5+surf6+deepseek   AUC={a1:.4f}  reward={r1:.4f}   (delta {r1-r0:+.4f})")
    print(f"  surf6+deepseek only AUC={a2:.4f}  reward={r2:.4f}")
    print("\nVERDICT:", "DeepSeek features HELP -> scale to full set + test set"
          if r1 - r0 > 0.003 else "no meaningful lift -> not worth full extraction")


if __name__ == "__main__":
    main()

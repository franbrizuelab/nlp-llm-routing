"""Error characterization: WHAT kind of wrong choices does the router make, and
on WHAT queries? Uses OOF predictions (held-out), real per-query-cost metric.

Two questions:
  A. Direction of error: do we OVER-SPEND (route too expensive) or UNDER-SERVE
     (stay on cheap K when we should upgrade)? -> regret split into perf vs cost.
  B. Fingerprint of error: how do mis-routed queries differ from correct ones on
     measurable features? -> tells us what feature we're blind to.
"""
from __future__ import annotations
import re, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold

MODELS = list("ABCDEFGHIJK")
PERF_W, COST_W = 0.85, 0.15


def pqnorm(c):
    q = c.max(1, keepdims=True); q[q == 0] = 1.0
    return c / q


def feats(q):
    t = str(q); n = max(len(t), 1)
    words = t.split()
    nonascii = sum(ord(ch) > 127 for ch in t)
    return {
        "len_chars": len(t),
        "n_words": len(words),
        "avg_word_len": np.mean([len(w) for w in words]) if words else 0.0,
        "has_code": float(bool(re.search(r"[\x60]{3}|def |class |import |#include|public ", t))),
        "has_math": float(bool(re.search(r"\\frac|\\begin|\\sum|\\int|=\s*\d|\^\d", t))),
        "has_url": float("http" in t),
        "n_digits": sum(ch.isdigit() for ch in t) / n,
        "nonascii_ratio": nonascii / n,
        "n_questions": t.count("?"),
        "n_newlines": t.count("\n"),
    }


def main():
    tr = pd.read_csv("data/train.csv")
    perf = tr[[f"Model_{m}_performance" for m in MODELS]].values
    cost = tr[[f"Model_{m}_cost" for m in MODELS]].values
    cn = pqnorm(cost); mean_nc = cn.mean(0)
    Xtr = np.load("notebooks/Xtr.npy")

    surf = np.array([[ # compact surface feats for the model input
        min(len(str(q))/4000., 4.), str(q).count(" ")/max(len(str(q)),1),
        float(bool(re.search(r"[\x60]{3}|def |class |import ", str(q)))),
        float(len(str(q)) > 2000), float(len(str(q)) > 5000),
        min(str(q).count(".")/20., 2.)] for q in tr["query"]], dtype=np.float32)
    X = np.hstack([Xtr, surf])

    rew_mat = PERF_W * perf - COST_W * cn
    oracle = rew_mat.argmax(1)
    kidx = MODELS.index("K")

    # OOF experts (P(perf=1)) + per-query cost regressors -> our best router
    kf = KFold(5, shuffle=True, random_state=42)
    oof_p1 = np.zeros((len(tr), 11)); oof_cn = np.zeros((len(tr), 11))
    t0 = time.time()
    for tri, vai in kf.split(X):
        for m in range(11):
            ym = (perf[tri, m] == 1.0).astype(int)
            if ym.sum() in (0, len(ym)):
                oof_p1[vai, m] = ym.mean()
            else:
                c = LogisticRegression(C=0.5, max_iter=300); c.fit(X[tri], ym)
                oof_p1[vai, m] = c.predict_proba(X[vai])[:, 1]
            rg = Ridge(alpha=10.0); rg.fit(X[tri], cn[tri, m])
            oof_cn[vai, m] = rg.predict(X[vai]).clip(0, 1)
    pred = (PERF_W * oof_p1 - COST_W * oof_cn).argmax(1)
    print(f"OOF fit {time.time()-t0:.0f}s")

    rows = np.arange(len(tr))
    perf_c, cn_c = perf[rows, pred], cn[rows, pred]
    perf_o, cn_o = perf[rows, oracle], cn[rows, oracle]
    regret = rew_mat[rows, oracle] - rew_mat[rows, pred]

    # ---- A. direction of error: perf shortfall vs cost overspend ----
    perf_term = PERF_W * (perf_o - perf_c)     # reward lost to picking lower-perf
    cost_term = COST_W * (cn_c - cn_o)         # reward lost to overpaying
    print("\n=== A. DIRECTION OF ERROR (total reward lost, decomposed) ===")
    print(f"  perf shortfall term : {perf_term.sum():8.1f}  ({100*perf_term.sum()/regret.sum():.0f}%)  (we missed performance)")
    print(f"  cost overspend term : {cost_term.sum():8.1f}  ({100*cost_term.sum()/regret.sum():.0f}%)  (we overpaid)")
    overspend = cn_c > cn_o + 1e-6
    underserve = perf_c < perf_o - 1e-6
    print(f"  queries where we OVERPAID  (picked pricier than oracle): {overspend.mean()*100:.1f}%")
    print(f"  queries where we UNDER-SERVED (picked lower perf):       {underserve.mean()*100:.1f}%")

    # ---- B. fingerprint: mis-routed vs correct queries ----
    err = regret > 0.05                         # meaningful miss
    F = pd.DataFrame([feats(q) for q in tr["query"]])
    print(f"\n=== B. ERROR FINGERPRINT  (error = regret>0.05, {err.mean()*100:.1f}% of queries) ===")
    print(f"{'feature':16s} {'correct':>10s} {'wrong':>10s} {'ratio':>7s}")
    for col in F.columns:
        a = F.loc[~err, col].mean(); b = F.loc[err, col].mean()
        print(f"{col:16s} {a:10.2f} {b:10.2f} {b/max(a,1e-9):7.2f}x")

    # break errors into the two kinds and fingerprint each
    over_err = err & overspend & ~underserve
    under_err = err & underserve & ~overspend
    print(f"\n  OVERPAY errors: {over_err.sum()}   UNDER-SERVE errors: {under_err.sum()}")
    print(f"{'feature':16s} {'overpay':>10s} {'underserve':>11s}")
    for col in ["len_chars", "n_words", "has_code", "has_math", "nonascii_ratio", "n_digits"]:
        print(f"{col:16s} {F.loc[over_err,col].mean():10.2f} {F.loc[under_err,col].mean():11.2f}")

    # ---- figure: regret split + length profile ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].bar(["perf\nshortfall", "cost\noverspend"],
              [perf_term.sum(), cost_term.sum()], color=["indianred", "goldenrod"])
    ax[0].set_ylabel("total reward lost"); ax[0].set_title("A. Why we lose reward")
    bins = [0, 200, 500, 1000, 2000, 5000, 1e7]; labels = ["<200","200-500","500-1k","1k-2k","2k-5k",">5k"]
    bi = np.digitize(F["len_chars"], bins) - 1
    err_rate = [err[bi == b].mean()*100 for b in range(len(labels))]
    ax[1].bar(labels, err_rate, color="steelblue")
    ax[1].set_ylabel("% mis-routed (regret>0.05)"); ax[1].set_xlabel("query length (chars)")
    ax[1].set_title("B. Error rate vs query length")
    plt.tight_layout(); plt.savefig("notebooks/figs/06_error_profile.png", dpi=110); plt.close()
    print("\nfig -> notebooks/figs/06_error_profile.png")


if __name__ == "__main__":
    main()

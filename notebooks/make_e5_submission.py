"""Build the submission with the winning recipe:
  e5 + surf6 + regularized LR per-model experts (P(perf=1)) + per-query cost regressors,
  routed by argmax(0.85*E[perf] - 0.15*pred_cost) with a CONSERVATIVE K-margin
  (only leave K when the predicted gain over K exceeds `margin`).

Step 1 tunes the margin on 5-fold OOF reward. Step 2 trains on ALL data, predicts test,
writes submission. Conservative routing is expected to lift reward AND shrink the CV->LB
gap (less overfit). Baseline always-K LB = 0.44.
"""
from __future__ import annotations
import re, time, argparse
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold

MODELS = list("ABCDEFGHIJK")
PERF_W, COST_W = 0.85, 0.15
KIDX = MODELS.index("K")
FOLDS = 5
MARGINS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]


def pqnorm(c):
    q = c.max(1, keepdims=True); q[q == 0] = 1.0
    return c / q


def surf6(q):
    return np.array([[min(len(t)/4000., 4.), t.count(" ")/max(len(t), 1),
                      float(bool(re.search(r"[\x60]{3}|def |class |import ", t))),
                      float(len(t) > 2000), float(len(t) > 5000),
                      min(t.count(".")/20., 2.)] for t in q], dtype=np.float32)


def route(eperf, pred_cn, margin):
    score = PERF_W * eperf - COST_W * pred_cn
    best_idx = score.argmax(1); best_val = score.max(1); kscore = score[:, KIDX]
    return np.where(best_val - kscore > margin, best_idx, KIDX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=None,
                    help="override K-margin; default = conservative (best OOF, min 0.03)")
    args = ap.parse_args()

    tr = pd.read_csv("data/train.csv"); te = pd.read_csv("data/test.csv")
    perf = tr[[f"Model_{m}_performance" for m in MODELS]].values
    cost = tr[[f"Model_{m}_cost" for m in MODELS]].values
    cn = pqnorm(cost)
    Xtr = np.hstack([np.load("notebooks/Xtr_e5.npy"), surf6(tr["query"].astype(str))])
    Xte = np.hstack([np.load("notebooks/Xte_e5.npy"), surf6(te["query"].astype(str))])
    print(f"always-K={(PERF_W*perf[:,KIDX]-COST_W*cn[:,KIDX]).mean():.4f}  "
          f"oracle={(PERF_W*perf-COST_W*cn).max(1).mean():.4f}")

    # ---- Step 1: OOF to tune the K-margin ----
    t0 = time.time()
    kf = KFold(FOLDS, shuffle=True, random_state=42)
    oof_p1 = np.zeros((len(tr), 11)); oof_cn = np.zeros((len(tr), 11))
    for tri, vai in kf.split(Xtr):
        for m in range(11):
            ym = (perf[tri, m] == 1.0).astype(int)
            if ym.sum() in (0, len(ym)):
                oof_p1[vai, m] = ym.mean()
            else:
                c = LogisticRegression(C=0.5, max_iter=300); c.fit(Xtr[tri], ym)
                oof_p1[vai, m] = c.predict_proba(Xtr[vai])[:, 1]
            rg = Ridge(alpha=10.0); rg.fit(Xtr[tri], cn[tri, m]); oof_cn[vai, m] = rg.predict(Xtr[vai]).clip(0, 1)
    r = np.arange(len(tr))
    print(f"\nK-margin sweep (OOF reward)  [fit {time.time()-t0:.0f}s]:")
    best = (-1, 0.0)
    for mg in MARGINS:
        ch = route(oof_p1, oof_cn, mg)
        rew = (PERF_W * perf[r, ch] - COST_W * cn[r, ch]).mean()
        nonk = (ch != KIDX).mean() * 100
        mark = ""
        if rew > best[0]: best = (rew, mg)
        print(f"  margin={mg:.3f}  reward={rew:.4f}  non-K={nonk:4.1f}%")
    # conservative default: our routers overfit (CV optimistic), so bias toward staying
    # on K. CV cost of margin<=0.03 is tiny (<0.003) but it cuts risky off-K routes a lot.
    chosen = args.margin if args.margin is not None else max(best[1], 0.03)
    print(f"\nbest OOF margin={best[1]} (reward {best[0]:.4f}); using margin={chosen} for submission")

    # ---- Step 2: train on ALL data, predict test ----
    eperf_te = np.zeros((len(te), 11)); cn_te = np.zeros((len(te), 11))
    for m in range(11):
        ym = (perf[:, m] == 1.0).astype(int)
        if ym.sum() in (0, len(ym)):
            eperf_te[:, m] = ym.mean()
        else:
            c = LogisticRegression(C=0.5, max_iter=300); c.fit(Xtr, ym)
            eperf_te[:, m] = c.predict_proba(Xte)[:, 1]
        rg = Ridge(alpha=10.0); rg.fit(Xtr, cn[:, m]); cn_te[:, m] = rg.predict(Xte).clip(0, 1)
    choice = route(eperf_te, cn_te, chosen)

    out = f"submission_e5_lr_margin{chosen:.3f}.csv"
    sub = pd.DataFrame({"ID": te["ID"], "pred_model": [f"Model_{MODELS[c]}" for c in choice]})
    sub.to_csv(out, index=False)
    print(f"\nwrote {out}")
    print("route dist:", {k: int(v) for k, v in sub['pred_model'].value_counts().items()})
    print(f"non-K on test: {(choice != KIDX).mean()*100:.1f}%")


if __name__ == "__main__":
    main()

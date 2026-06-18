"""Export OOF routing decisions to a browsable CSV for MANUAL pattern analysis.

Open errors_for_review.csv in a spreadsheet, sort by `regret` descending, and read
the actual queries where we lose the most reward. Key columns for spotting patterns:
  success_models : which models actually got perf=1 on this query (the "right answers")
  pred / oracle  : what we picked vs the cheapest-best model
  pred_ok        : 1 if our pick succeeded (perf=1), 0 if it flopped
  error_type     : underserve (missed perf) | overpay (too pricey) | mixed | correct
  pred_p1/oracle_p1 : our predicted P(perf=1) for each -> shows WHY we ranked it that way

Usage:
  python3 notebooks/export_errors.py            # 5-fold OOF (default)
  python3 notebooks/export_errors.py --folds 10 # 10-fold (trains on 90% per fold)
"""
from __future__ import annotations
import argparse, re, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold

MODELS = list("ABCDEFGHIJK")
PERF_W, COST_W = 0.85, 0.15
QCAP = 6000  # cap query text length in the CSV so the file stays openable


def pqnorm(c):
    q = c.max(1, keepdims=True); q[q == 0] = 1.0
    return c / q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="errors_for_review.csv")
    args = ap.parse_args()

    tr = pd.read_csv("data/train.csv")
    q = tr["query"].astype(str)
    perf = tr[[f"Model_{m}_performance" for m in MODELS]].values
    cost = tr[[f"Model_{m}_cost" for m in MODELS]].values
    cn = pqnorm(cost)
    Xtr = np.load("notebooks/Xtr.npy")
    surf = np.array([[min(len(t)/4000., 4.), t.count(" ")/max(len(t), 1),
                      float(bool(re.search(r"[\x60]{3}|def |class |import ", t))),
                      float(len(t) > 2000), float(len(t) > 5000),
                      min(t.count(".")/20., 2.)] for t in q], dtype=np.float32)
    X = np.hstack([Xtr, surf])

    rew_mat = PERF_W * perf - COST_W * cn
    oracle = rew_mat.argmax(1)

    # OOF per-model experts + per-query cost regressors
    kf = KFold(args.folds, shuffle=True, random_state=42)
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
    print(f"OOF {args.folds}-fold fit {time.time()-t0:.0f}s")

    rows = np.arange(len(tr))
    pred_perf = perf[rows, pred]; pred_cn = cn[rows, pred]
    orc_perf = perf[rows, oracle]; orc_cn = cn[rows, oracle]
    regret = rew_mat[rows, oracle] - rew_mat[rows, pred]

    def etype(i):
        if regret[i] <= 0.05: return "correct"
        lo_perf = pred_perf[i] < orc_perf[i] - 1e-6
        hi_cost = pred_cn[i] > orc_cn[i] + 1e-6
        if lo_perf and not hi_cost: return "underserve"
        if hi_cost and not lo_perf: return "overpay"
        return "mixed"

    success_models = ["".join(MODELS[m] for m in range(11) if perf[i, m] == 1.0) for i in rows]

    out = pd.DataFrame({
        "ID": tr["ID"] if "ID" in tr.columns else rows,
        "regret": regret.round(4),
        "error_type": [etype(i) for i in rows],
        "n_success": (perf == 1.0).sum(1),
        "success_models": success_models,
        "pred": [MODELS[m] for m in pred],
        "pred_ok": pred_perf.astype(int) if set(np.unique(perf)) <= {0, 1} else (pred_perf == 1.0).astype(int),
        "pred_perf": pred_perf,
        "pred_cost_norm": pred_cn.round(3),
        "pred_p1": oof_p1[rows, pred].round(3),
        "oracle": [MODELS[m] for m in oracle],
        "oracle_perf": orc_perf,
        "oracle_cost_norm": orc_cn.round(3),
        "oracle_p1": oof_p1[rows, oracle].round(3),
        "len_chars": q.str.len(),
        "has_code": surf[:, 2].astype(int),
        "n_digits_frac": [round(sum(ch.isdigit() for ch in t)/max(len(t), 1), 3) for t in q],
        "nonascii_frac": [round(sum(ord(ch) > 127 for ch in t)/max(len(t), 1), 3) for t in q],
        "query": q.str.slice(0, QCAP),
    })
    out = out.sort_values("regret", ascending=False).reset_index(drop=True)
    out.to_csv(args.out, index=False)

    n_err = (out["regret"] > 0.05).sum()
    print(f"wrote {args.out}  ({len(out)} rows, {n_err} with regret>0.05)")
    print("\nerror_type counts:")
    print(out["error_type"].value_counts().to_string())
    print(f"\nTop reward-leaking queries are at the TOP of the file.")
    print("Tip: filter error_type=='underserve' to see queries where we routed to a model")
    print("     that FAILED while a cheaper one in success_models would have worked.")


if __name__ == "__main__":
    main()

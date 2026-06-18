"""Feature experiment: is per-model success prediction CAPPED at ~0.75 AUC, or can
we push past it? Three axes:
  1. MODEL CAPACITY  : LR vs LightGBM on the SAME bge features (is LR the limit?)
  2. HAND FEATURES   : bge vs bge+rich vs rich-only (do explicit features add signal?)
  3. EMBEDDING       : bge vs e5-large (run after embed_strong.py finishes; --emb e5)

Reports mean per-model OOF AUC (P(perf=1)) and the routing reward for each setup.
If AUC barely moves across all of these -> the problem is information-capped (a big
chunk of per-query success is unpredictable from text). If it jumps -> winnable.
"""
from __future__ import annotations
import argparse, re, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

MODELS = list("ABCDEFGHIJK")
PERF_W, COST_W = 0.85, 0.15


def pqnorm(c):
    q = c.max(1, keepdims=True); q[q == 0] = 1.0
    return c / q


def rich_feats(q):
    """~25 explicit features targeting complexity, structure, and factual-recall cues."""
    out = []
    QW = ["what", "when", "where", "who", "whom", "which", "how", "why"]
    for t in q:
        t = str(t); n = max(len(t), 1); low = t.lower()
        words = t.split(); nw = max(len(words), 1)
        caps = sum(1 for w in words if w[:1].isupper())
        digits = sum(ch.isdigit() for ch in t)
        nonascii = sum(ord(ch) > 127 for ch in t)
        out.append([
            min(len(t)/4000., 5.), np.log1p(len(t)), nw/100.,
            np.mean([len(w) for w in words]) if words else 0.,
            t.count(" ")/n, t.count("\n")/n, t.count(".")/nw, t.count(",")/nw,
            # code / structure
            float("```" in t), float(bool(re.search(r"\bdef \b|\bclass \b|import |#include|public |function ", t))),
            (t.count("{")+t.count("}")+t.count(";"))/n, t.count("(")/n,
            # math
            float(bool(re.search(r"\\frac|\\sum|\\int|\\begin|\\sqrt", t))),
            float(bool(re.search(r"\d\s*[+\-*/=^]\s*\d", t))), digits/n,
            # multiple choice / instructions
            float(bool(re.search(r"answer choices|^[A-J]\.\s", t, re.I|re.M))),
            float("```" in t or "step by step" in low or "explain" in low),
            # question type (factual-recall cues)
            float(any(low.startswith(w) for w in QW)),
            float(low.startswith(("how many", "how much", "what year", "in what year", "what was", "who was"))),
            sum(low.count(w) for w in QW)/nw,
            caps/nw,                                # proper-noun density (factual)
            float(bool(re.search(r"\b(19|20)\d\d\b", t))),  # contains a year
            # language
            nonascii/n, float(bool(re.search(r"[一-鿿]", t))),
            float(bool(re.search(r"[Ѐ-ӿ]", t))),
            float("http" in low),
        ])
    return np.array(out, dtype=np.float32)


def evaluate(X, perf, cost, cn, mean_cn_unused, folds, model="lr", tag=""):
    rew_mat = PERF_W * perf - COST_W * cn
    oracle = rew_mat.argmax(1)
    kf = KFold(folds, shuffle=True, random_state=42)
    oof_p1 = np.zeros((len(perf), 11)); oof_cn = np.zeros((len(perf), 11))
    for tri, vai in kf.split(X):
        for m in range(11):
            ym = (perf[tri, m] == 1.0).astype(int)
            if ym.sum() in (0, len(ym)):
                oof_p1[vai, m] = ym.mean()
            elif model == "lr":
                c = LogisticRegression(C=0.5, max_iter=300); c.fit(X[tri], ym)
                oof_p1[vai, m] = c.predict_proba(X[vai])[:, 1]
            else:  # lightgbm
                c = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.6, verbose=-1)
                c.fit(X[tri], ym)
                oof_p1[vai, m] = c.predict_proba(X[vai])[:, 1]
            rg = Ridge(alpha=10.0); rg.fit(X[tri], cn[tri, m])
            oof_cn[vai, m] = rg.predict(X[vai]).clip(0, 1)
    aucs = [roc_auc_score((perf[:, m] == 1.0).astype(int), oof_p1[:, m])
            for m in range(11) if 0 < (perf[:, m] == 1.0).sum() < len(perf)]
    pred = (PERF_W * oof_p1 - COST_W * oof_cn).argmax(1)
    rows = np.arange(len(perf))
    reward = (PERF_W * perf[rows, pred] - COST_W * cn[rows, pred]).mean()
    print(f"  {tag:38s} meanAUC={np.mean(aucs):.4f}  reward={reward:.4f}")
    return np.mean(aucs), reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--emb", default="bge", help="bge | e5 (needs Xtr_e5.npy)")
    args = ap.parse_args()

    tr = pd.read_csv("data/train.csv"); q = tr["query"].astype(str)
    perf = tr[[f"Model_{m}_performance" for m in MODELS]].values
    cost = tr[[f"Model_{m}_cost" for m in MODELS]].values
    cn = pqnorm(cost); mean_cn = cn.mean(0)

    bge = np.load("notebooks/Xtr.npy")
    surf6 = np.array([[min(len(t)/4000., 4.), t.count(" ")/max(len(t), 1),
                       float(bool(re.search(r"[\x60]{3}|def |class |import ", t))),
                       float(len(t) > 2000), float(len(t) > 5000),
                       min(t.count(".")/20., 2.)] for t in q], dtype=np.float32)
    t0 = time.time(); rich = rich_feats(q); print(f"rich feats {rich.shape} in {time.time()-t0:.0f}s")
    # standardize rich for LR
    rich_z = (rich - rich.mean(0)) / (rich.std(0) + 1e-6)

    emb = bge if args.emb == "bge" else np.load(f"notebooks/Xtr_{args.emb}.npy")
    print(f"\nEMBEDDING = {args.emb}  ({emb.shape[1]}-d)")
    print(f"reference:  always-K reward={ (PERF_W*perf[:,MODELS.index('K')]-COST_W*cn[:,MODELS.index('K')]).mean():.4f}"
          f"   oracle={(PERF_W*perf-COST_W*cn).max(1).mean():.4f}")

    print("\n--- axis 2: HAND FEATURES (LogisticRegression) ---")
    evaluate(emb, perf, cost, cn, mean_cn, args.folds, "lr", f"{args.emb} only")
    evaluate(np.hstack([emb, surf6]), perf, cost, cn, mean_cn, args.folds, "lr", f"{args.emb} + surf6 (baseline)")
    evaluate(np.hstack([emb, rich_z]), perf, cost, cn, mean_cn, args.folds, "lr", f"{args.emb} + rich(~25)")
    evaluate(rich_z, perf, cost, cn, mean_cn, args.folds, "lr", "rich only (no embedding)")

    print("\n--- axis 1: MODEL CAPACITY (same features: emb+rich) ---")
    evaluate(np.hstack([emb, rich_z]), perf, cost, cn, mean_cn, args.folds, "lgbm", f"{args.emb}+rich  LightGBM")


if __name__ == "__main__":
    main()

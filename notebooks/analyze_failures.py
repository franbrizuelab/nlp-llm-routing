"""Failure-analysis dashboard for the router. Produces PNGs in notebooks/figs/.

Goal: see WHERE reward leaks so we can build a tailored model.
Everything uses OOF (out-of-fold) predictions so it reflects generalization,
not training fit. Metric = per-query cost normalization (the real one).
"""
from __future__ import annotations
import os, re, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, confusion_matrix

MODELS = list("ABCDEFGHIJK")
PERF_W, COST_W = 0.85, 0.15
FIGDIR = "notebooks/figs"
os.makedirs(FIGDIR, exist_ok=True)


def pqnorm(c):
    q = c.max(1, keepdims=True); q[q == 0] = 1.0
    return c / q


def sf(t):
    t = str(t); n = max(len(t), 1)
    return [min(len(t)/4000., 4.), t.count(" ")/n,
            float(bool(re.search(r"[\x60]{3}|def |class |import ", t))),
            float(len(t) > 2000), float(len(t) > 5000), min(t.count(".")/20., 2.)]


def main():
    tr = pd.read_csv("data/train.csv")
    perf = tr[[f"Model_{m}_performance" for m in MODELS]].values
    cost = tr[[f"Model_{m}_cost" for m in MODELS]].values
    cn = pqnorm(cost); mean_nc = cn.mean(0)
    Xtr = np.load("notebooks/Xtr.npy")
    surf = np.array([sf(q) for q in tr["query"]], dtype=np.float32)
    X = np.hstack([Xtr, surf])
    qlen = np.array([len(str(q)) for q in tr["query"]])

    rew_mat = PERF_W * perf - COST_W * cn          # (N,11) true reward per model
    oracle = rew_mat.argmax(1)
    oracle_rew = rew_mat.max(1)
    kidx = MODELS.index("K")

    # ---- OOF per-model experts: P(perf_m == 1) ----
    kf = KFold(5, shuffle=True, random_state=42)
    oof_p1 = np.zeros((len(tr), 11))
    per_model_auc = np.zeros(11)
    t0 = time.time()
    for tri, vai in kf.split(X):
        for m in range(11):
            ym = (perf[tri, m] == 1.0).astype(int)
            if ym.sum() in (0, len(ym)):
                oof_p1[vai, m] = ym.mean(); continue
            clf = LogisticRegression(C=0.5, max_iter=300)
            clf.fit(X[tri], ym)
            oof_p1[vai, m] = clf.predict_proba(X[vai])[:, 1]
    for m in range(11):
        yb = (perf[:, m] == 1.0).astype(int)
        per_model_auc[m] = roc_auc_score(yb, oof_p1[:, m]) if 0 < yb.sum() < len(yb) else 0.5
    print(f"OOF experts: {time.time()-t0:.0f}s  mean AUC={per_model_auc.mean():.3f}")

    score = PERF_W * oof_p1 - COST_W * mean_nc[None, :]
    pred = score.argmax(1)
    pred_rew = rew_mat[np.arange(len(tr)), pred]
    regret = oracle_rew - pred_rew                 # >=0, how much we lose per query

    # ============ FIG 1: reward vs K-upgrade margin ============
    kscore = score[:, kidx]; best = score.max(1); bestidx = score.argmax(1)
    margins = np.linspace(0, 0.20, 41); rewards = []
    for mg in margins:
        ch = np.where(best - kscore > mg, bestidx, kidx)
        r = (PERF_W * perf[np.arange(len(tr)), ch] - COST_W * cn[np.arange(len(tr)), ch]).mean()
        rewards.append(r)
    plt.figure(figsize=(7, 4))
    plt.plot(margins, rewards, lw=2)
    plt.axhline(rew_mat[:, kidx].mean(), ls="--", c="gray", label="always-K")
    plt.axhline(oracle_rew.mean(), ls=":", c="green", label="oracle")
    plt.xlabel("upgrade margin (only leave K if gain > margin)")
    plt.ylabel("CV reward"); plt.title("How aggressive should we be about leaving K?")
    plt.legend(); plt.tight_layout(); plt.savefig(f"{FIGDIR}/01_margin_sweep.png", dpi=110); plt.close()

    # ============ FIG 2: reward leak by oracle model ============
    leak = np.zeros(11)
    for m in range(11):
        leak[m] = regret[oracle == m].sum()
    plt.figure(figsize=(7, 4))
    bars = plt.bar(MODELS, leak / regret.sum() * 100,
                   color=["crimson" if MODELS[i] == "K" else "steelblue" for i in range(11)])
    plt.ylabel("% of total reward loss"); plt.xlabel("true oracle model")
    plt.title("Where reward leaks: queries whose BEST model is X but we mis-route")
    plt.tight_layout(); plt.savefig(f"{FIGDIR}/02_leak_by_oracle.png", dpi=110); plt.close()

    # ============ FIG 3: confusion (oracle vs predicted) ============
    cm = confusion_matrix(oracle, pred, labels=range(11))
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(cmn, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="row-normalized")
    plt.xticks(range(11), MODELS); plt.yticks(range(11), MODELS)
    plt.xlabel("predicted route"); plt.ylabel("true oracle")
    plt.title("Routing confusion (OOF)")
    for i in range(11):
        for j in range(11):
            if cmn[i, j] > 0.02:
                plt.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                         color="white" if cmn[i, j] < 0.6 else "black", fontsize=7)
    plt.tight_layout(); plt.savefig(f"{FIGDIR}/03_confusion.png", dpi=110); plt.close()

    # ============ FIG 4: per-model success predictability ============
    plt.figure(figsize=(7, 4))
    order = np.argsort(-per_model_auc)
    plt.bar([MODELS[i] for i in order], per_model_auc[order], color="teal")
    plt.axhline(0.5, ls="--", c="gray")
    plt.ylim(0.4, 1.0); plt.ylabel("OOF AUC  P(perf=1)")
    plt.title("How predictable is each model's success from the query?")
    plt.tight_layout(); plt.savefig(f"{FIGDIR}/04_model_predictability.png", dpi=110); plt.close()

    # ============ FIG 5: regret vs query length ============
    plt.figure(figsize=(7, 4))
    bins = np.array([0, 200, 500, 1000, 2000, 5000, 1e7])
    labels = ["<200", "200-500", "500-1k", "1k-2k", "2k-5k", ">5k"]
    idx = np.digitize(qlen, bins) - 1
    meanreg = [regret[idx == b].mean() for b in range(len(labels))]
    cnts = [(idx == b).sum() for b in range(len(labels))]
    plt.bar(labels, meanreg, color="indianred")
    for i, c in enumerate(cnts):
        plt.text(i, meanreg[i], f"n={c}", ha="center", va="bottom", fontsize=8)
    plt.ylabel("mean regret (reward lost)"); plt.xlabel("query length (chars)")
    plt.title("Do we fail more on long or short queries?")
    plt.tight_layout(); plt.savefig(f"{FIGDIR}/05_regret_by_length.png", dpi=110); plt.close()

    # ---- text summary ----
    print("\n=== SUMMARY ===")
    print(f"oracle reward      {oracle_rew.mean():.4f}")
    print(f"always-K           {rew_mat[:, kidx].mean():.4f}")
    print(f"expert route       {pred_rew.mean():.4f}")
    print(f"total regret       {regret.sum():.1f}  (mean {regret.mean():.4f}/query)")
    print(f"queries w/ regret>0: {(regret > 1e-6).mean()*100:.1f}%")
    print("\nWorst-routed oracle models (reward leak share):")
    for i in np.argsort(-leak)[:5]:
        print(f"  oracle={MODELS[i]}: {leak[i]/regret.sum()*100:4.1f}%  "
              f"(n={int((oracle==i).sum())}, we send them to "
              f"{MODELS[np.bincount(pred[oracle==i], minlength=11).argmax()]})")
    print(f"\nfigs written to {FIGDIR}/")


if __name__ == "__main__":
    main()

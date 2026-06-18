"""Two tuned knobs, both measured on the corrected per-query CV (our ground truth):

  PART A  regret cranking : sweep sample-weight base x power on the per-model LR experts.
  PART B  stage-1 TTA     : variance-reduction by averaging predictions across embedders
                            (bge + e5) -- free, no paraphraser. Gates whether paid
                            LLM-paraphrase TTA is worth trying.

Everything reported as 5-fold OOF routing reward (real metric). Baseline to beat:
  e5 + surf6 + LR = 0.4834 ;  always-K = 0.4509 ;  oracle = 0.6774
"""
from __future__ import annotations
import re, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold

MODELS = list("ABCDEFGHIJK")
PERF_W, COST_W = 0.85, 0.15
KIDX = MODELS.index("K")
FOLDS = 5


def pqnorm(c):
    q = c.max(1, keepdims=True); q[q == 0] = 1.0
    return c / q


def regret_weights(perf, cn, base, power):
    rew = PERF_W * perf - COST_W * cn
    adv = np.clip(rew.max(1) - rew[:, KIDX], 0, None)      # how much oracle beats K
    ratio = adv / (adv.mean() + 1e-9)
    w = base + ratio ** power
    return (w / w.mean()).astype(np.float32)


def fit_experts(X, perf, cn, sw=None):
    """OOF per-model P(perf=1) and per-query normalized cost."""
    kf = KFold(FOLDS, shuffle=True, random_state=42)
    oof_p1 = np.zeros((len(perf), 11)); oof_cn = np.zeros((len(perf), 11))
    for tri, vai in kf.split(X):
        wt = None if sw is None else sw[tri]
        for m in range(11):
            ym = (perf[tri, m] == 1.0).astype(int)
            if ym.sum() in (0, len(ym)):
                oof_p1[vai, m] = ym.mean()
            else:
                c = LogisticRegression(C=0.5, max_iter=300)
                c.fit(X[tri], ym, sample_weight=wt)
                oof_p1[vai, m] = c.predict_proba(X[vai])[:, 1]
            rg = Ridge(alpha=10.0); rg.fit(X[tri], cn[tri, m])
            oof_cn[vai, m] = rg.predict(X[vai]).clip(0, 1)
    return oof_p1, oof_cn


def reward(oof_p1, oof_cn, perf, cn):
    choice = (PERF_W * oof_p1 - COST_W * oof_cn).argmax(1)
    r = np.arange(len(perf))
    return (PERF_W * perf[r, choice] - COST_W * cn[r, choice]).mean()


def main():
    tr = pd.read_csv("data/train.csv"); q = tr["query"].astype(str)
    perf = tr[[f"Model_{m}_performance" for m in MODELS]].values
    cost = tr[[f"Model_{m}_cost" for m in MODELS]].values
    cn = pqnorm(cost)
    surf6 = np.array([[min(len(t)/4000., 4.), t.count(" ")/max(len(t), 1),
                       float(bool(re.search(r"[\x60]{3}|def |class |import ", t))),
                       float(len(t) > 2000), float(len(t) > 5000),
                       min(t.count(".")/20., 2.)] for t in q], dtype=np.float32)
    bge = np.hstack([np.load("notebooks/Xtr.npy"), surf6])
    e5 = np.hstack([np.load("notebooks/Xtr_e5.npy"), surf6])
    print(f"always-K={(PERF_W*perf[:,KIDX]-COST_W*cn[:,KIDX]).mean():.4f}  "
          f"oracle={(PERF_W*perf-COST_W*cn).max(1).mean():.4f}\n")

    # --- baseline (reused) ---
    t0 = time.time()
    e5_p1, e5_cn = fit_experts(e5, perf, cn)
    base_rew = reward(e5_p1, e5_cn, perf, cn)
    print(f"BASELINE  e5+surf6, no weight        reward={base_rew:.4f}  ({time.time()-t0:.0f}s)\n")

    # --- PART A: regret cranking grid ---
    print("=== PART A: REGRET CRANKING (sample_weight on e5+surf6 experts) ===")
    print(f"{'base':>6}{'power':>7}{'reward':>10}{'delta':>9}")
    bestA = (base_rew, "none")
    for base in (0.15, 0.05, 0.0):
        for power in (1.0, 1.5, 2.0):
            sw = regret_weights(perf, cn, base, power)
            p1, cnr = fit_experts(e5, perf, cn, sw=sw)
            r = reward(p1, cnr, perf, cn)
            flag = "  <-- best" if r > bestA[0] else ""
            if r > bestA[0]: bestA = (r, f"base={base} power={power}")
            print(f"{base:6.2f}{power:7.1f}{r:10.4f}{r-base_rew:+9.4f}{flag}")
    print(f"best regret config: {bestA[1]}  reward={bestA[0]:.4f}\n")

    # --- PART B: stage-1 TTA (embedder ensemble, free) ---
    print("=== PART B: STAGE-1 TTA (average across embedders, no paraphraser) ===")
    bge_p1, bge_cn = fit_experts(bge, perf, cn)
    print(f"  bge+surf6 alone                    reward={reward(bge_p1, bge_cn, perf, cn):.4f}")
    print(f"  e5+surf6 alone                     reward={base_rew:.4f}")
    ens_p1 = (bge_p1 + e5_p1) / 2; ens_cn = (bge_cn + e5_cn) / 2
    print(f"  ENSEMBLE avg(bge,e5) predictions   reward={reward(ens_p1, ens_cn, perf, cn):.4f}")
    concat_p1, concat_cn = fit_experts(np.hstack([np.load('notebooks/Xtr.npy'),
                                                  np.load('notebooks/Xtr_e5.npy'), surf6]), perf, cn)
    print(f"  CONCAT [bge,e5,surf6] single LR    reward={reward(concat_p1, concat_cn, perf, cn):.4f}\n")

    # --- PART C: best regret + ensemble together ---
    if bestA[1] != "none":
        b, p = [float(x.split("=")[1]) for x in bestA[1].split()]
        sw = regret_weights(perf, cn, b, p)
        e5w_p1, e5w_cn = fit_experts(e5, perf, cn, sw=sw)
        bgew_p1, bgew_cn = fit_experts(bge, perf, cn, sw=sw)
        cp = reward((e5w_p1 + bgew_p1)/2, (e5w_cn + bgew_cn)/2, perf, cn)
        print(f"=== PART C: best-regret + ensemble  reward={cp:.4f} ===")


if __name__ == "__main__":
    main()

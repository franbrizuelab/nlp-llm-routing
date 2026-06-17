"""Head-to-head CV comparison of routing strategies on Reward_{0.85}.

Phases implemented:
  1 - clf:         flat LightGBM classifier on bge-large embeddings
  3 - thresh:      confidence fallback to const_F (swept on best clf variant)
  4 - _w:          regret-weighted training (weight = reward_oracle - reward_F, norm to [0.1,1])
  5 - hier:        2-stage hierarchical: K vs not-K → sub-classify core models
  6 - _aug:        hand-crafted features concatenated to embedding

Run embed.py first to generate notebooks/Xtr.npy.
"""
import re, time
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

PW, CW   = 0.85, 0.15
MODELS   = list("ABCDEFGHIJK")
K_IDX    = MODELS.index('K')                          # 10
TAIL     = {MODELS.index(m) for m in ['A','B','G','I']}  # collapse to const_F in Stage 2
THRESHOLDS = np.linspace(0.10, 0.99, 20)

tr   = pd.read_csv("data/train.csv")
P    = tr[[f"Model_{m}_performance" for m in MODELS]].to_numpy(float)
C    = tr[[f"Model_{m}_cost"        for m in MODELS]].to_numpy(float)
X    = np.load("notebooks/Xtr.npy")
CMAX = C.max()
N, M = P.shape   # 10182 × 11

# ── reward helpers ────────────────────────────────────────────────────────────

def reward_vec(choices, idx):
    r = np.arange(len(idx))
    return PW * P[idx][r, choices] - CW * C[idx][r, choices] / CMAX

def reward(choices, idx):
    return reward_vec(choices, idx).mean()

def oracle_label(idx):
    return (PW * P[idx] - CW * C[idx] / CMAX).argmax(1)

ORACLE     = oracle_label(np.arange(N))
BEST_CONST = max(range(M), key=lambda j: PW*P[:,j].mean() - CW*C[:,j].mean()/CMAX)

# ── Phase 6: hand-crafted features ───────────────────────────────────────────

def hand_features(texts):
    """6 trivial features: char_len, token_count, has_code, has_math,
    sentence_count, avg_word_len. Scaled per CV fold before use."""
    out = []
    for t in texts:
        toks = t.split()
        out.append([
            float(len(t)),
            float(len(toks)),
            float(bool(re.search(r'```', t) or re.search(r'^    \S', t, re.MULTILINE))),
            float(bool(re.search(r'[\$\\]|\b\d[\+\-\*/=]\d|\bequation\b', t))),
            float(len(re.split(r'[.!?]+', t))),
            float(np.mean([len(w) for w in toks])) if toks else 0.0,
        ])
    return np.array(out, dtype=np.float32)

print("computing hand features...", flush=True)
HAND = hand_features(tr['query'].tolist())   # (N, 6)

# ── Phase 4: regret weights ───────────────────────────────────────────────────

def regret_weights(idx):
    """Per-query: reward(oracle) - reward(const_F), clipped ≥0, norm to [0.1, 1.0]."""
    r_oracle = reward_vec(oracle_label(idx), idx)
    r_f      = reward_vec(np.full(len(idx), BEST_CONST), idx)
    w = np.clip(r_oracle - r_f, 0, None)
    wmax = w.max()
    return (0.1 + 0.9 * w / wmax) if wmax > 0 else np.ones(len(idx))

# ── classifier builders ───────────────────────────────────────────────────────

def make_lgbm():
    return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=63,
                               subsample=0.8, colsample_bytree=0.8,
                               n_jobs=-1, verbose=-1)

def flat_clf(Xtr, ytr, Xva, weights=None):
    m = make_lgbm()
    m.fit(Xtr, ytr, sample_weight=weights)
    return m.predict(Xva), m.predict_proba(Xva), m.classes_

# ── Phase 5: hierarchical routing ────────────────────────────────────────────

def hierarchical_clf(Xtr, ytr, Xva, weights=None):
    # Stage 1 — binary: K vs not-K
    s1 = make_lgbm()
    s1.fit(Xtr, (ytr == K_IDX).astype(int), sample_weight=weights)
    is_k = s1.predict(Xva).astype(bool)

    # Stage 2 — multi-class: core non-K models (A/B/G/I collapsed to BEST_CONST)
    nk_mask = ytr != K_IDX
    y2 = ytr[nk_mask].copy()
    for t in TAIL:
        y2[y2 == t] = BEST_CONST
    w2 = weights[nk_mask] if weights is not None else None
    s2 = make_lgbm()
    s2.fit(Xtr[nk_mask], y2, sample_weight=w2)

    s2_pred = s2.predict(Xva)
    return np.where(is_k, K_IDX, s2_pred).astype(int)

# ── Phase 3: confidence threshold fallback ────────────────────────────────────

def threshold_route(proba, classes, t):
    pred = classes[proba.argmax(axis=1)]
    return np.where(proba.max(axis=1) >= t, pred, BEST_CONST).astype(int)

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    kf  = KFold(n_splits=5, shuffle=True, random_state=42)

    strat_keys = ["const_F", "oracle",
                  "clf", "clf_w", "clf_aug", "clf_w_aug",
                  "hier", "hier_w", "hier_aug", "hier_w_aug"]
    thresh_keys = [f"th{t:.2f}" for t in THRESHOLDS]
    res = {k: [] for k in strat_keys + thresh_keys}

    for fi, (ti, vi) in enumerate(kf.split(X)):
        t0 = time.time()

        res["const_F"].append(reward(np.full(len(vi), BEST_CONST), vi))
        res["oracle"].append(reward(ORACLE[vi], vi))

        # per-fold hand-feature scaler (fit on train fold only)
        scaler  = StandardScaler().fit(HAND[ti])
        Xtr_aug = np.hstack([X[ti], scaler.transform(HAND[ti])])
        Xva_aug = np.hstack([X[vi], scaler.transform(HAND[vi])])

        # Phase 4: regret weights for this fold
        w = regret_weights(ti)

        # ── flat classifier (4 variants) ──────────────────────────────────
        pred,     proba,  cls  = flat_clf(X[ti],     ORACLE[ti], X[vi])
        pred_w,   _,      _    = flat_clf(X[ti],     ORACLE[ti], X[vi],     weights=w)
        pred_aug, _,      _    = flat_clf(Xtr_aug,   ORACLE[ti], Xva_aug)
        pred_w_aug, proba_w_aug, cls_w_aug = flat_clf(Xtr_aug, ORACLE[ti], Xva_aug, weights=w)

        res["clf"].append(reward(pred, vi))
        res["clf_w"].append(reward(pred_w, vi))
        res["clf_aug"].append(reward(pred_aug, vi))
        res["clf_w_aug"].append(reward(pred_w_aug, vi))

        # Phase 3 threshold sweep on clf_w_aug (best expected variant)
        for t, tk in zip(THRESHOLDS, thresh_keys):
            res[tk].append(reward(threshold_route(proba_w_aug, cls_w_aug, t), vi))

        # ── hierarchical (4 variants) ─────────────────────────────────────
        res["hier"].append(reward(hierarchical_clf(X[ti],   ORACLE[ti], X[vi]),           vi))
        res["hier_w"].append(reward(hierarchical_clf(X[ti],   ORACLE[ti], X[vi], w),      vi))
        res["hier_aug"].append(reward(hierarchical_clf(Xtr_aug, ORACLE[ti], Xva_aug),     vi))
        res["hier_w_aug"].append(reward(hierarchical_clf(Xtr_aug, ORACLE[ti], Xva_aug, w), vi))

        print(f"fold {fi} done {time.time()-t0:.0f}s", flush=True)

    # ── results ───────────────────────────────────────────────────────────────
    const_r = np.mean(res["const_F"])
    print(f"\nbest constant = Model_{MODELS[BEST_CONST]}  ({const_r:.4f})")
    print(f"\n{'strategy':<16} {'mean reward':>12} {'vs const_F':>12} {'std':>8}")
    for k in sorted(strat_keys, key=lambda k: -np.mean(res[k])):
        mean = np.mean(res[k])
        marker = "  ***" if mean > const_r else ""
        print(f"{k:<16} {mean:>12.4f} {mean-const_r:>+12.4f} {np.std(res[k]):>8.4f}{marker}")

    print(f"\n--- threshold sweep on clf_w_aug (fallback → Model_{MODELS[BEST_CONST]}) ---")
    print(f"{'threshold':<12} {'mean reward':>12} {'vs const_F':>12}")
    best_tk = max(thresh_keys, key=lambda k: np.mean(res[k]))
    for t, tk in zip(THRESHOLDS, thresh_keys):
        mean = np.mean(res[tk])
        marker = " <--" if tk == best_tk else ""
        print(f"{t:<12.2f} {mean:>12.4f} {mean-const_r:>+12.4f}{marker}")

if __name__ == "__main__":
    main()

"""Build submission.csv using clf_w_aug at threshold=0.99.

Strategy: LightGBM classifier trained on bge-large embeddings + hand features,
with regret-weighted samples. Route to predicted model only when confidence >= 0.99,
else fall back to Model_F (best constant baseline). CV reward: 0.5027 vs 0.4936 const.
"""
import re
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

THRESHOLD  = 0.99
PW, CW     = 0.85, 0.15
MODELS     = list("ABCDEFGHIJK")
BEST_CONST = "Model_F"
BEST_CONST_IDX = MODELS.index('F')

# ── load data ─────────────────────────────────────────────────────────────────
tr = pd.read_csv("data/train.csv")
te = pd.read_csv("data/test.csv")
Xtr = np.load("notebooks/Xtr.npy")
Xte = np.load("notebooks/Xte.npy")

P    = tr[[f"Model_{m}_performance" for m in MODELS]].to_numpy(float)
C    = tr[[f"Model_{m}_cost"        for m in MODELS]].to_numpy(float)
CMAX = C.max()
N    = len(tr)

print(f"train: {Xtr.shape}  test: {Xte.shape}")

# ── Phase 6: hand features ────────────────────────────────────────────────────
def hand_features(texts):
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
hand_tr = hand_features(tr['query'].tolist())
hand_te = hand_features(te['query'].tolist())

scaler  = StandardScaler().fit(hand_tr)
Xtr_aug = np.hstack([Xtr, scaler.transform(hand_tr)])
Xte_aug = np.hstack([Xte, scaler.transform(hand_te)])
print(f"augmented features: {Xtr_aug.shape[1]} dims")

# ── Phase 4: regret weights ───────────────────────────────────────────────────
oracle_idx = (PW * P - CW * C / CMAX).argmax(1)
r_oracle   = PW * P[np.arange(N), oracle_idx] - CW * C[np.arange(N), oracle_idx] / CMAX
r_f        = PW * P[:, BEST_CONST_IDX]        - CW * C[:, BEST_CONST_IDX]        / CMAX
w = np.clip(r_oracle - r_f, 0, None)
wmax = w.max()
weights = (0.1 + 0.9 * w / wmax) if wmax > 0 else np.ones(N)
print(f"regret weights: min={weights.min():.3f} max={weights.max():.3f} mean={weights.mean():.3f}")

# ── train on full data ────────────────────────────────────────────────────────
print("training classifier on full training data...", flush=True)
clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=63,
                          subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1)
clf.fit(Xtr_aug, oracle_idx, sample_weight=weights)

# ── predict with confidence threshold ────────────────────────────────────────
print(f"predicting with threshold={THRESHOLD}...", flush=True)
proba      = clf.predict_proba(Xte_aug)
max_conf   = proba.max(axis=1)
pred_local = proba.argmax(axis=1)
pred_idx   = clf.classes_[pred_local]

routed     = (max_conf >= THRESHOLD).sum()
choice_idx = np.where(max_conf >= THRESHOLD, pred_idx, BEST_CONST_IDX)
choice_str = [f"Model_{MODELS[i]}" for i in choice_idx]

print(f"routed away from const_F: {routed}/{len(te)} ({100*routed/len(te):.1f}%)")
print(f"routing distribution: { {f'Model_{MODELS[k]}': int((choice_idx==k).sum()) for k in np.unique(choice_idx)} }")

# ── write submission ──────────────────────────────────────────────────────────
sub = pd.DataFrame({"ID": te["ID"], "pred_model": choice_str})
sub.to_csv("submission.csv", index=False)
print(f"\nwrote submission.csv ({len(sub)} rows)")
print(sub["pred_model"].value_counts().to_string())

"""Part B: fine-tune DeBERTa-v3 to predict per-model ternary performance, then
route by the metric-invariant oracle rule (cheapest model among predicted-best).

Why this design (see memory + HANDOFF):
- Performance is DISCRETE {0.0, 0.5, 1.0} -> model it as 3-class-per-model, not regression.
- The real Kaggle metric penalizes cost PER-QUERY -> Model_F (priciest) is a trap,
  Model_K (cheapest-decent) is the floor. So we never fall back to F; we fall back to K,
  and validation reward uses per-query cost normalization (matches the leaderboard).
- The oracle LABEL (cheapest among the perf-best) is metric-invariant, so predicting
  performance well + this routing rule is correct under the true scoring.

Run on the 3080:
    pip install -q torch transformers sentencepiece scikit-learn pandas numpy
    python notebooks/train_deberta_router.py                 # base, holdout
    python notebooks/train_deberta_router.py --large --epochs 4
    python notebooks/train_deberta_router.py --cv            # 5-fold reward CV

Outputs:
    submission.csv          (route for data/test.csv, K-fallback when unsure)
    notebooks/deberta_router.pt   (best weights by val reward)
"""
from __future__ import annotations
import argparse, re, time, math
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import KFold

MODELS = list("ABCDEFGHIJK")                       # Model_A .. Model_K
PERF_VALUES = np.array([0.0, 0.5, 1.0])            # class idx -> perf
PERF_W, COST_W = 0.85, 0.15

# ---- routing policy knobs (tune on CV) ----
EPS_TIE   = 1e-3   # models within EPS_TIE of best predicted E[perf] count as tied -> pick cheapest
TAU_WEAK  = 0.34   # if best predicted E[perf] < TAU_WEAK, assume unsolvable -> route K (cheapest)
KFALL     = "K"    # safety fallback model (NOT F: F is the most cost-penalized)

# ---------------------------------------------------------------- data utils
def perquery_cost_norm(cost_mat: np.ndarray) -> np.ndarray:
    """Cost / per-query max-over-models. This is the normalization that reproduces
    the leaderboard (always-F ~ 0.37, always-K ~ 0.45)."""
    qmax = cost_mat.max(1, keepdims=True); qmax[qmax == 0] = 1.0
    return cost_mat / qmax

def reward_of_choice(perf_mat, cost_mat, choice_idx) -> float:
    cn = perquery_cost_norm(cost_mat)
    rows = np.arange(len(choice_idx))
    return PERF_W * perf_mat[rows, choice_idx].mean() - COST_W * cn[rows, choice_idx].mean()

def route_from_eperf(eperf: np.ndarray, med_cost: np.ndarray) -> np.ndarray:
    """eperf: (N,11) predicted E[perf]. Return per-row model index.
    Rule: cheapest model among predicted-best; if all weak, fall back to K."""
    N = len(eperf); choice = np.empty(N, dtype=int)
    kidx = MODELS.index(KFALL)
    for i in range(N):
        best = eperf[i].max()
        if best < TAU_WEAK:
            choice[i] = kidx; continue
        cand = np.where(eperf[i] >= best - EPS_TIE)[0]
        choice[i] = cand[np.argmin(med_cost[cand])]
    return choice

_RE_BIGNUM  = re.compile(r"[0-9]{40,}")
_RE_BIGB64  = re.compile(r"[A-Za-z0-9+/=]{120,}")
_RE_BIGWORD = re.compile(r"\S{200,}")
def clean_payload(text: str) -> str:
    """Collapse base64/giant digit/JSON-ish blobs so the encoder spends attention on
    instructions, not payload, and learns 'this needs a big-context model'."""
    t = str(text)
    t = _RE_BIGB64.sub(" <LARGE_DATA_BLOCK> ", t)
    t = _RE_BIGNUM.sub(" <BIG_NUMBER> ", t)
    t = _RE_BIGWORD.sub(" <LONG_TOKEN> ", t)
    return t

def surface_feats(text: str) -> np.ndarray:
    t = str(text); n = max(len(t), 1)
    return np.array([
        min(len(t) / 4000.0, 4.0),                       # length (clipped)
        t.count(" ") / n,                                # word density
        float(bool(re.search(r"```|def |class |import |;\s*$", t))),  # code
        float(bool(re.search(r"\$.*\$|\\\(|\\begin|\\frac", t))),     # latex/math
        float(bool(re.search(r"[一-鿿぀-ヿЀ-ӿ]", t))),  # CJK/Cyrillic
        min(t.count(".") / 20.0, 2.0),                   # sentence-ish count
    ], dtype=np.float32)
NUM_SURF = 6

class QueryDS(Dataset):
    def __init__(self, texts, tok, max_len, targets=None, weights=None):
        self.tok, self.max_len = tok, max_len
        self.texts = [clean_payload(x) for x in texts]
        self.surf = np.stack([surface_feats(x) for x in texts])
        self.targets = targets; self.weights = weights
    def __len__(self): return len(self.texts)
    def _pack(self, text):
        ids = self.tok.encode(text, add_special_tokens=False)
        if len(ids) > self.max_len - 2:                  # head+tail packing
            h = (self.max_len - 2) // 2
            ids = ids[:h] + ids[-(self.max_len - 2 - h):]
        ids = [self.tok.cls_token_id] + ids + [self.tok.sep_token_id]
        mask = [1] * len(ids)
        pad = self.max_len - len(ids)
        ids += [self.tok.pad_token_id] * pad; mask += [0] * pad
        return torch.tensor(ids), torch.tensor(mask)
    def __getitem__(self, i):
        ids, mask = self._pack(self.texts[i])
        item = {"input_ids": ids, "attention_mask": mask,
                "surf": torch.tensor(self.surf[i])}
        if self.targets is not None:
            item["target"] = torch.tensor(self.targets[i])           # (11,) class idx
            item["weight"] = torch.tensor(self.weights[i] if self.weights is not None else 1.0)
        return item

# ---------------------------------------------------------------- model
class Router(nn.Module):
    def __init__(self, backbone, use_surf=True):
        super().__init__()
        self.enc = AutoModel.from_pretrained(backbone)
        h = self.enc.config.hidden_size
        self.use_surf = use_surf
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(h + (NUM_SURF if use_surf else 0), 11 * 3)
    def forward(self, input_ids, attention_mask, surf):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).float()
        pooled = (out * m).sum(1) / m.sum(1).clamp(min=1)             # mean-pool
        if self.use_surf:
            pooled = torch.cat([pooled, surf], dim=-1)
        return self.head(self.drop(pooled)).view(-1, 11, 3)          # (B,11,3)

def focal_ce(logits, target, gamma=1.0):
    # logits (B,11,3), target (B,11)
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    tgt = target.unsqueeze(-1)
    logpt = logp.gather(-1, tgt).squeeze(-1)                          # (B,11)
    pt = p.gather(-1, tgt).squeeze(-1)
    return (-((1 - pt) ** gamma) * logpt)                            # (B,11) per-model loss

# ---------------------------------------------------------------- train / eval
def make_targets(df):
    perf = df[[f"Model_{m}_performance" for m in MODELS]].values
    cost = df[[f"Model_{m}_cost" for m in MODELS]].values
    cls = np.searchsorted(PERF_VALUES, perf)                          # 0.0->0,0.5->1,1.0->2
    return perf, cost, cls.astype(np.int64)

def regret_weights(perf, cost):
    """Weight each query by how much the oracle beats always-K (corrected metric).
    Trivial/unsolvable queries (K already optimal) get ~base weight; decisive get more."""
    cn = perquery_cost_norm(cost)
    rew = PERF_W * perf - COST_W * cn
    kidx = MODELS.index("K")
    adv = rew.max(1) - rew[:, kidx]
    w = 0.15 + adv / (adv.mean() + 1e-9)
    return (w / w.mean()).astype(np.float32)

@torch.no_grad()
def predict_eperf(model, loader, device):
    model.eval(); outs = []
    for b in loader:
        logits = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                       b["surf"].to(device))
        p = F.softmax(logits.float(), dim=-1).cpu().numpy()           # (B,11,3)
        outs.append((p * PERF_VALUES).sum(-1))                        # E[perf] (B,11)
    return np.concatenate(outs)

def train_one(tr_df, va_df, args, med_cost, device):
    tok = AutoTokenizer.from_pretrained(args.backbone)
    perf_tr, cost_tr, cls_tr = make_targets(tr_df)
    w_tr = regret_weights(perf_tr, cost_tr) if args.regret else None
    ds_tr = QueryDS(tr_df["query"].tolist(), tok, args.max_len, cls_tr, w_tr)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True)

    model = Router(args.backbone, use_surf=not args.no_surf).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(dl_tr) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

    best = (-1e9, None)
    va_loader = None
    if va_df is not None:
        ds_va = QueryDS(va_df["query"].tolist(), tok, args.max_len)
        va_loader = DataLoader(ds_va, batch_size=args.batch * 2, num_workers=4)
        perf_va, cost_va, _ = make_targets(va_df)

    for ep in range(args.epochs):
        model.train(); t0 = time.time(); run = 0.0
        for step, b in enumerate(dl_tr):
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=device == "cuda"):
                logits = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                               b["surf"].to(device))
                per_model = focal_ce(logits, b["target"].to(device), gamma=args.gamma)  # (B,11)
                loss = (per_model.mean(1) * b["weight"].to(device)).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.item()
        msg = f"  epoch {ep+1}/{args.epochs}  loss={run/len(dl_tr):.4f}  ({time.time()-t0:.0f}s)"
        if va_loader is not None:
            ep_va = predict_eperf(model, va_loader, device)
            choice = route_from_eperf(ep_va, med_cost)
            r = reward_of_choice(perf_va, cost_va, choice)
            msg += f"  val_reward={r:.4f}"
            if r > best[0]:
                best = (r, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        print(msg)
    if best[1] is not None:
        model.load_state_dict(best[1])
    return model, tok, best[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="microsoft/deberta-v3-base")
    ap.add_argument("--large", action="store_true", help="use deberta-v3-large")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--gamma", type=float, default=1.0, help="focal gamma (0=plain CE)")
    ap.add_argument("--regret", action="store_true", default=True)
    ap.add_argument("--no_surf", action="store_true")
    ap.add_argument("--cv", action="store_true", help="5-fold reward CV (no submission)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.large: args.backbone = "microsoft/deberta-v3-large"; args.batch = min(args.batch, 8)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"backbone={args.backbone} device={device} batch={args.batch} epochs={args.epochs}")

    tr = pd.read_csv("data/train.csv")
    perf_all, cost_all, _ = make_targets(tr)
    med_cost = cost_all.mean(0) if False else np.median(np.where(cost_all > 0, cost_all, np.nan), axis=0)
    med_cost = np.nan_to_num(med_cost, nan=cost_all.mean())          # per-model median cost lookup

    # reference points under the TRUE (per-query-normalized) metric
    print("always-K reward:", round(reward_of_choice(perf_all, cost_all,
          np.full(len(tr), MODELS.index("K"))), 4))
    print("oracle reward:  ", round(reward_of_choice(perf_all, cost_all,
          (PERF_W*perf_all - COST_W*perquery_cost_norm(cost_all)).argmax(1)), 4))

    if args.cv:
        kf = KFold(5, shuffle=True, random_state=args.seed); scores = []
        for f, (tri, vai) in enumerate(kf.split(tr)):
            print(f"--- fold {f} ---")
            _, _, r = train_one(tr.iloc[tri], tr.iloc[vai], args, med_cost, device)
            scores.append(r)
        print(f"\nCV reward = {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
        return

    # holdout train + test submission
    n = len(tr); idx = np.random.permutation(n); cut = int(n * 0.9)
    tr_df, va_df = tr.iloc[idx[:cut]], tr.iloc[idx[cut:]]
    model, tok, r = train_one(tr_df, va_df, args, med_cost, device)
    print(f"best val reward = {r:.4f}")

    te = pd.read_csv("data/test.csv")
    ds_te = QueryDS(te["query"].tolist(), tok, args.max_len)
    dl_te = DataLoader(ds_te, batch_size=args.batch * 2, num_workers=4)
    ep_te = predict_eperf(model, dl_te, device)
    choice = route_from_eperf(ep_te, med_cost)
    sub = pd.DataFrame({"ID": te["ID"], "pred_model": [f"Model_{MODELS[c]}" for c in choice]})
    sub.to_csv("submission.csv", index=False)
    torch.save(model.state_dict(), "notebooks/deberta_router.pt")
    print("wrote submission.csv  route dist:",
          sub["pred_model"].value_counts().to_dict())

if __name__ == "__main__":
    main()

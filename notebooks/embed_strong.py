"""Generate a STRONGER embedding for the feature experiment, saved to separate files
so the committed bge embeddings (Xtr.npy/Xte.npy) are untouched.

Usage:  python notebooks/embed_strong.py intfloat/e5-large-v2 e5
Writes: notebooks/Xtr_<tag>.npy, notebooks/Xte_<tag>.npy
"""
import sys, time, numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer

EMB_MODEL = sys.argv[1] if len(sys.argv) > 1 else "intfloat/e5-large-v2"
TAG = sys.argv[2] if len(sys.argv) > 2 else "e5"
PREFIX = "query: " if "e5" in EMB_MODEL.lower() else ""   # e5 wants a 'query:' prefix
MAX_TOKENS = 512
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"model={EMB_MODEL} tag={TAG} prefix={PREFIX!r} device={dev}", flush=True)

tr = pd.read_csv("data/train.csv"); te = pd.read_csv("data/test.csv")
st = SentenceTransformer(EMB_MODEL, device=dev, trust_remote_code=True)
st.max_seq_length = MAX_TOKENS

def embed(texts):
    txt = [PREFIX + str(t) for t in texts]
    return st.encode(txt, batch_size=32, show_progress_bar=True,
                     convert_to_numpy=True, normalize_embeddings=True).astype("float32")

t0 = time.time()
Xtr = embed(tr["query"]); Xte = embed(te["query"])
np.save(f"notebooks/Xtr_{TAG}.npy", Xtr); np.save(f"notebooks/Xte_{TAG}.npy", Xte)
print(f"done {Xtr.shape} {Xte.shape} in {time.time()-t0:.1f}s -> notebooks/Xtr_{TAG}.npy", flush=True)

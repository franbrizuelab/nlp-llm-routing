"""Embed train/test queries -> cached .npy. Run this FIRST on the 3080 box.

On a 3080 this is ~1-2 min (vs ~20 min on CPU). VRAM need is tiny (~1-2 GB) since
this is inference only, so you can also swap EMB_MODEL for a stronger one:
  BAAI/bge-large-en-v1.5   (~1.3GB)   intfloat/e5-large-v2    Alibaba-NLP/gte-large-en-v1.5
  BAAI/bge-m3              (~2.3GB, multilingual)

Usage:  python notebooks/embed.py [model_name]
Writes: notebooks/Xtr.npy, notebooks/Xte.npy  (gitignored; regenerate per box)
"""
import sys, time, numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer

EMB_MODEL = sys.argv[1] if len(sys.argv) > 1 else "BAAI/bge-base-en-v1.5"
MAX_TOKENS = 512
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"model={EMB_MODEL} device={dev}")

tr = pd.read_csv("data/train.csv"); te = pd.read_csv("data/test.csv")
st = SentenceTransformer(EMB_MODEL, device=dev); st.max_seq_length = MAX_TOKENS

def embed(texts):
    return st.encode(list(texts.astype(str)), batch_size=64, show_progress_bar=True,
                     convert_to_numpy=True, normalize_embeddings=True).astype("float32")

t0 = time.time()
Xtr = embed(tr["query"]); Xte = embed(te["query"])
np.save("notebooks/Xtr.npy", Xtr); np.save("notebooks/Xte.npy", Xte)
print(f"done {Xtr.shape} {Xte.shape} in {time.time()-t0:.1f}s -> notebooks/Xtr.npy, notebooks/Xte.npy")

"""Extract per-query difficulty/category features from DeepSeek, to test whether an
external LLM adds TRANSFERABLE routing signal our embeddings lack (see FINDINGS.md sec 7).

The models we route to are ANONYMOUS, so DeepSeek cannot predict per-model success. It can
only judge the QUERY: how hard is it, is it obscure factual recall, would a small model get
it. Those map to our K-vs-upgrade decision.

Key handling:
  - reads DEEPSEEK_API_KEY from env (never hardcode / commit a key)
  - OpenAI-compatible endpoint (base_url=https://api.deepseek.com)
  - concurrent calls + retries + INCREMENTAL checkpoint (resumable; rerun skips done IDs)

Setup:  pip install openai pandas
        export DEEPSEEK_API_KEY=sk-...
Pilot:  python notebooks/deepseek_features.py --split train --n 2500 --model deepseek-chat
Full:   python notebooks/deepseek_features.py --split train --model deepseek-chat
        python notebooks/deepseek_features.py --split test  --model deepseek-chat
Writes: notebooks/deepseek_<split>.csv   (ID, category, difficulty, obscure, small_ok)
"""
from __future__ import annotations
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

SYS = (
    "You are a routing classifier. Given a user query that will be sent to a language "
    "model, judge ONLY the query (you do not need to answer it). Reply with STRICT JSON, "
    "no prose, with keys:\n"
    '  "category": one of "factual_recall","reasoning","coding","math","general"\n'
    '  "difficulty": integer 1-5 (1=trivial, 5=very hard)\n'
    '  "obscure_knowledge": true if it needs niche/specialized facts a typical model may not know\n'
    '  "small_model_can_answer": true if a small ~7B model would likely answer it correctly\n'
)
KEYS = ["category", "difficulty", "obscure_knowledge", "small_model_can_answer"]
CATS = {"factual_recall": 0, "reasoning": 1, "coding": 2, "math": 3, "general": 4}


def classify(client, model, query, max_chars=6000):
    q = str(query)[:max_chars]
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": q}],
                max_tokens=200,
            )
            d = json.loads(r.choices[0].message.content)
            return {
                "category": CATS.get(str(d.get("category", "general")).lower(), 4),
                "difficulty": int(d.get("difficulty", 3)),
                "obscure": int(bool(d.get("obscure_knowledge", False))),
                "small_ok": int(bool(d.get("small_model_can_answer", True))),
            }
        except Exception as e:
            if attempt == 3:
                return {"category": 4, "difficulty": 3, "obscure": 0, "small_ok": 1, "err": str(e)[:80]}
            time.sleep(2 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--model", default="deepseek-chat", help="deepseek-chat (V3) | deepseek-reasoner (R1)")
    ap.add_argument("--n", type=int, default=0, help="limit (pilot); 0 = all")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        sys.exit("set DEEPSEEK_API_KEY env var first")
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("pip install openai")
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")

    df = pd.read_csv(f"data/{args.split}.csv")
    if args.n:
        df = df.iloc[:args.n]
    out_path = f"notebooks/deepseek_{args.split}.csv"
    done = set()
    if os.path.exists(out_path):
        done = set(pd.read_csv(out_path)["ID"].tolist())
        print(f"resuming: {len(done)} already done")
    todo = [(int(r.ID), r.query) for r in df.itertuples() if int(r.ID) not in done]
    print(f"{len(todo)} queries to classify with {args.model} ({args.workers} workers)")

    t0 = time.time(); n_done = 0
    write_header = not os.path.exists(out_path)
    with open(out_path, "a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        if write_header:
            f.write("ID,category,difficulty,obscure,small_ok\n")
        futs = {ex.submit(classify, client, args.model, q): i for i, q in todo}
        for fut in as_completed(futs):
            i = futs[fut]; d = fut.result()
            f.write(f"{i},{d['category']},{d['difficulty']},{d['obscure']},{d['small_ok']}\n")
            n_done += 1
            if n_done % 50 == 0:
                f.flush()
                print(f"  {n_done}/{len(todo)}  ({(time.time()-t0)/n_done:.2f}s/query)", flush=True)
    print(f"done {n_done} in {time.time()-t0:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()

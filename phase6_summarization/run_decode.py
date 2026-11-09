from __future__ import annotations

import argparse
import itertools
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    from summ import data as D, metrics as M
    from summ.abs_data import encode_input
    from summ.abs_generate import summarise
    from summ.abs_train import load_trained, n_in_for

    model, cfg = load_trained(a.run)
    _, va = D.split(); va = va[:a.n]
    ins = {r["id"]: encode_input(r["document"], cfg.builder, n_in_for(cfg)) for r in va}
    refs = [r["summary"] for r in va]
    base = dict(max_new=460, min_new=200, greedy=True, repetition_penalty=1.15, no_repeat_ngram=4)
    grid = [{}]
    grid += [{"repetition_penalty": p} for p in (1.0, 1.05, 1.3)]
    grid += [{"no_repeat_ngram": n} for n in (0, 3, 6)]
    grid += [{"min_new": m} for m in (300, 380, 440)]
    grid += [{"greedy": False, "temperature": t, "top_p": 0.9} for t in (0.5, 0.8)]
    out = open(f"{a.run}/decode_sweep.jsonl", "a")
    for g in grid:
        kw = {**base, **g}; t = time.time()
        pred = summarise(model, ins, "cuda", batch=16, **kw)
        m = M.evaluate([pred[r["id"]] for r in va], refs, rougeL=False)
        rec = {"n": len(va), "gen": kw, **m, "sec": round(time.time() - t)}
        out.write(json.dumps(rec) + "\n"); out.flush()
        print(f"{json.dumps(g):55s} BERTScore {m['bertscore_f1']:.4f} R1 {m['rouge1']:.4f} "
              f"words {m['pred_words']:.0f} | {rec['sec']}s", flush=True)


if __name__ == "__main__":
    main()

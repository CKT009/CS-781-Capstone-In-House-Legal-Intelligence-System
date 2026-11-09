from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from summ import data as D
from summ import extractive as E
from summ import metrics as M

OUT = Path(__file__).resolve().parent / "runs" / "extractive"
PRIOR = OUT / "position_prior.npy"
_P = None


def make(name: str, budget: int, rec: dict) -> str:
    d = rec["document"]
    if name == "lead_raw":
        return " ".join(d.split()[:budget])
    if name == "lead_strip":
        return E.lead(d, budget)
    if name == "lead_strip_nofilt":
        return E.lead(d, budget, filt=False)
    if name == "oracle":
        return E.oracle(d, rec["summary"], budget)

    parts = name.split("+")
    method, kw = parts[0], {}
    for p in parts[1:]:
        if p.startswith("prior"):
            kw["prior"] = _P; kw["w_prior"] = float(p[5:] or 0.5)
        elif p.startswith("lead"):
            kw["lead_words"] = int(p[4:])
        elif p.startswith("mmr"):
            kw["mmr"] = float(p[3:])
    return E.select(d, budget, method=method, **kw)


def _init():
    global _P
    _P = np.load(PRIOR) if PRIOR.exists() else None


def _gen(args):
    name, budget, rec = args
    return make(name, budget, rec)


def generate(name, budget, recs, pool):
    return pool.map(_gen, [(name, budget, r) for r in recs], chunksize=8)


def _prior_chunk(recs):
    return E.fit_position_prior(recs, bins=20, budget=650)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["prior", "sweep", "final"])
    ap.add_argument("--configs", default="")
    ap.add_argument("--rougeL", action="store_true")
    ap.add_argument("--best", default="")
    ap.add_argument("--n_prior", type=int, default=2000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tr, va = D.split()

    if a.stage == "prior":
        recs = tr[:a.n_prior]
        chunks = [recs[i::12] for i in range(12)]
        with Pool(12) as p:
            parts = p.map(_prior_chunk, chunks)
        prior = np.mean(parts, 0)
        np.save(PRIOR, prior)
        print("position prior (20 bins):", np.round(prior, 3).tolist())
        return

    pool = Pool(12, initializer=_init)
    refs = [r["summary"] for r in va]
    if a.stage == "sweep":
        for cfg in a.configs.split(","):
            name, budget = cfg.split(":"); budget = int(budget)
            t = time.time()
            preds = generate(name, budget, va, pool)
            m = M.evaluate(preds, refs, bert=True, rougeL=a.rougeL)
            m.update(config=cfg, secs=round(time.time() - t))
            print(M.report(m), flush=True)
            with open(OUT / "results.jsonl", "a") as f:
                f.write(json.dumps(m) + "\n")
        return


    name, budget = a.best.split(":"); budget = int(budget)
    preds = generate(name, budget, va, pool)
    (OUT / "val_preds.json").write_text(json.dumps({r["id"]: p for r, p in zip(va, preds)}))
    te = D.load("test"); ids = [r["id"] for r in te]
    best = generate(name, budget, te, pool)
    M.write_submission(dict(zip(ids, best)), ids, OUT.parent.parent / "subs" / "extractive_best.zip")
    lead = generate("lead_strip", 400, te, pool)
    M.write_submission(dict(zip(ids, lead)), ids, OUT.parent.parent / "subs" / "lead_strip.zip")
    (OUT / "test_preds_best.json").write_text(json.dumps(dict(zip(ids, best))))
    print("wrote", a.best, "val_preds + subs/extractive_best.zip + subs/lead_strip.zip")


if __name__ == "__main__":
    main()

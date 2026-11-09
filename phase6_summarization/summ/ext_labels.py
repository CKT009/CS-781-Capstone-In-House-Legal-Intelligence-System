from __future__ import annotations

import pickle
import re
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from summ import data as D

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "ext_labels" / "labels.pkl"
OPEN_WORDS = 400
CAP_OPEN, CAP_FULL = 450, 900


def _toks(s):
    return re.findall(r"\w+", s.lower())


def _greedy(sent_toks, ref_toks, cap_words):
    ref1, ref2 = Counter(ref_toks), Counter(zip(ref_toks, ref_toks[1:]))
    if not ref1 or not sent_toks:
        return []
    v1 = {w: i for i, w in enumerate(ref1)}; v2 = {g: i for i, g in enumerate(ref2)}
    n = len(sent_toks)
    S1 = np.zeros((n, len(v1)), np.float32); S2 = np.zeros((n, max(1, len(v2))), np.float32)
    tot1 = np.zeros(n, np.float32); tot2 = np.zeros(n, np.float32)
    for i, t in enumerate(sent_toks):
        tot1[i] = len(t); tot2[i] = max(0, len(t) - 1)
        for w in t:
            j = v1.get(w)
            if j is not None:
                S1[i, j] += 1
        for g in zip(t, t[1:]):
            j = v2.get(g)
            if j is not None:
                S2[i, j] += 1
    R1 = np.array(list(ref1.values()), np.float32)
    R2 = np.array(list(ref2.values()), np.float32) if ref2 else np.zeros(1, np.float32)
    r1n, r2n = R1.sum(), max(R2.sum(), 1.0)

    def f(ov, tot, rn):
        p = ov / np.maximum(tot, 1); r = ov / rn
        return np.where(ov > 0, 2 * p * r / np.maximum(p + r, 1e-9), 0.0)

    cur1 = np.zeros_like(R1); cur2 = np.zeros_like(R2); c1 = c2 = 0.0
    chosen, best, words = [], 0.0, 0
    avail = np.ones(n, bool)
    while words < cap_words:
        ov1 = np.minimum(cur1[None] + S1, R1[None]).sum(1)
        ov2 = np.minimum(cur2[None] + S2, R2[None]).sum(1)
        sc = f(ov1, c1 + tot1, r1n) + f(ov2, c2 + tot2, r2n)
        sc[~avail] = -1
        j = int(sc.argmax())
        if sc[j] <= best + 1e-6:
            break
        best = float(sc[j]); chosen.append(j); avail[j] = False
        cur1 += S1[j]; cur2 += S2[j]; c1 += tot1[j]; c2 += tot2[j]; words += int(tot1[j])
    return sorted(chosen)


def _prec(sent_toks, ref_toks):
    ref1, ref2 = Counter(ref_toks), Counter(zip(ref_toks, ref_toks[1:]))
    r1p, r2p = [], []
    for t in sent_toks:
        c1, c2 = Counter(t), Counter(zip(t, t[1:]))
        r1p.append(sum((c1 & ref1).values()) / max(1, len(t)))
        r2p.append(sum((c2 & ref2).values()) / max(1, len(t) - 1))
    return np.array(r1p, np.float16), np.array(r2p, np.float16)


def label_one(rec):
    sents = D.sentences(rec["document"])
    st = [_toks(s) for s in sents]
    ref = _toks(rec["summary"])

    opening = _toks(" ".join(rec["summary"].split()[:OPEN_WORDS]))
    o1, o2 = _prec(st, opening); f1, f2 = _prec(st, ref)
    return rec["id"], dict(
        n_words=np.array([len(s.split()) for s in sents], np.int32),
        open=_greedy(st, opening, CAP_OPEN), full=_greedy(st, ref, CAP_FULL),
        r1p_open=o1, r2p_open=o2, r1p_full=f1, r2p_full=f2)


def build(procs=12):
    tr, va = D.split()
    with Pool(procs) as p:
        out = dict(p.imap_unordered(label_one, tr + va, chunksize=16))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    return out


_CACHE = None


def load():
    global _CACHE
    if _CACHE is None:
        with open(OUT, "rb") as fh:
            _CACHE = pickle.load(fh)
    return _CACHE


def hard(lab, kind="open"):
    y = np.zeros(len(lab["n_words"]), np.float32)
    y[lab[kind]] = 1
    return y


def stats(labels, ids):
    rows = [labels[i] for i in ids]
    n = np.array([len(r["n_words"]) for r in rows])
    out = {"docs": len(rows), "sents/doc median": float(np.median(n))}
    for k in ("open", "full"):
        k_n = np.array([len(r[k]) for r in rows])
        w = np.array([r["n_words"][r[k]].sum() for r in rows])
        first = np.array([np.mean(np.array(r[k]) / max(1, len(r["n_words"]) - 1))
                          if r[k] else np.nan for r in rows])
        out[k] = {"sel/doc": float(k_n.mean()), "pos_rate": float((k_n / n).mean()),
                  "words median": float(np.median(w)),
                  "mean rel. position": float(np.nanmean(first))}
    jac = [len(set(r["open"]) & set(r["full"])) / max(1, len(set(r["open"]) | set(r["full"])))
           for r in rows]
    out["jaccard open/full"] = float(np.mean(jac))
    return out


if __name__ == "__main__":
    import json
    import time
    t = time.time()
    L = build()
    tr, va = D.split()
    print(f"[labels] {len(L):,} docs in {time.time()-t:.0f}s -> {OUT}")
    print("train", json.dumps(stats(L, [r["id"] for r in tr]), indent=1))

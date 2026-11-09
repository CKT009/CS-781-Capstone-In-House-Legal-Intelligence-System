import argparse
import json
import os
import sys
import time

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lsi import data as D
from lsi.metrics import evaluate, report, decide, write_submission
sys.path.insert(0, os.path.join(ROOT, "baselines"))
import lexical as LX

N_JOBS = 5
L = 100
RUN = os.path.join(ROOT, "runs", "lexmention")
CACHE = os.environ.get("LEXM_CACHE", os.path.join(RUN, "_cache"))
LEX_CACHE = os.path.join(ROOT, "baselines", "_cache")
MASKS = ("<SECTION>", "<ACT>")


def mention_context(text, W):
    toks = text.split()
    pos = [i for i, w in enumerate(toks) if MASKS[0] in w or MASKS[1] in w]
    if not pos:
        return text, 0
    segs, (a, b) = [], (max(0, pos[0] - W), pos[0] + W + 1)
    for p in pos[1:]:
        lo, hi = max(0, p - W), p + W + 1
        if lo <= b:
            b = max(b, hi)
        else:
            segs.append((a, b)); a, b = lo, hi
    segs.append((a, b))
    return " ... ".join(" ".join(toks[s:e]) for s, e in segs), len(pos)


def contexts(docs, W):
    out = [mention_context(d.text, W) for d in docs]
    return [o[0] for o in out], np.array([o[1] for o in out])


def ctx_feats(W, splits):
    os.makedirs(CACHE, exist_ok=True)
    out = []
    for name in ("w12", "c25"):
        fs = [os.path.join(CACHE, f"ctx{W}_{name}_{k}.npz") for k in ("tr", "dv", "te")]
        if all(os.path.exists(f) for f in fs):
            out.append([sp.load_npz(f).tocsr() for f in fs]); continue
        t0 = time.time()
        kw, head, tail = LX.FEATS[name]
        txt = [[LX.truncate(t, head, tail) for t in contexts(splits[s], W)[0]]
               for s in ("train", "dev", "test")]
        vec = TfidfVectorizer(**kw)
        Xtr = vec.fit_transform(txt[0])
        def chunked(tx):
            ch = [tx[i:i + 2000] for i in range(0, len(tx), 2000)]
            return sp.vstack(Parallel(n_jobs=N_JOBS)(delayed(vec.transform)(c) for c in ch)).tocsr()
        Xs = [Xtr, chunked(txt[1]), chunked(txt[2])]
        for f, X in zip(fs, Xs):
            sp.save_npz(f, X, compressed=False)
        print(f"[ctx{W} {name}] {Xtr.shape} nnz/doc {Xtr.nnz/Xtr.shape[0]:.0f} "
              f"in {time.time()-t0:.0f}s", flush=True)
        out.append(Xs)
    return [sp.hstack([p[i] for p in out]).tocsr() for i in range(3)]


def full_feats():
    parts = [[sp.load_npz(os.path.join(LEX_CACHE, f"feat_{n}_{k}.npz")).tocsr()
              for k in ("tr", "dv", "te")] for n in ("w12", "c25")]
    return [sp.hstack([p[i] for p in parts]).tocsr() for i in range(3)]


def _fit_one(X, y, C, Xs):
    clf = LogisticRegression(C=C, solver="liblinear", max_iter=200)
    clf.fit(X, y)
    return [clf.predict_proba(x)[:, 1].astype(np.float32) for x in Xs]


def fit_ovr(Xtr, Ytr, Xs, C):
    res = Parallel(n_jobs=N_JOBS, prefer="processes", max_nbytes="1M")(
        delayed(_fit_one)(Xtr, Ytr[:, l], C, Xs) for l in range(L))
    return [np.stack([r[k] for r in res], 1) for k in range(len(Xs))]


def splits_():
    return {s: D.load(s) for s in ("train", "dev", "test")}


def ev_row(name, S, Y, secs=None):
    e = evaluate(S, Y)
    print(f"{name:34s} {report(e)}" + (f" ({secs:.0f}s)" if secs else ""), flush=True)
    return {k: (v if not isinstance(v, np.ndarray) else None) for k, v in e.items()} | (
        {"runtime_s": round(secs, 1)} if secs else {})


def log_result(name, row):
    p = os.path.join(RUN, "results.json")
    R = json.load(open(p)) if os.path.exists(p) else {}
    R[name] = row
    json.dump(R, open(p, "w"), indent=1)


def cmd_stats(a):
    S = splits_()
    stats = {}
    for W in (25, 50):
        for s, docs in S.items():
            ctx, n = contexts(docs, W)
            clen = np.array([len(c.split()) for c in ctx])
            flen = np.array([len(d.text.split()) for d in docs])
            has = n > 0
            stats[f"W{W}_{s}"] = dict(
                docs=len(docs), with_mask=int(has.sum()), frac_with_mask=round(float(has.mean()), 4),
                fallback_full=int((~has).sum()), median_mentions=float(np.median(n[has])),
                median_ctx_words=float(np.median(clen[has])), median_full_words=float(np.median(flen)),
                median_ctx_over_full=round(float(np.median(clen[has] / flen[has])), 4))
            print(f"W{W} {s}: {stats[f'W{W}_{s}']}", flush=True)
    os.makedirs(RUN, exist_ok=True)
    json.dump(stats, open(os.path.join(RUN, "mask_stats.json"), "w"), indent=1)


def cmd_fit(a):
    S = splits_()
    Ytr, Ydv = D.label_matrix(S["train"]).astype(np.int8), D.label_matrix(S["dev"])
    name = f"ctx{a.W}{'+full' if a.full else ''}_C{a.C:g}"
    t0 = time.time()
    Xtr, Xdv, Xte = ctx_feats(a.W, S)
    if a.full:
        F = full_feats()
        Xtr, Xdv, Xte = (sp.hstack([x, f]).tocsr() for x, f in zip((Xtr, Xdv, Xte), F))
    t1 = time.time()
    pdv, pte = fit_ovr(Xtr, Ytr, [Xdv, Xte], a.C)
    secs = time.time() - t1
    np.savez(os.path.join(CACHE, f"probs_{name}.npz"), dv=pdv, te=pte)
    row = ev_row(name, pdv, Ydv, secs) | {"feat_s": round(t1 - t0, 1), "n_features": Xtr.shape[1]}
    log_result(name, row)


def cmd_final(a):
    S = splits_()
    Ydv = D.label_matrix(S["dev"])
    z = np.load(os.path.join(CACHE, f"probs_{a.name}.npz"))
    pdv, pte = z["dv"].astype(np.float32), z["te"].astype(np.float32)
    e = evaluate(pdv, Ydv)
    np.save(os.path.join(RUN, "dev_probs.npy"), pdv)
    np.save(os.path.join(RUN, "test_probs.npy"), pte)
    np.save(os.path.join(RUN, "thresholds.npy"), e["thresholds"])
    P = decide(pte, e["thresholds"])
    write_submission(P, [d.id for d in S["test"]], os.path.join(ROOT, "subs", "lexmention.zip"))
    json.dump({"best": a.name, "dev": {k: v for k, v in e.items() if k != "thresholds"},
               "mean_test_labels": float(P.sum(1).mean())},
              open(os.path.join(RUN, "summary.json"), "w"), indent=1)
    print("final", a.name, report(e), "mean test labels", P.sum(1).mean())


def cmd_blend(a):
    Ydv = D.label_matrix(D.load("dev"))
    ld = lambda r: np.load(os.path.join(ROOT, "runs", r, "dev_probs.npy")).astype(np.float32)
    neu = np.mean([ld(r) for r in ("full_k4", "kd_half", "lastk_k4", "kd_control")], 0)
    lex, me = ld("lexical"), ld("lexmention")
    rows = {}
    S0 = 0.5 * neu + 0.5 * lex
    rows["S0"] = ev_row("S0 = .5 neural + .5 lexical", S0, Ydv)
    rows["mention_alone"] = ev_row("mention model alone", me, Ydv)
    rows["S0_lex->mention"] = ev_row(".5 neural + .5 mention", 0.5 * neu + 0.5 * me, Ydv)
    rows["S0_lex+mention"] = ev_row(".5 neural + .25 lex + .25 mention", 0.5 * neu + 0.25 * lex + 0.25 * me, Ydv)
    for w in (0.1, 0.2, 0.3, 0.4):
        rows[f"S0_w{w}"] = ev_row(f"(1-{w}) S0 + {w} mention", (1 - w) * S0 + w * me, Ydv)
    json.dump(rows, open(os.path.join(RUN, "blend.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats")
    f = sub.add_parser("fit"); f.add_argument("--W", type=int, default=25)
    f.add_argument("--C", type=float, default=16.0); f.add_argument("--full", action="store_true")
    g = sub.add_parser("final"); g.add_argument("--name", required=True)
    sub.add_parser("blend")
    a = ap.parse_args()
    os.makedirs(RUN, exist_ok=True)
    {"stats": cmd_stats, "fit": cmd_fit, "final": cmd_final, "blend": cmd_blend}[a.cmd](a)

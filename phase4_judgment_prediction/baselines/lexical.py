from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cjpe import data as D
from cjpe import metrics as M

OUT = ROOT / "baselines"
N_JOBS = 16
WS = re.compile(r"\s+")

SECTIONS = {"whole": None, "last5k": ("tail", 5000), "last10k": ("tail", 10000),
            "last20k": ("tail", 20000), "first5k": ("head", 5000)}
VECS = {
    "word": dict(analyzer="word", ngram_range=(1, 2), token_pattern=r"(?u)\b\w\w+\b"),
    "char": dict(analyzer="char", ngram_range=(3, 5)),
}
C_GRID = (1.0, 3.0, 10.0, 30.0, 100.0)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def clean(t: str) -> str:
    return WS.sub(" ", t.lower()).strip()


def cut(t: str, sec) -> str:
    if sec is None:
        return t
    how, n = sec
    return t[-n:] if how == "tail" else t[:n]


def _hash_chunk(texts, kind):
    hv = HashingVectorizer(n_features=2 ** 22, alternate_sign=False, norm=None,
                           dtype=np.float32, **VECS[kind])
    return hv.transform(texts)


def counts(texts, kind) -> sp.csr_matrix:
    step = max(1, len(texts) // (N_JOBS * 4))
    parts = Parallel(n_jobs=N_JOBS)(delayed(_hash_chunk)(texts[i:i + step], kind)
                                    for i in range(0, len(texts), step))
    return sp.vstack(parts).tocsr()


def tfidf(Xs: dict, fit_on: str) -> dict:
    tf = TfidfTransformer(sublinear_tf=True).fit(Xs[fit_on])
    return {k: tf.transform(v).astype(np.float32) for k, v in Xs.items()}


def year_feats(docs, scale=0.3) -> sp.csr_matrix:
    dec = np.clip((np.array([d.year for d in docs]) - 1940) // 10, 0, 8)
    return sp.csr_matrix((np.full(len(docs), scale, np.float32), (np.arange(len(docs)), dec)),
                         shape=(len(docs), 9))


def fit_lr(X, y, C):


    return LogisticRegression(C=C, solver="liblinear", dual=True, class_weight="balanced",
                              max_iter=1000).fit(X, y)


def score(p1, gold):
    t, f_t = M.best_threshold(p1, gold)
    f_05 = M.prf((p1 >= 0.5).astype(int), gold)["macro_f1"]
    return {"macro_f1_tuned": round(f_t, 4), "threshold": round(t, 3), "macro_f1_at_0.5": round(f_05, 4)}


def sweep_C(Xtr, ytr, Xdv, ydv, grid=C_GRID):
    models = Parallel(n_jobs=min(N_JOBS, len(grid)), prefer="threads")(
        delayed(fit_lr)(Xtr, ytr, C) for C in grid)
    res = []
    for C, m in zip(grid, models):
        r = score(m.predict_proba(Xdv)[:, 1], ydv)
        r["C"] = C
        res.append((r, m))
    return max(res, key=lambda rm: rm[0]["macro_f1_tuned"]), [r for r, _ in res]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-trdev", action="store_true", help="skip the train+dev refit")
    args = ap.parse_args()
    t0 = time.time()

    docs = {s: D.load(s) for s in ("train", "dev", "test")}
    y = {s: np.array([d.label for d in docs[s]]) for s in ("train", "dev")}
    text = {s: [clean(d.text) for d in docs[s]] for s in docs}
    log("loaded", {s: len(v) for s, v in docs.items()})

    table, feats = {}, {}
    for sec_name, sec in SECTIONS.items():
        for kind in VECS:
            t1 = time.time()
            Xc = {s: counts([cut(t, sec) for t in text[s]], kind) for s in text}
            feats[(sec_name, kind)] = tfidf(Xc, "train")
            del Xc
            log(f"features {sec_name}/{kind}: {time.time() - t1:.0f}s, "
                f"train nnz {feats[(sec_name, kind)]['train'].nnz / 1e6:.0f}M")
        for kind in ("word", "char", "word+char"):
            F = (feats[(sec_name, kind)] if kind in VECS else
                 {s: sp.hstack([feats[(sec_name, "word")][s], feats[(sec_name, "char")][s]]).tocsr()
                  for s in text})
            t1 = time.time()
            (best, _), allC = sweep_C(F["train"], y["train"], F["dev"], y["dev"])
            table.setdefault(sec_name, {})[kind] = {**best, "all_C": allC}
            log(f"{sec_name:8s} {kind:9s} {best}  ({time.time() - t1:.0f}s)")
        if sec_name not in ("last10k", "last20k", "whole"):

            for kind in VECS:
                feats.pop((sec_name, kind), None)
        (OUT / "lexical_sections.json").write_text(json.dumps(table, indent=1))


    sec_name, kind = max(((s, k) for s in table for k in table[s]),
                         key=lambda sk: table[sk[0]][sk[1]]["macro_f1_tuned"])
    if (sec_name, "word") not in feats:
        raise RuntimeError(f"best section {sec_name} was not kept in cache")
    F = (feats[(sec_name, kind)] if kind in VECS else
         {s: sp.hstack([feats[(sec_name, "word")][s], feats[(sec_name, "char")][s]]).tocsr()
          for s in text})
    feats.clear()
    base = table[sec_name][kind]
    C = base["C"]
    log(f"best section/features: {sec_name}/{kind} C={C}: {base['macro_f1_tuned']}")

    variants = {"lr": dict(base, year=False)}
    Fy = {s: sp.hstack([F[s], year_feats(docs[s])]).tocsr() for s in F}
    grid = tuple(c for c in C_GRID if C / 3.1 <= c <= C * 3.1)
    (ry, my), _ = sweep_C(Fy["train"], y["train"], Fy["dev"], y["dev"], grid)
    variants["lr+year"] = dict(ry, year=True)
    log("lr+year", ry)


    svm = LinearSVC(C=0.1 * C, class_weight="balanced").fit(F["train"], y["train"])
    s_dv = 1 / (1 + np.exp(-2 * svm.decision_function(F["dev"])))
    variants["svm"] = dict(score(s_dv, y["dev"]), C=0.1 * C)
    log("svm", variants["svm"])

    use_year = variants["lr+year"]["macro_f1_tuned"] > variants["lr"]["macro_f1_tuned"]
    if use_year:
        F, C = Fy, ry["C"]
    model = fit_lr(F["train"], y["train"], C)
    p_dv = model.predict_proba(F["dev"])[:, 1]
    p_te = model.predict_proba(F["test"])[:, 1]
    thr, f_t = M.best_threshold(p_dv, y["dev"])
    m05 = M.prf((p_dv >= 0.5).astype(int), y["dev"])
    log("final dev tuned:", M.report(M.prf((p_dv >= thr).astype(int), y["dev"])), f"thr={thr:.3f}")
    log("final dev @0.5:", M.report(m05))

    np.save(OUT / "lexical_dev_probs.npy", p_dv.astype(np.float32))
    np.save(OUT / "lexical_test_probs.npy", p_te.astype(np.float32))
    test_ids = [d.id for d in docs["test"]]
    pred = {i: int(p >= thr) for i, p in zip(test_ids, p_te)}
    M.write_submission(pred, test_ids, ROOT / "subs" / "lexical.zip")
    final = {"section": sec_name, "features": kind, "year": use_year, "C": C,
             "threshold": thr, "dev_macro_f1_tuned": round(f_t, 4),
             "dev_macro_f1_at_0.5": round(m05["macro_f1"], 4),
             "test_pred1_rate": round(float(np.mean(list(pred.values()))), 4),
             "variants": variants}

    if not args.no_trdev:


        log("refit on train+dev")
        trdev_docs = docs["train"] + docs["dev"]
        ytd = np.concatenate([y["train"], y["dev"]])
        sec = SECTIONS[sec_name]
        kinds = list(VECS) if kind == "word+char" else [kind]
        blocks = {"trdev": [], "test": []}
        tx = {"trdev": text["train"] + text["dev"], "test": text["test"]}
        for k in kinds:
            Xc = {s: counts([cut(t, sec) for t in tx[s]], k) for s in tx}
            Ft = tfidf(Xc, "trdev")
            for s in blocks:
                blocks[s].append(Ft[s])
        if use_year:
            blocks["trdev"].append(year_feats(trdev_docs))
            blocks["test"].append(year_feats(docs["test"]))
        G = {s: sp.hstack(b).tocsr() for s, b in blocks.items()}
        m2 = fit_lr(G["trdev"], ytd, C)
        p2 = m2.predict_proba(G["test"])[:, 1]
        np.save(OUT / "lexical_trdev_test_probs.npy", p2.astype(np.float32))
        pred2 = {i: int(p >= thr) for i, p in zip(test_ids, p2)}
        M.write_submission(pred2, test_ids, ROOT / "subs" / "lexical_trdev.zip")
        final["trdev_test_pred1_rate"] = round(float(np.mean(list(pred2.values()))), 4)
        final["trdev_vs_train_test_agreement"] = round(float(np.mean(
            [pred[i] == pred2[i] for i in test_ids])), 4)

    final["runtime_min"] = round((time.time() - t0) / 60, 1)
    (OUT / "lexical_final.json").write_text(json.dumps(final, indent=1))
    log("done", json.dumps(final))


if __name__ == "__main__":
    main()

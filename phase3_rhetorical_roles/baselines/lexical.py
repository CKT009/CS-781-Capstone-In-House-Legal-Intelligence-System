from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rr import data as D
from rr.metrics import (score, score_by_domain, domain_report, confusion,
                        confusion_report, validate_submission)

DATA = ROOT / "data" / "Phase_3_RR"
OUT = Path(__file__).resolve().parent
NEVER = ("Dissent", "PrecedentOverruled")
C1 = C2 = 4.0
SEED = 0

CUES = [r"learned counsel", r"\bappellant", r"\brespondent", r"\bpetitioner", r"\bsubmit",
        r"\bcontend", r"\bargu", r"\burged\b", r"\bheld\b", r"\bsection\b|\bs\.\s*\d",
        r"\barticle\b", r"\bact\b", r"\bv(?:s)?\.?\s", r"\bscc\b|\bair\b|\bitr\b|\bscr\b",
        r"\(\d{4}\)\s*\d+", r"\b(?:19|20)\d{2}\b", r"\bdismiss", r"\ballow", r"\bquash",
        r"\bset aside", r"\btribunal\b|\bhigh court\b|\bcommission\b|\bcit\b|\bito\b",
        r"\bwe\b", r"\bin our (?:view|opinion)", r"\bquestion\b|\bwhether\b|\bissue\b",
        r"\?", r"\bper curiam|\bj\.\s*$|^\s*\w+,\s*j\.", r"\bin view of", r"\bsupra\b",
        r"\bobserved\b|\breliance\b|\brelied\b", r"\bdistinguish", r"\bnot applicable\b",
        r"\bordered?\b", r"\bcosts?\b", r"\bdg\b|\bdirector general", r"\bassess",
        r"^\s*\d+(?:\.\d+)*\s", r'["“”]', r"\bthe facts\b"]
CUE_RE = [re.compile(c, re.I) for c in CUES]


def dense_feats(docs):
    rows = []
    for d in docs:
        n = len(d)
        for i, s in enumerate(d.sentences):
            r = i / max(n - 1, 1)
            bins = np.zeros(10); bins[min(int(r * 10), 9)] = 1
            toks = s.split()
            f = [r, r * r, float(i == 0), float(i == n - 1), np.log1p(len(toks)) / 5,
                 float(d.domain == "IT"), np.log1p(n) / 6]
            f += list(bins)
            f += [float(bool(c.search(s))) for c in CUE_RE]
            rows.append(f)
    return sp.csr_matrix(np.asarray(rows, dtype=np.float64))


def shift_rows(X, docs, k):
    idx, off = [], 0
    for d in docs:
        n = len(d)
        for i in range(n):
            j = i + k
            idx.append(off + j if 0 <= j < n else -1)
        off += n
    idx = np.asarray(idx)
    P = sp.csr_matrix((np.ones((idx >= 0).sum()), (np.nonzero(idx >= 0)[0], idx[idx >= 0])),
                      shape=(len(idx), X.shape[0]))
    return P @ X


class Featurizer:
    def fit(self, docs):
        sents = [s for d in docs for s in d.sentences]
        self.w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.5, sublinear_tf=True,
                                 lowercase=True, max_features=200_000, dtype=np.float32)
        self.c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                 sublinear_tf=True, lowercase=True, max_features=300_000,
                                 dtype=np.float32)
        self.w.fit(sents); self.c.fit(sents)
        return self

    def transform(self, docs):
        sents = [s for d in docs for s in d.sentences]
        W = self.w.transform(sents); Cc = self.c.transform(sents)
        ctx = [0.5 * shift_rows(W, docs, -1), 0.5 * shift_rows(W, docs, 1)]
        return sp.hstack([W, Cc, *ctx, dense_feats(docs)]).tocsr()


def ctx_prob_feats(P, docs):
    out, off = [], 0
    for d in docs:
        n = len(d); p = P[off:off + n]; off += n
        cs = np.vstack([np.zeros((1, p.shape[1])), np.cumsum(p, 0)])
        feats = [p]
        for k in (-3, -2, -1, 1, 2, 3):
            q = np.zeros_like(p)
            if k < 0: q[-k:] = p[:k]
            else: q[:-k] = p[k:]
            feats.append(q)
        for w in (5, 15):
            lo = np.clip(np.arange(n) - w, 0, n); hi = np.clip(np.arange(n) + w + 1, 0, n)
            feats.append((cs[hi] - cs[lo]) / (hi - lo)[:, None])

        feats.append(cs[np.arange(n)] / np.maximum(np.arange(n), 1)[:, None])
        feats.append((cs[n] - cs[np.arange(n) + 1]) / np.maximum(n - 1 - np.arange(n), 1)[:, None])
        feats.append(np.repeat(p.mean(0, keepdims=True), n, 0))
        out.append(np.hstack(feats))
    return sp.csr_matrix(np.vstack(out))


def _lr(C):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=300, tol=1e-3,
                              solver="lbfgs")


def labels_of(docs, vocab):
    li = {l: i for i, l in enumerate(vocab)}
    return np.array([li[l] for d in docs for l in d.labels])


def full_proba(clf, X, K):
    P = np.full((X.shape[0], K), 1e-9)
    P[:, clf.classes_] = clf.predict_proba(X)
    return P


def grouped_folds(docs, k, seed):
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(k)]
    for dom in ("CL", "IT"):
        ids = [i for i, d in enumerate(docs) if d.domain == dom]
        rng.shuffle(ids)
        for j, i in enumerate(ids):
            folds[j % k].append(i)
    return [sorted(f) for f in folds]


def _stage1_fit_predict(fit_docs, app_docs, vocab):
    fz = Featurizer().fit(fit_docs)
    clf = _lr(C1).fit(fz.transform(fit_docs), labels_of(fit_docs, vocab))
    return full_proba(clf, fz.transform(app_docs), len(vocab))


def oof_stage1(docs, vocab, k=5, n_jobs=5):
    folds = grouped_folds(docs, k, SEED + 1)
    res = Parallel(n_jobs=n_jobs)(delayed(_stage1_fit_predict)(
        [docs[i] for i in range(len(docs)) if i not in set(f)], [docs[i] for i in f], vocab)
        for f in folds)
    sizes = [len(d) for d in docs]
    starts = np.concatenate([[0], np.cumsum(sizes)])
    P = np.zeros((starts[-1], len(vocab)))
    for f, p in zip(folds, res):
        o = 0
        for i in f:
            P[starts[i]:starts[i + 1]] = p[o:o + sizes[i]]; o += sizes[i]
    return P


def fit_predict(fit_docs, app_docs, vocab, inner_jobs=5):
    K = len(vocab)
    fz = Featurizer().fit(fit_docs)
    Xf, Xa = fz.transform(fit_docs), fz.transform(app_docs)
    y = labels_of(fit_docs, vocab)
    s1 = _lr(C1).fit(Xf, y)
    P1 = full_proba(s1, Xa, K)
    Poof = oof_stage1(fit_docs, vocab, n_jobs=inner_jobs)
    s2 = _lr(C2).fit(sp.hstack([Xf, ctx_prob_feats(Poof, fit_docs)]).tocsr(), y)
    P2 = full_proba(s2, sp.hstack([Xa, ctx_prob_feats(P1, app_docs)]).tocsr(), K)
    T = transitions(fit_docs, vocab)
    return {"s1": P1, "s2": P2}, T


def transitions(docs, vocab):
    li = {l: i for i, l in enumerate(vocab)}; K = len(vocab)
    T = np.ones((K, K)); S = np.ones(K)
    for d in docs:
        S[li[d.labels[0]]] += 1
        for a, b in zip(d.labels, d.labels[1:]):
            T[li[a], li[b]] += 1
    logT = np.log(T / T.sum(1, keepdims=True))
    marg = T.sum(0) / T.sum()


    return {"plain": (np.log(S / S.sum()), logT),
            "pmi": (np.zeros(K), logT - np.log(marg)[None, :])}


def viterbi(logp, logS, logT):
    n, K = logp.shape
    dp = logS + logp[0]; bp = np.zeros((n, K), dtype=int)
    for t in range(1, n):
        cand = dp[:, None] + logT
        bp[t] = cand.argmax(0); dp = cand.max(0) + logp[t]
    path = [int(dp.argmax())]
    for t in range(n - 1, 0, -1):
        path.append(int(bp[t, path[-1]]))
    return path[::-1]


def decode(P, docs, vocab, T=None):
    mask = np.array([l in NEVER for l in vocab])
    logp = np.log(np.clip(P, 1e-12, 1)); logp[:, mask] = -1e9
    out, off = {}, 0
    for d in docs:
        lp = logp[off:off + len(d)]; off += len(d)
        idx = lp.argmax(1) if T is None else viterbi(lp, *T)
        out[d.id] = [vocab[i] for i in idx]
    return out


def all_variants(P, T, docs, vocab):
    out = {}
    for s in P:
        out[s] = decode(P[s], docs, vocab)
        out[s + "+viterbi"] = decode(P[s], docs, vocab, T["plain"])
        out[s + "+vitpmi"] = decode(P[s], docs, vocab, T["pmi"])
    return out


def run_cv(vocab, train, dev):
    t0 = time.time()
    folds = grouped_folds(train, 5, SEED)

    def one(f):
        fs = set(f)
        fit = [train[i] for i in range(len(train)) if i not in fs]
        app = [train[i] for i in f]
        P, T = fit_predict(fit, app, vocab, inner_jobs=5)
        return all_variants(P, T, app, vocab)

    res = Parallel(n_jobs=5)(delayed(one)(f) for f in folds)
    gold = {d.id: d.labels for d in train}
    variants = list(res[0])
    print(f"\n=== grouped 5-fold CV on 80 train docs ({time.time()-t0:.0f}s) ===")
    print(f"{'variant':18s} {'macro13 mean±sd':>18s} {'pooled':>8s} {'CL':>7s} {'IT':>7s}  per-fold")
    pooled_preds, cv_mean = {}, {}
    for v in variants:
        fs = []
        for f, r in zip(folds, res):
            fs.append(score(r[v], {train[i].id: train[i].labels for i in f}, vocab)["macro_f1"])
        pooled = {k: lab for r in res for k, lab in r[v].items()}
        pooled_preds[v] = pooled
        per = score_by_domain(pooled, train, vocab)
        cv_mean[v] = float(np.mean(fs))
        print(f"{v:18s} {np.mean(fs):9.4f} ± {np.std(fs):.4f} {per['ALL']['macro_f1']:8.4f} "
              f"{per['CL']['macro_f1']:7.4f} {per['IT']['macro_f1']:7.4f}  "
              + " ".join(f"{x:.3f}" for x in fs))
    best = max(cv_mean, key=cv_mean.get)
    print(f"\nCV-selected variant: {best}")
    print(f"\n--- CV pooled per-domain, {best} ---")
    print(domain_report(score_by_domain(pooled_preds[best], train, vocab)))
    print(confusion_report(confusion(pooled_preds[best], gold, vocab, top=10), gold))


    t0 = time.time()
    P, T = fit_predict(train, dev, vocab, inner_jobs=5)
    dv = all_variants(P, T, dev, vocab)
    print(f"\n=== dev (fit on 80 train docs, {time.time()-t0:.0f}s) ===")
    for v, p in dv.items():
        per = score_by_domain(p, dev, vocab)
        print(f"{v:18s} macro13 {per['ALL']['macro_f1']:.4f}  supported "
              f"{per['ALL']['macro_f1_supported']:.4f}  CL {per['CL']['macro_f1']:.4f}  "
              f"IT {per['IT']['macro_f1']:.4f}")
    print(f"\n--- dev per-domain, {best} ---")
    print(domain_report(score_by_domain(dv[best], dev, vocab)))
    gdev = {d.id: d.labels for d in dev}
    print(confusion_report(confusion(dv[best], gdev, vocab, top=10), gdev))
    (OUT / "lexical_dev_predictions.json").write_text(json.dumps(dv[best]))
    np.save(OUT / "lexical_dev_proba.npy", P["s2"])
    for v, p in dv.items():
        (OUT / f"lexical_dev_predictions.{v.replace('+', '_')}.json").write_text(json.dumps(p))
    print(f"wrote {OUT/'lexical_dev_predictions.json'} ({best})")


def run_submit(vocab, train, dev, variant):
    test = D.load_all(DATA, "test")
    fit = train + dev
    P, T = fit_predict(fit, test, vocab, inner_jobs=5)
    pred = all_variants(P, T, test, vocab)[variant]
    errs = validate_submission(pred, test, vocab)
    if errs:
        raise SystemExit("submission invalid:\n" + "\n".join(errs))
    subs = ROOT / "subs"; subs.mkdir(exist_ok=True)
    zp = subs / "lexical.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("prediction.json", json.dumps(pred))
    (OUT / "lexical_test_predictions.json").write_text(json.dumps(pred))
    np.save(OUT / "lexical_test_proba.npy", P["s2"])
    print(f"wrote {zp} ({variant}, {len(pred)} docs, {sum(map(len, pred.values()))} sentences)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--variant", default="s2",
                    choices=[f"{a}{b}" for a in ("s1", "s2") for b in ("", "+viterbi", "+vitpmi")])
    a = ap.parse_args()
    if not (a.cv or a.submit):
        ap.error("pass --cv and/or --submit")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    vocab = D.load_labels(DATA)
    train, dev = D.load_all(DATA, "train"), D.load_all(DATA, "dev")
    if a.cv:
        run_cv(vocab, train, dev)
    if a.submit:
        run_submit(vocab, train, dev, a.variant)


if __name__ == "__main__":
    main()

import argparse
import json
import os
import sys
import time

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "5"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lsi import data as D
from lsi.metrics import evaluate, report, scores, decide, tune_per_label, write_submission

OUT = os.path.join(ROOT, "runs", "lexical_kfold")
L, K, SEED, N_JOBS = 100, 5, 0, 5
C = 16.0
HEAD, TAIL = 30_000, 10_000
MIN_DF, MAX_DF, MAX_FEAT = 3, 0.9, 400_000
TOKEN_PATTERN = r"(?u)<\w+>|\b\w\w+\b"


def truncate(t, head=HEAD, tail=TAIL):
    return t if len(t) <= head + tail else t[:head] + " \n " + t[-tail:]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def counts(pool_txt, test_txt):
    cache = os.path.join(OUT, "_counts.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        n = int(z["n_pool"])
        X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        return X[:n], X[n:]
    cv = CountVectorizer(analyzer="word", ngram_range=(1, 2), min_df=MIN_DF,
                         token_pattern=TOKEN_PATTERN, dtype=np.float32)
    Xp = cv.fit_transform(pool_txt).tocsr()
    chunks = [test_txt[i:i + 1000] for i in range(0, len(test_txt), 1000)]
    Xt = sp.vstack(Parallel(n_jobs=N_JOBS)(delayed(cv.transform)(c) for c in chunks)).tocsr()
    X = sp.vstack([Xp, Xt]).tocsr()
    np.savez(cache, data=X.data, indices=X.indices, indptr=X.indptr,
             shape=np.array(X.shape), n_pool=len(pool_txt))
    return Xp, Xt


def tfidf_fold(Xc_tr, others):
    n = Xc_tr.shape[0]
    df = np.bincount(Xc_tr.indices, minlength=Xc_tr.shape[1])
    keep = (df >= MIN_DF) & (df <= MAX_DF * n)
    idx = np.flatnonzero(keep)
    if len(idx) > MAX_FEAT:
        tot = np.asarray(Xc_tr.sum(0)).ravel()[idx]
        idx = np.sort(idx[np.argsort(-tot, kind="stable")[:MAX_FEAT]])
    idf = (np.log((1 + n) / (1 + df[idx])) + 1).astype(np.float32)

    def tf(X):
        X = X[:, idx].tocsr().astype(np.float32)
        X.data = np.log(X.data) + 1
        X = X @ sp.diags(idf)
        return normalize(X, norm="l2", copy=False).tocsr()
    return tf(Xc_tr), [tf(X) for X in others]


def _fit_one(X, y, Xs):
    clf = LogisticRegression(C=C, solver="liblinear", max_iter=200)
    clf.fit(X, y)
    return [clf.predict_proba(x)[:, 1].astype(np.float32) for x in Xs]


def fit_ovr(Xtr, Ytr, Xs):
    res = Parallel(n_jobs=N_JOBS, prefer="processes")(
        delayed(_fit_one)(Xtr, Ytr[:, l], Xs) for l in range(L))
    return [np.stack([r[k] for r in res], 1) for k in range(len(Xs))]


def ev_line(e):
    return report(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lc", action="store_true", help="learning-curve probe after folds")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    T0 = time.time()
    tr, dv, te = D.load("train"), D.load("dev"), D.load("test")
    pool = tr + dv
    Y = D.label_matrix(pool)
    ntr, N = len(tr), len(pool)
    log(f"pooled {N} ({ntr} train + {len(dv)} dev), test {len(te)}")


    fpath = os.path.join(OUT, "folds.json")
    if os.path.exists(fpath):
        fd = json.load(open(fpath)); fold = np.array([fd[d.id] for d in pool])
    else:
        fold = np.random.RandomState(SEED).permutation(np.arange(N) % K)
        json.dump({d.id: int(f) for d, f in zip(pool, fold)}, open(fpath, "w"))
    log("fold sizes", np.bincount(fold).tolist())

    t0 = time.time()
    Xc, Xct = counts([truncate(d.text) for d in pool], [truncate(d.text) for d in te])
    log(f"counts {Xc.shape} test {Xct.shape} in {time.time()-t0:.0f}s")

    oof = np.zeros((N, L), np.float32)
    test = np.zeros((len(te), L), np.float32)
    fold_s = {}
    for k in range(K):
        ck = os.path.join(OUT, f"_fold{k}.npz")
        if os.path.exists(ck):
            z = np.load(ck); oof[fold == k] = z["oof"]; test += z["te"] / K
            fold_s[k] = float(z["sec"]); log(f"fold {k} loaded ({fold_s[k]:.0f}s)"); continue
        t0 = time.time()
        trn, hold = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        Xtr, (Xho, Xte) = tfidf_fold(Xc[trn], [Xc[hold], Xct])
        tv = time.time() - t0
        po, pt = fit_ovr(Xtr, Y[trn].astype(np.int8), [Xho, Xte])
        sec = time.time() - t0
        np.savez(ck, oof=po, te=pt, sec=sec)
        oof[hold] = po; test += pt / K; fold_s[k] = sec
        e = scores(decide(po, tune_per_label(po, Y[hold])), Y[hold])
        log(f"fold {k}: {Xtr.shape} tfidf {tv:.0f}s total {sec:.0f}s | in-fold per-label "
            f"macro-F1 {e['macro_f1']:.4f}")

    dv_oof, Ydv = oof[ntr:], Y[ntr:]
    tr_oof, Ytr = oof[:ntr], Y[:ntr]
    thr_tr = tune_per_label(tr_oof, Ytr)
    np.save(os.path.join(OUT, "oof_probs.npy"), oof)
    np.save(os.path.join(OUT, "dev_probs.npy"), dv_oof)
    np.save(os.path.join(OUT, "test_probs.npy"), test)
    np.save(os.path.join(OUT, "thresholds_trainoof.npy"), thr_tr)
    write_submission(decide(test, thr_tr), [d.id for d in te],
                     os.path.join(ROOT, "subs", "lexical_kfold.zip"))

    R = {"variant": "w12_C16_nobal", "fold_runtime_s": fold_s}
    m = lambda S, t: scores(decide(S, t), Ydv)

    a = m(dv_oof, thr_tr)
    R["a_trainoof_thr_on_dev_oof"] = {k: a[k] for k in ("macro_f1", "micro_f1", "subset_acc")}
    log(f"(a) dev OOF, thresholds from train OOF: macro-F1 {a['macro_f1']:.4f} "
        f"micro {a['micro_f1']:.4f}")

    eb = evaluate(dv_oof, Ydv)
    R["b_dev_oof_evaluate"] = {k: v for k, v in eb.items() if k != "thresholds"}
    log("(b) dev OOF evaluate:", report(eb))

    R["c"] = {"trainoof_thr": a["macro_f1"], "dev_insample_thr": eb["macro_f1_perlabel"],
              "dev_crossfit": eb["macro_f1_crossfit"]}

    rng = np.random.default_rng(0); idx = rng.permutation(len(dv_oof))
    h = (idx[: len(idx) // 2], idx[len(idx) // 2:])
    cf_dev, cf_tr = [], []
    for fit, evl in (h, h[::-1]):
        cf_dev.append(scores(decide(dv_oof[evl], tune_per_label(dv_oof[fit], Ydv[fit])),
                             Ydv[evl])["macro_f1"])
        cf_tr.append(scores(decide(dv_oof[evl], thr_tr), Ydv[evl])["macro_f1"])

    R["c"]["halves_trainoof_thr"] = float(np.mean(cf_tr))
    pool_cf = []
    for fit, evl in (h, h[::-1]):
        t = tune_per_label(np.vstack([tr_oof, dv_oof[fit]]), np.vstack([Ytr, Ydv[fit]]))
        pool_cf.append(scores(decide(dv_oof[evl], t), Ydv[evl])["macro_f1"])
    R["c"]["halves_trainoof+halfdev_thr"] = float(np.mean(pool_cf))
    log(f"(c) dev OOF: train-OOF thr {a['macro_f1']:.4f} | dev in-sample thr "
        f"{eb['macro_f1_perlabel']:.4f} | dev cross-fit {eb['macro_f1_crossfit']:.4f} | "
        f"same halves w/ train-OOF thr {np.mean(cf_tr):.4f} | train-OOF+other-half thr "
        f"{np.mean(pool_cf):.4f}")


    ref = {}
    cpath = os.path.join(ROOT, "baselines", "_cache", "probs_w12_C16_nobal.npz")
    if os.path.exists(cpath):
        old1 = np.load(cpath)["dv"]
        e1 = evaluate(old1, Ydv)
        ref["trainonly_w12_C16"] = {k: v for k, v in e1.items() if k != "thresholds"}
        ref["trainonly_w12_C16"]["with_trainoof_thr"] = m(old1, thr_tr)["macro_f1"]
        log("ref train-only w12_C16 dev:", report(e1),
            f"| with train-OOF thr {ref['trainonly_w12_C16']['with_trainoof_thr']:.4f}")
    old = np.load(os.path.join(ROOT, "runs", "lexical", "dev_probs.npy"))
    e2 = evaluate(old, Ydv)
    ref["trainonly_avg2"] = {k: v for k, v in e2.items() if k != "thresholds"}
    ref["trainonly_avg2"]["with_trainoof_thr"] = m(old, thr_tr)["macro_f1"]
    log("ref train-only avg2 (runs/lexical) dev:", report(e2),
        f"| with train-OOF thr {ref['trainonly_avg2']['with_trainoof_thr']:.4f}")
    R["reference"] = ref


    neu = np.mean([np.load(os.path.join(ROOT, "runs", r, "dev_probs.npy"))
                   for r in ("full_k4", "kd_half", "lastk_k4", "kd_control")], 0)
    eS0 = evaluate(0.5 * neu + 0.5 * old, Ydv)
    eS1 = evaluate(0.5 * neu + 0.5 * dv_oof, Ydv)
    R["blend"] = {"S0_old_lexical": {k: v for k, v in eS0.items() if k != "thresholds"},
                  "S1_kfold_oof": {k: v for k, v in eS1.items() if k != "thresholds"}}
    log("blend S0 (old lexical):", report(eS0))
    log("blend S1 (kfold OOF)  :", report(eS1))


    if args.lc:
        t0 = time.time()
        sub = np.sort(np.random.RandomState(1).permutation(ntr)[: int(0.8 * ntr)])
        Xs, (Xd,) = tfidf_fold(Xc[sub], [Xc[ntr:]])
        (pd_,) = fit_ovr(Xs, Ytr[sub].astype(np.int8), [Xd])
        el = evaluate(pd_, Ydv)
        R["lc_80pct_train"] = {k: v for k, v in el.items() if k != "thresholds"}
        R["lc_80pct_train"]["n"] = len(sub); R["lc_80pct_train"]["sec"] = time.time() - t0
        log(f"LC 80% train ({len(sub)} docs) -> dev:", report(el))

    R["total_runtime_s"] = time.time() - T0
    json.dump(R, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=float)
    log(f"done in {R['total_runtime_s']:.0f}s")


if __name__ == "__main__":
    main()

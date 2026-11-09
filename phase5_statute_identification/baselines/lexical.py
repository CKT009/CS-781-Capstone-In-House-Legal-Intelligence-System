import argparse
import json
import os
import time
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "Phase_5_LSI")
OUT = os.path.join(ROOT, "baselines")
CACHE = os.path.join(OUT, "_cache")
SUBS = os.path.join(ROOT, "subs")
L = 100
N_JOBS = 16


def load(path):
    with open(path) as f:
        return json.load(f)


def truncate(t, head, tail):
    if len(t) <= head + tail:
        return t
    return t[:head] + " \n " + t[-tail:]


def label_matrix(recs):
    Y = np.zeros((len(recs), L), dtype=np.int8)
    for i, r in enumerate(recs):
        for l in set(r["labels"]):
            Y[i, l] = 1
    return Y


FEATS = {

    "w12": (dict(analyzer="word", ngram_range=(1, 2), min_df=3, max_df=0.9,
                 max_features=400_000, sublinear_tf=True, dtype=np.float32,
                 token_pattern=r"(?u)<\w+>|\b\w\w+\b"), 30_000, 10_000),
    "c25": (dict(analyzer="char_wb", ngram_range=(2, 5), min_df=5, max_df=0.9,
                 max_features=400_000, sublinear_tf=True, dtype=np.float32),
            8_000, 4_000),
}


def _fit_transform_feat(name, tr, dv, te):
    kw, head, tail = FEATS[name]
    fs = [os.path.join(CACHE, f"feat_{name}_{k}.npz") for k in ("tr", "dv", "te")]
    if all(os.path.exists(f) for f in fs):
        return [sp.load_npz(f).tocsr() for f in fs]
    t0 = time.time()
    txt = lambda rs: [truncate(r["text"], head, tail) for r in rs]
    vec = TfidfVectorizer(**kw)
    Xtr = vec.fit_transform(txt(tr))

    def tf(ch):
        return vec.transform(ch)
    def chunked(rs):
        tx = txt(rs)
        chunks = [tx[i:i + 1000] for i in range(0, len(tx), 1000)]
        return sp.vstack(Parallel(n_jobs=N_JOBS)(delayed(tf)(c) for c in chunks)).tocsr()
    Xdv, Xte = chunked(dv), chunked(te)
    for f, X in zip(fs, (Xtr, Xdv, Xte)):
        sp.save_npz(f, X, compressed=False)
    print(f"[feat {name}] {Xtr.shape} in {time.time()-t0:.0f}s", flush=True)
    return Xtr, Xdv, Xte


def get_feats(names, tr, dv, te):
    parts = [_fit_transform_feat(n, tr, dv, te) for n in names]
    if len(parts) == 1:
        return parts[0]
    return [sp.hstack([p[i] for p in parts]).tocsr() for i in range(3)]


def _fit_one(X, y, C, cw, Xs):
    clf = LogisticRegression(C=C, solver="liblinear", class_weight=cw, max_iter=200)
    clf.fit(X, y)
    return [clf.predict_proba(x)[:, 1].astype(np.float32) for x in Xs]


def fit_ovr(Xtr, Ytr, Xs, C, cw):
    res = Parallel(n_jobs=N_JOBS, prefer="processes")(
        delayed(_fit_one)(Xtr, Ytr[:, l], C, cw, Xs) for l in range(L))
    return [np.stack([r[k] for r in res], 1) for k in range(len(Xs))]


def f1_parts(Y, P):
    tp = (Y & P).sum(0).astype(np.float64)
    fp = ((1 - Y) & P).sum(0)
    fn = (Y & (1 - P)).sum(0)
    return tp, fp, fn


def metrics(Y, P):
    P = P.astype(np.int8)
    tp, fp, fn = f1_parts(Y, P)
    d = 2 * tp + fp + fn
    f1 = np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0)
    micro = 2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)
    subset = float((Y == P).all(1).mean())
    return dict(macro_f1=float(f1.mean()), micro_f1=float(micro), subset_acc=subset), f1


def top1_fallback(prob, P):
    P = P.copy()
    empty = P.sum(1) == 0
    P[empty, prob[empty].argmax(1)] = 1
    return P


GRID = np.round(np.arange(0.02, 0.96, 0.01), 3)


def best_global(Y, prob):
    best = (-1, None)
    for t in GRID:
        m, _ = metrics(Y, prob >= t)
        if m["macro_f1"] > best[0]:
            best = (m["macro_f1"], t)
    return best[1]


def per_label_thr(Y, prob):
    thr = np.zeros(L, dtype=np.float32)
    for l in range(L):
        y = Y[:, l].astype(bool)
        p = prob[:, l]
        P = p[None, :] >= GRID[:, None]
        tp = (P & y).sum(1); fp = (P & ~y).sum(1); fn = y.sum() - tp
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        thr[l] = GRID[int(np.argmax(f1))]
    return thr


def evaluate(Y, prob):
    out = {}
    g = best_global(Y, prob)
    out["global_thr"] = float(g)
    m, _ = metrics(Y, prob >= g); out["global"] = m
    m, _ = metrics(Y, top1_fallback(prob, prob >= g)); out["global_top1"] = m
    thr = per_label_thr(Y, prob)
    m, f1 = metrics(Y, prob >= thr); out["perlabel"] = m
    m, _ = metrics(Y, top1_fallback(prob, prob >= thr)); out["perlabel_top1"] = m

    rng = np.random.RandomState(0)
    idx = rng.permutation(len(Y)); A, B = idx[: len(Y) // 2], idx[len(Y) // 2:]
    cf, cf1, cfg = [], [], []
    for fit, ev in ((A, B), (B, A)):
        t = per_label_thr(Y[fit], prob[fit])
        cf.append(metrics(Y[ev], prob[ev] >= t)[0])
        cf1.append(metrics(Y[ev], top1_fallback(prob[ev], prob[ev] >= t))[0])
        gg = best_global(Y[fit], prob[fit])
        cfg.append(metrics(Y[ev], prob[ev] >= gg)[0])
    avg = lambda ms: {k: float(np.mean([x[k] for x in ms])) for k in ms[0]}
    out["perlabel_crossfit"] = avg(cf)
    out["perlabel_top1_crossfit"] = avg(cf1)
    out["global_crossfit"] = avg(cfg)
    return out, thr, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True); os.makedirs(SUBS, exist_ok=True)
    T0 = time.time()

    tr = load(os.path.join(DATA, "Train", "train.json"))
    dv = load(os.path.join(DATA, "Train", "dev.json"))
    te = load(os.path.join(DATA, "Test", "test.json"))
    Ytr, Ydv = label_matrix(tr), label_matrix(dv)
    freq_tr = Ytr.sum(0); freq_dv = Ydv.sum(0)
    print(f"loaded {len(tr)}/{len(dv)}/{len(te)} in {time.time()-T0:.0f}s", flush=True)


    variants = [
        ("w12", 4.0, None), ("w12", 16.0, None), ("w12", 64.0, None),
        ("w12", 4.0, "balanced"), ("w12", 16.0, "balanced"),
    ]
    if not args.quick:
        variants += [("w12+c25", 16.0, None), ("w12+c25", 16.0, "balanced")]

    results, probs = {}, {}
    rpath = os.path.join(OUT, "lexical_results.json")
    for feat, C, cw in variants:
        name = f"{feat}_C{C:g}_{'bal' if cw else 'nobal'}"
        pf = os.path.join(CACHE, f"probs_{name}.npz")
        t0 = time.time()
        if os.path.exists(pf):
            z = np.load(pf); pdv, pte = z["dv"], z["te"]; fit_s = float(z["fit_s"])
        else:
            Xtr, Xdv, Xte = get_feats(feat.split("+"), tr, dv, te)
            pdv, pte = fit_ovr(Xtr, Ytr, [Xdv, Xte], C, cw)
            fit_s = time.time() - t0
            np.savez(pf, dv=pdv, te=pte, fit_s=fit_s)
        ev, thr, f1 = evaluate(Ydv, pdv)
        ev.update(features=feat, C=C, class_weight=cw, fit_runtime_s=round(fit_s, 1))
        results[name] = ev; probs[name] = (pdv, pte, thr, f1)
        print(f"{name:28s} global {ev['global']['macro_f1']:.4f} perlabel "
              f"{ev['perlabel']['macro_f1']:.4f} crossfit {ev['perlabel_crossfit']['macro_f1']:.4f} "
              f"micro {ev['perlabel']['micro_f1']:.4f} subset {ev['perlabel']['subset_acc']:.4f} "
              f"({fit_s:.0f}s)", flush=True)
        json.dump({"variants": results}, open(rpath, "w"), indent=1)


    ranked = sorted(results, key=lambda k: -results[k]["perlabel_crossfit"]["macro_f1"])
    if len(ranked) >= 2:
        for k in (2, 3):
            names = ranked[:k]
            pdv = np.mean([probs[n][0] for n in names], 0)
            pte = np.mean([probs[n][1] for n in names], 0)
            ev, thr, f1 = evaluate(Ydv, pdv)
            name = f"avg{k}[" + ",".join(names) + "]"
            ev.update(features="prob-average", members=names, fit_runtime_s=0.0)
            results[name] = ev; probs[name] = (pdv, pte, thr, f1)
            print(f"{name}: perlabel {ev['perlabel']['macro_f1']:.4f} crossfit "
                  f"{ev['perlabel_crossfit']['macro_f1']:.4f}", flush=True)


    best = max(results, key=lambda k: results[k]["perlabel_crossfit"]["macro_f1"])
    pdv, pte, thr, f1 = probs[best]
    ev = results[best]
    use_top1 = ev["perlabel_top1_crossfit"]["macro_f1"] > ev["perlabel_crossfit"]["macro_f1"]

    np.save(os.path.join(OUT, "lexical_dev_probs.npy"), pdv.astype(np.float32))
    np.save(os.path.join(OUT, "lexical_test_probs.npy"), pte.astype(np.float32))
    np.save(os.path.join(OUT, "lexical_thresholds.npy"), thr.astype(np.float32))

    P = pte >= thr
    if use_top1:
        P = top1_fallback(pte, P)
    preds = [{"id": r["id"], "pred": sorted(int(j) for j in np.nonzero(P[i])[0])}
             for i, r in enumerate(te)]
    zp = os.path.join(SUBS, "lexical.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("prediction.json", json.dumps(preds))

    worst = np.argsort(f1)[:10]
    worst10 = [dict(label=int(l), dev_f1=round(float(f1[l]), 4), train_freq=int(freq_tr[l]),
                    dev_freq=int(freq_dv[l]), thr=float(thr[l])) for l in worst]
    rank = np.argsort(np.argsort(freq_tr))
    corr = float(np.corrcoef(np.log(freq_tr), f1)[0, 1])
    summary = dict(best=best, use_top1_fallback=bool(use_top1),
                   selection="max per-label-threshold 2-fold cross-fit macro-F1 on dev",
                   worst10=worst10, corr_logfreq_f1=corr,
                   empty_test_preds_before_fallback=int(((pte >= thr).sum(1) == 0).sum()),
                   mean_test_labels=float(P.sum(1).mean()),
                   total_runtime_s=round(time.time() - T0, 1))
    json.dump({"summary": summary, "variants": results}, open(rpath, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

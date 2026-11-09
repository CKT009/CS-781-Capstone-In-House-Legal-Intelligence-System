import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "5"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lsi import data as D
from lsi.metrics import evaluate, report, decide, write_submission

CACHE = ROOT / "baselines" / "_cache"
OUT = ROOT / "runs" / "knn"
KMAX = 100
N_JOBS = 5


def feats(name):
    return [normalize(sp.load_npz(CACHE / f"feat_{name}_{s}.npz").tocsr()).astype(np.float32)
            for s in ("tr", "dv", "te")]


def _topk(Xq, XtrT):
    S = (Xq @ XtrT).toarray()
    idx = np.argpartition(-S, KMAX, axis=1)[:, :KMAX]
    sim = np.take_along_axis(S, idx, 1)
    o = np.argsort(-sim, 1)
    return np.take_along_axis(idx, o, 1).astype(np.int32), np.take_along_axis(sim, o, 1)


def topk(Xq, XtrT, chunk=500):
    res = Parallel(n_jobs=N_JOBS, prefer="processes")(
        delayed(_topk)(Xq[i:i + chunk], XtrT) for i in range(0, Xq.shape[0], chunk))
    return np.vstack([r[0] for r in res]), np.vstack([r[1] for r in res])


def vote(idx, sim, Ytr, k, p):
    w = np.maximum(sim[:, :k], 0) ** p
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
    return np.einsum("nk,nkl->nl", w, Ytr[idx[:, :k]]).astype(np.float32)


def main():
    T0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    dev, test = D.load("dev"), D.load("test")
    Ytr, Ydv = D.label_matrix(D.load("train")), D.label_matrix(dev)


    blocks = {"w12": feats("w12")}
    nbrs, timing = {}, {}
    t0 = time.time()
    Xtr, Xdv, Xte = blocks["w12"]
    nbrs["w12"] = (topk(Xdv, Xtr.T.tocsr()), topk(Xte, Xtr.T.tocsr()))
    timing["w12"] = time.time() - t0
    print(f"[knn w12] {timing['w12']:.0f}s", flush=True)

    t0 = time.time()
    c = feats("c25")
    s = np.float32(1 / np.sqrt(2))
    Xtr2, Xdv2, Xte2 = (sp.hstack([a * s, b * s]).tocsr() for a, b in zip(blocks["w12"], c))
    del c
    nbrs["w12+c25"] = (topk(Xdv2, Xtr2.T.tocsr()), topk(Xte2, Xtr2.T.tocsr()))
    timing["w12+c25"] = time.time() - t0
    print(f"[knn w12+c25] {timing['w12+c25']:.0f}s", flush=True)

    results, best = {}, None
    for f, ((di, ds), (ti, ts)) in nbrs.items():
        for k in (10, 25, 50, 100):
            for p in (1, 3):
                Sd = vote(di, ds, Ytr, k, p)
                e = evaluate(Sd, Ydv)
                name = f"{f}_k{k}_p{p}"
                results[name] = {kk: v for kk, v in e.items() if kk != "thresholds"}
                print(f"{name:18s} {report(e)}", flush=True)
                if best is None or e["macro_f1_crossfit"] > best[1]["macro_f1_crossfit"]:
                    best = (name, e, Sd, vote(ti, ts, Ytr, k, p))

    name, e, Sd, St = best
    print(f"best {name}: {report(e)}")
    np.save(OUT / "dev_probs.npy", Sd.astype(np.float32))
    np.save(OUT / "test_probs.npy", St.astype(np.float32))
    write_submission(decide(St, e["thresholds"]), [d.id for d in test], ROOT / "subs" / "knn.zip")


    R = ROOT / "runs"
    neu = np.mean([np.load(R / m / "dev_probs.npy") for m in
                   ("full_k4", "kd_half", "lastk_k4", "kd_control")], 0)
    S0 = 0.5 * neu + 0.5 * np.load(R / "lexical" / "dev_probs.npy")
    blend = {}
    e0 = evaluate(S0, Ydv)
    blend["w0.0"] = {kk: v for kk, v in e0.items() if kk != "thresholds"}
    print(f"S0        {report(e0)}")
    corr = float(np.corrcoef(Sd.ravel(), S0.ravel())[0, 1])
    pl = [np.corrcoef(Sd[:, j], S0[:, j])[0, 1] for j in range(100) if Sd[:, j].std() > 0]
    print(f"corr(knn,S0) pooled {corr:.4f} | mean per-label {np.mean(pl):.4f}")
    for w in (0.1, 0.2, 0.3):
        eb = evaluate((1 - w) * S0 + w * Sd, Ydv)
        blend[f"w{w}"] = {kk: v for kk, v in eb.items() if kk != "thresholds"}
        print(f"w={w:.1f}     {report(eb)}")

    json.dump({"best": name, "grid": results, "timing_s": timing,
               "corr_pooled": corr, "corr_perlabel_mean": float(np.mean(pl)), "blend": blend,
               "total_runtime_s": time.time() - T0},
              open(OUT / "results.json", "w"), indent=1)
    print(f"total {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()

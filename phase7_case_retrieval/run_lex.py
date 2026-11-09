from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from pcr import data as D, metrics as M, parse as P

OUT = ROOT / "runs" / "lex"
SPLITS = ("train", "dev", "test")


def note(msg):
    with open(OUT / "NOTES.md", "a") as f:
        f.write(f"- {time.strftime('%m-%d %H:%M')} {msg}\n")
    print(msg, flush=True)


def stage_parse():
    dates, meta = {}, {}
    for s in SPLITS:
        pr = P.parse_split(s, force=True)
        rep = P.date_report(pr)
        note(f"parse {s}: {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rep.items()})}")
        for k, r in pr.items():
            if r["date"]:
                dates[k] = r["date"]; meta[k] = r["date_src"]
    (OUT / "dates.json").write_text(json.dumps(dates))
    (OUT / "dates_src.json").write_text(json.dumps(meta))
    note(f"dates.json: {len(dates)} docs")


def doc_dates(split):
    pr = P.parse_split(split)
    return {k: (r["date"], r["date_src"], r["date_lb"]) for k, r in pr.items()}


def keep_fn(dd, mode="any"):
    def keep(qid, cid):
        qd, qs, _ = dd.get(qid, (None, None, None))
        cd, cs, _ = dd.get(cid, (None, None, None))
        if qd is None or cd is None:
            return True
        if mode == "exactq" and qs == "lb":
            return True
        if mode == "tol1":
            return int(cd[:4]) <= int(qd[:4]) + 1
        if mode == "exact_both" and (qs == "lb" or cs == "lb"):
            return True
        return cd <= qd
    return keep


def stage_temporal():
    for s in ("train", "dev"):
        q, pool = D.load(s)
        dd = doc_dates(s)
        for mode in ("any", "tol1", "exactq", "exact_both"):
            kp = keep_fn(dd, mode)
            n = rm = rel = relk = 0
            for qid, v in q.items():
                for c in D.candidates(qid, pool):
                    k = kp(qid, c); n += 1; rm += (not k)
                    if c in v["rel"]:
                        rel += 1; relk += k
            note(f"temporal {s} mode={mode}: pool removed {rm / n:.3f}, relevant kept {relk / rel:.4f} ({relk}/{rel})")

        kp = keep_fn(dd, "any")
        from collections import Counter
        viol = Counter()
        for qid, v in q.items():
            for c in v["rel"]:
                if not kp(qid, c):
                    viol[(dd[qid][1], dd[c][1])] += 1
        note(f"temporal {s} violations by (q_src,c_src): {dict(viol)}")


import numpy as np
from pcr import lexical as L

FIRST_N = 500
FIELDS = ("full", "head", "act", "first", "bi", "tri")
SHINGLE = {"bi": 2, "tri": 3}
_CACHE = {}


def field_text(r, f):
    if f == "full":
        return (r["headnote"] + " " + r["body"]) if r["has_header"] else r["body"]
    if f == "head":
        return r["headnote"]
    if f == "act":
        if r["act_field"]:
            return r["act_field"]
        return " ".join(r["acts"]) + " " + " ".join(x.replace("|", " ") for x in r["act_secs_all"])
    if f == "first":
        return L.first_words(r["body"], FIRST_N)


def shingles(t, n):
    return [" ".join(t[i:i + n]) for i in range(len(t) - n + 1)]


def query_text(r, f):
    t = field_text(r, f)
    return t if t.strip() else L.first_words(r["body"], FIRST_N)


def ctx(split):
    if split in _CACHE:
        return _CACHE[split]
    q, pool = D.load(split)
    pr = P.parse_split(split)
    ids = list(pool)
    idx = {c: i for i, c in enumerate(ids)}
    qids = list(q)
    from concurrent.futures import ProcessPoolExecutor
    toks = {}
    with ProcessPoolExecutor(16) as ex:
        for f in FIELDS:
            if f in SHINGLE:
                continue
            toks[f] = list(ex.map(L.tok, [field_text(pr[c], f) for c in ids], chunksize=64))
            toks["q_" + f] = list(ex.map(L.tok, [query_text(pr[x], f) for x in qids], chunksize=16))
    for f, n in SHINGLE.items():
        toks[f] = [shingles(t, n) for t in toks["full"]]
        toks["q_" + f] = [shingles(t, n) for t in toks["q_full"]]
    C = dict(q=q, pool=pool, pr=pr, ids=ids, idx=idx, qids=qids, toks=toks)
    _CACHE[split] = C
    return C


def bm25_scores(C, f, k1, b, qw="log"):
    bm = L.BM25(C["toks"][f], k1, b)
    return bm.score_matrix(C["toks"]["q_" + f], qw)


def masks(C, temporal):
    pr, ids, qids = C["pr"], C["ids"], C["qids"]
    A = np.ones((len(qids), len(ids)), bool)
    cd = np.array([pr[c]["date"] or "9999" for c in ids])
    has = np.array([pr[c]["date"] is not None for c in ids])
    for i, qid in enumerate(qids):
        A[i, C["idx"][qid]] = False
        r = pr[qid]
        if temporal and r["date"] and r["date_src"] != "lb":
            A[i] &= ~(has & (cd > r["date"]))
    return A


def rank_eval(C, S, allowed, k=7, tie=None):
    S = np.where(allowed, S + (0 if tie is None else 1e-6 * tie), -np.inf)
    top = np.argsort(-S, 1)[:, :k]
    ranked = {qid: [C["ids"][j] for j in top[i]] for i, qid in enumerate(C["qids"])}
    return M.score(ranked, C["q"]), ranked


def tune_bm25(C, f, A):
    best = None
    for qw in (("log",) if f in SHINGLE else ("log", "bin")):
        for k1 in ((1.2, 5.0, 20.0) if f in SHINGLE else (0.9, 1.6, 3.0, 5.0, 8.0, 12.0, 20.0, 50.0)):
            bm = L.BM25(C["toks"][f], k1, 0.75)
            Q = bm.qmatrix(C["toks"]["q_" + f], qw)
            for b in ((0.75, 1.0) if f in SHINGLE else (0.5, 0.75, 0.9, 1.0)):
                bm.set_params(k1, b)
                S = np.asarray((Q @ bm.W.T).todense())
                m, _ = rank_eval(C, S, A)
                if best is None or m["f1"] > best[0]:
                    best = (m["f1"], k1, b, qw)
    return best


def set_sim_matrix(C, key, kind="jac", idf=None):
    pr, ids, qids = C["pr"], C["ids"], C["qids"]
    sets = [set(pr[c][key]) for c in ids]
    S = np.zeros((len(qids), len(ids)), np.float32)
    for i, qid in enumerate(qids):
        a = set(pr[qid][key])
        if not a:
            continue
        for j, s in enumerate(sets):
            if s and (a & s):
                S[i, j] = L.jaccard(a, s) if kind == "jac" else L.idf_overlap(a, s, idf)
    return S


def popularity_matrix(C, loo):
    qtr, _ = D.load("train")
    pop = L.popularity(qtr)
    S = np.zeros((len(C["qids"]), len(C["ids"])), np.float32)
    v = np.array([pop.get(c, 0) for c in C["ids"]], np.float32)
    S[:] = v
    if loo:
        for i, qid in enumerate(C["qids"]):
            for c in C["q"][qid]["rel"]:
                S[i, C["idx"][c]] -= 1
    return S


def bench_matrix(C):
    return set_sim_matrix(C, "judges", "jac")


def stage_signals():
    params = {}
    Ct = ctx("train")
    At = masks(Ct, False)
    for f in FIELDS:
        best = tune_bm25(Ct, f, At)
        params[f] = dict(k1=best[1], b=best[2], qw=best[3])
        note(f"tune bm25[{f}] on TRAIN: F1 {best[0]:.4f} k1={best[1]} b={best[2]} qw={best[3]}")
    (OUT / "bm25_params.json").write_text(json.dumps(params))
    Cd = ctx("dev")
    A0, A1 = masks(Cd, False), masks(Cd, True)
    full = bm25_scores(Cd, "full", **params["full"])
    sig = {f"bm25_{f}": (full if f == "full" else bm25_scores(Cd, f, **params[f])) for f in FIELDS}
    allsets = [Cd["pr"][c]["act_secs_all"] for c in Cd["ids"]]
    sig["statute_jac"] = set_sim_matrix(Cd, "act_secs_all")
    sig["statute_idf"] = set_sim_matrix(Cd, "act_secs_all", "idf", L.set_idf(allsets))
    sig["acts_jac"] = set_sim_matrix(Cd, "acts")
    sig["actfield_jac"] = set_sim_matrix(Cd, "act_secs")
    sig["popularity"] = popularity_matrix(Cd, False)
    sig["bench_jac"] = bench_matrix(Cd)
    rows = []
    for n, S in sig.items():
        for tname, A in (("none", A0), ("exactq", A1)):
            m, _ = rank_eval(Cd, S, A, tie=None if n.startswith("bm25") else full / (full.max(1, keepdims=True) + 1e-9))
            rows.append((n, tname, m))
            note(f"dev signal {n:14s} temporal={tname:6s} {M.report(m)}")
    (OUT / "dev_signals.json").write_text(json.dumps([[a, b, c] for a, b, c in rows]))


TOPN = 200
LTR_EXCLUDE = ("popularity",)


def get_params():
    f = OUT / "bm25_params.json"
    return json.loads(f.read_text())


def rank_of(S):
    o = np.argsort(-S, 1)
    R = np.empty_like(o)
    rows = np.arange(S.shape[0])[:, None]
    R[rows, o] = np.arange(S.shape[1])[None, :]
    return R


def signal_matrices(split):
    C = ctx(split)
    params = get_params()
    sig = {}
    for f in FIELDS:
        sig[f"bm25_{f}"] = bm25_scores(C, f, **params[f]).astype(np.float32)

    bm = L.BM25(C["toks"]["full"], params["full"]["k1"], params["full"]["b"])
    Spp = bm.score_matrix(C["toks"]["full"], params["full"]["qw"])
    np.fill_diagonal(Spp, -np.inf)
    Rpp = rank_of(Spp)
    qi = [C["idx"][x] for x in C["qids"]]
    sig["rev_rank"] = Rpp[:, qi].T.astype(np.float32)
    Spp[~np.isfinite(Spp)] = np.nan
    hub = np.nanmean(Spp / np.nanmax(Spp, 1, keepdims=True), 0)
    sig["c_hub"] = np.broadcast_to(hub[None, :], sig["rev_rank"].shape).astype(np.float32).copy()
    allsets = [C["pr"][c]["act_secs_all"] for c in C["ids"]]
    sig["statute_jac"] = set_sim_matrix(C, "act_secs_all")
    sig["statute_idf"] = set_sim_matrix(C, "act_secs_all", "idf", L.set_idf(allsets))
    sig["acts_jac"] = set_sim_matrix(C, "acts")
    sig["actfield_jac"] = set_sim_matrix(C, "act_secs")
    sig["loose_sec_jac"] = set_sim_matrix(C, "secs_loose")
    sig["popularity"] = popularity_matrix(C, split == "train")
    sig["bench_jac"] = bench_matrix(C)
    return sig


def date_feats(C):
    pr, ids, qids = C["pr"], C["ids"], C["qids"]

    def yr(r, k="date"):
        return float(r[k][:4]) if r.get(k) else np.nan
    cy = np.array([yr(pr[c]) for c in ids]); qy = np.array([yr(pr[x]) for x in qids])
    cex = np.array([pr[c]["date_src"] in ("header", "cue") for c in ids], np.float32)
    qex = np.array([pr[x]["date_src"] in ("header", "cue") for x in qids], np.float32)
    qmax = np.array([yr(pr[x], "date_max") for x in qids])
    clen = np.log1p([len(C["toks"]["full"][C["idx"][c]]) for c in ids])
    qlen = np.log1p([len(t) for t in C["toks"]["q_full"]])
    chd = np.array([pr[c]["has_header"] for c in ids], np.float32)
    nq, nc = len(qids), len(ids)
    return {"year_diff": (cy[None, :] - qy[:, None]).astype(np.float32),
            "year_diff_qmax": (cy[None, :] - qmax[:, None]).astype(np.float32),
            "c_year": np.broadcast_to(cy[None, :], (nq, nc)).astype(np.float32),
            "q_year": np.broadcast_to(qy[:, None], (nq, nc)).astype(np.float32),
            "c_date_exact": np.broadcast_to(cex[None, :], (nq, nc)).copy(),
            "q_date_exact": np.broadcast_to(qex[:, None], (nq, nc)).copy(),
            "c_len": np.broadcast_to(clen[None, :], (nq, nc)).astype(np.float32),
            "q_len": np.broadcast_to(qlen[:, None], (nq, nc)).astype(np.float32),
            "c_has_header": np.broadcast_to(chd[None, :], (nq, nc)).copy()}


RANKED = ("bm25_full", "bm25_bi", "bm25_tri", "bm25_first", "bm25_act", "bm25_head")


def build_table(split, cand_gen=None):
    C = ctx(split)
    sig = signal_matrices(split)
    sig.update(date_feats(C))
    A = masks(C, True)
    for n in RANKED:
        S = np.where(A, sig[n], -np.inf)
        sig[n + "_rank"] = rank_of(S).astype(np.float32)
        mx = np.max(np.where(A, sig[n], -np.inf), 1, keepdims=True)
        sig[n + "_rel"] = (sig[n] / (mx + 1e-9)).astype(np.float32)
    gens = {"bm25_full": sig["bm25_full"], "bm25_tri": sig["bm25_tri"],
            "rrf_full_bi_tri": None}
    tot = np.zeros_like(sig["bm25_full"])
    for n in ("bm25_full", "bm25_bi", "bm25_tri"):
        tot += 1.0 / (60 + sig[n + "_rank"])
    gens["rrf_full_bi_tri"] = tot
    sig["rrf_full_bi_tri"] = tot.astype(np.float32)
    recall = {}
    if C["q"][C["qids"][0]]["rel"] is not None:
        g = sum(len(v["rel"]) for v in C["q"].values())
        for gn, S in gens.items():
            o = np.argsort(-np.where(A, S, -np.inf), 1)[:, :TOPN]
            recall[gn] = sum(C["ids"][j] in C["q"][qid]["rel"] for i, qid in enumerate(C["qids"]) for j in o[i]) / g
    gen = cand_gen or max(recall, key=recall.get)
    base = np.where(A, gens[gen], -np.inf)
    cand = np.argsort(-base, 1)[:, :TOPN]
    names = sorted(sig)
    rows = np.repeat(np.arange(len(C["qids"])), TOPN)
    cols = cand.ravel()
    X = np.stack([sig[n][rows, cols] for n in names], 1).astype(np.float32)
    y = None
    if C["q"][C["qids"][0]]["rel"] is not None:
        y = np.array([C["ids"][c] in C["q"][C["qids"][r]]["rel"] for r, c in zip(rows, cols)], np.int8)
    return dict(C=C, cand=cand, X=X, names=names, y=y, A=A, sig=sig, recall=recall, gen=gen)


def ranked_from_scores(T, s):
    C = T["C"]
    s = s.reshape(len(C["qids"]), TOPN)
    o = np.argsort(-s, 1)
    return {qid: [C["ids"][T["cand"][i, j]] for j in o[i]] for i, qid in enumerate(C["qids"])}


def rrf(T, names, k=60):
    sig = T["sig"]; A = T["A"]
    tot = np.zeros_like(sig["bm25_full"])
    for n in names:
        S = np.where(A, sig[n] + 1e-6 * sig["bm25_full"] / (sig["bm25_full"].max() + 1e-9), -np.inf)
        tot += 1.0 / (k + rank_of(S))
    return np.where(A, tot, -np.inf)


def hgb(seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
                                          min_samples_leaf=40, l2_regularization=1.0,
                                          early_stopping=False, random_state=seed)


def stage_fuse():
    import pickle
    T = {"train": build_table("train")}
    gen = T["train"]["gen"]
    note(f"fuse: TRAIN recall@{TOPN} by generator {json.dumps({k: round(v, 4) for k, v in T['train']['recall'].items()})} -> using {gen}")
    for s in ("dev", "test"):
        T[s] = build_table(s, gen)
    note(f"fuse: dev recall@{TOPN} by generator {json.dumps({k: round(v, 4) for k, v in T['dev']['recall'].items()})}")
    note("fuse: tables built " + ", ".join(f"{s} {T[s]['X'].shape}" for s in SPLITS))


    keep = [j for j, n in enumerate(T["train"]["names"]) if n not in LTR_EXCLUDE]
    for s in SPLITS:
        T[s]["Xall"], T[s]["names_all"] = T[s]["X"], T[s]["names"]
        T[s]["X"] = T[s]["X"][:, keep]; T[s]["names"] = [T[s]["names"][j] for j in keep]
    for s in ("train", "dev"):
        C = T[s]["C"]; g = sum(len(v["rel"]) for v in C["q"].values())
        note(f"fuse: {s} recall@{TOPN} of temporal-filtered bm25_full candidates: {T[s]['y'].sum() / g:.4f}")

    pool_sigs = ["bm25_full", "bm25_bi", "bm25_tri", "bm25_first", "bm25_act", "bm25_head",
                 "actfield_jac", "statute_jac", "statute_idf", "bench_jac"]
    import itertools
    res = []
    for r in (2, 3, 4):
        for combo in itertools.combinations(pool_sigs, r):
            if not any(c in ("bm25_full", "bm25_bi", "bm25_tri") for c in combo):
                continue
            m, _ = rank_eval(T["train"]["C"], rrf(T["train"], combo), T["train"]["A"])
            res.append((m["f1"], combo))
    res.sort(reverse=True)
    for f1, combo in res[:3]:
        m, _ = rank_eval(T["dev"]["C"], rrf(T["dev"], combo), T["dev"]["A"])
        note(f"RRF {'+'.join(combo)}: train F1 {f1:.4f} | dev {M.report(m)}")
    for combo in (("bm25_full", "bm25_bi", "bm25_tri"), ("bm25_bi", "actfield_jac"), ("bm25_full", "bm25_bi", "bm25_tri", "actfield_jac")):
        m, _ = rank_eval(T["dev"]["C"], rrf(T["dev"], combo), T["dev"]["A"])
        mt, _ = rank_eval(T["train"]["C"], rrf(T["train"], combo), T["train"]["A"])
        note(f"RRF(fixed) {'+'.join(combo)}: train F1 {mt['f1']:.4f} | dev {M.report(m)}")

    Xt, yt, names = T["train"]["X"], T["train"]["y"], T["train"]["names"]
    model = hgb().fit(Xt, yt)
    pickle.dump({"model": model, "names": names}, open(OUT / "ltr_hgb.pkl", "wb"))
    sd = model.predict_proba(T["dev"]["X"])[:, 1]
    md = M.score(ranked_from_scores(T["dev"], sd), T["dev"]["C"]["q"])
    note(f"LTR HGB (train -> dev, {len(names)} feats): dev {M.report(md)}")

    nq = len(T["train"]["C"]["qids"]); rng = np.random.RandomState(0); fold = rng.randint(0, 5, nq)
    oof = np.zeros(len(yt))
    rf = np.repeat(fold, TOPN)
    for k in range(5):
        mk = hgb(k).fit(Xt[rf != k], yt[rf != k])
        oof[rf == k] = mk.predict_proba(Xt[rf == k])[:, 1]
    mo = M.score(ranked_from_scores(T["train"], oof), T["train"]["C"]["q"])
    note(f"LTR HGB train 5-fold OOF: {M.report(mo)}")
    st = model.predict_proba(T["test"]["X"])[:, 1]

    imp = {}
    Xd = T["dev"]["X"]
    rng = np.random.RandomState(1)
    for j, n in enumerate(names):
        drops = []
        for rep in range(3):
            Xp = Xd.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(md["f1"] - M.score(ranked_from_scores(T["dev"], model.predict_proba(Xp)[:, 1]), T["dev"]["C"]["q"])["f1"])
        imp[n] = float(np.mean(drops))

    note("LTR permutation importance (dev F1 drop): " + ", ".join(f"{n}={v:+.4f}" for n, v in sorted(imp.items(), key=lambda x: -x[1])))
    (OUT / "ltr_importance.json").write_text(json.dumps(imp, indent=1))

    groups = {"no_date": [n for n in names if not n.startswith(("year", "c_year", "q_year", "c_date", "q_date"))],
              "no_statute": [n for n in names if not any(k in n for k in ("statute", "acts_jac", "actfield", "loose", "bm25_act"))],
              "no_c_hub": [n for n in names if n != "c_hub"],
              "no_bench": [n for n in names if n != "bench_jac"],
              "no_shingles": [n for n in names if "_bi" not in n and "_tri" not in n],
              "no_rev": [n for n in names if n not in ("rev_rank", "c_hub")]}
    na = T["train"]["names_all"]
    mg = hgb().fit(T["train"]["Xall"], yt)
    m = M.score(ranked_from_scores(T["dev"], mg.predict_proba(T["dev"]["Xall"])[:, 1]), T["dev"]["C"]["q"])
    note(f"LTR ablation WITH popularity ({len(na)} feats): dev {M.report(m)}")
    for g, keep in groups.items():
        ix = [names.index(n) for n in keep]
        mg = hgb().fit(Xt[:, ix], yt)
        m = M.score(ranked_from_scores(T["dev"], mg.predict_proba(T["dev"]["X"][:, ix])[:, 1]), T["dev"]["C"]["q"])
        note(f"LTR ablation {g}: dev {M.report(m)}")

    scores = {"train": oof, "dev": sd, "test": st}
    for s in SPLITS:
        C = T[s]["C"]; X = T[s]["Xall"]; out = {}
        sc = scores[s].reshape(len(C["qids"]), TOPN)
        for i, qid in enumerate(C["qids"]):
            d = {}
            for j in range(TOPN):
                row = X[i * TOPN + j]
                f = {n: (None if not np.isfinite(v) else round(float(v), 5)) for n, v in zip(T[s]["names_all"], row)}
                f["ltr"] = round(float(sc[i, j]), 6)
                d[C["ids"][T[s]["cand"][i, j]]] = f
            out[qid] = d
        (OUT / f"scores_{s}.json").write_text(json.dumps(out))
    note("wrote scores_{train,dev,test}.json (train ltr = 5-fold OOF)")
    C = T["train"]["C"]; S = np.where(T["train"]["A"], T["train"]["sig"]["bm25_full"], -np.inf)
    hn = {}
    for i, qid in enumerate(C["qids"]):
        o = np.argsort(-S[i])
        hn[qid] = [C["ids"][j] for j in o if np.isfinite(S[i, j]) and C["ids"][j] not in C["q"][qid]["rel"]][:50]
    (OUT / "hardneg_train.json").write_text(json.dumps(hn))
    S = np.where(T["train"]["A"], T["train"]["sig"]["bm25_tri"], -np.inf); hn = {}
    for i, qid in enumerate(C["qids"]):
        o = np.argsort(-S[i])
        hn[qid] = [C["ids"][j] for j in o if np.isfinite(S[i, j]) and C["ids"][j] not in C["q"][qid]["rel"]][:50]
    (OUT / "hardneg_train_tri.json").write_text(json.dumps(hn))
    note("wrote hardneg_train.json (bm25_full) + hardneg_train_tri.json (bm25_tri, stronger), exactq filter, 50/query")


if __name__ == "__main__":
    for st in sys.argv[1:]:
        globals()[f"stage_{st}"]()

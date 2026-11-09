from __future__ import annotations

import json
import pickle
import re
import time
from collections import Counter
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from summ import data as D
from summ import extractive as E

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "context"
TOK_PATH = str(Path(__file__).resolve().parents[1] / "artifacts" / "tokenizer_16k" / "tokenizer.json")
LR_PATH = OUT / "lr.pkl"


@lru_cache(1)
def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(TOK_PATH)


def n_tokens(text: str) -> int:
    return len(tokenizer().encode(text, add_special_tokens=False).ids)


_NUMWORD = ("of|in|on|at|to|from|by|for|and|or|under|section|sections|article|articles|rule|"
            "rules|order|clause|no|nos|dated|year|years|than|about|rs|s|u|month|months|days|"
            "day|para|paras|paragraph|item|entry|schedule|act|list|page|pages|exhibit|ex|"
            "between|over|upto|till|until|since|before|after|march|april|may|june|july|"
            "august|september|october|november|december|january|february|is|was|were|are|"
            "only|all|some|first|last|every|approximately|nearly|about|per|cent|plus|with")
def clean(text: str) -> str:
    out, last = [], 0
    for m in re.finditer(r"(?<=[a-z,;]) \d{1,4} (?=[a-z])", text):
        prev = re.findall(r"(\w+)\W*$", text[:m.start()])
        if prev and prev[0].lower() in _NUMSET:
            continue
        out.append(text[last:m.start()]); last = m.end() - 1
    out.append(text[last:])
    return re.sub(r"\s+", " ", "".join(out)).strip()


_NUMSET = set(_NUMWORD.split("|"))


def units(doc: str):
    sents = D.sentences(doc)
    body = E.strip_header(doc)
    cut = len(doc) - len(body) if body is not doc else 0
    hdr, pos = [], 0
    for s in sents:
        j = doc.find(s[:40], pos)
        if j < 0:
            j = pos
        hdr.append(j < cut)
        pos = j + 1
    junk = [E.is_junk(clean(s)) for s in sents]
    return sents, np.array(hdr, bool), np.array(junk, bool)


def fit_tokens(sents: list[str], idx, budget: int) -> str:
    out, used = [], 0
    for i in sorted(set(idx)):
        s = clean(sents[i])
        t = n_tokens(" " + s) if out else n_tokens(s)
        if used + t <= budget:
            out.append(s); used += t
            continue
        rem = budget - used
        if rem >= 12:
            w = s.split()
            lo, hi = 0, len(w)
            while lo < hi:
                m = (lo + hi + 1) // 2
                if n_tokens(" " + " ".join(w[:m])) <= rem:
                    lo = m
                else:
                    hi = m - 1
            if lo:
                out.append(" ".join(w[:lo]))
        break
    text = " ".join(out)
    while n_tokens(text) > budget:
        text = " ".join(text.split()[:-1])
    return text


def _greedy_fill(sents, order, budget, chosen=None, used=0, slack=1.15):
    chosen = list(chosen or [])
    for i in order:
        if used >= budget:
            break
        if i in chosen:
            continue
        chosen.append(i); used += n_tokens(sents[i])
    return chosen, used


CUES = {
    "facts": r"\b(?:fact|facts|briefly stated|brief facts|the case of the prosecution|"
             r"prosecution case|it is alleged|alleged)\b",
    "parties": r"\b(?:appellant|appellants|respondent|respondents|petitioner|petitioners|"
               r"accused|complainant|plaintiff|defendant)\b",
    "court": r"\b(?:high court|trial court|sessions judge|tribunal|district judge|magistrate|"
             r"division bench|single judge)\b",
    "proc": r"\b(?:filed|instituted|convicted|acquitted|sentenced|dismissed|allowed|decreed|"
            r"set aside|quashed|preferred|granted|rejected)\b",
    "issue": r"\b(?:question|questions|issue|issues|point for|contention|contended|urged|"
             r"argued|submitted)\b",
    "hold": r"\b(?:we hold|we are of the opinion|in our opinion|we agree|we think|held that|"
            r"accordingly|for these reasons|in the result|therefore)\b",
    "law": r"\b(?:section|sections|article|act|rule|rules|order|clause|sub-section)\b",
    "time": r"\b(?:19|20)\d\d\b|\bon \d{1,2}\b",
    "cite": r"\b(?:S\.?C\.?R|A\.?I\.?R|SCC|v\.|vs\.)",
    "quote": r"[\"“]",
}
_CUES = {k: re.compile(v, re.I) for k, v in CUES.items()}
NB = 10


def features(doc: str):
    sents, hdr, junk = units(doc)
    n = len(sents)
    if n == 0:
        return sents, hdr, junk, np.zeros((0, 1), np.float32)
    rel = np.arange(n) / max(1, n - 1)

    first_body = int(np.argmax(~hdr)) if (~hdr).any() else 0
    body_idx = np.arange(n) - first_body
    L = np.array([len(s.split()) for s in sents], float)
    cen = E.centrality(sents, "centroid")
    cen_r = E._rank01(cen)
    F = [rel, rel ** 2, np.log1p(np.maximum(body_idx, 0)), (body_idx < 0).astype(float),
         (body_idx == 0).astype(float), (body_idx < 5) & (body_idx >= 0),
         np.log1p(L), L / 100, cen, cen_r, hdr.astype(float), junk.astype(float),
         np.full(n, np.log(n))]
    bins = np.minimum(NB - 1, (rel * NB).astype(int))
    F += [(bins == b).astype(float) for b in range(NB)]
    for k, rx in _CUES.items():
        F.append(np.array([min(3, len(rx.findall(s))) for s in sents], float))
    digits = np.array([sum(c.isdigit() for c in s) / max(1, len(s)) for s in sents])
    caps = np.array([sum(w[:1].isupper() for w in s.split()) / max(1, len(s.split()))
                     for s in sents])
    F += [digits, caps]
    X = np.stack([np.asarray(f, float) for f in F], 1).astype(np.float32)
    return sents, hdr, junk, X


def _feat_lab(args):
    rec, lab = args
    sents, hdr, junk, X = features(rec["document"])
    assert len(sents) == len(lab["n_words"]), rec["id"]
    y = np.zeros(len(sents), np.int8); y[lab["open"]] = 1
    return X, y


def fit_lr(n_docs=3000, procs=12, model="lr"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from summ import ext_labels as XL
    tr, _ = D.split()
    L = XL.load()
    recs = tr[:n_docs]
    with Pool(procs) as p:
        parts = p.map(_feat_lab, [(r, L[r["id"]]) for r in recs], chunksize=16)
    X = np.concatenate([a for a, _ in parts]); y = np.concatenate([b for _, b in parts])
    if model == "lr":
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000,
                                                                class_weight="balanced"))
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                             max_leaf_nodes=31, random_state=0)
    clf.fit(X, y)
    LR_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path(model), "wb") as fh:
        pickle.dump(clf, fh)
    return clf, X.shape, float(y.mean())


_CLF = {}


def model_path(model="lr"):
    return LR_PATH if model == "lr" else OUT / f"{model}.pkl"


def lr_model(model="lr"):
    if model not in _CLF:
        with open(model_path(model), "rb") as fh:
            _CLF[model] = pickle.load(fh)
    return _CLF[model]


def lr_scores(doc, model="lr"):
    sents, hdr, junk, X = features(doc)
    if not sents:
        return sents, hdr, junk, np.zeros(0)
    p = lr_model(model).predict_proba(X)[:, 1]
    p = np.where(hdr | junk, p - 1.0, p)
    return sents, hdr, junk, p


def build_input(document: str, budget_tokens: int = 560, strategy: str = "lr_lead",
                head_frac: float = 0.5, k: int = 4, mmr: float = 0.0,
                model: str = "lr") -> str:
    if strategy == "lead":
        ss = E.doc_sentences(document, page_clean=False)
        return fit_tokens(ss, range(len(ss)), budget_tokens) or \
            fit_tokens([document], [0], budget_tokens)

    if strategy == "strat":
        ss = E.doc_sentences(document, page_clean=False)
        if not ss:
            return fit_tokens([document], [0], budget_tokens)
        head_b = int(head_frac * budget_tokens)
        chosen, used = _greedy_fill(ss, range(len(ss)), head_b)
        rest = list(range(max(chosen) + 1 if chosen else 0, len(ss)))
        if rest:
            segs = np.array_split(np.array(rest), k)
            share = (budget_tokens - used) / k
            for seg in segs:
                c, u = _greedy_fill(ss, seg.tolist(), share)
                chosen += c; used += u
        return fit_tokens(ss, chosen, budget_tokens)

    sents, hdr, junk, p = lr_scores(document, model)
    if not sents:
        return fit_tokens([document], [0], budget_tokens)
    body = [i for i in range(len(sents)) if not (hdr[i] or junk[i])] or list(range(len(sents)))

    if strategy == "lr":
        if mmr:
            X = E._tfidf(sents); order, cand = [], list(np.argsort(-p))
            used = 0
            while cand and used < budget_tokens:
                if order:
                    sim = (X[cand] @ X[order].T).max(1)
                    j = int(np.argmax(p[cand] - mmr * sim))
                else:
                    j = 0
                i = cand.pop(j); order.append(i); used += n_tokens(sents[i])
            return fit_tokens(sents, order, budget_tokens)
        chosen, _ = _greedy_fill(sents, list(np.argsort(-p)), budget_tokens)
        return fit_tokens(sents, chosen, budget_tokens)

    if strategy == "lr_lead":
        chosen, used = _greedy_fill(sents, body, int(head_frac * budget_tokens))
        chosen, _ = _greedy_fill(sents, list(np.argsort(-p)), budget_tokens, chosen, used)
        return fit_tokens(sents, chosen, budget_tokens)

    if strategy == "chunk":
        segs = np.array_split(np.array(body), k)
        w = np.array([2.0 ** -i for i in range(k)]) if head_frac > 0 else np.ones(k)
        w = w / w.sum()
        chosen = []
        for seg, wi in zip(segs, w):
            seg = seg.tolist()
            order = sorted(seg, key=lambda i: -p[i])
            c, _ = _greedy_fill(sents, order, wi * budget_tokens)
            chosen += c
        return fit_tokens(sents, chosen, budget_tokens)

    raise ValueError(strategy)


def _toks(s):
    return re.findall(r"\w+", s.lower())


def r1_recall(inp, ref):
    R = Counter(_toks(ref)); P = Counter(_toks(inp))
    return sum((P & R).values()) / max(1, sum(R.values()))


def parse(cfg):
    p = cfg.split(":")
    parts = p[0].split("+")
    kw = {"strategy": parts[0], "budget_tokens": int(p[1])}
    for q in parts[1:]:
        if q.startswith("mmr"):
            kw["mmr"] = float(q[3:])
        else:
            kw["model"] = q
    if len(p) > 2:
        kw["head_frac"] = float(p[2])
    if len(p) > 3:
        kw["k"] = int(p[3])
    return kw


def _build(args):
    cfg, doc = args
    return build_input(doc, **parse(cfg))


def generate(cfg, recs, pool):
    return pool.map(_build, [(cfg, r["document"]) for r in recs], chunksize=8)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit", "eval", "dump"])
    ap.add_argument("--configs", default="")
    ap.add_argument("--bert", action="store_true")
    ap.add_argument("--n_docs", type=int, default=3000)
    ap.add_argument("--model", default="lr")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "fit":
        t = time.time()
        clf, shape, pos = fit_lr(a.n_docs, model=a.model)
        print(f"[fit] {a.model} X {shape} pos-rate {pos:.3f} in {time.time()-t:.0f}s")
        return
    tr, va = D.split()
    pool = Pool(12)
    if a.stage == "eval":
        refs = [r["summary"] for r in va]
        open_ = [" ".join(r.split()[:400]) for r in refs]
        for cfg in a.configs.split(","):
            t = time.time()
            inp = generate(cfg, va, pool)
            m = {"config": cfg,
                 "r1rec_open400": float(np.mean([r1_recall(i, o) for i, o in zip(inp, open_)])),
                 "r1rec_full": float(np.mean([r1_recall(i, r) for i, r in zip(inp, refs)])),
                 "avg_tokens": float(np.mean(pool.map(n_tokens, inp, chunksize=16))),
                 "max_tokens": int(max(pool.map(n_tokens, inp, chunksize=16))),
                 "avg_words": float(np.mean([len(i.split()) for i in inp]))}
            if a.bert:
                from summ import metrics as M
                m["bertscore_f1"] = float(M.bertscore(inp, refs).mean())
            m["secs"] = round(time.time() - t)
            print(json.dumps(m), flush=True)
            with open(OUT / "results.jsonl", "a") as f:
                f.write(json.dumps(m) + "\n")
        return
    te = D.load("test")
    for cfg in a.configs.split(","):
        name = cfg.replace(":", "_")
        for split, recs in (("val", va), ("test", te)):
            inp = generate(cfg, recs, pool)
            (OUT / f"{name}_{split}.json").write_text(
                json.dumps({r["id"]: x for r, x in zip(recs, inp)}))
        print("wrote", name, "val/test")


if __name__ == "__main__":
    main()

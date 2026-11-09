from __future__ import annotations

import re
from collections import Counter

import numpy as np

from summ import data as D


_JUDGE = re.compile(
    r"(?:delivered by|Judgments? were delivered)[^.]{0,80}?\bJ\s*\.\s*:?"
    r"|(?:\b[A-Z][A-Z.\s]{2,40}?,\s*(?:C\.\s*)?J\s*\.\s*:?)", re.S)
_HDR = re.compile(
    r"for (?:the )?(?:appellants?|respondents?|petitioners?|interveners?|state|union|"
    r"accused|complainant|applicants?|opposite part)[^.]{0,120}\.\s"
    r"|(?:Appeals?|Petitions?) (?:by|under|from|on)[^.]{0,200}\.\s"
    r"|From the (?:Judgment|Order|Decree)[^.]{0,250}?\b(?:of|in)\s[^.]{0,150}\.\s"
    r"|\b(?:19|20)\d\d\.\s+[A-Z][a-z]+\.?\s+\d{1,2}\.\s", re.I | re.S)
_DATE_LINE = re.compile(r"\b(?:19|20)\d\d\.\s+[A-Z][a-z]+\.?\s+\d{1,2}\.\s")


def strip_header(doc: str, window: int = 3500) -> str:
    lim = min(window, int(0.4 * len(doc)))
    head = doc[:lim]
    m = None
    for m in _JUDGE.finditer(head):
        break
    if m is not None:
        cut = m.end()
    else:
        ends = [x.end() for x in _HDR.finditer(head)]
        cut = max(ends) if ends else 0
    body = doc[cut:].lstrip(" .:;,-")
    return body if len(body.split()) > 50 else doc


_PAGE = re.compile(r"(?<=[a-z,;]) \d{1,4} (?=[a-z])")
_CITE = re.compile(r"\b(?:\d{4}\]?\s*)?(?:S\.?C\.?R|A\.?I\.?R|SCC|All|Bom|Cal|Mad)\b")


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", _PAGE.sub(" ", text)).strip()


def is_junk(s: str, min_words: int = 8) -> bool:
    w = s.split()
    if len(w) < min_words:
        return True
    alpha = sum(t.isalpha() for t in w) / len(w)
    if alpha < 0.5:
        return True
    if len(_CITE.findall(s)) >= 2 and len(w) < 40:
        return True
    return False


def doc_sentences(doc: str, strip=True, filt=True, page_clean=True) -> list[str]:
    body = strip_header(doc) if strip else doc
    if page_clean:
        body = clean(body)
    ss = D.sentences(body)
    if filt:
        kept = [s for s in ss if not is_junk(s)]
        ss = kept if kept else ss
    return ss


def take(sents: list[str], idx, budget: int) -> str:
    out, n = [], 0
    for i in sorted(idx):
        w = sents[i].split()
        if n + len(w) > budget:
            rem = budget - n
            if rem >= 10:
                out.append(" ".join(w[:rem]))
            break
        out.append(sents[i]); n += len(w)
    return " ".join(out)


def lead(doc: str, budget: int = 400, **kw) -> str:
    ss = doc_sentences(doc, **kw)
    return take(ss, range(len(ss)), budget)


def _tok(s):
    return re.findall(r"\w+", s.lower())


def oracle_idx(sents: list[str], ref: str, budget: int) -> list[int]:
    R = Counter(_tok(ref)); nR = sum(R.values())
    T = [Counter(_tok(s)) for s in sents]
    L = [len(s.split()) for s in sents]
    cur, chosen, words, best = Counter(), [], 0, 0.0
    while True:
        cand = None
        for i, t in enumerate(T):
            if i in chosen or words + L[i] > budget + 30:
                continue
            c = cur + t
            ov = sum((c & R).values()); np_ = sum(c.values())
            if not ov:
                continue
            p, r = ov / np_, ov / nR
            f = 2 * p * r / (p + r)
            if f > best + 1e-6 and (cand is None or f > cand[0]):
                cand = (f, i)
        if cand is None:
            break
        best, i = cand
        chosen.append(i); cur += T[i]; words += L[i]
        if words >= budget:
            break
    return sorted(chosen)


def oracle(doc: str, ref: str, budget: int) -> str:
    ss = doc_sentences(doc)
    return take(ss, oracle_idx(ss, ref, budget), budget)


def fit_position_prior(records, bins: int = 20, budget: int = 650) -> np.ndarray:
    sel, tot = np.zeros(bins), np.zeros(bins)
    for r in records:
        ss = doc_sentences(r["document"])
        if not ss:
            continue
        chosen = set(oracle_idx(ss, r["summary"], budget))
        for i in range(len(ss)):
            b = min(bins - 1, int(bins * i / len(ss)))
            tot[b] += 1; sel[b] += i in chosen
    return sel / np.maximum(tot, 1)


_STOP = set("""a an the of to in and or for on by with as at from that this which is was be
been are were it its he she they his her their has have had not no but if so such any all
said shall may would could should will there these those than then also into upon under
other who whom whose what when where""".split())


def _tfidf(sents):
    toks = [[w for w in _tok(s) if w not in _STOP and not w.isdigit() and len(w) > 2]
            for s in sents]
    vocab = {}
    for t in toks:
        for w in t:
            vocab.setdefault(w, len(vocab))
    X = np.zeros((len(sents), max(1, len(vocab))))
    for i, t in enumerate(toks):
        for w, c in Counter(t).items():
            X[i, vocab[w]] = 1 + np.log(c)
    df = (X > 0).sum(0)
    X *= np.log((1 + len(sents)) / (1 + df)) + 1
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X


def centrality(sents, method="centroid"):
    X = _tfidf(sents)
    if method == "centroid":
        c = X.mean(0); c /= np.linalg.norm(c) + 1e-9
        return X @ c
    S = X @ X.T
    np.fill_diagonal(S, 0)
    S = S / (S.sum(1, keepdims=True) + 1e-9)
    n = len(sents); p = np.full(n, 1 / n)
    for _ in range(50):
        p = 0.15 / n + 0.85 * S.T @ p
    return p


def _rank01(x):
    if len(x) < 2:
        return np.ones_like(x, dtype=float)
    return np.argsort(np.argsort(x)) / (len(x) - 1)


def scores(sents, method="centroid", prior=None, w_prior=0.5):
    n = len(sents)
    s = _rank01(centrality(sents, method)) if method != "none" else np.zeros(n)
    if prior is not None:
        b = np.minimum(len(prior) - 1, (len(prior) * np.arange(n) / n).astype(int))
        s = (1 - w_prior) * s + w_prior * _rank01(prior[b] + 1e-6 * -np.arange(n))
    return s


def select(doc, budget, method="centroid", prior=None, w_prior=0.5, lead_words=0,
           mmr=0.0):
    ss = doc_sentences(doc)
    if not ss:
        return doc[:budget * 8]
    sc = scores(ss, method, prior, w_prior)
    chosen, words = [], 0
    if lead_words:
        for i, s in enumerate(ss):
            if words >= lead_words:
                break
            chosen.append(i); words += len(s.split())
    X = _tfidf(ss) if mmr else None
    order = list(np.argsort(-sc))
    while words < budget and order:
        if mmr and chosen:
            sim = (X[order] @ X[chosen].T).max(1)
            k = int(np.argmax(sc[order] - mmr * sim))
        else:
            k = 0
        i = order.pop(k)
        if i in chosen:
            continue
        chosen.append(i); words += len(ss[i].split())
    return take(ss, chosen, budget)

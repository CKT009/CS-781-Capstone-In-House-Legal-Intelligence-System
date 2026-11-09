from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import scipy.sparse as sp

STOP = set("""a an the of and or in on at to for by with from as is was were be been being are this that these
those it its he she his her they their them we our you your i me my which who whom whose what when where why how
not no nor but if then than so such there here also any all each other some may might shall should will would can
could must do does did done has have had having into upon under over about above below after before between
during against said same further thereof therein thereto hereby herein whereas viz ie eg etc mr ms shri smt
only very more most own one two three per sub vs v""".split())
RE_TOK = re.compile(r"<[A-Z]+>|[a-z][a-z0-9]+|\d+[a-z]?")


def tok(text: str, max_words: int | None = None) -> list[str]:
    toks = [t for t in RE_TOK.findall(text.lower()) if t[0] != "<" and t not in STOP and len(t) > 1]
    return toks[:max_words] if max_words else toks


def first_words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


class BM25:

    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75, vocab: dict | None = None):
        self.vocab = vocab if vocab is not None else {}
        rows, cols, vals = [], [], []
        for i, d in enumerate(docs):
            for t, c in Counter(d).items():
                j = self.vocab.get(t)
                if j is None:
                    if vocab is not None:
                        continue
                    j = self.vocab[t] = len(self.vocab)
                rows.append(i); cols.append(j); vals.append(c)
        self.tf = sp.csr_matrix((vals, (rows, cols)), shape=(len(docs), max(len(self.vocab), 1)), dtype=np.float32)
        self.dl = np.asarray(self.tf.sum(1)).ravel()
        df = np.bincount(self.tf.indices, minlength=self.tf.shape[1])
        N = len(docs)
        self.idf = np.log1p((N - df + 0.5) / (df + 0.5)).astype(np.float32)
        self.set_params(k1, b)

    def set_params(self, k1, b):
        self.k1, self.b = k1, b
        avg = max(self.dl.mean(), 1e-9)
        W = self.tf.tocoo(copy=True)
        norm = k1 * (1 - b + b * self.dl[W.row] / avg)
        W.data = (W.data * (k1 + 1) / (W.data + norm)) * self.idf[W.col]
        self.W = W.tocsr()

    def qmatrix(self, queries: list[list[str]], qweight: str = "log") -> sp.csr_matrix:
        rows, cols, vals = [], [], []
        for i, d in enumerate(queries):
            for t, c in Counter(d).items():
                j = self.vocab.get(t)
                if j is not None:
                    rows.append(i); cols.append(j)
                    vals.append(1.0 if qweight == "bin" else (1 + math.log(c) if qweight == "log" else c))
        return sp.csr_matrix((vals, (rows, cols)), shape=(len(queries), self.tf.shape[1]), dtype=np.float32)

    def score_matrix(self, queries, qweight="log") -> np.ndarray:
        return np.asarray((self.qmatrix(queries, qweight) @ self.W.T).todense())


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a and b else 0.0


def idf_overlap(a, b, idf: dict) -> float:
    a, b = set(a), set(b)
    if not a or not b:
        return 0.0
    tot = sum(idf.get(x, 0.0) for x in a)
    return sum(idf.get(x, 0.0) for x in a & b) / max(tot, 1e-9)


def set_idf(sets: list) -> dict:
    df = Counter(x for s in sets for x in set(s))
    N = len(sets)
    return {x: math.log1p((N - c + 0.5) / (c + 0.5)) for x, c in df.items()}


def popularity(q_train: dict) -> Counter:
    return Counter(c for v in q_train.values() for c in v["rel"])

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

LABEL_FILE = "Train/label_vocab.json"
DOMAINS = ("CL", "IT")
IGNORE = -100


def load_labels(data_dir) -> list[str]:
    p = Path(data_dir) / LABEL_FILE
    if not p.exists():
        raise FileNotFoundError(f"{p} not found -- run `python fetch.py`")
    v = json.loads(p.read_text())
    if isinstance(v, dict):
        return [k for k, _ in sorted(v.items(), key=lambda kv: kv[1])]
    return list(v)


@dataclass
class Doc:
    id: str
    domain: str
    sentences: list[str]
    labels: list[str] | None = None

    def __post_init__(self):
        if self.labels is not None and len(self.labels) != len(self.sentences):
            raise ValueError(
                f"{self.id}: {len(self.sentences)} sentences but {len(self.labels)} "
                f"labels. The phase guarantees len(text) == len(labels); a mismatch means "
                f"the file is not what this loader expects.")

    def __len__(self):
        return len(self.sentences)


def load_split(data_dir, domain: str, split: str) -> list[Doc]:
    if domain not in DOMAINS:
        raise ValueError(f"domain must be one of {DOMAINS}, got {domain!r}")
    sub = "Test" if split == "test" else "Train"
    p = Path(data_dir) / sub / f"{domain}_{split}.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found -- run `python fetch.py`")
    raw = json.loads(p.read_text())

    if split == "test":
        if not isinstance(raw, dict):
            raise ValueError(f"{p}: expected a dict of doc_id -> sentences")
        return [Doc(id=k, domain=domain, sentences=v) for k, v in raw.items()]

    if not isinstance(raw, list):
        raise ValueError(f"{p}: expected a list of records")
    return [Doc(id=r["id"], domain=domain, sentences=r["text"], labels=r["labels"])
            for r in raw]


def load_all(data_dir, split: str, domains=DOMAINS) -> list[Doc]:
    return [d for dom in domains for d in load_split(data_dir, dom, split)]


def check(data_dir, verbose=True) -> dict:
    vocab = load_labels(data_dir)
    vset, report = set(vocab), {"labels": vocab, "splits": {}}
    for dom in DOMAINS:
        for split in ("train", "dev", "test"):
            docs = load_split(data_dir, dom, split)
            n_sent = sum(len(d) for d in docs)
            bad = sorted({l for d in docs if d.labels for l in d.labels} - vset)
            if bad:
                raise ValueError(f"{dom}/{split}: labels outside label_vocab.json: {bad}")
            report["splits"][f"{dom}_{split}"] = {"docs": len(docs), "sentences": n_sent}
            if verbose:
                print(f"[check] {dom}_{split:5s} {len(docs):>3d} docs, {n_sent:>6d} sentences"
                      + ("" if split == "test" else "  labels ok"))
    tr = {d.id for d in load_all(data_dir, "train")}
    dv = {d.id for d in load_all(data_dir, "dev")}
    te = {d.id for d in load_all(data_dir, "test")}
    for a, b, na, nb in ((tr, dv, "train", "dev"), (tr, te, "train", "test"),
                         (dv, te, "dev", "test")):
        if a & b:
            raise ValueError(f"*** {na} and {nb} share {len(a & b)} document ids: "
                             f"{sorted(a & b)[:3]} -- the split is contaminated")
    if verbose:
        print(f"[check] train/dev/test document ids are disjoint "
              f"({len(tr)}/{len(dv)}/{len(te)})")
    return report


@dataclass
class EncodedDoc:
    doc_id: str
    domain: str
    ids: list[list[int]]
    mask: list[list[int]]
    labels: list[int] | None
    n_sent: int = field(init=False)

    def __post_init__(self):
        self.n_sent = len(self.ids)


def encode_doc(doc: Doc, tok, vocab: list[str], max_sent_len: int = 128) -> EncodedDoc:
    idx = {l: i for i, l in enumerate(vocab)}
    ids, mask = [], []
    for s in doc.sentences:
        e = tok.encode(s).ids[:max_sent_len]
        if not e:
            e = [0]
        pad = max_sent_len - len(e)
        ids.append(e + [0] * pad)
        mask.append([1] * len(e) + [0] * pad)
    labels = None
    if doc.labels is not None:
        labels = [idx[l] for l in doc.labels]
    return EncodedDoc(doc.id, doc.domain, ids, mask, labels)


def windows(n_sent: int, max_doc_sents: int, stride: int) -> list[tuple[int, int]]:
    if n_sent <= max_doc_sents:
        return [(0, n_sent)]
    step = max(1, max_doc_sents - stride)
    out, s = [], 0
    while s < n_sent:
        e = min(s + max_doc_sents, n_sent)
        out.append((s, e))
        if e == n_sent:
            break
        s += step
    return out


def owner(wins: list[tuple[int, int]], i: int) -> int:
    best, best_d = 0, -1
    for w, (s, e) in enumerate(wins):
        if s <= i < e:
            d = min(i - s, e - 1 - i)
            if d > best_d:
                best, best_d = w, d
    return best

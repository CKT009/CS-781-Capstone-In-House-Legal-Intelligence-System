from __future__ import annotations

import json
import zipfile
from pathlib import Path

K = 7


def score(ranked: dict[str, list[str]], q: dict, k: int = K) -> dict:
    hits = sum(len(set(ranked[qid][:k]) & q[qid]["rel"]) for qid in q)
    gold = sum(len(q[qid]["rel"]) for qid in q)
    p, r = hits / (k * len(q)), hits / max(gold, 1)
    return {"f1": 2 * p * r / max(p + r, 1e-12), "p": p, "r": r, "hits": hits}


def report(m: dict) -> str:
    return f"micro-F1@7 {m['f1']:.4f} | P@7 {m['p']:.4f} | R@7 {m['r']:.4f}"


def write_submission(ranked: dict[str, list[str]], q: dict, pool: dict, out_zip) -> Path:
    pred = {}
    for qid in q:
        ids = [c for c in dict.fromkeys(ranked[qid]) if c in pool and c != qid][:K]
        assert len(ids) == K, f"{qid}: only {len(ids)} valid candidates"
        pred[qid] = ids
    out_zip = Path(out_zip); out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("prediction.json", json.dumps(pred))
    return out_zip

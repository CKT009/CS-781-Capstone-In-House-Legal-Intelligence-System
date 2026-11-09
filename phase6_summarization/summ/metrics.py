from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from pathlib import Path

_SCORER = None


def _toks(s):
    return re.findall(r"\w+", s.lower())


def rouge_n(p, r, n=1):
    P, R = _toks(p), _toks(r)
    P = Counter(zip(*[P[i:] for i in range(n)])); R = Counter(zip(*[R[i:] for i in range(n)]))
    ov = sum((P & R).values())
    if not ov:
        return 0.0
    pr, rc = ov / sum(P.values()), ov / sum(R.values())
    return 2 * pr * rc / (pr + rc)


def rouge_l(p, r):
    a, b = _toks(p), _toks(r)
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    l = prev[-1]
    if not l:
        return 0.0
    pr, rc = l / len(a), l / len(b)
    return 2 * pr * rc / (pr + rc)


def bertscore(preds, refs, device=None, batch_size=16):
    global _SCORER
    if _SCORER is None:
        from bert_score import BERTScorer
        _SCORER = BERTScorer(lang="en", batch_size=batch_size, device=device,
                             rescale_with_baseline=False)
    P, R, F = _SCORER.score(preds, refs)
    return F.numpy()


def evaluate(preds: list[str], refs: list[str], bert=True, rougeL=True, device=None) -> dict:
    out = {"rouge1": sum(rouge_n(p, r) for p, r in zip(preds, refs)) / len(refs),
           "pred_words": sum(len(p.split()) for p in preds) / len(preds)}
    if rougeL:
        out["rougeL"] = sum(rouge_l(p, r) for p, r in zip(preds, refs)) / len(refs)
    if bert:
        out["bertscore_f1"] = float(bertscore(preds, refs, device).mean())
    return out


def report(m: dict) -> str:
    return " | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in m.items())


def write_submission(summaries: dict, test_ids: list[str], out_zip: Path) -> Path:
    assert set(summaries) == set(test_ids), "missing or extra test ids"
    rows = [{"id": i, "summary": summaries[i]} for i in test_ids]
    assert all(isinstance(r["summary"], str) and r["summary"].strip() for r in rows)
    out_zip = Path(out_zip); out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("prediction.json", json.dumps(rows))
    return out_zip

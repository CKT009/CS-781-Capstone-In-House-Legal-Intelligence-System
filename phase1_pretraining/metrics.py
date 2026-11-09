import math
from collections import Counter


def _tok(s):
    return s.lower().split()


def _ngrams(toks, n):
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 20))


def rouge_n(pred, ref, n=1):
    p, r = _ngrams(_tok(pred), n), _ngrams(_tok(ref), n)
    ov = sum((p & r).values())
    pr = ov / max(sum(p.values()), 1)
    rc = ov / max(sum(r.values()), 1)
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def rouge_l(pred, ref):
    a, b = _tok(pred), _tok(ref)
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    l = prev[-1]
    pr, rc = l / max(len(a), 1), l / max(len(b), 1)
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def bleu(preds, refs, n=4):
    num, den, plen, rlen = [0] * n, [0] * n, 0, 0
    for p, r in zip(preds, refs):
        pt, rt = _tok(p), _tok(r)
        plen += len(pt); rlen += len(rt)
        for k in range(1, n + 1):
            pg, rg = _ngrams(pt, k), _ngrams(rt, k)
            num[k - 1] += sum((pg & rg).values())
            den[k - 1] += sum(pg.values())
    if min(den) == 0 or min(num) == 0:
        return 0.0
    bp = 1.0 if plen > rlen else math.exp(1 - rlen / max(plen, 1))
    return bp * math.exp(sum(math.log(num[k] / den[k]) for k in range(n)) / n)


def distinct_n(text, n=3):
    g = _ngrams(_tok(text), n)
    return len(g) / max(sum(g.values()), 1)


def repetition_rate(text, n=4):
    g = _ngrams(_tok(text), n)
    return sum(c for c in g.values() if c > 1) / max(sum(g.values()), 1)


_SCORER = None


def bertscore(preds, refs, lang="en", model_type=None, batch_size=64):
    global _SCORER
    try:
        if _SCORER is None:
            from bert_score import BERTScorer
            _SCORER = BERTScorer(lang=lang, model_type=model_type, batch_size=batch_size,
                                 rescale_with_baseline=False)
        P, R, F = _SCORER.score(preds, refs, verbose=False)
        return {"bertscore_p": float(P.mean()), "bertscore_r": float(R.mean()),
                "bertscore_f1": float(F.mean())}
    except Exception as e:
        return {"bertscore_f1": None, "error": repr(e)[:160]}


def evaluate(preds: dict, refs: dict, with_bertscore=True) -> dict:
    ids = [i for i in refs if i in preds]
    P, R = [preds[i] for i in ids], [refs[i] for i in ids]
    n = max(len(ids), 1)
    out = {"n": len(ids),
           "rouge1": sum(rouge_n(p, r, 1) for p, r in zip(P, R)) / n,
           "rouge2": sum(rouge_n(p, r, 2) for p, r in zip(P, R)) / n,
           "rougeL": sum(rouge_l(p, r) for p, r in zip(P, R)) / n,
           "bleu4": bleu(P, R),
           "distinct3": sum(distinct_n(p) for p in P) / n,
           "repetition4": sum(repetition_rate(p) for p in P) / n,
           "mean_pred_words": sum(len(_tok(p)) for p in P) / n,
           "mean_ref_words": sum(len(_tok(r)) for r in R) / n}
    if with_bertscore:
        out.update(bertscore(P, R))
    return out

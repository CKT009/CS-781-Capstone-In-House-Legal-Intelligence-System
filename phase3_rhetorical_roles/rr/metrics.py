from __future__ import annotations

import collections


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def score(pred: dict, gold: dict, vocab: list[str]) -> dict:
    missing = set(gold) - set(pred)
    if missing:
        raise ValueError(f"{len(missing)} documents missing from predictions, "
                         f"e.g. {sorted(missing)[:3]}")
    tp = collections.Counter(); fp = collections.Counter(); fn = collections.Counter()
    support = collections.Counter(); predicted = collections.Counter()
    n = 0
    for d, g in gold.items():
        p = pred[d]
        if len(p) != len(g):
            raise ValueError(f"{d}: {len(p)} predictions for {len(g)} sentences. The "
                             f"submission must have exactly one label per sentence.")
        for a, b in zip(p, g):
            support[b] += 1; predicted[a] += 1; n += 1
            if a == b:
                tp[b] += 1
            else:
                fp[a] += 1; fn[b] += 1

    per = {}
    for l in vocab:
        p, r, f = _prf(tp[l], fp[l], fn[l])
        per[l] = {"p": p, "r": r, "f1": f, "support": support[l], "predicted": predicted[l]}

    seen = [l for l in vocab if support[l] > 0]
    mac = lambda ls, k: sum(per[l][k] for l in ls) / len(ls) if ls else 0.0
    return {
        "macro_f1": mac(vocab, "f1"),
        "macro_p": mac(vocab, "p"),
        "macro_r": mac(vocab, "r"),
        "macro_f1_supported": mac(seen, "f1"),
        "n_supported": len(seen),
        "accuracy": sum(tp.values()) / n if n else 0.0,
        "n_sentences": n,
        "per_label": per,
        "vocab": vocab,
    }


def report(s: dict) -> str:
    out = [f"{'label':24s}{'P':>7s}{'R':>7s}{'F1':>7s}{'gold':>7s}{'pred':>7s}", "-" * 59]
    for l in s["vocab"]:
        d = s["per_label"][l]
        flag = "  <- no gold support" if d["support"] == 0 else ""
        out.append(f"{l:24s}{d['p']:7.3f}{d['r']:7.3f}{d['f1']:7.3f}"
                   f"{d['support']:7d}{d['predicted']:7d}{flag}")
    out += ["-" * 59,
            f"{'MACRO (all 13)':24s}{s['macro_p']:7.3f}{s['macro_r']:7.3f}"
            f"{s['macro_f1']:7.3f}   <- OFFICIAL",
            f"{'macro (supported only)':24s}{'':7s}{'':7s}{s['macro_f1_supported']:7.3f}"
            f"   ({s['n_supported']}/{len(s['vocab'])} labels have gold here)",
            f"{'accuracy':24s}{'':7s}{'':7s}{s['accuracy']:7.3f}"
            f"   (diagnostic only, NOT the metric)",
            f"\n{s['n_sentences']} sentences scored"]
    if s["n_supported"] < len(s["vocab"]):
        out.append(
            f"NOTE: {len(s['vocab']) - s['n_supported']} label(s) have no gold support in "
            f"this split, so the official column is mechanically capped at "
            f"{s['n_supported']}/{len(s['vocab'])} = {s['n_supported']/len(s['vocab']):.3f}. "
            f"Compare models on it anyway -- it is the same cap for all of them -- but do "
            f"not read it as an estimate of the leaderboard score.")
    return "\n".join(out)


def validate_submission(pred: dict, docs, vocab: list[str]) -> list[str]:
    errs, vset = [], set(vocab)
    by_id = {d.id: d for d in docs}
    for miss in sorted(set(by_id) - set(pred)):
        errs.append(f"{miss}: missing from predictions")
    for extra in sorted(set(pred) - set(by_id)):
        errs.append(f"{extra}: not a test document")
    for d, labs in pred.items():
        if d not in by_id:
            continue
        want = len(by_id[d])
        if len(labs) != want:
            errs.append(f"{d}: {len(labs)} labels for {want} sentences")
        bad = sorted(set(labs) - vset)
        if bad:
            errs.append(f"{d}: labels outside label_vocab.json: {bad[:5]}")
    return errs


def score_by_domain(pred: dict, docs, vocab: list[str]) -> dict:
    out = {}
    for dom in sorted({d.domain for d in docs}):
        ids = [d.id for d in docs if d.domain == dom]
        g = {d.id: d.labels for d in docs if d.domain == dom and d.labels is not None}
        if not g:
            continue
        out[dom] = score({k: pred[k] for k in ids if k in pred}, g, vocab)
    g_all = {d.id: d.labels for d in docs if d.labels is not None}
    out["ALL"] = score(pred, g_all, vocab)
    return out


def domain_report(per: dict) -> str:
    doms = [d for d in per if d != "ALL"] + ["ALL"]
    out = [f"{'':24s}" + "".join(f"{d:>10s}" for d in doms), "-" * (24 + 10 * len(doms))]
    vocab = per[doms[0]]["vocab"]
    for l in vocab:
        if all(per[d]["per_label"][l]["support"] == 0 for d in doms):
            continue
        row = f"{l:24s}"
        for d in doms:
            s = per[d]["per_label"][l]
            row += f"{s['f1']:10.3f}" if s["support"] else f"{'--':>10s}"
        out.append(row)
    out.append("-" * (24 + 10 * len(doms)))
    out.append(f"{'MACRO (all 13)':24s}" + "".join(f"{per[d]['macro_f1']:10.3f}" for d in doms))
    out.append(f"{'macro (supported)':24s}" +
               "".join(f"{per[d]['macro_f1_supported']:10.3f}" for d in doms))
    out.append(f"{'labels with gold':24s}" +
               "".join(f"{per[d]['n_supported']:10d}" for d in doms))
    out.append(f"{'sentences':24s}" + "".join(f"{per[d]['n_sentences']:10d}" for d in doms))
    out.append("\n'--' means that label has no gold in that domain, so it contributes 0 to "
               "that\ncolumn's macro. The per-domain caps therefore differ and the columns "
               "are not\ndirectly comparable to each other -- compare models within a column.")
    return "\n".join(out)


def confusion(pred: dict, gold: dict, vocab: list[str], top: int = 12) -> list[tuple]:
    c = collections.Counter()
    for d, g in gold.items():
        for p, t in zip(pred[d], g):
            if p != t:
                c[(t, p)] += 1
    return c.most_common(top)


def confusion_report(pairs, gold: dict) -> str:
    n_err = sum(n for _, n in pairs)
    tot = sum(len(v) for v in gold.values())
    out = [f"{'gold':24s}{'predicted as':24s}{'count':>7s}{'% of all sentences':>20s}",
           "-" * 75]
    for (t, p), n in pairs:
        out.append(f"{t:24s}{p:24s}{n:7d}{100*n/tot:19.1f}%")
    out.append("-" * 75)
    out.append(f"top {len(pairs)} pairs account for {n_err} errors "
               f"({100*n_err/tot:.1f}% of all sentences)")
    return "\n".join(out)

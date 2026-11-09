from __future__ import annotations

import warnings


LABELS: list[str] = [
    "APP",
    "RESP",
    "A.COUNSEL",
    "R.COUNSEL",
    "JUDGE",
    "WIT",
    "AUTH",
    "COURT",
    "STAT",
    "PREC",
    "DATE",
    "CASENO",
]


def _norm_spans(raw, labels_set, unknown_counter):
    out = set()
    for item in raw:
        if len(item) < 3:
            raise ValueError(f"span needs at least [start, end, label], got {item!r}")
        start, end, label = item[0], item[1], item[2]
        start, end = int(start), int(end)
        if end < start:
            raise ValueError(f"span end < start: {item!r}")
        if label not in labels_set:


            unknown_counter[label] = unknown_counter.get(label, 0) + 1
            continue
        out.add((start, end, label))
    return out


def _index(pred, gold, labels, on_extra_docs="raise"):
    if not isinstance(labels, (list, tuple)) or not labels:
        raise ValueError("labels must be a non-empty list")
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate labels in label list: {labels}")
    labels_set = set(labels)

    extra = [d for d in pred if d not in gold]
    if extra:
        msg = (
            f"{len(extra)} doc_id(s) in predictions are absent from gold "
            f"(e.g. {sorted(map(str, extra))[:5]}). This normally means a document-ID "
            f"mismatch, which the official evaluator rejects."
        )
        if on_extra_docs == "raise":
            raise ValueError(msg + " Pass on_extra_docs='warn' to score anyway.")
        elif on_extra_docs == "warn":
            warnings.warn(msg, stacklevel=3)
        else:
            raise ValueError("on_extra_docs must be 'raise' or 'warn'")

    unknown: dict[str, int] = {}
    g = {d: _norm_spans(gold[d], labels_set, unknown) for d in gold}


    p = {d: _norm_spans(pred.get(d, []), labels_set, unknown) for d in gold}
    if unknown:
        warnings.warn(
            f"ignored {sum(unknown.values())} span(s) with labels outside the "
            f"{len(labels)}-label schema: {sorted(unknown)}",
            stacklevel=3,
        )
    return p, g


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1


def strict_prf(pred: dict, gold: dict, labels: list[str] = LABELS,
               *, on_extra_docs: str = "raise") -> dict:
    p_doc, g_doc = _index(pred, gold, labels, on_extra_docs)

    tp_c = {lab: 0 for lab in labels}
    fp_c = {lab: 0 for lab in labels}
    fn_c = {lab: 0 for lab in labels}

    for doc in g_doc:
        gs, ps = g_doc[doc], p_doc[doc]
        for (_, _, lab) in ps & gs:
            tp_c[lab] += 1
        for (_, _, lab) in ps - gs:
            fp_c[lab] += 1
        for (_, _, lab) in gs - ps:
            fn_c[lab] += 1

    per_label = {}
    for lab in labels:
        p, r, f1 = _prf(tp_c[lab], fp_c[lab], fn_c[lab])
        per_label[lab] = {
            "p": p, "r": r, "f1": f1,
            "tp": tp_c[lab], "fp": fp_c[lab], "fn": fn_c[lab],
            "support": tp_c[lab] + fn_c[lab],
        }

    n = len(labels)
    macro_p = sum(per_label[l]["p"] for l in labels) / n
    macro_r = sum(per_label[l]["r"] for l in labels) / n


    macro_f1 = sum(per_label[l]["f1"] for l in labels) / n

    TP = sum(tp_c.values())
    FP = sum(fp_c.values())
    FN = sum(fn_c.values())
    micro_p, micro_r, micro_f1 = _prf(TP, FP, FN)

    present = [l for l in labels if per_label[l]["support"] > 0]
    macro_sup = (sum(per_label[l]["f1"] for l in present) / len(present)) if present else 0.0

    return {
        "macro_f1": macro_f1, "macro_p": macro_p, "macro_r": macro_r,
        "micro_f1": micro_f1, "micro_p": micro_p, "micro_r": micro_r,
        "support": TP + FN,
        "per_label": per_label,

        "tp": TP, "fp": FP, "fn": FN,
        "n_pred": TP + FP,
        "n_labels": n,
        "n_labels_with_support": len(present),
        "macro_f1_labels_with_support": macro_sup,
    }


def _overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def partial_credit_report(pred, gold, labels=LABELS, *, on_extra_docs="raise",
                          max_examples=20) -> dict:
    p_doc, g_doc = _index(pred, gold, labels, on_extra_docs)

    pred_keys = ["exact", "label", "boundary", "boundary_label", "spurious"]
    gold_keys = ["exact", "label", "boundary", "boundary_label", "missed"]
    pred_side = {k: 0 for k in pred_keys}
    gold_side = {k: 0 for k in gold_keys}
    examples: dict[str, list] = {k: [] for k in set(pred_keys) | set(gold_keys)}

    def _note(cat, doc, span, partner=None):
        if len(examples[cat]) < max_examples:
            examples[cat].append((doc, span, partner))

    def _classify(span, others):
        s, e, lab = span
        if span in others:
            return "exact", span
        same_span = [o for o in others if o[0] == s and o[1] == e]
        if same_span:
            return "label", same_span[0]
        ov = [o for o in others if _overlaps(span, o)]
        same_lab = [o for o in ov if o[2] == lab]
        if same_lab:
            return "boundary", same_lab[0]
        if ov:
            return "boundary_label", ov[0]
        return None, None

    for doc in g_doc:
        gs, ps = g_doc[doc], p_doc[doc]
        for sp in sorted(ps):
            cat, partner = _classify(sp, gs)
            cat = cat or "spurious"
            pred_side[cat] += 1
            _note(cat, doc, sp, partner)
        for sp in sorted(gs):
            cat, partner = _classify(sp, ps)
            cat = cat or "missed"
            gold_side[cat] += 1
            if cat == "missed":
                _note("missed", doc, sp, None)

    n_pred = sum(len(p_doc[d]) for d in g_doc)
    n_gold = sum(len(g_doc[d]) for d in g_doc)
    out = {
        "n_pred": n_pred,
        "n_gold": n_gold,
        "pred_side": pred_side,
        "gold_side": gold_side,
        "examples": examples,
    }

    out["exact"] = pred_side["exact"]
    out["boundary"] = pred_side["boundary"]
    out["label"] = pred_side["label"]
    out["boundary_label"] = pred_side["boundary_label"]
    out["spurious"] = pred_side["spurious"]
    out["missed"] = gold_side["missed"]
    return out


def confusion(pred, gold, labels=LABELS, *, on_extra_docs="raise") -> dict:
    p_doc, g_doc = _index(pred, gold, labels, on_extra_docs)
    m = {g: {p: 0 for p in labels} for g in labels}
    for doc in g_doc:
        by_off: dict[tuple, str] = {(s, e): lab for (s, e, lab) in p_doc[doc]}
        for (s, e, glab) in g_doc[doc]:
            plab = by_off.get((s, e))
            if plab is not None:
                m[glab][plab] += 1
    return m


def format_report(res: dict) -> str:
    per = res["per_label"]


    order = sorted(per, key=lambda l: (-per[l]["support"], l))
    w = max([len("label")] + [len(l) for l in order])

    lines = [
        f"{'label':<{w}}  {'P':>6} {'R':>6} {'F1':>6}  {'TP':>5} {'FP':>5} {'FN':>5} {'supp':>5}",
        "-" * (w + 48),
    ]
    for l in order:
        d = per[l]
        lines.append(
            f"{l:<{w}}  {d['p']:>6.3f} {d['r']:>6.3f} {d['f1']:>6.3f}  "
            f"{d['tp']:>5d} {d['fp']:>5d} {d['fn']:>5d} {d['support']:>5d}"
        )
    lines.append("-" * (w + 48))
    lines.append(
        f"{'MACRO':<{w}}  {res['macro_p']:>6.3f} {res['macro_r']:>6.3f} "
        f"{res['macro_f1']:>6.3f}   <- official metric (strict, "
        f"/{res.get('n_labels', len(per))} labels)"
    )
    lines.append(
        f"{'micro':<{w}}  {res['micro_p']:>6.3f} {res['micro_r']:>6.3f} "
        f"{res['micro_f1']:>6.3f}   (diagnostic only, NOT official)"
    )
    lines.append(
        f"gold spans: {res['support']}   predicted spans: {res.get('n_pred', '?')}"
    )
    if "macro_f1_labels_with_support" in res:


        lines.append(
            f"diagnostic: macro-F1 over the {res['n_labels_with_support']} labels with gold"
            f" support = {res['macro_f1_labels_with_support']:.3f}"
        )
        lines.append(
            "            (zero-support labels count as F1=0.0 in the official number above)"
        )
    return "\n".join(lines)


__all__ = ["LABELS", "strict_prf", "partial_credit_report", "confusion", "format_report"]

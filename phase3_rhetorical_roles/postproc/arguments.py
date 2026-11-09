from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

P, R = "ArgumentPetitioner", "ArgumentRespondent"
ARG = (P, R)
DATA = ROOT / "data" / "Phase_3_RR"


_PN = r"(?:petitioners?|appellants?|informants?|applicants?|complainants?|plaintiffs?)"
_RN = r"(?:respondents?|opposite\s+part(?:y|ies)|OPs?\b|OP-?\s?\d|defendants?|non-applicants?)"
_REVN = r"(?:revenue|department|income[- ]tax\s+(?:officer|department)|commissioner)"
_ASSN = r"(?:assessees?)"
_SPEAKER = (r"(?:counsel|advocates?|advocate[- ]general|solicitor[- ]general|attorney[- ]general|"
            r"senior\s+counsel|appearing|on\s+behalf\s+of|behalf\s+of|argued\s+for|for\s+and\s+on\s+behalf)")
_VERB = (r"(?:has\s+|have\s+|had\s+)?(?:also\s+|further\s+|then\s+|next\s+)?"
         r"(?:submit|submitted|submits|contend|contended|contends|argue|argued|argues|urge|urged|"
         r"urges|allege|alleged|alleges|claim|claimed|claims|stated|states|plead|pleaded|pleads|"
         r"assert|asserted|asserts|maintain|maintained|maintains|rel(?:y|ies|ied)|denied|denies|"
         r"seek|seeks|sought|raised?|challenge[sd]?)")
_NOUN_OF = r"(?:case|contention|submission|argument|stand|plea|grievance|say)s?\s+of\s+(?:the\s+)?(?:learned\s+)?"


def _party_rx(noun):
    return re.compile(
        "|".join([
            _SPEAKER + r"[^.;]{0,60}?\b" + noun,
            r"\b" + noun + r"(?:['’]s?)?\s*(?:\([^)]{0,30}\)\s*)?(?:,[^,]{0,60},\s*)?" + _VERB + r"\b",
            r"^\W*(?:\w+\W+){0,4}?" + _NOUN_OF + r"(?:counsel\s+for\s+the\s+)?" + noun,
            r"^\W*according\s+to\s+(?:the\s+)?(?:learned\s+counsel\s+for\s+the\s+)?" + noun,
        ]), re.I)


_RX = {
    "P": _party_rx(_PN),
    "R": _party_rx(_RN),
    "REV": _party_rx(_REVN),
    "ASS": _party_rx(_ASSN),
}
_RX_SW = re.compile(r"^\W*(?:\w+\W+){0,5}?(?:per\s+contra\b|"
                    r"in\s+reply\b(?!\s+(?:to|dated|filed))|in\s+response\b(?!\s+to)|replying|countering|"
                    r"refuting|controverting|opposing|resisting)", re.I)

_RX_ANSWER = re.compile(r"(?:refut|respond|answer|meet|meeting|counter|rebut|disput|den(?:y|ied|ies)|"
                        r"repl(?:y|ied|ying)|controvert|oppos|resist)\w*\s+(?:to\s+)?(?:\w+\s+){0,5}?$", re.I)


_RX_B = re.compile(r"^\W*(?:\w+\W+){0,5}?(?:on\s+the\s+other\s+hand|on\s+the\s+contrary)", re.I)
_RX_RJ = re.compile(r"\b(?:rejoinder|in\s+rebuttal|by\s+way\s+of\s+rebuttal)", re.I)
_RX_SUPPORT = re.compile(r"in\s+support\s+of\s+(?:the|these|this)\s+(?:appeal|petition|writ)", re.I)


HEAD = 200
_FLIP = {"P": "R", "R": "P", "REV": "ASS", "ASS": "REV"}


def cue(sentence: str) -> str | None:
    s = sentence[:HEAD]
    hits = []
    for k, rx in _RX.items():
        m = rx.search(s)
        if m:
            if _RX_ANSWER.search(s[:m.start() + 1][-80:]) and k in _FLIP:
                k = _FLIP[k]
            hits.append((m.end(), k))
    m = _RX_SUPPORT.search(s)
    if m:
        hits.append((m.end(), "P"))
    if hits:
        hits.sort()
        return hits[0][1]
    if _RX_RJ.search(s):
        return "RJ"
    if _RX_SW.search(s):
        return "SW"
    if _RX_B.search(s):
        return "B"
    return None


def regions(is_arg: list[bool]) -> list[tuple[int, int]]:
    out, i, n = [], 0, len(is_arg)
    while i < n:
        if is_arg[i]:
            j = i
            while j < n and is_arg[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def segments(sents: list[str], is_arg: list[bool]):
    segs = []
    for r, (a, b) in enumerate(regions(is_arg)):
        cur = None
        for i in range(a, b):
            k = cue(sents[i])
            if k is not None or cur is None:
                if k is None and i == a and a > 0:
                    k = cue(sents[a - 1])
                    k = k if k else None
                cur = {"idx": [], "key": k, "region": r}
                segs.append(cur)
            cur["idx"].append(i)
    return segs


def resolve(segs, votes=None):
    party = [None] * len(segs)
    for i, s in enumerate(segs):
        k = s["key"]
        if k in ("P", "R"):
            party[i] = k
        elif k == "RJ":
            party[i] = "P"

    if votes is not None:
        tally = collections.Counter()
        for s, v in zip(segs, votes):
            if s["key"] == "REV":
                tally["REV=P"] += v["P"]; tally["REV=R"] += v["R"]
            elif s["key"] == "ASS":
                tally["REV=R"] += v["P"]; tally["REV=P"] += v["R"]
        if tally["REV=P"] != tally["REV=R"]:
            rev = "P" if tally["REV=P"] > tally["REV=R"] else "R"
            ass = "R" if rev == "P" else "P"
            for i, s in enumerate(segs):
                if s["key"] == "REV":
                    party[i] = rev
                elif s["key"] == "ASS":
                    party[i] = ass

    for i, s in enumerate(segs):
        if s["key"] == "SW":
            if i > 0 and segs[i - 1]["region"] == s["region"] and party[i - 1]:
                party[i] = "R" if party[i - 1] == "P" else "P"
            else:
                party[i] = "R"
    return party


MODE_UNCUED = "keep"


def apply_doc(sents, labels, mode_uncued=MODE_UNCUED):
    is_arg = [l in ARG for l in labels]
    segs = segments(sents, is_arg)
    votes = [collections.Counter(labels[i][8] for i in s["idx"]) for s in segs]
    party = resolve(segs, votes)
    out = list(labels)
    for s, p, v in zip(segs, party, votes):
        if p is None:
            if mode_uncued == "majority" and v["P"] != v["R"]:
                p = "P" if v["P"] > v["R"] else "R"
            else:
                continue
        lab = P if p == "P" else R
        for i in s["idx"]:
            out[i] = lab
    return out


def apply(pred: dict, docs: list, mode_uncued=MODE_UNCUED) -> dict:
    by = {d.id: d for d in docs}
    return {k: (apply_doc(by[k].sentences, v, mode_uncued) if k in by else list(v))
            for k, v in pred.items()}


def train_simulation(docs, verbose=True):
    stats = collections.Counter()
    cue_tab = collections.Counter()
    for d in docs:
        is_arg = [l in ARG for l in d.labels]
        segs = segments(d.sentences, is_arg)
        gold_votes = [collections.Counter(d.labels[i][8] for i in s["idx"]) for s in segs]


        party = resolve(segs, None)
        for s, p, gv in zip(segs, party, gold_votes):
            n = len(s["idx"])
            stats["sent"] += n
            maj = max(gv["P"], gv["R"])
            stats["purity_hits"] += maj
            cue_tab[(s["key"], "P")] += gv["P"]; cue_tab[(s["key"], "R")] += gv["R"]
            if p is None:
                stats["uncued"] += n
            else:
                stats["cued"] += n
                stats["cued_correct"] += gv[p]

        tal = collections.Counter()
        for s, gv in zip(segs, gold_votes):
            if s["key"] == "REV":
                tal["REV=P"] += gv["P"]; tal["REV=R"] += gv["R"]
            elif s["key"] == "ASS":
                tal["REV=R"] += gv["P"]; tal["REV=P"] += gv["R"]
        stats["ra_sent"] += tal["REV=P"] + tal["REV=R"]
        stats["ra_oracle_hits"] += max(tal["REV=P"], tal["REV=R"])
    if verbose:
        print("== train simulation (gold argument regions, party hidden) ==")
        print(f"argument sentences            : {stats['sent']}")
        print(f"cue-segment purity (oracle)   : {stats['purity_hits']/stats['sent']:.3f}  "
              f"(upper bound of any per-segment relabel)")
        print(f"sentences in label-free cued segments: {stats['cued']} "
              f"({stats['cued']/stats['sent']:.1%}); party accuracy {stats['cued_correct']/max(1,stats['cued']):.3f}")
        print(f"REV/ASS segments: {stats['ra_sent']} sentences; best per-doc mapping accuracy "
              f"{stats['ra_oracle_hits']/max(1,stats['ra_sent']):.3f} (consistency of the side)")
        print(f"uncued (left to model)        : {stats['uncued']} ({stats['uncued']/stats['sent']:.1%})")
        print("per cue kind, gold sentences P/R in the segments it opens:")
        for k in ("P", "R", "SW", "B", "RJ", "REV", "ASS", None):
            print(f"   {str(k):5s} P {cue_tab[(k,'P')]:5d}  R {cue_tab[(k,'R')]:5d}")
    return stats


def _conf(pred, gold):
    c = collections.Counter()
    for k, g in gold.items():
        for a, b in zip(pred[k], g):
            if a == P and b == R:
                c["R->P"] += 1
            elif a == R and b == P:
                c["P->R"] += 1
    return c


RUNS = ["bitfit40_s3", "bitfit40", "bitfit40_s2", "ia3", "adapter40", "adapter", "qlora", "lora"]
EXTRA = ["bitfit", "bitfit_boost", "bitfit_boost2", "full", "rslora", "lora_boost", "lora_boost2",
         "prefix", "prompt", "no_crf", "no_doc"]


def _dev_file(run):
    for f in (ROOT / "runs" / "B_grown" / run / "dev_predictions.json",
              ROOT / "runs" / run / "dev_predictions.json"):
        if f.exists():
            return f
    return None


def evaluate(mode_uncued=MODE_UNCUED, runs=RUNS):
    from rr import data as D
    from rr.metrics import score
    vocab = D.load_labels(DATA)
    train = D.load_all(DATA, "train")
    train_simulation(train)
    dev = D.load_all(DATA, "dev")
    gold = {d.id: d.labels for d in dev}
    print(f"\n== dev (check only; 10 docs) mode_uncued={mode_uncued} ==")
    print(f"{'run':12s}{'macroF1':>17s}{'F1 ArgPet':>17s}{'F1 ArgResp':>17s}"
          f"{'gold P->pred R':>16s}{'gold R->pred P':>16s}")
    better = 0
    n = 0
    for run in runs:
        f = _dev_file(run)
        if f is None:
            print(f"{run:12s} missing"); continue
        pred = json.loads(f.read_text())
        post = apply(pred, dev, mode_uncued)
        a, b = score(pred, gold, vocab), score(post, gold, vocab)
        ca, cb = _conf(pred, gold), _conf(post, gold)
        fp = lambda s, l: s["per_label"][l]["f1"]
        better += b["macro_f1"] > a["macro_f1"]; n += 1
        print(f"{run:12s}{a['macro_f1']:.4f}->{b['macro_f1']:.4f}"
              f"   {fp(a,P):.3f}->{fp(b,P):.3f}    {fp(a,R):.3f}->{fp(b,R):.3f}"
              f"     {ca['P->R']:4d}->{cb['P->R']:<4d}      {ca['R->P']:4d}->{cb['R->P']:<4d}")
    print(f"improved macro-F1 on {better}/{n} runs")


def apply_zip(src, out):
    from rr import data as D
    from rr.metrics import validate_submission
    src, out = Path(src), Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    if not out.name.endswith("_arg.zip"):
        raise SystemExit("output name must end in _arg.zip")
    with zipfile.ZipFile(src) as z:
        name = [n for n in z.namelist() if n.endswith("prediction.json")][0]
        pred = json.loads(z.read(name))
    vocab = D.load_labels(DATA)
    docs = D.load_all(DATA, "test")
    post = apply(pred, docs)
    errs = validate_submission(post, docs, vocab)
    if errs:
        raise SystemExit("validation failed: " + "; ".join(errs[:5]))
    n_changed = sum(a != b for k in pred for a, b in zip(pred[k], post[k]))
    cnt = lambda p, l: sum(x == l for v in p.values() for x in v)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("prediction.json", json.dumps(post))
    print(f"wrote {out}: {n_changed} sentences changed; "
          f"P {cnt(pred,P)}->{cnt(post,P)}  R {cnt(pred,R)}->{cnt(post,R)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--mode", default=MODE_UNCUED, choices=["keep", "majority"])
    ap.add_argument("--all", action="store_true", help="also evaluate the other B_grown runs")
    ap.add_argument("--apply")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.eval:
        evaluate(a.mode, RUNS + (EXTRA if a.all else []))
    if a.apply:
        if not a.out:
            raise SystemExit("--out required")
        apply_zip(a.apply, a.out)

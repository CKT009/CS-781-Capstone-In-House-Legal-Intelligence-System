from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "lex"

MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"])}
MONTHS.update({m[:3]: v for m, v in list(MONTHS.items())})
MONTHS["sept"] = 9
MON = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
YR = r"(1[789]\d\d|20[0-2]\d)"
MAXY = 2024

RE_NUM = re.compile(r"\b(\d{1,2})\s?[./-]\s?(\d{1,2})\s?[./-]\s?" + YR + r"\b")
RE_DMY = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s*(?:day\s+of\s+)?(" + MON + r")\s*,?\s*" + YR + r"\b", re.I)
RE_MDY = re.compile(r"\b(" + MON + r")\s*(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*" + YR + r"\b", re.I)
RE_MY = re.compile(r"\b(" + MON + r")\s*,?\s*" + YR + r"\b", re.I)

RE_YCTX = re.compile(r"(?:\bof|\bin|\byear|\bAct\s*,?|\bdated|\bsince|\buntil|\btill|\bupto|\bAIR|\bSCR|\bSCC|\(|\[)\s*" + YR + r"\b", re.I)
RE_YRANGE = re.compile(r"\b" + YR + r"\s*-\s*(\d{2})\b")

CUES = re.compile(r"(?:\bDATED\b\s*:?|\bDated\s*(?::|this\b)|(?i:\bDate\s+of\s+(?:Decision|Judgment|Order|pronouncement)\s*:?|"
                  r"\bDecided\s+on\s*:?|\bPronounced\s+on\s*:?|\bDelivered\s+on\s*:?|"
                  r"\bNew\s+Delhi\s*[,;:.]?|\bJudgment\s+(?:delivered|pronounced)\s+on\s*:?))\s*")


def _ok(y, m, d):
    return 1800 <= y <= MAXY and 1 <= m <= 12 and 1 <= d <= 31


def dates_in(text: str):
    out = []
    for r in RE_NUM.finditer(text):
        d, m, y = int(r.group(1)), int(r.group(2)), int(r.group(3))
        if _ok(y, m, d):
            out.append((y, m, d, r.start()))
    for r in RE_DMY.finditer(text):
        d, m, y = int(r.group(1)), MONTHS[r.group(2).lower().rstrip(".")[:3]], int(r.group(3))
        if _ok(y, m, d):
            out.append((y, m, d, r.start()))
    for r in RE_MDY.finditer(text):
        m, d, y = MONTHS[r.group(1).lower().rstrip(".")[:3]], int(r.group(2)), int(r.group(3))
        if _ok(y, m, d):
            out.append((y, m, d, r.start()))
    for r in RE_MY.finditer(text):
        m, y = MONTHS[r.group(1).lower().rstrip(".")[:3]], int(r.group(2))
        if _ok(y, m, 1):
            out.append((y, m, 0, r.start()))
    for r in RE_YCTX.finditer(text):
        y = int(r.group(1))
        if 1800 <= y <= MAXY:
            out.append((y, 0, 0, r.start()))
    for r in RE_YRANGE.finditer(text):
        y = int(r.group(1)); y2 = (y // 100) * 100 + int(r.group(2))
        if y2 == y + 1 and y2 <= MAXY:
            out.append((y2, 0, 0, r.start()))
    return out


def _fmt(y, m, d):
    return f"{y:04d}-{max(m, 1):02d}-{max(d, 1):02d}"


def _cue_date(text):
    zones = [(0, text[:800]), (max(0, len(text) - 800), text[-800:])]
    for off, z in zones:
        for c in CUES.finditer(z):
            ds = [x for x in dates_in(z[c.end():c.end() + 40]) if x[3] < 6 and x[2] > 0]
            if ds:
                y, m, d, _ = ds[0]
                return _fmt(y, m, d)
    return None


LABELS = [("case_no", r"CASE NO\.?\s*:"), ("petitioner", r"PETITIONER\s*:"),
          ("respondent", r"RESPONDENT\s*:"), ("date", r"DATE OF JUDGMENT\s*:?"),
          ("bench", r"BENCH\s*:"), ("citation", r"CITATION\s*:"),
          ("citator", r"CITATOR INFO\s*:"), ("act", r"\bACT\s*:"),
          ("headnote", r"HEADNOTE\s*:"), ("judgment", r"\bJUDGMENT\s*:")]


def _deorg(s):
    return re.sub(r"(?<=\S)<ORG>", ".", s)


def header(text: str):
    if "PETITIONER" not in text[:600] and "DATE OF JUDGMENT" not in text[:3000]:
        return None
    pos, cur = {}, 0
    for name, pat in LABELS:
        lim = len(text) if name == "judgment" else 6000
        r = re.compile(pat).search(text, cur, lim if name != "judgment" else len(text))
        if r is None:
            continue
        pos[name] = (r.start(), r.end())
        cur = r.end()
    if not pos:
        return None
    names = sorted(pos, key=lambda n: pos[n][0])
    f = {}
    for i, n in enumerate(names):
        end = pos[names[i + 1]][0] if i + 1 < len(names) else len(text)
        f[n] = text[pos[n][1]:end].strip() if n != "judgment" else ""
    f["body_start"] = pos["judgment"][1] if "judgment" in pos else (pos[names[-1]][1] if names[-1] != "headnote" else len(text))
    if "date" in f:
        r = re.match(r"\s*(\d{1,2})/(\d{1,2})/(\d{4})", f["date"])
        f["date"] = _fmt(int(r.group(3)), int(r.group(2)), int(r.group(1))) if r and _ok(int(r.group(3)), int(r.group(2)), int(r.group(1))) else None

    if "bench" in f:
        b = f["bench"]
        f["bench"] = b
    return f


def judges(bench: str):
    if not bench:
        return []
    b = re.sub(r"<[A-Z]+>", " ", bench)
    b = b.split("BENCH:")[-1]
    out = []
    for m in re.finditer(r"\b([A-Z][A-Z\.]{2,})\s*,", b):
        s = m.group(1).strip(".")
        if s not in ("J", "CJ", "CJI", "JJ") and s not in out:
            out.append(s)
    if not out:
        for part in re.split(r"&|\band\b", b):
            w = re.findall(r"[A-Z]{3,}", part)
            if w and w[-1] not in out:
                out.append(w[-1])
    return out


ALIASES = [
    (r"code of civil procedure|civil procedure code|c\.\s?p\.\s?c\.?|\bcpc\b", "code of civil procedure"),
    (r"code of criminal procedure|criminal procedure code|cr\.\s?p\.\s?c\.?|crpc|cr\.p\.c", "code of criminal procedure"),
    (r"indian penal code|penal code|i\.\s?p\.\s?c\.?|\bipc\b", "indian penal code"),
    (r"constitution", "constitution"),
    (r"income[\s-]*tax act|i\.\s?t\.\s?act", "income tax act"),
    (r"evidence act", "evidence act"),
    (r"limitation act", "limitation act"),
    (r"transfer of property act|t\.\s?p\.\s?act", "transfer of property act"),
    (r"contract act", "contract act"),
    (r"specific relief act", "specific relief act"),
    (r"industrial disputes act|i\.\s?d\.\s?act", "industrial disputes act"),
    (r"arbitration", "arbitration act"),
    (r"motor vehicles act|m\.\s?v\.\s?act", "motor vehicles act"),
    (r"land acquisition act|l\.\s?a\.\s?act", "land acquisition act"),
    (r"negotiable instruments act|n\.\s?i\.\s?act", "negotiable instruments act"),
    (r"prevention of corruption act", "prevention of corruption act"),
    (r"narcotic drugs|n\.\s?d\.\s?p\.\s?s\.?", "ndps act"),
    (r"general clauses act", "general clauses act"),
    (r"companies act", "companies act"),
    (r"central excise", "central excise act"),
    (r"customs act", "customs act"),
    (r"sales tax act", "sales tax act"),
    (r"hindu succession act", "hindu succession act"),
    (r"hindu marriage act", "hindu marriage act"),
    (r"representation of the people act", "representation of the people act"),
    (r"wealth[\s-]*tax act", "wealth tax act"),
    (r"stamp act", "stamp act"),
    (r"registration act", "registration act"),
]
ALIAS_RE = [(re.compile(a, re.I), n) for a, n in ALIASES]
STOP = {"the", "of", "and", "indian", "in", "under", "said", "this", "that", "an", "a", "said"}
RE_ACT = re.compile(r"((?:(?:[A-Z]\.){1,4}\s*|[A-Z][\w'&\-]*\.?\s+|\(\s*[\w\s,&'\-]+\)\s+|of\s+|and\s+|the\s+|&\s+){0,9}?(?:[A-Z][\w'&\-]*\s+)?Act)\b\s*,?\s*(?:of\s+)?(\d{4})?")
RE_CODE = re.compile(r"(Code of Civil Procedure|Civil Procedure Code|C\.\s?P\.\s?C\.?|Code of Criminal Procedure|Criminal Procedure Code|Cr\.\s?P\.\s?C\.?|Indian Penal Code|Penal Code|I\.\s?P\.\s?C\.?|\bIPC\b|\bCPC\b|Constitution)", re.I)
RE_SEC = re.compile(r"\b(?:Sections?|Secs?\.|Ss?\.|Articles?|Arts?\.|Orders?|O\.)\s*((?:\d+[A-Z]{0,2}(?:\s*\([\w]+\))*(?:\s*(?:,|and|&|/|to|-)\s*)?)+)", re.I)
RE_SECNUM = re.compile(r"(\d+[A-Z]{0,2})")
GENERIC = {"act", "the act", "said act", "this act", "that act", "principal act", "amending act", "parent act", "new act", "old act", "central act", "state act"}


def norm_act(name: str):
    s = re.sub(r"<[A-Z]+>", " ", name)
    s = re.split(r"(?<![A-Z])[.,;:]\s", s)[-1]
    name = s
    for r, n in ALIAS_RE:
        if r.search(s):
            return n
    s = re.sub(r"\([^)]*\)", " ", s.lower())
    s = re.sub(r"[^a-z ]", " ", s)
    w = [x for x in s.split() if x not in STOP]

    raw = re.findall(r"[A-Za-z]+", re.sub(r"\([^)]*\)", " ", re.sub(r"<[A-Z]+>", " ", name)))
    if len(raw) < 2 or not raw[-2][0].isupper():
        return None
    if not w or w[-1] != "act" or len(w) < 2:
        return None
    w = w[-6:]
    n = " ".join(w)
    return None if n in GENERIC else n


def statutes(text: str):
    t = _deorg(text)
    mentions = []
    for m in RE_ACT.finditer(t):
        n = norm_act(m.group(1))
        if n:
            mentions.append((m.start(), m.end(), n, int(m.group(2)) if m.group(2) else None))
    for m in RE_CODE.finditer(t):
        n = norm_act(m.group(1))
        if n:
            mentions.append((m.start(), m.end(), n, None))
    mentions.sort()
    acts = {}
    for _, _, n, y in mentions:
        if n not in acts or (acts[n] is None and y):
            acts[n] = y
    by_end = sorted(mentions, key=lambda x: x[1])
    ends = [x[1] for x in by_end]
    act_sec, loose = set(), set()
    import bisect
    for m in RE_SEC.finditer(t):
        kind = m.group(0)[:2].lower()
        nums = RE_SECNUM.findall(re.sub(r"\([^)]*\)", " ", m.group(1)))[:6]
        if not nums:
            continue

        i = bisect.bisect_right(ends, m.end())
        j = bisect.bisect_right(ends, m.start()) - 1
        tgt = None
        if i < len(by_end):
            gap = t[m.end():by_end[i][1]]
            if len(gap) < 90 and (re.match(r"\s*,?\s*(?:of|under)\b", gap) or len(gap) < 25):
                tgt = by_end[i][2]
        if tgt is None and j >= 0 and m.start() - by_end[j][1] < 40:
            tgt = by_end[j][2]
        if kind == "ar":
            tgt = tgt or "constitution"
        pre = "art" if kind == "ar" else ("ord" if kind in ("or", "o.") else "s")
        for x in nums:
            if tgt:
                act_sec.add(f"{tgt}|{pre}{x.upper()}")
            else:
                loose.add(f"{pre}{x.upper()}")
    return acts, act_sec, loose


def parse_doc(text: str) -> dict:
    h = header(text)
    rec = {"has_header": h is not None}
    date, src = None, None
    if h and h.get("date"):
        date, src = h["date"], "header"
    if date is None:
        c = _cue_date(text)
        if c:
            date, src = c, "cue"
    ds = dates_in(_deorg(text[h["body_start"]:] if h and h.get("body_start", 0) < len(text) else text))


    srt = sorted((y, m, d) for y, m, d, _ in ds)
    lb = srt[-2] if len(srt) > 1 else (srt[-1] if srt else None)
    rec["date_max"] = _fmt(*srt[-1]) if srt else None
    rec["date_lb"] = _fmt(*lb) if lb else None
    if date is None and lb:
        date, src = _fmt(*lb), "lb"
    rec["date"], rec["date_src"] = date, src
    hn = (h or {}).get("headnote", "")
    actf = (h or {}).get("act", "")
    rec["petitioner"] = (h or {}).get("petitioner", "")[:300]
    rec["respondent"] = (h or {}).get("respondent", "")[:300]
    rec["bench_raw"] = (h or {}).get("bench", "")[:400]
    rec["judges"] = judges(rec["bench_raw"])
    rec["citation"] = (h or {}).get("citation", "")[:300]
    rec["act_field"] = _deorg(actf)[:3000]
    rec["headnote"] = _deorg(hn)
    body = text[h["body_start"]:] if h and h.get("body_start", 0) < len(text) else text
    rec["body"] = body
    a1, s1, _ = statutes(actf) if actf else ({}, set(), set())
    a2, s2, l2 = statutes(body)
    rec["act_acts"] = sorted(a1)
    rec["act_secs"] = sorted(s1)
    acts = dict(a2); acts.update({k: v for k, v in a1.items() if v or k not in acts})
    rec["acts"] = {k: acts[k] for k in sorted(acts)}
    rec["act_secs_all"] = sorted(s1 | s2)
    rec["secs_loose"] = sorted(l2)
    return rec


def parse_split(split: str, force: bool = False) -> dict:
    from pcr import data as D
    f = OUT / f"parsed_{split}.json"
    if f.exists() and not force:
        return json.loads(f.read_text())
    q, pool = D.load(split)
    docs = dict(pool)
    docs.update({k: v["text"] for k, v in q.items()})
    from concurrent.futures import ProcessPoolExecutor
    keys = list(docs)
    with ProcessPoolExecutor(16) as ex:
        recs = list(ex.map(parse_doc, [docs[k] for k in keys], chunksize=32))
    out = dict(zip(keys, recs))
    OUT.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out))
    return out


def date_report(parsed: dict) -> dict:
    c = Counter(r["date_src"] for r in parsed.values())
    n = len(parsed)
    return {"n": n, **{str(k): v for k, v in c.items()},
            "exact_rate": (c["header"] + c["cue"]) / n, "any_rate": (n - c[None]) / n}

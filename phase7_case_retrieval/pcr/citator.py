from __future__ import annotations

import re

# About a third of the judgments have a header like
#   CITATION: 1966 AIR 1053 1966 SCR (2) 944
#   CITATOR INFO : F 1967 SC 819 (5) R 1969 SC 840 (11) ...
# The CITATOR INFO block lists the later cases that cite this one. If query Q cites
# candidate C, Q's own citation appears in C's block. Reporters disagree on the series
# name (AIR vs SC), so only the (year, number) pair is matched.

NUM = re.compile(r"\b(1[89]\d\d|20\d\d)\s+(?:AIR|SC|SCR|SCC)\s*(?:\(\d+\))?\s*(\d{1,4})")
CITED_BY = re.compile(r"CITATOR INFO(.{0,4000}?)(?:ACT\s*:|HEADNOTE\s*:|JUDGMENT|$)", re.S)
OWN = re.compile(r"CITATION\s*:(.{0,300}?)(?:CITATOR INFO|ACT\s*:|HEADNOTE\s*:|$)", re.S)


def _find(pat, text):
    m = pat.search(text)
    return set(NUM.findall(m.group(1))) if m else set()


def cited_by(text: str) -> set[tuple[str, str]]:
    return _find(CITED_BY, text)


def own_citation(text: str) -> set[tuple[str, str]]:
    return _find(OWN, text)


def matches(qid: str, pool: dict[str, str]) -> list[str]:
    q = own_citation(pool[qid]) if qid in pool else set()
    if not q:
        return []
    return [c for c, t in pool.items() if c != qid and (cited_by(t) & q)]


def promote(ranked: dict[str, list[str]], pool: dict[str, str]):
    out, touched = {}, 0
    for qid, order in ranked.items():
        front = matches(qid, pool)
        touched += bool(front)
        seen = set(front)
        out[qid] = front + [c for c in order if c not in seen]
    return out, touched

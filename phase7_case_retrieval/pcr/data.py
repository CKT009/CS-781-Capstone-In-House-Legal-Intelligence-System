from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Phase_7_PCR"
FILES = {"train": ("Train/train.json", "Train/train_candidates.json"),
         "dev": ("Dev/dev.json", "Dev/dev_candidates.json"),
         "test": ("Test/test.json", "Test/test_candidates.json")}
K = 7


def load(split: str):
    qf, cf = FILES[split]
    qraw = json.loads((DATA / qf).read_text())
    craw = json.loads((DATA / cf).read_text())
    q = {k: {"text": " ".join(v["text"]),
             "rel": set(v["relevant_candidates"]) if "relevant_candidates" in v else None}
         for k, v in qraw.items()}
    pool = {k: " ".join(v["text"]) for k, v in craw.items()}
    if split != "test":
        assert all(r["rel"] <= set(pool) for r in q.values()), "relevant id outside pool"
    return q, pool


def candidates(qid: str, pool: dict) -> list[str]:
    return [c for c in pool if c != qid]

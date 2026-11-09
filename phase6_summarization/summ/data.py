from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FILES = {"train": ROOT / "data" / "train.json", "test": ROOT / "data" / "test.json"}
N_VAL, SEED = 500, 0


def load(split: str) -> list[dict]:
    rows = json.loads(FILES[split].read_text())
    assert len({r["id"] for r in rows}) == len(rows), f"{split}: duplicate ids"
    return rows


def split() -> tuple[list[dict], list[dict]]:
    rows = load("train")
    idx = np.random.default_rng(SEED).permutation(len(rows))
    va = set(idx[:N_VAL].tolist())
    return ([r for i, r in enumerate(rows) if i not in va],
            [r for i, r in enumerate(rows) if i in va])


_SENT = re.compile(r"(?<=[.?!])\s+(?=[A-Z(\"'])")


def sentences(text: str, min_words: int = 4) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if len(s.split()) >= min_words]

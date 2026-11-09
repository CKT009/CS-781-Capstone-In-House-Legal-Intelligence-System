from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Phase_5_LSI"
CACHE = ROOT / "artifacts" / "tokens"
TOKENIZER = ROOT / "artifacts" / "tokenizer_16k" / "tokenizer.json"
FILES = {"train": "Train/train.json", "dev": "Train/dev.json", "test": "Test/test.json"}
N_LABELS = 100


@dataclass
class Doc:
    id: str
    text: str
    labels: list[int] | None


def load(split: str) -> list[Doc]:
    rows = json.loads((DATA / FILES[split]).read_text())
    out = []
    for r in rows:
        labs = sorted(set(int(l) for l in r["labels"])) if "labels" in r else None
        out.append(Doc(str(r["id"]), r["text"], labs))
    if split == "test":
        assert all(d.labels is None for d in out), "test should carry no labels"
    else:
        assert all(d.labels and 0 <= min(d.labels) and max(d.labels) < N_LABELS for d in out)
    assert len({d.id for d in out}) == len(out), f"{split}: duplicate ids"
    return out


def label_matrix(docs: list[Doc]) -> np.ndarray:
    Y = np.zeros((len(docs), N_LABELS), dtype=np.float32)
    for i, d in enumerate(docs):
        Y[i, d.labels] = 1.0
    return Y


def build_cache(splits=("train", "dev", "test")) -> None:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER))
    CACHE.mkdir(parents=True, exist_ok=True)
    for s in splits:
        docs = load(s)
        flat, index, pos, B = [], {}, 0, 512
        for i in range(0, len(docs), B):
            for d, e in zip(docs[i:i + B], tok.encode_batch([d.text for d in docs[i:i + B]])):
                a = np.asarray(e.ids, dtype=np.int32)
                flat.append(a); index[d.id] = [pos, len(a)]; pos += len(a)
        np.save(CACHE / f"{s}.ids.npy", np.concatenate(flat))
        (CACHE / f"{s}.idx.json").write_text(json.dumps(index))
        n = np.array([v[1] for v in index.values()])
        print(f"[tokens] {s}: {len(docs):,} docs, {pos/1e6:.1f}M tokens, median "
              f"{int(np.median(n)):,}, p75 {int(np.percentile(n, 75)):,}, "
              f"p90 {int(np.percentile(n, 90)):,}, max {n.max():,}")


def tokens(split: str) -> dict[str, np.ndarray]:
    p = CACHE / f"{split}.ids.npy"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing -- run `python -c 'from lsi import data; "
                                f"data.build_cache()'`")
    flat = np.load(p, mmap_mode="r")
    idx = json.loads((CACHE / f"{split}.idx.json").read_text())
    return {k: flat[a:a + n] for k, (a, n) in idx.items()}


def window_starts(n: int, L: int, K: int, keep: str) -> list[int]:
    if n <= L:
        return [0]
    starts = list(range(0, n - L + 1, L))
    if starts[-1] != n - L:
        starts.append(n - L)
    if len(starts) <= K:
        return starts
    if keep == "head":
        return starts[:K]
    if keep == "tail":
        return starts[-K:]
    if keep == "ends":
        h = K // 2
        return starts[:h] + starts[-(K - h):]
    if keep == "spread":
        idx = np.linspace(0, len(starts) - 1, K).round().astype(int)
        return [starts[i] for i in sorted(set(idx))]
    raise ValueError(keep)

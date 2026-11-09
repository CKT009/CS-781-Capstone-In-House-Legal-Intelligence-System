from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "Phase_4_CJPE"
CACHE = Path(__file__).resolve().parent.parent / "artifacts" / "tokens"
TOKENIZER = Path(__file__).resolve().parent.parent / "artifacts" / "tokenizer_16k" / "tokenizer.json"
FILES = {"train": "Train/train.json", "dev": "Train/dev.json", "test": "Test/test.json"}


@dataclass
class Doc:
    id: str
    text: str
    label: int | None

    @property
    def year(self) -> int:
        return int(self.id.split("_")[0])


def load(split: str) -> list[Doc]:
    rows = json.loads((DATA / FILES[split]).read_text())
    out, seen = [], set()
    for r in rows:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        lab = None if split == "test" else int(r["label"])
        out.append(Doc(r["id"], r["text"], lab))
    if split == "test":
        assert all(d.label is None for d in out)
    else:
        assert all(d.label in (0, 1) for d in out), f"{split}: label outside {{0,1}}"
    return out


def clean_dev_mask(dev_docs: list[Doc]) -> np.ndarray:
    test_ids = {d.id for d in load("test")}
    return np.array([d.id not in test_ids for d in dev_docs])


DECISION = re.compile(
    r"(appeals?|petitions?|writ petitions?|special leave petitions?|suits?|revisions?)\s+"
    r"(is|are|stands?|must be|should be|shall be|be|were|was)\s+"
    r"(accordingly\s+|hereby\s+|therefore\s+|partly\s+|partially\s+)?"
    r"(allowed|dismissed|disposed of|rejected|accepted|succeeds?|fails?)"
    r"|we\s+(accordingly\s+|therefore\s+|hereby\s+)?(allow|dismiss|set aside|quash|uphold|confirm)\b"
    r"|(appeals?|petitions?)\s+(succeeds?|fails?)"
    r"|no order as to costs|rule\s+(is\s+)?(made absolute|discharged)"
    r"|(impugned|the)\s+(order|judgment|decree)s?\s+(of the high court\s+)?(is|are)\s+"
    r"(hereby\s+)?(set aside|quashed|restored|upheld|affirmed|confirmed)"
    r"|accordingly\s+(allowed|dismissed)", re.I)


def redact(text: str, window: int = 2500) -> str:
    off = max(0, len(text) - window)
    m = DECISION.search(text, off)
    if not m:
        return text
    cut = text.rfind(". ", off, m.start())
    cut = cut + 1 if cut >= 0 else m.start()
    return text[:cut].rstrip()


def build_redacted_cache() -> None:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER))
    docs = load("train")
    flat, index, pos, n_cut, chars = [], {}, 0, 0, 0
    B = 512
    for i in range(0, len(docs), B):
        texts = []
        for d in docs[i:i + B]:
            t = redact(d.text)
            if len(t) < len(d.text):
                n_cut += 1; chars += len(d.text) - len(t)
            texts.append(t)
        for d, e in zip(docs[i:i + B], tok.encode_batch(texts)):
            a = np.asarray(e.ids, dtype=np.int32)
            flat.append(a); index[d.id] = [pos, len(a)]; pos += len(a)
    np.save(CACHE / "train_redacted.ids.npy", np.concatenate(flat))
    (CACHE / "train_redacted.idx.json").write_text(json.dumps(index))
    print(f"[redact] cut {n_cut:,} of {len(docs):,} train docs ({n_cut/len(docs):.1%}), "
          f"mean {chars/max(n_cut,1):,.0f} chars removed from each")


def build_cache(splits=("train", "dev", "test")) -> None:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER))
    CACHE.mkdir(parents=True, exist_ok=True)
    for s in splits:
        docs = load(s)
        flat, index, pos = [], {}, 0
        B = 512
        for i in range(0, len(docs), B):
            enc = tok.encode_batch([d.text for d in docs[i:i + B]])
            for d, e in zip(docs[i:i + B], enc):
                a = np.asarray(e.ids, dtype=np.int32)
                flat.append(a)
                index[d.id] = [pos, len(a)]
                pos += len(a)
        np.save(CACHE / f"{s}.ids.npy", np.concatenate(flat))
        (CACHE / f"{s}.idx.json").write_text(json.dumps(index))
        n = np.array([v[1] for v in index.values()])
        print(f"[tokens] {s}: {len(docs):,} docs, {pos/1e6:.1f}M tokens, "
              f"median {int(np.median(n)):,}, p90 {int(np.percentile(n, 90)):,}")


def tokens(split: str) -> dict[str, np.ndarray]:
    p = CACHE / f"{split}.ids.npy"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing -- run `python run_train.py --build-cache`")
    flat = np.load(p, mmap_mode="r")
    idx = json.loads((CACHE / f"{split}.idx.json").read_text())
    return {k: flat[a:a + n] for k, (a, n) in idx.items()}


def chunks(ids: np.ndarray, L: int, stride: int | None = None, max_chunks: int = 0,
           keep: str = "tail") -> list[np.ndarray]:
    stride = stride or L
    n = len(ids)
    if n <= L:
        return [np.asarray(ids)]
    starts = list(range(0, n - L + 1, stride))
    if starts[-1] != n - L:
        starts.append(n - L)
    out = [np.asarray(ids[s:s + L]) for s in starts]
    if max_chunks and len(out) > max_chunks:
        if keep == "tail":
            out = out[-max_chunks:]
        elif keep == "head":
            out = out[:max_chunks]
        elif keep == "ends":
            h = max_chunks // 2
            out = out[:h] + out[-(max_chunks - h):]
        else:
            raise ValueError(keep)
    return out


def view(ids: np.ndarray, how: str, L: int) -> np.ndarray:
    ids = np.asarray(ids)
    if len(ids) <= L:
        return ids
    if how == "head":
        return ids[:L]
    if how == "tail":
        return ids[-L:]
    if how == "ends":
        h = L // 2
        return np.concatenate([ids[:h], ids[-(L - h):]])
    raise ValueError(how)

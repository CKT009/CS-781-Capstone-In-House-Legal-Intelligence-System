from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER = Path(__file__).resolve().parents[1] / "artifacts" / "tokenizer_16k" / "tokenizer.json"
CACHE = ROOT / "runs" / "neural" / "tok"
L_MAX = 1024
MAX_TOK = 16 * L_MAX


def build_cache() -> None:
    os.environ.setdefault("RAYON_NUM_THREADS", "8")
    from tokenizers import Tokenizer
    from pcr import data as D
    tok = Tokenizer.from_file(str(TOKENIZER))
    docs = {}
    for s in ("train", "dev", "test"):
        docs.update(D.load(s)[1])
    ids = sorted(docs)
    flat, index, pos, full = [], {}, 0, []
    for i in range(0, len(ids), 256):
        chunk = ids[i:i + 256]
        for d, e in zip(chunk, tok.encode_batch([docs[d] for d in chunk])):
            full.append(len(e.ids))
            a = np.asarray(e.ids[:MAX_TOK], dtype=np.int32)
            flat.append(a); index[d] = [pos, len(a)]; pos += len(a)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / "ids.npy", np.concatenate(flat))
    (CACHE / "idx.json").write_text(json.dumps(index))
    n = np.array(full)
    stats = {"docs": len(ids), "median": int(np.median(n)), "p75": int(np.percentile(n, 75)),
             "p90": int(np.percentile(n, 90)), "max": int(n.max()),
             **{f"whole_in_K{k}": round(float((n <= k * L_MAX).mean()), 3) for k in (1, 2, 4, 8)}}
    (CACHE / "stats.json").write_text(json.dumps(stats, indent=1))
    print(f"[tokens] {stats}", flush=True)


_TOK = None


def tokens() -> dict[str, np.ndarray]:
    global _TOK
    if _TOK is None:
        if not (CACHE / "idx.json").exists():
            build_cache()
        flat = np.load(CACHE / "ids.npy", mmap_mode="r")
        idx = json.loads((CACHE / "idx.json").read_text())
        _TOK = {k: flat[s:s + n] for k, (s, n) in idx.items()}
    return _TOK


def collate(seqs, K: int, L: int = L_MAX):
    per = []
    for t in seqs:
        n = max(1, len(t))
        per.append([np.asarray(t[s:s + L]) for s in range(0, min(n, K * L), L)] or [np.zeros(1)])
    B, Kb = len(seqs), max(len(p) for p in per)
    Lb = max(len(c) for p in per for c in p)
    x = torch.zeros(B, Kb, Lb, dtype=torch.long)
    m = torch.zeros(B, Kb, Lb, dtype=torch.long)
    cm = torch.zeros(B, Kb, dtype=torch.bool)
    for i, p in enumerate(per):
        for k, c in enumerate(p):
            x[i, k, :len(c)] = torch.from_numpy(c.astype(np.int64))
            m[i, k, :len(c)] = 1
            cm[i, k] = True
    return x, m, cm


@torch.no_grad()
def embed_docs(model, ids, K, L=L_MAX, device="cuda", batch=16, chunks=False):
    tok = tokens()
    was = model.training
    model.eval()
    order = np.argsort([-len(tok[i]) for i in ids], kind="stable")
    out = cmo = None
    for b in range(0, len(ids), batch):
        sel = order[b:b + batch]
        x, m, cm = collate([tok[ids[j]] for j in sel], K, L)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
            if chunks:
                v = model.chunk_vectors(x.to(device), m.to(device), cm.to(device))
            else:
                v = model(x.to(device), m.to(device), cm.to(device))
        v = v.float().cpu().numpy()
        if out is None:
            out = np.zeros((len(ids), K, v.shape[-1]) if chunks else (len(ids), v.shape[-1]),
                           dtype=np.float32)
            cmo = np.zeros((len(ids), K), dtype=bool)
        if chunks:
            out[sel, :v.shape[1]] = v
            cmo[sel, :cm.shape[1]] = cm.numpy()
        else:
            out[sel] = v
    model.train(was)
    return (out, cmo) if chunks else out


def pool_chunks(v: np.ndarray, cm: np.ndarray, pool: str, K: int | None = None) -> np.ndarray:
    if K is not None:
        v, cm = v[:, :K], cm[:, :K]
    if pool == "first":
        return v[:, 0]
    w = cm[..., None].astype(np.float32)
    if pool == "mean":
        return (v * w).sum(1) / np.maximum(w.sum(1), 1)
    if pool == "max":
        return np.where(cm[..., None], v, -np.inf).max(1)
    raise ValueError(pool)

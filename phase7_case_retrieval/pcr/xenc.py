from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from pcr import data as D

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER = Path(__file__).resolve().parents[1] / "artifacts" / "tokenizer_16k" / "tokenizer.json"
BOS, EOS, SEP, PAD = 2, 3, 5, 0
HALF = 510


def shortlist(split: str, n: int = 30) -> dict[str, list[tuple[str, float]]]:
    S = json.loads((ROOT / "runs" / "lex" / f"scores_{split}.json").read_text())
    return {q: sorted(((c, f["ltr"]) for c, f in cs.items() if c != q), key=lambda x: -x[1])[:n]
            for q, cs in S.items()}


def openings(split: str) -> dict[str, list[int]]:
    cache = ROOT / "runs" / "xenc" / f"open_{split}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER))
    q, pool = D.load(split)
    texts = {**pool, **{k: v["text"] for k, v in q.items()}}
    ids = list(texts)
    enc = tok.encode_batch([texts[i][:6000] for i in ids])
    out = {i: e.ids[:HALF] for i, e in zip(ids, enc)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def pair(qi: list[int], ci: list[int]) -> list[int]:
    return [BOS] + qi + [SEP] + ci + [EOS]


def batchify(seqs: list[list[int]], device):
    L = max(map(len, seqs))
    x = torch.full((len(seqs), L), PAD, dtype=torch.long)
    m = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s); m[i, :len(s)] = 1
    return x.to(device), m.to(device)


class CrossEncoder(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        d = backbone.d_model
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, ids, mask):
        h = self.backbone(ids, mask)
        m = mask.unsqueeze(-1).to(h.dtype)
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1)).squeeze(-1)


@torch.no_grad()
def score_pairs(model, opens, short, device, batch=32) -> dict[str, list[tuple[str, float]]]:
    model.eval()
    flat = [(q, c) for q, cs in short.items() for c, _ in cs]
    out: dict[str, list] = {q: [] for q in short}
    for i in range(0, len(flat), batch):
        chunk = flat[i:i + batch]
        x, m = batchify([pair(opens[q], opens[c]) for q, c in chunk], device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            s = model(x, m).float().cpu().numpy()
        for (q, c), v in zip(chunk, s):
            out[q].append((c, float(v)))
    model.train()
    return out


def fuse(ce: dict, short: dict, w: float) -> dict[str, list[str]]:
    ranked = {}
    for q, cs in short.items():
        lex = np.array([s for _, s in cs]); cand = [c for c, _ in cs]
        cmap = dict(ce[q]); ces = np.array([cmap[c] for c in cand])
        z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
        s = w * z(ces) + (1 - w) * z(lex)
        ranked[q] = [cand[j] for j in np.argsort(-s)]
    return ranked


def groups(q: dict, short: dict, n_neg: int, rng: random.Random):
    out = []
    for qid, v in q.items():
        negs = [c for c, _ in short.get(qid, []) if c not in v["rel"]]
        if not negs:
            continue
        for pos in v["rel"]:
            out.append((qid, pos, rng.sample(negs, min(n_neg, len(negs)))))
    rng.shuffle(out)
    return out

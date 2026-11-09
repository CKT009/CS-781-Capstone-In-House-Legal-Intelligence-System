from __future__ import annotations

import numpy as np
import torch

from summ.lm import BOS, EOS, PAD, SEP, tokenizer

IN, OUT = 560, 460


def get_builder(name: str):
    from summ import context
    parts = name.split("+")
    kw = {"strategy": parts[0]}
    for p in parts[1:]:
        if p == "gbt":
            kw["model"] = "gbt"
        elif p.startswith("mmr"):
            kw["mmr"] = float(p[3:])
        else:
            raise ValueError(f"unknown builder option {p!r} in {name!r}")
    return lambda doc, budget_tokens=IN: context.build_input(doc, budget_tokens, **kw)


def encode_input(document: str, builder="lead", n_in: int = IN) -> list[int]:
    ids = tokenizer().encode(get_builder(builder)(document, n_in)).ids[:n_in]
    return [BOS] + ids + [SEP]


def encode_pair(document: str, summary: str, builder="lead", n_in=IN, n_out=OUT):
    x = encode_input(document, builder, n_in)
    y = tokenizer().encode(summary).ids
    y = y[:n_out] if len(y) > n_out else y + [EOS]
    return x, y


class SummDataset(torch.utils.data.Dataset):

    def __init__(self, records, builder="lead", targets: dict | None = None, n_in=IN, n_out=OUT):
        self.items = [encode_pair(r["document"], (targets or {}).get(r["id"], r["summary"]),
                                  builder, n_in, n_out) for r in records]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    L = max(len(x) + len(y) for x, y in batch)
    ids = torch.full((len(batch), L), PAD, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for b, (x, y) in enumerate(batch):
        s = x + y
        ids[b, :len(s)] = torch.tensor(s)
        lab[b, len(x):len(s)] = torch.tensor(y)
        att[b, :len(s)] = 1
    return ids, lab, att

from __future__ import annotations

import math

import torch
import torch.nn as nn

N_LABELS = 100


def pos_feats(p, k=8):
    f = [p[..., None]]
    for j in range(1, k + 1):
        f += [torch.sin(math.pi * j * p)[..., None], torch.cos(math.pi * j * p)[..., None]]
    return torch.cat(f, -1)


class Head(nn.Module):
    def __init__(self, d, pool="attn", n_out=N_LABELS, dropout=0.1):
        super().__init__()
        self.pool = pool
        self.norm = nn.LayerNorm(d)
        if pool == "attn":
            self.pos = nn.Linear(17, d)
            nn.init.zeros_(self.pos.weight); nn.init.zeros_(self.pos.bias)
            self.score = nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, 1))
        elif pool != "mean":
            raise ValueError(pool)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d, n_out)

    def forward(self, v, cmask, pos):
        h = self.norm(v)
        if self.pool == "attn":
            h = h + self.pos(pos_feats(pos))
            a = self.score(h).squeeze(-1).float().masked_fill(~cmask, -1e4).softmax(-1)
            z = (a[..., None].to(h.dtype) * h).sum(1)
        else:
            m = cmask[..., None].to(h.dtype)
            z = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.out(self.drop(z))


class LSIClassifier(nn.Module):
    def __init__(self, backbone, pool="attn", dropout=0.1, chunk_batch=8):
        super().__init__()
        self.backbone = backbone
        self.head = Head(backbone.d_model, pool, N_LABELS, dropout)
        self.chunk_batch = chunk_batch

    def encode(self, ids, mask):
        out = []
        for i in range(0, ids.shape[0], self.chunk_batch):
            x, m = ids[i:i + self.chunk_batch], mask[i:i + self.chunk_batch]
            h = self.backbone(x, m)
            mf = m.unsqueeze(-1).to(h.dtype)
            out.append((h * mf).sum(1) / mf.sum(1).clamp(min=1))
        return torch.cat(out, 0)

    def forward(self, ids, mask, cmask, pos):
        B, K, L = ids.shape
        real = cmask.view(-1)
        v = torch.zeros(B * K, self.backbone.d_model, device=ids.device,
                        dtype=torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled()
                        else torch.float32)
        if real.any():
            v[real] = self.encode(ids.view(B * K, L)[real], mask.view(B * K, L)[real]).to(v.dtype)
        return self.head(v.view(B, K, -1), cmask, pos)

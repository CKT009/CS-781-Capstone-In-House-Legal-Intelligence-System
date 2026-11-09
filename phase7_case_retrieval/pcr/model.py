from __future__ import annotations

import math

import torch
import torch.nn as nn

POOLS = ("first", "mean", "max", "attn")


def pos_feats(p, k=8):
    f = [p[..., None]]
    for j in range(1, k + 1):
        f += [torch.sin(math.pi * j * p)[..., None], torch.cos(math.pi * j * p)[..., None]]
    return torch.cat(f, -1)


class DocHead(nn.Module):
    def __init__(self, d, pool="mean", proj=True, init_temp=0.05):
        super().__init__()
        if pool not in POOLS:
            raise ValueError(f"pool {pool!r} not in {POOLS}")
        self.pool = pool
        if pool == "attn":
            self.score = nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, 1))
            self.pos = nn.Linear(17, 1)
            for lin in (self.score[2], self.pos):
                nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
        self.proj = None
        if proj:
            self.proj = nn.Linear(d, d)
            with torch.no_grad():
                self.proj.weight.copy_(torch.eye(d)); self.proj.bias.zero_()

        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / init_temp)))

    def scale(self):
        return self.logit_scale.exp().clamp(max=100.0)

    def forward(self, v, cm):
        if self.pool == "first":
            z = v[:, 0]
        elif self.pool == "max":
            z = v.masked_fill(~cm[..., None], float("-inf")).amax(1)
        else:
            w = cm[..., None].to(v.dtype)
            if self.pool == "attn":


                p = torch.arange(v.shape[1], device=v.device, dtype=torch.float32) / 16
                a = (self.score(v).squeeze(-1).float() + self.pos(pos_feats(p)).squeeze(-1))
                w = a.masked_fill(~cm, -1e4).softmax(-1)[..., None].to(v.dtype)
                z = (w * v).sum(1)
            else:
                z = (v * w).sum(1) / w.sum(1).clamp(min=1)
        return self.proj(z) if self.proj is not None else z


class DocEncoder(nn.Module):
    def __init__(self, backbone, pool="mean", proj=True, chunk_batch=16):
        super().__init__()
        self.backbone = backbone
        self.head = DocHead(backbone.d_model, pool, proj)
        self.chunk_batch = chunk_batch

    def _encode(self, ids, mask):
        out = []
        for i in range(0, ids.shape[0], self.chunk_batch):
            x, m = ids[i:i + self.chunk_batch], mask[i:i + self.chunk_batch]
            h = self.backbone(x, m)
            mf = m.unsqueeze(-1).to(h.dtype)
            out.append((h * mf).sum(1) / mf.sum(1).clamp(min=1))
        return torch.cat(out, 0)

    def chunk_vectors(self, ids, mask, cmask):
        B, K, L = ids.shape
        real = cmask.reshape(-1)
        dt = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else torch.float32
        v = torch.zeros(B * K, self.backbone.d_model, device=ids.device, dtype=dt)
        if real.any():
            v[real] = self._encode(ids.reshape(B * K, L)[real],
                                   mask.reshape(B * K, L)[real]).to(dt)
        return v.view(B, K, -1)

    def forward(self, ids, mask, cmask):
        return self.head(self.chunk_vectors(ids, mask, cmask), cmask)

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cjpe.metrics import best_threshold, prf, report, write_submission

EMB = Path(__file__).resolve().parent.parent / "artifacts" / "emb"
ROOT = Path(__file__).resolve().parent.parent
HEADS = ["first", "middle", "last", "mean", "mean_last4", "attn", "transformer"]
MAX_CHUNKS = 32


def load_split(tag, split):
    z = np.load(EMB / tag / f"{split}.npz")
    vec = torch.from_numpy(z["vec"].astype(np.float32))
    doc, pos = z["doc"], z["pos"]
    n = len(z["ids"])
    starts = np.searchsorted(doc, np.arange(n))
    ends = np.searchsorted(doc, np.arange(n), side="right")
    return {"vec": vec, "pos": torch.from_numpy(pos), "s": starts, "e": ends,
            "ids": list(z["ids"]), "y": torch.from_numpy(z["label"].astype(np.int64))}


def fixed_features(S, head):
    out = []
    for s, e in zip(S["s"], S["e"]):
        v = S["vec"][s:e]
        if head == "first":
            out.append(v[0])
        elif head == "last":
            out.append(v[-1])
        elif head == "middle":
            p = S["pos"][s:e]
            out.append(v[int(torch.argmin((p - 0.5).abs()))])
        elif head == "mean":
            out.append(v.mean(0))
        elif head == "mean_last4":
            out.append(v[-4:].mean(0))
    return torch.stack(out)


def padded(S):
    n, d = len(S["s"]), S["vec"].shape[1]
    X = torch.zeros(n, MAX_CHUNKS, d)
    M = torch.zeros(n, MAX_CHUNKS, dtype=torch.bool)
    P = torch.zeros(n, MAX_CHUNKS)
    for i, (s, e) in enumerate(zip(S["s"], S["e"])):
        s = max(s, e - MAX_CHUNKS)
        k = e - s
        X[i, :k] = S["vec"][s:e]
        M[i, :k] = True
        P[i, :k] = S["pos"][s:e]
    return X, M, P


class Linear(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(0.1)
        self.out = nn.Linear(d, 1)

    def forward(self, x, m=None, p=None):
        return self.out(self.drop(self.norm(x))).squeeze(-1)


def pos_feats(p, k=8):
    f = [p[..., None]]
    for j in range(1, k + 1):
        f += [torch.sin(math.pi * j * p)[..., None], torch.cos(math.pi * j * p)[..., None]]
    return torch.cat(f, -1)


class Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.pos = nn.Linear(17, d)
        self.score = nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, 1))
        self.drop = nn.Dropout(0.1)
        self.out = nn.Linear(d, 1)

    def forward(self, x, m, p):
        h = self.norm(x) + self.pos(pos_feats(p))
        a = self.score(h).squeeze(-1).masked_fill(~m, -1e4).softmax(-1)
        return self.out(self.drop((a[..., None] * h).sum(1))).squeeze(-1)


class ChunkTransformer(nn.Module):
    def __init__(self, d, layers=2, heads=8, dim=256):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, dim))
        self.pos = nn.Linear(17, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, 0.1, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, x, m, p):
        h = self.inp(x) + self.pos(pos_feats(p))
        h = torch.cat([self.cls.expand(len(h), -1, -1), h], 1)
        m = torch.cat([torch.ones(len(m), 1, dtype=torch.bool, device=m.device), m], 1)
        return self.out(self.enc(h, src_key_padding_mask=~m)[:, 0]).squeeze(-1)


def train_head(head, tag, epochs=30, lr=1e-3, seed=0, device="cpu", verbose=False):
    torch.manual_seed(seed)
    tr, dv = load_split(tag, "train"), load_split(tag, "dev")
    d = tr["vec"].shape[1]
    if head in ("attn", "transformer"):
        Xtr, Mtr, Ptr = padded(tr)
        Xdv, Mdv, Pdv = padded(dv)
        model = (Attn(d) if head == "attn" else ChunkTransformer(d)).to(device)
    else:
        Xtr, Xdv = fixed_features(tr, head), fixed_features(dv, head)
        Mtr = Ptr = Mdv = Pdv = None
        model = Linear(d).to(device)
    ytr = tr["y"].float()
    pw = torch.tensor((ytr == 0).sum() / (ytr == 1).sum())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n, B = len(ytr), 256
    steps = epochs * math.ceil(n / B)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)

    def run(X, M, P, idx):
        a = [X[idx].to(device)]
        a += [M[idx].to(device), P[idx].to(device)] if M is not None else [None, None]
        return model(*a)

    @torch.no_grad()
    def predict(X, M, P):
        model.eval()
        out = torch.cat([run(X, M, P, torch.arange(i, min(i + 1024, len(X))))
                         for i in range(0, len(X), 1024)]).sigmoid().cpu().numpy()
        model.train()
        return out

    best = {"score": -1}
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, B):
            idx = perm[i:i + B]
            loss = F.binary_cross_entropy_with_logits(
                run(Xtr, Mtr, Ptr, idx), ytr[idx].to(device), pos_weight=pw.to(device))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        p1 = predict(Xdv, Mdv, Pdv)
        t, s = best_threshold(p1, dv["y"].numpy())
        if s > best["score"]:
            best = {"score": s, "thr": t, "epoch": ep + 1, "p1": p1,
                    "at05": prf((p1 >= 0.5).astype(int), dv["y"].numpy())["macro_f1"],
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        if verbose:
            print(f"  ep {ep+1:2d} dev macro-F1 {s:.4f} @thr {t:.2f}")
    model.load_state_dict(best["state"])
    best["model"] = model
    best["params"] = sum(p.numel() for p in model.parameters())
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", choices=HEADS, default="last")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--tag", default="L1024")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sub", action="store_true", help="write subs/frozen_<tag>_<head>.zip")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    res = {}
    for h in (HEADS if a.all else [a.head]):
        b = train_head(h, a.tag, a.epochs, a.lr, a.seed, a.device, a.verbose)
        dv = load_split(a.tag, "dev")
        m = prf((b["p1"] >= b["thr"]).astype(int), dv["y"].numpy())
        print(f"{h:12s} {report(m)}  | @0.5 {b['at05']:.4f}  thr {b['thr']:.2f}  "
              f"ep {b['epoch']}  params {b['params']:,}", flush=True)
        res[h] = {"dev_macro_f1": b["score"], "dev_at_0.5": b["at05"], "thr": b["thr"],
                  "epoch": b["epoch"], "params": b["params"]}
        if a.sub:
            te = load_split(a.tag, "test")
            model = b["model"].eval()
            with torch.no_grad():
                if h in ("attn", "transformer"):
                    X, M, P = padded(te)
                    p1 = model(X.to(a.device), M.to(a.device), P.to(a.device)).sigmoid().cpu().numpy()
                else:
                    p1 = model(fixed_features(te, h).to(a.device)).sigmoid().cpu().numpy()
            pred = {i: int(p >= b["thr"]) for i, p in zip(te["ids"], p1)}
            z = write_submission(pred, te["ids"], ROOT / "subs" / f"frozen_{a.tag}_{h}.zip")
            print(f"  [sub] {z}  pred=1 rate {np.mean(list(pred.values())):.3f}")
    out = ROOT / "runs" / "frozen" / f"{a.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(out.read_text()) if out.exists() else {}
    old.update(res)
    out.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main()

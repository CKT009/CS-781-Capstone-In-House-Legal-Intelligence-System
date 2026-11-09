from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "emb"
BACKBONE = "../checkpoints/B_grown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--L", type=int, default=1024)
    ap.add_argument("--layer", type=int, default=0,
                    help="0 = final normed output; -k = output of block n_layers-k")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--splits", nargs="+", default=["dev", "test", "train"])
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N docs")
    a = ap.parse_args()
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

    import torch
    from cjpe import data as D
    from cjpe.backbones import load_backbone

    bb = load_backbone(BACKBONE, "best.pt", device="cuda").eval()
    hook_out = {}
    if a.layer:
        blocks = [m for m in bb.modules() if m.__class__.__name__ == "Block"]
        blocks[a.layer].register_forward_hook(
            lambda m, i, o: hook_out.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    tag = f"L{a.L}" + (f"_layer{a.layer}" if a.layer else "")
    (OUT / tag).mkdir(parents=True, exist_ok=True)

    for split in a.splits:
        docs = D.load(split)
        toks = D.tokens(split)
        if a.limit:
            docs = docs[:a.limit]
        items = []
        for r, d in enumerate(docs):
            ids = toks[d.id]
            cs = D.chunks(ids, a.L)
            n = max(len(ids), 1)
            step = a.L
            for k, c in enumerate(cs):
                start = min(k * step, max(n - a.L, 0))
                items.append((r, (start + len(c) / 2) / n, c))

        order = np.argsort([-len(c) for _, _, c in items], kind="stable")
        vec = np.zeros((len(items), bb.d_model), dtype=np.float16)
        t0 = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for b in range(0, len(order), a.batch):
                sel = order[b:b + a.batch]
                Lb = max(len(items[i][2]) for i in sel)
                x = torch.zeros(len(sel), Lb, dtype=torch.long)
                m = torch.zeros(len(sel), Lb, dtype=torch.long)
                for j, i in enumerate(sel):
                    c = items[i][2]
                    x[j, :len(c)] = torch.from_numpy(c.astype(np.int64))
                    m[j, :len(c)] = 1
                x, m = x.cuda(), m.cuda()
                h = bb(x, m)
                if a.layer:
                    h = hook_out["h"]
                mf = m.unsqueeze(-1).to(h.dtype)
                v = (h * mf).sum(1) / mf.sum(1).clamp(min=1)
                vec[sel] = v.float().cpu().numpy().astype(np.float16)
        np.savez(OUT / tag / f"{split}.npz", vec=vec,
                 doc=np.array([r for r, _, _ in items], dtype=np.int32),
                 pos=np.array([p for _, p, _ in items], dtype=np.float32),
                 ids=np.array([d.id for d in docs]),
                 label=np.array([-1 if d.label is None else d.label for d in docs], dtype=np.int8))
        print(f"[embed] {tag}/{split}: {len(docs):,} docs, {len(items):,} chunks, "
              f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

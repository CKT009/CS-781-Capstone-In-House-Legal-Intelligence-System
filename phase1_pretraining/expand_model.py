from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path

import torch

from config import Config
from models import build

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--src", default="runs/A_long/best.pt")
ap.add_argument("--out", default="runs/A_grown")
ap.add_argument("--layers", type=int, default=20)
ap.add_argument("--check", action="store_true", help="verify only, write nothing")
ap.add_argument("--batch", type=int, default=4)
a = ap.parse_args()

src = Path(a.src)
old_cfg = Config.load(src.parent / "config.json")
print(f"[src] {src}  {old_cfg.arch}  d_model={old_cfg.d_model} layers={old_cfg.n_layers} "
      f"tie={old_cfg.tie_embeddings}  {old_cfg.n_params()/1e6:.1f}M")

n_old, n_new = old_cfg.n_layers, a.layers
n_add = n_new - n_old
if n_add <= 0:
    raise SystemExit(f"--layers {n_new} must exceed the source's {n_old}")

new_cfg = Config(**{**json.loads((src.parent / "config.json").read_text()),
                    "name": Path(a.out).name, "n_layers": n_new,
                    "tie_embeddings": False, "init_from": ""})
print(f"[dst] {new_cfg.arch}  d_model={new_cfg.d_model} layers={n_new} "
      f"tie=False  {new_cfg.n_params()/1e6:.1f}M  (+{n_add} blocks, untied head)")


after = [round((j + 1) * n_old / (n_add + 1)) for j in range(n_add)]
plan, oi = [], 0
for pos in range(n_new):
    if oi < n_old and (len(plan) - oi) < n_add and oi in after and plan.count(None) < n_add \
       and sum(1 for p in plan if p is None) < len([x for x in after if x <= oi]):
        plan.append(None)
    else:
        plan.append(oi); oi += 1

if sum(1 for p in plan if p is None) != n_add or oi != n_old:
    plan, oi, ins = [], 0, set(after)
    for _ in range(n_new):
        if oi in ins and oi < n_old:
            plan.append(None); ins.discard(oi)
        else:
            plan.append(oi); oi += 1
    while sum(1 for p in plan if p is None) < n_add:
        plan.append(None)
    plan = plan[:n_new]
assert sum(1 for p in plan if p is None) == n_add, plan
assert [p for p in plan if p is not None] == list(range(n_old)), plan
print(f"[plan] new stack: " + " ".join("NEW" if p is None else str(p) for p in plan))


state = torch.load(src, map_location="cpu", weights_only=False)
old = build(old_cfg); old.load_state_dict(state["model"]); old.eval()
new = build(new_cfg).eval()


BLK = "stack.blocks."
inv = {srci: dst for dst, srci in enumerate(plan) if srci is not None}
old_sd, mapped = old.state_dict(), {}
for k, v in old_sd.items():
    if k.startswith(BLK):
        i, tail = k[len(BLK):].split(".", 1)
        mapped[f"{BLK}{inv[int(i)]}.{tail}"] = v
    else:
        mapped[k] = v
missing, unexpected = new.load_state_dict(mapped, strict=False)
fresh = [dst for dst, srci in enumerate(plan) if srci is None]


stray = [k for k in missing
         if not any(k.startswith(f"{BLK}{d}.") for d in fresh)]
if stray:
    raise SystemExit(f"these tensors got no value and are not in a fresh block: {stray[:8]}")
if unexpected:
    raise SystemExit(f"source had tensors the grown model has no slot for: {unexpected[:8]}")

head = next((h for h in ("lm_head", "mlm_head") if hasattr(new, h)), None)
if head is None:
    raise SystemExit(f"no output head found on {type(new).__name__}")
with torch.no_grad():
    getattr(new, head).weight.copy_(getattr(old, head).weight)
assert getattr(new, head).weight.data_ptr() != new.stack.embed_tokens.weight.data_ptr(), \
    f"{head} is still tied to the embedding -- tie_embeddings did not take effect"

for dst in fresh:
    with torch.no_grad():
        new.stack.blocks[dst].attn.o_proj.weight.zero_()
        new.stack.blocks[dst].mlp.down_proj.weight.zero_()
print(f"[fill] copied {n_old} trained blocks via state-dict remap (head={head}); "
      f"zero-initialised {fresh} as identity")


torch.manual_seed(0)
ok = True
for L in (16, 128, 512):
    x = torch.randint(0, old_cfg.vocab_size, (a.batch, L))
    with torch.no_grad():
        d = (old(x)["logits"] - new(x)["logits"]).abs().max().item()
    flag = "ok" if d < 1e-5 else "MISMATCH"
    ok &= d < 1e-5
    print(f"[verify] seq_len {L:4d}: max |logit difference| = {d:.3e}   {flag}")
if not ok:
    raise SystemExit("expanded model is NOT equivalent to the source -- refusing to write")
print(f"[verify] the {new_cfg.n_params()/1e6:.1f}M model computes exactly what "
      f"{src} computes.")

if a.check:
    print("\n--check: nothing written.")
    raise SystemExit(0)

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
torch.save({"model": new.state_dict(), "cfg": new_cfg, "step": 0, "tokens": 0,
            "grown_from": str(src), "plan": plan, "fresh_blocks": fresh},
           out / "init.pt")
new_cfg.save()
_LETTER = {"decoder": "A", "encoder": "B", "mamba": "M", "diffusion": "F"}
print(f"""
[ok] {out/'init.pt'}   {new_cfg.n_params()/1e6:.1f}M parameters, {n_new} layers, untied head

Train it (pick a free GPU):

  python train_continued.py --model {_LETTER.get(new_cfg.arch, '?')} --gpu <N> \\
      --name {out.name} --init {out}/init.pt --steps 40000

Judge it on the task metric, never on val loss -- the shard set changed, and its
held-out slice is one court-year cluster rather than a representative sample.
""")

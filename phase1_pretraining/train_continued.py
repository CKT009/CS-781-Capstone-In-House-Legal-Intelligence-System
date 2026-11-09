from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

SHARDS = "artifacts/shards_full"


SRC = {"A": ("runs/A_long/best.pt",  1.5e-4),
       "M": ("runs/M_mamba/best.pt", 3.0e-4),


       "B": ("runs/B_encoder/best.pt", 1.0e-4)}

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", choices=["A", "M", "B"])
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--pack", action="store_true", help="build the full-corpus shards and exit")
ap.add_argument("--budget", type=float, default=1e10,
                help="per-run token cap before the mixture split; 1e10 exhausts every source")
ap.add_argument("--steps", type=int, default=40000,
                help="40,000 x 65,536 = 2.62B tokens, ~20 h at 37.1k tok/s")
ap.add_argument("--lr", type=float, default=None, help="default: half the original peak")
ap.add_argument("--warmup", type=int, default=1000)
ap.add_argument("--decay-frac", type=float, default=0.25)
ap.add_argument("--seed", type=int, default=90071,
                help="must differ from 1337, which both originals used")
ap.add_argument("--name", default=None)
ap.add_argument("--init", default=None,
                help="checkpoint to warm-start from; default is the model's original run. "
                     "Use runs/A_grown/init.pt for the depth-expanded 150M model, whose "
                     "config.json carries n_layers=20 and tie_embeddings=False.")
ap.add_argument("--minutes", type=float, default=0.0)
a = ap.parse_args()

if a.pack:
    print(f"packing the FULL corpus -> {SHARDS}  (~4.13B tokens, 8.2 GB)\n")
    sys.exit(subprocess.run([sys.executable, "run.py", "pack",
                             "--budget", repr(a.budget), "--out", SHARDS]).returncode)

if not a.model:
    sys.exit("pass --model A or --model M (or --pack first)")
if not Path(SHARDS, "index.json").exists():
    sys.exit(f"{SHARDS} not built yet -- run: python train_continued.py --pack")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

import json
from config import Config
from train import train

init, peak = SRC[a.model]
if a.init:
    init = a.init
if not Path(init).exists():
    sys.exit(f"missing {init}")
assert a.seed != 1337, "both originals used 1337; use a different seed"
name = a.name or f"{a.model}_full"
assert name not in ("A_long", "M_mamba"), "that would overwrite a submitted checkpoint"

old = json.loads(Path(init).parent.joinpath("config.json").read_text())
cfg = Config(**{**old,
    "name": name,
    "init_from": init,
    "shard_dir": SHARDS,
    "out_dir": "runs",
    "max_steps": a.steps,
    "warmup": a.warmup,
    "schedule": "wsd",
    "decay_frac": a.decay_frac,
    "lr": a.lr if a.lr else peak / 2,
    "min_lr_ratio": 0.1,
    "seed": a.seed,
    "eval_every": 500,
    "save_every_min": 20,
    "log_every": 20,
    "max_minutes": a.minutes})

idx = json.loads(Path(SHARDS, "index.json").read_text())
tot = idx["total_tokens"]
print(f"""
{'=' * 78}
  {cfg.name}   ->  runs/{cfg.name}        ({Path(init).parent.name} is NOT written)
{'=' * 78}
  from        {init}
  corpus      {tot/1e9:.2f}B tokens  (old shard set: 1.33B -- {tot/1.327e9:.1f}x)
  this run    {cfg.max_steps:,} steps = {cfg.tokens_per_step*cfg.max_steps/1e9:.2f}B tokens
              = {cfg.tokens_per_step*cfg.max_steps/tot:.2f} epochs over the full pool
              ~{cfg.tokens_per_step*cfg.max_steps/37100/3600:.0f} h at 37.1k tok/s
  lr          warm {cfg.warmup} -> {cfg.lr:.2e} (half the original {peak:.1e}),
              decay last {cfg.decay_frac:.0%} -> {cfg.lr*cfg.min_lr_ratio:.2e}
  seed        {cfg.seed}
{'=' * 78}
  val_loss here is NOT comparable to the old runs -- different held-out text.
  Judge on the task metric:  python heldout.py --run runs/{cfg.name}
{'=' * 78}
""")

train(cfg)

print(f"""
done -> runs/{cfg.name}/best.pt

  python tune_gen.py --run runs/{cfg.name} --draws 3 --axes top_k --values top_k=50

Compare against the originals on the SAME proxy: A_long 0.8342, M_mamba 0.8325
(baseline decoding config). Anything under +0.0008 is inside the measured noise floor.
""")

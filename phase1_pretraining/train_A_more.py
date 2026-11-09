from __future__ import annotations
import argparse
import os
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--epochs", type=float, default=2.0,
                help="passes over the 54.6M unseen tokens (1 pass = 832 steps, ~25 min)")
ap.add_argument("--steps", type=int, default=0, help="override --epochs with a step count")
ap.add_argument("--lr", type=float, default=3e-5,
                help="peak after re-warm. A_long ENDED at 1.5e-5. Much above ~5e-5 risks "
                     "the re-warm loss spike Ibrahim et al. 2024 document")
ap.add_argument("--warmup", type=int, default=100)
ap.add_argument("--decay-frac", type=float, default=0.6)
ap.add_argument("--seed", type=int, default=20261006)
ap.add_argument("--name", default="A_more")
ap.add_argument("--init", default="runs/A_long/best.pt")
ap.add_argument("--minutes", type=float, default=0.0)
ap.add_argument("--all-data", action="store_true",
                help="ignore the unseen set and sample the whole split (the plain "
                     "more-steps run, for contrast)")
ap.add_argument("--build", action="store_true",
                help="recompute artifacts/fresh_windows.npy and exit")
a = ap.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

import numpy as np
import train as T
from config import Config, PRESETS
from dataset import ShardDataset, Sampler

WIN = Path("artifacts/fresh_windows.npy")
BLOCK, BATCH = 1024, 64


def build_unseen(shard_dir="artifacts/shards", seen_seed=1337, seen_steps=40000):
    idx = ("index_train.json" if Path(shard_dir, "index_train.json").exists()
           else "index.json")
    ds = ShardDataset(shard_dir, index_file=idx)
    N = ds.n_tokens
    ms_ = max(1, N - BLOCK - 1)
    print(f"[build] train split {N:,} tokens; replaying seed {seen_seed} "
          f"x {seen_steps:,} steps")
    offs = np.empty(seen_steps * BATCH, dtype=np.int64)
    for st in range(seen_steps):
        r = np.random.default_rng((seen_seed * 1_000_003 + st) % (2**63 - 1))
        offs[st*BATCH:(st+1)*BATCH] = r.integers(0, ms_, size=BATCH, dtype=np.int64)
    offs.sort()
    ends = offs + BLOCK
    ks, ke = [offs[0]], [ends[0]]
    for s, e in zip(offs[1:], ends[1:]):
        if s <= ke[-1]:
            ke[-1] = max(ke[-1], e)
        else:
            ks.append(s); ke.append(e)
    ms, me = np.array(ks), np.array(ke)
    cov = (me - ms).sum()
    gs = np.concatenate(([0], me)); ge = np.concatenate((ms, [N]))
    keep = (ge - gs) >= (BLOCK + 1)
    w = np.concatenate([np.arange(x, y - BLOCK, BLOCK, dtype=np.int64)
                        for x, y in zip(gs[keep], ge[keep])])
    print(f"[build] seen {cov/N*100:.1f}% | unseen {(N-cov)/1e6:.0f}M tok | "
          f"{len(w):,} clean windows = {len(w)*BLOCK/1e6:.1f}M tok")
    WIN.parent.mkdir(parents=True, exist_ok=True)
    np.save(WIN, w)
    return w


if a.build:
    build_unseen()
    raise SystemExit(0)


class FreshSampler(Sampler):
    windows: np.ndarray | None = None
    train_tokens: int | None = None

    def _mine(self):
        return (FreshSampler.windows is not None
                and self.ds.n_tokens == FreshSampler.train_tokens
                and self.seed != 999)

    def batch(self, step: int):
        if not self._mine():
            return super().batch(step)
        w = FreshSampler.windows
        k = (step * self.bs) % len(w)
        take = np.arange(k, k + self.bs) % len(w)
        offs = w[take]
        chunk = np.stack([self.ds.read(int(o), self.block + 1) for o in offs])
        return chunk[:, :-1], chunk[:, 1:]


if not a.all_data:
    if not WIN.exists():
        print(f"[fresh] {WIN} missing -- building it now (~2 min)")
        build_unseen()
    w = np.load(WIN)
    idx = ("index_train.json" if Path("artifacts/shards/index_train.json").exists()
           else "index.json")
    FreshSampler.train_tokens = ShardDataset("artifacts/shards", index_file=idx).n_tokens
    FreshSampler.windows = np.random.default_rng(a.seed).permutation(w)
    T.Sampler = FreshSampler
    per_epoch = len(w) // BATCH
    steps = a.steps or max(1, int(round(a.epochs * per_epoch)))
    print(f"[fresh] {len(w):,} unseen windows ({len(w)*BLOCK/1e6:.1f}M tok) | "
          f"1 epoch = {per_epoch:,} steps | running {steps:,}")
else:
    steps = a.steps or 5000
    print(f"[all-data] whole split, seed {a.seed}; {steps:,} steps")

assert a.seed != 1337, "seed 1337 replays A_decoder's batches -- use another"
assert a.name != "A_long", "would overwrite the 0.8364 checkpoint"

cfg = Config(**{**PRESETS["A"],
    "name": a.name, "init_from": a.init,
    "shard_dir": "artifacts/shards", "tokenizer_dir": "artifacts/tokenizer_16k",
    "out_dir": "runs", "block_size": BLOCK, "batch_size": BATCH, "grad_accum": 1,
    "max_steps": steps, "warmup": a.warmup, "schedule": "wsd",
    "decay_frac": a.decay_frac, "lr": a.lr, "min_lr_ratio": 0.05,
    "amp": "bf16", "grad_checkpoint": True,
    "eval_every": 100, "save_every_min": 10, "log_every": 10,
    "max_minutes": a.minutes})

print(f"""
{'='*76}
  {cfg.name}  ->  runs/{cfg.name}        (runs/A_long is NOT written)
{'='*76}
  from       {a.init}
  data       {'WHOLE SPLIT' if a.all_data else '54.6M tokens A_long provably never read'}
  steps      {cfg.max_steps:,} = {cfg.tokens_per_step*cfg.max_steps/1e6:.0f}M tok, ~{cfg.tokens_per_step*cfg.max_steps/37100/60:.0f} min
  lr         warm {cfg.warmup} -> {cfg.lr:.1e}, decay last {cfg.decay_frac:.0%} -> {cfg.lr*cfg.min_lr_ratio:.1e}
  seed       {cfg.seed}
{'='*76}
  A_long: val 1.4771, BERTScore 0.8364. Judge this on BERTScore, not val loss --
  Section 14 is the whole reason: loss moved the wrong way against the metric six
  times out of six. Val loss here is also NOT comparable to A_long's in the usual
  way: this trains on a subset A_long never saw, so a drop is partly just coverage.
{'='*76}
""")

T.train(cfg)

print(f"""
done -> runs/{cfg.name}/best.pt

1) proxy FIRST, it is free and it predicted every official score today to 0.002:
   python tune_gen.py --run runs/{cfg.name} --draws 3 --axes top_k --values top_k=50

   A_long scores 0.8364 on the leaderboard. If the proxy does not clear its A_long
   number by more than the 0.0004 noise floor, stop and submit nothing.

2) only if it clears, generate:
   python run.py predict --model A --ckpt runs/{cfg.name}/best.pt \\
       --no-repeat-ngram 0 --rep-penalty 1.0 --temperature 0.8 --top-p 0.92 --top-k 50
   cp runs/A_decoder/submission/submission.zip subs/{cfg.name}_tuned.zip

3) the ensemble is your best score (0.8372), so also try swapping this in for A_long:
   python ensemble_gen.py --run-a runs/{cfg.name} --run-b runs/M_mamba --sweep
""")

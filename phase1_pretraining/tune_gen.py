from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--run", default="runs/M_mamba")
ap.add_argument("--ckpt", default="best.pt")
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--heldout", default="data/heldout_phase1.json")
ap.add_argument("--tokenizer", default="artifacts/tokenizer_16k")
ap.add_argument("--n", type=int, default=174, help="proxy examples; SE ~0.0023 at 174")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--draws", type=int, default=4, help="seeds for the noise floor")
ap.add_argument("--max-new-tokens", type=int, default=220)
ap.add_argument("--axes", nargs="*", default=None)
ap.add_argument("--values", nargs="*", default=[], metavar="AXIS=V1,V2")
ap.add_argument("--base", nargs="*", default=[], metavar="AXIS=V")
ap.add_argument("--out", default=None)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

import torch
sys.path.insert(0, str(Path(__file__).parent))
from config import Config
from dataset import load_tokenizer
from generate import generate, truncate_words, strip_echo
from metrics import bertscore, rouge_l, bleu, distinct_n
from models import build

TARGET_WORDS = 128

AXES = {
    "no_repeat_ngram":    [0, 3, 4, 5, 6],
    "repetition_penalty": [1.0, 1.05, 1.15, 1.3],
    "temperature":        [0.6, 0.7, 0.8, 0.9, 1.0],
    "top_p":              [0.85, 0.92, 0.97, 1.0],
    "top_k":              [0, 20, 50, 100],
}
BASE = dict(no_repeat_ngram=4, repetition_penalty=1.15,
            temperature=0.8, top_p=0.92, top_k=50)


def _coerce(axis, s):
    proto = AXES[axis][0]
    return type(proto)(float(s))


for spec in a.values:
    k, _, v = spec.partition("=")
    if k not in AXES:
        sys.exit(f"--values: unknown axis {k!r}; choose from {list(AXES)}")
    AXES[k] = [_coerce(k, x) for x in v.split(",")]
for spec in a.base:
    k, _, v = spec.partition("=")
    if k not in BASE:
        sys.exit(f"--base: unknown axis {k!r}; choose from {list(BASE)}")
    BASE[k] = _coerce(k, v)


run = Path(a.run)
cfg = Config.load(run / "config.json")
p = Path(a.ckpt) if Path(a.ckpt).exists() else run / a.ckpt
if not p.exists():
    sys.exit(f"no checkpoint at {a.ckpt!r} or {run / a.ckpt}")
st = torch.load(p, map_location="cuda", weights_only=False)
model = build(cfg).cuda().eval()
model.load_state_dict(st["model"])
tok = load_tokenizer(a.tokenizer)
print(f"[tune] {p} @ step {st['step']:,}  arch={cfg.arch}")

data = json.loads(Path(a.heldout).read_text())
keys = list(data)[:a.n]
prompts = [data[k]["prompt"] for k in keys]
golds = [data[k]["gold"] for k in keys]
enc = [tok.encode(x).ids[-cfg.max_seq_len:] for x in prompts]
print(f"[tune] {len(keys)} examples | ctx {cfg.max_seq_len} | "
      f"{a.max_new_tokens} new tokens")


@torch.no_grad()
def run_setting(s: dict, seed: int) -> dict:
    t0 = time.time()
    preds = []
    for i in range(0, len(enc), a.batch):
        chunk = enc[i:i + a.batch]
        L = max(len(e) for e in chunk)


        x = torch.zeros(len(chunk), L, dtype=torch.long)
        for j, e in enumerate(chunk):
            x[j, L - len(e):] = torch.tensor(e)
        torch.manual_seed(seed * 100003 + i)
        out = generate(model, x.cuda(), max_new_tokens=a.max_new_tokens, **s)
        for j in range(len(chunk)):
            txt = truncate_words(
                strip_echo(prompts[i + j], tok.decode(out[j].tolist())), TARGET_WORDS)
            preds.append(txt if txt.strip() else ".")
    gen_s = time.time() - t0

    bs = bertscore(preds, golds)
    if bs.get("bertscore_f1") is None:
        sys.exit(f"bert_score failed: {bs.get('error')}")
    n = len(preds)
    return {**s, "seed": seed, "bertscore_f1": bs["bertscore_f1"],
            "rougeL": sum(rouge_l(q, g) for q, g in zip(preds, golds)) / n,
            "bleu4": bleu(preds, golds),
            "distinct4": sum(distinct_n(q, 4) for q in preds) / n,
            "mean_words": sum(len(q.split()) for q in preds) / n,
            "min_words": min(len(q.split()) for q in preds),
            "gen_s": round(gen_s, 1)}


def show(r, tag=""):
    print(f"    {tag:30s} bert {r['bertscore_f1']:.4f}  rougeL {r['rougeL']:.4f}  "
          f"bleu4 {r['bleu4']:.4f}  words {r['mean_words']:5.1f}/{r['min_words']:3d}  "
          f"d4 {r['distinct4']:.3f}  {r['gen_s']:5.1f}s")


results = []
print(f"\n{'=' * 78}\n  stage 0 -- noise floor: {a.draws} seeds at the SUBMITTED setting"
      f"\n{'=' * 78}\n  {BASE}")
draws = []
for d in range(a.draws):
    r = run_setting(BASE, seed=a.seed + d)
    draws.append(r["bertscore_f1"])
    results.append({**r, "stage": "noise"})
    show(r, f"seed {a.seed + d}")
mean = sum(draws) / len(draws)
sigma = (sum((x - mean) ** 2 for x in draws) / max(len(draws) - 1, 1)) ** 0.5
THRESH = 2 * sigma
print(f"\n  mean {mean:.4f}   sigma {sigma:.4f}   spread {max(draws)-min(draws):.4f}")
print(f"  -> a setting must beat the incumbent by {THRESH:.4f} to be adopted.")

incumbent, best_score = dict(BASE), mean
for axis in (a.axes or list(AXES)):
    if axis not in AXES:
        print(f"\n  [skip] unknown axis {axis!r}")
        continue
    print(f"\n{'=' * 78}\n  axis: {axis}   "
          f"(incumbent {incumbent[axis]!r} @ {best_score:.4f})\n{'=' * 78}")
    winner, winner_score = incumbent[axis], best_score
    for v in AXES[axis]:
        if v == incumbent[axis]:
            print(f"    {axis}={v!r:<26s} incumbent, {best_score:.4f}")
            continue
        r = run_setting({**incumbent, axis: v}, seed=a.seed)
        results.append({**r, "stage": axis})
        show(r, f"{axis}={v!r}")
        if r["bertscore_f1"] > winner_score:
            winner, winner_score = v, r["bertscore_f1"]
    gain = winner_score - best_score
    if winner != incumbent[axis] and gain > THRESH:
        print(f"    -> ADOPT {axis}={winner!r}  (+{gain:.4f} > {THRESH:.4f})")
        incumbent[axis], best_score = winner, winner_score
    else:
        print(f"    -> keep {axis}={incumbent[axis]!r}"
              + (f": best alternative gained only +{gain:.4f}, inside noise"
                 if winner != incumbent[axis] else ""))

out = a.out or f"{run}/decoding_sweep.json"
Path(out).write_text(json.dumps(
    {"ckpt": str(p), "step": st["step"], "n": a.n, "sigma": sigma,
     "threshold": THRESH, "baseline": BASE, "chosen": incumbent,
     "baseline_score": mean, "chosen_score": best_score, "results": results}, indent=2))

print(f"\n{'=' * 78}\n  RANKED\n{'=' * 78}")
for r in sorted(results, key=lambda r: -r["bertscore_f1"])[:10]:
    print(f"  {r['bertscore_f1']:.4f}  ngram={r['no_repeat_ngram']:<2} "
          f"rp={r['repetition_penalty']:<5} T={r['temperature']:<4} "
          f"p={r['top_p']:<5} k={r['top_k']:<4} [{r['stage']}]")
print(f"\n  submitted setting: {mean:.4f}")
print(f"  chosen:            {best_score:.4f}   ({best_score - mean:+.4f})")
if best_score - mean <= THRESH:
    print("\n  NULL RESULT -- nothing beat the submitted setting by more than the noise "
          "floor. That is a finding: decoding is not where the remaining score is, and "
          "it is now measured with PAIRED draws rather than inferred from an "
          "underpowered unpaired grid.")
else:
    print(f"\n  Regenerate with:  python run.py predict --model <key> "
          + " ".join(f"--{k.replace('_','-')} {v}" for k, v in incumbent.items()
                     if k != "no_repeat_ngram"))
    print(f"  (no_repeat_ngram={incumbent['no_repeat_ngram']} is NOT exposed by "
          f"run.py predict -- see the note in generate.py)")
print(f"\n  -> {out}")

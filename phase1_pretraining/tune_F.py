from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--run", default="runs/F_diffusion")
ap.add_argument("--ckpt", default="best.pt")
ap.add_argument("--gpu", type=int, default=2, help="idle card; 0 and 1 may be training")
ap.add_argument("--heldout", default="data/heldout_phase1.json")
ap.add_argument("--tokenizer", default="artifacts/tokenizer_16k")
ap.add_argument("--n", type=int, default=100, help="examples; SE ~ 0.003 at 100, "
                                                  "~0.0016 at 350 (Section 14)")
ap.add_argument("--batch", type=int, default=25)
ap.add_argument("--draws", type=int, default=4, help="seeds for the noise floor")
ap.add_argument("--axes", nargs="*", default=None,
                help="subset of: steps schedule gumbel temperature top_p top_k")
ap.add_argument("--values", nargs="*", default=[], metavar="AXIS=V1,V2",
                help="override an axis grid, e.g. gumbel=2,3,4,6 temperature=1.0,1.2. "
                     "Use when a sweep's winner sits at the EDGE of its grid -- a "
                     "monotone trend to the boundary means the optimum is outside it.")
ap.add_argument("--base", nargs="*", default=[], metavar="AXIS=V",
                help="override the starting point, e.g. schedule=linear gumbel=2.0")
ap.add_argument("--out", default="runs/F_decoding_sweep.json")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

import torch
sys.path.insert(0, str(Path(__file__).parent))
from config import Config
from dataset import load_tokenizer
from generate import generate_diffusion, truncate_words, strip_echo
from models import build
from metrics import bertscore, rouge_l, bleu, distinct_n

TARGET_WORDS = 128


AXES = {
    "steps":       [8, 16, 32, 64, 128, 256],
    "schedule":    ["cosine", "linear"],
    "gumbel":      [0.0, 0.5, 1.0, 2.0],
    "temperature": [0.6, 0.7, 0.8, 0.9, 1.0],
    "top_p":       [0.85, 0.92, 0.97, 1.0],
    "top_k":       [0, 20, 50, 100],
}


run = Path(a.run)
cfg = Config.load(run / "config.json")
st = torch.load(run / a.ckpt, map_location="cuda", weights_only=False)
model = build(cfg).cuda().eval()
model.load_state_dict(st["model"])
tok = load_tokenizer(a.tokenizer)
_vl = (st.get("best") or {}).get("val_loss")
print(f"[tune] {run / a.ckpt} @ step {st['step']:,}  "
      f"(val_loss {_vl:.4f})" if isinstance(_vl, float) else
      f"[tune] {run / a.ckpt} @ step {st['step']:,}")

data = json.loads(Path(a.heldout).read_text())
keys = list(data)[:a.n]
prompts = [data[k]["prompt"] for k in keys]
golds = [data[k]["gold"] for k in keys]

S = cfg.block_size - cfg.diff_target_len
enc = [tok.encode(p).ids[-S:] for p in prompts]
print(f"[tune] {len(keys)} examples | prompt budget {S} tok | "
      f"target {cfg.diff_target_len} tok")

def _coerce(axis, s, check=True):
    proto = AXES[axis][0]
    if isinstance(proto, str):
        if check and s not in AXES[axis]:
            raise SystemExit(f"{axis}={s!r} is not one of {AXES[axis]}. "
                             f"(Separate items with SPACES, not commas: "
                             f"--base schedule=linear gumbel=2.0)")
        return s
    try:
        return type(proto)(float(s))
    except ValueError:
        raise SystemExit(f"{axis}={s!r} is not numeric. (Separate items with SPACES, "
                         f"not commas: --values gumbel=2,4 temperature=1.0,1.2)")


for spec in a.values:
    k, _, v = spec.partition("=")
    if k not in AXES:
        raise SystemExit(f"--values: unknown axis {k!r}; choose from {list(AXES)}")
    AXES[k] = [_coerce(k, x) for x in v.split(",")]

BASE = dict(steps=cfg.diff_steps, schedule="cosine", gumbel=1.0,
            temperature=0.8, top_p=0.92, top_k=50)
for spec in a.base:
    k, _, v = spec.partition("=")
    if k not in BASE:
        raise SystemExit(f"--base: unknown axis {k!r}; choose from {list(BASE)}")
    BASE[k] = _coerce(k, v)


@torch.no_grad()
def run_setting(s: dict, seed: int) -> dict:
    t0 = time.time()
    preds = []
    for i in range(0, len(enc), a.batch):
        torch.manual_seed(seed * 100003 + i)
        rows = generate_diffusion(
            model, enc[i:i + a.batch],
            max_new_tokens=cfg.diff_target_len,
            steps=s["steps"], schedule=s["schedule"], gumbel=s["gumbel"],
            temperature=s["temperature"], top_p=s["top_p"], top_k=s["top_k"])
        for j, r in enumerate(rows):
            txt = truncate_words(
                strip_echo(prompts[i + j], tok.decode(r)), TARGET_WORDS)
            preds.append(txt if txt.strip() else ".")
    gen_s = time.time() - t0

    bs = bertscore(preds, golds)
    if bs.get("bertscore_f1") is None:
        raise SystemExit(f"bert_score failed: {bs.get('error')}")
    n = len(preds)
    return {**s, "seed": seed,
            "bertscore_f1": bs["bertscore_f1"],
            "rougeL": sum(rouge_l(p, g) for p, g in zip(preds, golds)) / n,
            "bleu4": bleu(preds, golds),
            "distinct1": sum(distinct_n(p, 1) for p in preds) / n,
            "distinct4": sum(distinct_n(p, 4) for p in preds) / n,
            "mean_words": sum(len(p.split()) for p in preds) / n,
            "min_words": min(len(p.split()) for p in preds),
            "gen_s": round(gen_s, 1)}


def show(r, tag=""):
    print(f"    {tag:34s} bert {r['bertscore_f1']:.4f}  rougeL {r['rougeL']:.4f}  "
          f"bleu4 {r['bleu4']:.4f}  words {r['mean_words']:5.1f}/{r['min_words']:3d}  "
          f"d1 {r['distinct1']:.3f}  {r['gen_s']:5.1f}s")


results = []


print(f"\n{'=' * 78}\n  stage 0 -- noise floor: {a.draws} seeds at the baseline setting\n{'=' * 78}")
print(f"  baseline: {BASE}")
draws = []
for d in range(a.draws):
    r = run_setting(BASE, seed=a.seed + d)
    draws.append(r["bertscore_f1"]); results.append({**r, "stage": "noise"})
    show(r, f"seed {a.seed + d}")

mean = sum(draws) / len(draws)
sigma = (sum((x - mean) ** 2 for x in draws) / max(len(draws) - 1, 1)) ** 0.5
THRESH = 2 * sigma
print(f"\n  mean {mean:.4f}   sigma {sigma:.4f}   spread {max(draws) - min(draws):.4f}")
print(f"  -> a setting must beat the incumbent by {THRESH:.4f} to be adopted.")
print(f"  (Section 12.3 measured sigma ~ 0.0004 at n=1000 for causal sampling; this is")
print(f"   n={a.n} and a different decoder, so it is measured here rather than assumed.)")


incumbent, best_score = dict(BASE), mean
axes = a.axes or list(AXES)
for axis in axes:
    if axis not in AXES:
        print(f"\n  [skip] unknown axis {axis!r}"); continue
    print(f"\n{'=' * 78}\n  axis: {axis}   (incumbent {incumbent[axis]!r} @ {best_score:.4f})\n{'=' * 78}")
    winner, winner_score = incumbent[axis], best_score
    for v in AXES[axis]:
        if v == incumbent[axis]:
            print(f"    {axis}={v!r:<28s} incumbent, {best_score:.4f}")
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
    elif winner != incumbent[axis]:
        print(f"    -> keep {axis}={incumbent[axis]!r}: best alternative gained only "
              f"+{gain:.4f}, inside noise")
    else:
        print(f"    -> keep {axis}={incumbent[axis]!r}")


Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(
    {"ckpt": str(run / a.ckpt), "step": st["step"], "n": a.n,
     "sigma": sigma, "threshold": THRESH, "baseline": BASE,
     "chosen": incumbent, "results": results}, indent=2))

print(f"\n{'=' * 78}\n  RANKED (all settings tried)\n{'=' * 78}")
for r in sorted(results, key=lambda r: -r["bertscore_f1"])[:12]:
    print(f"  {r['bertscore_f1']:.4f}  steps={r['steps']:<4d} {r['schedule']:<7s} "
          f"g={r['gumbel']:<4} T={r['temperature']:<4} p={r['top_p']:<5} "
          f"k={r['top_k']:<4} [{r['stage']}]")

print(f"\n  chosen: {incumbent}")
print(f"  gain over baseline: {best_score - mean:+.4f}  (noise floor {THRESH:.4f})")
if best_score - mean <= THRESH:
    print("\n  NULL RESULT -- no setting beat the baseline by more than the noise floor.")
    print("  That is a finding, not a failure: it is what Section 12.2 found for A, and")
    print("  it means decoding is not where the remaining score is. Report it and move on.")
print(f"\n  cost note: steps is linear in inference time. A +0.001 gain for 2x the")
print(f"  passes is a bad trade across 1,000 test examples -- check the gen_s column.")
print(f"\n  -> {a.out}")
print(f"  then: python predict.py-equivalent via run.py predict, or heldout.py export")
print(f"        with the chosen flags, and confirm on the official set.")

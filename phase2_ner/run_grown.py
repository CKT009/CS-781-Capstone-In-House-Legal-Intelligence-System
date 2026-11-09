from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

BACKBONE = "../checkpoints/A_grown"
TAG = "_g"
EPOCHS = 20


FAST = ["adapter", "lora", "qlora", "distill"]


GRID = {
    "head_only": "linear probe on frozen features -- the floor, and it measures how much "
                 "of the task the Phase-1 pretraining already solved",
    "bitfit":    "biases and norm gains only; no weight matrix moves",
    "last4":     "partial/layer-wise: only the top 4 of 20 blocks adapt",
    "lora":      "low-rank reparameterisation of q_proj/v_proj",
    "qlora":     "the same rank-8 adapter over a 4-bit NF4 base -- paired with lora, it "
                 "isolates the cost of quantising the frozen weights",
    "adapter":   "Houlsby/Pfeiffer bottleneck modules -- added capacity, not reparameterised",
    "prefix":    "learned key/value vectors at every attention layer",
    "prompt":    "learned virtual tokens at the input embedding",
}

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--backbone", default=None,
                help="Phase-1 run to adapt (default: A_grown). Use "
                     "../checkpoints/B_grown for the bidirectional 153M model.")
ap.add_argument("--tag", default=None,
                help="run-name suffix; must differ per backbone or cells collide. "
                     "Defaults to _g, or _bg when --backbone names B_grown.")
ap.add_argument("--fast", action="store_true",
                help=f"only the cells that have ever mattered: {FAST}")
ap.add_argument("--epochs", type=int, default=EPOCHS)
ap.add_argument("--only", nargs="*", default=None,
                help="subset of: " + " ".join(list(GRID) + ["distill"]))
ap.add_argument("--list", action="store_true")
ap.add_argument("--force", action="store_true", help="retrain cells that already exist")
ap.add_argument("--submit", action="store_true",
                help="after the grid, retrain on fold1+fold2 at each honest best epoch "
                     "and write fold3 submissions")
ap.add_argument("--teacher", default=None,
                help="override the distillation teacher (default: best grid cell)")
ap.add_argument("--keep-every", type=int, default=2)
a = ap.parse_args()

if a.backbone:
    BACKBONE = a.backbone
TAG = a.tag or ("_bg" if "B_grown" in BACKBONE else "_g")

PY = sys.executable
cells = a.only or (FAST if a.fast else list(GRID) + ["distill"])
bad = [c for c in cells if c not in GRID and c != "distill"]
if bad:
    sys.exit(f"unknown cell(s) {bad}; choose from {list(GRID) + ['distill']}")


def name_of(exp):
    return f"{exp}{TAG}"


def summary(run):
    p = Path("runs") / run / "summary.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    c = json.loads((Path("runs") / run / "config.json").read_text())
    if set(c.get("train_folds") or []) & set(c.get("dev_folds") or []):
        return {"contaminated": True, **d}
    return d


def sh(cmd, label):
    print(f"\n{'=' * 78}\n  {label}\n  $ {' '.join(cmd)}\n{'=' * 78}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd)
    if r.returncode:
        print(f"[FAIL] {label} exited {r.returncode} -- continuing with the rest",
              flush=True)
        return False
    print(f"[done] {label} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return True


if a.list:
    print(f"\nbackbone: {BACKBONE}")
    bp = Path(BACKBONE) / "config.json"
    if bp.exists():
        c = json.loads(bp.read_text())
        print(f"  arch={c['arch']} d_model={c['d_model']} layers={c['n_layers']} "
              f"tie_embeddings={c.get('tie_embeddings')}")
    print(f"\n{'cell':12s}{'run dir':16s}{'exists':>8s}  why")
    print("-" * 100)
    for exp, why in GRID.items():
        s = summary(name_of(exp))
        mark = f"{s['best']['macro_f1']:.4f}" if s and "best" in s else "-"
        print(f"{exp:12s}{'runs/'+name_of(exp):16s}{mark:>8s}  {why}")
    d = summary(f"distill{TAG}_kd")
    dmark = f"{d['best']['macro_f1']:.4f}" if d and "best" in d else "-"
    print(f"{'distill':12s}{'runs/distill' + TAG + '_kd':16s}{dmark:>8s}"
          f"  MANDATORY: teacher = best grid cell, student = every {a.keep_every}nd block")
    print(f"\n{len(cells)} cell(s) selected, {a.epochs} epochs each, train fold1 / dev fold2")
    print("nothing was trained.")
    sys.exit(0)

print(f"[plan] backbone {BACKBONE}")
print(f"[plan] cells: {' '.join(cells)}   epochs {a.epochs}   gpu {a.gpu}")
if not Path(BACKBONE, "best.pt").exists():
    sys.exit(f"{BACKBONE}/best.pt not found")


for exp in [c for c in cells if c in GRID]:
    run = name_of(exp)
    if summary(run) and not a.force:
        print(f"[skip] runs/{run} already has a summary (use --force to retrain)")
        continue
    ok = sh([PY, "run_train.py", "--exp", exp, "--name", run, "--gpu", str(a.gpu),
             "--backbone", BACKBONE, "--set", f"epochs={a.epochs}",
             'train_folds=["fold1"]', 'dev_folds=["fold2"]'],
            f"{exp} on A_grown  ({GRID[exp]})")
    if ok:
        s = summary(run)
        if s and s.get("contaminated"):
            sys.exit(f"{run}: train and dev folds overlap -- refusing to continue")


if "distill" in cells:
    done = {e: summary(name_of(e)) for e in GRID}
    scored = {e: s["best"]["macro_f1"] for e, s in done.items() if s and "best" in s}
    teacher = a.teacher or (f"runs/{name_of(max(scored, key=scored.get))}" if scored else None)
    if not teacher:
        print("[distill] no scored grid cell to use as teacher -- skipping")
    elif summary(f"distill{TAG}_kd") and not a.force:
        print(f"[skip] runs/distill{TAG}_kd already exists")
    else:
        print(f"[distill] teacher = {teacher}" + (
            f"  (best grid cell, {max(scored.values()):.4f})" if scored and not a.teacher else ""))
        code = (
            "import sys; sys.path.insert(0,'.')\n"
            "from ner.distill import distill, DistillConfig\n"
            f"distill(DistillConfig(name='distill{TAG}', teacher_run={teacher!r},\n"
            f"    keep_every={a.keep_every}, epochs={a.epochs},\n"
            "    train_folds=('fold1',), dev_folds=('fold2',), control=True))\n")
        sh([PY, "-c", code], f"distillation from {teacher} (+ gold-only control)")


rows = []
for exp in GRID:
    s = summary(name_of(exp))
    if s and "best" in s:
        rows.append((exp, s["best"]["macro_f1"], s["best"]["epoch"], s.get("epochs"),
                     s.get("trainable"), s.get("trainable_pct"), s.get("wall_min")))
for mode in ("kd", "control"):
    s = summary(f"distill{TAG}_{mode}")
    if s and "best" in s:
        rows.append((f"distill:{mode}", s["best"]["macro_f1"], s["best"]["epoch"],
                     s.get("epochs"), s.get("params"), None, s.get("wall_min")))
rows.sort(key=lambda r: -r[1])

print(f"\n{'=' * 84}\n  A_grown (150.4M, causal, 20 layers) -- honest fold1 -> fold2"
      f"\n{'=' * 84}")
print(f"{'method':14s}{'macroF1':>9s}{'best ep':>9s}{'trainable':>13s}{'%':>8s}{'min':>7s}")
print("-" * 84)
for n, f1, ep, tot, tr, pct, mn in rows:
    print(f"{n:14s}{f1:9.4f}{f'{ep}/{tot}':>9s}"
          f"{(format(tr, ',') if tr else '-'):>13s}"
          f"{(format(pct, '.2f') if pct else '-'):>8s}"
          f"{(format(mn, '.0f') if mn else '-'):>7s}")
print("-" * 84)
lo = {n: f for n, f, *_ in rows}
if "lora" in lo and "qlora" in lo:
    print(f"  QLoRA vs LoRA: {lo['qlora'] - lo['lora']:+.4f}  "
          f"(same rank/targets/lr; the only difference is the 4-bit base)")
if "distill:kd" in lo and "distill:control" in lo:
    print(f"  distillation gap: {lo['distill:kd'] - lo['distill:control']:+.4f}  "
          f"(teacher-supervised minus gold-only; THIS is the claim, not the score)")
print("\n  B_encoder (98M, bidirectional) for reference: adapter 0.3738, lora 0.3439, "
      "bitfit 0.3165,\n  last_k 0.3128, prompt 0.2486, prefix 0.2242, head_only 0.2208")
Path("runs").mkdir(exist_ok=True)
Path(f"runs/grown_grid{TAG}.json").write_text(json.dumps(
    {"backbone": BACKBONE, "epochs": a.epochs, "folds": {"train": ["fold1"], "dev": ["fold2"]},
     "rows": [dict(zip(("method", "macro_f1", "best_epoch", "epochs", "trainable",
                        "trainable_pct", "wall_min"), r)) for r in rows]}, indent=2))
print(f"\n-> runs/grown_grid{TAG}.json")


if a.submit:
    print(f"\n{'=' * 78}\n  submissions: retrain on fold1+fold2 at each honest best epoch"
          f"\n{'=' * 78}")
    for exp in [c for c in cells if c in GRID]:
        s = summary(name_of(exp))
        if not (s and "best" in s):
            print(f"[skip] {exp}: no honest result to take an epoch from")
            continue
        ep, run = s["best"]["epoch"], f"{exp}{TAG}_f2"
        if not Path("runs", run, "last.pt").exists() or a.force:
            sh([PY, "run_train.py", "--exp", exp, "--name", run, "--gpu", str(a.gpu),
                "--backbone", BACKBONE, "--set", f"epochs={ep}",
                'train_folds=["fold1","fold2"]', 'dev_folds=["fold2"]'],
               f"{exp} on both folds, {ep} epochs fixed from the honest run")
        sh([PY, "run_predict.py", "--run", f"runs/{run}", "--ckpt", "last.pt",
            "--gpu", str(a.gpu)], f"predict fold3 from runs/{run}/last.pt")
    print("\nThe dev scores those submission runs print are CONTAMINATED by construction "
          "(fold2 is in train). Ignore them; the honest number for each method is the "
          "table above.")

print(f"""
{'=' * 78}
  next
{'=' * 78}
  ensemble on honest dev predictions (free, no GPU) -- span voting was worth
  +0.08 macro-F1 on the B backbone, far more than any single method:

    python ensemble.py --dev runs/lora{TAG} runs/adapter{TAG} runs/qlora{TAG} \\
        runs/last4{TAG} runs/distill{TAG}_kd

  DO NOT mix A_grown cells into the B_encoder ensemble. That was this script's advice
  until it was tested, and it is wrong. The 9-way mix (6 B + adapter_g, qlora_g, lora_g)
  scored the BEST honest dev of any ensemble tried, 0.5059 against ens_crf's 0.4997 --
  and then came FOURTH on the leaderboard, 0.5590 against 0.5732.

  The reason is calibration, not noise. Across four B-family ensembles the dev->test
  offset sat in a +0.074..+0.080 band; the mixed one came in at +0.053. The offset
  measures what the second training fold buys, and the A_grown cells peak at earlier
  epochs and gain less from it, so an offset learned on one member population does not
  transfer to another. A dev gain inside 0.01 is below this proxy's resolution anyway.

  Diversity is still the input where it applies -- two runs of the same method on the
  same backbone scored BELOW a single model -- but it must come from members whose
  dev->test behaviour matches the ensemble you are calibrating against.
""")

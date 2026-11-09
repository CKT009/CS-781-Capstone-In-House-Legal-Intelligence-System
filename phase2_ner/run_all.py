from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = "../checkpoints"
BACKBONES = {
    "B": (f"{ROOT}/B_encoder",   "bidirectional MLM encoder -- a token-level pretraining objective"),
    "A": (f"{ROOT}/A_decoder",   "causal decoder -- no right context"),
    "M": (f"{ROOT}/M_mamba",     "Mamba SSM -- causal, no attention"),
    "F": (f"{ROOT}/F_diffusion", "diffusion encoder -- rank-collapsed, expect ~0.06"),
}


METHODS = {
    "full":      (dict(peft="full",      lr=3e-5), "baseline: everything trains"),
    "head_only": (dict(peft="head_only", lr=3e-5, head_lr=3e-3),
                  "frozen backbone -- the linear-probe floor"),
    "last4":     (dict(peft="last_k",    lr=5e-5, peft_kw=dict(k=4)),
                  "partial/layer-wise: top 4 blocks"),
    "bitfit":    (dict(peft="bitfit",    lr=1e-3), "biases + norm gains, no matrices"),
    "lora":      (dict(peft="lora",      lr=3e-4, peft_kw=dict(r=8, alpha=16)),
                  "low-rank reparameterisation"),
    "adapter":   (dict(peft="adapter",   lr=3e-4, peft_kw=dict(bottleneck=64)),
                  "bottleneck adapter modules"),
    "prefix":    (dict(peft="prefix",    lr=1e-3, peft_kw=dict(n_prefix=16)),
                  "learned KV at every attention layer"),
    "prompt":    (dict(peft="prompt",    lr=1e-3, peft_kw=dict(n_prompt=20)),
                  "learned virtual tokens at the input"),
}
EPOCHS = 20


def cell_overrides(method: str, bb_key: str) -> dict | None:
    over, _ = METHODS[method]
    over = {k: (dict(v) if isinstance(v, dict) else v) for k, v in over.items()}
    over["backbone"] = BACKBONES[bb_key][0]
    if bb_key == "M":
        if method == "prefix":
            return None
        if method == "lora":
            over["peft_kw"]["targets"] = ("in_proj", "out_proj")
    return over


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="*", default=None, choices=sorted(METHODS))
    ap.add_argument("--backbones", nargs="*", default=None, choices=sorted(BACKBONES))
    ap.add_argument("--with-F", action="store_true", help="include the collapsed backbone")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--suffix", default="",
                    help="append to every cell name, e.g. --suffix _both. USE THIS when "
                         "retraining on a different split: without it --force overwrites "
                         "the runs your dev numbers and vote threshold came from.")
    ap.add_argument("--shard", default=None, metavar="i/n",
                    help="run only every n-th cell, offset i -- for three cards at once")
    ap.add_argument("--set", nargs="*", default=[], metavar="K=V",
                    help="override a TrainConfig field for EVERY cell")
    a = ap.parse_args()

    methods = a.methods or list(METHODS)
    bbs = a.backbones or (["B", "A", "M", "F"] if a.with_F else ["B", "A", "M"])

    cells = []
    skipped_undefined = []
    for bb in bbs:
        for m in methods:
            over = cell_overrides(m, bb)
            name = f"{m}_{bb}{a.suffix}"
            if over is None:
                skipped_undefined.append((name, m, bb))
            else:
                cells.append((name, m, bb, over))

    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        cells = [c for j, c in enumerate(cells) if j % n == i]

    if a.list:
        print(f"{'cell':14s} {'method':10s} {'backbone':9s} {'done?':6s}  note")
        for name, m, bb, _ in cells:
            done = (Path("runs") / name / "summary.json").exists()
            print(f"{name:14s} {m:10s} {bb:9s} {'yes' if done else '-':6s}  "
                  f"{METHODS[m][1]}")
        for name, m, bb in skipped_undefined:
            print(f"{name:14s} {m:10s} {bb:9s} {'n/a':6s}  "
                  f"UNDEFINED: prefix tuning needs attention layers; Mamba has none")
        todo = sum(1 for n_, _, _, _ in cells
                   if not (Path('runs') / n_ / 'summary.json').exists())
        print(f"\n{len(cells)} cells, {todo} outstanding "
              f"(~{todo * a.epochs * 0.6 / 60:.1f} h on one card at {a.epochs} epochs)")
        return

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    sys.path.insert(0, str(Path(__file__).parent))
    from ner.train import TrainConfig, train

    common = {}
    for s in a.set:
        k, _, v = s.partition("=")
        try:
            common[k] = json.loads(v)
        except json.JSONDecodeError:
            common[k] = v


    if a.force and not a.suffix and ("train_folds" in common or "dev_folds" in common):
        clashes = [c[0] for c in cells
                   if (Path("runs") / c[0] / "summary.json").exists()]
        if clashes:
            sys.exit(
                f"refusing to overwrite {len(clashes)} existing run(s) while changing the "
                f"data split:\n  {', '.join(clashes)}\n"
                f"Those runs hold your honest dev scores. Pass --suffix (e.g. "
                f"--suffix _both) to train the new split under new names.")

    pending = [c for c in cells
               if a.force or not (Path("runs") / c[0] / "summary.json").exists()]
    done_already = [c[0] for c in cells if c not in pending]
    if done_already:
        print(f"[grid] already done: {', '.join(done_already)}  (--force to redo)")
    for name, m, bb in skipped_undefined:
        print(f"[grid] skipping {name}: prefix tuning is undefined without attention")
    eff_epochs = common.get("epochs", a.epochs)
    folds = common.get("train_folds", ["fold1"])
    print(f"[grid] {len(pending)} cell(s) on cuda:{a.gpu}, {eff_epochs} epochs each, "
          f"train on {folds}  (~{len(pending) * eff_epochs * 0.6 / 60:.1f} h)")
    if common:
        print(f"[grid] applied to every cell: {common}")
    print()

    cache: dict = {}
    t_all = time.time()
    for i, (name, m, bb, over) in enumerate(pending, 1):


        cfg = TrainConfig(name=name, **{"epochs": a.epochs, **over, **common})
        eta = (time.time() - t_all) / max(i - 1, 1) * (len(pending) - i + 1) / 60
        print(f"\n{'=' * 78}\n  [{i}/{len(pending)}]  {name}   "
              f"{'(eta ' + format(eta, '.0f') + ' min)' if i > 1 else ''}\n"
              f"  {METHODS[m][1]}  |  {BACKBONES[bb][1]}\n{'=' * 78}")
        t0 = time.time()
        try:
            train(cfg, cache=cache)
        except Exception:
            print(f"\n*** {name} FAILED after {(time.time()-t0)/60:.1f} min:")
            traceback.print_exc()

    report(cells, skipped_undefined, (time.time() - t_all) / 60)


def report(cells, undefined, minutes):
    print(f"\n{'=' * 94}\n  GRID COMPLETE in {minutes:.1f} min\n{'=' * 94}")
    rows = []
    for name, m, bb, _ in cells:
        p = Path("runs") / name / "summary.json"
        rows.append((name, m, bb, json.loads(p.read_text()) if p.exists() else None))
    rows.sort(key=lambda r: -(r[3]["best"]["macro_f1"] if r[3] else -1))

    print(f"{'cell':14s} {'method':10s} {'bb':3s} {'macro-F1':>9s} {'@ep':>4s} "
          f"{'trainable':>11s} {'%':>8s} {'min':>6s}")
    print("-" * 94)
    for name, m, bb, s in rows:
        if s is None:
            print(f"{name:14s} {m:10s} {bb:3s} {'-- not run --':>24s}")
            continue
        flag = " <- dev contaminated" if s.get("dev_contaminated") else ""
        print(f"{name:14s} {m:10s} {bb:3s} {s['best']['macro_f1']:9.4f} "
              f"{s['best']['epoch']:4d} {s['trainable']:11,d} "
              f"{s['trainable_pct']:7.3f}% {s['wall_min']:6.1f}{flag}")
    for name, m, bb in undefined:
        print(f"{name:14s} {m:10s} {bb:3s} {'n/a (no attention)':>24s}")
    print("-" * 94)
    print("Selected on STRICT MACRO-F1 on the dev fold, never on loss.")
    good = [n for n, _, _, s in rows if s][:5]
    dirty = [n for n, _, _, s in rows if s and s.get("dev_contaminated")]
    if dirty:
        print(f"\n{len(dirty)} run(s) were scored on their own training data. Those "
              f"macro-F1 values measure MEMORISATION, not quality -- do not quote them "
              f"and do not rank anything by them. They are submission models: score them "
              f"on the leaderboard, and take the ensemble vote threshold from an "
              f"uncontaminated run.")
        print("\nNext:  python run_predict.py --run runs/<name>    for each, then")
        print("       python ensemble.py --subs <names> --name ens --vote <k from dev>")
    elif good:
        print("\nNext:  python ensemble.py --dev " + " ".join(f"runs/{n}" for n in good))
        print("       voting across different mechanisms has beaten every single model "
              "so far (0.4560 vs 0.4088).")


if __name__ == "__main__":
    main()

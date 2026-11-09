from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cjpe import data as D
from cjpe.metrics import best_threshold, prf, report, write_submission

ROOT = Path(__file__).resolve().parent


def ranks(p):
    return p.argsort().argsort() / (len(p) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--name")
    ap.add_argument("--weights", nargs="*", type=float)
    ap.add_argument("--select-on", choices=["clean", "all"], default="clean")
    ap.add_argument("--rank-thr", type=float, default=None,
                    help="fixed rank threshold (for train_on_dev runs, whose dev is not held out)")
    a = ap.parse_args()
    w = np.array(a.weights or [1.0] * len(a.runs))
    assert len(w) == len(a.runs)
    w = w / w.sum()

    dev_docs = D.load("dev")
    clean = D.clean_dev_mask(dev_docs)
    yall = np.array([d.label for d in dev_docs])
    sel = clean if a.select_on == "clean" else np.ones_like(clean)
    ydv = yall[sel]
    dv_all = []
    test_ids = [d.id for d in D.load("test")]
    dv, te = [], []
    for r in a.runs:
        d, t = ROOT / "runs" / r / "dev_probs.npy", ROOT / "runs" / r / "test_probs.npy"
        if not t.exists():
            raise SystemExit(f"{t} missing -- run: python run_predict.py --run runs/{r}")
        full = np.load(d)
        pd = full[sel]
        dv.append(ranks(pd)); te.append(ranks(np.load(t))); dv_all.append(ranks(full))
        print(f"  {r:18s} clean dev {best_threshold(full[clean], yall[clean])[1]:.4f} | "
              f"all dev {best_threshold(full, yall)[1]:.4f}")
    sdv = np.tensordot(w, np.stack(dv), 1)
    ste = np.tensordot(w, np.stack(te), 1)
    thr, score = best_threshold(sdv, ydv, grid=np.linspace(0.2, 0.8, 241))
    if a.rank_thr is not None:
        print(f"[ensemble] dev-tuned rank thr {thr:.3f} IGNORED -> fixed {a.rank_thr:.3f}")
        thr = a.rank_thr
    sall = np.tensordot(w, np.stack(dv_all), 1)
    print(f"[ensemble] selected on {a.select_on} dev: rank thr {thr:.3f}")
    print(f"[ensemble] clean dev (994)  {report(prf((sall[clean] >= thr).astype(int), yall[clean]))}")
    print(f"[ensemble] all dev (2,511)  {report(prf((sall >= thr).astype(int), yall))}")
    name = a.name or "ens_" + "+".join(a.runs)
    pred = {i: int(s >= thr) for i, s in zip(test_ids, ste)}
    z = write_submission(pred, test_ids, ROOT / "subs" / f"{name}.zip")
    print(f"[ok] {z}  predicted 1 for {np.mean(list(pred.values())):.1%} of test")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lsi import data as D
from lsi.metrics import decide, evaluate, report, write_submission

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--weights", nargs="*", type=float)
    ap.add_argument("--name")
    a = ap.parse_args()
    w = np.array(a.weights or [1.0] * len(a.runs)); assert len(w) == len(a.runs)
    w = w / w.sum()

    Ydv = D.label_matrix(D.load("dev"))
    test_ids = [d.id for d in D.load("test")]
    dv, te = [], []
    for r in a.runs:
        d, t = ROOT / "runs" / r / "dev_probs.npy", ROOT / "runs" / r / "test_probs.npy"
        if not t.exists():
            raise SystemExit(f"{t} missing -- run: python run_predict.py --run runs/{r}")
        dv.append(np.load(d)); te.append(np.load(t))
        print(f"  {r:18s} {report(evaluate(dv[-1], Ydv))}")
    sdv = np.tensordot(w, np.stack(dv), 1)
    ste = np.tensordot(w, np.stack(te), 1)
    e = evaluate(sdv, Ydv)
    print(f"[ensemble] {report(e)}")
    pred = decide(ste, e["thresholds"])
    name = a.name or "ens_" + "+".join(a.runs)
    z = write_submission(pred, test_ids, ROOT / "subs" / f"{name}.zip")
    print(f"[ok] {z}  ({pred.sum(1).mean():.2f} statutes/doc predicted)")


if __name__ == "__main__":
    main()

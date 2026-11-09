from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lsi import data as D
from lsi.metrics import decide, evaluate, report, scores, tune_per_label

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()

    tr, dv = D.load("train"), D.load("dev")
    ids = [d.id for d in tr] + [d.id for d in dv]
    row = {i: r for r, i in enumerate(ids)}
    oof = np.full((len(ids), D.N_LABELS), np.nan, dtype=np.float32)
    test = []
    for f in range(a.k):
        run = ROOT / "runs" / f"{a.exp}_f{f}"
        if not (run / "test_probs.npy").exists():
            raise SystemExit(f"{run} is not finished (no test_probs.npy)")
        fid = json.loads((run / "oof_ids.json").read_text())
        oof[[row[i] for i in fid]] = np.load(run / "dev_probs.npy")
        test.append(np.load(run / "test_probs.npy"))
    assert not np.isnan(oof).any(), "some pooled docs have no OOF score: folds incomplete"

    out = ROOT / "runs" / f"{a.exp}_kf"
    out.mkdir(parents=True, exist_ok=True)
    n = len(tr)
    np.save(out / "oof_probs.npy", oof)
    np.save(out / "dev_probs.npy", oof[n:])
    np.save(out / "test_probs.npy", np.mean(test, 0).astype(np.float32))

    Ytr, Ydv = D.label_matrix(tr), D.label_matrix(dv)
    thr = tune_per_label(oof[:n], Ytr)
    np.save(out / "thresholds_trainoof.npy", thr)
    held = scores(decide(oof[n:], thr), Ydv)
    print(f"[{a.exp}_kf] dev OOF, train-OOF thresholds: macro-F1 {held['macro_f1']:.4f} "
          f"micro {held['micro_f1']:.4f}")
    print(f"[{a.exp}_kf] dev OOF, evaluate(): {report(evaluate(oof[n:], Ydv))}")
    print(f"[ok] {out}")


if __name__ == "__main__":
    main()

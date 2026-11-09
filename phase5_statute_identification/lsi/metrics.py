from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

GRID = np.linspace(0.02, 0.98, 97)


def scores(P: np.ndarray, Y: np.ndarray) -> dict:
    P, Y = P.astype(bool), Y.astype(bool)
    tp = (P & Y).sum(0); fp = (P & ~Y).sum(0); fn = (~P & Y).sum(0)
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    TP, FP, FN = tp.sum(), fp.sum(), fn.sum()
    return {"macro_f1": float(f1.mean()),
            "micro_f1": float(2 * TP / max(2 * TP + FP + FN, 1)),
            "subset_acc": float((P == Y).all(1).mean()),
            "macro_p": float(np.mean(np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0))),
            "macro_r": float(np.mean(np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0))),
            "per_label_f1": f1}


def decide(S: np.ndarray, thr, top1: bool = True) -> np.ndarray:
    P = S >= np.asarray(thr)
    if top1:
        empty = ~P.any(1)
        P[np.where(empty)[0], S[empty].argmax(1)] = True
    return P


def tune_global(S, Y, top1=True) -> tuple[float, float]:
    best = max(((scores(decide(S, t, top1), Y)["macro_f1"], t) for t in GRID))
    return float(best[1]), float(best[0])


def tune_per_label(S, Y) -> np.ndarray:
    S, Y = np.asarray(S), np.asarray(Y).astype(bool)
    thr = np.empty(S.shape[1], dtype=np.float32)
    for j in range(S.shape[1]):
        s, y = S[:, j], Y[:, j]
        P = s[:, None] >= GRID[None, :]
        tp = (P & y[:, None]).sum(0); fp = (P & ~y[:, None]).sum(0); fn = y.sum() - tp
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        thr[j] = GRID[int(np.argmax(f1))]
    return thr


def crossfit(S, Y, seed=0) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(S)); a, b = idx[: len(S) // 2], idx[len(S) // 2:]
    out = []
    for fit, ev in ((a, b), (b, a)):
        out.append(scores(decide(S[ev], tune_per_label(S[fit], Y[fit])), Y[ev])["macro_f1"])
    return float(np.mean(out))


def evaluate(S, Y) -> dict:
    tg, fg = tune_global(S, Y)
    tl = tune_per_label(S, Y)
    m = scores(decide(S, tl), Y)
    return {"global_thr": tg, "macro_f1_global": fg, "macro_f1_perlabel": m["macro_f1"],
            "macro_f1_crossfit": crossfit(S, Y), "micro_f1": m["micro_f1"],
            "subset_acc": m["subset_acc"], "thresholds": tl}


def report(e: dict) -> str:
    return (f"macro-F1 per-label {e['macro_f1_perlabel']:.4f} (cross-fit "
            f"{e['macro_f1_crossfit']:.4f}) | global {e['macro_f1_global']:.4f} @"
            f"{e['global_thr']:.2f} | micro {e['micro_f1']:.4f} | subset {e['subset_acc']:.4f}")


def write_submission(P: np.ndarray, test_ids: list[str], out_zip: Path) -> Path:
    assert P.shape == (len(test_ids), 100)
    rows = [{"id": i, "pred": [int(j) for j in np.flatnonzero(p)]} for i, p in zip(test_ids, P)]
    assert all(len(set(r["pred"])) == len(r["pred"]) for r in rows)
    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("prediction.json", json.dumps(rows))
    return out_zip

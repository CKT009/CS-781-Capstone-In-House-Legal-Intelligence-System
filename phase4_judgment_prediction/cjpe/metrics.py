from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np


def prf(pred, gold) -> dict:
    pred, gold = np.asarray(pred), np.asarray(gold)
    out = {}
    for c in (0, 1):
        tp = int(((pred == c) & (gold == c)).sum())
        fp = int(((pred == c) & (gold != c)).sum())
        fn = int(((pred != c) & (gold == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out[c] = {"p": p, "r": r, "f1": f, "n_gold": int((gold == c).sum()),
                  "n_pred": int((pred == c).sum())}
    out["macro_f1"] = (out[0]["f1"] + out[1]["f1"]) / 2
    out["macro_p"] = (out[0]["p"] + out[1]["p"]) / 2
    out["macro_r"] = (out[0]["r"] + out[1]["r"]) / 2
    out["accuracy"] = float((pred == gold).mean())
    return out


def best_threshold(p1, gold, grid=None) -> tuple[float, float]:
    p1, gold = np.asarray(p1), np.asarray(gold)
    grid = np.linspace(0.05, 0.95, 181) if grid is None else grid
    scores = [prf((p1 >= t).astype(int), gold)["macro_f1"] for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def report(m: dict) -> str:
    return (f"macro-F1 {m['macro_f1']:.4f}  (P {m['macro_p']:.4f}  R {m['macro_r']:.4f}  "
            f"acc {m['accuracy']:.4f})  | class0 F1 {m[0]['f1']:.4f} pred {m[0]['n_pred']}"
            f"/{m[0]['n_gold']}  class1 F1 {m[1]['f1']:.4f} pred {m[1]['n_pred']}/{m[1]['n_gold']}")


def write_submission(pred: dict, test_ids: list[str], out_zip: Path) -> Path:
    missing = [i for i in test_ids if i not in pred]
    if missing:
        raise ValueError(f"{len(missing)} test ids have no prediction, e.g. {missing[:3]}")
    rows = [{"id": i, "pred": int(pred[i])} for i in test_ids]
    assert all(r["pred"] in (0, 1) for r in rows)
    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("prediction.json", json.dumps(rows))
    return out_zip

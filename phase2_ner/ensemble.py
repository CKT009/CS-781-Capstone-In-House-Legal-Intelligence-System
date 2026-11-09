from __future__ import annotations

import argparse
import collections
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def _load_dev(run: Path):
    p = run / "dev_predictions.json"
    if not p.exists():
        sys.exit(f"{p} not found -- that run has no saved dev predictions.")
    return {k: set(tuple(s[:3]) for s in v) for k, v in json.loads(p.read_text()).items()}


def _load_sub(name: str, subs_dir="subs"):
    p = Path(subs_dir) / f"{name}.zip"
    if not p.exists():
        sys.exit(f"{p} not found -- run run_predict.py for that run first.")
    d = json.loads(zipfile.ZipFile(p).read("prediction.json"))
    return {k: set(tuple(s[:3]) for s in v) for k, v in d.items()}


def vote(models: list[dict], k: int) -> dict:
    docs = set().union(*(set(m) for m in models))
    out = {}
    for d in docs:
        c = collections.Counter(s for m in models for s in m.get(d, ()))
        out[d] = sorted(s for s, n in c.items() if n >= k)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run dirs (--dev) or submission names (--subs)")
    ap.add_argument("--dev", action="store_true", help="score every threshold on the dev fold")
    ap.add_argument("--subs", action="store_true", help="combine subs/<name>.zip files")
    ap.add_argument("--vote", type=int, default=None, help="votes required (default: majority)")
    ap.add_argument("--name", default="ensemble", help="output name under subs/")
    ap.add_argument("--dev-fold", default="fold2")
    ap.add_argument("--data-dir", default="data/Phase_2_LNER")
    a = ap.parse_args()

    from ner import data as D
    from ner.metrics import strict_prf, format_report

    if a.dev:
        models = [_load_dev(Path(r)) for r in a.runs]
        _, gold = D.load_fold(a.data_dir, a.dev_fold)
        n = len(models)
        print(f"{'model / ensemble':34s} {'macroF1':>8s} {'P':>7s} {'R':>7s} {'spans':>7s}")
        for r, m in zip(a.runs, models):
            s = strict_prf({k: list(v) for k, v in m.items()}, gold)
            print(f"{Path(r).name:34s} {s['macro_f1']:8.4f} {s['macro_p']:7.4f} "
                  f"{s['macro_r']:7.4f} {sum(len(v) for v in m.values()):7d}")
        print("-" * 68)
        best = (None, -1.0)
        for k in range(1, n + 1):
            e = vote(models, k)
            s = strict_prf(e, gold)
            tag = {1: "union", n: "intersection"}.get(k, "majority" if k * 2 > n else "weak")
            print(f"{f'vote >={k} ({tag})':34s} {s['macro_f1']:8.4f} {s['macro_p']:7.4f} "
                  f"{s['macro_r']:7.4f} {sum(len(v) for v in e.values()):7d}")
            if s["macro_f1"] > best[1]:
                best = (k, s["macro_f1"])
        print(f"\ngold spans: {sum(len(v) for v in gold.values())}")
        print(f"best threshold on this dev fold: --vote {best[0]}  ({best[1]:.4f})")
        print("\nThat threshold was chosen by looking at dev. With few models the "
              "selection bias is small, but quote the single-model number too.")
        return

    if not a.subs:
        sys.exit("pass --dev (score on the dev fold) or --subs (build a submission)")

    models = [_load_sub(r) for r in a.runs]
    k = a.vote or (len(models) + 1) // 2
    ens = vote(models, k)
    print(f"[ens] {len(models)} models, keeping spans with >= {k} votes")
    for r, m in zip(a.runs, models):
        print(f"    {r:24s} {sum(len(v) for v in m.values()):6d} spans")
    print(f"    {'-> ensemble':24s} {sum(len(v) for v in ens.values()):6d} spans")

    texts, _ = D.load_fold(a.data_dir, "fold3")
    from ner.predict import write_submission
    out = Path("runs") / a.name / "submission"
    zp = write_submission({d: ens.get(d, []) for d in texts}, out, texts)
    dst = Path("subs") / f"{a.name}.zip"
    dst.parent.mkdir(exist_ok=True)
    import shutil
    shutil.copy2(zp, dst)
    (Path("subs") / f"{a.name}.log").write_text(json.dumps(
        {"method": "ensemble vote", "votes_required": k, "members": list(a.runs),
         "n_docs": len(ens), "n_spans": sum(len(v) for v in ens.values())}, indent=2))
    print(f"\n[ok] {dst}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def _dev(run: Path) -> dict:
    p = run / "dev_predictions.json"
    if not p.exists():
        sys.exit(f"{p} not found -- run `python run_predict.py --run {run} --dev` first.")
    return json.loads(p.read_text())


def _sub(name: str, subs="subs") -> dict:
    p = Path(subs) / f"{name}.zip"
    if not p.exists():
        sys.exit(f"{p} not found -- run run_predict.py for that run first.")
    return json.loads(zipfile.ZipFile(p).read("prediction.json"))


def vote(models: list[dict]) -> dict:
    docs = set().union(*(set(m) for m in models))
    out = {}
    for d in sorted(docs):
        seqs = [m[d] for m in models if d in m]
        n = len(seqs[0])
        if any(len(s) != n for s in seqs):
            sys.exit(f"{d}: members disagree on sentence count {[len(s) for s in seqs]}")
        res = []
        for i in range(n):
            c = collections.Counter(s[i] for s in seqs)
            top = max(c.values())

            res.append(next(s[i] for s in seqs if c[s[i]] == top))
        out[d] = res
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--subs", action="store_true")
    ap.add_argument("--name", default="ens")
    ap.add_argument("--data-dir", default="data/Phase_3_RR")
    a = ap.parse_args()

    from rr import data as D
    from rr.metrics import score, report, validate_submission
    vocab = D.load_labels(a.data_dir)

    if a.dev:
        models = [_dev(Path(r)) for r in a.runs]
        gold = {d.id: d.labels for d in D.load_all(a.data_dir, "dev")}
        print(f"{'run':34s}{'macroF1':>9s}{'supported':>11s}{'acc':>8s}")
        print("-" * 62)
        for r, m in zip(a.runs, models):
            s = score(m, gold, vocab)
            print(f"{Path(r).name:34s}{s['macro_f1']:9.4f}{s['macro_f1_supported']:11.4f}"
                  f"{s['accuracy']:8.4f}")
        s = score(vote(models), gold, vocab)
        print("-" * 62)
        print(f"{'MAJORITY VOTE':34s}{s['macro_f1']:9.4f}{s['macro_f1_supported']:11.4f}"
              f"{s['accuracy']:8.4f}")
        best = max(score(m, gold, vocab)["macro_f1"] for m in models)
        print(f"\nbest single member {best:.4f}   vote gain {s['macro_f1'] - best:+.4f}")
        return

    if not a.subs:
        sys.exit("pass --dev (score) or --subs (build a submission)")
    models = [_sub(r) for r in a.runs]
    ens = vote(models)
    docs = D.load_all(a.data_dir, "test")
    errs = validate_submission(ens, docs, vocab)
    if errs:
        sys.exit("ensemble invalid:\n  - " + "\n  - ".join(errs[:10]))
    out = Path("runs") / a.name / "submission"
    out.mkdir(parents=True, exist_ok=True)
    pj = out / "prediction.json"
    pj.write_text(json.dumps(ens, ensure_ascii=False))
    zp = out / "submission.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    Path("subs").mkdir(exist_ok=True)
    shutil.copy2(zp, Path("subs") / f"{a.name}.zip")
    (Path("subs") / f"{a.name}.log").write_text(json.dumps(
        {"method": "per-sentence majority vote", "members": list(a.runs),
         "n_docs": len(ens), "n_sentences": sum(len(v) for v in ens.values())}, indent=2))
    dist = collections.Counter(l for v in ens.values() for l in v)
    print(f"[ok] subs/{a.name}.zip  ({len(ens)} docs, "
          f"{sum(len(v) for v in ens.values())} sentences)")
    print("[ok] " + "  ".join(f"{l}={dist[l]}" for l in vocab if dist[l]))
    miss = [l for l in vocab if not dist[l]]
    if miss:
        print(f"[warn] never predicted: {miss} -- each costs {1/len(vocab):.3f} of macro-F1")


if __name__ == "__main__":
    main()

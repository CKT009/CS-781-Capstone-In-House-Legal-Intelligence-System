from __future__ import annotations

import argparse
import collections
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def _is_final_fit(run: Path) -> bool:
    p = run / "config.json"
    return p.exists() and bool(json.loads(p.read_text()).get("final_fit", False))


def load_probs(run, split: str, vocab: list[str]) -> dict:
    run = Path(run)
    if split == "dev" and _is_final_fit(run):
        sys.exit(f"{run} is a FINAL-FIT run: dev was its training data, so its dev "
                 f"probabilities are contaminated and cannot be scored. Use --subs.")
    p = run / f"{split}_probs.json"
    if not p.exists():
        sys.exit(f"{p} not found -- run `python run_predict.py --run {run} --probs-only` "
                 f"first.")
    c = run / "crf.json"
    if c.exists():
        v = json.loads(c.read_text()).get("vocab")
        if v is not None and list(v) != list(vocab):
            sys.exit(f"{run}: probabilities were written in vocab order {v}, "
                     f"not the data's {vocab}")
    raw = json.loads(p.read_text())
    out = {}
    for d, rows in raw.items():
        a = np.asarray(rows, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != len(vocab):
            sys.exit(f"{p}: {d} has shape {a.shape}, expected [n_sent, {len(vocab)}]")
        out[d] = a
    return out


def load_crf(run) -> dict | None:
    c = Path(run) / "crf.json"
    if not c.exists():
        return None
    cp = json.loads(c.read_text()).get("crf")
    return None if cp is None else {k: np.asarray(v, dtype=np.float64) for k, v in cp.items()}


def average(members: list[dict], weights=None) -> dict:
    if not members:
        raise ValueError("no members")
    w = np.ones(len(members)) if weights is None else np.asarray(weights, dtype=np.float64)
    if len(w) != len(members) or (w < 0).any() or w.sum() <= 0:
        raise ValueError(f"bad weights {weights!r} for {len(members)} members")
    w = w / w.sum()
    docs = set(members[0])
    for i, m in enumerate(members[1:], 1):
        if set(m) != docs:
            raise ValueError(f"member {i} covers different documents "
                             f"({len(set(m) ^ docs)} differ)")
    out = {}
    for d in sorted(docs):
        shapes = [m[d].shape for m in members]
        if len(set(shapes)) != 1:
            raise ValueError(f"{d}: members disagree on shape {shapes}")
        out[d] = sum(wi * m[d] for wi, m in zip(w, members))
    return out


def decode_argmax(avg: dict, vocab: list[str]) -> dict:
    return {d: [vocab[int(k)] for k in p.argmax(1)] for d, p in avg.items()}


def mean_crf(crfs: list[dict]) -> dict:
    if any(c is None for c in crfs):
        raise ValueError("viterbi decoding needs every member's crf.json with a CRF")
    return {k: np.mean([c[k] for c in crfs], axis=0) for k in ("transitions", "start", "end")}


def viterbi(emis: np.ndarray, trans: np.ndarray, start: np.ndarray, end: np.ndarray):
    T, K = emis.shape
    a = start + emis[0]
    back = np.zeros((T, K), dtype=np.int64)
    for t in range(1, T):
        sc = a[:, None] + trans
        back[t] = sc.argmax(0)
        a = sc.max(0) + emis[t]
    cur = int((a + end).argmax())
    seq = [cur]
    for t in range(T - 1, 0, -1):
        cur = int(back[t, cur])
        seq.append(cur)
    return seq[::-1]


def decode_viterbi(avg: dict, crf: dict, vocab: list[str], eps: float = 1e-8) -> dict:
    return {d: [vocab[k] for k in viterbi(np.log(p + eps), crf["transitions"],
                                          crf["start"], crf["end"])]
            for d, p in avg.items()}


def decode(avg, vocab, how, crfs=None):
    if how == "argmax":
        return decode_argmax(avg, vocab)
    if how == "viterbi":
        return decode_viterbi(avg, mean_crf(crfs), vocab)
    raise ValueError(f"unknown decode {how!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories, e.g. runs/B_grown/bitfit40_s3")
    ap.add_argument("--dev", action="store_true", help="score on dev")
    ap.add_argument("--subs", action="store_true", help="build subs/<name>.zip from test")
    ap.add_argument("--name", default=None)
    ap.add_argument("--decode", choices=("argmax", "viterbi"), default="viterbi")
    ap.add_argument("--weights", type=float, nargs="*", default=None)
    ap.add_argument("--data-dir", default="data/Phase_3_RR")
    a = ap.parse_args()
    if a.dev == a.subs:
        sys.exit("pass exactly one of --dev (score) or --subs (build a submission)")

    from rr import data as D
    from rr.metrics import score, validate_submission
    vocab = D.load_labels(a.data_dir)
    runs = [Path(r) for r in a.runs]

    if a.dev:
        from ensemble import vote
        members = [load_probs(r, "dev", vocab) for r in runs]
        crfs = [load_crf(r) for r in runs]
        gold = {d.id: d.labels for d in D.load_all(a.data_dir, "dev")}
        hard = []
        for r in runs:
            p = r / "dev_predictions.json"
            if not p.exists():
                sys.exit(f"{p} not found -- needed for the hard-vote comparison")
            hard.append(json.loads(p.read_text()))

        def row(label, pred):
            s = score(pred, gold, vocab)
            print(f"{label:40s}{s['macro_f1']:9.4f}{s['macro_f1_supported']:11.4f}"
                  f"{s['accuracy']:8.4f}")
            return s["macro_f1"]

        print(f"{'':40s}{'macroF1':>9s}{'supported':>11s}{'acc':>8s}")
        print("-" * 68)
        singles = []
        for r, h, m in zip(runs, hard, members):
            singles.append(row(f"{r.name}  viterbi (as trained)", h))
            mb = decode_argmax(m, vocab)
            row(f"{r.name}  argmax-marginal", mb)
            agree = np.mean([x == y for d in h for x, y in zip(h[d], mb[d])])
            print(f"{'':40s}  (agrees with its viterbi on {agree:.1%} of sentences)")
            c = load_crf(r)
            if c is not None:
                row(f"{r.name}  viterbi-on-marginals", decode_viterbi(m, c, vocab))
        print("-" * 68)
        hv = row("HARD VOTE (ensemble.vote)", vote(hard))
        avg = average(members, a.weights)
        sa = row("SOFT VOTE  argmax", decode_argmax(avg, vocab))
        sv = None
        if all(c is not None for c in crfs):
            sv = row("SOFT VOTE  viterbi (mean transitions)",
                     decode_viterbi(avg, mean_crf(crfs), vocab))
        best = max(singles)
        print(f"\nbest single {best:.4f}   hard {hv - best:+.4f}   soft-argmax "
              f"{sa - best:+.4f}" + ("" if sv is None else f"   soft-viterbi {sv - best:+.4f}"))
        print(f"soft-argmax vs hard vote: {sa - hv:+.4f}")
        return

    if not a.name:
        sys.exit("--subs needs --name")
    zp = Path("subs") / f"{a.name}.zip"
    if zp.exists():
        sys.exit(f"{zp} already exists -- refusing to overwrite; choose another --name")
    members = [load_probs(r, "test", vocab) for r in runs]
    crfs = [load_crf(r) for r in runs] if a.decode == "viterbi" else None
    avg = average(members, a.weights)
    ens = decode(avg, vocab, a.decode, crfs)
    docs = D.load_all(a.data_dir, "test")
    errs = validate_submission(ens, docs, vocab)
    if errs:
        sys.exit("ensemble invalid:\n  - " + "\n  - ".join(errs[:10]))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("prediction.json", json.dumps(ens, ensure_ascii=False))
    zp.parent.mkdir(exist_ok=True)
    zp.write_bytes(buf.getvalue())
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["prediction.json"], z.namelist()
        again = json.loads(z.read("prediction.json"))
    assert validate_submission(again, docs, vocab) == []
    (Path("subs") / f"{a.name}.log").write_text(json.dumps(
        {"method": f"soft vote: mean CRF marginals, {a.decode} decode",
         "members": [str(r) for r in runs], "weights": a.weights,
         "final_fit_members": [str(r) for r in runs if _is_final_fit(r)],
         "n_docs": len(ens), "n_sentences": sum(len(v) for v in ens.values())}, indent=2))
    dist = collections.Counter(l for v in ens.values() for l in v)
    print(f"[ok] {zp}  ({len(ens)} docs, {sum(len(v) for v in ens.values())} sentences)")
    print("[ok] " + "  ".join(f"{l}={dist[l]}" for l in vocab if dist[l]))
    miss = [l for l in vocab if not dist[l]]
    if miss:
        print(f"[warn] never predicted: {miss} -- each costs {1/len(vocab):.3f} of macro-F1")


if __name__ == "__main__":
    main()

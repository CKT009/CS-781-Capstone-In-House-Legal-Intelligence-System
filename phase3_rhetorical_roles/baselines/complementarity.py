from __future__ import annotations

import argparse
import collections
import json
import sys
import zipfile

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rr import data as D
from rr.metrics import score, score_by_domain, validate_submission

NEURAL = ["bitfit40_s3", "bitfit40", "bitfit40_s2"]
NEVER = ("Dissent", "PrecedentOverruled")


def run_dir(n):
    for p in (ROOT / "runs/B_grown" / n, ROOT / "runs" / n):
        if (p / "dev_predictions.json").exists():
            return p
    raise FileNotFoundError(n)


def stack_probs(probs: dict, docs):
    return np.vstack([np.asarray(probs[d.id]) for d in docs])


def argmax_pred(P, docs, vocab):
    P = P.copy(); P[:, [vocab.index(l) for l in NEVER]] = -1
    out, off = {}, 0
    for d in docs:
        out[d.id] = [vocab[i] for i in P[off:off + len(d)].argmax(1)]; off += len(d)
    return out


def soft(neural_P, lex_P, w_lex):
    return (1 - w_lex) * np.mean(neural_P, 0) + w_lex * lex_P


def vote(members):
    out = {}
    for d in members[0]:
        seqs = [m[d] for m in members]
        lab = []
        for i in range(len(seqs[0])):
            c = collections.Counter(s[i] for s in seqs)
            top = max(c.values())
            lab.append(next(s[i] for s in seqs if c[s[i]] == top))
        out[d] = lab
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lex", default=str(ROOT / "baselines/lexical_dev_predictions.json"))
    ap.add_argument("--write-test", action="store_true",
                    help="also write subs/lexical_vote4.zip and subs/lexical_soft4.zip")
    a = ap.parse_args()
    data = ROOT / "data/Phase_3_RR"
    vocab = D.load_labels(data)
    dev = D.load_all(data, "dev")
    gold = {d.id: d.labels for d in dev}
    nn = {n: json.loads((run_dir(n) / "dev_predictions.json").read_text()) for n in NEURAL}
    lex = json.loads(Path(a.lex).read_text())

    def agree(p, q):
        tot = sum(len(v) for v in gold.values())
        return sum(x == y for d in gold for x, y in zip(p[d], q[d])) / tot

    print("agreement (fraction of dev sentences with the same label)")
    for n in NEURAL:
        print(f"  lexical vs {n:12s} {agree(lex, nn[n]):.3f}")
    print(f"  {NEURAL[0]} vs {NEURAL[1]:12s} {agree(nn[NEURAL[0]], nn[NEURAL[1]]):.3f}")
    print(f"  {NEURAL[0]} vs {NEURAL[2]:12s} {agree(nn[NEURAL[0]], nn[NEURAL[2]]):.3f}")


    v3 = vote([nn[n] for n in NEURAL])
    lr = nr = both = 0
    for d in gold:
        for g, p, q in zip(gold[d], lex[d], v3[d]):
            lr += (p == g) and (q != g); nr += (q == g) and (p != g); both += (p == g) and (q == g)
    print(f"  lexical right & 3-vote wrong: {lr}   3-vote right & lexical wrong: {nr}   both right: {both}")

    cands = {n: nn[n] for n in NEURAL}
    cands["lexical"] = lex
    cands["vote3 (neural)"] = v3
    cands["vote4 (3 neural + lexical)"] = vote([nn[n] for n in NEURAL] + [lex])

    d3 = {}
    for d in gold:
        lab = []
        for i in range(len(gold[d])):
            c = collections.Counter(nn[n][d][i] for n in NEURAL)
            lab.append(lex[d][i] if max(c.values()) == 1 else v3[d][i])
        d3[d] = lab
    cands["vote3, lexical breaks 3-way splits"] = d3

    cands["vote3 (s3, bitfit40, lexical)"] = vote([nn["bitfit40_s3"], nn["bitfit40"], lex])


    nP = [stack_probs(json.loads((run_dir(n) / "dev_probs.json").read_text()), dev) for n in NEURAL]
    lP = np.load(ROOT / "baselines/lexical_dev_proba.npy")
    cands["soft: mean neural marginals"] = argmax_pred(np.mean(nP, 0), dev, vocab)
    cands["soft4: equal-weight pool (w_lex=1/4)"] = argmax_pred(soft(nP, lP, 0.25), dev, vocab)
    cands["soft: w_lex=1/2"] = argmax_pred(soft(nP, lP, 0.5), dev, vocab)

    print(f"\n{'dev':40s} {'macro13':>8s} {'CL':>7s} {'IT':>7s}")
    for k, p in cands.items():
        per = score_by_domain(p, dev, vocab)
        print(f"{k:40s} {per['ALL']['macro_f1']:8.4f} {per['CL']['macro_f1']:7.4f} "
              f"{per['IT']['macro_f1']:7.4f}")

    print("\nper-label dev F1: lexical vs bitfit40_s3 vs vote4")
    sl, sn, s4 = (score(cands[k], gold, vocab)["per_label"] for k in
                  ("lexical", "bitfit40_s3", "vote4 (3 neural + lexical)"))
    for l in vocab:
        if sl[l]["support"]:
            print(f"  {l:24s} {sl[l]['f1']:.3f}  {sn[l]['f1']:.3f}  {s4[l]['f1']:.3f}  (n={sl[l]['support']})")

    if a.write_test:
        write_test(vocab, data)


def write_test(vocab, data):
    test = D.load_all(data, "test")
    lex_pred = json.loads((ROOT / "baselines/lexical_test_predictions.json").read_text())
    lP = np.load(ROOT / "baselines/lexical_test_proba.npy")
    nn = {}
    for n in NEURAL:
        with zipfile.ZipFile(ROOT / f"subs/{n}.zip") as z:
            nn[n] = json.loads(z.read("prediction.json"))
    nP = [stack_probs(json.loads((run_dir(n) / "test_probs.json").read_text()), test)
          for n in NEURAL]

    chk = argmax_pred(nP[0], test, vocab)
    agr = np.mean([x == y for d in test for x, y in zip(chk[d.id], nn[NEURAL[0]][d.id])])
    print(f"test sanity: {NEURAL[0]} marginal-argmax vs its submission agree {agr:.3f}")
    outs = {"lexical_vote4": vote([nn[n] for n in NEURAL] + [lex_pred]),
            "lexical_soft4": argmax_pred(soft(nP, lP, 0.25), test, vocab)}
    for name, pred in outs.items():
        errs = validate_submission(pred, test, vocab)
        if errs:
            raise SystemExit(f"{name} invalid:\n" + "\n".join(errs))
        zp = ROOT / "subs" / f"{name}.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("prediction.json", json.dumps(pred))
        print(f"wrote {zp}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pcr import citator as Z, data as D, metrics as M, xenc as X

ROOT = Path(__file__).resolve().parent
CE = ROOT / "runs" / "xenc"


def main():
    ap = argparse.ArgumentParser(description="Blend the cross-encoder with the lexical ranker; optional citator promotion")
    ap.add_argument("--w", type=float, default=0.12, help="weight on the cross-encoder score")
    ap.add_argument("--shortlist", type=int, default=30)
    ap.add_argument("--citator", action="store_true", help="move candidates whose CITATOR INFO names the query to the front")
    ap.add_argument("--sweep", action="store_true", help="print the dev curve over w and exit")
    ap.add_argument("--out", default=None, help="zip to write, e.g. subs/xenc_fused_w12_cit.zip")
    a = ap.parse_args()

    q_dev, pool_dev = D.load("dev")
    short_dev = X.shortlist("dev", a.shortlist)
    ce_dev = json.loads((CE / "ce_dev.json").read_text())

    if a.sweep:
        for w in (0.0, 0.05, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.5, 0.7, 1.0):
            print(f"w={w:.2f}  {M.report(M.score(X.fuse(ce_dev, short_dev, w), q_dev))}")
        return

    r = X.fuse(ce_dev, short_dev, a.w)
    if a.citator:
        r, n = Z.promote(r, pool_dev)
        print(f"citator matches for {n}/{len(q_dev)} dev queries")
    print(f"dev w={a.w}: {M.report(M.score(r, q_dev))}")

    if a.out:
        q_te, pool_te = D.load("test")
        ce_te = json.loads((CE / "ce_test.json").read_text())
        r = X.fuse(ce_te, X.shortlist("test", a.shortlist), a.w)
        if a.citator:
            r, n = Z.promote(r, pool_te)
            print(f"citator matches for {n}/{len(q_te)} test queries")
        print("wrote", M.write_submission(r, q_te, pool_te, ROOT / a.out))


if __name__ == "__main__":
    main()

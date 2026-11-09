from __future__ import annotations

import argparse
import json
from pathlib import Path

from pcr import data as D, metrics as M

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description="Submission from the lexical ranker's stored scores (runs/lex/scores_<split>.json)")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--out", default=None, help="zip to write; omit to only score")
    a = ap.parse_args()

    S = json.loads((ROOT / f"runs/lex/scores_{a.split}.json").read_text())
    q, pool = D.load(a.split)
    ranked = {qid: [c for c, _ in sorted(S[qid].items(), key=lambda kv: -kv[1]["ltr"])] for qid in q}
    if a.split != "test":
        print(f"{a.split}: {M.report(M.score(ranked, q))}")
    if a.out:
        print("wrote", M.write_submission(ranked, q, pool, ROOT / a.out))


if __name__ == "__main__":
    main()

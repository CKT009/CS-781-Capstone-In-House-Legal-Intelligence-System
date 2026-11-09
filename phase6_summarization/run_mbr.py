from __future__ import annotations

import argparse
import json
from pathlib import Path

from summ import data as D, metrics as M

RUNS = ["full", "full_6ep", "qlora", "pfeiffer", "pfeiffer_6ep", "houlsby", "lorma", "lora",
        "lora_chunk", "lora_lead", "kd", "kd_control", "adalora", "prefix", "prompt"]


def cut(s, n):
    return " ".join(s.split()[:n])


def select(preds_by_run, ids, n=400):
    # BERTScore truncates at 512 tokens, so agreement is measured on the first 400 words only
    out = {}
    for i in ids:
        c = [cut(p[i], n) for p in preds_by_run]
        agree = [sum(M.rouge_n(c[a], c[b], 1) + M.rouge_n(c[a], c[b], 2)
                     for b in range(len(c)) if b != a) for a in range(len(c))]
        out[i] = preds_by_run[max(range(len(c)), key=agree.__getitem__)][i]
    return out


def main():
    ap = argparse.ArgumentParser(description="Consensus (MBR) choice among the runs' generated summaries")
    ap.add_argument("--runs", nargs="+", default=RUNS)
    ap.add_argument("--out", default="subs/mbr_all15.zip")
    ap.add_argument("--val", action="store_true", help="also score the selection on val500")
    a = ap.parse_args()

    root = Path("runs")
    test_ids = [str(d["id"]) for d in D.load("test")]
    test = select([json.loads((root / r / "test_preds.json").read_text()) for r in a.runs], test_ids)
    print(M.write_submission(test, test_ids, Path(a.out)))

    if a.val:
        _, val = D.split()
        ids = [str(d["id"]) for d in val]
        sel = select([json.loads((root / r / "val_preds.json").read_text()) for r in a.runs], ids)
        print(M.report(M.evaluate([sel[i] for i in ids], [d["summary"] for d in val])))


if __name__ == "__main__":
    main()

import argparse, json, os, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default=None,
                    help="default: best.pt, or last.pt for a final-fit run")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dev", action="store_true", help="score dev instead of submitting")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-probs", action="store_true",
                    help="also write <split>_probs.json + crf.json for ensemble_soft.py")
    ap.add_argument("--probs-only", action="store_true",
                    help="write dev_probs.json, test_probs.json, crf.json and nothing else")
    a = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    sys.path.insert(0, str(Path(__file__).parent))

    import torch
    from rr import data as D
    from rr.predict import (load_tagger, tag, write_submission, tag_probs, crf_params,
                            is_final_fit)
    from rr.metrics import (score, report, score_by_domain, domain_report,
                            confusion, confusion_report)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, vocab = load_tagger(a.run, a.ckpt, dev)
    run = Path(a.run)
    ff = is_final_fit(run)
    sfx = "_CONTAMINATED" if ff else ""
    if ff:
        print("!" * 78 + "\n!! FINAL-FIT run: trained on train+dev. Any dev score is "
              "CONTAMINATED.\n" + "!" * 78)

    def save_probs(split, docs):
        probs = tag_probs(model, docs, cfg, vocab, dev)
        name = f"{split}_probs{sfx if split == 'dev' else ''}.json"
        (run / name).write_text(json.dumps(probs))
        cp = crf_params(model)
        (run / "crf.json").write_text(json.dumps({"vocab": vocab, "crf": cp}))
        print(f"-> {run / name}  ({len(probs)} docs)\n-> {run / 'crf.json'}")

    if a.probs_only:
        for split in ("dev", "test"):
            docs = D.load_all(cfg.data_dir, split, cfg.domains)
            print(f"[{split}] {len(docs)} documents, {sum(len(d) for d in docs)} sentences")
            save_probs(split, docs)
        return

    if a.dev:
        docs = D.load_all(cfg.data_dir, "dev", cfg.domains)
        print(f"[dev] {len(docs)} documents, {sum(len(d) for d in docs)} sentences")
        pred = tag(model, docs, cfg, vocab, dev)
        gold = {d.id: d.labels for d in docs}
        s = score(pred, gold, vocab)
        print("\n" + report(s))


        per = score_by_domain(pred, docs, vocab)
        print("\n=== per domain ===\n" + domain_report(per))
        pairs = confusion(pred, gold, vocab, 12)
        print("\n=== most frequent confusions ===\n" + confusion_report(pairs, gold))
        if ff:
            print("\n*** CONTAMINATED: dev was training data for this run. ***")
        (run / f"domain_report{sfx}.txt").write_text(domain_report(per))
        (run / f"confusion{sfx}.txt").write_text(confusion_report(pairs, gold))
        (run / f"dev_report{sfx}.json").write_text(json.dumps(
            {**s, "by_domain": {k: {kk: vv for kk, vv in v.items() if kk != "vocab"}
                                for k, v in per.items()},
             "confusions": [[list(a), b] for a, b in pairs]}, indent=2, default=str))
        (run / f"dev_predictions{sfx}.json").write_text(json.dumps(pred))
        print(f"\n-> {run/f'dev_report{sfx}.json'}\n-> {run/f'dev_predictions{sfx}.json'}")
        if a.save_probs:
            save_probs("dev", docs)
        return

    docs = D.load_all(cfg.data_dir, "test", cfg.domains)
    print(f"[test] {len(docs)} documents, {sum(len(d) for d in docs)} sentences")
    pred = tag(model, docs, cfg, vocab, dev)
    if a.save_probs:
        save_probs("test", docs)
    zp = write_submission(pred, docs, vocab, a.out or run / "submission")
    import shutil
    Path("subs").mkdir(exist_ok=True)
    shutil.copy2(zp, Path("subs") / f"{run.name}.zip")
    print(f"[ok] subs/{run.name}.zip\n\nUpload that zip.")


if __name__ == "__main__":
    main()

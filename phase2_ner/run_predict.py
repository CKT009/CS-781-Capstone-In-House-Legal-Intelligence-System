from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--fold", default=None,
                    help="with --dev, score THIS fold instead of cfg.dev_folds. Use it to "
                         "produce training-fold predictions for boosting; note those are "
                         "resubstitution predictions (the model trained on them), so the "
                         "error rates are optimistic and only their ORDERING is meaningful.")
    ap.add_argument("--dev", action="store_true",
                    help="score the dev fold locally instead of writing a submission")
    ap.add_argument("--out", default="subs")
    a = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    sys.path.insert(0, str(Path(__file__).parent))

    import torch
    from ner import data as D
    from ner.metrics import strict_prf, partial_credit_report, format_report
    from ner.predict import load_tagger, tag_documents, write_submission

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = Path(a.run)
    model, cfg = load_tagger(run, a.ckpt, device)

    if a.dev:
        texts, golds = {}, {}
        for f in ([a.fold] if a.fold else cfg.dev_folds):
            t, g = D.load_fold(cfg.data_dir, f)
            texts.update(t)
            golds.update(g)
        print(f"[dev] {len(texts)} documents, "
              f"{sum(len(v) for v in golds.values())} gold spans")
        preds = tag_documents(model, texts, cfg, device, a.batch)
        res = strict_prf(preds, golds)
        res["partial"] = partial_credit_report(preds, golds)
        print("\n" + format_report(res))
        print("\n[dev] error breakdown (diagnostic, not the metric):")
        for k, v in res["partial"].items():
            if isinstance(v, (int, float)):
                print(f"    {k:16s} {v}")
        (run / "dev_report.json").write_text(json.dumps(res, indent=2, default=str))


        _name = "train_predictions.json" if a.fold and a.fold in cfg.train_folds \
            else "dev_predictions.json"
        (run / _name).write_text(json.dumps(
            {d: [list(s) for s in v] for d, v in preds.items()}))
        print(f"\n-> {run / 'dev_report.json'}")
        print(f"-> {run / _name}")
        return


    texts, _ = D.load_fold(cfg.data_dir, "fold3")
    print(f"[test] {len(texts)} documents")
    preds = tag_documents(model, texts, cfg, device, a.batch)
    zp = write_submission(preds, run / "submission", texts)


    out = Path(a.out)
    out.mkdir(exist_ok=True)
    dst = out / f"{run.name}.zip"
    shutil.copy2(zp, dst)
    state = __import__("torch").load(run / a.ckpt, map_location="cpu",
                                     weights_only=False)
    (out / f"{run.name}.log").write_text(json.dumps({
        "run": str(run), "ckpt": str(run / a.ckpt),
        "epoch": state["epoch"], "step": state["step"],
        "dev_macro_f1": state["best"]["macro_f1"],
        "backbone": cfg.backbone, "peft": cfg.peft, "peft_kw": cfg.peft_kw,
        "use_crf": cfg.use_crf, "class_weight": cfg.class_weight,
        "n_docs": len(preds), "n_spans": sum(len(v) for v in preds.values()),
    }, indent=2))
    print(f"\n[ok] {dst}")
    print(f"[ok] {out / f'{run.name}.log'}  (dev macro-F1 "
          f"{state['best']['macro_f1']:.4f} @ epoch {state['epoch']})")
    print("\nUpload that zip. Codabench takes 15-30 min and caps at 10/day.")


if __name__ == "__main__":
    main()

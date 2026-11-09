from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

    import numpy as np
    import torch
    from lsi import data as D
    from lsi.metrics import decide, write_submission
    from lsi.train import TrainConfig, build_model, predict_probs

    run = Path(a.run)
    state = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    fields = TrainConfig.__dataclass_fields__
    cfg = TrainConfig(**{k: v for k, v in state["config"].items() if k in fields})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.peft == "distill":
        from lsi.distill import build_student
        model = build_student(cfg, device)
    else:
        model = build_model(cfg, device)
    model.load_state_dict(state["model"])

    docs = D.load("test")
    tok = D.tokens("test")
    P = predict_probs(model, [(tok[d.id], None) for d in docs], cfg, device)
    np.save(run / "test_probs.npy", P.astype(np.float32))
    pred = decide(P, state["thresholds"])
    z = write_submission(pred, [d.id for d in docs], ROOT / "subs" / f"{a.name or run.name}.zip")
    print(f"[ok] {z}  ({len(docs)} docs, {pred.sum(1).mean():.2f} statutes/doc predicted; "
          f"dev gold averages 3.92)")


if __name__ == "__main__":
    main()

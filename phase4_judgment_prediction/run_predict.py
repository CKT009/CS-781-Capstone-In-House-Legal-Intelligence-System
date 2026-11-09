from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--name", default=None, help="zip name (default: run name)")
    a = ap.parse_args()
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

    import numpy as np
    import torch
    from cjpe import data as D
    from cjpe.metrics import write_submission
    from cjpe.train import TrainConfig, build_model, predict_probs

    run = Path(a.run)
    state = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    fields = TrainConfig.__dataclass_fields__
    cfg = TrainConfig(**{k: v for k, v in state["config"].items() if k in fields})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.peft == "distill":
        from cjpe.distill import build_student
        model = build_student(cfg, device)
    else:
        model = build_model(cfg, device)
    model.load_state_dict(state["model"])
    thr = state["thr"] if a.thr is None else a.thr

    docs = D.load("test")
    tok = D.tokens("test")
    p1 = predict_probs(model, [(tok[d.id], None) for d in docs], cfg, device)
    np.save(run / "test_probs.npy", p1)
    ids = [d.id for d in docs]
    pred = {i: int(p >= thr) for i, p in zip(ids, p1)}
    z = write_submission(pred, ids, ROOT / "subs" / f"{a.name or run.name}.zip")
    rate = sum(pred.values()) / len(pred)
    print(f"[ok] {z}  ({len(ids)} cases, threshold {thr:.2f}, predicted 1 for {rate:.1%})")
    if not 0.3 <= rate <= 0.7:
        print(f"[warn] predicted-1 rate {rate:.1%} is far from dev/test's ~50% balance")


if __name__ == "__main__":
    main()

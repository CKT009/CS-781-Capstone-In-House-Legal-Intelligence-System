from __future__ import annotations

import argparse
import json
import os
import random
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--name", default="xenc")
    ap.add_argument("--peft", default="full")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--groups-per-step", type=int, default=2)
    ap.add_argument("--n-neg", type=int, default=7)
    ap.add_argument("--shortlist", type=int, default=30)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

    import numpy as np
    import torch
    import torch.nn.functional as F
    from pcr import data as D, metrics as M
    from pcr import xenc as X
    from pcr.backbones import load_backbone
    from pcr.peft import apply_peft, trainable_report

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable -- not falling back to the CPU")
    dev = "cuda"
    name = a.name + ("_smoke" if a.smoke else "")
    out = X.ROOT / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=1))
    log = open(out / "log.jsonl", "w")
    rng = random.Random(0); torch.manual_seed(0)

    qtr, _ = D.load("train"); qdv, pdv = D.load("dev"); qte, pte = D.load("test")
    opens = {**X.openings("train"), **X.openings("dev"), **X.openings("test")}
    s_tr, s_dv, s_te = (X.shortlist(s, a.shortlist) for s in ("train", "dev", "test"))
    if a.smoke:
        qdv = dict(list(qdv.items())[:12]); s_dv = {k: s_dv[k] for k in qdv}

    bb = load_backbone("../checkpoints/B_grown", "best.pt", device="cpu")
    model = X.CrossEncoder(bb)
    apply_peft(model, a.peft)
    model.to(dev)
    if a.peft == "full":
        model.backbone.gradient_checkpointing(True)
    rep = trainable_report(model)
    print(f"[xenc] {a.peft}: {rep['trainable']:,} / {rep['total']:,} trainable ({rep['pct']:.2f}%)", flush=True)

    head = [p for n, p in model.named_parameters() if n.startswith("head.")]
    body = [p for n, p in model.named_parameters() if not n.startswith("head.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": body, "lr": a.lr}, {"params": head, "lr": a.head_lr}],
                            weight_decay=0.01)
    G = X.groups(qtr, s_tr, a.n_neg, rng)
    if a.smoke:
        G = G[:40]
    total = max(1, int(len(G) * a.epochs / a.groups_per_step))
    warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    print(f"[data] {len(G):,} training groups (1 relevant + {a.n_neg} lexical negatives), "
          f"{total:,} steps | dev {len(qdv)} queries x top {a.shortlist}", flush=True)

    best, t0 = -1.0, time.time()

    def evaluate(step):
        nonlocal best
        ce = X.score_pairs(model, opens, s_dv, dev)
        m = M.score(X.fuse(ce, s_dv, 1.0), qdv)
        rec = {"step": step, "f1": m["f1"], "p": m["p"], "r": m["r"], "min": round((time.time() - t0) / 60, 1)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"[eval] step {step}: cross-encoder alone {M.report(m)} | {rec['min']} min", flush=True)
        if m["f1"] > best:
            best = m["f1"]
            torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                       out / "best.pt")

    evaluate(0)
    model.train()
    step = 0
    for i in range(0, total * a.groups_per_step, a.groups_per_step):
        batch = [G[j % len(G)] for j in range(i, i + a.groups_per_step)]
        seqs = [X.pair(opens[q], opens[c]) for q, pos, negs in batch for c in [pos] + negs]
        x, m = X.batchify(seqs, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s = model(x, m).float()
        s = s.view(len(batch), -1)
        loss = F.cross_entropy(s, torch.zeros(len(batch), dtype=torch.long, device=dev))
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 25 == 0:
            log.write(json.dumps({"step": step, "loss": round(loss.item(), 4)}) + "\n")
            print(f"[train] step {step}/{total} loss {loss.item():.4f} | "
                  f"{torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
        if step % a.eval_every == 0:
            evaluate(step)
    if step % a.eval_every:
        evaluate(step)


    sd = torch.load(out / "best.pt")
    model.load_state_dict(sd, strict=False)
    ce_dv = X.score_pairs(model, opens, s_dv, dev)
    ce_te = X.score_pairs(model, opens, s_te, dev)
    json.dump(ce_dv, open(out / "ce_dev.json", "w")); json.dump(ce_te, open(out / "ce_test.json", "w"))
    table = {w: M.score(X.fuse(ce_dv, s_dv, w), qdv) for w in (1.0, 0.7, 0.5, 0.3, 0.0)}
    for w, m in table.items():
        print(f"[fuse] w={w}: {M.report(m)}", flush=True)
    w_best = max((0.7, 0.5, 0.3), key=lambda w: table[w]["f1"])
    M.write_submission(X.fuse(ce_te, s_te, 1.0), qte, pte, X.ROOT / "subs" / f"{name}.zip")
    M.write_submission(X.fuse(ce_te, s_te, w_best), qte, pte, X.ROOT / "subs" / f"{name}_fused.zip")
    summ = {"name": name, "peft": a.peft, "trainable": rep["trainable"], "total": rep["total"],
            "dev_alone": table[1.0], "dev_fused": table[w_best], "w_fused": w_best,
            "dev_lexical": table[0.0], "wall_min": round((time.time() - t0) / 60, 1),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(f"[done] {name}: alone {M.report(table[1.0])} | fused (w={w_best}) {M.report(table[w_best])}", flush=True)


if __name__ == "__main__":
    main()

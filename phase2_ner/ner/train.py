from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ner import data as D
from ner.metrics import strict_prf, partial_credit_report, format_report
from ner.model import NERTagger, class_weights_from_counts
from ner.peft import apply_peft, trainable_report, describe


@dataclass
class TrainConfig:

    name: str = "run"
    backbone: str = "../checkpoints/B_encoder"
    ckpt: str = "best.pt"
    peft: str = "full"
    peft_kw: dict = field(default_factory=dict)
    use_crf: bool = False


    data_dir: str = "data/Phase_2_LNER"
    train_folds: tuple = ("fold1",)
    dev_folds: tuple = ("fold2",)
    max_len: int = 512
    stride: int = 128
    boost_from: str = ""


    bootstrap_seed: int = 0


    class_weight: str = "inv_sqrt"
    class_weight_cap: float = 20.0


    epochs: int = 5
    batch_size: int = 8
    grad_accum: int = 1
    lr: float = 3e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_frac: float = 0.1
    grad_clip: float = 1.0
    dropout: float = 0.1
    amp: str = "bf16"
    grad_checkpoint: bool = False
    seed: int = 1337


    out_dir: str = "runs"
    eval_every_epochs: float = 1.0
    log_every: int = 20
    device: str = "cuda"
    num_workers: int = 2

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.name


def seed_all(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log(run_dir, **kv):
    kv["time"] = time.time()
    with open(Path(run_dir) / "log.jsonl", "a") as f:
        f.write(json.dumps(kv, default=str) + "\n")


def build_optimizer(model, cfg: TrainConfig):
    head, bb_decay, bb_nodecay = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("head") or "crf" in n:
            head.append(p)
        elif p.ndim < 2 or "embed" in n or "norm" in n or n.endswith(".bias"):
            bb_nodecay.append(p)
        else:
            bb_decay.append(p)
    groups = [
        {"params": head,       "lr": cfg.head_lr, "weight_decay": cfg.weight_decay},
        {"params": bb_decay,   "lr": cfg.lr,      "weight_decay": cfg.weight_decay},
        {"params": bb_nodecay, "lr": cfg.lr,      "weight_decay": 0.0},
    ]
    groups = [g for g in groups if g["params"]]
    for g in groups:
        g["initial_lr"] = g["lr"]
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)


def lr_scale(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return max(0.0, 1.0 - p)


@torch.no_grad()
def evaluate(model, chunks_by_doc, texts, golds, cfg, device) -> dict:
    model.eval()
    preds = {}
    flat = [c for cs in chunks_by_doc.values() for c in cs]
    order = {id(c): i for i, c in enumerate(flat)}
    tag_out = [None] * len(flat)

    loader = DataLoader(flat, batch_size=cfg.batch_size, shuffle=False,
                        collate_fn=D.collate, num_workers=0)
    i = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        am = batch["attention_mask"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=device.startswith("cuda") and cfg.amp != "off"):
            tags = model.predict_tags(ids, am)
        n = ids.shape[0]
        for b in range(n):
            L = int(am[b].sum())
            tag_out[i + b] = tags[b, :L].tolist()
        i += n

    for doc_id, cs in chunks_by_doc.items():
        preds[doc_id] = D.decode_chunks(
            texts[doc_id], cs, [tag_out[order[id(c)]] for c in cs])

    res = strict_prf(preds, golds)
    res["partial"] = partial_credit_report(preds, golds)
    model.train()
    return res, preds


def train(cfg: TrainConfig, cache: dict | None = None):
    seed_all(cfg.seed)
    run = cfg.run_dir
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
    device = cfg.device if torch.cuda.is_available() else "cpu"


    from tokenizers import Tokenizer
    cache = {} if cache is None else cache
    if "tok" not in cache:
        cache["tok"] = Tokenizer.from_file("artifacts/tokenizer_16k/tokenizer.json")
    tok = cache["tok"]

    from ner.backbones import load_backbone
    bb = load_backbone(cfg.backbone, cfg.ckpt, device="cpu",
                       state_cache=cache.setdefault("_ckpt", {}))
    max_len = min(cfg.max_len, bb.max_positions)
    if max_len < cfg.max_len:
        print(f"[data] max_len clamped {cfg.max_len} -> {max_len} "
              f"(the backbone was only trained at {bb.max_positions} positions)")

    def prep(folds):
        texts, golds, chunks = {}, {}, {}
        for f in folds:
            t, g = D.load_fold(cfg.data_dir, f)
            texts.update(t)
            golds.update(g)
            for doc_id, txt in t.items():
                chunks[doc_id] = D.encode_document(
                    txt, g.get(doc_id, []), tok, max_len=max_len, stride=cfg.stride)
        return texts, golds, chunks

    def prep_cached(folds):


        key = (cfg.data_dir, tuple(folds), max_len, cfg.stride)
        if key not in cache.setdefault("_data", {}):
            cache["_data"][key] = prep(folds)
        return cache["_data"][key]

    tr_texts, tr_golds, tr_chunks = prep_cached(cfg.train_folds)
    dv_texts, dv_golds, dv_chunks = prep_cached(cfg.dev_folds)
    if cfg.bootstrap_seed:


        import random as _rnd
        _docs = sorted(tr_chunks)
        _rng = _rnd.Random(cfg.bootstrap_seed)
        if cfg.boost_from:


            import json as _j
            from pathlib import Path as _P
            from ner.metrics import strict_prf as _prf
            _pf = _P(cfg.boost_from) / "train_predictions.json"
            if not _pf.exists():
                _pf = _P(cfg.boost_from) / "dev_predictions.json"
            _pred = {k: [tuple(x[:3]) for x in v] for k, v in
                     _j.loads(_pf.read_text()).items()}


            _cov = len(set(_docs) & set(_pred))
            if _cov < 0.5 * len(_docs):
                raise RuntimeError(
                    f"boost_from={cfg.boost_from}: {_pf.name} covers {_cov}/{len(_docs)} "
                    f"training documents. Boosting needs predictions on the TRAINING fold "
                    f"({list(cfg.train_folds)}), not the dev fold. Generate them with "
                    f"`python run_predict.py --run {cfg.boost_from} --dev --fold "
                    f"{cfg.train_folds[0]}` and re-run.")
            _w = []
            for d in _docs:
                g = {d: tr_golds.get(d, [])}
                pr = {d: _pred.get(d, [])}
                f1 = _prf(pr, g)["macro_f1"] if g[d] else 0.0
                _w.append(max(1.0 - f1, 0.05))
            _tot = sum(_w)
            print(f"[data] BOOSTING from {cfg.boost_from}: document weights "
                  f"min {min(_w)/_tot*len(_docs):.2f}x max {max(_w)/_tot*len(_docs):.2f}x mean 1.00x")
            _picked = _rng.choices(_docs, weights=_w, k=len(_docs))
        else:
            _picked = [_rng.choice(_docs) for _ in _docs]
        train_flat = [c for d in _picked for c in tr_chunks[d]]
        print(f"[data] BOOTSTRAP seed {cfg.bootstrap_seed}: "
              f"{len(set(_picked))}/{len(_docs)} unique documents "
              f"({100*len(set(_picked))/len(_docs):.0f}%), "
              f"{len(train_flat)} chunks (vs {sum(len(c) for c in tr_chunks.values())} "
              f"without resampling)")
    else:
        train_flat = [c for cs in tr_chunks.values() for c in cs]
    overlap = set(cfg.train_folds) & set(cfg.dev_folds)
    if overlap:
        print(f"[data] *** DEV CONTAMINATED: {sorted(overlap)} is in BOTH train and dev. "
              f"Every macro-F1 this run reports is measured on data it trained on and "
              f"means nothing. That is the accepted price of training a submission model "
              f"on all the data -- fix the epoch count and any ensemble threshold from "
              f"an honest run first, and never quote these numbers.")
    print(f"[data] train {len(tr_texts)} docs -> {len(train_flat)} chunks | "
          f"dev {len(dv_texts)} docs | {sum(len(v) for v in tr_golds.values())} gold spans")

    counts = {}
    for c in train_flat:
        for t in c.tag_ids:
            if t != D.IGNORE:
                counts[t] = counts.get(t, 0) + 1
    weights = class_weights_from_counts(counts, len(D.TAGS), cfg.class_weight,
                                        cap=cfg.class_weight_cap)
    n_capped = int((weights >= cfg.class_weight_cap - 1e-6).sum())
    print(f"[data] class weights: min {weights.min():.2f} max {weights.max():.2f} "
          f"({n_capped} tag(s) at the cap)")
    if n_capped:
        uncapped = class_weights_from_counts(counts, len(D.TAGS), cfg.class_weight,
                                             cap=float("inf"))
        print(f"[data] {n_capped}/{len(D.TAGS)} tags share the cap; uncapped "
              f"{cfg.class_weight} would reach {uncapped.max():.1f}x. "
              f"If precision ends up well below recall, LOWER this.")


    model = NERTagger(bb, n_tags=len(D.TAGS), dropout=cfg.dropout,
                      class_weights=weights, use_crf=cfg.use_crf).to(device)
    apply_peft(model, cfg.peft, **cfg.peft_kw)
    if cfg.grad_checkpoint:
        model.backbone.gradient_checkpointing(True)
    rep = trainable_report(model)
    print(f"[peft] {describe(cfg.peft, **cfg.peft_kw)}")
    print(f"[peft] trainable {rep['trainable']:,} / {rep['total']:,} "
          f"({rep['pct']:.2f}%)")

    opt = build_optimizer(model, cfg)
    loader = DataLoader(train_flat, batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=D.collate, num_workers=cfg.num_workers,
                        drop_last=False)
    steps_per_epoch = math.ceil(len(loader) / cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    warmup = int(total_steps * cfg.warmup_frac)
    print(f"[train] {total_steps} optimiser steps "
          f"({cfg.epochs} epochs x {steps_per_epoch}), warmup {warmup}")

    amp_dtype = torch.bfloat16 if cfg.amp == "bf16" else torch.float16
    use_amp = device.startswith("cuda") and cfg.amp != "off"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and cfg.amp == "fp16")

    best = {"macro_f1": -1.0, "epoch": None, "step": None}
    step, t0 = 0, time.time()
    model.train()

    for epoch in range(1, cfg.epochs + 1):
        opt.zero_grad(set_to_none=True)
        run_loss, seen = 0.0, 0
        for i, batch in enumerate(loader):
            ids = batch["input_ids"].to(device, non_blocking=True)
            am = batch["attention_mask"].to(device, non_blocking=True)
            lab = batch["labels"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                loss = model(ids, am, labels=lab)["loss"]
            scaler.scale(loss / cfg.grad_accum).backward()
            run_loss += float(loss.detach())
            seen += 1

            if (i + 1) % cfg.grad_accum == 0 or i + 1 == len(loader):
                if cfg.grad_clip:
                    scaler.unscale_(opt)
                    gnorm = torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        cfg.grad_clip)
                else:
                    gnorm = torch.tensor(float("nan"))
                scale = lr_scale(step, total_steps, warmup)
                for g in opt.param_groups:
                    g["lr"] = g["initial_lr"] * scale
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1

                if step % cfg.log_every == 0:
                    log(run, step=step, epoch=epoch, train_loss=run_loss / max(seen, 1),
                        lr=opt.param_groups[0]["lr"], grad_norm=float(gnorm))
                    print(f"  step {step}/{total_steps}  loss {run_loss/max(seen,1):.4f}"
                          f"  |g| {float(gnorm):.2f}")
                    run_loss, seen = 0.0, 0


        res, preds = evaluate(model, dv_chunks, dv_texts, dv_golds, cfg, device)
        log(run, step=step, epoch=epoch, **{k: v for k, v in res.items()
                                            if k not in ("per_label", "partial")},
            per_label=res["per_label"], partial=res["partial"])
        print(f"\n[eval] epoch {epoch}  "
              f"macro-F1 {res['macro_f1']:.4f}  macro-P {res['macro_p']:.4f}  "
              f"macro-R {res['macro_r']:.4f}  (micro-F1 {res['micro_f1']:.4f})")
        print(format_report(res))

        if res["macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": res["macro_f1"], "epoch": epoch, "step": step}
            _save(run / "best.pt", model, opt, cfg, step, epoch, best)
            (run / "dev_predictions.json").write_text(json.dumps(preds))
            print(f"[ckpt] new best macro-F1 {res['macro_f1']:.4f} -> best.pt")
        _save(run / "last.pt", model, opt, cfg, step, epoch, best)

    summary = {
        "name": cfg.name, "backbone": cfg.backbone, "peft": cfg.peft,
        "train_folds": list(cfg.train_folds), "dev_folds": list(cfg.dev_folds),
        "dev_contaminated": bool(set(cfg.train_folds) & set(cfg.dev_folds)),
        "peft_kw": cfg.peft_kw, "use_crf": cfg.use_crf,
        "best": best, "epochs": cfg.epochs, "steps": step,
        "wall_min": round((time.time() - t0) / 60, 1),
        "trainable": rep["trainable"], "total": rep["total"], "trainable_pct": rep["pct"],
        "peak_vram_mb": (round(torch.cuda.max_memory_allocated() / 2**20, 1)
                         if device.startswith("cuda") else None),
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps(summary, indent=2))
    return summary


def _save(path, model, opt, cfg, step, epoch, best):
    path = Path(path)
    payload = {"model": model.state_dict(), "opt": opt.state_dict(),
               "cfg": asdict(cfg), "step": step, "epoch": epoch, "best": best}
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)

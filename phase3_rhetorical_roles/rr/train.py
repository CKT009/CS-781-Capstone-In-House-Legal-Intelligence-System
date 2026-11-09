from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from rr import data as D
from rr.metrics import score as score_fn, report as report_fn


@dataclass
class TrainConfig:
    name: str = "run"
    backbone: str = "../checkpoints/B_grown"
    ckpt: str = "best.pt"
    tokenizer: str = "artifacts/tokenizer_16k/tokenizer.json"
    data_dir: str = "data/Phase_3_RR"
    out_dir: str = "runs"
    domains: tuple = D.DOMAINS


    peft: str = "lora"


    peft_kw: dict = field(default_factory=dict)
    doc_layers: int = 2
    doc_heads: int = 8
    use_crf: bool = True
    dropout: float = 0.1
    max_sent_len: int = 128
    max_doc_sents: int = 256
    doc_stride: int = 64
    sent_batch: int = 32


    epochs: int = 20
    lr: float = 3e-4
    head_lr: float = 1e-3
    train_doc: bool = False


    ctx_tokens: int = 0

    doc_lr: float = 0.0
    weight_decay: float = 0.01
    warmup_frac: float = 0.1
    grad_clip: float = 1.0
    grad_accum: int = 2
    class_weight: str = "inv_sqrt"
    class_weight_cap: float = 20.0
    aux_ce: float = 0.5

    class_weight_boost: dict = field(default_factory=dict)


    amp: str = "bf16"
    seed: int = 1337
    bootstrap_seed: int = 0
    device: str = "cuda"
    log_every: int = 20


    final_fit: bool = False


    def to_dict(self) -> dict:
        d = asdict(self)
        if not d.get("final_fit"):
            d.pop("final_fit", None)
        return d

    @property
    def backbone_name(self) -> str:
        return Path(self.backbone).name

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.backbone_name / self.name

    def save(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text(json.dumps(self.to_dict(), indent=2,
                                                            default=str))


def seed_all(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def class_weights(counts, n, mode, cap):
    if mode == "none":
        return None
    w = np.ones(n, dtype=np.float32)
    tot = sum(counts.values()) or 1
    for i in range(n):
        c = counts.get(i, 0)
        if c == 0:
            w[i] = cap
        elif mode == "inv":
            w[i] = tot / (n * c)
        else:
            w[i] = math.sqrt(tot / (n * c))
    return np.minimum(w, cap)


_CACHE: dict = {}


def _prepare(cfg, tok, vocab, split):
    key = (cfg.data_dir, split, tuple(cfg.domains), cfg.max_sent_len)
    if key not in _CACHE:
        docs = D.load_all(cfg.data_dir, split, cfg.domains)
        _CACHE[key] = (docs, [D.encode_doc(d, tok, vocab, cfg.max_sent_len) for d in docs])
    return _CACHE[key]


def _examples(docs, encs, cfg):
    out = []
    for d, e in zip(docs, encs):
        for lo, hi in D.windows(len(d), cfg.max_doc_sents, cfg.doc_stride):
            out.append((e, lo, hi))
    return out


@torch.no_grad()
def evaluate(model, docs, encs, cfg, vocab, device):
    model.eval()
    pred = {}
    for d, e in zip(docs, encs):
        wins = D.windows(len(d), cfg.max_doc_sents, cfg.doc_stride)
        tags = [None] * len(d)
        per_win = []
        for lo, hi in wins:
            ids = torch.tensor(e.ids[lo:hi], device=device)
            msk = torch.tensor(e.mask[lo:hi], device=device)
            per_win.append(model.predict(ids, msk).tolist())
        for i in range(len(d)):
            w = D.owner(wins, i)
            lo, _ = wins[w]
            tags[i] = per_win[w][i - lo]
        pred[d.id] = [vocab[t] for t in tags]
    model.train()
    gold = {d.id: d.labels for d in docs}
    return pred, score_fn(pred, gold, vocab)


def train(cfg: TrainConfig):
    seed_all(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    cfg.save()
    from tokenizers import Tokenizer
    from rr.backbones import load_backbone
    from rr.model import RRTagger
    from rr.peft import apply_peft, trainable_report

    vocab = D.load_labels(cfg.data_dir)
    tok = Tokenizer.from_file(cfg.tokenizer)
    tr_docs, tr_encs = _prepare(cfg, tok, vocab, "train")
    dv_docs, dv_encs = _prepare(cfg, tok, vocab, "dev")
    if set(d.id for d in tr_docs) & set(d.id for d in dv_docs):
        raise RuntimeError("train and dev share document ids -- refusing to train")
    if cfg.final_fit:
        n_tr = len(tr_docs)
        tr_docs, tr_encs = list(tr_docs) + list(dv_docs), list(tr_encs) + list(dv_encs)
        print("\n" + "!" * 78)
        print(f"!! FINAL-FIT MODE: training on train ({n_tr}) + dev ({len(dv_docs)}) = "
              f"{len(tr_docs)} documents")
        print(f"!! for exactly {cfg.epochs} epochs. NO checkpoint selection; last.pt only.")
        print("!! Any dev score this run prints is CONTAMINATED (dev is training data) and")
        print("!! must not be compared with anything.")
        print("!" * 78 + "\n")

    if cfg.bootstrap_seed:
        rng = random.Random(cfg.bootstrap_seed)
        picked = [rng.randrange(len(tr_docs)) for _ in tr_docs]
        tr_docs = [tr_docs[i] for i in picked]
        tr_encs = [tr_encs[i] for i in picked]
        print(f"[data] BOOTSTRAP seed {cfg.bootstrap_seed}: "
              f"{len(set(picked))}/{len(picked)} unique documents")

    counts = {}
    for e in tr_encs:
        for l in e.labels:
            counts[l] = counts.get(l, 0) + 1
    w = class_weights(counts, len(vocab), cfg.class_weight, cfg.class_weight_cap)
    if cfg.class_weight_boost and w is not None:
        bad = sorted(set(cfg.class_weight_boost) - set(vocab))
        if bad:
            raise ValueError(f"class_weight_boost names labels not in the vocabulary: {bad}")
        for lab, mult in cfg.class_weight_boost.items():
            i = vocab.index(lab)
            print(f"[data] boost {lab}: weight {w[i]:.2f} -> {w[i]*mult:.2f} ({mult}x)")
            w[i] = w[i] * mult
    unseen = [vocab[i] for i in range(len(vocab)) if counts.get(i, 0) == 0]
    print(f"[data] train {len(tr_docs)} docs / {sum(len(d) for d in tr_docs)} sentences | "
          f"dev {len(dv_docs)} docs / {sum(len(d) for d in dv_docs)} sentences")
    if unseen:
        print(f"[data] labels with NO training support: {unseen} "
              f"(weighted at the cap; they cannot be learned, only guessed)")

    bb = load_backbone(cfg.backbone, cfg.ckpt, device="cpu")
    model = RRTagger(bb, n_labels=len(vocab), doc_layers=cfg.doc_layers,
                     doc_heads=cfg.doc_heads, dropout=cfg.dropout, use_crf=cfg.use_crf,
                     max_sents=cfg.max_doc_sents + 8, sent_batch=cfg.sent_batch,
                     ctx_tokens=cfg.ctx_tokens,
                     class_weights=w, aux_ce=cfg.aux_ce).to(device)
    apply_peft(model, cfg.peft, **cfg.peft_kw)
    rep = trainable_report(model)
    print(f"[peft] {cfg.peft}: {rep['trainable']:,} / {rep['total']:,} "
          f"({rep['pct']:.2f}%) trainable")

    if cfg.train_doc and model.doc is not None:
        for p in model.doc.parameters():
            p.requires_grad_(True)
        rep = trainable_report(model)
        print(f"[doc] DocEncoder trainable -> {rep['trainable']:,} trainable in total")
    head_names = ("head", "doc", "crf")
    doc_lr = cfg.doc_lr or cfg.head_lr
    groups = [
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and not n.startswith(head_names)], "lr": cfg.lr},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and n.startswith(("head", "crf"))], "lr": cfg.head_lr},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and n.startswith("doc")], "lr": doc_lr},
    ]
    opt = torch.optim.AdamW([g for g in groups if g["params"]],
                            weight_decay=cfg.weight_decay)
    ex = _examples(tr_docs, tr_encs, cfg)
    steps = max(1, (len(ex) * cfg.epochs) // cfg.grad_accum)
    warm = max(1, int(steps * cfg.warmup_frac))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else max(0.0, (steps - s) / max(1, steps - warm)))
    amp = device.startswith("cuda") and cfg.amp != "off"
    dt = torch.bfloat16 if cfg.amp == "bf16" else torch.float16
    print(f"[train] {len(ex)} windows/epoch x {cfg.epochs} epochs "
          f"= {steps} optimiser steps (accum {cfg.grad_accum}), warmup {warm}")

    best = {"macro_f1": -1.0, "epoch": None}
    log = (cfg.run_dir / "log.jsonl").open("w")
    t0 = time.time()
    gstep = 0
    for ep in range(1, cfg.epochs + 1):
        random.Random(cfg.seed + ep).shuffle(ex)
        model.train()
        run_loss, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, (e, lo, hi) in enumerate(ex):
            ids = torch.tensor(e.ids[lo:hi], device=device)
            msk = torch.tensor(e.mask[lo:hi], device=device)
            lab = torch.tensor(e.labels[lo:hi], device=device)
            with torch.autocast("cuda", dtype=dt, enabled=amp):
                out = model(ids, msk, labels=lab)
            (out["loss"] / cfg.grad_accum).backward()
            run_loss += out["loss"].item(); nb += 1
            if (i + 1) % cfg.grad_accum == 0 or i + 1 == len(ex):
                gn = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); gstep += 1
                if gstep % cfg.log_every == 0:
                    print(f"  step {gstep}/{steps}  loss {run_loss/max(nb,1):.4f}  "
                          f"|g| {float(gn):.2f}", flush=True)
        if cfg.final_fit:
            print(f"[epoch] {ep:2d}/{cfg.epochs}  loss {run_loss/max(nb,1):.4f}  "
                  f"(final-fit: no dev evaluation, no selection)", flush=True)
            log.write(json.dumps({"epoch": ep, "train_loss": run_loss / max(nb, 1),
                                  "final_fit": True}) + "\n"); log.flush()
            continue
        pred, s = evaluate(model, dv_docs, dv_encs, cfg, vocab, device)
        print(f"[eval] epoch {ep:2d}  macro-F1 {s['macro_f1']:.4f}  "
              f"(supported {s['macro_f1_supported']:.4f})  acc {s['accuracy']:.4f}  "
              f"loss {run_loss/max(nb,1):.4f}", flush=True)
        log.write(json.dumps({"epoch": ep, "train_loss": run_loss / max(nb, 1),
                              "macro_f1": s["macro_f1"],
                              "macro_f1_supported": s["macro_f1_supported"],
                              "accuracy": s["accuracy"]}) + "\n"); log.flush()
        if s["macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": s["macro_f1"],
                    "macro_f1_supported": s["macro_f1_supported"], "epoch": ep}
            torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(), "epoch": ep,
                        "best": best}, cfg.run_dir / "best.pt")
            (cfg.run_dir / "dev_predictions.json").write_text(json.dumps(pred))
            print(f"[ckpt] new best {s['macro_f1']:.4f} -> best.pt")
    if cfg.final_fit:
        best = {"macro_f1": None, "macro_f1_supported": None, "epoch": cfg.epochs,
                "final_fit": True}
    torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(), "epoch": cfg.epochs,
                "best": best}, cfg.run_dir / "last.pt")
    log.close()

    pred, s = evaluate(model, dv_docs, dv_encs, cfg, vocab, device)
    if cfg.final_fit:
        warn = ("*** CONTAMINATED: final-fit run, dev was TRAINING data. This is a "
                "training-fit diagnostic, not a validation score. ***")
        (cfg.run_dir / "dev_report_CONTAMINATED.txt").write_text(
            warn + "\n" + report_fn(s))
        print("\n" + warn + "\n" + report_fn(s) + "\n" + warn)
    else:
        (cfg.run_dir / "dev_report.txt").write_text(report_fn(s))
        print("\n" + report_fn(s))
    summary = {"name": cfg.name, "backbone": cfg.backbone, "peft": cfg.peft,
               "use_crf": cfg.use_crf, "doc_layers": cfg.doc_layers, "best": best,
               **({"final_fit": True, "n_train_docs": len(tr_docs)}
                  if cfg.final_fit else {}),
               "epochs": cfg.epochs, "wall_min": (time.time() - t0) / 60,
               "trainable": rep["trainable"], "total": rep["total"],
               "trainable_pct": rep["pct"],
               "peak_vram_mb": torch.cuda.max_memory_allocated() / 1e6 if amp else None}
    (cfg.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary

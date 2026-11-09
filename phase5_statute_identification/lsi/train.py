from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lsi import data as D
from lsi.metrics import evaluate, report

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class TrainConfig:
    name: str = "debug"
    backbone: str = "../checkpoints/B_grown"
    ckpt: str = "best.pt"
    out_dir: str = "runs"

    peft: str = "full"
    peft_kw: dict = field(default_factory=dict)
    gradual_stages: int = 4

    L: int = 1024
    K: int = 4
    keep: str = "ends"
    pool: str = "attn"

    epochs: float = 3.0
    batch_docs: int = 8
    grad_accum: int = 2
    lr: float = 2e-5
    head_lr: float = 1e-3
    e_lr_mult: float = 1.0
    weight_decay: float = 0.01
    warmup_frac: float = 0.06
    grad_clip: float = 1.0
    label_pw: str = "none"


    fold: int = -1
    folds_file: str = "runs/lexical_kfold/folds.json"
    eval_every: int = 1000
    max_train: int = 0
    seed: int = 1337
    device: str = "cuda"

    @property
    def run_dir(self) -> Path:
        return ROOT / self.out_dir / self.name

    def save(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text(json.dumps(asdict(self), indent=1))


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def collate(batch, cfg):
    per = []
    for ids, _ in batch:
        n = len(ids)
        per.append([(np.asarray(ids[s:s + cfg.L]), (s + min(cfg.L, n) / 2) / max(n, 1))
                    for s in D.window_starts(n, cfg.L, cfg.K, cfg.keep)])
    B, K = len(batch), max(len(p) for p in per)
    L = max(len(c) for p in per for c, _ in p)
    x = torch.zeros(B, K, L, dtype=torch.long)
    m = torch.zeros(B, K, L, dtype=torch.long)
    cm = torch.zeros(B, K, dtype=torch.bool)
    pos = torch.zeros(B, K)
    for i, p in enumerate(per):
        for k, (c, ps) in enumerate(p):
            x[i, k, :len(c)] = torch.from_numpy(c.astype(np.int64))
            m[i, k, :len(c)] = 1
            cm[i, k] = True
            pos[i, k] = ps
    y = None if batch[0][1] is None else torch.from_numpy(np.stack([l for _, l in batch]))
    return x, m, cm, pos, y


@torch.no_grad()
def predict_logits(model, items, cfg, device, batch=16):
    model.eval()
    order = np.argsort([-len(ids) for ids, _ in items], kind="stable")
    out = np.zeros((len(items), D.N_LABELS), dtype=np.float32)
    for i in range(0, len(items), batch):
        sel = order[i:i + batch]
        x, m, cm, pos, _ = collate([(items[j][0], None) for j in sel], cfg)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            lg = model(x.to(device), m.to(device), cm.to(device), pos.to(device))
        out[sel] = lg.float().cpu().numpy()
    model.train()
    return out


def predict_probs(model, items, cfg, device, batch=16):
    return 1 / (1 + np.exp(-predict_logits(model, items, cfg, device, batch)))


def build_model(cfg, device):
    from lsi.backbones import load_backbone
    from lsi.model import LSIClassifier
    from lsi.peft import apply_peft
    bb = load_backbone(cfg.backbone, cfg.ckpt, device="cpu")
    model = LSIClassifier(bb, pool=cfg.pool).to(device)
    if cfg.peft == "gradual":
        apply_peft(model, "head_only")
    elif cfg.peft == "adalora":
        from lsi.adalora import apply_adalora
        apply_adalora(model, **cfg.peft_kw)
    else:
        apply_peft(model, cfg.peft, **dict(cfg.peft_kw))
    return model


def blocks_of(model):
    return [m for m in model.backbone.modules() if m.__class__.__name__ == "Block"]


def train(cfg: TrainConfig):
    seed_all(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    cfg.save()
    from lsi.peft import trainable_report

    tr_docs, dv_docs = D.load("train"), D.load("dev")
    tok = {**D.tokens("train"), **D.tokens("dev")}
    if cfg.fold >= 0:
        folds = json.loads((ROOT / cfg.folds_file).read_text())
        pool = tr_docs + dv_docs
        tr_docs = [d for d in pool if folds[d.id] != cfg.fold]
        dv_docs = [d for d in pool if folds[d.id] == cfg.fold]
        (cfg.run_dir / "oof_ids.json").write_text(json.dumps([d.id for d in dv_docs]))
        print(f"[fold {cfg.fold}] train {len(tr_docs):,} pooled docs | held out "
              f"{len(dv_docs):,} | final checkpoint kept, test predicted at the end", flush=True)
    if cfg.max_train:
        tr_docs = random.Random(cfg.seed).sample(tr_docs, cfg.max_train)
    Ytr, Ydv = D.label_matrix(tr_docs), D.label_matrix(dv_docs)
    tr = [(tok[d.id], Ytr[i]) for i, d in enumerate(tr_docs)]
    dv = [(tok[d.id], None) for d in dv_docs]

    t_logit = None
    if cfg.peft == "distill":
        from lsi.distill import build_student, kd_loss, teacher_logits
        t_logit = teacher_logits(cfg, tr, [d.id for d in tr_docs], device)
        model = build_student(cfg, device)
        cfg.save()
        alpha = float(cfg.peft_kw.get("alpha", 0.5)); tau = float(cfg.peft_kw.get("tau", 2.0))
        print(f"[distill] alpha={alpha} tau={tau}"
              f"{' (CONTROL: gold only)' if alpha >= 1 else ''}", flush=True)
    else:
        model = build_model(cfg, device)
    ada = getattr(model, "_adalora", None) if cfg.peft == "adalora" else None
    rep = trainable_report(model)
    print(f"[peft] {cfg.peft}: {rep['trainable']:,} / {rep['total']:,} ({rep['pct']:.2f}%) "
          f"trainable | groups {rep['groups']}", flush=True)

    head_p = [p for n, p in model.named_parameters() if n.startswith("head.")]
    body_p = [p for n, p in model.named_parameters() if not n.startswith("head.")]
    if cfg.peft == "gradual":
        groups = [{"params": body_p, "lr": cfg.lr}]
    else:
        groups = [{"params": [p for p in body_p if p.requires_grad], "lr": cfg.lr}]
    groups.append({"params": head_p, "lr": cfg.head_lr})
    if cfg.peft == "adalora" and cfg.e_lr_mult != 1.0:
        e_ids = {id(p) for n, p in model.named_parameters() if n.endswith("lora_E")}
        groups[0]["params"] = [p for p in groups[0]["params"] if id(p) not in e_ids]
        groups.append({"params": [p for n, p in model.named_parameters() if n.endswith("lora_E")],
                       "lr": cfg.lr * cfg.e_lr_mult, "weight_decay": 0.0})
    opt = torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=cfg.weight_decay)

    steps_per_epoch = math.ceil(len(tr) / (cfg.batch_docs * cfg.grad_accum))
    total = max(1, int(steps_per_epoch * cfg.epochs))
    warm = max(1, int(total * cfg.warmup_frac))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))

    npos = Ytr.sum(0)
    pw = None
    if cfg.label_pw == "sqrt":
        pw = torch.from_numpy(np.sqrt((len(Ytr) - npos) / npos)).float().to(device)
    print(f"[data] train {len(tr):,} dev {len(dv):,} | statutes/doc {Ytr.sum(1).mean():.2f} | "
          f"{steps_per_epoch:,} steps/epoch, {total:,} total | view K={cfg.K} L={cfg.L} "
          f"keep={cfg.keep} pool={cfg.pool} | label pos_weight {cfg.label_pw}", flush=True)

    blocks, stage_done = blocks_of(model), 0

    def maybe_unfreeze(step):
        nonlocal stage_done
        if cfg.peft != "gradual":
            return
        S = cfg.gradual_stages
        stage = min(S, int(step / total * (S + 1)))
        while stage_done < stage:
            stage_done += 1
            k = math.ceil(len(blocks) * stage_done / S)
            for b in blocks[-k:]:
                for p in b.parameters():
                    p.requires_grad_(True)
            inner = {id(x) for b in blocks for x in b.modules()}
            for nm in model.backbone.modules():
                if nm.__class__.__name__ in ("LayerNorm", "RMSNorm") and id(nm) not in inner:
                    for p in nm.parameters():
                        p.requires_grad_(True)
            print(f"[gradual] step {step}: stage {stage_done}/{S} -> top {k} blocks, "
                  f"{trainable_report(model)['trainable']:,} trainable", flush=True)

    log = open(cfg.run_dir / "log.jsonl", "w")
    best = {"score": -1.0}
    t0 = time.time()

    def run_eval(step):
        P = predict_probs(model, dv, cfg, device)
        e = evaluate(P, Ydv)
        rec = {"step": step, "epoch": round(step / steps_per_epoch, 3),
               **{k: v for k, v in e.items() if k != "thresholds"},
               "min": round((time.time() - t0) / 60, 1)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"[eval] step {step} ep {rec['epoch']:.2f}: {report(e)} | {rec['min']} min",
              flush=True)
        if cfg.fold >= 0 or e["macro_f1_crossfit"] > best["score"]:
            best.update(score=e["macro_f1_crossfit"], step=step, e=e)
            torch.save({"model": model.state_dict(), "thresholds": e["thresholds"],
                        "global_thr": e["global_thr"], "config": asdict(cfg), "step": step},
                       cfg.run_dir / "best.pt")
            np.save(cfg.run_dir / "dev_probs.npy", P.astype(np.float32))

    step, micro, ep = 0, 0, 0
    model.train()
    order = list(range(len(tr)))
    while step < total:
        random.Random(cfg.seed + ep).shuffle(order)
        for i in range(0, len(order), cfg.batch_docs):
            maybe_unfreeze(step)
            idx = order[i:i + cfg.batch_docs]
            x, m, cm, pos, y = collate([tr[j] for j in idx], cfg)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                lg = model(x.to(device), m.to(device), cm.to(device), pos.to(device))
            y = y.to(device)
            if t_logit is not None:
                loss = kd_loss(lg.float(), t_logit[idx].to(device), y, alpha, tau)
            else:
                loss = F.binary_cross_entropy_with_logits(lg.float(), y, pos_weight=pw)
            obj = loss if ada is None else loss + ada.orth_loss()
            (obj / cfg.grad_accum).backward()
            micro += 1
            if micro % cfg.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                           cfg.grad_clip)
            opt.step(); sched.step()
            if ada is not None:
                ada.update_and_mask(step, total)
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                log.write(json.dumps({"step": step, "loss": round(loss.item(), 5),
                                      "lr": sched.get_last_lr()[0]}) + "\n")
            if step % cfg.eval_every == 0:
                run_eval(step)
            if step >= total:
                break
        ep += 1
    if step % cfg.eval_every:
        run_eval(step)
    if cfg.fold >= 0:
        te_docs, te_tok = D.load("test"), D.tokens("test")
        Pte = predict_probs(model, [(te_tok[d.id], None) for d in te_docs], cfg, device)
        np.save(cfg.run_dir / "test_probs.npy", Pte.astype(np.float32))

    e = best["e"]
    summ = {"name": cfg.name, "peft": cfg.peft, "K": cfg.K, "keep": cfg.keep, "pool": cfg.pool,
            "best_step": best["step"], "dev": {k: v for k, v in e.items() if k != "thresholds"},
            "trainable": rep["trainable"], "total": rep["total"],
            "wall_min": round((time.time() - t0) / 60, 1),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
            if device.startswith("cuda") else None}
    (cfg.run_dir / "summary.json").write_text(json.dumps(summ, indent=1))
    print(f"[done] {cfg.name}: best step {best['step']} | {report(e)}", flush=True)
    return summ

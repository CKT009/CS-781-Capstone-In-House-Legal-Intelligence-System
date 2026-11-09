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

from cjpe import data as D
from cjpe.metrics import best_threshold, prf, report

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
    K: int = 1
    keep: str = "tail"
    pool: str = "mean"

    epochs: float = 2.0
    batch_docs: int = 8
    grad_accum: int = 2
    lr: float = 2e-5
    head_lr: float = 1e-3
    e_lr_mult: float = 1.0
    weight_decay: float = 0.01
    warmup_frac: float = 0.06
    grad_clip: float = 1.0
    balance: bool = True
    grad_checkpoint: bool = False
    eval_every: int = 500
    max_train: int = 0
    redact_train: bool = False
    era_weight: bool = False
    era_clip: tuple = (0.1, 4.0)
    select_on: str = "clean"
    train_on_dev: bool = False

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


def doc_chunks(ids, cfg):
    n = len(ids)
    if n <= cfg.L:
        return [(np.asarray(ids), 0.5)]
    starts = list(range(0, n - cfg.L + 1, cfg.L))
    if starts[-1] != n - cfg.L:
        starts.append(n - cfg.L)
    if len(starts) > cfg.K:
        if cfg.keep == "tail":
            starts = starts[-cfg.K:]
        elif cfg.keep == "head":
            starts = starts[:cfg.K]
        elif cfg.keep == "ends":
            h = cfg.K // 2
            starts = starts[:h] + starts[-(cfg.K - h):]
        else:
            raise ValueError(cfg.keep)
    return [(np.asarray(ids[s:s + cfg.L]), (s + cfg.L / 2) / n) for s in starts]


def collate(batch, cfg):
    per = [doc_chunks(ids, cfg) for ids, _ in batch]
    B, K = len(batch), max(len(p) for p in per)
    L = max(len(c) for p in per for c, _ in p)
    x = torch.zeros(B, K, L, dtype=torch.long)
    m = torch.zeros(B, K, L, dtype=torch.long)
    cm = torch.zeros(B, K, dtype=torch.bool)
    pos = torch.zeros(B, K)
    for i, p in enumerate(per):
        for k, (c, ps) in enumerate(p):
            x[i, k, :len(c)] = torch.from_numpy(np.asarray(c, dtype=np.int64))
            m[i, k, :len(c)] = 1
            cm[i, k] = True
            pos[i, k] = ps
    y = torch.tensor([-1 if l is None else l for _, l in batch], dtype=torch.float)
    return x, m, cm, pos, y


@torch.no_grad()
def predict_probs(model, items, cfg, device, batch=16):
    model.eval()
    out = []
    for i in range(0, len(items), batch):
        x, m, cm, pos, _ = collate(items[i:i + batch], cfg)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            lg = model(x.to(device), m.to(device), cm.to(device), pos.to(device))
        out.append(lg.float().sigmoid().cpu())
    model.train()
    return torch.cat(out).numpy()


def build_model(cfg, device):
    from cjpe.backbones import load_backbone
    from cjpe.model import CJPEClassifier
    from cjpe.peft import apply_peft
    bb = load_backbone(cfg.backbone, cfg.ckpt, device="cpu")
    model = CJPEClassifier(bb, pool=cfg.pool).to(device)
    if cfg.peft == "gradual":
        apply_peft(model, "head_only")
    elif cfg.peft == "adalora":
        from cjpe.adalora import apply_adalora
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
    from cjpe.peft import trainable_report

    tr_docs, dv_docs = D.load("train"), D.load("dev")
    tr_tok = D.tokens("train_redacted" if cfg.redact_train else "train")
    dv_tok = D.tokens("dev")
    if cfg.max_train:
        rng = random.Random(cfg.seed)
        tr_docs = rng.sample(tr_docs, cfg.max_train)
    tr = [(tr_tok[d.id], d.label) for d in tr_docs]
    tr_ids = [d.id for d in tr_docs]
    dv = [(dv_tok[d.id], d.label) for d in dv_docs]
    if cfg.train_on_dev:


        tr = tr + dv
        tr_ids = tr_ids + [d.id for d in dv_docs]
        cfg.eval_every = 10 ** 9
        print(f"[train_on_dev] training on train+dev ({len(tr):,} docs); dev scores below "
              f"are NOT held-out; final checkpoint is kept", flush=True)
    ydv = np.array([d.label for d in dv_docs])


    clean = D.clean_dev_mask(dv_docs)
    sel = clean if cfg.select_on == "clean" else np.ones_like(clean)
    print(f"[dev] selecting on {sel.sum():,} dev cases ({cfg.select_on}); reporting both "
          f"clean ({clean.sum():,}) and all ({len(clean):,}) dev", flush=True)

    t_logit = None
    if cfg.peft == "distill":
        from cjpe.distill import build_student, kd_loss, teacher_logits
        t_logit = teacher_logits(cfg, tr, tr_ids, device)
        model = build_student(cfg, device)
        cfg.save()
        alpha = float(cfg.peft_kw.get("alpha", 0.5))
        tau = float(cfg.peft_kw.get("tau", 2.0))
        print(f"[distill] alpha={alpha} tau={tau} "
              f"{'(CONTROL: gold only)' if alpha >= 1 else ''}", flush=True)
    else:
        model = build_model(cfg, device)
    if cfg.grad_checkpoint:
        model.backbone.gradient_checkpointing(True)
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
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm))
        if s >= warm else (s + 1) / warm)


    w_doc = np.ones(len(tr), dtype=np.float32)
    if cfg.era_weight:
        dec = lambda y: min(y // 10 * 10, 2010)
        td = np.array([dec(d.year) for d in tr_docs])
        dd = np.array([dec(d.year) for d in dv_docs])
        ratio = {k: (dd == k).mean() / max((td == k).mean(), 1e-9) for k in set(td)}
        w_doc = np.clip(np.array([ratio[k] for k in td]), *cfg.era_clip).astype(np.float32)
        w_doc /= w_doc.mean()
        print("[era] weight by decade: " + " ".join(
            f"{k}s:{w_doc[td == k][0]:.2f}" for k in sorted(set(td))) +
            f" | effective sample size {w_doc.sum()**2 / (w_doc**2).sum():,.0f} of {len(tr):,}",
            flush=True)
    w_doc = torch.from_numpy(w_doc)
    labels = np.array([l for _, l in tr])
    n1 = float(w_doc[labels == 1].sum()); n0 = float(w_doc[labels == 0].sum())
    pw = torch.tensor(n0 / n1 if cfg.balance else 1.0, device=device)
    print(f"[data] train {len(tr):,} (0:{n0:,.0f} 1:{n1:,.0f}) dev {len(dv):,} | {steps_per_epoch:,} "
          f"steps/epoch, {total:,} total | view K={cfg.K} L={cfg.L} keep={cfg.keep} "
          f"pool={cfg.pool} | pos_weight {pw.item():.3f}", flush=True)

    blocks = blocks_of(model)
    stage_done = 0

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
            for nm in model.backbone.modules():
                if nm.__class__.__name__ in ("LayerNorm", "RMSNorm") and nm not in \
                        [x for b in blocks for x in b.modules()]:
                    for p in nm.parameters():
                        p.requires_grad_(True)
            r = trainable_report(model)
            print(f"[gradual] step {step}: stage {stage_done}/{S} -> top {k} blocks, "
                  f"{r['trainable']:,} trainable", flush=True)

    log = open(cfg.run_dir / "log.jsonl", "w")
    best = {"score": -1.0}
    t0 = time.time()

    def evaluate(step, epoch):
        p1 = predict_probs(model, dv, cfg, device)
        thr, s = best_threshold(p1[sel], ydv[sel])
        if cfg.train_on_dev:
            thr = float(np.median(p1))
        m05 = prf((p1[sel] >= 0.5).astype(int), ydv[sel])
        s_clean = prf((p1[clean] >= thr).astype(int), ydv[clean])["macro_f1"]
        s_all = prf((p1 >= thr).astype(int), ydv)["macro_f1"]
        rec = {"step": step, "epoch": round(epoch, 3), "dev_macro_f1": s, "thr": thr,
               "dev_at_0.5": m05["macro_f1"], "dev_clean": s_clean, "dev_all": s_all,
               "min": round((time.time() - t0) / 60, 1)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"[eval] step {step} ep {epoch:.2f}: dev macro-F1 clean {s_clean:.4f} | "
              f"all {s_all:.4f} @thr {thr:.2f} (selecting on {cfg.select_on}) | "
              f"{rec['min']} min", flush=True)
        if s > best["score"] or cfg.train_on_dev:
            best.update(score=s, thr=thr, step=step, epoch=epoch, at05=m05["macro_f1"])
            torch.save({"model": model.state_dict(), "thr": thr, "config": asdict(cfg),
                        "step": step}, cfg.run_dir / "best.pt")
            np.save(cfg.run_dir / "dev_probs.npy", p1)

    dt = torch.bfloat16
    step, micro = 0, 0
    model.train()
    order = list(range(len(tr)))
    ep = 0
    while step < total:
        random.Random(cfg.seed + ep).shuffle(order)
        for i in range(0, len(order), cfg.batch_docs):
            maybe_unfreeze(step)
            idx = order[i:i + cfg.batch_docs]
            x, m, cm, pos, y = collate([tr[j] for j in idx], cfg)
            with torch.autocast("cuda", dtype=dt, enabled=device.startswith("cuda")):
                lg = model(x.to(device), m.to(device), cm.to(device), pos.to(device))
            if t_logit is not None:
                loss = kd_loss(lg.float(), t_logit[idx].to(device), y.to(device), pw, alpha, tau)
            else:
                wb = w_doc[idx].to(device)
                loss = (F.binary_cross_entropy_with_logits(
                    lg.float(), y.to(device), pos_weight=pw, reduction="none") * wb).mean()
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
                log.write(json.dumps({"step": step, "loss": round(loss.item(), 4),
                                      "lr": sched.get_last_lr()[0]}) + "\n")
            if step % cfg.eval_every == 0:
                evaluate(step, step / steps_per_epoch)
            if step >= total:
                break
        ep += 1
    if step % cfg.eval_every:
        evaluate(step, step / steps_per_epoch)

    m = prf((np.load(cfg.run_dir / "dev_probs.npy")[sel] >= best["thr"]).astype(int),
            ydv[sel])
    summ = {"name": cfg.name, "peft": cfg.peft, "K": cfg.K, "keep": cfg.keep, "pool": cfg.pool,
            "best": {k: best[k] for k in ("score", "thr", "step", "epoch", "at05")},
            "dev_report": report(m), "trainable": rep["trainable"], "total": rep["total"],
            "wall_min": round((time.time() - t0) / 60, 1),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
            if device.startswith("cuda") else None}
    (cfg.run_dir / "summary.json").write_text(json.dumps(summ, indent=1))
    print(f"[done] {cfg.name}: best dev macro-F1 {best['score']:.4f} @thr {best['thr']:.2f} "
          f"(step {best['step']}) | {report(m)}", flush=True)
    return summ

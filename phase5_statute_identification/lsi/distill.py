from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent


def load_teacher(teacher_run, device):
    from lsi.train import TrainConfig, build_model
    run = Path(teacher_run)
    run = run if run.is_absolute() else ROOT / run
    state = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    fields = TrainConfig.__dataclass_fields__
    tcfg = TrainConfig(**{k: v for k, v in state["config"].items() if k in fields})
    teacher = build_model(tcfg, device)
    teacher.load_state_dict(state["model"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher, tcfg, run


def make_student(teacher, keep_every=2):
    bb = teacher.backbone
    student = copy.deepcopy(teacher)
    keep = list(range(0, len(bb.inner.blocks), keep_every))
    student.backbone.inner.blocks = torch.nn.ModuleList(
        [student.backbone.inner.blocks[i] for i in keep])
    if hasattr(student.backbone.inner, "cfg"):
        student.backbone.inner.cfg = copy.deepcopy(bb.inner.cfg)
        student.backbone.inner.cfg.n_layers = len(keep)
    for p in student.parameters():
        p.requires_grad_(True)
    n_t = sum(p.numel() for p in teacher.parameters())
    n_s = sum(p.numel() for p in student.parameters())
    print(f"[student] blocks {keep} of {len(bb.inner.blocks)} | {n_s/1e6:.1f}M params vs "
          f"teacher {n_t/1e6:.1f}M ({100*n_s/n_t:.0f}%)", flush=True)
    return student


def build_student(cfg, device):
    teacher, tcfg, _ = load_teacher(cfg.peft_kw["teacher_run"], device)
    for f in ("L", "K", "keep", "pool"):
        setattr(cfg, f, getattr(tcfg, f))
    return make_student(teacher, int(cfg.peft_kw.get("keep_every", 2))).to(device)


def teacher_logits(cfg, train_items, train_ids, device):
    from lsi.train import predict_logits
    teacher, tcfg, run = load_teacher(cfg.peft_kw["teacher_run"], device)

    cache = run / ("train_logits.npz" if cfg.fold < 0 else f"train_logits_fold{cfg.fold}.npz")
    if cache.exists():
        z = np.load(cache)
        lut = {i: k for k, i in enumerate(z["ids"])}
        if all(i in lut for i in train_ids):
            return torch.from_numpy(z["logit"][[lut[i] for i in train_ids]])
    logit = predict_logits(teacher, [(ids, None) for ids, _ in train_items], tcfg, device, 32)
    np.savez(cache, ids=np.array(train_ids), logit=logit.astype(np.float32))
    del teacher
    torch.cuda.empty_cache()
    print(f"[teacher] logits for {len(train_ids):,} train docs cached at {cache}", flush=True)
    return torch.from_numpy(logit)


def kd_loss(s, t, y, alpha, tau):
    gold = F.binary_cross_entropy_with_logits(s, y)
    if alpha >= 1.0:
        return gold
    soft = F.binary_cross_entropy_with_logits(s / tau, torch.sigmoid(t / tau))
    return alpha * gold + (1 - alpha) * tau ** 2 * soft

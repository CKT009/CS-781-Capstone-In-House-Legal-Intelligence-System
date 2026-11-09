from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent


def load_teacher(teacher_run, device):
    from cjpe.train import TrainConfig, build_model
    run = Path(teacher_run)
    if not run.is_absolute():
        run = ROOT / run
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
    kw = cfg.peft_kw
    teacher, tcfg, _ = load_teacher(kw["teacher_run"], device)

    for f in ("L", "K", "keep", "pool"):
        setattr(cfg, f, getattr(tcfg, f))
    return make_student(teacher, int(kw.get("keep_every", 2))).to(device)


def teacher_logits(cfg, train_items, train_ids, device):
    from cjpe.train import predict_probs
    teacher, tcfg, run = load_teacher(cfg.peft_kw["teacher_run"], device)
    cache = run / "train_logits.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        lut = dict(zip(z["ids"], z["logit"]))
        if all(i in lut for i in train_ids):
            return torch.tensor([lut[i] for i in train_ids], dtype=torch.float)
    p = predict_probs(teacher, train_items, tcfg, device, batch=32)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    np.savez(cache, ids=np.array(train_ids), logit=logit.astype(np.float32))
    del teacher
    torch.cuda.empty_cache()
    print(f"[teacher] logits for {len(train_ids):,} train docs cached at {cache}", flush=True)
    return torch.tensor(logit, dtype=torch.float)


def kd_loss(s_logit, t_logit, y, pos_weight, alpha, tau):
    gold = F.binary_cross_entropy_with_logits(s_logit, y, pos_weight=pos_weight)
    if alpha >= 1.0:
        return gold
    soft = F.binary_cross_entropy_with_logits(s_logit / tau, torch.sigmoid(t_logit / tau))
    return alpha * gold + (1 - alpha) * tau ** 2 * soft

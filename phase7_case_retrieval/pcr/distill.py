from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent


def _run(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_teacher(teacher_run, device):
    from pcr.train import TrainConfig, build_model
    run = _run(teacher_run)
    state = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    fields = TrainConfig.__dataclass_fields__
    tcfg = TrainConfig(**{k: v for k, v in state["config"].items() if k in fields})
    tcfg.device = device
    teacher = build_model(tcfg, device)
    teacher.load_state_dict(state["model"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher, tcfg, run


def drop_blocks(model, keep_every=2):
    inner = model.backbone.inner
    n = len(inner.blocks)
    keep = list(range(keep_every - 1, n, keep_every))
    inner.blocks = torch.nn.ModuleList([inner.blocks[i] for i in keep])
    inner.cfg = copy.deepcopy(inner.cfg)
    inner.cfg.n_layers = len(keep)
    return keep, n


class KD:
    def __init__(self, t_emb, t_ids, t_scale, alpha, tau, device):
        self.z = F.normalize(torch.from_numpy(t_emb).float(), dim=-1).to(device)
        self.pos = {c: i for i, c in enumerate(t_ids)}
        self.scale, self.alpha, self.tau = float(t_scale), float(alpha), float(tau)

    def loss(self, nce, s, qids, cands):
        if self.alpha >= 1.0:
            return nce, {"nce": nce.item()}
        dev = s.device
        tq = self.z[[self.pos[q] for q in qids]]
        tc = self.z[[self.pos[c] for c in cands]]
        t = self.scale * tq @ tc.T
        selfm = torch.tensor([[c == q for c in cands] for q in qids], device=dev)
        T = self.tau
        lt = (t / T).masked_fill(selfm, -1e4)
        ls = (s.float() / T).masked_fill(selfm, -1e4)
        kl = F.kl_div(ls.log_softmax(-1), lt.log_softmax(-1), log_target=True,
                      reduction="batchmean") * T * T
        return self.alpha * nce + (1 - self.alpha) * kl, {"nce": nce.item(), "kl": kl.item()}


def setup_distill(cfg, device, qtr, pool_tr):
    from pcr import encode as E
    from pcr.backbones import load_backbone
    from pcr.model import DocEncoder
    from pcr.peft import apply_peft
    kw = dict(cfg.peft_kw)
    teacher, tcfg, run = load_teacher(kw["teacher_run"], device)
    for f in ("K", "K_eval", "L", "pool", "proj", "centre_rank", "centre_init"):
        setattr(cfg, f, getattr(tcfg, f))
    t_scale = teacher.head.scale().item()


    ids = sorted(pool_tr)
    cache = run / "train_emb.npy"
    if cache.exists() and json.loads((run / "train_emb_ids.json").read_text()) == ids:
        t_emb = np.load(cache).astype(np.float32)
    else:
        t_emb = E.embed_docs(teacher, ids, tcfg.k_eval, tcfg.L, device)
        np.save(cache, t_emb.astype(np.float16))
        (run / "train_emb_ids.json").write_text(json.dumps(ids))
        print(f"[teacher] embeddings for {len(ids):,} train-pool docs cached at {cache}",
              flush=True)

    keep_every = int(kw.get("keep_every", 2))
    init = kw.get("student_init", "phase1")
    if init == "teacher":
        student = copy.deepcopy(teacher)
    elif init == "phase1":
        bb = load_backbone(cfg.backbone, cfg.ckpt, device="cpu")
        student = DocEncoder(bb, pool=cfg.pool, proj=cfg.proj, chunk_batch=cfg.chunk_batch)
    else:
        raise ValueError(f"student_init {init!r}: phase1|teacher")
    keep, n = drop_blocks(student, keep_every)
    student = student.to(device)
    if init == "phase1":
        from pcr.train import centre_head
        if cfg.proj and cfg.centre_init:
            centre_head(student, cfg, device)
        apply_peft(student, "full")
    for p in student.parameters():
        p.requires_grad_(True)
    n_t = sum(p.numel() for p in teacher.parameters())
    n_s = sum(p.numel() for p in student.parameters())
    alpha = float(kw.get("alpha", 0.5)); tau = float(kw.get("tau", 1.0))
    print(f"[student] init={init} blocks {keep} of {n} | {n_s/1e6:.1f}M params vs teacher "
          f"{n_t/1e6:.1f}M ({100*n_s/n_t:.0f}%) | teacher 1/T {t_scale:.1f} | alpha={alpha} "
          f"tau={tau}{' (CONTROL: gold only)' if alpha >= 1 else ''}", flush=True)
    del teacher
    torch.cuda.empty_cache()
    return student, KD(t_emb, ids, t_scale, alpha, tau, device)

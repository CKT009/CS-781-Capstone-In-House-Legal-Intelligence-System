from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from summ import data as D
from summ import metrics as M
from summ.abs_data import IN, OUT, SummDataset, collate, encode_input
from summ.abs_generate import summarise
from summ.abs_peft import POSITIONS_USED, adapt, report
from summ.lm import load_lm

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Cfg:
    name: str = "debug"
    backbone: str = "A_grown"
    method: str = "lora"
    kw: dict = field(default_factory=dict)
    builder: str = "lr+gbt+mmr0.6"
    n_in: int = IN
    n_out: int = OUT
    epochs: float = 3.0
    batch: int = 8
    accum: int = 2
    lr: float = 2e-4
    e_lr_mult: float = 20.0
    weight_decay: float = 0.0
    warmup: float = 0.05
    grad_ckpt: bool = False

    teacher: str = ""
    keep_every: int = 2
    kd_alpha: float = 0.5
    kd_tau: float = 2.0
    max_train: int = 0
    sel_docs: int = 100
    eval_docs: int = 0
    gen: dict = field(default_factory=lambda: dict(max_new=OUT, min_new=200, greedy=True,
                                                   repetition_penalty=1.15, no_repeat_ngram=4))
    seed: int = 0

    @property
    def dir(self) -> Path:
        return ROOT / "runs" / self.name


def _load_params(model, sd):
    missing = [n for n, _ in model.named_parameters() if n not in sd]
    assert not missing, f"{len(missing)} parameters missing: {missing[:5]}"
    model.load_state_dict(sd, strict=False)
    return model


def _student_from(teacher_lm, keep_every):
    import copy
    s = copy.deepcopy(teacher_lm)
    s.stack.blocks = torch.nn.ModuleList(list(s.stack.blocks)[::keep_every])
    s.cfg = copy.deepcopy(teacher_lm.cfg); s.cfg.n_layers = len(s.stack.blocks)
    return s


def build(cfg: Cfg, device="cuda"):
    lm = load_lm(cfg.backbone)
    teacher = None
    if cfg.method == "student":
        tdir = ROOT / cfg.teacher
        _load_params(lm, torch.load(tdir / "best.pt", map_location="cpu")["trainable"])
        if cfg.kd_alpha < 1.0:
            teacher = lm.to(device).eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
        lm = _student_from(lm, cfg.keep_every)
    model = adapt(lm, cfg.method, **cfg.kw).to(device)
    return model, teacher


def load_trained(run_dir, device="cuda"):
    run = Path(run_dir)
    c = json.loads((run / "config.json").read_text())
    cfg = Cfg(**{k: v for k, v in c.items() if k in Cfg.__dataclass_fields__})
    if cfg.method == "student":
        lm = load_lm(cfg.backbone)
        _load_params(lm, torch.load(ROOT / cfg.teacher / "best.pt", map_location="cpu")["trainable"])
        model = adapt(_student_from(lm, cfg.keep_every), "student")
    else:
        model = adapt(load_lm(cfg.backbone), cfg.method, **cfg.kw)
    sd = torch.load(run / "best.pt", map_location="cpu")["trainable"]
    missing = set(sd) - set(dict(model.named_parameters())) - set(dict(model.named_buffers()))
    assert not missing, f"checkpoint tensors not in model: {list(missing)[:5]}"
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval(), cfg


def n_in_for(cfg):
    used = cfg.kw.get(POSITIONS_USED.get(cfg.method, ""), 0) or \
        {"prefix": 16, "prompt": 20}.get(cfg.method, 0)
    return cfg.n_in - used


def trainable_state(model, full):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if full or p.requires_grad} | \
        ({n: b.detach().cpu() for n, b in model.named_buffers()} if not full else {})


def train(cfg: Cfg):
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    cfg.dir.mkdir(parents=True, exist_ok=True)
    (cfg.dir / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    dev = "cuda"
    tr, va = D.split()
    if cfg.max_train:
        tr = random.Random(cfg.seed).sample(tr, cfg.max_train)
    n_in = n_in_for(cfg)


    key = f"{cfg.builder}_{n_in}_{cfg.n_out}_{len(tr)}_{hashlib.md5(' '.join(r['id'] for r in tr).encode()).hexdigest()[:8]}"
    cache = ROOT / "runs" / "_cache" / f"{key}.json"
    if cache.exists():
        ds = SummDataset([], cfg.builder); ds.items = [tuple(x) for x in json.loads(cache.read_text())]
    else:
        ds = SummDataset(tr, cfg.builder, n_in=n_in, n_out=cfg.n_out)
        cache.parent.mkdir(parents=True, exist_ok=True); cache.write_text(json.dumps(ds.items))
    sel = va[:cfg.sel_docs]
    sel_in = {r["id"]: encode_input(r["document"], cfg.builder, n_in) for r in sel}

    model, teacher = build(cfg, dev)
    if cfg.grad_ckpt:
        model.cfg.grad_checkpoint = True
    rep = report(model)
    print(f"[{cfg.name}] {cfg.method}: {rep['trainable']:,} / {rep['total']:,} trainable "
          f"({rep['pct']:.2f}%) | input {n_in} tok, output {cfg.n_out} | train {len(ds):,}"
          + (f" | KD from {cfg.teacher} alpha={cfg.kd_alpha} tau={cfg.kd_tau}"
             if cfg.method == "student" else ""), flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": [p for n, p in model.named_parameters()
                          if p.requires_grad and not n.endswith("lora_E")], "lr": cfg.lr}]
    E = [p for n, p in model.named_parameters() if p.requires_grad and n.endswith("lora_E")]
    if E:
        groups.append({"params": E, "lr": cfg.lr * cfg.e_lr_mult, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch, shuffle=True, collate_fn=collate,
                                     drop_last=True, num_workers=2)
    total = int(len(dl) * cfg.epochs / cfg.accum)
    warm = max(1, int(total * cfg.warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    ada = getattr(model, "_adalora", None)
    log = open(cfg.dir / "log.jsonl", "w")
    best, step, micro, t0 = -1.0, 0, 0, time.time()

    def select(epoch):
        nonlocal best
        out = summarise(model, sel_in, dev, batch=16, **cfg.gen)
        m = M.evaluate([out[r["id"]] for r in sel], [r["summary"] for r in sel], bert=False, rougeL=False)
        rec = {"epoch": epoch, "step": step, "sel_rouge1": m["rouge1"], "words": m["pred_words"],
               "min": round((time.time() - t0) / 60, 1)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"[select] epoch {epoch:.2f} step {step}: ROUGE-1 {m['rouge1']:.4f} on {len(sel)} val "
              f"| {m['pred_words']:.0f} words | {rec['min']} min", flush=True)
        if m["rouge1"] > best:
            best = m["rouge1"]
            torch.save({"trainable": trainable_state(model, cfg.method in ("full", "student")),
                        "step": step, "epoch": epoch}, cfg.dir / "best.pt")
        model.train()

    model.train()
    ep = 0.0
    while step < total:
        for ids, lab, att in dl:
            ids, lab = ids.to(dev), lab.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(ids)["logits"]
            lg = lg[:, :-1].float(); tgt = lab[:, 1:]
            ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), ignore_index=-100)
            loss = ce
            if teacher is not None:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    tl = teacher(ids)["logits"][:, :-1].float()
                m = tgt != -100
                kl = F.kl_div(F.log_softmax(lg[m] / cfg.kd_tau, -1), F.log_softmax(tl[m] / cfg.kd_tau, -1),
                              log_target=True, reduction="batchmean") * cfg.kd_tau ** 2
                loss = cfg.kd_alpha * ce + (1 - cfg.kd_alpha) * kl
            if ada is not None:
                loss = loss + ada.orth_loss()
            (loss / cfg.accum).backward()
            micro += 1
            if micro % cfg.accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            if ada is not None:
                ada.update_and_mask(step, total)
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0:
                log.write(json.dumps({"step": step, "ce": round(ce.item(), 4), "loss": round(loss.item(), 4),
                                      "lr": sched.get_last_lr()[0]}) + "\n"); log.flush()
            if step % max(1, len(dl) // cfg.accum) == 0:
                ep = step * cfg.accum / len(dl)
                select(ep)
            if step >= total:
                break
    if step % max(1, len(dl) // cfg.accum):
        select(step * cfg.accum / len(dl))
    del opt
    if teacher is not None:
        del teacher
    torch.cuda.empty_cache()
    return finish(cfg.dir, dev, wall=(time.time() - t0) / 60, rep=rep)


def finish(run_dir, dev="cuda", wall=None, rep=None, gen=None, tag=""):
    model, cfg = load_trained(run_dir, dev)
    g = gen or cfg.gen
    n_in = n_in_for(cfg)
    _, va = D.split(); te = D.load("test")
    va = va[:cfg.eval_docs] if cfg.eval_docs else va
    t = time.time()
    vp = summarise(model, {r["id"]: encode_input(r["document"], cfg.builder, n_in) for r in va}, dev, **g)
    gen_min = (time.time() - t) / 60
    tp = summarise(model, {r["id"]: encode_input(r["document"], cfg.builder, n_in) for r in te}, dev, **g)
    del model; torch.cuda.empty_cache()


    run = Path(run_dir)
    (run / f"val_preds{tag}.json").write_text(json.dumps(vp))
    (run / f"test_preds{tag}.json").write_text(json.dumps(tp))
    z = M.write_submission(tp, [r["id"] for r in te], ROOT / "subs" / f"{run.name}{tag}.zip")
    m = M.evaluate([vp[r["id"]] for r in va], [r["summary"] for r in va], device=dev)
    summ = {"name": run.name + tag, "method": cfg.method, "val500": m, "gen": g,
            "val_gen_min": round(gen_min, 1), "wall_min": round(wall, 1) if wall else None,
            "trainable": rep["trainable"] if rep else None, "total": rep["total"] if rep else None,
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    (run / f"summary{tag}.json").write_text(json.dumps(summ, indent=1))
    print(f"[done] {run.name}{tag}: {M.report(m)} | {z}", flush=True)
    return summ

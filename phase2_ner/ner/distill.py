from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ner import data as D
from ner.metrics import strict_prf, format_report
from ner.model import NERTagger, class_weights_from_counts
from ner.train import TrainConfig, build_optimizer, lr_scale, log, seed_all, evaluate, _save

IGNORE = D.IGNORE


@dataclass
class DistillConfig:
    name: str = "distill"
    teacher_run: str = "runs/lora"
    keep_every: int = 2
    alpha: float = 0.5
    temperature: float = 2.0
    control: bool = True


    data_dir: str = "data/Phase_2_LNER"
    train_folds: tuple = ("fold1",)
    dev_folds: tuple = ("fold2",)
    max_len: int = 512
    stride: int = 128
    class_weight: str = "inv_sqrt"
    epochs: int = 5
    batch_size: int = 8
    lr: float = 5e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_frac: float = 0.1
    grad_clip: float = 1.0
    dropout: float = 0.1
    amp: str = "bf16"
    seed: int = 1337
    out_dir: str = "runs"
    log_every: int = 20
    device: str = "cuda"

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.name


def make_student(teacher: NERTagger, keep_every: int = 2) -> NERTagger:
    import copy

    bb = teacher.backbone
    if not hasattr(bb.inner, "blocks"):
        raise NotImplementedError(
            f"backbone {bb.arch!r} has no .blocks list to thin; "
            f"distillation here supports the stack-based architectures")

    student_bb = copy.deepcopy(bb)
    keep = list(range(0, len(bb.inner.blocks), keep_every))
    student_bb.inner.blocks = torch.nn.ModuleList(
        [student_bb.inner.blocks[i] for i in keep])
    student_bb.cfg = copy.deepcopy(bb.cfg)
    student_bb.cfg.n_layers = len(keep)

    student = NERTagger(student_bb, n_tags=teacher.n_tags,
                        dropout=teacher.head_dropout.p,
                        class_weights=teacher.class_weights.clone(),
                        use_crf=teacher.crf is not None)
    student.head_proj.load_state_dict(teacher.head_proj.state_dict())
    if teacher.crf is not None:
        student.crf.load_state_dict(teacher.crf.state_dict())


    for q in student.parameters():
        q.requires_grad_(True)
    n_train = sum(q.numel() for q in student.parameters() if q.requires_grad)

    nt = sum(p.numel() for p in teacher.parameters())
    ns = sum(p.numel() for p in student.parameters())
    print(f"[student] kept blocks {keep} of {len(bb.inner.blocks)} | "
          f"{ns/1e6:.1f}M vs teacher {nt/1e6:.1f}M ({100*ns/nt:.0f}%) | "
          f"{n_train/1e6:.1f}M trainable")
    if n_train < ns * 0.9:
        raise RuntimeError(
            f"student has {n_train:,} trainable of {ns:,} parameters -- it was built "
            f"from a frozen teacher and cannot learn. Distillation would silently "
            f"measure nothing.")
    return student


def kd_loss(student_logits, teacher_logits, labels, tau: float):
    mask = labels != IGNORE
    if not mask.any():
        return student_logits.sum() * 0.0
    s = student_logits[mask].float() / tau
    t = teacher_logits[mask].float() / tau
    return F.kl_div(F.log_softmax(s, -1), F.softmax(t, -1),
                    reduction="batchmean") * (tau ** 2)


def distill(cfg: DistillConfig):
    from ner.predict import load_tagger

    seed_all(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    teacher, tcfg = load_tagger(cfg.teacher_run, "best.pt", device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("artifacts/tokenizer_16k/tokenizer.json")
    max_len = min(cfg.max_len, teacher.backbone.max_positions)

    def prep(folds):
        texts, golds, chunks = {}, {}, {}
        for f in folds:
            t, g = D.load_fold(cfg.data_dir, f)
            texts.update(t)
            golds.update(g)
            for d, txt in t.items():
                chunks[d] = D.encode_document(txt, g.get(d, []), tok,
                                              max_len=max_len, stride=cfg.stride)
        return texts, golds, chunks

    tr_texts, tr_golds, tr_chunks = prep(cfg.train_folds)
    dv_texts, dv_golds, dv_chunks = prep(cfg.dev_folds)
    train_flat = [c for cs in tr_chunks.values() for c in cs]

    results = {}
    for mode in (["kd", "control"] if cfg.control else ["kd"]):
        name = f"{cfg.name}_{mode}"
        print(f"\n{'=' * 72}\n  {name}  "
              f"({'teacher + gold' if mode == 'kd' else 'gold only -- the CONTROL'})"
              f"\n{'=' * 72}")
        results[mode] = _train_student(
            cfg, mode, name, teacher, tcfg, train_flat,
            dv_chunks, dv_texts, dv_golds, device)

    if cfg.control:
        gap = results["kd"]["best"]["macro_f1"] - results["control"]["best"]["macro_f1"]
        print(f"\n{'=' * 72}")
        print(f"  DISTILLATION GAP: {gap:+.4f} macro-F1")
        print(f"    with teacher {results['kd']['best']['macro_f1']:.4f}")
        print(f"    gold only    {results['control']['best']['macro_f1']:.4f}")
        print(f"  This gap, not the absolute score, is the claim.")
        print(f"{'=' * 72}")
        (cfg.run_dir.parent / f"{cfg.name}_comparison.json").write_text(
            json.dumps({"gap": gap, **results}, indent=2, default=str))
    return results


def _train_student(cfg, mode, name, teacher, tcfg, train_flat,
                   dv_chunks, dv_texts, dv_golds, device):
    run = Path(cfg.out_dir) / name
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(
        {**asdict(tcfg), **asdict(cfg), "name": name, "mode": mode},
        indent=2, default=str))

    seed_all(cfg.seed)
    student = make_student(teacher, cfg.keep_every).to(device)

    opt = build_optimizer(student, cfg)
    loader = DataLoader(train_flat, batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=D.collate, num_workers=0)
    total = len(loader) * cfg.epochs
    warmup = int(total * cfg.warmup_frac)
    use_amp = device.startswith("cuda") and cfg.amp != "off"

    best, step, t0 = {"macro_f1": -1.0, "epoch": None, "step": None}, 0, time.time()
    student.train()
    for epoch in range(1, cfg.epochs + 1):
        for batch in loader:
            ids = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            lab = batch["labels"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = student(ids, am, labels=lab)
                loss = out["loss"]
                if mode == "kd":
                    with torch.no_grad():
                        t_logits = teacher(ids, am)["logits"]
                    loss = (cfg.alpha * out["loss"]
                            + (1 - cfg.alpha) * kd_loss(out["logits"], t_logits, lab,
                                                        cfg.temperature))
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in student.parameters() if p.requires_grad], cfg.grad_clip)
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * lr_scale(step, total, warmup)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % cfg.log_every == 0:
                log(run, step=step, epoch=epoch, mode=mode, loss=float(loss.detach()))

        res, _ = evaluate(student, dv_chunks, dv_texts, dv_golds, cfg, device)
        log(run, step=step, epoch=epoch, mode=mode,
            **{k: v for k, v in res.items() if k not in ("per_label", "partial")})
        print(f"[eval] {name} epoch {epoch}  macro-F1 {res['macro_f1']:.4f}")
        if res["macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": res["macro_f1"], "epoch": epoch, "step": step}
            _save(run / "best.pt", student, opt, cfg, step, epoch, best)
        _save(run / "last.pt", student, opt, cfg, step, epoch, best)

    print(format_report(res))
    summary = {"name": name, "mode": mode, "best": best,
               "params": sum(p.numel() for p in student.parameters()),
               "teacher_params": sum(p.numel() for p in teacher.parameters()),
               "wall_min": round((time.time() - t0) / 60, 1)}
    (run / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary

from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import torch
import torch.nn.functional as F

from rr import data as D
from rr.metrics import score as score_fn, report as report_fn


@dataclass
class DistillConfig:
    name: str = "distill"
    teacher_run: str = "runs/lora"
    keep_every: int = 2
    alpha: float = 0.5
    temperature: float = 2.0
    control: bool = True
    epochs: int = 20
    lr: float = 5e-5
    head_lr: float = 1e-3
    seed: int = 1337
    out_dir: str = "runs"
    device: str = "cuda"

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.name


def make_student(teacher, keep_every: int = 2):
    bb = teacher.backbone
    if not hasattr(bb.inner, "blocks"):
        raise NotImplementedError(f"backbone {bb.arch!r} has no .blocks to thin")
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
    print(f"[student] kept blocks {keep} of {len(bb.inner.blocks)} | "
          f"{n_s/1e6:.1f}M vs teacher {n_t/1e6:.1f}M ({100*n_s/n_t:.0f}%)")
    assert sum(p.numel() for p in student.parameters() if p.requires_grad) == n_s, \
        "student has frozen parameters -- it was copied from a frozen teacher"
    return student


def kd_loss(s_logits, t_logits, mask, tau: float):
    m = mask.bool().reshape(-1)
    if not m.any():
        return s_logits.sum() * 0.0
    s = s_logits.reshape(-1, s_logits.shape[-1])[m].float() / tau
    t = t_logits.reshape(-1, t_logits.shape[-1])[m].float() / tau
    return F.kl_div(F.log_softmax(s, -1), F.softmax(t, -1),
                    reduction="batchmean") * (tau ** 2)


def distill(cfg: DistillConfig):
    from rr.predict import load_tagger
    from rr.train import TrainConfig, _prepare, _examples, evaluate, seed_all
    from tokenizers import Tokenizer

    seed_all(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    teacher, tcfg, vocab = load_tagger(cfg.teacher_run, "best.pt", device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    tok = Tokenizer.from_file(tcfg.tokenizer)
    tr_docs, tr_encs = _prepare(tcfg, tok, vocab, "train")
    dv_docs, dv_encs = _prepare(tcfg, tok, vocab, "dev")
    ex = _examples(tr_docs, tr_encs, tcfg)

    results = {}
    for mode in (["kd", "control"] if cfg.control else ["kd"]):
        name = f"{cfg.name}_{mode}"


        run = Path(cfg.out_dir) / Path(tcfg.backbone).name / name
        run.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 72}\n  {name}  ({'teacher + gold' if mode=='kd' else 'gold only'})"
              f"\n{'=' * 72}")
        seed_all(cfg.seed)
        student = make_student(teacher, cfg.keep_every).to(device)
        head = ("head", "doc", "crf")
        opt = torch.optim.AdamW([
            {"params": [p for n, p in student.named_parameters()
                        if not n.startswith(head)], "lr": cfg.lr},
            {"params": [p for n, p in student.named_parameters()
                        if n.startswith(head)], "lr": cfg.head_lr}], weight_decay=0.01)
        best = {"macro_f1": -1.0, "epoch": None}
        t0 = time.time()
        for ep in range(1, cfg.epochs + 1):
            random.Random(cfg.seed + ep).shuffle(ex)
            student.train()
            tot, nb = 0.0, 0
            for e, lo, hi in ex:
                ids = torch.tensor(e.ids[lo:hi], device=device)
                msk = torch.tensor(e.mask[lo:hi], device=device)
                lab = torch.tensor(e.labels[lo:hi], device=device)
                dm = torch.ones(1, hi - lo, dtype=torch.long, device=device)
                out = student(ids, msk, labels=lab)
                loss = out["loss"]
                if mode == "kd":
                    with torch.no_grad():
                        tl = teacher(ids, msk)["logits"]
                    loss = cfg.alpha * loss + (1 - cfg.alpha) * kd_loss(
                        out["logits"], tl, dm, cfg.temperature)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                tot += float(loss); nb += 1
            pred, s = evaluate(student, dv_docs, dv_encs, tcfg, vocab, device)
            print(f"[eval] epoch {ep:2d}  macro-F1 {s['macro_f1']:.4f}  "
                  f"loss {tot/max(nb,1):.4f}", flush=True)
            if s["macro_f1"] > best["macro_f1"]:
                best = {"macro_f1": s["macro_f1"],
                        "macro_f1_supported": s["macro_f1_supported"], "epoch": ep}
                torch.save({"model": student.state_dict(), "cfg": asdict(tcfg),
                            "epoch": ep, "best": best}, run / "best.pt")
                (run / "dev_predictions.json").write_text(json.dumps(pred))
        (run / "config.json").write_text(json.dumps({**asdict(tcfg), "name": name,
                                                     **asdict(cfg), "mode": mode},
                                                    indent=2, default=str))
        summary = {"name": name, "mode": mode, "best": best,
                   "params": sum(p.numel() for p in student.parameters()),
                   "teacher_params": sum(p.numel() for p in teacher.parameters()),
                   "wall_min": (time.time() - t0) / 60}
        (run / "summary.json").write_text(json.dumps(summary, indent=2))
        results[mode] = summary
        del student
        torch.cuda.empty_cache()

    if "control" in results:
        gap = results["kd"]["best"]["macro_f1"] - results["control"]["best"]["macro_f1"]
        print(f"\n{'=' * 72}\n  DISTILLATION GAP: {gap:+.4f} macro-F1\n"
              f"    with teacher {results['kd']['best']['macro_f1']:.4f}\n"
              f"    gold only    {results['control']['best']['macro_f1']:.4f}\n"
              f"  This gap, not the absolute score, is the claim.\n{'=' * 72}")
        (Path(cfg.out_dir) / Path(tcfg.backbone).name /
         f"{cfg.name}_comparison.json").write_text(
            json.dumps({"gap": gap, **results}, indent=2))
    return results

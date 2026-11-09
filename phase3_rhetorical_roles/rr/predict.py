from __future__ import annotations

import json
import zipfile
from pathlib import Path

import torch

from rr import data as D
from rr.metrics import score as score_fn, report as report_fn, validate_submission


def _resolve_run(path, backbone):
    p = Path(path)
    if (p / "config.json").exists():
        return p
    alt = p.parent / Path(backbone).name / p.name
    if (alt / "config.json").exists():
        return alt
    raise FileNotFoundError(
        f"teacher run {path!r} not found, and not at {alt} either. If the runs tree was "
        f"reorganised, pass the current path explicitly.")


def is_final_fit(run_dir) -> bool:
    p = Path(run_dir) / "config.json"
    return p.exists() and bool(json.loads(p.read_text()).get("final_fit", False))


def default_ckpt(run_dir) -> str:
    return "last.pt" if is_final_fit(run_dir) else "best.pt"


def load_tagger(run_dir, ckpt=None, device="cuda"):
    import dataclasses
    from rr.backbones import load_backbone
    from rr.model import RRTagger
    from rr.peft import apply_peft
    from rr.train import TrainConfig

    run = Path(run_dir)
    raw = json.loads((run / "config.json").read_text())
    known = {f.name for f in dataclasses.fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in raw.items() if k in known})
    cfg.domains = tuple(cfg.domains)

    if ckpt is None:
        ckpt = default_ckpt(run)
    if raw.get("final_fit") and Path(ckpt).name == "best.pt":
        raise FileNotFoundError(
            f"{run} is a FINAL-FIT run (trained on train+dev, no selection): it has no "
            f"best.pt by design. Use last.pt (the default when --ckpt is omitted).")
    p = Path(ckpt) if Path(ckpt).exists() else run / ckpt
    if not p.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt!r} or {run / ckpt}")
    state = torch.load(p, map_location="cpu", weights_only=False)
    vocab = D.load_labels(cfg.data_dir)

    if raw.get("keep_every") and raw.get("teacher_run"):


        from rr.distill import make_student
        t = _resolve_run(raw["teacher_run"], cfg.backbone)
        teacher, _, _ = load_tagger(t, None, "cpu")
        model = make_student(teacher, int(raw["keep_every"]))
        model.load_state_dict(state["model"])
    else:
        bb = load_backbone(cfg.backbone, cfg.ckpt, device="cpu")
        model = RRTagger(bb, n_labels=len(vocab), doc_layers=cfg.doc_layers,
                         doc_heads=cfg.doc_heads, dropout=cfg.dropout, use_crf=cfg.use_crf,
                         max_sents=cfg.max_doc_sents + 8, sent_batch=cfg.sent_batch,
                         ctx_tokens=getattr(cfg, "ctx_tokens", 0))


        apply_peft(model, cfg.peft, **cfg.peft_kw)
        model.load_state_dict(state["model"])
    model.to(device).eval()
    b = state.get("best", {}) or {}
    mf = b.get("macro_f1")
    mf = "n/a (final-fit: nothing selected on dev)" if mf is None else f"{mf:.4f}"
    print(f"[load] {p} @ epoch {state.get('epoch')} (dev macro-F1 {mf})")
    print(f"[load] backbone={cfg.backbone} peft={cfg.peft} crf={cfg.use_crf} "
          f"doc_layers={cfg.doc_layers}")
    return model, cfg, vocab


@torch.no_grad()
def tag(model, docs, cfg, vocab, device="cuda"):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(cfg.tokenizer)
    out = {}
    for n, d in enumerate(docs, 1):
        e = D.encode_doc(d, tok, vocab, cfg.max_sent_len)
        wins = D.windows(len(d), cfg.max_doc_sents, cfg.doc_stride)
        per_win = []
        for lo, hi in wins:
            ids = torch.tensor(e.ids[lo:hi], device=device)
            msk = torch.tensor(e.mask[lo:hi], device=device)
            per_win.append(model.predict(ids, msk).tolist())
        tags = []
        for i in range(len(d)):
            w = D.owner(wins, i)
            tags.append(per_win[w][i - wins[w][0]])
        out[d.id] = [vocab[t] for t in tags]
        if n % 5 == 0 or n == len(docs):
            print(f"  {n}/{len(docs)} documents")
    return out


@torch.no_grad()
def tag_probs(model, docs, cfg, vocab, device="cuda"):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(cfg.tokenizer)
    out = {}
    for n, d in enumerate(docs, 1):
        e = D.encode_doc(d, tok, vocab, cfg.max_sent_len)
        wins = D.windows(len(d), cfg.max_doc_sents, cfg.doc_stride)
        per_win = []
        for lo, hi in wins:
            ids = torch.tensor(e.ids[lo:hi], device=device)
            msk = torch.tensor(e.mask[lo:hi], device=device)
            per_win.append(model.marginals(ids, msk).cpu())
        rows = []
        for i in range(len(d)):
            w = D.owner(wins, i)
            rows.append([float(x) for x in per_win[w][i - wins[w][0]]])
        out[d.id] = rows
        if n % 5 == 0 or n == len(docs):
            print(f"  {n}/{len(docs)} documents")
    return out


def crf_params(model) -> dict | None:
    if getattr(model, "crf", None) is None:
        return None
    c = model.crf
    return {"transitions": c.transitions.detach().float().cpu().tolist(),
            "start": c.start.detach().float().cpu().tolist(),
            "end": c.end.detach().float().cpu().tolist()}


def write_submission(pred: dict, docs, vocab, out_dir) -> Path:
    errs = validate_submission(pred, docs, vocab)
    if errs:
        raise SystemExit("submission invalid:\n  - " + "\n  - ".join(errs[:15]))
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pj = out / "prediction.json"
    pj.write_text(json.dumps(pred, ensure_ascii=False))
    zp = out / "submission.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["prediction.json"], z.namelist()
    n = sum(len(v) for v in pred.values())
    import collections
    dist = collections.Counter(l for v in pred.values() for l in v)
    print(f"[ok] {zp}  ({len(pred)} documents, {n} sentences)")
    print("[ok] predicted label distribution: " +
          "  ".join(f"{l}={dist[l]}" for l in vocab if dist[l]))
    missing = [l for l in vocab if not dist[l]]
    if missing:
        print(f"[warn] {len(missing)} label(s) never predicted: {missing}. Each scores "
              f"F1=0 and costs 1/{len(vocab)} = {1/len(vocab):.3f} of the macro metric.")
    return zp

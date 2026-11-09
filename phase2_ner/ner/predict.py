from __future__ import annotations

import json
import zipfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ner import data as D


def load_tagger(run_dir, ckpt: str = "best.pt", device: str = "cuda"):
    from ner.backbones import load_backbone
    from ner.model import NERTagger
    from ner.peft import apply_peft
    from ner.train import TrainConfig

    import dataclasses

    run = Path(run_dir)
    raw = json.loads((run / "config.json").read_text())


    known = {f.name for f in dataclasses.fields(TrainConfig)}
    extra = {k: v for k, v in raw.items() if k not in known}
    cfg = TrainConfig(**{k: v for k, v in raw.items() if k in known})

    p = Path(ckpt)
    if not p.exists():
        p = run / ckpt
    if not p.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt!r} or {run / ckpt}")

    state = torch.load(p, map_location="cpu", weights_only=False)

    if "teacher_run" in extra:


        from ner.distill import make_student
        teacher, _ = load_tagger(extra["teacher_run"], "best.pt", device="cpu")
        model = make_student(teacher, extra.get("keep_every", 2))
    else:
        bb = load_backbone(cfg.backbone, cfg.ckpt, device="cpu")
        model = NERTagger(bb, n_tags=len(D.TAGS), dropout=cfg.dropout, use_crf=cfg.use_crf)


        apply_peft(model, cfg.peft, **cfg.peft_kw)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    print(f"[load] {p} @ epoch {state['epoch']} step {state['step']} "
          f"(dev macro-F1 {state['best']['macro_f1']:.4f})")
    print(f"[load] backbone={cfg.backbone} peft={cfg.peft} crf={cfg.use_crf}")
    return model, cfg


@torch.no_grad()
def tag_documents(model, texts: dict, cfg, device="cuda", batch_size=8) -> dict:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("artifacts/tokenizer_16k/tokenizer.json")
    max_len = min(cfg.max_len, model.backbone.max_positions)

    chunks = {d: D.encode_document(t, [], tok, max_len=max_len, stride=cfg.stride)
              for d, t in texts.items()}
    flat = [c for cs in chunks.values() for c in cs]
    order = {id(c): i for i, c in enumerate(flat)}
    tag_out = [None] * len(flat)

    loader = DataLoader(flat, batch_size=batch_size, shuffle=False,
                        collate_fn=D.collate, num_workers=0)
    i = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        am = batch["attention_mask"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=str(device).startswith("cuda")):
            tags = model.predict_tags(ids, am)
        for b in range(ids.shape[0]):
            L = int(am[b].sum())
            tag_out[i + b] = tags[b, :L].tolist()
        i += ids.shape[0]
        if i % (batch_size * 20) == 0:
            print(f"  {i}/{len(flat)} chunks")

    return {d: D.decode_chunks(texts[d], cs, [tag_out[order[id(c)]] for c in cs])
            for d, cs in chunks.items()}


def validate(preds: dict, texts: dict) -> list[str]:
    errs = []
    missing = set(texts) - set(preds)
    extra = set(preds) - set(texts)
    if missing:
        errs.append(f"{len(missing)} test ids have no entry, e.g. {sorted(missing)[:5]}")
    if extra:
        errs.append(f"{len(extra)} unknown ids, e.g. {sorted(extra)[:5]}")

    for doc_id, spans in preds.items():
        if doc_id not in texts:
            continue
        n = len(texts[doc_id])
        for s in spans:
            if not (isinstance(s, (list, tuple)) and len(s) in (3, 4)):
                errs.append(f"{doc_id}: malformed span {s!r}")
                continue
            a, b, lab = s[0], s[1], s[2]
            if not (isinstance(a, int) and isinstance(b, int)):
                errs.append(f"{doc_id}: non-integer offsets {s!r}")
            elif not (0 <= a < b <= n):
                errs.append(f"{doc_id}: span [{a},{b}) outside document of length {n}")
            elif texts[doc_id][a] .isspace() or texts[doc_id][b - 1].isspace():


                errs.append(f"{doc_id}: span [{a},{b}) has whitespace at an edge: "
                            f"{texts[doc_id][a:b]!r}")
            if lab not in D.LABELS:
                errs.append(f"{doc_id}: unknown label {lab!r}")
    return errs


def write_submission(preds: dict, out_dir, texts: dict | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if texts is not None:
        errs = validate(preds, texts)
        if errs:
            raise SystemExit("submission invalid:\n  - " + "\n  - ".join(errs[:20])
                             + (f"\n  ... and {len(errs)-20} more" if len(errs) > 20 else ""))

    payload = {d: [list(s) for s in spans] for d, spans in preds.items()}
    pj = out / "prediction.json"
    pj.write_text(json.dumps(payload, ensure_ascii=False))

    zp = out / "submission.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["prediction.json"], z.namelist()

    n = [len(v) for v in preds.values()]
    per_label = {}
    for spans in preds.values():
        for s in spans:
            per_label[s[2]] = per_label.get(s[2], 0) + 1
    print(f"[ok] {zp}  ({len(preds)} docs, {sum(n)} spans, "
          f"mean {sum(n)/max(len(n),1):.1f}/doc, max {max(n) if n else 0})")
    print("[ok] per label: " + "  ".join(f"{k}={per_label.get(k,0)}" for k in D.LABELS))
    missing_labels = [k for k in D.LABELS if not per_label.get(k)]
    if missing_labels:


        print(f"[warn] NEVER predicted: {missing_labels} -- each costs 1/12 of macro-F1")
    return zp


def predict(run_dir, test_file=None, out_dir=None, ckpt="best.pt",
            device="cuda", batch_size=8):
    model, cfg = load_tagger(run_dir, ckpt, device)
    test_file = test_file or f"{cfg.data_dir}/Test/fold3-text.json"
    texts, _ = D.load_fold(str(Path(test_file).parent.parent), "fold3")
    print(f"[data] {len(texts)} test documents")
    preds = tag_documents(model, texts, cfg, device, batch_size)
    return write_submission(preds, out_dir or Path(run_dir) / "submission", texts)

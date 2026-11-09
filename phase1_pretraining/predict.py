from __future__ import annotations
import json, zipfile
from pathlib import Path

import torch

from config import Config
from dataset import load_tokenizer
from generate import generate, generate_diffusion, generate_mlm, generate_encdec, truncate_words, strip_echo
from models import build


def load_model(run_dir, ckpt="best.pt", device="cuda"):
    run = Path(run_dir)
    cfg = Config.load(run / "config.json")
    p = Path(ckpt)
    if not p.exists():
        p = run / ckpt
    if not p.exists():
        print(f"[load] WARNING: {ckpt!r} not found under {run} -- falling back to last.pt")
        p = run / "last.pt"
    state = torch.load(p, map_location=device, weights_only=False)
    model = build(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"[load] {p} @ step {state['step']} ({state['tokens']/1e6:.0f}M tokens)")
    return model, cfg


def predict(run_dir, test_file, out_dir=None, ckpt="best.pt", device="cuda",
            batch_size=8, **decode):
    model, cfg = load_model(run_dir, ckpt, device)
    tok = load_tokenizer(cfg.tokenizer_dir)


    diff = {k: decode.pop(k, None) for k in ("steps", "schedule", "gumbel")}
    if cfg.arch == "diffusion":
        diff = {"steps": diff["steps"] or cfg.diff_steps,
                "schedule": diff["schedule"] or cfg.diff_schedule,
                "gumbel": cfg.diff_gumbel if diff["gumbel"] is None else diff["gumbel"]}


        print(f"[decode] {diff} + {dict(decode)}")
    prompts = {str(k): (v["text"] if isinstance(v, dict) else v)
               for k, v in json.loads(Path(test_file).read_text()).items()}
    ids = list(prompts)
    preds = {}

    for i in range(0, len(ids), batch_size):
        keys = ids[i:i + batch_size]
        enc = [tok.encode(prompts[k]).ids[-cfg.max_seq_len:] for k in keys]
        L = max(len(e) for e in enc)


        x = torch.zeros(len(enc), L, dtype=torch.long)
        for j, e in enumerate(enc):
            x[j, L - len(e):] = torch.tensor(e)
        if cfg.arch == "encoder":

            rows = generate_mlm(model, enc, **{k: v for k, v in decode.items()
                                               if k != "use_cache"})
            texts = [tok.decode(r) for r in rows]

        elif cfg.arch == "diffusion":


            rows = generate_diffusion(model, enc, **diff,
                                      **{k: v for k, v in decode.items()
                                         if k != "use_cache"})
            texts = [tok.decode(r) for r in rows]
        elif cfg.arch == "encdec":
            xs = x.to(device)
            gen = generate_encdec(model, xs, (xs != 0).long(),
                                  **{k: v for k, v in decode.items() if k != "use_cache"})
            texts = [tok.decode(gen[j].tolist()) for j in range(len(keys))]
        else:
            gen = generate(model, x.to(device), **decode)
            texts = [tok.decode(gen[j].tolist()) for j in range(len(keys))]
        for j, k in enumerate(keys):
            preds[k] = {"text": truncate_words(strip_echo(prompts[k], texts[j]), 128)}
        print(f"  {min(i+batch_size, len(ids))}/{len(ids)}")

    preds = _rescue_degenerate(model, cfg, tok, prompts, preds, device, decode)

    out = Path(out_dir or Path(run_dir) / "submission")
    return write_submission(preds, out, prompts)


_DEGENERATE_WORDS = 30
_SAFE = dict(repetition_penalty=1.15, no_repeat_ngram=4)


def _rescue_degenerate(model, cfg, tok, prompts, preds, device, decode):
    bad = [k for k, v in preds.items()
           if len(v["text"].split()) < _DEGENERATE_WORDS]
    if not bad:
        print("[rescue] no degenerate outputs")
        return preds
    if cfg.arch not in ("decoder", "efficient", "mamba"):
        print(f"[rescue] {len(bad)} short output(s) but arch={cfg.arch} has no "
              f"repetition controls to fall back on; leaving them")
        return preds

    print(f"[rescue] {len(bad)}/{len(preds)} output(s) under {_DEGENERATE_WORDS} words "
          f"-- regenerating with {_SAFE}")
    safe = {**decode, **_SAFE}
    for i in range(0, len(bad), 8):
        keys = bad[i:i + 8]
        enc = [tok.encode(prompts[k]).ids[-cfg.max_seq_len:] for k in keys]
        L = max(len(e) for e in enc)
        x = torch.zeros(len(enc), L, dtype=torch.long)
        for j, e in enumerate(enc):
            x[j, L - len(e):] = torch.tensor(e)
        gen = generate(model, x.to(device), **{k: v for k, v in safe.items()
                                               if k != "use_cache"})
        for j, k in enumerate(keys):
            txt = truncate_words(
                strip_echo(prompts[k], tok.decode(gen[j].tolist())), 128)
            before = len(preds[k]["text"].split())
            if len(txt.split()) > before:
                preds[k] = {"text": txt}
            print(f"[rescue]   {k}: {before} -> {len(preds[k]['text'].split())} words")
    return preds


def validate(preds: dict, prompts: dict) -> list[str]:
    errs = []
    missing = set(prompts) - set(preds)
    extra = set(preds) - set(prompts)
    if missing:
        errs.append(f"{len(missing)} missing ids, e.g. {sorted(missing)[:5]}")
    if extra:
        errs.append(f"{len(extra)} unknown ids, e.g. {sorted(extra)[:5]}")
    for k, v in preds.items():
        t = v["text"] if isinstance(v, dict) else v
        if not isinstance(t, str) or not t.strip():
            errs.append(f"{k}: empty text")
        elif len(t.split()) > 128:
            errs.append(f"{k}: {len(t.split())} words > 128")
    return errs


def write_submission(preds, out_dir, prompts=None):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if prompts:
        errs = validate(preds, prompts)
        if errs:
            raise SystemExit("submission invalid:\n  - " + "\n  - ".join(errs[:15]))
    pj = out / "prediction.json"
    pj.write_text(json.dumps(preds, ensure_ascii=False))
    zp = out / "submission.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["prediction.json"], z.namelist()
    n = [len(v["text"].split()) for v in preds.values()]
    print(f"[ok] {zp}  ({len(preds)} items, mean {sum(n)/len(n):.0f} words, "
          f"min {min(n)}, max {max(n)})")
    return zp

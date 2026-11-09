from __future__ import annotations
import json, math, os, random, time
from pathlib import Path

import numpy as np
import torch

from dataset import ShardDataset, Sampler
from loss import get_loss
from models import build, count_params
from tqdm.auto import tqdm


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    s = {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def _as_byte_cpu(x):
    if hasattr(x, "cpu"):
        x = x.cpu()
        return x.to(torch.uint8) if x.dtype != torch.uint8 else x
    return torch.as_tensor(x, dtype=torch.uint8)

def _strip_prefixes(sd):
    out = {}
    for k, v in sd.items():
        while True:
            for p in ("_orig_mod.", "module."):
                if k.startswith(p):
                    k = k[len(p):]
                    break
            else:
                break
        out[k] = v
    return out

def load_init_weights(model, path):


    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if (isinstance(ck, dict) and "model" in ck) else ck
    sd = _strip_prefixes(sd)

    tgt = model.state_dict()
    ok, bad, missing = {}, [], []
    for k, v in tgt.items():
        if k not in sd:
            missing.append(k)
        elif tuple(sd[k].shape) != tuple(v.shape):
            bad.append((k, tuple(sd[k].shape), tuple(v.shape)))
        else:
            ok[k] = sd[k]
    extra = [k for k in sd if k not in tgt]

    model.load_state_dict(ok, strict=False)

    print(f"[init_from] {path}")
    if isinstance(ck, dict):
        c = ck.get("cfg")
        src = c.get("name") if isinstance(c, dict) else getattr(c, "name", "?")
        print(f"[init_from] source: {src} @ step {ck.get('step','?')} "
              f"val_loss {ck.get('val_loss','?')}")
    print(f"[init_from] loaded {len(ok)}/{len(tgt)} tensors "
          f"({sum(t.numel() for t in ok.values())/1e6:.1f}M params)")
    if missing:
        print(f"[init_from] kept at random init ({len(missing)}): "
              f"{missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if bad:
        print(f"[init_from] SHAPE MISMATCH ({len(bad)}) -- kept at random init:")
        for k, a, b in bad[:8]:
            print(f"[init_from]   {k}: ckpt {a} vs model {b}")
    if extra:
        print(f"[init_from] unused in ckpt ({len(extra)}): "
              f"{extra[:8]}{' ...' if len(extra) > 8 else ''}")


def load_rng(s):
    if not s:
        return
    try:
        random.setstate(s["py"])
        np.random.set_state(s["np"])
        torch.set_rng_state(_as_byte_cpu(s["torch"]))
        if "cuda" in s and torch.cuda.is_available():
            states = [_as_byte_cpu(x) for x in s["cuda"]]
            if len(states) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(states)
            else:
                torch.cuda.set_rng_state(states[0])
    except Exception as e:
        print(f"[resume] WARNING: could not restore RNG state ({type(e).__name__}: {e}). "
              f"Continuing -- batch order is unaffected (it is a pure function of step), "
              f"but dropout/init randomness will differ from the original run.")


def build_optimizer(model, lr, weight_decay):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim < 2 or "embed" in n or "norm" in n else decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    try:
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), fused=torch.cuda.is_available())
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


def set_lr(opt, step, cfg):
    lr = cfg.lr * lr_at(step, cfg)
    for g in opt.param_groups:
        g["lr"] = lr
    return lr


def lr_at(step, cfg):
    if step < cfg.warmup:
        return (step + 1) / max(cfg.warmup, 1)

    sched = getattr(cfg, "schedule", "cosine")
    if sched == "constant":
        return 1.0

    if sched == "wsd":
        start = cfg.decay_start or int(cfg.max_steps * (1 - cfg.decay_frac))
        if step < start:
            return 1.0
        p = min(1.0, (step - start) / max(cfg.max_steps - start, 1))
        return 1.0 - p * (1 - cfg.min_lr_ratio)

    p = min(1.0, (step - cfg.warmup) / max(cfg.max_steps - cfg.warmup, 1))
    if sched == "linear":
        return 1.0 - p * (1 - cfg.min_lr_ratio)
    return cfg.min_lr_ratio + 0.5 * (1 - cfg.min_lr_ratio) * (1 + math.cos(math.pi * p))


def save_ckpt(path, model, opt, scaler, step, tokens, cfg, best, wall):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "opt": opt.state_dict(),
               "scaler": scaler.state_dict(), "step": step, "tokens": tokens,
               "cfg": cfg.__dict__, "cfg_hash": cfg.hash(), "shape_hash": cfg.shape_hash(),
               "best": best,
               "rng": rng_state(), "wall_s": wall}
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def log(run_dir, **kv):
    kv["time"] = time.time()
    with open(Path(run_dir) / "log.jsonl", "a") as f:
        f.write(json.dumps(kv, default=str) + "\n")


def train(cfg, teacher=None):
    cfg.check()
    seed_all(cfg.seed)
    run = cfg.run_dir
    run.mkdir(parents=True, exist_ok=True)
    cfg.save()
    device = cfg.device if torch.cuda.is_available() else "cpu"

    model = build(cfg).to(device)
    print(f"[model] {count_params(model)}")
    opt = build_optimizer(model, cfg.lr, cfg.weight_decay)
    set_lr(opt, 0, cfg)
    use_amp = device.startswith("cuda") and cfg.amp != "off"
    amp_dtype = torch.bfloat16 if cfg.amp == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and cfg.amp == "fp16")

    split = Path(cfg.shard_dir) / "index_train.json"
    ds = ShardDataset(cfg.shard_dir, index_file="index_train.json" if split.exists()
                      else "index.json")
    if not split.exists():
        print("[data] WARNING: no index_train.json -- training on ALL shards including any "
              "you evaluate on. Run: python run.py split")
    sampler = Sampler(ds, cfg.block_size, cfg.batch_size, cfg.seed)
    try:
        val = Sampler(ShardDataset(cfg.shard_dir, index_file="index_val.json"),
                      cfg.block_size, cfg.batch_size, seed=999)
    except Exception:
        val = None
        print("[data] no index_val.json -- validation disabled")
    print(f"[data] {ds.n_tokens/1e9:.3f}B tokens | {cfg.tokens_per_step:,} tokens/step")


    step, tokens, wall0, best = 0, 0, 0.0, {"val_loss": None, "step": None}
    ck = run / "last.pt"
    if ck.exists():
        p = torch.load(ck, map_location=device, weights_only=False)
        if p.get("shape_hash") != cfg.shape_hash():
            raise SystemExit(
                f"{ck} was built with a different MODEL or DATA config, so its weights and "
                f"batch order do not match this one.\n  changed: "
                f"{ {k: v for k, v in cfg.diff(p.get('cfg', {})).items() if k in cfg.SHAPE_KEYS} }"
                f"\n  Use a new --model/name, or restore the old settings. Do not resume.")
        changed = {k: v for k, v in cfg.diff(p.get("cfg", {})).items()
                   if k not in cfg.SHAPE_KEYS and k not in ("max_minutes", "device", "out_dir")}
        if changed:
            print(f"[resume] schedule changed since the checkpoint: {changed}")
        model.load_state_dict(p["model"]); opt.load_state_dict(p["opt"])
        scaler.load_state_dict(p["scaler"]); load_rng(p.get("rng"))
        step, tokens, best, wall0 = p["step"], p["tokens"], p["best"], p.get("wall_s", 0.0)
        print(f"[resume] step {step}, "
              f"{f'{tokens/1e6:.1f}M' if tokens >= 1e5 else f'{tokens:,}'} tokens seen")

    elif cfg.init_from:
        load_init_weights(model, cfg.init_from)
        print(f"[init_from] step 0, 0 tokens seen")
    loss_fn = get_loss(cfg.arch)
    model.train()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t_start = t_save = t_log = time.time()
    acc_loss = acc_tok = 0
    pbar = tqdm(total=cfg.max_steps, initial=step, unit="step", dynamic_ncols=True,
                desc=cfg.name, smoothing=0.1)

    while step < cfg.max_steps:
        opt.zero_grad(set_to_none=True)
        for micro in range(cfg.grad_accum):
            batch = sampler.torch_batch(step * cfg.grad_accum + micro, device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                if teacher is not None:
                    from loss import distill_lm_loss
                    l, ntok = distill_lm_loss(model, batch, cfg, teacher)
                else:
                    l, ntok = loss_fn(model, batch, cfg)
            scaler.scale(l / cfg.grad_accum).backward()
            acc_loss += float(l.detach()); acc_tok += ntok

        if cfg.grad_clip:
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        else:
            gnorm = torch.tensor(float("nan"))
        scaler.step(opt); scaler.update()
        step += 1; tokens += acc_tok
        set_lr(opt, step, cfg)

        if step % cfg.log_every == 0:
            dt = time.time() - t_log; t_log = time.time()
            tps = acc_tok * cfg.log_every / max(dt, 1e-9)
            mean_loss = acc_loss / cfg.grad_accum
            vram = (torch.cuda.max_memory_allocated() / 2**20) if device.startswith("cuda") else 0
            log(run, step=step, tokens=tokens, train_loss=mean_loss,
                ppl=math.exp(min(mean_loss, 20)), lr=opt.param_groups[0]["lr"],
                grad_norm=float(gnorm), tokens_per_s=round(tps, 1), vram_mb=round(vram, 1))
            pbar.set_postfix(loss=f"{mean_loss:.3f}", ppl=f"{math.exp(min(mean_loss,20)):.0f}",
                             lr=f"{opt.param_groups[0]['lr']:.1e}",
                             tok_s=f"{tps/1e3:.1f}k", vram=f"{vram:.0f}M", refresh=False)
        acc_loss = acc_tok = 0
        pbar.update(1)

        if val is not None and cfg.eval_every and step % cfg.eval_every == 0:
            vl = evaluate_loss(model, val, cfg, device, loss_fn)
            log(run, step=step, val_loss=vl, val_ppl=math.exp(min(vl, 20)))
            tqdm.write(f"  eval @ {step}: val_loss {vl:.4f}  ppl {math.exp(min(vl,20)):.1f}")
            if best["val_loss"] is None or vl < best["val_loss"]:
                best = {"val_loss": vl, "step": step}
                save_ckpt(run / "best.pt", model, opt, scaler, step, tokens, cfg, best,
                          time.time() - t_start + wall0)
            model.train()

        if (time.time() - t_save) / 60 >= cfg.save_every_min:
            save_ckpt(run / "last.pt", model, opt, scaler, step, tokens, cfg, best,
                      time.time() - t_start + wall0)
            t_save = time.time()
            tqdm.write(f"  [ckpt] step {step}  ({(time.time()-t_start)/60:.0f} min elapsed)")

        if cfg.max_minutes and (time.time() - t_start) / 60 >= cfg.max_minutes:
            tqdm.write("[stop] max_minutes reached -- saving for the next session")
            break
        if (run / "STOP").exists():
            (run / "STOP").unlink()
            tqdm.write("[stop] STOP file")
            break

    pbar.close()
    save_ckpt(run / "last.pt", model, opt, scaler, step, tokens, cfg, best,
              time.time() - t_start + wall0)
    summary = {"steps": step, "tokens": tokens, "best": best,
               "wall_min": round((time.time() - t_start + wall0) / 60, 1),
               "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)
               if device.startswith("cuda") else None,
               "params_M": count_params(model)["total_M"],
               "tokens_per_s": _steady_tps(run)}
    (run / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def _steady_tps(run_dir):
    p = Path(run_dir) / "log.jsonl"
    if not p.exists():
        return None
    vals = [json.loads(l)["tokens_per_s"] for l in p.read_text().splitlines()
            if l.strip() and "tokens_per_s" in l]
    vals = vals[3:] or vals
    return round(float(np.median(vals)), 1) if vals else None


@torch.no_grad()
def evaluate_loss(model, sampler, cfg, device, loss_fn):
    model.eval()
    tot = 0.0
    for i in tqdm(range(cfg.eval_batches), desc="  eval", leave=False, unit="batch"):
        b = sampler.torch_batch(i, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            l, _ = loss_fn(model, b, cfg)
        tot += float(l)
    return tot / cfg.eval_batches

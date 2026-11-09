from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pcr import data as D
from pcr import metrics as M
from pcr import encode as E

ROOT = Path(__file__).resolve().parent.parent
LEX = ROOT / "runs" / "lex"
TOPN = 100


@dataclass
class TrainConfig:
    name: str = "debug"
    backbone: str = "../checkpoints/B_grown"
    ckpt: str = "best.pt"
    out_dir: str = "runs"

    peft: str = "full"
    peft_kw: dict = field(default_factory=dict)

    K: int = 4
    K_eval: int = 8


    L: int = 1024
    pool: str = "mean"


    proj: bool = True

    batch_q: int = 16
    n_hard: int = 1
    hardneg: str = "auto"
    hardneg_top: int = 30
    hardneg_file: str = "runs/lex/hardneg_train.json"
    date_filter: str = "auto"
    centre_rank: bool = True
    centre_init: bool = True

    epochs: float = 3.0
    lr: float = 2e-5
    head_lr: float = 2e-4
    temp_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_frac: float = 0.06
    grad_clip: float = 1.0
    chunk_batch: int = 16
    eval_every: int = 0
    eval_at_start: bool = True

    max_train_q: int = 0
    max_steps: int = 0
    seed: int = 1337
    device: str = "cuda"

    @property
    def run_dir(self) -> Path:
        return ROOT / self.out_dir / self.name

    @property
    def k_eval(self) -> int:
        return self.K_eval or self.K

    def save(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text(json.dumps(asdict(self), indent=1))


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _parse_date(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return (int(v), 0, 0)
    if isinstance(v, (list, tuple)) and v:
        return tuple(int(x) for x in v) + (0,) * (3 - len(v))
    s = str(v).strip()
    try:
        if "/" in s:
            d, m, y = s.split("/")[:3]
            return (int(y), int(m), int(d))
        p = s.split("T")[0].split("-")
        return tuple(int(x) for x in p) + (0,) * (3 - len(p))
    except ValueError:
        return None


EXACT_SRC = ("header", "cue")


def load_dates(mode="auto"):
    f = LEX / "dates.json"
    if mode == "off" or (mode == "auto" and not f.exists()):
        return None
    raw = json.loads(f.read_text())
    srcf = LEX / "dates_src.json"
    src = json.loads(srcf.read_text()) if srcf.exists() else {}
    flat = {}
    for k, v in raw.items():
        if isinstance(v, dict) and k in ("train", "dev", "test", "pool", "dates"):
            flat.update(v)
        else:
            flat[k] = v
    any_mode = mode == "any"
    out = {k: (_parse_date(v.get("date") if isinstance(v, dict) else v),
               any_mode or src.get(k) in EXACT_SRC) for k, v in flat.items()}
    out = {k: v for k, v in out.items() if v[0] is not None}
    print(f"[dates] {len(out):,} docs dated, {sum(e for _, e in out.values()):,} usable as "
          f"query anchors (rule {'any' if any_mode else 'exactq'}) ({f})", flush=True)
    return out


def load_hardneg(mode="auto", path="runs/lex/hardneg_train.json"):
    f = ROOT / path
    if mode == "none" or (mode == "auto" and not f.exists()):
        return None
    if mode == "bm25" and not f.exists():
        raise FileNotFoundError(f"hardneg=bm25 but {f} does not exist yet")
    raw = json.loads(f.read_text())
    out = {q: [c[0] if isinstance(c, (list, tuple)) else c for c in v] for q, v in raw.items()}
    print(f"[hardneg] hard negatives for {len(out):,} queries ({f})", flush=True)
    return out


def rank(emb: np.ndarray, ids: list[str], q: dict, dates=None, topn=TOPN, centre=True):
    z = torch.from_numpy(emb).float()
    if centre:
        z = z - z.mean(0, keepdim=True)
    z = F.normalize(z, dim=-1)
    pos = {c: i for i, c in enumerate(ids)}
    S = (z[[pos[k] for k in q]] @ z.T).numpy()
    out = {}
    for r, qid in enumerate(q):
        s = S[r].copy()
        s[pos[qid]] = -np.inf
        key = s.copy()
        if dates is not None and qid in dates and dates[qid][1]:
            qd = dates[qid][0]
            late = np.array([c in dates and dates[c][0] > qd for c in ids])
            key[late] -= 10.0
        top = np.argsort(-key, kind="stable")[:topn]
        out[qid] = [[ids[j], round(float(s[j]), 5)] for j in top if np.isfinite(s[j])]
    return out


def evaluate(rk, q):
    m = M.score({k: [c for c, _ in v] for k, v in rk.items()}, q)
    gold = sum(len(q[k]["rel"]) for k in q)
    m["r100"] = sum(len({c for c, _ in rk[k]} & q[k]["rel"]) for k in q) / max(gold, 1)
    return m


def pool_ids(split):
    return sorted(D.load(split)[1])


def build_model(cfg, device):
    from pcr.backbones import load_backbone
    from pcr.model import DocEncoder
    from pcr.peft import apply_peft
    bb = load_backbone(cfg.backbone, cfg.ckpt, device="cpu")
    model = DocEncoder(bb, pool=cfg.pool, proj=cfg.proj, chunk_batch=cfg.chunk_batch).to(device)
    kw = dict(cfg.peft_kw)
    if cfg.peft == "prompt":


        cfg.L = E.L_MAX - int(kw.get("n_prompt", 20))
    if cfg.proj and cfg.centre_init:
        centre_head(model, cfg, device)
    apply_peft(model, cfg.peft, **kw)
    return model


def centre_head(model, cfg, device, n=256):
    ids = random.Random(0).sample(pool_ids("train"), n)
    with torch.no_grad():
        model.head.proj.bias.zero_()
        mu = E.embed_docs(model, ids, cfg.K, cfg.L, device).mean(0)
        model.head.proj.bias.copy_(-torch.from_numpy(mu))
    print(f"[centre] proj bias = -mean of {n} train-pool docs (|mu| {np.linalg.norm(mu):.2f})",
          flush=True)


def make_batches(pairs, B, rng):
    rng.shuffle(pairs)
    batches, cur, used, left = [], [], set(), []
    for p in pairs:
        if p[0] in used:
            left.append(p); continue
        cur.append(p); used.add(p[0])
        if len(cur) == B:
            batches.append(cur); cur, used = [], set()
            pending, left = left, []
            for lp in pending:
                if lp[0] in used:
                    left.append(lp)
                else:
                    cur.append(lp); used.add(lp[0])
                    if len(cur) == B:
                        batches.append(cur); cur, used = [], set()
    if len(cur) >= 2:
        batches.append(cur)
    return batches


def info_nce(zq, zc, scale, mask):
    s = scale * F.normalize(zq.float(), dim=-1) @ F.normalize(zc.float(), dim=-1).T
    loss = F.cross_entropy(s.masked_fill(mask, -1e4), torch.arange(zq.shape[0], device=zq.device))
    return loss, s


def train(cfg: TrainConfig):
    seed_all(cfg.seed)
    torch.set_num_threads(8)


    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for a GPU run (driver init failed?) -- not "
                           "falling back to the CPU; relaunch")
    device = cfg.device if torch.cuda.is_available() else "cpu"
    cfg.save()
    from pcr.peft import trainable_report

    tok = E.tokens()
    qtr, pool_tr = D.load("train")
    qdv, _ = D.load("dev")
    dv_ids, te_ids = pool_ids("dev"), pool_ids("test")
    if cfg.max_train_q:
        keep = random.Random(cfg.seed).sample(sorted(qtr), cfg.max_train_q)
        qtr = {k: qtr[k] for k in keep}
    dates = load_dates(cfg.date_filter)
    hn = load_hardneg(cfg.hardneg, cfg.hardneg_file)

    kd = None
    if cfg.peft == "distill":
        from pcr.distill import setup_distill
        model, kd = setup_distill(cfg, device, qtr, pool_tr)
        cfg.save()
    else:
        model = build_model(cfg, device)
    cfg.save()
    rep = trainable_report(model)
    print(f"[peft] {cfg.peft}: {rep['trainable']:,} / {rep['total']:,} ({rep['pct']:.2f}%) "
          f"trainable | groups {rep['groups']}", flush=True)

    temp_p = [p for n, p in model.named_parameters() if n == "head.logit_scale"]
    head_p = [p for n, p in model.named_parameters()
              if n.startswith("head.") and n != "head.logit_scale"]
    body_p = [p for n, p in model.named_parameters()
              if not n.startswith("head.") and p.requires_grad]
    groups = [{"params": body_p, "lr": cfg.lr, "weight_decay": cfg.weight_decay},

              {"params": head_p, "lr": cfg.head_lr, "weight_decay": 0.0},
              {"params": temp_p, "lr": cfg.temp_lr, "weight_decay": 0.0}]
    opt = torch.optim.AdamW([g for g in groups if g["params"]])

    pairs = [(qid, c) for qid in sorted(qtr) for c in sorted(qtr[qid]["rel"])]
    steps_per_epoch = math.ceil(len(pairs) / cfg.batch_q)
    total = max(1, int(steps_per_epoch * cfg.epochs))
    if cfg.max_steps:
        total = min(total, cfg.max_steps)
    warm = max(1, int(total * cfg.warmup_frac))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    print(f"[data] train {len(qtr)} queries, {len(pairs):,} pairs | {steps_per_epoch} "
          f"steps/epoch, {total} total | batch {cfg.batch_q} q + {cfg.n_hard} hard neg each "
          f"({'BM25' if hn else 'none: in-batch only'}) | K={cfg.K} (eval {cfg.k_eval}) "
          f"L={cfg.L} pool={cfg.pool} proj={cfg.proj} | date filter "
          f"{'ON' if dates else 'off'}", flush=True)

    log = open(cfg.run_dir / "log.jsonl", "a")
    best = {"key": (-1.0, -1.0)}
    t0 = time.time()

    def run_eval(step):
        te = time.time()
        emb = E.embed_docs(model, dv_ids, cfg.k_eval, cfg.L, device)
        rk = rank(emb, dv_ids, qdv, dates, centre=cfg.centre_rank)
        m = evaluate(rk, qdv)
        rec = {"step": step, "epoch": round(step / steps_per_epoch, 3), **m,
               "scale": round(model.head.scale().item(), 2),
               "eval_s": round(time.time() - te, 1), "min": round((time.time() - t0) / 60, 1)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"[eval] step {step} ep {rec['epoch']:.2f}: {M.report(m)} | R@100 "
              f"{m['r100']:.4f} | 1/T {rec['scale']} | eval {rec['eval_s']}s | "
              f"{rec['min']} min", flush=True)
        key = (round(m["f1"], 4), m["r100"])
        if key > best["key"]:
            best.update(key=key, step=step, m=m)
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "step": step},
                       cfg.run_dir / "best.pt")

    if cfg.eval_at_start:
        run_eval(0)

    rng = random.Random(cfg.seed)
    step, ep, s_hist = 0, 0, []
    peak = 0.0
    model.train()
    while step < total:
        for batch in make_batches(list(pairs), cfg.batch_q, random.Random(cfg.seed + ep)):
            ts = time.time()
            qids = [p[0] for p in batch]
            cands = [p[1] for p in batch]
            if hn and cfg.n_hard:
                for qid in qids:
                    gold = qtr[qid]["rel"]
                    pool = [c for c in hn.get(qid, []) if c not in gold and c != qid][:cfg.hardneg_top]
                    cands += rng.sample(pool, min(cfg.n_hard, len(pool)))
            mask = torch.tensor([[c == qid or (c in qtr[qid]["rel"] and j != i)
                                  for j, c in enumerate(cands)] for i, qid in enumerate(qids)],
                                device=device)
            x, m, cm = E.collate([tok[d] for d in qids + cands], cfg.K, cfg.L)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                z = model(x.to(device), m.to(device), cm.to(device))
            zq, zc = z[:len(qids)], z[len(qids):]
            loss, s = info_nce(zq, zc, model.head.scale(), mask)
            parts = {"nce": loss.item()}
            if kd is not None:
                loss, parts = kd.loss(loss, s, qids, cands)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                           cfg.grad_clip)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if device.startswith("cuda"):
                torch.cuda.synchronize()
                peak = max(peak, torch.cuda.max_memory_allocated() / 1e9)
            s_hist.append(time.time() - ts)
            if step % 10 == 0 or step <= 3:
                rec = {"step": step, "loss": round(loss.item(), 4),
                       **{k: round(v, 4) for k, v in parts.items()},
                       "lr": sched.get_last_lr()[0], "s_step": round(s_hist[-1], 2)}
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(f"[train] {rec} | {peak:.1f} GB", flush=True)
            if cfg.eval_every and step % cfg.eval_every == 0 and step < total:
                run_eval(step)
            if step >= total:
                break
        ep += 1
        run_eval(step)


    state = torch.load(cfg.run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    qte, pool_te = D.load("test")
    out = {}
    for split, ids, qs in (("dev", dv_ids, qdv), ("test", te_ids, qte)):
        emb = E.embed_docs(model, ids, cfg.k_eval, cfg.L, device)
        np.save(cfg.run_dir / f"emb_{split}.npy", emb.astype(np.float16))
        (cfg.run_dir / f"emb_ids_{split}.json").write_text(json.dumps(ids))
        rk = rank(emb, ids, qs, dates, centre=cfg.centre_rank)
        (cfg.run_dir / f"rank_{split}.json").write_text(json.dumps(rk))
        out[split] = rk
    dm = evaluate(out["dev"], qdv)
    zp = M.write_submission({k: [c for c, _ in v] for k, v in out["test"].items()}, qte,
                            pool_te, ROOT / "subs" / f"{cfg.name}.zip")
    summ = {"name": cfg.name, "peft": cfg.peft, "peft_kw": cfg.peft_kw, "K": cfg.K,
            "K_eval": cfg.k_eval, "pool": cfg.pool, "best_step": best["step"],
            "dev": dm, "date_filter": dates is not None, "hardneg": hn is not None,
            "trainable": rep["trainable"], "total": rep["total"], "pct": round(rep["pct"], 3),
            "steps": step, "s_per_step": round(float(np.median(s_hist)), 2) if s_hist else None,
            "wall_min": round((time.time() - t0) / 60, 1), "peak_vram_gb": round(peak, 2),
            "submission": str(zp)}
    (cfg.run_dir / "summary.json").write_text(json.dumps(summ, indent=1))
    print(f"[done] {cfg.name}: best step {best['step']} | dev {M.report(dm)} | "
          f"{summ['s_per_step']} s/step | {summ['peak_vram_gb']} GB | {zp}", flush=True)
    return summ


def zeroshot(device="cuda", Ks=(1, 2, 4, 8), out_name="zeroshot"):
    from pcr.backbones import load_backbone
    from pcr.model import DocEncoder
    torch.set_num_threads(8)
    out_dir = ROOT / "runs" / "neural"
    bb = load_backbone(TrainConfig.backbone, "best.pt", device="cpu")
    model = DocEncoder(bb, pool="mean", proj=False).to(device).eval()
    qdv, _ = D.load("dev")
    ids = pool_ids("dev")
    dmodes = {"off": None, "exactq": load_dates("exactq"), "any": load_dates("any")} \
        if (LEX / "dates.json").exists() else {"off": None}
    t = time.time()
    cache = out_dir / "zs_dev_chunks.npy"
    if cache.exists() and np.load(out_dir / "zs_dev_cmask.npy").shape == (len(ids), max(Ks)):
        v, cm = np.load(cache).astype(np.float32), np.load(out_dir / "zs_dev_cmask.npy")
        enc_s = json.loads((out_dir / f"{out_name}.json").read_text())["encode_seconds"] \
            if (out_dir / f"{out_name}.json").exists() else None
    else:
        v, cm = E.embed_docs(model, ids, max(Ks), E.L_MAX, device, chunks=True)
        enc_s = time.time() - t
        np.save(cache, v.astype(np.float16))
        np.save(out_dir / "zs_dev_cmask.npy", cm)
    print(f"[zeroshot] {len(ids)} dev docs x {max(Ks)} chunks ({cm.sum():,} real chunks), "
          f"encode {enc_s}s", flush=True)
    res = []
    for pool in ("first", "mean", "max"):
        for K in (Ks if pool != "first" else (1,)):
            z = E.pool_chunks(v, cm, pool, K)
            for centred in (False, True):
                for dname, df in dmodes.items():
                    m = evaluate(rank(z, ids, qdv, df, centre=centred), qdv)
                    r = {"pool": pool, "K": K, "centred": centred, "date_filter": dname,
                         "f1": round(m["f1"], 4), "p": round(m["p"], 4), "r": round(m["r"], 4),
                         "r100": round(m["r100"], 4)}
                    res.append(r)
                    print(f"  {pool:5s} K={K} centred={centred!s:5s} date={dname:6s}"
                          f" F1@7 {r['f1']:.4f} R@100 {r['r100']:.4f}", flush=True)
    (out_dir / f"{out_name}.json").write_text(json.dumps(
        {"encode_seconds": round(enc_s, 1) if enc_s else None, "chunks": int(cm.sum()), "results": res}, indent=1))
    return res

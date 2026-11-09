from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--run-a", default="runs/A_long", help="first model")
ap.add_argument("--run-b", default="runs/M_mamba", help="second model")
ap.add_argument("--ckpt", default="best.pt")
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--lam", type=float, default=0.5, help="weight on run-a")
ap.add_argument("--sweep", action="store_true", help="try several lambdas on the proxy")
ap.add_argument("--lams", default="0.0,0.3,0.5,0.7,1.0")
ap.add_argument("--submit", action="store_true", help="write a submission from the test set")
ap.add_argument("--n", type=int, default=174)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--max-new-tokens", type=int, default=220)
ap.add_argument("--temperature", type=float, default=0.8)
ap.add_argument("--top-p", type=float, default=0.92)
ap.add_argument("--top-k", type=int, default=50)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--heldout", default="data/heldout_phase1.json")
ap.add_argument("--test", default="data/Phase_1_Pretraining/Test/test.json")
ap.add_argument("--tokenizer", default="artifacts/tokenizer_16k")
ap.add_argument("--out", default="runs/ensemble")
a = ap.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

import torch
sys.path.insert(0, str(Path(__file__).parent))
from config import Config
from dataset import load_tokenizer
from generate import filter_logits, truncate_words, strip_echo
from metrics import bertscore, rouge_l, bleu, distinct_n
from models import build

TARGET_WORDS = 128


def load(run_dir):
    run = Path(run_dir)
    cfg = Config.load(run / "config.json")
    st = torch.load(run / a.ckpt, map_location="cuda", weights_only=False)
    m = build(cfg).cuda().eval()
    m.load_state_dict(st["model"])
    print(f"[load] {run/a.ckpt} arch={cfg.arch} vocab={cfg.vocab_size} "
          f"ctx={cfg.max_seq_len} @ step {st['step']:,}")
    return m, cfg


@torch.no_grad()
def generate_ensemble(models, ids, lam, max_new_tokens, temperature, top_k, top_p):
    mA, mB = models
    ctx = 1024
    cur = ids[:, -ctx:]
    pastA = pastB = None
    step_in, out = cur, []
    for _ in range(max_new_tokens):
        rA = mA(step_in, past_kvs=pastA, use_cache=True)
        rB = mB(step_in, past_kvs=pastB, use_cache=True)
        pastA, pastB = rA["past_kvs"], rB["past_kvs"]
        t = max(temperature, 1e-6)
        pA = (rA["logits"][:, -1, :].float() / t).softmax(-1)
        pB = (rB["logits"][:, -1, :].float() / t).softmax(-1)
        p = lam * pA + (1.0 - lam) * pB


        p = filter_logits(p.clamp_min(1e-12).log(), top_k, top_p).softmax(-1)
        nxt = torch.multinomial(p, 1).squeeze(-1)
        out.append(nxt)
        step_in = nxt[:, None]
    return torch.stack(out, 1)


def run(models, tok, prompts, encs, lam, seed):
    t0 = time.time()
    preds = []
    for i in range(0, len(encs), a.batch):
        chunk = encs[i:i + a.batch]
        L = max(len(e) for e in chunk)
        x = torch.zeros(len(chunk), L, dtype=torch.long)
        for j, e in enumerate(chunk):
            x[j, L - len(e):] = torch.tensor(e)
        torch.manual_seed(seed * 100003 + i)
        g = generate_ensemble(models, x.cuda(), lam, a.max_new_tokens,
                              a.temperature, a.top_k, a.top_p)
        for j in range(len(chunk)):
            txt = truncate_words(
                strip_echo(prompts[i + j], tok.decode(g[j].tolist())), TARGET_WORDS)
            preds.append(txt if txt.strip() else ".")
    return preds, time.time() - t0


mA, cfgA = load(a.run_a)
mB, cfgB = load(a.run_b)
if cfgA.vocab_size != cfgB.vocab_size:
    sys.exit(f"vocabularies differ ({cfgA.vocab_size} vs {cfgB.vocab_size}); the "
             f"distributions are not on the same support and cannot be averaged.")
tok = load_tokenizer(a.tokenizer)
models = (mA, mB)

if a.submit:
    raw = json.loads(Path(a.test).read_text())
    keys = list(raw)
    prompts = [(raw[k]["text"] if isinstance(raw[k], dict) else raw[k]) for k in keys]
    encs = [tok.encode(p).ids[-1024:] for p in prompts]
    print(f"[test] {len(keys)} documents, lambda={a.lam}")
    preds, secs = run(models, tok, prompts, encs, a.lam, a.seed)
    from predict import write_submission
    out = Path(a.out) / "submission"
    pm = {k: {"text": t} for k, t in zip(keys, preds)}
    write_submission(pm, out, {k: p for k, p in zip(keys, prompts)})
    print(f"[ok] {secs/60:.1f} min")
    sys.exit(0)

data = json.loads(Path(a.heldout).read_text())
keys = list(data)[:a.n]
prompts = [data[k]["prompt"] for k in keys]
golds = [data[k]["gold"] for k in keys]
encs = [tok.encode(p).ids[-1024:] for p in prompts]
print(f"[proxy] {len(keys)} examples\n")

lams = [float(x) for x in a.lams.split(",")] if a.sweep else [a.lam]
print(f"{'lambda':>7s}  {'bert':>7s} {'rougeL':>7s} {'bleu4':>7s} {'words':>11s} {'d4':>6s} {'min':>5s}")
print("-" * 60)
rows = []
for lam in lams:
    preds, secs = run(models, tok, prompts, encs, lam, a.seed)
    bs = bertscore(preds, golds)
    n = len(preds)
    r = dict(lam=lam, bert=bs["bertscore_f1"],
             rougeL=sum(rouge_l(q, g) for q, g in zip(preds, golds)) / n,
             bleu4=bleu(preds, golds),
             d4=sum(distinct_n(q, 4) for q in preds) / n,
             words=sum(len(q.split()) for q in preds) / n,
             minw=min(len(q.split()) for q in preds), secs=round(secs, 1))
    rows.append(r)
    tag = {0.0: " (B only)", 1.0: " (A only)"}.get(lam, "")
    print(f"{lam:7.2f}  {r['bert']:7.4f} {r['rougeL']:7.4f} {r['bleu4']:7.4f} "
          f"{r['words']:7.1f}/{r['minw']:<3d} {r['d4']:6.3f} {r['secs']:5.0f}s{tag}")

Path(a.out).mkdir(parents=True, exist_ok=True)
Path(a.out, "lambda_sweep.json").write_text(json.dumps(
    {"run_a": a.run_a, "run_b": a.run_b, "decoding":
     dict(temperature=a.temperature, top_p=a.top_p, top_k=a.top_k), "rows": rows},
    indent=2))

best = max(rows, key=lambda r: r["bert"])
ends = {r["lam"]: r["bert"] for r in rows if r["lam"] in (0.0, 1.0)}
print("-" * 60)
print(f"best lambda={best['lam']}  bert {best['bert']:.4f}")
if ends:
    solo = max(ends.values())
    print(f"best single model in this sweep: {solo:.4f}   "
          f"ensemble gain {best['bert'] - solo:+.4f}")
    if best["lam"] in (0.0, 1.0):
        print("The blend never beat a single model -- the two make the same mistakes, so "
              "averaging has nothing to cancel. Report it and keep the single model.")
print(f"\n-> {Path(a.out, 'lambda_sweep.json')}")

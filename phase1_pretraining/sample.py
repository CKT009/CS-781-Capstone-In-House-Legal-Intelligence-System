from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import torch

from config import Config
from dataset import load_tokenizer
from generate import generate, generate_mlm, generate_encdec, truncate_words, strip_echo
from models import build
from metrics import distinct_n, repetition_rate


def load(run_dir, ckpt, device):
    run = Path(run_dir)
    cfg = Config.load(run / "config.json")
    p = run / ckpt
    if not p.exists():
        p = run / "last.pt"
    st = torch.load(p, map_location=device, weights_only=False)
    model = build(cfg).to(device).eval()
    model.load_state_dict(st["model"])
    print(f"[load] {p}")
    print(f"       step {st['step']:,} | {st['tokens']/1e9:.3f}B tokens | "
          f"best val_loss {st['best'].get('val_loss')}")
    decay_at = int(cfg.max_steps * (1 - cfg.decay_frac)) if cfg.schedule == "wsd" else None
    if decay_at and st["step"] < decay_at:
        print(f"       NOTE: lr decay starts at step {decay_at:,} -- this checkpoint is "
              f"NOT annealed, so quality is below what the finished run will give.")
    return model, cfg


@torch.no_grad()
def run_one(model, tok, prompt, device, **dec):
    ids = tok.encode(prompt).ids[-model.cfg.max_seq_len:]
    if model.cfg.arch == "encoder":
        text = tok.decode(generate_mlm(model, [ids], **dec)[0])
    elif model.cfg.arch == "encdec":
        xs = torch.tensor([ids], device=device)
        text = tok.decode(generate_encdec(model, xs, (xs != 0).long(), **dec)[0].tolist())
    else:
        text = tok.decode(generate(model, torch.tensor([ids], device=device), **dec)[0].tolist())
    return truncate_words(strip_echo(prompt, text), 128)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/A_decoder")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--test", default="data/Phase_1_Pretraining/Test/test.json")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer_16k")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--compare", action="store_true",
                    help="one prompt, several decoding settings side by side")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    device = a.device if torch.cuda.is_available() else "cpu"
    model, cfg = load(a.run, a.ckpt, device)
    tok = load_tokenizer(a.tokenizer)
    prompts = json.loads(Path(a.test).read_text())
    prompts = {str(k): (v["text"] if isinstance(v, dict) else v) for k, v in prompts.items()}
    keys = list(prompts)[:a.n]

    SETTINGS = {
        "greedy":       dict(greedy=True, max_new_tokens=220),
        "t0.7 p0.90":   dict(temperature=0.7, top_p=0.90, top_k=50, repetition_penalty=1.15,
                             no_repeat_ngram=4, max_new_tokens=220),
        "t0.9 p0.95":   dict(temperature=0.9, top_p=0.95, top_k=50, repetition_penalty=1.10,
                             no_repeat_ngram=0, max_new_tokens=220),
        "t1.0 no-pen":  dict(temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
                             no_repeat_ngram=0, max_new_tokens=220),
    }

    if a.compare:
        k = keys[0]
        print("\n" + "=" * 78)
        print("PROMPT (last 60 words):")
        print("  ..." + " ".join(prompts[k].split()[-60:]))
        for name, dec in SETTINGS.items():
            txt = run_one(model, tok, prompts[k], device, **dec)
            print("\n" + "-" * 78)
            print(f"[{name}]  distinct-3 {distinct_n(txt,3):.2f}  "
                  f"repeat-4 {repetition_rate(txt,4):.2f}  words {len(txt.split())}")
            print(txt)
        print("\n" + "=" * 78)
        print("distinct-3 near 1.0 = varied; below 0.6 = looping.")
        print("repeat-4 above 0.2 = repeating whole phrases.")
        return

    dec = SETTINGS["t0.7 p0.90"]
    d3 = r4 = 0.0
    for k in keys:
        txt = run_one(model, tok, prompts[k], device, **dec)
        d3 += distinct_n(txt, 3); r4 += repetition_rate(txt, 4)
        print("\n" + "=" * 78)
        print(f"[{k}] PROMPT ...{' '.join(prompts[k].split()[-40:])}")
        print("-" * 78)
        print(txt)
    n = max(len(keys), 1)
    print("\n" + "=" * 78)
    print(f"mean distinct-3 {d3/n:.2f} | mean repeat-4 {r4/n:.2f} | over {n} prompts")


if __name__ == "__main__":
    sys.exit(main())

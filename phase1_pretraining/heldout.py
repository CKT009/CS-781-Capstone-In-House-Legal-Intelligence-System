from __future__ import annotations
import argparse, itertools, json, sys
from pathlib import Path

PROMPT_WORDS = 512
TARGET_WORDS = 128


def build(args):
    sys.path.insert(0, str(Path(__file__).parent))
    from dataset import stream_docs, SOURCES

    idx = json.loads((Path(args.shards) / "index.json").read_text())
    packed = {}
    for s in idx["shards"]:
        packed[s["source"]] = packed.get(s["source"], 0) + s["n_tokens"]


    CHARS_PER_TOKEN = 4.6
    out, usable = {}, [s for s in SOURCES if packed.get(s, 0) > 0]
    per_source = max(1, args.n // max(len(usable), 1))

    for src in SOURCES:
        used_tokens = packed.get(src, 0)
        if used_tokens == 0:
            print(f"  {src:16s} skipped (not packed)")
            continue
        budget_chars = used_tokens * CHARS_PER_TOKEN * (1 + args.margin)
        seen_chars, kept, exhausted = 0, 0, True
        for doc in stream_docs(args.data, src, min_chars=200):
            seen_chars += len(doc)
            if seen_chars < budget_chars:
                continue
            exhausted = False
            words = doc.split()
            if len(words) < PROMPT_WORDS + TARGET_WORDS:
                continue
            out[f"{src}_{kept}"] = {
                "prompt": " ".join(words[:PROMPT_WORDS]),
                "gold": " ".join(words[PROMPT_WORDS:PROMPT_WORDS + TARGET_WORDS])}
            kept += 1
            if kept >= per_source:
                break
        if exhausted:
            print(f"  {src:16s} 0 -- corpus was fully consumed during packing, "
                  f"so it contains no unseen text")
        else:
            print(f"  {src:16s} {kept} examples (skipped ~{used_tokens/1e6:.0f}M "
                  f"trained-on tokens first)")

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"[heldout] {len(out)} examples -> {p}")
    if not out:
        print("  WARNING: no examples. Every corpus was exhausted during packing; you can")
        print("  only evaluate on the validation shards, which the model also did not train")
        print("  on but which are only 20M tokens.")


GRID = {
    "temperature": [0.6, 0.7, 0.8, 0.9],
    "top_p": [0.85, 0.92, 0.97],
    "repetition_penalty": [1.0, 1.1, 1.2],
}


def tune(args):
    import torch
    sys.path.insert(0, str(Path(__file__).parent))
    from config import Config
    from dataset import load_tokenizer
    from generate import (generate, generate_mlm, generate_encdec, generate_diffusion,
                          truncate_words, strip_echo)
    from models import build as build_model
    from metrics import rouge_n, rouge_l, distinct_n, repetition_rate, bertscore

    device = args.device if torch.cuda.is_available() else "cpu"
    run = Path(args.run)
    cfg = Config.load(run / "config.json")
    st = torch.load(run / args.ckpt, map_location=device, weights_only=False)
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(st["model"])
    tok = load_tokenizer(args.tokenizer)
    print(f"[tune] {run/args.ckpt} @ step {st['step']:,}")

    data = json.loads(Path(args.heldout).read_text())
    keys = list(data)[:args.n]
    print(f"[tune] {len(keys)} examples x {len(list(itertools.product(*GRID.values())))} settings")

    @torch.no_grad()
    def gen(prompt, **dec):
        ids = tok.encode(prompt).ids[-cfg.max_seq_len:]
        if cfg.arch == "encoder":
            txt = tok.decode(generate_mlm(model, [ids], max_new_tokens=220, **dec)[0])
        elif cfg.arch == "diffusion":

            txt = tok.decode(generate_diffusion(model, [ids], steps=cfg.diff_steps,
                                                **dec)[0])
        elif cfg.arch == "encdec":
            xs = torch.tensor([ids], device=device)
            txt = tok.decode(generate_encdec(model, xs, (xs != 0).long(),
                                             max_new_tokens=220, **dec)[0].tolist())
        else:
            o = generate(model, torch.tensor([ids], device=device), max_new_tokens=220, **dec)
            txt = tok.decode(o[0].tolist())
        return truncate_words(strip_echo(prompt, txt), TARGET_WORDS)

    rows = []
    for t, p, rp in itertools.product(*GRID.values()):
        dec = dict(temperature=t, top_p=p, top_k=50, repetition_penalty=rp, no_repeat_ngram=4)
        preds = [gen(data[k]["prompt"], **dec) for k in keys]
        gold = [data[k]["gold"] for k in keys]
        n = len(keys)
        row = {"temperature": t, "top_p": p, "repetition_penalty": rp,
               "rouge1": sum(rouge_n(a, b, 1) for a, b in zip(preds, gold)) / n,
               "rougeL": sum(rouge_l(a, b) for a, b in zip(preds, gold)) / n,
               "distinct3": sum(distinct_n(a) for a in preds) / n,
               "repeat4": sum(repetition_rate(a) for a in preds) / n}
        if args.bertscore:
            bs = bertscore(preds, gold)
            if bs.get("bertscore_f1") is None:
                raise SystemExit(
                    f"bert_score failed: {bs.get('error')}\n"
                    f"  pip install bert-score\n"
                    f"  It downloads a scoring model on first use, so it needs network\n"
                    f"  access once. Re-run without --bertscore to rank by ROUGE-L instead.")
            row.update(bs)
        rows.append(row)
        print(f"  t={t} p={p} rp={rp} -> rougeL {row['rougeL']:.4f} "
              f"{'bertF1 ' + format(row.get('bertscore_f1', 0), '.4f') if args.bertscore else ''}")

    key = "bertscore_f1" if args.bertscore and "bertscore_f1" in rows[0] else "rougeL"
    rows.sort(key=lambda r: -r[key])
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\n[tune] ranked by {key} -> {args.out}")
    print("  best 5:")
    for r in rows[:5]:
        print(f"    t={r['temperature']} p={r['top_p']} rp={r['repetition_penalty']}  "
              f"{key}={r[key]:.4f}  distinct3={r['distinct3']:.2f}")
    b = rows[0]
    print(f"\n  use: --temperature {b['temperature']} --top-p {b['top_p']} "
          f"--rep-penalty {b['repetition_penalty']}")
    print("  CAVEAT: tuned on corpus text, not the official test prompts. Treat it as a")
    print("  starting point and confirm with one or two Codabench submissions.")


def export(args):
    import torch
    sys.path.insert(0, str(Path(__file__).parent))
    from config import Config
    from dataset import load_tokenizer
    from generate import (generate, generate_mlm, generate_encdec, generate_diffusion,
                          truncate_words, strip_echo)
    from models import build as build_model

    data = json.loads(Path(args.heldout).read_text())
    keys = list(data)[:args.n] if args.n else list(data)

    truth = {k: {"text": data[k]["gold"]} for k in keys}
    Path(args.truth_out).write_text(json.dumps(truth, ensure_ascii=False, indent=1))
    print(f"[export] truth   -> {args.truth_out} ({len(truth)} records)")

    device = args.device if torch.cuda.is_available() else "cpu"
    run = Path(args.run)
    cfg = Config.load(run / "config.json")
    st = torch.load(run / args.ckpt, map_location=device, weights_only=False)
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(st["model"])
    tok = load_tokenizer(args.tokenizer)
    print(f"[export] generating from {run/args.ckpt} @ step {st['step']:,}")

    dec = dict(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
               repetition_penalty=args.rep_penalty, no_repeat_ngram=args.no_repeat_ngram,
               max_new_tokens=args.max_new_tokens)
    pred = {}
    with torch.no_grad():
        for n, k in enumerate(keys, 1):
            prompt = data[k]["prompt"]
            ids = tok.encode(prompt).ids[-cfg.max_seq_len:]
            if cfg.arch == "encoder":
                raw = tok.decode(generate_mlm(model, [ids], **dec)[0])
            elif cfg.arch == "diffusion":
                raw = tok.decode(generate_diffusion(model, [ids],
                                                    steps=cfg.diff_steps, **dec)[0])
            elif cfg.arch == "encdec":
                xs = torch.tensor([ids], device=device)
                raw = tok.decode(generate_encdec(model, xs, (xs != 0).long(),
                                                 **dec)[0].tolist())
            else:
                raw = tok.decode(generate(model, torch.tensor([ids], device=device),
                                          **dec)[0].tolist())
            txt = truncate_words(strip_echo(prompt, raw), TARGET_WORDS)

            pred[k] = {"text": txt if txt.strip() else "."}
            if n % 25 == 0:
                print(f"  {n}/{len(keys)}")
    Path(args.pred_out).write_text(json.dumps(pred, ensure_ascii=False, indent=1))
    print(f"[export] predict -> {args.pred_out}")
    print(f"\nnow run:\n  python evaluate_phase1.py --truth {args.truth_out} "
          f"--prediction {args.pred_out}")


def score(args):
    sys.path.insert(0, str(Path(__file__).parent))
    from metrics import evaluate
    data = json.loads(Path(args.heldout).read_text())
    pred = json.loads(Path(args.pred).read_text())
    preds = {k: (v["text"] if isinstance(v, dict) else v) for k, v in pred.items()}
    gold = {k: v["gold"] for k, v in data.items()}
    print(json.dumps(evaluate(preds, gold, with_bertscore=args.bertscore), indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--data", default="data")
    b.add_argument("--n", type=int, default=300)
    b.add_argument("--shards", default="artifacts/shards",
                   help="read index.json to learn how much of each corpus was trained on")
    b.add_argument("--margin", type=float, default=0.10,
                   help="extra fraction to skip past the training region, for safety")
    b.add_argument("--out", default="data/heldout_phase1.json")
    b.set_defaults(func=build)

    t = sub.add_parser("tune")
    t.add_argument("--run", default="runs/A_decoder")
    t.add_argument("--ckpt", default="best.pt")
    t.add_argument("--tokenizer", default="artifacts/tokenizer_16k")
    t.add_argument("--heldout", default="data/heldout_phase1.json")
    t.add_argument("--n", type=int, default=100)
    t.add_argument("--bertscore", action="store_true", help="the official metric; slower")
    t.add_argument("--device", default="cuda")
    t.add_argument("--out", default="runs/decoding_grid.json")
    t.set_defaults(func=tune)

    e = sub.add_parser("export", help="write truth.json + prediction.json for the "
                                      "official evaluate_phase1.py")
    e.add_argument("--run", default="runs/A_decoder")
    e.add_argument("--ckpt", default="best.pt")
    e.add_argument("--tokenizer", default="artifacts/tokenizer_16k")
    e.add_argument("--heldout", default="data/heldout_phase1.json")
    e.add_argument("--n", type=int, default=0, help="0 = all")
    e.add_argument("--truth-out", default="truth.json")
    e.add_argument("--pred-out", default="pred_heldout.json")
    e.add_argument("--temperature", type=float, default=0.8)
    e.add_argument("--top-p", type=float, default=0.97)
    e.add_argument("--top-k", type=int, default=50)
    e.add_argument("--rep-penalty", type=float, default=1.0)
    e.add_argument("--no-repeat-ngram", type=int, default=4)
    e.add_argument("--max-new-tokens", type=int, default=260)
    e.add_argument("--device", default="cuda")
    e.set_defaults(func=export)

    s = sub.add_parser("score")
    s.add_argument("--pred", required=True)
    s.add_argument("--heldout", default="data/heldout_phase1.json")
    s.add_argument("--bertscore", action="store_true")
    s.set_defaults(func=score)

    a = ap.parse_args()
    a.func(a)

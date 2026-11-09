import argparse, json, sys
from pathlib import Path

from config import Config, PRESETS


MODEL = "D"

TRAIN = dict(
    shard_dir="artifacts/shards",
    tokenizer_dir="artifacts/tokenizer_16k",
    out_dir="runs",
    max_steps=40000,
    warmup=500,
    max_minutes=0,
    schedule="wsd",
    decay_frac=0.2,
    lr=3e-4,
    amp="bf16",
    batch_size=16,
    grad_accum=1,
    grad_checkpoint=True,
    eval_every=500,
    save_every_min=20,
    log_every=20,
)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="`python run.py` with no arguments trains MODEL using the TRAIN dict "
                    "at the top of this file.")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("params")

    p = sub.add_parser("tokenizer")
    p.add_argument("--vocab", type=int, default=16000)
    p.add_argument("--sample", type=int, default=50000)
    p.add_argument("--data", default="data")
    p.add_argument("--out", default=None)

    p = sub.add_parser("pack")
    p.add_argument("--budget", type=float, default=1.5e9)
    p.add_argument("--data", default="data")
    p.add_argument("--tokenizer", default="artifacts/tokenizer_16k")
    p.add_argument("--out", default="artifacts/shards")
    p.add_argument("--only", nargs="*", default=None)

    p = sub.add_parser("train")
    p.add_argument("--model", default="A", choices=list(PRESETS))
    p.add_argument("--minutes", type=float, default=0)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--shards", default=None)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--set", nargs="*", default=[], help="override any Config field: lr=1e-4")

    p = sub.add_parser("predict")
    p.add_argument("--model", default="A")
    p.add_argument("--test", default="data/Phase_1_Pretraining/Test/test.json")
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.92)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--rep-penalty", type=float, default=1.15)


    p.add_argument("--no-repeat-ngram", type=int, default=4)


    p.add_argument("--diff-steps", type=int, default=None)
    p.add_argument("--schedule", default=None, choices=["cosine", "linear"])
    p.add_argument("--gumbel", type=float, default=None)

    p = sub.add_parser("split")
    p.add_argument("--shards", default="artifacts/shards")
    p.add_argument("--val-tokens", type=int, default=5_000_000,
                   help="held-out tokens per corpus")

    p = sub.add_parser("describe")
    p.add_argument("--shards", default="artifacts/shards")

    p = sub.add_parser("plot")
    p.add_argument("--runs", default="runs")

    a = ap.parse_args(argv)


    if a.cmd is None:
        from train import train
        cfg = Config(**{**PRESETS[MODEL], **TRAIN})
        print(f"[run] training preset {MODEL} from the TRAIN dict in run.py")
        train(cfg)
        return

    if a.cmd == "params":
        print(f"{'key':5s} {'name':14s} {'params':>10s}  50M-300M?")
        for k in PRESETS:
            c = Config(**PRESETS[k]); n = c.n_params()
            print(f"{k:5s} {c.name:14s} {n/1e6:9.1f}M  {'yes' if 50e6<=n<=300e6 else 'NO'}")
        return

    if a.cmd == "tokenizer":
        from dataset import stream_all, train_tokenizer, fertility, probe_tokens
        out = a.out or f"artifacts/tokenizer_{a.vocab//1000}k"
        tok = train_tokenizer(stream_all(a.data, max_docs_per_source=a.sample // 4),
                              out, vocab_size=a.vocab)
        sample = list(stream_all(a.data, max_docs_per_source=40))
        rep = {"vocab": a.vocab, "fertility": fertility(tok, sample),
               "probe_token_counts": probe_tokens(tok)}
        Path(out, "fertility.json").write_text(json.dumps(rep, indent=2))
        print(json.dumps(rep, indent=2))
        print("\nCompare tokens_per_word across vocab sizes, pick one, then FREEZE it.")
        return

    if a.cmd == "pack":
        from dataset import stream_docs, load_tokenizer, pack, write_index, SOURCES
        tok = load_tokenizer(a.tokenizer)
        mixture = {"high_courts": .40, "supreme_court": .25, "tribunals": .20, "acts": .15}
        shards = []
        for src, ratio in mixture.items():
            if a.only and src not in a.only:
                continue
            shards += pack(stream_docs(a.data, src), tok, a.out, src,
                           max_tokens=int(a.budget * ratio))
        write_index(a.out, shards, {"mixture": mixture, "tokenizer": a.tokenizer})
        from dataset import holdout_split, describe
        describe(a.out)
        holdout_split(a.out)
        return

    if a.cmd == "train":
        from train import train
        over = {}
        for s in a.set:
            k, _, v = s.partition("=")
            over[k] = json.loads(v) if v[0] in "0123456789[{\"-." else v
        base = {**PRESETS[a.model], **TRAIN}
        if a.shards:
            base["shard_dir"] = a.shards
        if a.tokenizer:
            base["tokenizer_dir"] = a.tokenizer
        if a.minutes:
            base["max_minutes"] = a.minutes
        cfg = Config(**{**base, **over})
        if a.steps:
            cfg.max_steps = a.steps
        train(cfg)
        return

    if a.cmd == "predict":
        from predict import predict
        name = Config(**PRESETS[a.model]).name
        predict(f"runs/{name}", a.test, ckpt=a.ckpt, temperature=a.temperature,
                top_p=a.top_p, top_k=a.top_k, repetition_penalty=a.rep_penalty,
                no_repeat_ngram=a.no_repeat_ngram,
                steps=a.diff_steps, schedule=a.schedule, gumbel=a.gumbel)
        return

    if a.cmd == "split":
        from dataset import holdout_split
        holdout_split(a.shards, a.val_tokens)
        return

    if a.cmd == "describe":
        from dataset import describe
        describe(a.shards)
        return

    if a.cmd == "plot":
        import plot
        runs = [d for d in Path(a.runs).iterdir() if (d / "log.jsonl").exists()]
        for d in runs:
            plot.loss_curve(d)
        if len(runs) > 1:
            plot.compare(runs); plot.throughput(runs)
        print("\n" + plot.summary_table(runs))
        return


if __name__ == "__main__":
    sys.exit(main())

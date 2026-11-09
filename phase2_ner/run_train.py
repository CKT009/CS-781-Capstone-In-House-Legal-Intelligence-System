from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

B_ENC = "../checkpoints/B_encoder"
F_DIF = "../checkpoints/F_diffusion"
A_DEC = "../checkpoints/A_decoder"
M_MAM = "../checkpoints/M_mamba"


EXPERIMENTS = {


    "full":      dict(peft="full",      lr=3e-5,
                      _why="full fine-tuning -- the baseline every other method is judged against"),
    "head_only": dict(peft="head_only", lr=3e-5, head_lr=3e-3, epochs=8,
                      _why="linear probe on frozen Phase-1 features -- the floor, and it "
                           "measures how much of the task the pretraining already solved"),
    "bitfit":    dict(peft="bitfit",    lr=1e-3,
                      _why="BitFit: biases and norms only, a sparse method with no added parameters"),
    "lora":      dict(peft="lora",      lr=3e-4, peft_kw=dict(r=8, alpha=16),
                      _why="LoRA on the attention projections -- low-rank reparameterisation"),
    "qlora":     dict(peft="qlora",     lr=3e-4, peft_kw=dict(r=8, alpha=16),
                      _why="QLoRA: the same rank-8 adapter as `lora` over a base "
                           "quantised to 4-bit NF4. Paired with `lora` it isolates "
                           "exactly what quantising the frozen weights costs"),
    "adapter":   dict(peft="adapter",   lr=3e-4, peft_kw=dict(bottleneck=64),
                      _why="Houlsby/Pfeiffer bottleneck adapters -- added modules, not reparameterisation"),
    "last4":     dict(peft="last_k",    lr=5e-5,  peft_kw=dict(k=4),
                      _why="partial/layer-wise: only the top 4 blocks adapt"),
    "prompt":    dict(peft="prompt",    lr=1e-3,  peft_kw=dict(n_prompt=20),
                      _why="prompt tuning: learned virtual tokens at the input embedding, the backbone untouched"),
    "prefix":    dict(peft="prefix",    lr=1e-3,  peft_kw=dict(n_prefix=16),
                      _why="prefix tuning: learned key/value vectors, the backbone untouched"),


    "lora_F":    dict(peft="lora", backbone=F_DIF, lr=3e-4, peft_kw=dict(r=8, alpha=16),
                      _why="the diffusion model's encoder -- bidirectional, but never had "
                           "a token-level loss of its own"),
    "lora_A":    dict(peft="lora", backbone=A_DEC, lr=3e-4, peft_kw=dict(r=8, alpha=16),
                      _why="causal decoder: a token cannot see its own right context. "
                           "Expected to trail B; that gap IS the result"),


    "lora_M":    dict(peft="lora", backbone=M_MAM, lr=3e-4,
                      peft_kw=dict(r=8, alpha=16, targets=("in_proj", "out_proj")),
                      _why="Mamba, the best Phase-1 generator, but causal and it ignores "
                           "the padding mask (sequences are right-padded for it)"),


    "lora_crf":  dict(peft="lora", lr=3e-4, use_crf=True, peft_kw=dict(r=8, alpha=16),
                      _why="CRF decoding: argmax can emit I-X after O, which is not a "
                           "valid BIO sequence and usually costs a whole span under "
                           "strict matching"),
    "lora_noweight": dict(peft="lora", lr=3e-4, class_weight="none",
                          peft_kw=dict(r=8, alpha=16),
                          _why="class-weighting ablation -- macro-F1 weights all 12 "
                               "labels equally, so the rare ones matter more than their "
                               "token counts suggest"),


    "distill":   dict(peft="full", lr=3e-5,
                      _why="MANDATORY: distil the best teacher into the smaller Phase-1 "
                           "student. See ner/distill.py -- run_train.py does not drive this."),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default=None, help="a name from EXPERIMENTS")
    ap.add_argument("--list", action="store_true", help="print the grid and exit")
    ap.add_argument("--check", action="store_true",
                    help="load data and backbone, verify offsets, then exit")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--name", default=None, help="run directory name (default: --exp)")
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="K=V",
                    help="override any TrainConfig field, e.g. epochs=8 lr=5e-5")
    a = ap.parse_args()

    if a.list:
        print(f"{'name':16s} {'peft':11s} {'backbone':12s}  why")
        for k, v in EXPERIMENTS.items():
            bb = Path(v.get("backbone", B_ENC)).name
            print(f"{k:16s} {v['peft']:11s} {bb:12s}  {v['_why']}")
        print(f"\n{len(EXPERIMENTS)} experiments defined; the phase requires >=6 "
              f"submissions and >=5 distinct adaptation techniques.")
        return

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    sys.path.insert(0, str(Path(__file__).parent))

    from ner.train import TrainConfig, train

    over = {}
    if a.exp:
        if a.exp not in EXPERIMENTS:
            sys.exit(f"unknown --exp {a.exp!r}; choose from {sorted(EXPERIMENTS)}")
        over = {k: v for k, v in EXPERIMENTS[a.exp].items() if not k.startswith("_")}
    if a.backbone:
        over["backbone"] = a.backbone
    for s in a.set:
        k, _, v = s.partition("=")
        try:
            over[k] = json.loads(v)
        except json.JSONDecodeError:
            over[k] = v

    cfg = TrainConfig(name=a.name or a.exp or "run", **over)

    if a.check:
        return check(cfg)

    print(f"\n{'=' * 72}\n  {cfg.name}  ->  {cfg.run_dir}\n{'=' * 72}")
    if a.exp:
        print(f"  {EXPERIMENTS[a.exp]['_why']}")
    print(f"  backbone {cfg.backbone}")
    print(f"  method   {cfg.peft} {cfg.peft_kw or ''}  crf={cfg.use_crf}")
    print(f"  data     train={cfg.train_folds} dev={cfg.dev_folds}  "
          f"max_len={cfg.max_len} stride={cfg.stride}")
    print(f"  select   on STRICT MACRO-F1, never on loss (Phase-1 Section 14)")
    print(f"{'=' * 72}\n")
    train(cfg)


def check(cfg):
    from tokenizers import Tokenizer
    from ner import data as D
    from ner.backbones import load_backbone, available

    print("[check] Phase-1 runs usable as a backbone:")
    for k, v in available().items():
        print(f"    {k:20s} {v}")

    d = Path(cfg.data_dir)
    if not d.exists():
        print(f"\n[check] data not found at {d} -- run `python fetch.py` first "
              f"(Phase 2 opens 7 October 2026).")
        return

    tok = Tokenizer.from_file("artifacts/tokenizer_16k/tokenizer.json")
    for fold in (*cfg.train_folds, *cfg.dev_folds):
        texts, golds = D.load_fold(cfg.data_dir, fold)
        n_spans = sum(len(v) for v in golds.values())
        print(f"\n[check] {fold}: {len(texts)} docs, {n_spans} gold spans")


        import collections
        exact = total = 0
        reasons, by_label = collections.Counter(), collections.Counter()
        for doc_id, txt in texts.items():
            spans = golds.get(doc_id, [])
            if not spans:
                continue
            chunks = D.encode_document(txt, spans, tok,
                                       max_len=cfg.max_len, stride=cfg.stride)
            back = set(map(tuple, D.decode_chunks(
                txt, chunks, [c.tag_ids for c in chunks])))
            want = set(map(tuple, spans))
            exact += len(back & want)
            total += len(want)

            starts, ends = set(), set()
            for c in chunks:
                for a, b in c.offsets:
                    a2, b2 = D.trim_span(txt, a, b)
                    starts.add(a2)
                    ends.add(b2)
            for s, e, lab in want - back:
                aligned = (s in starts) and (e in ends)
                reasons["BUG: aligned but lost" if aligned else "subword boundary"] += 1
                if not aligned:
                    by_label[lab] += 1

        pct = 100 * exact / max(total, 1)
        bugs = reasons["BUG: aligned but lost"]
        print(f"[check] gold -> BIO -> gold: {exact}/{total} recovered ({pct:.2f}%)")
        for r, n in reasons.most_common():
            print(f"[check]    {n:5d}  {r}")
        if by_label:
            print("[check]    subword losses by label: "
                  + "  ".join(f"{l}={n}" for l, n in by_label.most_common(6)))
        print(f"[check] => STRICT-F1 CEILING for token-level tagging: {pct:.2f}%. "
              f"Quote this in the report.")
        if bugs:
            print(f"[check] *** {bugs} span(s) were token-aligned and still lost. "
                  f"That is a defect in our encoding, not the tokenizer. Fix it before "
                  f"training -- likely an overlapping gold pair or a chunk-ownership "
                  f"error.")

    print("\n[check] building the backbone...")
    bb = load_backbone(cfg.backbone, cfg.ckpt, device="cpu")
    print(f"[check] ok. max_len will be clamped to {min(cfg.max_len, bb.max_positions)}")


if __name__ == "__main__":
    main()

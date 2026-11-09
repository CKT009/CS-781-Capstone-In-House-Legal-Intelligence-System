import argparse, json, os, sys
from pathlib import Path

B_GROWN = "../checkpoints/B_grown"
B_ENC   = "../checkpoints/B_encoder"
A_GROWN = "../checkpoints/A_grown"
M_MAMBA = "../checkpoints/M_mamba"

EXPERIMENTS = {

    "full":      dict(peft="full", lr=3e-5,
                      _why="full fine-tuning -- the baseline every other method is judged against"),
    "head_only": dict(peft="head_only", lr=3e-5, head_lr=3e-3,
                      _why="linear probe on frozen features: the floor, and it measures how "
                           "much of the task the document transformer alone can do"),
    "bitfit":    dict(peft="bitfit", lr=1e-3,
                      _why="BitFit: biases and norm gains only, no weight matrix moves"),
    "lora":      dict(peft="lora", lr=3e-4, peft_kw=dict(r=8, alpha=16),
                      _why="LoRA on the attention projections -- low-rank reparameterisation"),
    "qlora":     dict(peft="qlora", lr=3e-4, peft_kw=dict(r=8, alpha=16),
                      _why="the same rank-8 adapter over a 4-bit NF4 base; paired with lora "
                           "it isolates the cost of quantising the frozen weights"),
    "rslora":    dict(peft="rslora", lr=3e-4, peft_kw=dict(r=8, alpha=16),
                      _why="rank-stabilised LoRA: alpha/sqrt(r) scaling. Paired with lora "
                           "at the same rank it isolates the scaling rule alone"),
    "ia3":       dict(peft="ia3", lr=3e-3,
                      _why="IA^3: a learned scale per output unit on k/v and the FFN "
                           "intermediate. The cheapest method here -- no matrices at all"),
    "adapter":   dict(peft="adapter", lr=3e-4, peft_kw=dict(bottleneck=64),
                      _why="Houlsby/Pfeiffer bottleneck adapters -- added modules"),
    "last4":     dict(peft="last_k", lr=5e-5, peft_kw=dict(k=4),
                      _why="partial/layer-wise: only the top 4 blocks adapt"),
    "prefix":    dict(peft="prefix", lr=1e-3, peft_kw=dict(n_prefix=16),
                      _why="prefix tuning: learned key/value vectors, the backbone untouched"),
    "prompt":    dict(peft="prompt", lr=1e-3, peft_kw=dict(n_prompt=20),
                      _why="prompt tuning: learned virtual tokens at the input embedding"),


    "no_doc":    dict(peft="lora", lr=3e-4, doc_layers=0, peft_kw=dict(r=8, alpha=16),
                      _why="ABLATION: no document transformer. Sentences classified without "
                           "knowing their neighbours -- measures what document context is worth"),
    "no_crf":    dict(peft="lora", lr=3e-4, use_crf=False, peft_kw=dict(r=8, alpha=16),
                      _why="ABLATION: no CRF. Measures what modelling label TRANSITIONS is "
                           "worth, separately from document context"),
    "doc4":      dict(peft="lora", lr=3e-4, doc_layers=4, peft_kw=dict(r=8, alpha=16),
                      _why="a deeper document transformer"),


    "lora_Benc": dict(peft="lora", backbone=B_ENC, lr=3e-4, peft_kw=dict(r=8, alpha=16),
                      _why="the un-upgraded encoder: does the Phase-2 backbone gain carry here?"),
    "lora_A":    dict(peft="lora", backbone=A_GROWN, lr=3e-4, peft_kw=dict(r=8, alpha=16),
                      _why="causal decoder. A sentence vector is its LAST token state; "
                           "expected to trail the bidirectional encoder"),
    "lora_M":    dict(peft="lora", backbone=M_MAMBA, lr=3e-4,
                      peft_kw=dict(r=8, alpha=16, targets=("in_proj", "out_proj")),
                      _why="Mamba: no attention, so LoRA targets the fused in/out projections"),


    "bitfit_doc":  dict(peft="bitfit", lr=1e-3, train_doc=True, doc_lr=3e-4, epochs=40,
                        _why="bitfit40 with the document transformer actually trained"),
    "bitfit_doc_hi": dict(peft="bitfit", lr=1e-3, train_doc=True, doc_lr=1e-3, epochs=40,
                        _why="as bitfit_doc, doc encoder at head_lr (what the optimizer intended)"),
    "lora_doc":    dict(peft="lora", lr=3e-4, train_doc=True, doc_lr=3e-4, epochs=40,
                        peft_kw=dict(r=8, alpha=16),
                        _why="LoRA with the document transformer actually trained"),


    "bitfit_ctx":  dict(peft="bitfit", lr=1e-3, epochs=40, ctx_tokens=512,
                        _why="bitfit40, but each sentence is encoded inside ~512 tokens of "
                             "running text rather than alone"),
    "bitfit_ctx1k": dict(peft="bitfit", lr=1e-3, epochs=40, ctx_tokens=1024,
                        _why="as bitfit_ctx with the backbone's full pretraining width"),
    "lora_ctx":    dict(peft="lora", lr=3e-4, epochs=40, ctx_tokens=512,
                        peft_kw=dict(r=8, alpha=16),
                        _why="LoRA with in-context sentence encoding"),

    "distill":   dict(peft="full", lr=3e-5,
                      _why="MANDATORY: distil the best teacher into a thinner student. "
                           "See rr/distill.py -- run_train.py does not drive this"),
}


def check(cfg):
    from rr import data as D
    from rr.backbones import load_backbone
    rep = D.check(cfg.data_dir)
    vocab = rep["labels"]
    print(f"[check] {len(vocab)} labels: {vocab}")
    print("[check] building the backbone...")
    bb = load_backbone(cfg.backbone, cfg.ckpt, device="cpu")
    print(f"[check] ok. d_model={bb.d_model} bidirectional={bb.bidirectional} "
          f"max_positions={bb.max_positions}")
    print(f"[check] sentences truncated at {cfg.max_sent_len} tokens; documents windowed "
          f"at {cfg.max_doc_sents} sentences with stride {cfg.doc_stride}")
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--name")
    ap.add_argument("--backbone")
    ap.add_argument("--set", nargs="*", default=[], metavar="K=V")
    a = ap.parse_args()

    if a.list:
        print(f"{'name':12s} {'peft':10s} {'backbone':12s}  why")
        for k, v in EXPERIMENTS.items():
            bb = Path(v.get("backbone", B_GROWN)).name
            print(f"{k:12s} {v['peft']:10s} {bb:12s}  {v['_why']}")
        print(f"\n{len(EXPERIMENTS)} experiments; the phase requires >=6 submissions and "
              f">=5 distinct techniques, one using distillation.")
        return

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    sys.path.insert(0, str(Path(__file__).parent))
    from rr.train import TrainConfig, train

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
    print(f"  method   {cfg.peft} {cfg.peft_kw}  crf={cfg.use_crf} doc_layers={cfg.doc_layers}")
    print(f"  data     domains={list(cfg.domains)}  train/dev are the GIVEN splits")
    if cfg.final_fit:
        print(f"  data     FINAL FIT: train+dev, {cfg.epochs} fixed epochs")
        print(f"  select   NOTHING -- last.pt is the model; dev scores are contaminated")
    else:
        print(f"  select   on dev MACRO-F1, never on loss")
    print(f"{'=' * 72}\n")
    train(cfg)


if __name__ == "__main__":
    main()

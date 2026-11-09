from __future__ import annotations

import argparse
import ast
import os

LORA_ALL = ("q_proj", "k_proj", "v_proj", "o_proj")

EXPERIMENTS = {

    "full_k4":      dict(peft="full", lr=2e-5,
                         _why="full fine-tuning: the reference, and the KD teacher"),
    "lastk_k4":     dict(peft="last_k", peft_kw=dict(k=6), lr=5e-5,
                         _why="partial: only the top 6 of 18 blocks train"),
    "gradual_k4":   dict(peft="gradual", gradual_stages=4, lr=3e-5,
                         _why="head only, then top blocks unfrozen in 4 stages"),
    "lora_k4":      dict(peft="lora", peft_kw=dict(r=16, alpha=32, targets=LORA_ALL), lr=1e-4,
                         _why="LoRA r=16 on q/k/v/o (Phase 4: 3e-4 collapsed, 1e-4 worked)"),
    "adapter_k4":   dict(peft="adapter", peft_kw=dict(bottleneck=64), lr=3e-4,
                         _why="bottleneck adapters (64) after every block"),
    "adalora_k4":   dict(peft="adalora", peft_kw=dict(r=12, target_r=6), lr=1e-4, e_lr_mult=20,
                         _why="AdaLoRA with E at 20x lr (Phase 4: without it E never grew)"),

    "kd_half":      dict(peft="distill", lr=3e-5, peft_kw=dict(teacher_run="runs/full_k4",
                         keep_every=2, alpha=0.5, tau=2.0),
                         _why="KD: 9-block student, teacher logits + gold"),
    "kd_control":   dict(peft="distill", lr=3e-5, peft_kw=dict(teacher_run="runs/full_k4",
                         keep_every=2, alpha=1.0),
                         _why="KD control: the same student, gold only"),

    "full_k1":      dict(peft="full", lr=2e-5, K=1, keep="head", pool="mean",
                         _why="first 1,024 tokens only (57% of test docs fit whole)"),
    "full_k8":      dict(peft="full", lr=2e-5, K=8, batch_docs=4, grad_accum=4,
                         _why="up to 8 windows (98% of docs read whole)"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp")
    ap.add_argument("--name")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--fold", type=int, default=-1,
                    help="k-fold over train+dev: train without fold k (0-4) -> runs/<exp>_f<k>")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.list or not a.exp:
        for k, v in EXPERIMENTS.items():
            print(f"  {k:12s} {v['_why']}")
        return
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    from lsi.train import TrainConfig, train
    kw = {k: v for k, v in EXPERIMENTS[a.exp].items() if not k.startswith("_")}
    kw["name"] = a.name or a.exp
    if a.fold >= 0:
        kw.update(fold=a.fold, name=f"{kw['name']}_f{a.fold}")
        kw.setdefault("eval_every", 2700)
    for s in a.set:
        k, v = s.split("=", 1)
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            pass
        kw[k] = v
    if a.smoke:
        kw.update(name=kw["name"] + "_smoke", max_train=300, epochs=1.0, eval_every=10**9)
    train(TrainConfig(**kw))


if __name__ == "__main__":
    main()

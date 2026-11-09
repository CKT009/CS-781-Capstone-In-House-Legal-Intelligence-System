from __future__ import annotations

import argparse
import ast
import os

BACKBONE = "../checkpoints/B_grown"

EXPERIMENTS = {

    "full_tail":    dict(peft="full", lr=2e-5,
                         _why="full fine-tuning on the last 1,024 tokens: the reference"),
    "lastk_tail":   dict(peft="last_k", peft_kw=dict(k=6), lr=5e-5,
                         _why="partial / layer-wise: only the top 6 of 18 blocks train"),
    "gradual_tail": dict(peft="gradual", gradual_stages=4, lr=3e-5,
                         _why="freezing/unfreezing: head only, then top blocks unfrozen "
                              "in 4 stages (ULMFiT-style)"),
    "lora_tail":    dict(peft="lora", peft_kw=dict(r=16, alpha=32,
                         targets=("q_proj", "k_proj", "v_proj", "o_proj")), lr=3e-4,
                         _why="LoRA r=16 on all attention projections"),
    "adapter_tail": dict(peft="adapter", peft_kw=dict(bottleneck=64), lr=3e-4,
                         _why="bottleneck adapters (64) after every block"),
    "adalora_tail": dict(peft="adalora", peft_kw=dict(r=12, target_r=6), lr=3e-4,
                         _why="AdaLoRA: SVD-form LoRA with importance-based rank pruning"),
    "adalora_e":    dict(peft="adalora", peft_kw=dict(r=12, target_r=6), lr=1e-4, e_lr_mult=20,
                         _why="AdaLoRA with E at 20x lr, no decay (adalora_tail's E never grew)"),

    "kd_half":      dict(peft="distill", lr=3e-5, peft_kw=dict(teacher_run="runs/full_tail",
                         keep_every=2, alpha=0.5, tau=2.0),
                         _why="KD: half-depth student, teacher logits + gold"),
    "kd_control":   dict(peft="distill", lr=3e-5, peft_kw=dict(teacher_run="runs/full_tail",
                         keep_every=2, alpha=1.0),
                         _why="KD control: the same student, gold only"),

    "full_head":    dict(peft="full", lr=2e-5, keep="head",
                         _why="first 1,024 tokens only (opening: parties, facts)"),
    "full_k4_attn": dict(peft="full", lr=2e-5, K=4, pool="attn", batch_docs=2, grad_accum=8,
                         _why="hierarchical: last 4 chunks, attention-pooled"),
    "adapter_k4_attn": dict(peft="adapter", peft_kw=dict(bottleneck=64), lr=3e-4, K=4,
                         pool="attn", batch_docs=2, grad_accum=8,
                         _why="hierarchical with adapters (best single-window method)"),
    "full_k8_attn": dict(peft="full", lr=2e-5, K=8, pool="attn", batch_docs=2, grad_accum=8,
                         _why="hierarchical: last 8 chunks (whole doc for 84% of test cases)"),
    "full_k16_attn": dict(peft="full", lr=2e-5, K=16, pool="attn", batch_docs=1, grad_accum=16,
                         _why="hierarchical: last 16 chunks (whole doc for 97% of test cases)"),

    "k8_redact":    dict(peft="full", lr=2e-5, K=8, pool="attn", batch_docs=2, grad_accum=8,
                         redact_train=True,
                         _why="full_k8_attn with decision wording cut from train"),
    "k8_era":       dict(peft="full", lr=2e-5, K=8, pool="attn", batch_docs=2, grad_accum=8,
                         era_weight=True,
                         _why="full_k8_attn with train docs weighted towards dev/test decades"),
    "k8_redact_era": dict(peft="full", lr=2e-5, K=8, pool="attn", batch_docs=2, grad_accum=8,
                         redact_train=True, era_weight=True,
                         _why="both"),
    "adapter_k8_attn": dict(peft="adapter", peft_kw=dict(bottleneck=64), lr=3e-4, K=8,
                         pool="attn", batch_docs=2, grad_accum=8,
                         _why="hierarchical: adapters over the last 8 chunks"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp")
    ap.add_argument("--name")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.list or not a.exp:
        for k, v in EXPERIMENTS.items():
            print(f"  {k:14s} {v['_why']}")
        return
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    from cjpe.train import TrainConfig, train
    kw = {k: v for k, v in EXPERIMENTS[a.exp].items() if not k.startswith("_")}
    kw["name"] = a.name or a.exp
    for s in a.set:
        k, v = s.split("=", 1)
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            pass
        kw[k] = v
    if a.smoke:
        kw.update(name=kw["name"] + "_smoke", max_train=200, epochs=1.0, eval_every=10**9)
    train(TrainConfig(**kw))


if __name__ == "__main__":
    main()

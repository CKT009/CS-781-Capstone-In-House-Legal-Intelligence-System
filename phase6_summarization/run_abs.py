from __future__ import annotations

import argparse
import ast
import os

ATT = ("q_proj", "k_proj", "v_proj", "o_proj")
EXPERIMENTS = {

    "houlsby":  dict(method="houlsby", kw=dict(bottleneck=64), lr=5e-4,
                     _why="Houlsby adapters: after attention and after MLP in every block"),
    "pfeiffer": dict(method="pfeiffer", kw=dict(bottleneck=128), lr=5e-4,
                     _why="Pfeiffer adapters: one per block, after the MLP"),
    "lora":     dict(method="lora", kw=dict(r=16, alpha=32, targets=ATT), lr=2e-4,
                     _why="LoRA r=16 on q/k/v/o"),
    "qlora":    dict(method="qlora", kw=dict(r=16, alpha=32, targets=ATT), lr=2e-4,
                     _why="QLoRA: NF4-quantised frozen base + LoRA r=16"),
    "adalora":  dict(method="adalora", kw=dict(r=12, target_r=6), lr=2e-4, e_lr_mult=20,
                     _why="AdaLoRA r=12 -> 6 budget; E at 20x lr (Phase 4 lesson)"),
    "lorma":    dict(method="lorma", kw=dict(r=16, alpha=32), lr=2e-4,
                     _why="LoRMA: multiplicative low-rank (I + BA)W on q/k/v/o"),
    "prefix":   dict(method="prefix", kw=dict(n_prefix=16, hidden=512), lr=1e-3,
                     _why="prefix tuning: 16 learned KV positions per layer, MLP-reparameterised"),
    "prompt":   dict(method="prompt", kw=dict(n_prompt=20), lr=3e-2,
                     _why="prompt tuning: 20 learned input embeddings"),

    "lora_lead":  dict(method="lora", kw=dict(r=16, alpha=32, targets=ATT), lr=2e-4,
                       builder="lead", _why="LoRA, input = header-stripped lead (control)"),
    "lora_chunk": dict(method="lora", kw=dict(r=16, alpha=32, targets=ATT), lr=2e-4,
                       builder="chunk", _why="LoRA, input = top sentences from 4 equal chunks"),

    "full":     dict(method="full", lr=3e-5, weight_decay=0.01, grad_ckpt=True,
                     _why="full fine-tune: the KD TEACHER (not one of the listed techniques)"),
    "kd":       dict(method="student", teacher="runs/full", keep_every=2, kd_alpha=0.5,
                     kd_tau=2.0, lr=5e-5, weight_decay=0.01,
                     _why="KD: 10-block student from the teacher, gold CE + teacher logits"),
    "kd_control": dict(method="student", teacher="runs/full", keep_every=2, kd_alpha=1.0,
                       lr=5e-5, weight_decay=0.01,
                       _why="KD control: the same student, gold only"),
}


def _kv(items):
    out = {}
    for s in items or []:
        k, v = s.split("=", 1)
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            pass
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp"); ap.add_argument("--name"); ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--set", nargs="*", help="Cfg overrides key=value")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--finish", help="re-evaluate a trained run dir (e.g. new decoding)")
    ap.add_argument("--gen", nargs="*", help="decoding overrides for --finish")
    ap.add_argument("--tag", default="")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list or not (a.exp or a.finish):
        for k, v in EXPERIMENTS.items():
            print(f"  {k:11s} {v['_why']}")
        return
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    from summ.abs_train import Cfg, finish, load_trained, train
    if a.finish:
        import json
        from pathlib import Path
        c = json.loads((Path(a.finish) / "config.json").read_text())
        finish(a.finish, gen={**c["gen"], **_kv(a.gen)}, tag=a.tag)
        return
    kw = {k: v for k, v in EXPERIMENTS[a.exp].items() if not k.startswith("_")}
    kw["name"] = a.name or a.exp
    kw.update(_kv(a.set))
    if a.smoke:
        kw.update(name=kw["name"] + "_smoke", max_train=200, epochs=1.0, sel_docs=8, eval_docs=16)
    train(Cfg(**kw))


if __name__ == "__main__":
    main()

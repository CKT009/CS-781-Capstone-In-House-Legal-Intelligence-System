from __future__ import annotations

import argparse
import ast
import os

QKVO = ("q_proj", "k_proj", "v_proj", "o_proj")

EXPERIMENTS = {

    "full":    dict(peft="full", lr=2e-5,
                    _why="full fine-tuning: the reference, and the KD teacher"),
    "partial": dict(peft="last_k", peft_kw=dict(k=6), lr=5e-5,
                    _why="partial / layer-wise: only the top 6 of 18 blocks (+ final norm)"),
    "lora":    dict(peft="lora", peft_kw=dict(r=16, alpha=32, targets=QKVO), lr=1e-4,
                    _why="LoRA r=16 on q/k/v/o (Phase 4/5: 1e-4 works, 3e-4 collapsed)"),
    "qlora":   dict(peft="qlora", peft_kw=dict(r=16, alpha=32, targets=QKVO), lr=1e-4,
                    _why="QLoRA: same adapters as `lora` over an NF4-quantised frozen base"),
    "adapter": dict(peft="adapter", peft_kw=dict(bottleneck=64), lr=3e-4,
                    _why="bottleneck adapters (64) after every block"),
    "prefix":  dict(peft="prefix", peft_kw=dict(n_prefix=16), lr=3e-3,
                    _why="prefix tuning: 16 learned K/V per layer, backbone frozen"),
    "prompt":  dict(peft="prompt", peft_kw=dict(n_prompt=20), lr=3e-3,
                    _why="prompt tuning: 20 soft input tokens (text window 1004)"),
    "ia3":     dict(peft="ia3", lr=3e-3,
                    _why="IA3: learned rescaling of k, v and the FFN up-projection"),


    "kd_student": dict(peft="distill", lr=5e-5, epochs=5.0,
                       peft_kw=dict(teacher_run="runs/full", keep_every=2, alpha=0.5, tau=1.0,
                                    student_init="phase1"),
                       _why="KD: 9-block Phase-1 student, KL to teacher similarity "
                            "distribution + gold InfoNCE"),
    "kd_control": dict(peft="distill", lr=5e-5, epochs=5.0,
                       peft_kw=dict(teacher_run="runs/full", keep_every=2, alpha=1.0,
                                    student_init="phase1"),
                       _why="KD control: the same student, gold InfoNCE only"),


    "kd_student_tinit": dict(peft="distill", lr=3e-5,
                             peft_kw=dict(teacher_run="runs/full", keep_every=2, alpha=0.5,
                                          tau=1.0, student_init="teacher"),
                             _why="KD, student initialised from the teacher's blocks"),
    "kd_control_tinit": dict(peft="distill", lr=3e-5,
                             peft_kw=dict(teacher_run="runs/full", keep_every=2, alpha=1.0,
                                          student_init="teacher"),
                             _why="KD control for the teacher-initialised student"),
}

EXPERIMENTS["kd_teacher"] = dict(EXPERIMENTS["full"], _why="alias of `full` (the KD teacher)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp")
    ap.add_argument("--name")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--zeroshot", action="store_true")
    a = ap.parse_args()
    if a.list or not (a.exp or a.zeroshot):
        for k, v in EXPERIMENTS.items():
            print(f"  {k:11s} {v['_why']}")
        return
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("RAYON_NUM_THREADS", "8")
    if a.zeroshot:
        from pcr.train import zeroshot
        zeroshot()
        return
    from pcr.train import TrainConfig, train
    kw = {k: v for k, v in EXPERIMENTS[a.exp].items() if not k.startswith("_")}
    kw["peft_kw"] = dict(kw.get("peft_kw", {}))
    kw["name"] = a.name or a.exp
    if a.smoke:


        kw.update(name=a.name or kw["name"] + "_smoke", max_train_q=50, max_steps=3,
                  K_eval=1, eval_at_start=False)
        if kw["peft"] == "distill":
            kw["peft_kw"]["teacher_run"] = "runs/full_smoke"
    for s in a.set:
        k, v = s.split("=", 1)
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            pass
        if k.startswith("peft_kw."):
            kw["peft_kw"][k.split(".", 1)[1]] = v
        else:
            kw[k] = v
    train(TrainConfig(**kw))


if __name__ == "__main__":
    main()

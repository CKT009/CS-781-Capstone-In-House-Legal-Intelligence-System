from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

B_ENC = "../checkpoints/B_encoder"
A_DEC = "../checkpoints/A_decoder"
M_MAM = "../checkpoints/M_mamba"


PLAN = {
    "adapter_B_f2": ("adapter", B_ENC, 12, 0.3738,
                     "TECHNIQUE 4/5: bottleneck adapters -- added modules. Missing from subs/"),
    "last4_B_f2":   ("last4",   B_ENC,  7, 0.3126,
                     "TECHNIQUE 5/5: partial/layer-wise, top 4 blocks. Missing from subs/"),
    "adapter_A_f2": ("adapter", A_DEC, 10, 0.3435,
                     "ensemble member -- adapter on the causal decoder"),
    "adapter_M_f2": ("adapter", M_MAM, 13, 0.3385,
                     "ensemble member -- adapter on Mamba"),
}
DISTILL = dict(name="distill_f2", teacher="runs/full_B_both", epochs=13, honest=0.3812)


ENSEMBLE = ["full_B_both", "adapter_B_f2", "adapter_A_f2", "adapter_M_f2", "lora_B_both"]
VOTE = 3

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--list", action="store_true")
ap.add_argument("--only", default=None, help="one of: " + ", ".join(list(PLAN) + ["distill", "ensemble"]))
ap.add_argument("--folds", default='["fold1","fold2"]')
a = ap.parse_args()

PY = sys.executable


def sh(cmd):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode:
        sys.exit(f"FAILED ({r.returncode}): {' '.join(cmd)}")


if a.list:
    print(f"\ntrain on {a.folds}, fixed epochs, submission from last.pt\n")
    print(f"{'run':16s}{'exp':10s}{'backbone':12s}{'ep':>4s}{'honestF1':>10s}  why")
    print("-" * 100)
    for n, (e, b, ep, h, why) in PLAN.items():
        print(f"{n:16s}{e:10s}{Path(b).name:12s}{ep:4d}{h:10.4f}  {why}")
    print(f"{DISTILL['name']+'_kd':16s}{'distill':10s}{'B_encoder':12s}"
          f"{DISTILL['epochs']:4d}{DISTILL['honest']:10.4f}  MANDATORY: knowledge distillation")
    print(f"\nthen: ensemble.py --subs {' '.join(ENSEMBLE)} --vote {VOTE} --name ens_best")
    print("\nFINAL SUBMISSION SET (7 zips, 6 techniques + ensemble):")
    for z, t, f1 in [("ens_best", "span-vote ensemble", 0.4877),
                     ("full_B_both", "full fine-tuning", 0.4088),
                     ("adapter_B_f2", "adapters", 0.3738),
                     ("distill_f2_kd", "knowledge distillation (REQUIRED)", 0.3812),
                     ("lora_B_both", "LoRA", 0.3439),
                     ("bitfit_B_both", "BitFit", 0.3156),
                     ("last4_B_f2", "last-4 layers", 0.3126)]:
        print(f"    subs/{z + '.zip':22s} {t:36s} honest dev {f1:.4f}")
    print("\n  NOT ens2.zip (0.3911, below a single model) and NOT final.zip "
          "(LoRA at epoch 37; its honest optimum is 14).")
    sys.exit(0)

todo = [a.only] if a.only else list(PLAN) + ["distill", "ensemble"]

for name in todo:
    if name in PLAN:
        exp, backbone, epochs, honest, why = PLAN[name]
        print(f"\n{'=' * 78}\n  {name}   {why}\n  honest dev F1 of its fold1-trained twin: "
              f"{honest:.4f}  (the dev score THIS run prints is contaminated)\n{'=' * 78}")
        sh([PY, "run_train.py", "--exp", exp, "--name", name, "--gpu", str(a.gpu),
            "--backbone", backbone, "--set", f"epochs={epochs}", f"train_folds={a.folds}"])
        sh([PY, "run_predict.py", "--run", f"runs/{name}", "--ckpt", "last.pt",
            "--gpu", str(a.gpu)])

    elif name == "distill":
        print(f"\n{'=' * 78}\n  {DISTILL['name']}   MANDATORY knowledge-distillation submission"
              f"\n  teacher {DISTILL['teacher']}; trains the KD student AND a gold-only control,"
              f"\n  so the claim is the GAP, not the student's score alone.\n{'=' * 78}")
        code = (
            "import json,sys; sys.path.insert(0,'.')\n"
            "from ner.distill import distill, DistillConfig\n"
            f"distill(DistillConfig(name='{DISTILL['name']}', teacher_run='{DISTILL['teacher']}',\n"
            f"    epochs={DISTILL['epochs']}, train_folds=('fold1','fold2'), dev_folds=('fold2',),\n"
            "    control=True))\n")
        sh([PY, "-c", code])
        sh([PY, "run_predict.py", "--run", f"runs/{DISTILL['name']}_kd", "--ckpt", "last.pt",
            "--gpu", str(a.gpu)])

    elif name == "ensemble":
        missing = [m for m in ENSEMBLE if not Path("subs", f"{m}.zip").exists()]
        if missing:
            sys.exit(f"cannot build the ensemble, missing submissions: {missing}")
        print(f"\n{'=' * 78}\n  ensemble: vote >={VOTE} of {len(ENSEMBLE)}"
              f"\n  honest dev estimate 0.4877 (vs 0.4560 for ens5, 0.4088 best single)"
              f"\n{'=' * 78}")
        sh([PY, "ensemble.py", "--subs", *ENSEMBLE, "--vote", str(VOTE), "--name", "ens_best"])

print("""
================================================================================
  SUBMIT, in this order (Codabench allows 10/day):

    1  subs/ens_best.zip        span-vote ensemble            honest 0.4877
    2  subs/full_B_both.zip     full fine-tuning              honest 0.4088
    3  subs/adapter_B_f2.zip    adapters                      honest 0.3738
    4  subs/distill_f2_kd.zip   knowledge distillation  <-- REQUIRED
    5  subs/lora_B_both.zip     LoRA                          honest 0.3439
    6  subs/bitfit_B_both.zip   BitFit                        honest 0.3156
    7  subs/last4_B_f2.zip      last-4-layer tuning           honest 0.3126

  Six distinct techniques plus the ensemble, which satisfies ">=6 submissions,
  >=5 distinct techniques, one distillation".

  Do NOT submit ens2.zip or final.zip -- see this file's docstring.
================================================================================
""")

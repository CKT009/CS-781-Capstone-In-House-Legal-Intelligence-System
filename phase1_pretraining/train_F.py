import argparse
import os

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--gpu", type=int, default=0, help="physical card index")
ap.add_argument("--probe", action="store_true",
                help="2,000 steps into a throwaway run dir: checks the loss descends "
                     "and measures throughput before you commit ~20 GPU-hours")
ap.add_argument("--steps", type=int, default=40000)
ap.add_argument("--minutes", type=float, default=0.0, help="0 = no wall-clock limit")
ap.add_argument("--name", default="F_diffusion")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--target-len", type=int, default=256,
                help="tokens diffused per sequence; block_size minus this is the clean "
                     "prompt the encoder reads. IN SHAPE_KEYS -- changing it blocks "
                     "resume, so decide before step 1.")
ap.add_argument("--weighting", default="nelbo", choices=["nelbo", "uniform"],
                help="nelbo: loss is an upper bound on per-token NLL, so it is in the "
                     "same units as A's 1.5343 and belongs in Table 21. uniform: lower "
                     "variance, but NOT comparable to A -- give it its own column.")
ap.add_argument("--lr", type=float, default=3e-4)
a = ap.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

from config import Config, PRESETS
from train import train

cfg = Config(**{
    **PRESETS["F"],
    "name": f"{a.name}_probe" if a.probe else a.name,
    "shard_dir": "artifacts/shards",
    "tokenizer_dir": "artifacts/tokenizer_16k",
    "out_dir": "runs",


    "block_size": 1024,
    "batch_size": a.batch,
    "grad_accum": 1,
    "max_steps": 2000 if a.probe else a.steps,
    "warmup": 500,
    "schedule": "wsd",
    "decay_frac": 0.2,
    "lr": a.lr,
    "amp": "bf16",
    "grad_checkpoint": True,


    "diff_target_len": a.target_len,
    "diff_weighting": a.weighting,
    "diff_cond_dim": 256,
    "diff_steps": 64,


    "eval_every": 250 if a.probe else 500,
    "eval_batches": 20,
    "save_every_min": 20,
    "log_every": 20,
    "max_minutes": a.minutes,
})

FERTILITY = 1.289
prompt_tok = cfg.block_size - cfg.diff_target_len
supervised = 0.5 * cfg.diff_target_len * cfg.batch_size

print(f"""
{'=' * 72}
  model F -- masked-diffusion encoder-decoder     ->  runs/{cfg.name}
{'=' * 72}
  card              cuda:{a.gpu} (physical)
  parameters        {cfg.n_params() / 1e6:.1f}M        (A 100.4M, M 100.3M; band 50-300M)
  tokens/step       {cfg.tokens_per_step:,}       matched to A, B, C, D, M
  total tokens      {cfg.tokens_per_step * cfg.max_steps / 1e9:.3f}B

  prompt            {prompt_tok} tok  ~= {prompt_tok / FERTILITY:.0f} words   (test prompt is 512)
  target            {cfg.diff_target_len} tok  ~= {cfg.diff_target_len / FERTILITY:.0f} words   (need 128)
  supervised/step   {supervised:,.0f}        <- only masked positions carry loss;
                                  A supervises all {cfg.tokens_per_step:,}
  supervised total  {supervised * cfg.max_steps / 1e6:.0f}M       vs A's 2,621M (Section 6.1 is
                                  the same arithmetic for C)

  loss              {cfg.diff_weighting}{'  (comparable to A)' if cfg.diff_weighting == 'nelbo' else '  (NOT comparable to A)'}
  resume            re-run this exact command
  stop early        touch runs/{cfg.name}/STOP
{'=' * 72}
""")

if a.probe:
    print("PROBE: 2,000 steps into a separate run dir.\n"
          "  Watch for  (1) val_loss falling below ln(16000) = 9.68 within a few hundred\n"
          "                 steps -- above that it has not learned the unigram distribution\n"
          "             (2) tokens_per_s in runs/%s/summary.json -- multiply out to check\n"
          "                 40,000 steps fits the window before 6 Oct 23:59\n" % cfg.name)

train(cfg.check())

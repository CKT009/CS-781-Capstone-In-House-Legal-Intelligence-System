import json, shutil, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import Config, PRESETS

TMP = Path("smoke_tmp")


def fake_shards(d, vocab=1000, n_tokens=200_000):
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    shards = []
    for i in range(2):
        arr = rng.integers(10, vocab, size=n_tokens // 2, dtype=np.uint16)
        arr.tofile(d / f"acts_{i:04d}.bin")
        shards.append({"file": f"acts_{i:04d}.bin", "source": "acts", "n_tokens": int(arr.size)})
    (d / "index.json").write_text(json.dumps({"shards": shards}))
    (d / "index_val.json").write_text(json.dumps({"shards": shards[:1]}))


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    fake_shards(TMP / "shards")
    ok = []

    for arch in ["decoder", "encoder", "encdec", "efficient"]:
        cfg = Config(**{**PRESETS["tiny"], "name": f"smoke_{arch}", "arch": arch},
                     shard_dir=str(TMP / "shards"), out_dir=str(TMP / "runs"),
                     max_steps=3, warmup=1, log_every=1, eval_every=2, eval_batches=2,
                     device="cpu", amp="off", grad_checkpoint=False, save_every_min=0)
        from train import train
        s = train(cfg)
        ok.append((arch, s["steps"] == 3))
        print(f"  --> {arch}: {s['steps']} steps, loss logged\n")


    cfg = Config(**{**PRESETS["tiny"], "name": "smoke_decoder"}, shard_dir=str(TMP / "shards"),
                 out_dir=str(TMP / "runs"), max_steps=6, warmup=1, log_every=1, eval_every=0,
                 device="cpu", amp="off", grad_checkpoint=False, save_every_min=0)
    from train import train
    s = train(cfg)
    log = [json.loads(l) for l in
           (TMP / "runs" / "smoke_decoder" / "log.jsonl").read_text().splitlines() if l.strip()]
    steps_after_resume = [r["step"] for r in log if "train_loss" in r]
    ok.append(("resume reaches step 6", s["steps"] == 6))

    ok.append(("resume did not restart at 0", steps_after_resume.count(1) == 1))


    import torch
    from models import build
    from generate import generate, truncate_words
    from predict import write_submission
    m = build(Config(**PRESETS["tiny"]))
    out = generate(m, torch.randint(10, 900, (2, 16)), max_new_tokens=12, eos_id=3)
    ok.append(("generate", tuple(out.shape) == (2, 12)))
    preds = {"1": {"text": truncate_words("word " * 300, 128)}}
    z = write_submission(preds, TMP / "sub", prompts={"1": "p"})
    import zipfile
    ok.append(("submission", zipfile.ZipFile(z).namelist() == ["prediction.json"]))
    ok.append(("128 words", len(preds["1"]["text"].split()) == 128))

    print("\n" + "=" * 50)
    for name, passed in ok:
        print(f"  [{'x' if passed else ' '}] {name}")
    bad = [n for n, p in ok if not p]
    print("=" * 50)
    print("SMOKE TEST PASSED" if not bad else f"FAILED: {bad}")
    shutil.rmtree(TMP, ignore_errors=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

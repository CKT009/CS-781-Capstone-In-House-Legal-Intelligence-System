from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backbone import Config, build

RUNS = ROOT.parent / "checkpoints"
TOKENIZER = ROOT / "artifacts" / "tokenizer_16k" / "tokenizer.json"
PAD, UNK, BOS, EOS, MASK, SEP = 0, 1, 2, 3, 4, 5


def load_lm(name: str = "A_grown", ckpt: str = "best.pt", device: str = "cpu"):
    run = RUNS / name
    cfg = Config.load(run / "config.json")
    assert cfg.arch == "decoder", f"{name} is {cfg.arch}; summarisation needs the decoder"
    sd = torch.load(run / ckpt, map_location="cpu", weights_only=False)["model"]
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}
    model = build(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"{len(missing)} tensors missing from {run / ckpt}: {missing[:5]}")
    return model.to(device)


_TOK = None


def tokenizer():
    global _TOK
    if _TOK is None:
        from tokenizers import Tokenizer
        _TOK = Tokenizer.from_file(str(TOKENIZER))
    return _TOK

from __future__ import annotations


import json
from pathlib import Path

import torch
import torch.nn as nn

from backbone import Config, build


_SPEC = {
    "decoder":   dict(attr="stack",   bidirectional=False, honours_pad_mask=True),
    "encoder":   dict(attr="stack",   bidirectional=True,  honours_pad_mask=True),
    "diffusion": dict(attr="encoder", bidirectional=True,  honours_pad_mask=True),
    "mamba":     dict(attr=None,      bidirectional=False, honours_pad_mask=False),
}


class Backbone(nn.Module):

    def __init__(self, inner: nn.Module, cfg: Config, max_positions: int):
        super().__init__()


        if cfg.arch != "mamba" and not hasattr(inner, "blocks"):
            raise TypeError(
                f"Backbone expects the inner Stack for arch={cfg.arch!r}, got "
                f"{type(inner).__name__}. Use load_backbone(), or pass model.stack "
                f"(decoder/encoder) or model.encoder (diffusion).")
        self.inner = inner
        self.cfg = cfg
        self.arch = cfg.arch
        self.d_model = cfg.d_model
        self.max_positions = max_positions
        spec = _SPEC[cfg.arch]
        self.bidirectional = spec["bidirectional"]
        self.honours_pad_mask = spec["honours_pad_mask"]

    def forward(self, input_ids, attention_mask=None):
        if self.arch == "mamba":


            return self.inner(ids=input_ids, return_hidden=True)["hidden"]
        return self.inner(input_ids, pad_mask=attention_mask)[0]

    def gradient_checkpointing(self, enable: bool):
        self.cfg.grad_checkpoint = enable
        if hasattr(self.inner, "cfg"):
            self.inner.cfg.grad_checkpoint = enable


def _strip_prefixes(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        while True:
            for p in ("_orig_mod.", "module."):
                if k.startswith(p):
                    k = k[len(p):]
                    break
            else:
                break
        out[k] = v
    return out


def load_backbone(run_dir, ckpt: str = "best.pt", device: str = "cpu",
                  state_cache: dict | None = None) -> Backbone:
    run = Path(run_dir)
    cfg = Config.load(run / "config.json")

    p = Path(ckpt)
    if not p.exists():
        p = run / ckpt
    if not p.exists():
        raise FileNotFoundError(
            f"no checkpoint at {ckpt!r} or {run / ckpt}. Available: "
            f"{sorted(x.name for x in run.glob('*.pt'))}")

    if cfg.arch not in _SPEC:
        raise ValueError(f"arch {cfg.arch!r} is not wired up for Phase 2; "
                         f"supported: {sorted(_SPEC)}")


    key = str(p.resolve())
    if state_cache is not None and key in state_cache:
        sd = state_cache[key]
    else:
        sd = _strip_prefixes(torch.load(p, map_location="cpu",
                                        weights_only=False)["model"])
        if state_cache is not None:
            state_cache[key] = sd
    model = build(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(
            f"{len(missing)} tensors missing from {p} -- the checkpoint does not match "
            f"the architecture it claims. First few: {missing[:5]}")


    spec = _SPEC[cfg.arch]
    inner = model if spec["attr"] is None else getattr(model, spec["attr"])


    if cfg.arch == "diffusion":
        max_pos = cfg.block_size - cfg.diff_target_len
    else:
        max_pos = cfg.max_seq_len

    bb = Backbone(inner, cfg, max_pos).to(device)
    n = sum(q.numel() for q in bb.parameters())
    print(f"[backbone] {p} | arch={cfg.arch} d_model={cfg.d_model} "
          f"layers={cfg.n_layers} | {n/1e6:.1f}M params kept | "
          f"{'bidirectional' if bb.bidirectional else 'CAUSAL (left context only)'} | "
          f"max_positions={max_pos}"
          + (f" | dropped {len(unexpected)} unused tensors" if unexpected else ""))
    if not bb.honours_pad_mask:
        print("[backbone] NOTE: this architecture ignores the attention mask; "
              "sequences must be right-padded.")
    return bb


def available(runs_root="../checkpoints") -> dict:
    out = {}
    for d in sorted(Path(runs_root).glob("*/config.json")):
        try:
            cfg = Config.load(d)
        except Exception as e:
            out[d.parent.name] = f"unreadable config ({type(e).__name__})"
            continue
        out[d.parent.name] = (f"ok ({cfg.arch})" if cfg.arch in _SPEC
                              else f"unsupported arch {cfg.arch!r}")
    return out

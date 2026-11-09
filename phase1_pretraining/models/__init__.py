from .decoder import DecoderLM
from .encoder import EncoderMLM
from .encdec import EncoderDecoder, shift_right
from .efficient import EfficientLM
from .mamba import MambaLM, MambaBlock
from .diffusion import DiffusionEncDec, DiTBlock
from .ssd import ssd_chunked, ssd_naive, ssd_step
from .layers import Stack, Block, Attention, RMSNorm, RoPE, SwiGLU

ARCHS = {"decoder": DecoderLM, "encoder": EncoderMLM, "encdec": EncoderDecoder,
         "efficient": EfficientLM, "mamba": MambaLM, "diffusion": DiffusionEncDec}


def build(cfg):
    if cfg.arch not in ARCHS:
        raise ValueError(f"unknown arch {cfg.arch!r}; choose from {sorted(ARCHS)}")
    return ARCHS[cfg.arch](cfg)


def count_params(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "total_M": round(total / 1e6, 2)}


__all__ = ["build", "count_params", "ARCHS", "DecoderLM", "EncoderMLM", "EncoderDecoder",
           "EfficientLM", "MambaLM", "MambaBlock", "DiffusionEncDec", "DiTBlock",
           "shift_right", "Stack", "Block",
           "Attention", "RMSNorm", "RoPE", "SwiGLU",
           "ssd_chunked", "ssd_naive", "ssd_step"]

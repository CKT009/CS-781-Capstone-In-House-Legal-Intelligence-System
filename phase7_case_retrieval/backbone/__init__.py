from .config import Config, PRESETS
from .layers import Stack, Block, Attention, RMSNorm, RoPE, SwiGLU, build_mask
from .decoder import DecoderLM
from .encoder import EncoderMLM
from .mamba import MambaLM, MambaBlock
from .ssd import ssd_chunked, ssd_naive, ssd_step
from .diffusion import DiffusionEncDec, DiTBlock

ARCHS = {"decoder": DecoderLM, "encoder": EncoderMLM,
         "mamba": MambaLM, "diffusion": DiffusionEncDec}


def build(cfg):
    if cfg.arch not in ARCHS:
        raise ValueError(f"unknown arch {cfg.arch!r}; choose from {sorted(ARCHS)}")
    return ARCHS[cfg.arch](cfg)


__all__ = ["Config", "PRESETS", "build", "ARCHS", "Stack", "Block", "Attention",
           "RMSNorm", "RoPE", "SwiGLU", "build_mask", "DecoderLM", "EncoderMLM",
           "MambaLM", "MambaBlock", "DiffusionEncDec", "DiTBlock",
           "ssd_chunked", "ssd_naive", "ssd_step"]

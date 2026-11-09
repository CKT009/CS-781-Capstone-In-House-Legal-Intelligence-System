import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Stack, make_norm


class EncoderMLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.stack = Stack(cfg, mode="bidirectional")
        self.transform = nn.Linear(cfg.d_model, cfg.d_model)
        self.norm = make_norm(cfg.norm, cfg.d_model)
        self.mlm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.mlm_head.weight = self.stack.embed_tokens.weight

    def encode(self, ids, pad_mask=None):
        return self.stack(ids, pad_mask=pad_mask)[0]

    def forward(self, ids=None, embeds=None, pad_mask=None, return_hidden=False, **kw):
        h = self.stack(ids, embeds, pad_mask)[0]
        logits = self.mlm_head(self.norm(F.gelu(self.transform(h))))
        out = {"logits": logits}
        if return_hidden:
            out["hidden"] = h
        return out

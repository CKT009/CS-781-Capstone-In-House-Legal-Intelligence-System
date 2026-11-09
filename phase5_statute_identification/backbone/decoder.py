import torch
import torch.nn as nn
from .layers import Stack


class DecoderLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.stack = Stack(cfg, mode="causal")
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.stack.embed_tokens.weight

    def forward(self, ids=None, embeds=None, pad_mask=None, past_kvs=None, use_cache=False,
                return_hidden=False):
        h, present = self.stack(ids, embeds, pad_mask, past_kvs, use_cache)
        out = {"logits": self.lm_head(h), "past_kvs": present}
        if return_hidden:
            out["hidden"] = h
        return out

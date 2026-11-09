import torch
import torch.nn as nn
from .layers import Stack


class EncoderDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = Stack(cfg, n_layers=cfg.n_layers, mode="bidirectional")
        self.decoder = Stack(cfg, n_layers=cfg.n_dec_layers, cross=True,
                             share_embed=self.encoder.embed_tokens, mode="causal")
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.encoder.embed_tokens.weight

    def encode(self, ids, pad_mask=None):
        return self.encoder(ids, pad_mask=pad_mask)[0]

    def forward(self, ids=None, dec_ids=None, pad_mask=None, enc=None, past_kvs=None,
                use_cache=False, return_hidden=False, **kw):
        if enc is None:
            enc = self.encode(ids, pad_mask)
        h, present = self.decoder(dec_ids, past_kvs=past_kvs, use_cache=use_cache,
                                  enc=enc, enc_pad_mask=pad_mask)
        out = {"logits": self.lm_head(h), "enc": enc, "past_kvs": present}
        if return_hidden:
            out["hidden"] = h
        return out


def shift_right(labels, pad_id=0):
    out = labels.new_zeros(labels.shape)
    out[:, 1:] = labels[:, :-1].clone()
    out[:, 0] = pad_id
    return out.masked_fill(out == -100, pad_id)

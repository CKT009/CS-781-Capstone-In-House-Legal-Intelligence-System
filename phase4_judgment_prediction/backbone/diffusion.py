import math

import torch
import torch.nn as nn

from .layers import Attention, RoPE, Stack, make_mlp


def timestep_embedding(t, dim, max_period=10000.0):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t.float()[:, None] * 1000.0 * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):

    def __init__(self, cfg, rope=None):
        super().__init__()
        d = cfg.d_model

        self.attn_norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(cfg, rope)
        self.cross_norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(cfg, cross=True)
        self.mlp_norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.mlp = make_mlp(cfg.act, d, cfg.d_ff, cfg.bias, cfg.dropout)
        self.adaLN = nn.Linear(cfg.diff_cond_dim, 6 * d, bias=True)

    def forward(self, x, cond, enc=None, enc_mask=None):
        sh_a, sc_a, g_a, sh_m, sc_m, g_m = self.adaLN(cond).chunk(6, dim=-1)


        x = x + g_a[:, None] * self.attn(modulate(self.attn_norm(x), sh_a, sc_a))[0]
        if enc is not None:
            x = x + self.cross_attn(self.cross_norm(x), kv_input=enc, mask=enc_mask)[0]
        x = x + g_m[:, None] * self.mlp(modulate(self.mlp_norm(x), sh_m, sc_m))
        return x


class DiffusionEncDec(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, c = cfg.d_model, cfg.diff_cond_dim


        self.encoder = Stack(cfg, n_layers=cfg.n_layers, mode="bidirectional")
        self.embed_tokens = self.encoder.embed_tokens

        self.rope = RoPE(cfg.head_dim, cfg.max_seq_len) if cfg.pos == "rope" else None
        self.embed_pos = (nn.Embedding(cfg.max_seq_len, d)
                          if cfg.pos == "learned" else None)
        self.t_embed = nn.Sequential(nn.Linear(c, 4 * c), nn.SiLU(), nn.Linear(4 * c, c))
        self.blocks = nn.ModuleList(
            [DiTBlock(cfg, self.rope) for _ in range(cfg.n_dec_layers)])
        self.final_norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Linear(c, 2 * d, bias=True)
        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self._init_decoder()


    @staticmethod
    def _basic(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def _init_decoder(self):
        mods = [self.t_embed, self.blocks, self.final_adaLN]
        if self.embed_pos is not None:
            mods.append(self.embed_pos)
        if not self.cfg.tie_embeddings:
            mods.append(self.lm_head)
        for m in mods:
            m.apply(self._basic)

        n = len(self.blocks)
        for blk in self.blocks:

            nn.init.zeros_(blk.adaLN.weight)
            nn.init.zeros_(blk.adaLN.bias)

            nn.init.zeros_(blk.cross_attn.o_proj.weight)
            if blk.cross_attn.o_proj.bias is not None:
                nn.init.zeros_(blk.cross_attn.o_proj.bias)

            nn.init.normal_(blk.attn.o_proj.weight, 0.0, 0.02 / math.sqrt(2 * n))
            nn.init.normal_(blk.mlp.down_proj.weight, 0.0, 0.02 / math.sqrt(2 * n))
        nn.init.zeros_(self.final_adaLN.weight)
        nn.init.zeros_(self.final_adaLN.bias)


    def encode(self, ids, pad_mask=None):
        return self.encoder(ids, pad_mask=pad_mask)[0]

    def forward(self, ids=None, dec_ids=None, t=None, pad_mask=None, enc=None,
                return_hidden=False, **kw):
        if enc is None:
            enc = self.encode(ids, pad_mask)
        B, L = dec_ids.shape
        h = self.embed_tokens(dec_ids)
        if self.embed_pos is not None:
            h = h + self.embed_pos(torch.arange(L, device=h.device))[None]

        if t is None:
            t = torch.ones(B, device=h.device)
        cond = self.t_embed(
            timestep_embedding(t, self.cfg.diff_cond_dim).to(h.dtype))

        enc_mask = (pad_mask.bool()[:, None, None, :]
                    if pad_mask is not None else None)
        for blk in self.blocks:
            if self.cfg.grad_checkpoint and self.training:
                h = torch.utils.checkpoint.checkpoint(
                    lambda _h, _b=blk: _b(_h, cond, enc=enc, enc_mask=enc_mask),
                    h, use_reentrant=False)
            else:
                h = blk(h, cond, enc=enc, enc_mask=enc_mask)

        sh, sc = self.final_adaLN(cond).chunk(2, dim=-1)
        h = modulate(self.final_norm(h), sh, sc)
        out = {"logits": self.lm_head(h), "enc": enc}
        if return_hidden:
            out["hidden"] = h
        return out

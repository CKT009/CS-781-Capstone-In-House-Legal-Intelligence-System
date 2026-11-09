import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm
from .ssd import ssd_chunked, ssd_step


class MambaBlock(nn.Module):

    def __init__(self, cfg, layer_idx=0):
        super().__init__()
        self.cfg = cfg
        self.d_inner = cfg.expand * cfg.d_model
        self.n_heads = self.d_inner // cfg.ssm_head_dim
        self.head_dim = cfg.ssm_head_dim
        self.d_state = cfg.d_state
        self.d_conv = cfg.d_conv
        self.chunk = cfg.chunk_size
        if self.d_inner % cfg.ssm_head_dim:
            raise ValueError("expand*d_model must be divisible by ssm_head_dim")

        self.norm = RMSNorm(cfg.d_model)

        self.in_proj = nn.Linear(
            cfg.d_model, 2 * self.d_inner + 2 * self.d_state + self.n_heads, bias=False)


        conv_ch = self.d_inner + 2 * self.d_state
        self.conv1d = nn.Conv1d(conv_ch, conv_ch, kernel_size=cfg.d_conv,
                                groups=conv_ch, padding=0, bias=True)


        self.A_log = nn.Parameter(torch.log(torch.empty(self.n_heads).uniform_(1, 16)))
        self.D = nn.Parameter(torch.ones(self.n_heads))


        dt = torch.exp(torch.rand(self.n_heads) * (math.log(0.1) - math.log(0.001))
                       + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

        self.norm_gate = RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, cfg.d_model, bias=False)

    def _project(self, h):
        zxbcdt = self.in_proj(h)
        z, xbc, dt = torch.split(
            zxbcdt, [self.d_inner, self.d_inner + 2 * self.d_state, self.n_heads], dim=-1)
        return z, xbc, dt

    def _finish(self, y, z, x):
        y = y + self.D[:, None] * x
        y = y.reshape(y.shape[0], y.shape[1], self.d_inner)
        y = self.norm_gate(y) * F.silu(z)
        return self.out_proj(y)

    def forward(self, h, cache=None, use_cache=False):
        b, l, _ = h.shape
        resid = h
        z, xbc, dt = self._project(self.norm(h))

        if cache is not None and l == 1:
            return resid + self._step(z, xbc, dt, cache), cache

        pad = self.d_conv - 1
        xbc_t = xbc.transpose(1, 2)


        left = cache[0] if cache is not None else xbc_t.new_zeros(b, xbc_t.shape[1], pad)
        padded = torch.cat([left, xbc_t], dim=-1)
        xbc = F.silu(self.conv1d(padded)).transpose(1, 2)

        x, B, C = torch.split(xbc, [self.d_inner, self.d_state, self.d_state], dim=-1)
        x = x.reshape(b, l, self.n_heads, self.head_dim)
        dt = F.softplus(dt + self.dt_bias)
        A = -torch.exp(self.A_log.float())

        prev = cache[1] if cache is not None else None
        y, state = ssd_chunked(x, dt, A, B, C, chunk=self.chunk, state=prev)

        if cache is not None:
            cache[0].copy_(padded[:, :, -pad:])
            cache[1].copy_(state)
        return resid + self._finish(y, z, x), cache
    def _step(self, z, xbc, dt, cache):
        conv_state, ssm_state = cache
        b = z.shape[0]
        window = torch.cat([conv_state, xbc.transpose(1, 2)], dim=-1)
        conv_out = F.silu((window * self.conv1d.weight.squeeze(1)).sum(-1)
                          + self.conv1d.bias)
        cache[0].copy_(window[:, :, 1:])

        x, B, C = torch.split(conv_out, [self.d_inner, self.d_state, self.d_state], dim=-1)
        x = x.reshape(b, self.n_heads, self.head_dim)
        dt = F.softplus(dt.squeeze(1) + self.dt_bias)
        A = -torch.exp(self.A_log.float())
        y, new_state = ssd_step(x, dt, A, B, C, ssm_state)
        cache[1].copy_(new_state)
        return self._finish(y[:, None], z, x[:, None])


class MambaLM(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([MambaBlock(cfg, i) for i in range(cfg.n_layers)])
        self.norm_out = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("out_proj.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layers))

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def allocate_cache(self, batch, device, dtype=torch.float32):
        out = []
        for blk in self.blocks:
            out.append([
                torch.zeros(batch, blk.d_inner + 2 * blk.d_state, blk.d_conv - 1,
                            device=device, dtype=dtype),
                torch.zeros(batch, blk.n_heads, blk.head_dim, blk.d_state,
                            device=device, dtype=dtype)])
        return out

    def forward(self, ids=None, embeds=None, pad_mask=None, past_kvs=None,
                use_cache=False, return_hidden=False):
        h = self.embed_tokens(ids) if embeds is None else embeds
        if use_cache and past_kvs is None:
            past_kvs = self.allocate_cache(h.shape[0], h.device, h.dtype)
        caches = []
        for i, blk in enumerate(self.blocks):
            cache = past_kvs[i] if past_kvs is not None else None
            if self.cfg.grad_checkpoint and self.training and not use_cache:
                h = torch.utils.checkpoint.checkpoint(
                    lambda _h, _b=blk: _b(_h)[0], h, use_reentrant=False)
                caches.append(None)
            else:
                h, c = blk(h, cache=cache, use_cache=use_cache)
                caches.append(c)
        h = self.norm_out(h)
        out = {"logits": self.lm_head(h), "past_kvs": past_kvs if use_cache else None}
        if return_hidden:
            out["hidden"] = h
        return out

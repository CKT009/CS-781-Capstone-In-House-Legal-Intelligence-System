import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dt) * self.weight


def make_norm(kind, d):
    return RMSNorm(d) if kind == "rms" else nn.LayerNorm(d)


class RoPE(nn.Module):

    def __init__(self, head_dim, max_seq_len, theta=10000.0):
        super().__init__()
        self.hd, self.theta = head_dim, theta
        self._build(max_seq_len)

    def _build(self, n):
        inv = 1.0 / (self.theta ** (torch.arange(0, self.hd, 2).float() / self.hd))
        f = torch.outer(torch.arange(n).float(), inv)
        self.register_buffer("cos", f.cos()[None, None], persistent=False)
        self.register_buffer("sin", f.sin()[None, None], persistent=False)

    def forward(self, seq_len, offset=0, device=None, dtype=None):
        if offset + seq_len > self.cos.shape[2]:
            self._build(offset + seq_len)
        sl = slice(offset, offset + seq_len)
        return (self.cos[:, :, sl].to(device=device, dtype=dtype),
                self.sin[:, :, sl].to(device=device, dtype=dtype))


def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class SwiGLU(nn.Module):

    def __init__(self, d, d_ff, bias=False, dropout=0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d, d_ff, bias=bias)
        self.up_proj = nn.Linear(d, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class GeluMLP(nn.Module):
    def __init__(self, d, d_ff, bias=True, dropout=0.0):
        super().__init__()
        self.up_proj = nn.Linear(d, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.down_proj(F.gelu(self.up_proj(x), approximate="tanh")))


def make_mlp(act, d, d_ff, bias, dropout):
    return SwiGLU(d, d_ff, bias, dropout) if act == "swiglu" else GeluMLP(d, d_ff, bias, dropout)


def build_mask(mode, q_len, k_len, device, window=512, n_global=0, pad_mask=None):
    offset = k_len - q_len
    qi = torch.arange(q_len, device=device)[:, None] + offset
    kj = torch.arange(k_len, device=device)[None, :]
    if mode == "bidirectional":
        allow = torch.ones(q_len, k_len, dtype=torch.bool, device=device)
    elif mode == "causal":
        allow = kj <= qi
    elif mode == "sliding":
        allow = (kj <= qi) & ((kj > qi - window) | (kj < n_global))


    else:
        raise ValueError(mode)
    allow = allow[None, None]
    if pad_mask is not None:
        allow = allow & pad_mask.bool()[:, None, None, :]
    return allow


class Attention(nn.Module):

    def __init__(self, cfg, rope=None, cross=False):
        super().__init__()
        self.nh, self.nkv, self.hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.rep = self.nh // self.nkv
        self.q_proj = nn.Linear(cfg.d_model, self.nh * self.hd, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.d_model, self.nkv * self.hd, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.d_model, self.nkv * self.hd, bias=cfg.bias)
        self.o_proj = nn.Linear(self.nh * self.hd, cfg.d_model, bias=cfg.bias)
        self.rope = None if cross else rope
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, kv_input=None, mask=None, past_kv=None, use_cache=False, offset=0):
        B, T, _ = x.shape
        src = x if kv_input is None else kv_input
        q = self.q_proj(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(src).view(B, src.shape[1], self.nkv, self.hd).transpose(1, 2)
        v = self.v_proj(src).view(B, src.shape[1], self.nkv, self.hd).transpose(1, 2)

        if self.rope is not None:
            cos, sin = self.rope(T, offset, x.device, q.dtype)
            q = apply_rope(q, cos, sin)


            kcos, ksin = self.rope(k.shape[2], offset, x.device, k.dtype)
            k = apply_rope(k, kcos, ksin)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], 2)
            v = torch.cat([past_kv[1], v], 2)
        present = (k, v) if use_cache else None

        if self.rep > 1:
            k = k.repeat_interleave(self.rep, 1)
            v = v.repeat_interleave(self.rep, 1)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.drop(self.o_proj(out.transpose(1, 2).reshape(B, T, self.nh * self.hd))), present


class Block(nn.Module):

    def __init__(self, cfg, rope=None, cross=False):
        super().__init__()
        self.attn_norm = make_norm(cfg.norm, cfg.d_model)
        self.attn = Attention(cfg, rope)
        self.cross = cross
        if cross:
            self.cross_norm = make_norm(cfg.norm, cfg.d_model)
            self.cross_attn = Attention(cfg, cross=True)
        self.mlp_norm = make_norm(cfg.norm, cfg.d_model)
        self.mlp = make_mlp(cfg.act, cfg.d_model, cfg.d_ff, cfg.bias, cfg.dropout)

    def forward(self, x, mask=None, past_kv=None, use_cache=False, offset=0,
                enc=None, enc_mask=None):
        h, present = self.attn(self.attn_norm(x), mask=mask, past_kv=past_kv,
                               use_cache=use_cache, offset=offset)
        x = x + h
        if self.cross and enc is not None:
            x = x + self.cross_attn(self.cross_norm(x), kv_input=enc, mask=enc_mask)[0]
        return x + self.mlp(self.mlp_norm(x)), present


class Stack(nn.Module):

    def __init__(self, cfg, n_layers=None, cross=False, share_embed=None, mode=None):
        super().__init__()
        self.cfg = cfg
        self.mode = mode or {"decoder": "causal", "encoder": "bidirectional",
                             "efficient": "sliding", "encdec": "causal"}[cfg.arch]
        n_layers = n_layers or cfg.n_layers
        self.embed_tokens = share_embed or nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embed_pos = nn.Embedding(cfg.max_seq_len, cfg.d_model) if cfg.pos == "learned" else None
        self.rope = RoPE(cfg.head_dim, cfg.max_seq_len) if cfg.pos == "rope" else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg, self.rope, cross) for _ in range(n_layers)])
        self.norm_out = make_norm(cfg.norm, cfg.d_model)
        self.apply(self._init)


        for n, p in self.named_parameters():
            if n.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * n_layers))

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, ids=None, embeds=None, pad_mask=None, past_kvs=None, use_cache=False,
                enc=None, enc_pad_mask=None):
        h = self.embed_tokens(ids) if embeds is None else embeds
        B, T, _ = h.shape
        offset = past_kvs[0][0].shape[2] if past_kvs else 0
        if self.embed_pos is not None:
            h = h + self.embed_pos(torch.arange(offset, offset + T, device=h.device))[None]
        h = self.drop(h)
        mask = build_mask(self.mode, T, offset + T, h.device, self.cfg.window,
                          self.cfg.n_global, pad_mask)
        enc_mask = enc_pad_mask.bool()[:, None, None, :] if enc_pad_mask is not None else None
        presents = []
        for i, blk in enumerate(self.blocks):
            if self.cfg.grad_checkpoint and self.training and not use_cache:
                h = torch.utils.checkpoint.checkpoint(
                    lambda _h, _b=blk: _b(_h, mask=mask, offset=offset, enc=enc,
                                          enc_mask=enc_mask)[0], h, use_reentrant=False)
                presents.append(None)
            else:
                h, p = blk(h, mask=mask, past_kv=past_kvs[i] if past_kvs else None,
                           use_cache=use_cache, offset=offset, enc=enc, enc_mask=enc_mask)
                presents.append(p)
        return self.norm_out(h), (presents if use_cache else None)

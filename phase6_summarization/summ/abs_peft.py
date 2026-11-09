from __future__ import annotations

import math

import torch
import torch.nn as nn

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")


def _blocks(model):
    return list(model.stack.blocks)


def _freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)


def report(model) -> dict:
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"trainable": tr, "total": tot, "pct": 100 * tr / tot}


class Bottleneck(nn.Module):
    def __init__(self, d, b):
        super().__init__()
        self.down, self.up = nn.Linear(d, b), nn.Linear(b, d)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def forward(self, h):
        return h + self.up(torch.nn.functional.gelu(self.down(h)))


def _hook_adapter(module, adapter, tuple_out):
    def hook(_m, _a, out):
        if tuple_out:
            return (adapter(out[0]),) + tuple(out[1:])
        return adapter(out)
    module.register_forward_hook(hook)


def _adapters(model, b, where):
    d = model.cfg.d_model
    for blk in _blocks(model):
        if "attn" in where:
            blk.peft_attn_adapter = Bottleneck(d, b)
            _hook_adapter(blk.attn, blk.peft_attn_adapter, tuple_out=True)
        if "mlp" in where:
            blk.peft_mlp_adapter = Bottleneck(d, b)
            _hook_adapter(blk.mlp, blk.peft_mlp_adapter, tuple_out=False)


class LoRMALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=16, alpha=32):
        super().__init__()
        self.base, self.scale = base, alpha / r
        n = base.out_features
        self.A = nn.Parameter(torch.randn(r, n) / math.sqrt(n))
        self.B = nn.Parameter(torch.zeros(n, r))

    def forward(self, x):
        y = self.base(x)
        return y + self.scale * (y @ self.A.t().to(y.dtype)) @ self.B.t().to(y.dtype)


def _lorma(model, r, alpha, targets=ATTN):
    for blk in _blocks(model):
        for t in targets:
            setattr(blk.attn, t, LoRMALinear(getattr(blk.attn, t), r, alpha))


class PrefixTuned(nn.Module):

    def __init__(self, lm, n_prefix=16, hidden=512):
        super().__init__()
        self.lm, self.cfg, self.n = lm, lm.cfg, n_prefix
        c = lm.cfg
        self.L, self.nkv, self.hd = c.n_layers, c.n_kv_heads, c.head_dim
        self.emb = nn.Parameter(torch.randn(n_prefix, c.d_model) * 0.02)
        self.mlp = nn.Sequential(nn.Linear(c.d_model, hidden), nn.Tanh(),
                                 nn.Linear(hidden, self.L * 2 * self.nkv * self.hd))
        nn.init.normal_(self.mlp[-1].weight, std=0.02); nn.init.zeros_(self.mlp[-1].bias)
        self.stack = lm.stack

    def prefix(self, B, dtype):
        kv = self.mlp(self.emb).view(self.n, self.L, 2, self.nkv, self.hd)
        kv = kv.permute(1, 2, 3, 0, 4).to(dtype)
        return [(kv[i, 0].expand(B, -1, -1, -1), kv[i, 1].expand(B, -1, -1, -1))
                for i in range(self.L)]

    def forward(self, ids=None, past_kvs=None, use_cache=False, **kw):
        if past_kvs is None:
            dt = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else torch.float32
            past_kvs = self.prefix(ids.shape[0], dt)
        h, present = self.lm.stack(ids, None, None, past_kvs, True)
        return {"logits": self.lm.lm_head(h), "past_kvs": present if use_cache else None}


class PromptTuned(nn.Module):
    def __init__(self, lm, n_prompt=20, init_ids=None):
        super().__init__()
        self.lm, self.cfg, self.n = lm, lm.cfg, n_prompt
        E = lm.stack.embed_tokens.weight

        idx = init_ids if init_ids is not None else torch.randint(7, E.shape[0], (n_prompt,))
        self.prompt = nn.Parameter(E[idx].detach().clone())
        self.stack = lm.stack

    def forward(self, ids=None, past_kvs=None, use_cache=False, **kw):
        if past_kvs is None:
            e = self.lm.stack.embed_tokens(ids)
            e = torch.cat([self.prompt.to(e.dtype).expand(ids.shape[0], -1, -1), e], 1)
            h, present = self.lm.stack(None, e, None, None, use_cache)
            h = h[:, self.n:]
        else:
            h, present = self.lm.stack(ids, None, None, past_kvs, use_cache)
        return {"logits": self.lm.lm_head(h), "past_kvs": present}


METHODS = ("full", "student", "houlsby", "pfeiffer", "prefix", "prompt", "lora", "qlora",
           "adalora", "lorma")
POSITIONS_USED = {"prefix": "n_prefix", "prompt": "n_prompt"}


def adapt(lm, method, **kw):
    if method in ("full", "student"):
        for p in lm.parameters():
            p.requires_grad_(True)
        return lm
    _freeze(lm)
    if method in ("houlsby", "pfeiffer"):
        _adapters(lm, kw.get("bottleneck", 64), ("attn", "mlp") if method == "houlsby" else ("mlp",))
        model = lm
    elif method == "lorma":
        _lorma(lm, kw.get("r", 16), kw.get("alpha", 32))
        model = lm
    elif method in ("lora", "qlora"):
        from summ.peft import apply_peft
        apply_peft(lm, method, r=kw.get("r", 16), alpha=kw.get("alpha", 32),
                   targets=kw.get("targets", ATTN))
        _freeze_keep(lm, ("lora_",))
        model = lm
    elif method == "adalora":
        from summ.adalora import apply_adalora
        apply_adalora(lm, r=kw.get("r", 12), target_r=kw.get("target_r", 6), targets=ATTN)
        _freeze_keep(lm, ("lora_",))
        model = lm
    elif method == "prefix":
        model = PrefixTuned(lm, kw.get("n_prefix", 16), kw.get("hidden", 512))
    elif method == "prompt":
        model = PromptTuned(lm, kw.get("n_prompt", 20))
    else:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")

    if method in ("houlsby", "pfeiffer"):
        trainable = [p for n, p in model.named_parameters() if "peft_" in n]
    elif method == "lorma":
        trainable = [p for m in model.modules() if isinstance(m, LoRMALinear) for p in (m.A, m.B)]
    elif method == "prefix":
        trainable = [model.emb, *model.mlp.parameters()]
    elif method == "prompt":
        trainable = [model.prompt]
    else:
        trainable = []
    for p in trainable:
        p.requires_grad_(True)
    assert report(model)["trainable"] > 0, f"{method}: nothing trainable"
    assert not any(p.requires_grad for n, p in model.named_parameters()
                   if n.endswith("lm_head.weight") or "embed_tokens" in n), "lm head must stay frozen"
    return model


def _freeze_keep(model, keys):
    for n, p in model.named_parameters():
        p.requires_grad_(any(k in n for k in keys))

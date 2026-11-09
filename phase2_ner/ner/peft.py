from __future__ import annotations

import inspect
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.layers import Attention, Block, RMSNorm
from backbone.mamba import MambaBlock

try:
    from backbone.diffusion import DiTBlock
except Exception:
    DiTBlock = None

METHODS = ["full", "head_only", "last_k", "bitfit", "lora", "qlora", "adapter",
           "prefix", "prompt"]


_APPLIED = "_peft_method"
_HOOKS = "_peft_hook_handles"
_ADAPTER_ATTR = "peft_adapter"
_PREFIX_ATTR = "peft_prefix"

_BLOCK_TYPES = tuple(t for t in (Block, MambaBlock, DiTBlock) if isinstance(t, type))
_NORM_TYPES = (RMSNorm, nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d)


_ALWAYS_TRAIN_PREFIXES = ("head", "crf")


def _is_head_param(name: str) -> bool:
    return any(part.startswith(_ALWAYS_TRAIN_PREFIXES) for part in name.split("."))


def _set_all(model: nn.Module, flag: bool) -> None:
    for p in model.parameters():
        p.requires_grad_(flag)


def _unfreeze_head(model: nn.Module) -> int:
    n = 0
    for name, p in model.named_parameters():
        if _is_head_param(name):
            p.requires_grad_(True)
            n += 1
    if n == 0:
        warnings.warn(
            "apply_peft: no parameter matched the head pattern (a dotted name component "
            f"starting with one of {_ALWAYS_TRAIN_PREFIXES}). If this model has a Phase-2 "
            "classifier, rename it to `head*` -- otherwise it is frozen and the run "
            "cannot learn anything.",
            RuntimeWarning, stacklevel=3)
    return n


def _blocks(model: nn.Module) -> list[tuple[str, nn.Module]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, _BLOCK_TYPES)]


def _final_norms(model: nn.Module) -> list[tuple[str, nn.Module]]:
    return [(n, m) for n, m in model.named_modules() if n.split(".")[-1] == "norm_out"]


def _self_attentions(model: nn.Module) -> list[tuple[str, nn.Module]]:
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, Attention) and n.split(".")[-1] == "attn"]


def _get_parent(model: nn.Module, name: str):
    parts = name.split(".")
    parent = model.get_submodule(".".join(parts[:-1])) if len(parts) > 1 else model
    return parent, parts[-1]


def _replace_submodule(model: nn.Module, name: str, new: nn.Module) -> None:
    parent, key = _get_parent(model, name)
    if isinstance(parent, (nn.ModuleList, nn.Sequential)) and key.isdigit():
        parent[int(key)] = new
    else:
        setattr(parent, key, new)


def _block_width(block: nn.Module) -> int:
    attn = getattr(block, "attn", None)
    if attn is not None and hasattr(attn, "o_proj"):
        return attn.o_proj.out_features
    if hasattr(block, "out_proj"):
        return block.out_proj.out_features
    for m in block.modules():
        if isinstance(m, _NORM_TYPES) and getattr(m, "weight", None) is not None:
            return m.weight.shape[0]
    raise RuntimeError(f"cannot infer d_model for block of type {type(block).__name__}")


def _mark(model: nn.Module, method: str) -> None:
    object.__setattr__(model, _APPLIED, method)


def _add_hook(model: nn.Module, handle) -> None:
    if not hasattr(model, _HOOKS):
        object.__setattr__(model, _HOOKS, [])
    getattr(model, _HOOKS).append(handle)


_NF4_LEVELS = (
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0)


def nf4_quantize(w: torch.Tensor, block: int = 64):
    levels = torch.tensor(_NF4_LEVELS, device=w.device, dtype=torch.float32)
    flat = w.detach().reshape(-1).float()
    pad = (-flat.numel()) % block
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(-1, block)
    scale = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    idx = (blocks / scale).unsqueeze(-1).sub(levels.view(1, 1, -1)).abs().argmin(-1)
    idx = idx.to(torch.uint8)
    packed = (idx[:, 0::2] << 4) | idx[:, 1::2]
    return packed, scale.squeeze(1).half(), pad, tuple(w.shape)


def nf4_dequantize(packed, scale, pad, shape, dtype=torch.float32, block: int = 64):
    levels = torch.tensor(_NF4_LEVELS, device=packed.device, dtype=dtype)
    hi, lo = (packed >> 4).long(), (packed & 0x0F).long()
    idx = torch.stack((hi, lo), dim=-1).view(packed.shape[0], block)
    out = (levels[idx] * scale.unsqueeze(1).to(dtype)).reshape(-1)
    if pad:
        out = out[:-pad]
    return out.view(shape)


class QLoRALinear(nn.Module):

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0, block: int = 64):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"QLoRALinear expects nn.Linear, got {type(base).__name__}")
        if r <= 0:
            raise ValueError("QLoRA rank r must be >= 1")
        if block % 2:
            raise ValueError("block must be even (two 4-bit indices pack into one byte)")
        w = base.weight
        self._dtype, self._shape, self.block = w.dtype, tuple(w.shape), int(block)
        packed, scale, pad, shape = nf4_quantize(w, block)


        self.register_buffer("w_packed", packed)
        self.register_buffer("w_scale", scale)
        self.pad = pad
        self.register_buffer("bias", base.bias.detach().clone()
                             if base.bias is not None else None)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.r, self.alpha = int(r), float(alpha)
        self.scaling = self.alpha / self.r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features,
                                               device=w.device, dtype=w.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r,
                                               device=w.device, dtype=w.dtype))
        nn.init.normal_(self.lora_A, mean=0.0, std=self.r ** -0.5)
        self.drop = nn.Dropout(dropout)
        self._w0_err = None

    @property
    def weight(self) -> torch.Tensor:
        return nf4_dequantize(self.w_packed, self.w_scale, self.pad, self._shape,
                              self._dtype, self.block)

    def quant_error(self, original: torch.Tensor) -> dict:
        w = self.weight.float()
        o = original.detach().float().to(w.device)
        return {"rel_fro": ((w - o).norm() / o.norm().clamp_min(1e-12)).item(),
                "max_abs": (w - o).abs().max().item()}

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight.to(x.dtype),
                     None if self.bias is None else self.bias.to(x.dtype))
        return y + F.linear(F.linear(self.drop(x), self.lora_A), self.lora_B) * self.scaling

    def extra_repr(self) -> str:
        bits = 4 + 16 / self.block
        return (f"in={self.in_features}, out={self.out_features}, r={self.r}, "
                f"alpha={self.alpha}, nf4 block={self.block} (~{bits:.2f} bits/weight)")


class LoRALinear(nn.Module):

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base).__name__}")
        if r <= 0:
            raise ValueError("LoRA rank r must be >= 1")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.alpha = int(r), float(alpha)
        self.scaling = self.alpha / self.r
        w = base.weight
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features,
                                               device=w.device, dtype=w.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r,
                                               device=w.device, dtype=w.dtype))
        nn.init.normal_(self.lora_A, mean=0.0, std=self.r ** -0.5)


        self.drop = nn.Dropout(dropout)

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(self.drop(x), self.lora_A),
                                       self.lora_B) * self.scaling


    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:g}"


def merge_lora(model: nn.Module) -> nn.Module:
    names = [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    if not names:
        raise RuntimeError("merge_lora: no LoRALinear modules found -- was apply_peft("
                           "model, 'lora') ever called on this model?")
    for name in names:
        mod = model.get_submodule(name)
        with torch.no_grad():
            mod.base.weight.add_(mod.delta_weight().to(mod.base.weight.dtype))
        _replace_submodule(model, name, mod.base)
    if getattr(model, _APPLIED, None) == "lora":
        _mark(model, "lora(merged)")
    return model


class Adapter(nn.Module):

    def __init__(self, d: int, bottleneck: int = 64):
        super().__init__()
        self.down_proj = nn.Linear(d, bottleneck)
        self.up_proj = nn.Linear(bottleneck, d)
        nn.init.normal_(self.down_proj.weight, 0.0, 1e-2)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.up_proj(F.gelu(self.down_proj(h)))


def _adapter_forward_hook(block: nn.Module, args, output):
    adapter = getattr(block, _ADAPTER_ATTR)
    if isinstance(output, tuple):
        h = output[0]
        return (h + adapter(h),) + tuple(output[1:])
    return output + adapter(output)


class PrefixKV(nn.Module):

    def __init__(self, n_kv_heads: int, n_prefix: int, head_dim: int, ref: torch.Tensor):
        super().__init__()
        self.n_prefix = int(n_prefix)
        self.k = nn.Parameter(torch.randn(n_kv_heads, n_prefix, head_dim,
                                          device=ref.device, dtype=ref.dtype) * 0.02)
        self.v = nn.Parameter(torch.zeros(n_kv_heads, n_prefix, head_dim,
                                          device=ref.device, dtype=ref.dtype))


def _prefix_pre_hook(attn: nn.Module, args, kwargs):
    pre = getattr(attn, _PREFIX_ATTR)
    if kwargs.get("use_cache"):
        raise RuntimeError(
            "prefix tuning and the KV cache both occupy Attention's past_kv slot: the "
            "returned `present` would contain the prefix and the next decode step would "
            "prepend it a second time. Run the prefix model with use_cache=False (which "
            "is what a tagger does anyway), or write cache-aware prefix bookkeeping.")
    if kwargs.get("past_kv") is not None:
        raise RuntimeError("prefix tuning cannot be combined with an external past_kv")

    x = kwargs["x"] if "x" in kwargs else args[0]
    B = x.shape[0]
    kwargs = dict(kwargs)
    kwargs["past_kv"] = (pre.k.unsqueeze(0).expand(B, -1, -1, -1),
                         pre.v.unsqueeze(0).expand(B, -1, -1, -1))
    mask = kwargs.get("mask")
    if mask is not None:

        pad = mask.new_ones(*mask.shape[:-1], pre.n_prefix)
        kwargs["mask"] = torch.cat([pad, mask], dim=-1)
    return args, kwargs


def apply_peft(model: nn.Module, method: str, **kw) -> nn.Module:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")


    _ref = next(model.parameters(), None)
    _dev = _ref.device if _ref is not None else None

    previous = getattr(model, _APPLIED, None)
    if previous is not None:
        raise RuntimeError(
            f"apply_peft: {previous!r} has already been applied to this "
            f"{type(model).__name__}; refusing to apply {method!r} on top. Applying twice "
            f"would wrap LoRA in LoRA, stack two adapters per block, or register a second "
            f"prefix hook -- all of which train, log a plausible parameter count, and "
            f"quietly measure the wrong architecture. Rebuild the model instead.")


    if method == "prefix":
        attns = _self_attentions(model)
        if not attns:
            if any(isinstance(m, MambaBlock) for m in model.modules()):
                raise NotImplementedError(
                    "prefix tuning is not defined for the 'mamba' backbone "
                    "(backbone.mamba.MambaBlock, selective state-space / Mamba-2 SSD): it "
                    "has no attention layers, so there are no keys or values to prepend a "
                    "prefix to. Its mixing is a fixed-size recurrent state, not a "
                    "set of attendable positions. Use 'lora' (on in_proj/out_proj), "
                    "'adapter', 'bitfit' or 'last_k' for the Mamba backbone instead.")
            raise NotImplementedError(
                f"prefix tuning found no self-attention layers in "
                f"{type(model).__name__}; it cannot be applied to this backbone.")

    if method == "full":
        _set_all(model, True)

    elif method == "head_only":
        _set_all(model, False)

    elif method == "last_k":
        k = int(kw.pop("k", 4))
        _set_all(model, False)
        blocks = _blocks(model)
        if not blocks:
            raise RuntimeError(f"last_k: found no blocks in {type(model).__name__}")


        for _, blk in blocks[-max(1, min(k, len(blocks))):]:
            for p in blk.parameters():
                p.requires_grad_(True)


        for _, nm in _final_norms(model):
            for p in nm.parameters():
                p.requires_grad_(True)

    elif method == "bitfit":
        _set_all(model, False)
        for name, p in model.named_parameters():
            if "bias" in name.split(".")[-1]:
                p.requires_grad_(True)


        for _, nm in model.named_modules():
            if isinstance(nm, _NORM_TYPES):
                for p in nm.parameters(recurse=False):
                    p.requires_grad_(True)
    elif method == "lora":
        r = int(kw.pop("r", 8))
        alpha = float(kw.pop("alpha", 16))
        dropout = float(kw.pop("dropout", 0.0))
        targets = tuple(kw.pop("targets", ("q_proj", "v_proj")))
        _set_all(model, False)

        hits = [n for n, m in model.named_modules()
                if isinstance(m, nn.Linear) and n.split(".")[-1] in targets]
        if not hits:
            raise RuntimeError(
                f"lora: no nn.Linear matched targets={targets} in "
                f"{type(model).__name__}. Transformer blocks expose q_proj/k_proj/"
                f"v_proj/o_proj/gate_proj/up_proj/down_proj; MambaBlock exposes "
                f"in_proj/out_proj.")
        for name in hits:
            _replace_submodule(model, name,
                               LoRALinear(model.get_submodule(name), r, alpha, dropout))

    elif method == "qlora":


        r = int(kw.pop("r", 8))
        alpha = float(kw.pop("alpha", 16))
        dropout = float(kw.pop("dropout", 0.0))
        block = int(kw.pop("block", 64))
        targets = tuple(kw.pop("targets", ("q_proj", "v_proj")))
        _set_all(model, False)
        hits = [n for n, m in model.named_modules()
                if isinstance(m, nn.Linear) and n.split(".")[-1] in targets]
        if not hits:
            raise RuntimeError(
                f"qlora: no nn.Linear matched targets={targets} in "
                f"{type(model).__name__}. See the `lora` branch for the available names.")
        errs = []
        for name in hits:
            base = model.get_submodule(name)


            measurable = not base.weight.is_meta
            original = base.weight.detach().clone() if measurable else None
            q = QLoRALinear(base, r, alpha, dropout, block)
            if measurable:
                errs.append(q.quant_error(original)["rel_fro"])
            _replace_submodule(model, name, q)


        if errs:
            print(f"[qlora] {len(hits)} layers quantised to NF4 (block={block}); "
                  f"relative Frobenius error mean {sum(errs)/len(errs):.4f} "
                  f"max {max(errs):.4f}")

    elif method == "adapter":
        bottleneck = int(kw.pop("bottleneck", 64))
        _set_all(model, False)
        blocks = _blocks(model)
        if not blocks:
            raise RuntimeError(f"adapter: found no blocks in {type(model).__name__}")
        for name, blk in blocks:
            if hasattr(blk, _ADAPTER_ATTR):
                raise RuntimeError(
                    f"adapter: {name} already carries a {_ADAPTER_ATTR}")
            blk.add_module(_ADAPTER_ATTR, Adapter(_block_width(blk), bottleneck))
            _add_hook(model, blk.register_forward_hook(_adapter_forward_hook))

    elif method == "prefix":
        n_prefix = int(kw.pop("n_prefix", 16))
        _set_all(model, False)
        for name, attn in _self_attentions(model):
            if hasattr(attn, _PREFIX_ATTR):
                raise RuntimeError(f"prefix: {name} already carries a {_PREFIX_ATTR}")
            attn.add_module(_PREFIX_ATTR,
                            PrefixKV(attn.nkv, n_prefix, attn.hd, attn.q_proj.weight))
            _add_hook(model, attn.register_forward_pre_hook(
                _prefix_pre_hook, with_kwargs=True))

    elif method == "prompt":
        _set_all(model, False)
        _apply_prompt(model, int(kw.pop("n_prompt", 20)))

    if kw:
        raise TypeError(f"apply_peft({method!r}) got unexpected kwargs: {sorted(kw)}")

    if _dev is not None:
        model.to(device=_dev)


    _unfreeze_head(model)
    _mark(model, method)
    return model


_PROMPT_ATTR = "peft_prompt"


class SoftPrompt(nn.Module):

    def __init__(self, embed: nn.Embedding, n: int):
        super().__init__()
        self.n = n
        with torch.no_grad():
            idx = torch.randint(0, embed.num_embeddings, (n,))
            init = embed.weight[idx].detach().clone()
        self.prompt = nn.Parameter(init)


def _embedding_owner(model: nn.Module) -> tuple[str, nn.Module]:
    cands = [(n, m) for n, m in model.named_modules()
             if isinstance(getattr(m, "embed_tokens", None), nn.Embedding)
             and "embeds" in inspect.signature(m.forward).parameters]
    if not cands:
        raise NotImplementedError(
            "prompt tuning needs a module exposing `embed_tokens` and accepting "
            "`embeds=` (Stack, MambaLM). None found on this model.")
    return min(cands, key=lambda nm: len(nm[0].split(".")))


def _prompt_pre_hook(mod, args, kwargs):
    sp = getattr(mod, _PROMPT_ATTR)
    args = list(args)
    ids, slot = kwargs.pop("ids", None), None
    if ids is None and args and torch.is_tensor(args[0]) and args[0].dtype == torch.long:
        ids, slot = args[0], 0
    if ids is None:
        return tuple(args), kwargs

    emb = mod.embed_tokens(ids)
    B = emb.shape[0]
    emb = torch.cat([sp.prompt[None].expand(B, -1, -1).to(emb.dtype), emb], dim=1)

    if slot is None:
        kwargs["embeds"] = emb
    else:
        args[slot] = None
        if len(args) > slot + 1:
            args[slot + 1] = emb
        else:
            kwargs["embeds"] = emb

    pm_slot = None if slot is None else slot + 2
    if pm_slot is not None and len(args) > pm_slot and torch.is_tensor(args[pm_slot]):
        args[pm_slot] = torch.cat([args[pm_slot].new_ones(B, sp.n), args[pm_slot]], dim=1)
    elif torch.is_tensor(kwargs.get("pad_mask")):
        pm = kwargs["pad_mask"]
        kwargs["pad_mask"] = torch.cat([pm.new_ones(B, sp.n), pm], dim=1)
    return tuple(args), kwargs


def _prompt_post_hook(mod, args, kwargs, out):
    n = getattr(mod, _PROMPT_ATTR).n

    def cut(x):
        return x[:, n:] if torch.is_tensor(x) and x.ndim >= 2 and x.shape[1] > n else x

    if torch.is_tensor(out):
        return cut(out)
    if isinstance(out, dict):
        return {k: cut(v) for k, v in out.items()}
    if isinstance(out, tuple):
        return tuple(cut(v) for v in out)
    return out


def _apply_prompt(model: nn.Module, n_prompt: int) -> None:
    name, owner = _embedding_owner(model)
    if hasattr(owner, _PROMPT_ATTR):
        raise RuntimeError(f"prompt: {name} already carries a soft prompt")
    owner.add_module(_PROMPT_ATTR, SoftPrompt(owner.embed_tokens, n_prompt))
    _add_hook(model, owner.register_forward_pre_hook(_prompt_pre_hook, with_kwargs=True))
    _add_hook(model, owner.register_forward_hook(_prompt_post_hook, with_kwargs=True))


def trainable_report(model: nn.Module) -> dict:
    seen, trainable, total, groups = set(), 0, 0, {}
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            key = name.split(".")[0]
            groups[key] = groups.get(key, 0) + p.numel()
    return {"trainable": trainable, "total": total,
            "pct": 100.0 * trainable / max(total, 1),
            "groups": dict(sorted(groups.items()))}


def describe(method: str, **kw) -> str:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    if method == "full":
        return "full fine-tuning: every parameter trainable (baseline / upper reference)"
    if method == "prompt":
        n = int(kw.get("n_prompt", 20))
        return (f"prompt tuning: {n} learned virtual tokens prepended to the input "
                f"embeddings, initialised from sampled vocabulary embeddings; the "
                f"backbone is untouched and this is the only PEFT method here that "
                f"works on an attention-free (Mamba) backbone")
    if method == "head_only":
        return ("head-only linear probe: backbone frozen, classifier only -- measures how "
                "much of the task is linearly decodable from frozen Phase-1 features")
    if method == "last_k":
        k = int(kw.get("k", 4))
        return (f"partial fine-tuning: top {k} block(s) + final norm + head trainable, "
                f"all lower layers frozen")
    if method == "bitfit":
        return ("BitFit (bias + norm gain): only additive bias terms and elementwise norm "
                "gains train -- no weight matrix moves; norm gains included because "
                "RMSNorm has no bias and bias=False presets would make it vacuous")
    if method == "lora":
        r, alpha = int(kw.get("r", 8)), float(kw.get("alpha", 16))
        tgt = ",".join(kw.get("targets", ("q_proj", "v_proj")))
        d = float(kw.get("dropout", 0.0))
        return (f"LoRA: frozen weights + rank-{r} update B@A on [{tgt}], scaling "
                f"alpha/r={alpha / max(r, 1):g}, dropout={d:g}, B zero-init (identity at "
                f"step 0), mergeable at inference")
    if method == "adapter":
        b = int(kw.get("bottleneck", 64))
        return (f"bottleneck adapter: frozen backbone + a d->{b}->d GELU adapter added to "
                f"each block output via a forward hook, up-projection zero-init "
                f"(identity at step 0)")
    n = int(kw.get("n_prefix", 16))
    return (f"prefix tuning: frozen backbone + {n} learned key/value vectors prepended to "
            f"every self-attention layer (position-free, injected through Attention's "
            f"past_kv path); undefined for the Mamba backbone")


__all__ = ["METHODS", "apply_peft", "trainable_report", "describe", "merge_lora",
           "LoRALinear", "Adapter", "PrefixKV"]

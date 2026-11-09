import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backbone.config import Config
from backbone.encoder import EncoderMLM
from backbone.mamba import MambaLM
from ner.peft import (METHODS, Adapter, LoRALinear, apply_peft,
                      describe, merge_lora, trainable_report)

N_TAGS = 9
SEQ = 16
BATCH = 2


class TinyTagger(nn.Module):

    def __init__(self, cfg, n_tags=N_TAGS):
        super().__init__()
        self.backbone = EncoderMLM(cfg)
        self.head = nn.Linear(cfg.d_model, n_tags)

    def forward(self, ids):
        h = self.backbone(ids=ids, return_hidden=True)["hidden"]
        return self.head(h)


def tiny_cfg(**over):


    base = dict(name="tiny_peft", arch="encoder", d_model=64, n_layers=2, n_heads=4,
                n_kv_heads=4, vocab_size=100, max_seq_len=32, block_size=32,
                norm="layer", pos="learned", act="gelu", bias=True,
                tie_embeddings=True, dropout=0.0, grad_checkpoint=False, device="cpu")
    base.update(over)
    return Config(**base)


def build(seed=0, **over):
    torch.manual_seed(seed)
    return TinyTagger(tiny_cfg(**over)).eval()


def ids():
    g = torch.Generator().manual_seed(1234)
    return torch.randint(0, 100, (BATCH, SEQ), generator=g)


DEFAULT_KW = {"full": {}, "head_only": {}, "last_k": {}, "bitfit": {},
              "lora": {}, "qlora": {"block": 8}, "adapter": {}, "prefix": {},
              "prompt": {"n_prompt": 4}}
assert set(DEFAULT_KW) == set(METHODS), (
    "a method was added to peft.METHODS without default kwargs here, which turns every "
    f"parametrised test into a KeyError: {set(METHODS) ^ set(DEFAULT_KW)}")


@pytest.mark.parametrize("method", METHODS)
def test_method_runs_and_report_is_sane(method):
    x = ids()
    ref = build()(x)
    model = apply_peft(build(), method, **DEFAULT_KW[method])
    out = model(x)

    assert out.shape == ref.shape == (BATCH, SEQ, N_TAGS)
    assert torch.isfinite(out).all(), f"{method} produced non-finite output"

    rep = trainable_report(model)
    assert rep["total"] > 0
    assert 0 < rep["trainable"] <= rep["total"]
    assert rep["pct"] == pytest.approx(100.0 * rep["trainable"] / rep["total"])
    if method == "full":
        assert rep["trainable"] == rep["total"]
    else:
        assert rep["trainable"] < rep["total"], f"{method} froze nothing"

    assert rep["groups"].get("head", 0) == sum(p.numel() for p in model.head.parameters())
    assert isinstance(describe(method, **DEFAULT_KW[method]), str)


def test_parameter_count_ordering():
    counts = {}
    for method in METHODS:
        counts[method] = trainable_report(
            apply_peft(build(), method, **DEFAULT_KW[method]))["trainable"]

    order = ["head_only", "bitfit", "lora", "adapter", "last_k", "full"]
    got = [counts[m] for m in order]
    assert got == sorted(got) and len(set(got)) == len(got), (
        f"expected strictly increasing {order}, got "
        f"{ {m: counts[m] for m in order} }")


    assert counts["head_only"] < counts["prefix"] < counts["last_k"]

    for m in ("head_only", "bitfit", "lora", "prefix", "adapter"):
        pct = 100.0 * counts[m] / counts["full"]
        assert pct < 25.0, f"{m} trains {pct:.1f}% of the model -- not parameter-efficient"


def test_last_k_is_monotone_in_k():
    c1 = trainable_report(apply_peft(build(), "last_k", k=1))["trainable"]
    c2 = trainable_report(apply_peft(build(), "last_k", k=2))["trainable"]
    c99 = trainable_report(apply_peft(build(), "last_k", k=99))["trainable"]
    assert c1 < c2 == c99


def test_lora_is_identity_at_init():
    x = ids()
    base, lora = build(), apply_peft(build(), "lora")
    with torch.no_grad():
        assert torch.allclose(base(x), lora(x), atol=0, rtol=0), (
            "LoRA changed the function at step 0: B is not zero-initialised")

    wrapped = [n for n, m in lora.named_modules() if isinstance(m, LoRALinear)]
    assert len(wrapped) == 4
    assert all(n.split(".")[-1] in ("q_proj", "v_proj") for n in wrapped)
    assert all(not m.base.weight.requires_grad and m.lora_A.requires_grad
               and m.lora_B.requires_grad
               for m in lora.modules() if isinstance(m, LoRALinear))


def test_adapter_is_identity_at_init():
    x = ids()
    base, ad = build(), apply_peft(build(), "adapter")
    with torch.no_grad():
        assert torch.allclose(base(x), ad(x), atol=0, rtol=0), (
            "adapter changed the function at step 0: up_proj is not zero-initialised")
    adapters = [m for m in ad.modules() if isinstance(m, Adapter)]
    assert len(adapters) == 2

    assert all(a.down_proj.weight.abs().sum() > 0 for a in adapters)

    assert "backbone.stack.blocks.0.attn.q_proj.weight" in dict(ad.named_parameters())
    assert "backbone.stack.blocks.0.peft_adapter.up_proj.weight" in dict(ad.named_parameters())


def test_adapter_moves_the_function_once_trained():
    x = ids()
    ad = apply_peft(build(), "adapter")
    with torch.no_grad():
        before = ad(x).clone()
        for m in ad.modules():
            if isinstance(m, Adapter):
                nn.init.normal_(m.up_proj.weight, 0.0, 0.1)
        after = ad(x)
    assert not torch.allclose(before, after), "adapter forward hook never fired"
    assert torch.isfinite(after).all()


def test_merge_lora_matches_unmerged():
    x = ids()
    model = apply_peft(build(), "lora")
    torch.manual_seed(7)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                nn.init.normal_(m.lora_A, 0.0, 0.5)
                nn.init.normal_(m.lora_B, 0.0, 0.5)
        unmerged = model(x).clone()
        assert not torch.allclose(unmerged, build()(x)), "randomised LoRA did nothing"
        merged = merge_lora(model)(x)
    assert torch.allclose(unmerged, merged, atol=1e-5, rtol=1e-4)

    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    assert "backbone.stack.blocks.0.attn.q_proj.weight" in dict(model.named_parameters())


def test_merge_lora_without_lora_raises():
    with pytest.raises(RuntimeError, match="no LoRALinear"):
        merge_lora(apply_peft(build(), "bitfit"))


@pytest.mark.parametrize("method", [m for m in METHODS if m != "full"])
def test_frozen_parameters_get_no_gradient(method):
    model = apply_peft(build(), method, **DEFAULT_KW[method])
    frozen = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
    assert frozen, f"{method} left nothing frozen"

    model(ids()).square().mean().backward()

    for n, p in frozen:
        assert p.requires_grad is False
        assert p.grad is None, f"{method}: frozen {n} received a gradient"

    got = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is not None]
    assert got, f"{method}: no trainable parameter received a gradient"
    assert any(n.split(".")[-1].startswith(("weight", "bias")) and "head" in n
               for n in got), f"{method}: the classifier head received no gradient"


@pytest.mark.parametrize("method", [m for m in METHODS if m != "full"])
def test_expected_groups_are_frozen(method):
    model = apply_peft(build(), method, **DEFAULT_KW[method])
    g = dict(model.named_parameters())
    emb = g["backbone.stack.embed_tokens.weight"]
    assert not emb.requires_grad, f"{method} left the embedding matrix trainable"
    if method == "qlora":


        assert "backbone.stack.blocks.0.attn.q_proj.weight" not in g
        assert "backbone.stack.blocks.0.attn.q_proj.base.weight" not in g
        qm = model.get_submodule("backbone.stack.blocks.0.attn.q_proj")
        assert {n for n, _ in qm.named_parameters()} == {"lora_A", "lora_B"}, \
            "qlora left something other than the adapter trainable"
        q0 = None
    else:
        q0 = g["backbone.stack.blocks.0.attn.q_proj.weight"] if method != "lora" else \
            g["backbone.stack.blocks.0.attn.q_proj.base.weight"]
    if q0 is not None and method != "last_k":
        assert not q0.requires_grad, f"{method} left a pretrained weight matrix trainable"
    if method == "bitfit":
        assert g["backbone.stack.blocks.0.attn.q_proj.bias"].requires_grad
        assert g["backbone.stack.blocks.0.attn_norm.weight"].requires_grad


@pytest.mark.parametrize("method", METHODS)
def test_applying_twice_raises(method):
    model = apply_peft(build(), method, **DEFAULT_KW[method])
    with pytest.raises(RuntimeError, match="already been applied"):
        apply_peft(model, method, **DEFAULT_KW[method])


@pytest.mark.parametrize("second", ["lora", "adapter", "prefix", "full"])
def test_stacking_different_methods_also_raises(second):
    model = apply_peft(build(), "lora")
    with pytest.raises(RuntimeError, match="already been applied"):
        apply_peft(model, second)


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown method"):
        apply_peft(build(), "not_a_real_method")
    with pytest.raises(TypeError, match="unexpected kwargs"):
        apply_peft(build(), "lora", rank=4)


def mamba_cfg():

    return Config(name="tiny_mamba", arch="mamba", d_model=64, n_layers=2, n_heads=4,
                  n_kv_heads=4, vocab_size=100, max_seq_len=32, block_size=32,
                  norm="rms", expand=2, d_state=16, ssm_head_dim=32, d_conv=4,
                  chunk_size=8, dropout=0.0, grad_checkpoint=False, device="cpu")


class TinyMambaTagger(nn.Module):
    def __init__(self, cfg, n_tags=N_TAGS):
        super().__init__()
        self.backbone = MambaLM(cfg)
        self.head = nn.Linear(cfg.d_model, n_tags)

    def forward(self, ids):
        return self.head(self.backbone(ids=ids, return_hidden=True)["hidden"])


def build_mamba(seed=0):
    torch.manual_seed(seed)
    return TinyMambaTagger(mamba_cfg()).eval()


def test_prefix_on_mamba_raises_not_implemented():
    model = build_mamba()
    with pytest.raises(NotImplementedError) as e:
        apply_peft(model, "prefix")
    msg = str(e.value)
    assert "mamba" in msg.lower(), msg
    assert "MambaBlock" in msg, msg
    assert "attention" in msg.lower(), msg


    assert all(p.requires_grad for p in model.parameters())
    apply_peft(model, "adapter")
    out = model(ids())
    assert out.shape == (BATCH, SEQ, N_TAGS) and torch.isfinite(out).all()


@pytest.mark.parametrize("method", [m for m in METHODS if m != "prefix"])
def test_other_methods_work_on_mamba(method):
    x = ids()
    ref = build_mamba()(x)


    kw = ({**DEFAULT_KW[method], "targets": ("in_proj", "out_proj")}
          if method in ("lora", "qlora") else DEFAULT_KW[method])
    model = apply_peft(build_mamba(), method, **kw)
    out = model(x)
    assert out.shape == ref.shape and torch.isfinite(out).all()
    rep = trainable_report(model)
    assert (rep["trainable"] == rep["total"]) if method == "full" else \
        (0 < rep["trainable"] < rep["total"])
    if method in ("lora", "adapter"):
        assert torch.allclose(out, ref, atol=1e-6)
    if method == "qlora":


        assert not torch.allclose(out, ref, atol=1e-6), \
            "qlora left the function unchanged -- quantisation did not take effect"


def test_tied_weights_counted_once():
    model = build()
    assert model.backbone.mlm_head.weight is model.backbone.stack.embed_tokens.weight
    rep = trainable_report(model)
    deduped = sum(p.numel() for p in model.parameters())
    raw = sum(p.numel() for _, p in model.named_parameters(remove_duplicate=False))
    assert rep["total"] == deduped < raw
    assert raw - deduped == model.backbone.stack.embed_tokens.weight.numel()


def test_head_only_trains_exactly_the_head():
    model = apply_peft(build(), "head_only")
    rep = trainable_report(model)
    assert set(rep["groups"]) == {"head"}
    assert rep["trainable"] == sum(p.numel() for p in model.head.parameters())

    assert not model.backbone.mlm_head.weight.requires_grad


def test_prefix_changes_the_attention_pattern():
    x = ids()
    base = build()(x)
    model = apply_peft(build(), "prefix", n_prefix=8)
    out = model(x)
    assert torch.isfinite(out).all()
    assert not torch.allclose(base, out, atol=0, rtol=0)
    with torch.no_grad():
        for m in model.modules():
            if hasattr(m, "k") and hasattr(m, "v") and isinstance(m, nn.Module) \
                    and type(m).__name__ == "PrefixKV":
                nn.init.normal_(m.v, 0.0, 0.5)
        assert not torch.allclose(out, model(x))


def test_prefix_rejects_kv_cache():
    cfg = tiny_cfg(arch="decoder")
    torch.manual_seed(0)
    from backbone.decoder import DecoderLM
    dec = DecoderLM(cfg).eval()
    with pytest.warns(RuntimeWarning, match="no parameter matched the head"):
        apply_peft(dec, "prefix")
    assert torch.isfinite(dec(ids=ids())["logits"]).all()
    with pytest.raises(RuntimeError, match="KV cache"):
        dec(ids=ids(), use_cache=True)


def test_fresh_phase2_capacity_is_always_trainable():

    class CRFish(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.transitions = nn.Parameter(torch.zeros(n, n))

    class Realistic(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.backbone = EncoderMLM(cfg)
            self.head_dropout = nn.Dropout(0.0)
            self.head_proj = nn.Linear(cfg.d_model, N_TAGS)
            self.crf = CRFish(N_TAGS)

        def forward(self, x):
            return self.head_proj(self.head_dropout(
                self.backbone(ids=x, return_hidden=True)["hidden"]))

    for method in METHODS:
        torch.manual_seed(0)
        model = apply_peft(Realistic(tiny_cfg()).eval(), method, **DEFAULT_KW[method])
        named = dict(model.named_parameters())
        assert named["head_proj.weight"].requires_grad, method
        assert named["head_proj.bias"].requires_grad, method
        assert named["crf.transitions"].requires_grad, method
        assert not named["backbone.stack.embed_tokens.weight"].requires_grad or \
            method == "full", method
        assert set(trainable_report(model)["groups"]) >= {"head_proj", "crf"}, method


def test_describe_is_a_single_informative_line():
    for method in METHODS:
        s = describe(method)
        assert "\n" not in s and 20 < len(s) < 400
    assert "rank-4" in describe("lora", r=4)
    assert "top 2" in describe("last_k", k=2)
    assert "32 learned" in describe("prefix", n_prefix=32)
    with pytest.raises(ValueError):
        describe("nope")


@pytest.mark.parametrize("method", METHODS)
def test_injected_modules_land_on_the_models_device(method):
    model = build().to("meta")
    apply_peft(model, method, **DEFAULT_KW[method])
    off = [n for n, p in model.named_parameters() if p.device.type != "meta"]
    assert not off, f"{method} left {len(off)} parameter(s) off the model's device: {off[:4]}"

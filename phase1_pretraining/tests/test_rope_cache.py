import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
import torch

from config import Config
from models import build

ATOL = 2e-4


def tiny(arch="decoder", **kw):
    return Config(name="t", arch=arch, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
                  vocab_size=256, max_seq_len=64, block_size=64, norm="rms",
                  pos="rope", act="swiglu", dropout=0.0, grad_checkpoint=False,
                  **kw)


def _uncached(model, ids):
    out = []
    for t in range(1, ids.shape[1] + 1):
        out.append(model(ids[:, :t])["logits"][:, -1])
    return torch.stack(out, dim=1)


def _cached(model, ids, prefill):
    res = model(ids[:, :prefill], use_cache=True)
    past = res["past_kvs"]
    out = [res["logits"][:, -1]]
    for t in range(prefill, ids.shape[1] - 1):
        res = model(ids[:, t:t + 1], past_kvs=past, use_cache=True)
        past = res["past_kvs"]
        out.append(res["logits"][:, -1])
    return torch.stack(out, dim=1)


@pytest.mark.parametrize("arch", ["decoder", "efficient"])
@pytest.mark.parametrize("prefill", [1, 5, 12])
def test_cached_equals_uncached(arch, prefill):
    torch.manual_seed(0)
    cfg = tiny(arch) if arch == "decoder" else tiny(arch, window=8, n_global=2)
    model = build(cfg).eval()
    ids = torch.randint(10, 250, (2, 16))

    with torch.no_grad():
        ref = _uncached(model, ids)[:, prefill - 1:-1]
        got = _cached(model, ids, prefill)

    d = (ref - got).abs().max().item()
    agree = (ref.argmax(-1) == got.argmax(-1)).float().mean().item()
    assert d < ATOL, (
        f"{arch} prefill={prefill}: cached decoding differs from uncached by {d:.3e} "
        f"(argmax agreement {agree:.4f}). RoPE is being applied at the wrong position "
        f"to cached keys.")


def test_rope_angles_advance_with_position():
    torch.manual_seed(0)
    cfg = tiny()
    model = build(cfg).eval()
    attn = model.stack.blocks[0].attn
    tokenA = torch.randint(10, 250, (1, 1))

    with torch.no_grad():
        h = model.stack.embed_tokens(tokenA)

        _, kv0 = attn(h, use_cache=True, offset=0)
        fake_past = (torch.zeros(1, attn.nkv, 7, attn.hd),
                     torch.zeros(1, attn.nkv, 7, attn.hd))
        _, kv7 = attn(h, past_kv=fake_past, use_cache=True, offset=7)

    k0 = kv0[0][:, :, -1]
    k7 = kv7[0][:, :, -1]
    assert not torch.allclose(k0, k7, atol=1e-6), (
        "a token at position 7 produced the same key as at position 0 -- the cached "
        "path is rotating new keys at offset 0 regardless of where they actually sit")


def test_prefill_alone_is_correct():
    torch.manual_seed(0)
    model = build(tiny()).eval()
    ids = torch.randint(10, 250, (2, 16))
    with torch.no_grad():
        full = model(ids)["logits"]
        step = _uncached(model, ids)
    assert (full - step).abs().max() < ATOL

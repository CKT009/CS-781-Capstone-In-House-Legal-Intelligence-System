import sys, pathlib, math

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
import torch

from config import Config
from models import ARCHS
from loss import diffusion_loss, _sample_t, MASK_ID
from generate import generate_diffusion, _keep_after


def tiny(**kw):
    base = dict(name="tiny_diffusion", arch="diffusion", d_model=128, n_layers=2,
                n_dec_layers=2, n_heads=4, n_kv_heads=4, norm="rms", pos="rope",
                act="swiglu", vocab_size=1000, max_seq_len=128, block_size=128,
                batch_size=4, grad_accum=1, grad_checkpoint=False,
                diff_target_len=32, diff_cond_dim=64)
    return Config(**{**base, **kw})


def model_of(cfg):
    torch.manual_seed(0)
    return ARCHS[cfg.arch](cfg).eval()


@pytest.mark.parametrize("schedule", ["cosine", "linear"])
def test_schedule_reaches_zero_and_is_monotone(schedule):
    for S in (1, 8, 64, 200):
        k = [_keep_after(128, s, S, schedule) for s in range(S)]
        assert k[-1] == 0, f"{schedule} steps={S} leaves {k[-1]} positions masked"
        assert all(a >= b for a, b in zip(k, k[1:])), "schedule not monotone"


def test_stratified_t_covers_the_unit_interval():
    t = _sample_t(64, tiny(), torch.device("cpu"))
    assert t.min() < 0.05 and t.max() > 0.95
    assert ((t[1:] - t[:-1]) > 0).all(), "stratified grid should be sorted"
    assert (t > 0).all() and (t <= 1.0).all()


def test_prompt_is_never_masked():
    cfg = tiny()
    seen = {}

    class Spy(torch.nn.Module):
        cfg_ = cfg

        def __init__(self):
            super().__init__()
            self.cfg = cfg

        def forward(self, ids=None, dec_ids=None, t=None, **kw):
            seen["prompt"], seen["target"], seen["t"] = ids, dec_ids, t
            return {"logits": torch.zeros(dec_ids.shape[0], dec_ids.shape[1],
                                          cfg.vocab_size)}

    x = torch.randint(10, cfg.vocab_size, (8, cfg.block_size))
    diffusion_loss(Spy(), x, cfg)

    S = cfg.block_size - cfg.diff_target_len
    assert seen["prompt"].shape == (8, S)
    assert seen["target"].shape == (8, cfg.diff_target_len)
    assert torch.equal(seen["prompt"], x[:, :S]), "prompt was modified"
    assert (seen["prompt"] != MASK_ID).all(), "prompt contains <mask>"

    clean = x[:, S:]
    assert ((seen["target"] == clean) | (seen["target"] == MASK_ID)).all(), \
        "corruption is not pure absorbing -- a BERT-style 10%-random crept in"
    assert seen["t"].shape == (8,), "the model must be conditioned on the noise level"


def test_mask_rate_tracks_t():
    cfg = tiny(diff_stratified=False)
    rates, ts = [], []

    class Spy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = cfg

        def forward(self, ids=None, dec_ids=None, t=None, **kw):
            rates.append((dec_ids == MASK_ID).float().mean(1))
            ts.append(t)
            return {"logits": torch.zeros(*dec_ids.shape, cfg.vocab_size)}

    m = Spy()
    for _ in range(40):
        diffusion_loss(m, torch.randint(10, cfg.vocab_size, (16, cfg.block_size)), cfg)
    r, t = torch.cat(rates), torch.cat(ts)
    assert (r - t).abs().mean() < 0.10, "mask rate does not track the drawn t"
    assert r.max() > 0.9 and r.min() < 0.1, "t is not spanning [0,1]"


@pytest.mark.parametrize("weighting", ["uniform", "nelbo"])
def test_loss_is_finite_and_near_uniform_at_init(weighting):
    cfg = tiny(diff_weighting=weighting)
    model = model_of(cfg)
    x = torch.randint(10, cfg.vocab_size, (16, cfg.block_size))
    loss, ntok = diffusion_loss(model, x, cfg)
    assert torch.isfinite(loss), f"{weighting} loss is {loss}"
    assert ntok > 0

    assert 0.3 * math.log(cfg.vocab_size) < loss.item() < 3.0 * math.log(cfg.vocab_size)


def test_loss_accepts_the_sampler_tuple():
    cfg = tiny()
    model = model_of(cfg)
    x = torch.randint(10, cfg.vocab_size, (4, cfg.block_size))
    y = torch.roll(x, -1, dims=1)
    a, _ = diffusion_loss(model, (x, y), cfg)
    b, _ = diffusion_loss(model, x, cfg)
    assert torch.isfinite(a) and torch.isfinite(b)


def test_target_longer_than_block_is_rejected():
    with pytest.raises(ValueError, match="no prompt"):
        cfg = tiny(diff_target_len=128)
        diffusion_loss(model_of(cfg),
                       torch.randint(10, 999, (2, cfg.block_size)), cfg)


def test_gradients_reach_the_encoder():
    cfg = tiny()
    torch.manual_seed(0)
    model = ARCHS[cfg.arch](cfg).train()
    for blk in model.blocks:
        torch.nn.init.normal_(blk.cross_attn.o_proj.weight, 0.0, 0.02)
        torch.nn.init.normal_(blk.adaLN.weight, 0.0, 0.02)
    loss, _ = diffusion_loss(model, torch.randint(10, 999, (4, cfg.block_size)), cfg)
    loss.backward()
    g = model.encoder.blocks[0].attn.q_proj.weight.grad
    assert g is not None and torch.isfinite(g).all() and (g != 0).any()


def test_identity_at_init():
    cfg = tiny()
    model = model_of(cfg)
    dec = torch.randint(10, cfg.vocab_size, (2, cfg.diff_target_len))
    S = cfg.block_size - cfg.diff_target_len
    a = model(ids=torch.randint(10, 999, (2, S)), dec_ids=dec,
              t=torch.full((2,), 0.9))["logits"]
    b = model(ids=torch.randint(10, 999, (2, S)), dec_ids=dec,
              t=torch.full((2,), 0.1))["logits"]
    assert torch.allclose(a, b, atol=1e-5), \
        "decoder is not the identity at init (adaLN-Zero broken)"


def test_conditioning_is_wired_up():
    cfg = tiny()
    torch.manual_seed(0)
    model = ARCHS[cfg.arch](cfg).eval()
    for blk in model.blocks:
        torch.nn.init.normal_(blk.adaLN.weight, 0.0, 0.05)
    dec = torch.randint(10, cfg.vocab_size, (2, cfg.diff_target_len))
    src = torch.randint(10, 999, (2, cfg.block_size - cfg.diff_target_len))
    with torch.no_grad():
        a = model(ids=src, dec_ids=dec, t=torch.full((2,), 0.9))["logits"]
        b = model(ids=src, dec_ids=dec, t=torch.full((2,), 0.1))["logits"]
    assert not torch.allclose(a, b, atol=1e-4), "t never reaches the blocks"


def test_prompt_changes_the_output():
    cfg = tiny()
    torch.manual_seed(0)
    model = ARCHS[cfg.arch](cfg).eval()
    for blk in model.blocks:
        torch.nn.init.normal_(blk.cross_attn.o_proj.weight, 0.0, 0.05)
    dec = torch.randint(10, cfg.vocab_size, (2, cfg.diff_target_len))
    S = cfg.block_size - cfg.diff_target_len
    t = torch.full((2,), 0.5)
    with torch.no_grad():
        a = model(ids=torch.randint(10, 500, (2, S)), dec_ids=dec, t=t)["logits"]
        b = model(ids=torch.randint(500, 999, (2, S)), dec_ids=dec, t=t)["logits"]
    assert not torch.allclose(a, b, atol=1e-4), "the prompt does not reach the decoder"


@pytest.mark.parametrize("steps", [1, 4, 16, 64])
def test_sampler_fills_every_position(steps):
    cfg = tiny(diff_steps=steps)
    model = model_of(cfg)
    ctxs = [list(range(10, 40)), list(range(10, 25))]
    out = generate_diffusion(model, ctxs, steps=steps, greedy=True)
    assert len(out) == 2
    for o in out:
        assert len(o) == cfg.diff_target_len, "wrong output length"
        assert MASK_ID not in o, "sampler committed <mask> -- ban_ids not applied"


def test_structural_tokens_are_never_emitted():
    cfg = tiny()
    model = model_of(cfg)
    banned = (0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
    out = generate_diffusion(model, [list(range(10, 60))], steps=8, greedy=False)
    assert not (set(out[0]) & set(banned)), \
        f"emitted structural tokens: {sorted(set(out[0]) & set(banned))}"


def test_ban_can_be_disabled():
    cfg = tiny()
    model = model_of(cfg)
    out = generate_diffusion(model, [list(range(10, 60))], steps=4, greedy=True,
                             ban_ids=())
    assert len(out[0]) == cfg.diff_target_len


def test_sampler_output_length_is_exact():
    cfg = tiny()
    model = model_of(cfg)
    for n in (8, 32, 64):
        out = generate_diffusion(model, [list(range(10, 50))], max_new_tokens=n,
                                 steps=7, greedy=True)
        assert len(out[0]) == n


def test_sampler_ignores_autoregressive_decode_flags():
    cfg = tiny()
    model = model_of(cfg)
    out = generate_diffusion(model, [list(range(10, 50))], steps=4, greedy=True,
                             repetition_penalty=1.15, no_repeat_ngram=4)
    assert len(out[0]) == cfg.diff_target_len


def test_batching_does_not_change_a_sequence():
    cfg = tiny()
    model = model_of(cfg)
    a, b = list(range(10, 60)), list(range(200, 250))
    torch.manual_seed(1)
    both = generate_diffusion(model, [a, b], steps=8, greedy=True, gumbel=0.0)
    torch.manual_seed(1)
    solo_a = generate_diffusion(model, [a], steps=8, greedy=True, gumbel=0.0)
    assert both[0] == solo_a[0], "batching changed the output of sequence 0"

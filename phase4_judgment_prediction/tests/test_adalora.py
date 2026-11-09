import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
import torch.nn as nn

from cjpe.adalora import SVDLinear, apply_adalora, budget_schedule


class Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj, self.k_proj = nn.Linear(d, d), nn.Linear(d, d)
        self.v_proj, self.o_proj = nn.Linear(d, d), nn.Linear(d, d)

    def forward(self, x):
        a = torch.softmax(self.q_proj(x) @ self.k_proj(x).transpose(-1, -2), -1)
        return x + self.o_proj(a @ self.v_proj(x))


class Tiny(nn.Module):
    def __init__(self, d=16, n=3):
        super().__init__()
        self.backbone = nn.ModuleList([Attn(d) for _ in range(n)])
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        for b in self.backbone:
            x = b(x)
        return self.head(x.mean(1)).squeeze(-1)


def make(seed=0, **kw):
    torch.manual_seed(seed)
    m = Tiny()
    apply_adalora(m, verbose=False, **kw)
    return m


def test_wrapped_equals_base_at_init():
    torch.manual_seed(0)
    base = nn.Linear(16, 8)
    w = SVDLinear(base, r=4, alpha=8)
    x = torch.randn(5, 16)
    assert torch.equal(w(x), base(x))
    torch.manual_seed(0)
    ref = Tiny(); x = torch.randn(2, 7, 16)
    torch.manual_seed(0)
    m = Tiny(); apply_adalora(m, r=4, target_r=2, verbose=False)
    assert torch.allclose(m(x), ref(x))


def test_freezing_and_head():
    m = make(r=4, target_r=2)
    for n, p in m.named_parameters():
        trainable = any(part.startswith("head") for part in n.split(".")) \
            or n.split(".")[-1] in ("lora_A", "lora_B", "lora_E")
        assert p.requires_grad == trainable, n
    assert sum(isinstance(x, SVDLinear) for x in m.modules()) == 12
    with pytest.raises(RuntimeError):
        apply_adalora(m)


def test_budget_schedule():
    T, b0, bT = 1000, 864, 432
    assert budget_schedule(0, T, b0, bT, 0.1, 0.3) == (b0, "warmup")
    assert budget_schedule(99, T, b0, bT, 0.1, 0.3)[0] == b0
    assert budget_schedule(100, T, b0, bT, 0.1, 0.3)[0] == b0
    assert budget_schedule(T - 1, T, b0, bT, 0.1, 0.3) == (bT, "final")
    seq = [budget_schedule(t, T, b0, bT, 0.1, 0.3)[0] for t in range(T)]
    assert all(a >= b for a, b in zip(seq, seq[1:]))
    assert seq[-1] == bT and min(seq) == bT


def _train(m, total, lr=1e-2):
    ada = m._adalora
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr)
    torch.manual_seed(1)
    x, y = torch.randn(8, 5, 16), torch.randint(0, 2, (8,)).float()
    for step in range(total):
        loss = nn.functional.binary_cross_entropy_with_logits(m(x), y) + ada.orth_loss()
        loss.backward()
        opt.step()
        ada.update_and_mask(step, total)
        if step >= int(0.7 * total) + 1:
            for s in (x for x in m.modules() if isinstance(x, SVDLinear)):
                assert torch.all(s.lora_E[~s.rank_mask] == 0)
        opt.zero_grad(set_to_none=True)
    return ada


def test_budget_reaches_target_and_masked_stay_zero():
    m = make(r=4, target_r=2)
    ada = _train(m, total=60)
    assert ada.budget == ada.bT == 2 * 12
    assert ada.kept() == 24
    frozen = [k.clone() for k in ada.frozen_mask]
    for s, k in zip((x for x in m.modules() if isinstance(x, SVDLinear)), frozen):
        assert torch.equal(s.rank_mask, k)
        assert torch.all(s.lora_E[~k] == 0)
    kept_E = torch.cat([s.lora_E[s.rank_mask] for s in m.modules() if isinstance(s, SVDLinear)])
    assert kept_E.numel() == 24 and (kept_E != 0).all()


def test_orth_reg_zero_for_orthonormal():
    m = make(r=4, target_r=2)
    with torch.no_grad():
        for s in (x for x in m.modules() if isinstance(x, SVDLinear)):
            q, _ = torch.linalg.qr(torch.randn(16, 4))
            s.lora_B.copy_(q)
            q2, _ = torch.linalg.qr(torch.randn(16, 4))
            s.lora_A.copy_(q2.T)
    assert m._adalora.orth_loss().item() < 1e-10
    m2 = make(r=4, target_r=2)
    assert m2._adalora.orth_loss().item() > 0


def test_state_dict_round_trip():
    m = make(r=4, target_r=2)
    _train(m, total=40)
    sd = m.state_dict()
    assert any(k.endswith("lora_E") for k in sd) and any(k.endswith("rank_mask") for k in sd)
    m2 = make(seed=123, r=4, target_r=2)
    m2.load_state_dict(sd)
    x = torch.randn(3, 6, 16)
    m.eval(); m2.eval()
    assert torch.equal(m(x), m2(x))
    for a, b in zip((x for x in m.modules() if isinstance(x, SVDLinear)),
                    (x for x in m2.modules() if isinstance(x, SVDLinear))):
        assert torch.equal(a.rank_mask, b.rank_mask)
        assert torch.all(b.lora_E[~b.rank_mask] == 0)

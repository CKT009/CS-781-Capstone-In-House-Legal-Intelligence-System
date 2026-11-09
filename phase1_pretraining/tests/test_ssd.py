import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from models.ssd import ssd_chunked, ssd_naive, ssd_step, segsum

D = torch.float64


def rand(l, b=2, h=4, p=16, n=8, seed=0):
    torch.manual_seed(seed)
    return (torch.randn(b, l, h, p, dtype=D),
            F.softplus(torch.randn(b, l, h, dtype=D)),
            -torch.rand(h, dtype=D) * 2 - 0.05,
            torch.randn(b, l, n, dtype=D),
            torch.randn(b, l, n, dtype=D))


def test_segsum_shape_and_values():
    s = segsum(torch.tensor([1., 2., 3., 4.]))
    assert (s.diag() == 0).all()
    assert s[2, 0] == 5.0
    assert torch.isinf(s[0, 2]) and s[0, 2] < 0


def test_matches_recurrence_at_every_chunk_size():
    x, dt, A, B, C = rand(256)
    yn, sn = ssd_naive(x, dt, A, B, C)
    for ck in (16, 32, 64, 128, 256):
        yc, sc = ssd_chunked(x, dt, A, B, C, chunk=ck)
        assert (yc - yn).abs().max() < 1e-10, ck
        assert (sc - sn).abs().max() < 1e-10, ck


def test_ragged_lengths():
    for l in (1, 7, 63, 64, 65, 100, 129):
        x, dt, A, B, C = rand(l)
        yn, sn = ssd_naive(x, dt, A, B, C)
        yc, sc = ssd_chunked(x, dt, A, B, C, chunk=64)
        assert yc.shape == yn.shape == (2, l, 4, 16)
        assert (yc - yn).abs().max() < 1e-10, l
        assert (sc - sn).abs().max() < 1e-10, l


def test_initial_state_is_honoured():
    x, dt, A, B, C = rand(256)
    s0 = torch.randn(2, 4, 16, 8, dtype=D)
    yn, sn = ssd_naive(x, dt, A, B, C, state=s0)
    yc, sc = ssd_chunked(x, dt, A, B, C, chunk=64, state=s0)
    assert (yc - yn).abs().max() < 1e-10
    assert (sc - sn).abs().max() < 1e-10


def test_split_equals_whole():
    x, dt, A, B, C = rand(256)
    whole, _ = ssd_chunked(x, dt, A, B, C, chunk=64)
    out, s = [], None
    for a, b in [(0, 64), (64, 160), (160, 256)]:
        y, s = ssd_chunked(x[:, a:b], dt[:, a:b], A, B[:, a:b], C[:, a:b], chunk=64, state=s)
        out.append(y)
    assert (torch.cat(out, 1) - whole).abs().max() < 1e-10


def test_step_equals_chunked():
    x, dt, A, B, C = rand(128)
    y, ref_state = ssd_chunked(x, dt, A, B, C, chunk=32)
    s, ys = torch.zeros(2, 4, 16, 8, dtype=D), []
    for t in range(128):
        yt, s = ssd_step(x[:, t], dt[:, t], A, B[:, t], C[:, t], s)
        ys.append(yt)
    assert (torch.stack(ys, 1) - y).abs().max() < 1e-10
    assert (s - ref_state).abs().max() < 1e-10


def test_causality():
    x, dt, A, B, C = rand(128)
    y0, _ = ssd_chunked(x, dt, A, B, C, chunk=32)
    t0 = 70
    x2 = x.clone(); x2[:, t0] += 5.0
    y1, _ = ssd_chunked(x2, dt, A, B, C, chunk=32)
    d = (y1 - y0).abs().amax(dim=(0, 2, 3))
    assert d[:t0].max() == 0.0
    assert d[t0] > 1e-3


def test_stable_where_naive_factorisation_overflows():
    torch.manual_seed(0)
    x = torch.randn(1, 1024, 4, 16)
    dt = F.softplus(torch.randn(1, 1024, 4)) * 4
    A = -torch.rand(4) * 4 - 1.0
    B, C = torch.randn(1, 1024, 8), torch.randn(1, 1024, 8)
    y, s = ssd_chunked(x, dt, A, B, C, chunk=64)
    assert torch.isfinite(y).all() and torch.isfinite(s).all()
    assert float((dt * A).sum(1).min()) < -1000


def test_gradients_flow():
    x, dt, A, B, C = rand(128)
    x = x.float().requires_grad_(True)
    y, _ = ssd_chunked(x, dt.float(), A.float(), B.float(), C.float(), chunk=32)
    y.sum().backward()
    assert torch.isfinite(x.grad).all() and (x.grad != 0).any()

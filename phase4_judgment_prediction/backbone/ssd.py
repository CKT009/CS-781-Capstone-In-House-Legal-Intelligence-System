import torch


def segsum(x):
    L = x.shape[-1]
    cum = x.cumsum(dim=-1)
    out = cum[..., :, None] - cum[..., None, :]
    mask = torch.ones(L, L, dtype=torch.bool, device=x.device).tril()
    return out.masked_fill(~mask, -float("inf"))


def ssd_chunked(x, dt, A, B, C, chunk=64, state=None):
    b, l0, h, p = x.shape
    n = B.shape[-1]


    pad = (-l0) % chunk
    if pad:
        x = torch.cat([x, x.new_zeros(b, pad, h, p)], 1)
        dt = torch.cat([dt, dt.new_zeros(b, pad, h)], 1)
        B = torch.cat([B, B.new_zeros(b, pad, n)], 1)
        C = torch.cat([C, C.new_zeros(b, pad, n)], 1)
    l = l0 + pad
    nc = l // chunk

    xdt = (x * dt[..., None]).reshape(b, nc, chunk, h, p)
    Bc = B.reshape(b, nc, chunk, n)
    Cc = C.reshape(b, nc, chunk, n)
    dA = (dt * A).reshape(b, nc, chunk, h)


    decay = torch.exp(segsum(dA.permute(0, 1, 3, 2)))
    CB = torch.einsum("bcln,bcsn->bcls", Cc, Bc)
    y = torch.einsum("bchls,bcshp->bclhp", CB[:, :, None] * decay, xdt)


    cum = dA.cumsum(dim=2)
    to_end = torch.exp(cum[:, :, -1:, :] - cum)
    chunk_states = torch.einsum("bcshp,bcsn->bchpn", xdt * to_end[..., None], Bc)


    if state is None:
        state = xdt.new_zeros(b, h, p, n)
    total = cum[:, :, -1, :]
    total_ext = torch.cat([total.new_zeros(b, 1, h), total], dim=1)
    states_ext = torch.cat([state[:, None], chunk_states], dim=1)
    carry = torch.exp(segsum(total_ext.permute(0, 2, 1)))
    out = torch.einsum("bhij,bjhpn->bihpn", carry, states_ext)
    entering, state = out[:, :nc], out[:, nc]


    y = y + torch.einsum("bcln,bchpn->bclhp", Cc, entering) * torch.exp(cum)[..., None]
    return y.reshape(b, l, h, p)[:, :l0], state


def ssd_naive(x, dt, A, B, C, state=None):
    b, l, h, p = x.shape
    n = B.shape[-1]
    s = x.new_zeros(b, h, p, n) if state is None else state.clone()
    ys = []
    for t in range(l):
        decay = torch.exp(dt[:, t] * A)
        s = decay[..., None, None] * s + torch.einsum(
            "bhp,bn->bhpn", x[:, t] * dt[:, t][..., None], B[:, t])
        ys.append(torch.einsum("bhpn,bn->bhp", s, C[:, t]))
    return torch.stack(ys, dim=1), s


def ssd_step(x_t, dt_t, A, B_t, C_t, state):
    decay = torch.exp(dt_t * A)
    state = decay[..., None, None] * state + torch.einsum(
        "bhp,bn->bhpn", x_t * dt_t[..., None], B_t)
    return torch.einsum("bhpn,bn->bhp", state, C_t), state

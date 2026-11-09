import torch
import torch.nn.functional as F
from dataset import MASK, PAD


def causal_lm_loss(model, batch, cfg=None):
    x, y = batch
    logits = model(x)["logits"]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1),
                           ignore_index=-100)
    return loss, x.numel()


def mask_tokens(x, vocab_size, mlm_prob=0.15, mask_id=MASK, generator=None):
    labels = x.clone()
    prob = torch.full(x.shape, mlm_prob, device=x.device)
    chosen = torch.bernoulli(prob, generator=generator).bool() & (x != PAD)
    labels[~chosen] = -100

    to_mask = torch.bernoulli(torch.full(x.shape, 0.8, device=x.device),
                              generator=generator).bool() & chosen
    x = x.clone()
    x[to_mask] = mask_id
    to_rand = (torch.bernoulli(torch.full(x.shape, 0.5, device=x.device),
                               generator=generator).bool() & chosen & ~to_mask)
    x[to_rand] = torch.randint(vocab_size, (int(to_rand.sum()),), device=x.device,
                               generator=generator)
    return x, labels


def mlm_loss(model, batch, cfg):
    x, _ = batch
    corrupted, labels = mask_tokens(x, cfg.vocab_size, cfg.mlm_prob)
    logits = model(corrupted)["logits"]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1),
                           ignore_index=-100)
    return loss, int((labels != -100).sum())


def span_corrupt(x, rate=0.15, mean_span=3, sentinel_start=7, n_sentinels=8):
    B, T = x.shape
    n_spans = max(1, int(T * rate / mean_span))
    src_rows, tgt_rows = [], []
    for b in range(B):
        starts = torch.randperm(max(T - mean_span, 1), device=x.device)[:n_spans].sort().values
        taken, src, tgt, prev, s_id = set(), [], [], 0, 0
        for st in starts.tolist():
            if st < prev or s_id >= n_sentinels:
                continue
            end = min(st + mean_span, T)
            src.append(x[b, prev:st])
            src.append(torch.tensor([sentinel_start + s_id], device=x.device))
            tgt.append(torch.tensor([sentinel_start + s_id], device=x.device))
            tgt.append(x[b, st:end])
            prev, s_id = end, s_id + 1
        src.append(x[b, prev:])
        src_rows.append(torch.cat(src)[:T])
        tgt_rows.append(torch.cat(tgt)[:T] if tgt else x[b, :1])
    pad = lambda rows: torch.stack([F.pad(r, (0, max(len(q) for q in rows) - len(r)), value=PAD)
                                    for r in rows])
    return pad(src_rows), pad(tgt_rows)


def span_corruption_loss(model, batch, cfg):
    from models import shift_right
    x, _ = batch
    src, tgt = span_corrupt(x, cfg.span_rate, cfg.mean_span)
    labels = tgt.masked_fill(tgt == PAD, -100)
    logits = model(ids=src, dec_ids=shift_right(labels))["logits"]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1),
                           ignore_index=-100)
    return loss, int((labels != -100).sum())


def kd_loss(student_logits, teacher_logits, temperature=2.0):
    T = temperature
    s = F.log_softmax(student_logits.float() / T, -1)
    t = F.softmax(teacher_logits.float() / T, -1)
    return F.kl_div(s, t, reduction="batchmean") * (T ** 2)


def distill_lm_loss(model, batch, cfg, teacher=None, alpha=0.5, temperature=2.0):
    x, y = batch
    logits = model(x)["logits"]
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
    with torch.no_grad():
        t_logits = teacher(x)["logits"]
    kd = kd_loss(logits, t_logits, temperature)
    return alpha * ce + (1 - alpha) * kd, x.numel()


MASK_ID = MASK


def _ids_of(batch):
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, (tuple, list)):
        return batch[0]
    for k in ("ids", "input_ids", "x", "tokens"):
        if k in batch:
            return batch[k]
    raise KeyError(f"no token ids in batch: {list(batch)}")


def _sample_t(B, cfg, device):
    eps = cfg.diff_eps
    if cfg.diff_stratified:
        u = torch.rand(1, device=device)
        t = (torch.arange(B, device=device) + u) / B
    else:
        t = torch.rand(B, device=device)
    return t.clamp(min=eps, max=1.0)


def diffusion_loss(model, batch, cfg):
    x = _ids_of(batch)
    B, T = x.shape
    L = cfg.diff_target_len
    if L >= T:
        raise ValueError(
            f"diff_target_len={L} leaves no prompt in a block of {T}; "
            f"it must be < block_size")
    prompt, x0 = x[:, :T - L], x[:, T - L:]

    t = _sample_t(B, cfg, x.device)
    m = torch.rand(B, L, device=x.device) < t[:, None]
    xt = torch.where(m, torch.full_like(x0, MASK_ID), x0)

    logits = model(ids=prompt, dec_ids=xt, t=t)["logits"]
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                         x0.reshape(-1), reduction="none").view(B, L)
    ce = ce * m

    if cfg.diff_weighting == "nelbo":
        loss = ((ce.sum(dim=1) / t) / L).mean()
    else:
        loss = ce.sum() / m.sum().clamp(min=1)


    return loss, int(m.sum())


LOSSES = {
    "decoder":   causal_lm_loss,
    "encoder":   mlm_loss,
    "encdec":    span_corruption_loss,
    "efficient": causal_lm_loss,
    "mamba":     causal_lm_loss,
    "diffusion": diffusion_loss,
}


def get_loss(arch):
    return LOSSES[arch]

import torch
import torch.nn.functional as F
import math

def filter_logits(logits, top_k=0, top_p=0.0):
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
        logits = logits.masked_fill(logits < kth, -float("inf"))
    if 0.0 < top_p < 1.0:
        s, idx = torch.sort(logits, descending=True, dim=-1)
        probs = s.softmax(-1)
        drop = probs.cumsum(-1) - probs > top_p
        drop[..., 0] = False
        logits = logits.scatter(-1, idx, s.masked_fill(drop, -float("inf")))
    return logits


def repetition_penalty_(logits, seen, penalty=1.0):
    if penalty == 1.0:
        return logits
    for b in range(seen.size(0)):
        ids = torch.unique(seen[b])
        v = logits[b, ids]
        logits[b, ids] = torch.where(v < 0, v * penalty, v / penalty)
    return logits


def block_ngrams_(logits, seen, n=0):
    if n <= 0 or seen.size(1) < n:
        return logits
    for b in range(seen.size(0)):
        seq = seen[b].tolist()
        prefix = tuple(seq[-(n - 1):]) if n > 1 else ()
        banned = [seq[i + n - 1] for i in range(len(seq) - n + 1)
                  if tuple(seq[i:i + n - 1]) == prefix]
        if banned:
            logits[b, banned] = -float("inf")
    return logits


@torch.no_grad()
def generate(model, ids, max_new_tokens=220, temperature=0.8, top_k=50, top_p=0.92,
             repetition_penalty=1.15, no_repeat_ngram=4, eos_id=3, greedy=False):
    model.eval()
    ctx = getattr(model.cfg, "max_seq_len", 1024)
    cur = ids[:, -ctx:]
    past, out = None, []
    step_in = cur
    for _ in range(max_new_tokens):
        res = model(step_in, past_kvs=past, use_cache=True)
        past = res["past_kvs"]
        logits = res["logits"][:, -1, :].float()
        seen = torch.cat([cur] + ([torch.stack(out, 1)] if out else []), 1)
        logits = repetition_penalty_(logits, seen, repetition_penalty)
        logits = block_ngrams_(logits, seen, no_repeat_ngram)
        if greedy:
            nxt = logits.argmax(-1)
        else:
            nxt = torch.multinomial(
                filter_logits(logits / max(temperature, 1e-6), top_k, top_p).softmax(-1), 1
            ).squeeze(-1)
        out.append(nxt)
        step_in = nxt[:, None]
    return torch.stack(out, 1)


@torch.no_grad()
def generate_mlm(model, contexts, mask_id=4, pad_id=0, max_new_tokens=200,
                 temperature=0.8, top_k=50, top_p=0.97, repetition_penalty=1.0,
                 no_repeat_ngram=4, greedy=False, max_len=None):
    device = next(model.parameters()).device
    max_len = max_len or getattr(model.cfg, "max_seq_len", 1024)
    ctxs = [list(c)[-(max_len - 1):] for c in contexts]
    out = [[] for _ in ctxs]

    for _ in range(max_new_tokens):
        L = min(max(len(c) for c in ctxs) + 1, max_len)
        x = torch.full((len(ctxs), L), pad_id, dtype=torch.long, device=device)
        m = torch.zeros((len(ctxs), L), dtype=torch.long, device=device)
        for i, c in enumerate(ctxs):
            c = c[-(L - 1):]
            x[i, L - 1 - len(c):L - 1] = torch.tensor(c, device=device)
            x[i, L - 1] = mask_id
            m[i, L - 1 - len(c):] = 1
        logits = model(x, pad_mask=m)["logits"][:, -1, :].float()

        for i, c in enumerate(ctxs):
            seen = torch.tensor(c + out[i], device=device)[None]
            logits[i:i + 1] = repetition_penalty_(logits[i:i + 1], seen, repetition_penalty)
            logits[i:i + 1] = block_ngrams_(logits[i:i + 1], seen, no_repeat_ngram)
        if greedy:
            nxt = logits.argmax(-1)
        else:
            nxt = torch.multinomial(
                filter_logits(logits / max(temperature, 1e-6), top_k, top_p).softmax(-1),
                1).squeeze(-1)
        for i, tok in enumerate(nxt.tolist()):
            out[i].append(tok)
            ctxs[i].append(tok)
    return out


@torch.no_grad()
def generate_encdec(model, src_ids, src_mask=None, max_new_tokens=220, temperature=0.8,
                    top_k=50, top_p=0.97, repetition_penalty=1.0, no_repeat_ngram=4,
                    greedy=False, eos_id=3, pad_id=0, bos_id=0,
                    ban_ids=(0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)):
    model.eval()
    B = src_ids.size(0)
    device = src_ids.device
    if src_mask is None:
        src_mask = (src_ids != pad_id).long()

    enc = model.encode(src_ids, src_mask)

    dec = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    past, out = None, []
    done = torch.zeros(B, dtype=torch.bool, device=device)
    step_in = dec
    for _ in range(max_new_tokens):
        res = model(dec_ids=step_in, enc=enc, pad_mask=src_mask,
                    past_kvs=past, use_cache=True)
        past = res["past_kvs"]
        logits = res["logits"][:, -1, :].float()
        if out:
            seen = torch.cat([dec, torch.stack(out, 1)], 1)
        else:
            seen = dec
        logits = repetition_penalty_(logits, seen, repetition_penalty)
        logits = block_ngrams_(logits, seen, no_repeat_ngram)
        if ban_ids:
            logits[:, list(ban_ids)] = -float("inf")
        if greedy:
            nxt = logits.argmax(-1)
        else:
            nxt = torch.multinomial(
                filter_logits(logits / max(temperature, 1e-6), top_k, top_p).softmax(-1),
                1).squeeze(-1)
        if eos_id is not None:
            nxt = torch.where(done, torch.full_like(nxt, pad_id), nxt)
            done |= nxt == eos_id
        out.append(nxt)
        step_in = nxt[:, None]
        if eos_id is not None and bool(done.all()):
            break
    return torch.stack(out, 1) if out else src_ids.new_zeros(B, 0)


def truncate_words(text, n=128):
    return " ".join(text.split()[:n])


def strip_echo(prompt, continuation, n=20):
    tail = " ".join(prompt.split()[-n:]).lower()
    toks = continuation.split()
    return " ".join(toks[n:]) if tail and " ".join(toks[:n]).lower() == tail else continuation


def _keep_after(n, s, S, schedule="cosine"):
    frac = (s + 1) / S
    r = (1.0 - frac) if schedule == "linear" else math.cos(math.pi / 2 * frac)
    return max(0, int(n * r))


@torch.no_grad()
def generate_diffusion(model, contexts, max_new_tokens=None, steps=64,
                       mask_id=4, pad_id=0, temperature=0.8, top_k=50, top_p=0.92,
                       greedy=False, schedule="cosine", gumbel=1.0,
                       ban_ids=(0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
                       **_ignored):
    device = next(model.parameters()).device
    cfg = model.cfg
    L = max_new_tokens or cfg.diff_target_len
    S = cfg.block_size - cfg.diff_target_len

    ctxs = [list(c)[-S:] for c in contexts]
    B = len(ctxs)
    Sm = max(len(c) for c in ctxs)

    src = torch.full((B, Sm), pad_id, dtype=torch.long, device=device)
    pad = torch.zeros((B, Sm), dtype=torch.long, device=device)
    for i, c in enumerate(ctxs):
        src[i, :len(c)] = torch.tensor(c, device=device)
        pad[i, :len(c)] = 1
    enc = model.encode(src, pad)

    x = torch.full((B, L), mask_id, dtype=torch.long, device=device)
    todo = torch.ones((B, L), dtype=torch.bool, device=device)

    for s in range(steps):

        t = todo.float().mean(dim=1).clamp(min=cfg.diff_eps)
        logits = model(dec_ids=x, t=t, enc=enc, pad_mask=pad)["logits"]
        if ban_ids:
            logits[..., list(ban_ids)] = -float("inf")
        keep = _keep_after(L, s, steps, schedule)

        for i in range(B):
            pos = todo[i].nonzero(as_tuple=True)[0]
            if pos.numel() == 0:
                continue
            lg = logits[i, pos].float()
            if greedy:
                p = lg.softmax(-1)
                conf, cand = p.max(-1)
            else:
                p = filter_logits(lg / max(temperature, 1e-6), top_k, top_p).softmax(-1)
                cand = torch.multinomial(p, 1).squeeze(-1)
                conf = p.gather(-1, cand[:, None]).squeeze(-1)

            score = conf.clamp_min(1e-9).log()
            if gumbel > 0:
                u = torch.rand_like(score).clamp_(1e-9, 1 - 1e-9)
                score = score + gumbel * (-torch.log(-torch.log(u)))

            n_unmask = max(1, pos.numel() - keep)
            sel = score.topk(min(n_unmask, pos.numel())).indices
            x[i, pos[sel]] = cand[sel]
            todo[i, pos[sel]] = False

        if not todo.any():
            break

    return [x[i].tolist() for i in range(B)]

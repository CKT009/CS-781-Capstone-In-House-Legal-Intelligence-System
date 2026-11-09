import torch
import torch.nn.functional as F


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

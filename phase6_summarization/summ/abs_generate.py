from __future__ import annotations

from collections import defaultdict

import torch

from summ.decoding import block_ngrams_, filter_logits, repetition_penalty_
from summ.lm import EOS, tokenizer


@torch.no_grad()
def generate_batch(model, ids: torch.Tensor, max_new=460, min_new=200, greedy=True,
                   temperature=0.8, top_k=50, top_p=0.92, repetition_penalty=1.15,
                   no_repeat_ngram=4, use_cache=True, amp=True):
    model.eval()
    B = ids.shape[0]
    past, step_in, out = None, ids, []
    done = torch.zeros(B, dtype=torch.bool, device=ids.device)
    for t in range(max_new):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and ids.is_cuda):
            if use_cache:
                res = model(step_in, past_kvs=past, use_cache=True)
                past = res["past_kvs"]
            else:
                res = model(torch.cat([ids] + ([torch.stack(out, 1)] if out else []), 1))
        logits = res["logits"][:, -1, :].float()
        gen = torch.stack(out, 1) if out else ids[:, :0]
        logits = repetition_penalty_(logits, gen, repetition_penalty)
        logits = block_ngrams_(logits, gen, no_repeat_ngram)
        if t < min_new:
            logits[:, EOS] = -float("inf")
        if greedy:
            nxt = logits.argmax(-1)
        else:
            nxt = torch.multinomial(filter_logits(logits / max(temperature, 1e-6), top_k,
                                                  top_p).softmax(-1), 1).squeeze(-1)
        nxt = torch.where(done, torch.full_like(nxt, EOS), nxt)
        out.append(nxt)
        done |= nxt == EOS
        if done.all():
            break
        step_in = nxt[:, None]
    gen = torch.stack(out, 1).tolist()
    return [g[:g.index(EOS)] if EOS in g else g for g in gen]


def summarise(model, inputs: dict[str, list[int]], device="cuda", batch=16, **kw) -> dict:
    groups = defaultdict(list)
    for k, v in inputs.items():
        groups[len(v)].append(k)
    tok, out = tokenizer(), {}
    for L, keys in sorted(groups.items()):
        for i in range(0, len(keys), batch):
            ks = keys[i:i + batch]
            x = torch.tensor([inputs[k] for k in ks], device=device)
            for k, g in zip(ks, generate_batch(model, x, **kw)):
                out[k] = tok.decode(g).strip() or "."
    return out

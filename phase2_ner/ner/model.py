from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE = -100


class NERTagger(nn.Module):

    def __init__(self, backbone, n_tags: int, dropout: float = 0.1,
                 class_weights: torch.Tensor | None = None, use_crf: bool = False):
        super().__init__()
        self.backbone = backbone
        self.n_tags = n_tags
        self.head_dropout = nn.Dropout(dropout)
        self.head_proj = nn.Linear(backbone.d_model, n_tags)
        nn.init.normal_(self.head_proj.weight, 0.0, 0.02)
        nn.init.zeros_(self.head_proj.bias)


        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(n_tags),
            persistent=True)

        self.crf = CRF(n_tags) if use_crf else None

    @property
    def d_model(self):
        return self.backbone.d_model

    def forward(self, input_ids, attention_mask=None, labels=None):
        h = self.backbone(input_ids, attention_mask)
        logits = self.head_proj(self.head_dropout(h))
        out = {"logits": logits, "hidden": h}

        if labels is not None:
            if self.crf is not None:

                mask = labels != IGNORE
                safe = labels.masked_fill(~mask, 0)
                out["loss"] = -self.crf.log_likelihood(logits, safe, mask)
            else:
                out["loss"] = F.cross_entropy(
                    logits.reshape(-1, self.n_tags).float(),
                    labels.reshape(-1),
                    weight=self.class_weights.float(),
                    ignore_index=IGNORE)
        return out

    @torch.no_grad()
    def predict_tags(self, input_ids, attention_mask=None):
        logits = self(input_ids, attention_mask)["logits"]
        if self.crf is not None:
            mask = attention_mask.bool() if attention_mask is not None else \
                torch.ones_like(input_ids, dtype=torch.bool)
            return self.crf.decode(logits, mask)
        return logits.argmax(-1)


class CRF(nn.Module):

    def __init__(self, n_tags: int):
        super().__init__()
        self.n_tags = n_tags
        self.transitions = nn.Parameter(torch.zeros(n_tags, n_tags))
        self.start = nn.Parameter(torch.zeros(n_tags))
        self.end = nn.Parameter(torch.zeros(n_tags))

    def _score(self, logits, tags, mask):
        B, T, _ = logits.shape
        m = mask.float()
        s = self.start[tags[:, 0]] + logits[:, 0].gather(1, tags[:, :1]).squeeze(1) * m[:, 0]
        for t in range(1, T):
            trans = self.transitions[tags[:, t - 1], tags[:, t]]
            emit = logits[:, t].gather(1, tags[:, t:t + 1]).squeeze(1)
            s = s + (trans + emit) * m[:, t]
        last = mask.long().sum(1) - 1
        s = s + self.end[tags.gather(1, last.clamp(min=0)[:, None]).squeeze(1)]
        return s

    def _logZ(self, logits, mask):
        B, T, K = logits.shape
        a = self.start[None] + logits[:, 0]
        for t in range(1, T):
            nxt = torch.logsumexp(a[:, :, None] + self.transitions[None], dim=1) \
                  + logits[:, t]
            m = mask[:, t][:, None]
            a = torch.where(m, nxt, a)
        return torch.logsumexp(a + self.end[None], dim=1)

    def log_likelihood(self, logits, tags, mask):
        n = mask.float().sum(1).clamp(min=1.0)
        return ((self._score(logits, tags, mask) - self._logZ(logits, mask)) / n).mean()

    @torch.no_grad()
    def decode(self, logits, mask):
        B, T, K = logits.shape
        a = self.start[None] + logits[:, 0]
        back = []
        for t in range(1, T):
            sc = a[:, :, None] + self.transitions[None]
            best, idx = sc.max(dim=1)
            nxt = best + logits[:, t]
            m = mask[:, t][:, None]
            a = torch.where(m, nxt, a)
            back.append(torch.where(m, idx, torch.zeros_like(idx)))
        best_last = (a + self.end[None]).argmax(1)

        out = torch.zeros(B, T, dtype=torch.long, device=logits.device)
        lens = mask.long().sum(1)
        for b in range(B):
            L = int(lens[b])
            cur = int(best_last[b])
            seq = [cur]
            for t in range(L - 2, -1, -1):
                cur = int(back[t][b, cur])
                seq.append(cur)
            out[b, :L] = torch.tensor(seq[::-1], device=logits.device)
        return out


def class_weights_from_counts(counts, n_tags: int, scheme: str = "inv_sqrt",
                              o_id: int = 0, cap: float = 20.0) -> torch.Tensor:
    w = torch.ones(n_tags)
    if scheme == "none":
        return w
    c = torch.tensor([max(float(counts.get(i, 0)), 1.0) for i in range(n_tags)])
    w = 1.0 / (c if scheme == "inv" else c.sqrt())
    w = w / w[o_id]
    return w.clamp(max=cap)

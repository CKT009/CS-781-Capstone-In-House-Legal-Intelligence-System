from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE = -100


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
        return s + self.end[tags.gather(1, last.clamp(min=0)[:, None]).squeeze(1)]

    def _logZ(self, logits, mask):
        B, T, K = logits.shape
        a = self.start[None] + logits[:, 0]
        for t in range(1, T):
            nxt = torch.logsumexp(a[:, :, None] + self.transitions[None], dim=1) + logits[:, t]
            a = torch.where(mask[:, t][:, None], nxt, a)
        return torch.logsumexp(a + self.end[None], dim=1)

    def log_likelihood(self, logits, tags, mask):
        n = mask.float().sum(1).clamp(min=1.0)
        return ((self._score(logits, tags, mask) - self._logZ(logits, mask)) / n).mean()

    @torch.no_grad()
    def marginals(self, logits, mask):
        logits = logits.float()
        B, T, K = logits.shape
        mask = mask.bool()
        trans = self.transitions.float()[None]
        end = self.end.float()[None]

        alphas = [self.start.float()[None] + logits[:, 0]]
        for t in range(1, T):
            nxt = torch.logsumexp(alphas[-1][:, :, None] + trans, dim=1) + logits[:, t]
            alphas.append(torch.where(mask[:, t][:, None], nxt, alphas[-1]))

        betas = [None] * T
        betas[T - 1] = end.expand(B, K)
        for t in range(T - 2, -1, -1):
            nxt = torch.logsumexp(trans + (logits[:, t + 1] + betas[t + 1])[:, None, :],
                                  dim=2)
            betas[t] = torch.where(mask[:, t + 1][:, None], nxt, end.expand(B, K))
        a = torch.stack(alphas, 1)
        b = torch.stack(betas, 1)
        logZ = torch.logsumexp(alphas[-1] + end, dim=1)
        p = torch.exp(a + b - logZ[:, None, None])
        return p * mask[:, :, None].float()

    @torch.no_grad()
    def decode(self, logits, mask):
        B, T, K = logits.shape
        a = self.start[None] + logits[:, 0]
        back = []
        for t in range(1, T):
            sc = a[:, :, None] + self.transitions[None]
            best, idx = sc.max(dim=1)
            a = torch.where(mask[:, t][:, None], best + logits[:, t], a)
            back.append(torch.where(mask[:, t][:, None], idx, torch.zeros_like(idx)))
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


class DocEncoder(nn.Module):

    def __init__(self, d_model, n_layers=2, n_heads=8, dropout=0.1, max_sents=512):
        super().__init__()
        self.pos = nn.Embedding(max_sents, d_model)
        nn.init.normal_(self.pos.weight, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.max_sents = max_sents

    def forward(self, x, mask):
        S = x.shape[1]
        if S > self.max_sents:
            raise ValueError(f"document has {S} sentences, DocEncoder was built for "
                             f"{self.max_sents}; raise max_sents or lower max_doc_sents")
        p = self.pos(torch.arange(S, device=x.device))[None]

        return self.enc(x + p, src_key_padding_mask=~mask.bool())


class RRTagger(nn.Module):
    def __init__(self, backbone, n_labels: int, doc_layers: int = 2, doc_heads: int = 8,
                 dropout: float = 0.1, use_crf: bool = True, max_sents: int = 512,
                 sent_batch: int = 64, class_weights=None, aux_ce: float = 0.5,
                 ctx_tokens: int = 0):
        super().__init__()
        self.backbone = backbone


        self.ctx_tokens = int(ctx_tokens)
        self.n_labels = n_labels
        d = backbone.d_model
        self.sent_batch = sent_batch
        self.doc = DocEncoder(d, doc_layers, doc_heads, dropout, max_sents) \
            if doc_layers > 0 else None
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d, n_labels)
        self.crf = CRF(n_labels) if use_crf else None


        self.aux_ce = float(aux_ce)
        self.register_buffer("class_weights",
                             None if class_weights is None
                             else torch.as_tensor(class_weights, dtype=torch.float),
                             persistent=False)

    def pool(self, h, mask):
        m = mask.unsqueeze(-1).float()
        if self.backbone.bidirectional:
            return (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        last = mask.long().sum(1).clamp(min=1) - 1
        return h[torch.arange(h.shape[0], device=h.device), last]

    def encode_packed(self, ids, mask):
        lens = mask.long().sum(1).clamp(min=1).tolist()
        T, chunks, spans, cur = max(self.ctx_tokens, ids.shape[1]), [], [], []
        for i, n in enumerate(lens):
            if cur and sum(len(x) for x in cur) + n > T:
                chunks.append(cur); cur = []
            cur.append(ids[i, :n])
            spans.append((len(chunks), sum(len(x) for x in cur) - n, n))
        chunks.append(cur)
        C = len(chunks)
        cid = torch.zeros(C, T, dtype=ids.dtype, device=ids.device)
        cmask = torch.zeros(C, T, dtype=mask.dtype, device=ids.device)
        for c, parts in enumerate(chunks):
            flat = torch.cat(parts)
            cid[c, :len(flat)] = flat
            cmask[c, :len(flat)] = 1
        per = max(1, (self.sent_batch * ids.shape[1]) // T)
        hs = [self.backbone(cid[i:i + per], cmask[i:i + per]) for i in range(0, C, per)]
        h = torch.cat(hs, 0)
        if not self.backbone.bidirectional:
            return torch.stack([h[c, s + n - 1] for c, s, n in spans])
        return torch.stack([h[c, s:s + n].mean(0) for c, s, n in spans])

    def encode_sentences(self, ids, mask):
        if self.ctx_tokens > 0:
            return self.encode_packed(ids, mask)
        out = []
        for i in range(0, ids.shape[0], self.sent_batch):
            h = self.backbone(ids[i:i + self.sent_batch], mask[i:i + self.sent_batch])
            out.append(self.pool(h, mask[i:i + self.sent_batch]))
        return torch.cat(out, 0)

    def forward(self, ids, mask, labels=None, doc_mask=None):
        single = ids.dim() == 2
        if single:
            ids, mask = ids[None], mask[None]
            if labels is not None:
                labels = labels[None]
        B, S, L = ids.shape
        if doc_mask is None:
            doc_mask = torch.ones(B, S, dtype=torch.long, device=ids.device)
        sent = self.encode_sentences(ids.reshape(B * S, L), mask.reshape(B * S, L))
        sent = sent.view(B, S, -1)
        if self.doc is not None:
            sent = self.doc(sent, doc_mask)
        logits = self.head(self.drop(sent))

        out = {"logits": logits}
        if labels is not None:
            if self.crf is not None:
                safe = labels.clamp(min=0)
                loss = -self.crf.log_likelihood(logits, safe, doc_mask.bool())
                if self.aux_ce > 0 and self.class_weights is not None:
                    masked = labels.masked_fill(~doc_mask.bool(), IGNORE)
                    loss = loss + self.aux_ce * F.cross_entropy(
                        logits.reshape(-1, self.n_labels), masked.reshape(-1),
                        weight=self.class_weights.to(logits.dtype), ignore_index=IGNORE)
                out["loss"] = loss
            else:
                out["loss"] = F.cross_entropy(
                    logits.reshape(-1, self.n_labels), labels.reshape(-1),
                    weight=self.class_weights, ignore_index=IGNORE)
        return out

    @torch.no_grad()
    def predict(self, ids, mask, doc_mask=None):
        single = ids.dim() == 2
        o = self.forward(ids, mask, doc_mask=doc_mask)
        logits = o["logits"]
        if doc_mask is None:
            doc_mask = torch.ones(logits.shape[:2], dtype=torch.long, device=logits.device)
        tags = self.crf.decode(logits, doc_mask.bool()) if self.crf is not None \
            else logits.argmax(-1)
        return tags[0] if single else tags

    @torch.no_grad()
    def marginals(self, ids, mask, doc_mask=None):
        single = ids.dim() == 2
        logits = self.forward(ids, mask, doc_mask=doc_mask)["logits"].float()
        if doc_mask is None:
            doc_mask = torch.ones(logits.shape[:2], dtype=torch.long, device=logits.device)
        p = self.crf.marginals(logits, doc_mask.bool()) if self.crf is not None \
            else logits.softmax(-1) * doc_mask[..., None].float()
        return p[0] if single else p

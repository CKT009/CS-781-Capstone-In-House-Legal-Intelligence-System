from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F

from summ.peft import _APPLIED, _mark, _replace_submodule, _set_all, _unfreeze_head

_ALLOC_ATTR = "_adalora"


class SVDLinear(nn.Module):

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0,
                 init_std: float = 0.02):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"SVDLinear expects nn.Linear, got {type(base).__name__}")
        if r <= 0:
            raise ValueError("AdaLoRA rank r must be >= 1")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.alpha = int(r), float(alpha)
        self.scaling = self.alpha / self.r
        w = base.weight
        kw = dict(device=w.device, dtype=w.dtype)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, **kw))
        self.lora_E = nn.Parameter(torch.zeros(r, **kw))
        self.lora_B = nn.Parameter(torch.empty(base.out_features, r, **kw))
        nn.init.normal_(self.lora_A, 0.0, init_std)
        nn.init.normal_(self.lora_B, 0.0, init_std)
        self.register_buffer("rank_mask", torch.ones(r, dtype=torch.bool, device=w.device))
        self.drop = nn.Dropout(dropout)

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_B * self.lora_E) @ self.lora_A * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qe = self.lora_A * self.lora_E[:, None]
        return self.base(x) + F.linear(F.linear(self.drop(x), qe), self.lora_B) * self.scaling

    @property
    def rank(self) -> int:
        return int(self.rank_mask.sum())

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:g}"


def budget_schedule(step: int, total: int, b0: int, bT: int,
                    init_warmup_frac: float, final_warmup_frac: float) -> tuple[int, str]:
    ti = int(round(total * init_warmup_frac))
    tf = int(round(total * final_warmup_frac))
    if step < ti:
        return b0, "warmup"
    if step > total - tf or total - ti - tf <= 0:
        return bT, "final"
    mul = 1.0 - (step - ti) / (total - ti - tf)
    return int(bT + (b0 - bT) * mul ** 3), "prune"


class RankAllocator:

    def __init__(self, modules: dict[str, SVDLinear], r: int, target_r: int,
                 init_warmup_frac: float, final_warmup_frac: float, beta1: float,
                 beta2: float, orth_reg: float, mask_interval: int = 1, verbose: bool = True):
        if not 0 < target_r <= r:
            raise ValueError(f"need 0 < target_r <= r, got target_r={target_r}, r={r}")
        if not (init_warmup_frac >= 0 and final_warmup_frac >= 0
                and init_warmup_frac + final_warmup_frac <= 1):
            raise ValueError("warmup fractions must be >= 0 and sum to <= 1")
        self.modules = modules
        self.names = list(modules)
        self.r, self.target_r = r, target_r
        self.b0, self.bT = r * len(modules), target_r * len(modules)
        self.init_warmup_frac, self.final_warmup_frac = init_warmup_frac, final_warmup_frac
        self.beta1, self.beta2, self.orth_reg = beta1, beta2, orth_reg
        self.mask_interval = max(1, int(mask_interval))
        self.verbose = verbose
        self.ipt: dict[int, torch.Tensor] = {}
        self.unc: dict[int, torch.Tensor] = {}
        self.budget, self.phase = self.b0, "warmup"
        self.frozen_mask: list[torch.Tensor] | None = None
        self._last_logged = None


    def orth_loss(self) -> torch.Tensor:
        terms = []
        for m in self.modules.values():
            P, Q = m.lora_B.float(), m.lora_A.float()
            eye = torch.eye(m.r, device=P.device)
            terms.append(((P.T @ P - eye) ** 2).sum())
            terms.append(((Q @ Q.T - eye) ** 2).sum())
        return self.orth_reg * torch.stack(terms).mean()


    @torch.no_grad()
    def _update_ipt(self, p: torch.Tensor) -> torch.Tensor | None:
        if p.grad is None:
            return None
        k = id(p)
        cur = (p.detach().float() * p.grad.detach().float()).abs()
        if k not in self.ipt:
            self.ipt[k] = torch.zeros_like(cur)
            self.unc[k] = torch.zeros_like(cur)
        self.ipt[k].mul_(self.beta1).add_(cur, alpha=1 - self.beta1)
        self.unc[k].mul_(self.beta2).add_((cur - self.ipt[k]).abs(), alpha=1 - self.beta2)
        return self.ipt[k] * self.unc[k]

    @torch.no_grad()
    def triplet_scores(self) -> torch.Tensor:
        rows = []
        for m in self.modules.values():
            sE = self._update_ipt(m.lora_E)
            sP = self._update_ipt(m.lora_B)
            sQ = self._update_ipt(m.lora_A)
            row = torch.zeros(m.r, device=m.lora_E.device)
            if sE is not None:
                row += sE
            if sP is not None:
                row += sP.mean(0)
            if sQ is not None:
                row += sQ.mean(1)
            rows.append(row)
        return torch.stack(rows)

    @torch.no_grad()
    def _apply(self, masks: list[torch.Tensor]) -> None:
        for m, keep in zip(self.modules.values(), masks):
            m.rank_mask.copy_(keep)
            m.lora_E.masked_fill_(~keep, 0.0)

    @torch.no_grad()
    def update_and_mask(self, step: int, total: int) -> int:
        budget, phase = budget_schedule(step, total, self.b0, self.bT,
                                        self.init_warmup_frac, self.final_warmup_frac)
        entering_final = phase == "final" and self.frozen_mask is None
        if phase == "final" and not entering_final:
            self._apply(self.frozen_mask)
        else:
            scores = self.triplet_scores()
            if phase != "warmup" and (entering_final or step % self.mask_interval == 0):
                flat = scores.flatten()
                keep = torch.zeros_like(flat, dtype=torch.bool)
                keep[torch.topk(flat, budget).indices] = True
                masks = list(keep.view(len(self.modules), self.r))
                self._apply(masks)
                if entering_final:
                    self.frozen_mask = [k.clone() for k in masks]
            elif phase != "warmup":
                self._apply([m.rank_mask for m in self.modules.values()])
        self.budget, self.phase = budget, phase
        if self.verbose:
            every = max(1, total // 20)
            if step % every == 0 or entering_final or step == total - 1:
                print(f"[adalora] step {step}/{total} {phase}: budget {budget}/{self.b0} "
                      f"(kept {self.kept()}) | {self.summary()}", flush=True)
        return budget


    def kept(self) -> int:
        return sum(m.rank for m in self.modules.values())

    def rank_pattern(self) -> dict[str, int]:
        return {n: m.rank for n, m in self.modules.items()}

    def summary(self) -> str:
        agg: dict[str, int] = {}
        for n, m in self.modules.items():
            t = n.split(".")[-1]
            agg[t] = agg.get(t, 0) + m.rank
        return " ".join(f"{k} {v}" for k, v in agg.items())


def apply_adalora(model: nn.Module, r: int = 12, target_r: int = 6,
                  targets=("q_proj", "k_proj", "v_proj", "o_proj"), alpha=None,
                  init_warmup_frac: float = 0.1, final_warmup_frac: float = 0.3,
                  beta1: float = 0.85, beta2: float = 0.85, orth_reg: float = 0.1,
                  **kw) -> nn.Module:
    dropout = float(kw.pop("dropout", 0.0))
    init_std = float(kw.pop("init_std", 0.02))
    mask_interval = int(kw.pop("mask_interval", 1))
    verbose = bool(kw.pop("verbose", True))
    if kw:
        raise TypeError(f"apply_adalora got unexpected kwargs: {sorted(kw)}")
    previous = getattr(model, _APPLIED, None)
    if previous is not None:
        raise RuntimeError(f"apply_adalora: {previous!r} already applied; rebuild the model")
    r, target_r = int(r), int(target_r)
    alpha = float(2 * r if alpha is None else alpha)
    targets = tuple(targets)

    _ref = next(model.parameters(), None)
    _dev = _ref.device if _ref is not None else None
    _set_all(model, False)
    hits = [n for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and n.split(".")[-1] in targets]
    if not hits:
        raise RuntimeError(f"adalora: no nn.Linear matched targets={targets}")
    for name in hits:
        _replace_submodule(model, name, SVDLinear(model.get_submodule(name), r, alpha,
                                                  dropout, init_std))
    if _dev is not None:
        model.to(device=_dev)
    _unfreeze_head(model)
    mods = {n: model.get_submodule(n) for n in hits}
    alloc = RankAllocator(mods, r, target_r, init_warmup_frac, final_warmup_frac,
                          beta1, beta2, orth_reg, mask_interval, verbose)
    object.__setattr__(model, _ALLOC_ATTR, alloc)
    _mark(model, "adalora")
    return model


__all__ = ["SVDLinear", "RankAllocator", "apply_adalora", "budget_schedule"]

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_log(run_dir):
    p = Path(run_dir) / "log.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def ema(values, alpha=0.05):
    out, acc = [], None
    for v in values:
        acc = v if acc is None else alpha * v + (1 - alpha) * acc
        out.append(acc)
    return out


def loss_curve(run_dir, out=None, x="step", logy=True, alpha=0.05):
    rows = read_log(run_dir)
    if not rows:
        print(f"[plot] no log.jsonl in {run_dir}")
        return None
    tr = [r for r in rows if "train_loss" in r]
    va = [r for r in rows if "val_loss" in r]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    if tr:
        xs = [r.get(x, r["step"]) for r in tr]
        ys = [r["train_loss"] for r in tr]
        ax.plot(xs, ys, lw=0.6, alpha=0.25, color="tab:blue")
        ax.plot(xs, ema(ys, alpha), lw=1.6, color="tab:blue", label="train (EMA)")
    if va:
        ax.plot([r.get(x, r["step"]) for r in va], [r["val_loss"] for r in va],
                marker="o", ms=3.5, lw=1.4, color="tab:red", label="validation")
        best = min(va, key=lambda r: r["val_loss"])
        ax.axhline(best["val_loss"], ls=":", lw=1, color="tab:red", alpha=0.6)
        ax.annotate(f"best {best['val_loss']:.3f} @ {best['step']:,}",
                    (best.get(x, best["step"]), best["val_loss"]),
                    textcoords="offset points", xytext=(-8, 10), fontsize=8,
                    color="tab:red", ha="right")


    cfg_p = Path(run_dir) / "config.json"
    if cfg_p.exists():
        cfg = json.loads(cfg_p.read_text())
        if cfg.get("schedule") == "wsd" and tr:
            d = cfg.get("decay_start") or int(cfg["max_steps"] * (1 - cfg.get("decay_frac", .15)))
            if d < max(r["step"] for r in tr):
                dx = d if x == "step" else d * cfg.get("block_size", 1024) * \
                    cfg.get("batch_size", 1) * cfg.get("grad_accum", 1)
                ax.axvline(dx, ls="--", lw=1, color="gray", alpha=0.7)
                ax.text(dx, ax.get_ylim()[1], " lr decay", fontsize=8, color="gray", va="top")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("tokens" if x == "tokens" else "step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(Path(run_dir).name)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    out = out or Path(run_dir) / "loss.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")
    return out


def compare(run_dirs, key="val_loss", out="runs/compare.png", x="tokens", logy=True):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    drawn = 0
    for d in run_dirs:
        rows = [r for r in read_log(d) if key in r]
        if not rows:
            continue
        xs = [r.get(x, r["step"]) for r in rows]
        ys = ema([r[key] for r in rows]) if key == "train_loss" else [r[key] for r in rows]
        ax.plot(xs, ys, lw=1.4, label=Path(d).name)
        drawn += 1
    if not drawn:
        print(f"[plot] nothing logged with key {key!r}")
        plt.close(fig)
        return None
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("tokens seen" if x == "tokens" else "step")
    ax.set_ylabel(key)
    ax.set_title(f"{key} vs {x}")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")
    return out


def throughput(run_dirs, out="runs/cost.png"):
    names, tps, vram, params = [], [], [], []
    for d in run_dirs:
        rows = [r for r in read_log(d) if "tokens_per_s" in r]
        if not rows:
            continue
        mid = sorted(r["tokens_per_s"] for r in rows)
        names.append(Path(d).name)
        tps.append(mid[len(mid) // 2] / 1e3)
        vram.append(max(r.get("vram_reserved_mb", r.get("vram_mb", 0)) for r in rows) / 1024)
        s = Path(d) / "summary.json"
        params.append(json.loads(s.read_text()).get("params_M", 0) if s.exists() else 0)
    if not names:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, vals, lab, col in zip(axes, [tps, vram, params],
                                  ["median k tokens/s", "peak VRAM (GB)", "parameters (M)"],
                                  ["tab:blue", "tab:orange", "tab:green"]):
        ax.bar(names, vals, color=col)
        ax.set_ylabel(lab)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")
    return out


def summary_table(run_dirs, fmt="md") -> str:
    import math
    cols = ["run", "params (M)", "steps", "tokens (M)", "best val loss", "val ppl",
            "wall (h)", "VRAM (GB)", "tok/s"]
    body = []
    for d in run_dirs:
        s = Path(d) / "summary.json"
        if not s.exists():
            continue
        v = json.loads(s.read_text())
        bl = (v.get("best") or {}).get("val_loss")
        vram = v.get("peak_vram_reserved_mb") or v.get("peak_vram_mb") or 0
        body.append([
            Path(d).name,
            f"{v.get('params_M', 0):.1f}",
            f"{v.get('steps', 0):,}",
            f"{v.get('tokens', 0) / 1e6:,.0f}",
            f"{bl:.4f}" if bl else "--",
            f"{math.exp(bl):.2f}" if bl else "--",
            f"{v.get('wall_min', 0) / 60:.1f}",
            f"{vram / 1024:.1f}",
            f"{v.get('tokens_per_s', 0) / 1e3:.1f}k" if v.get("tokens_per_s") else "--",
        ])
    if not body:
        return "(no summary.json found -- a run must finish before it can be tabulated)"
    if fmt == "md":
        return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                         + ["| " + " | ".join(r) + " |" for r in body])
    esc = lambda s: s.replace("_", r"\_").replace("%", r"\%")
    return "\n".join([r"\begin{tabular}{l" + "r" * (len(cols) - 1) + "}", r"\toprule",
                      " & ".join(esc(c) for c in cols) + r" \\", r"\midrule"]
                     + [" & ".join(esc(x) for x in r) + r" \\" for r in body]
                     + [r"\bottomrule", r"\end{tabular}"])

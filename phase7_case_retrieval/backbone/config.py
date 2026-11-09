from dataclasses import dataclass, asdict, field
import json, hashlib
from pathlib import Path


@dataclass
class Config:

    name: str = "A_decoder"
    arch: str = "decoder"
    seed: int = 1337


    vocab_size: int = 16000
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4
    d_ff: int = 0
    max_seq_len: int = 1024
    norm: str = "rms"
    pos: str = "rope"
    act: str = "swiglu"
    dropout: float = 0.0
    bias: bool = False
    tie_embeddings: bool = True
    n_dec_layers: int = 6


    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    ssm_head_dim: int = 64
    chunk_size: int = 64


    window: int = 512
    n_global: int = 16


    data_dir: str = "data"
    shard_dir: str = "artifacts/shards"
    tokenizer_dir: str = "artifacts/tokenizer_16k"
    block_size: int = 1024


    batch_size: int = 8
    grad_accum: int = 8
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    warmup: int = 500
    max_steps: int = 20000
    schedule: str = "wsd"
    decay_frac: float = 0.15
    decay_start: int = 0
    grad_clip: float = 1.0
    amp: str = "fp16"
    grad_checkpoint: bool = True


    mlm_prob: float = 0.15
    span_rate: float = 0.15
    mean_span: int = 3


    out_dir: str = "runs"
    log_every: int = 20
    eval_every: int = 500
    eval_batches: int = 20
    save_every_min: float = 20.0
    max_minutes: float = 0.0
    device: str = "cuda"

    init_from: str = ""


    diff_eps: float = 1e-3
    diff_weighting: str = "uniform"
    diff_stratified: bool = True
    diff_steps: int = 64


    diff_schedule: str = "linear"
    diff_gumbel: float = 3.0


    diff_target_len: int = 192


    diff_cond_dim: int = 256


    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0
        if not self.d_ff:
            self.d_ff = (int(8 * self.d_model / 3) // 64 * 64 if self.act == "swiglu"
                         else 4 * self.d_model)


    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.name

    @property
    def tokens_per_step(self) -> int:
        return self.block_size * self.batch_size * self.grad_accum

    def n_params(self) -> int:
        if self.arch == "mamba":
            di = self.expand * self.d_model
            h = di // self.ssm_head_dim
            per_layer = (
                self.d_model * (2 * di + 2 * self.d_state + h)
                + (di + 2 * self.d_state) * (self.d_conv + 1)
                + 3 * h
                + di
                + di * self.d_model
                + self.d_model)
            head = 0 if self.tie_embeddings else self.vocab_size * self.d_model
            return (self.vocab_size * self.d_model + self.n_layers * per_layer
                    + self.d_model + head)
        if self.arch == "diffusion":
            hd, kv, d, c = self.head_dim, self.n_kv_heads, self.d_model, self.diff_cond_dim
            attn = 2 * d * d + 2 * d * kv * hd
            mlp = (3 if self.act == "swiglu" else 2) * d * self.d_ff
            enc_block = attn + mlp + 2 * d
            dec_block = 2 * attn + mlp + 6 * c * d + 6 * d

            t_emb = 2 * (c * 4 * c) + 4 * c + c
            final = 2 * c * d + 2 * d
            emb = self.vocab_size * d
            pos = self.max_seq_len * d if self.pos == "learned" else 0
            head = 0 if self.tie_embeddings else self.vocab_size * d
            return (emb + pos + self.n_layers * enc_block
                    + self.n_dec_layers * dec_block + t_emb + final + head + d)

        hd, kv = self.head_dim, self.n_kv_heads
        emb = self.vocab_size * self.d_model
        pos = self.max_seq_len * self.d_model if self.pos == "learned" else 0
        attn = 2 * self.d_model * self.d_model + 2 * self.d_model * kv * hd
        mlp = (3 if self.act == "swiglu" else 2) * self.d_model * self.d_ff
        block = attn + mlp + 2 * self.d_model
        n = self.n_layers + (self.n_dec_layers if self.arch == "encdec" else 0)
        cross = self.n_dec_layers * (attn + self.d_model) if self.arch == "encdec" else 0
        head = 0 if self.tie_embeddings else self.vocab_size * self.d_model
        return emb + pos + n * block + cross + head + self.d_model

    def check(self) -> "Config":
        n = self.n_params()
        smoke = self.name.startswith(("tiny", "smoke"))
        if not 50e6 <= n <= 300e6 and not smoke:
            raise ValueError(f"{n/1e6:.1f}M params -- rules require 50M-300M. Fix the config "
                             f"before spending GPU hours.")
        if smoke:
            print(f"[config] SMOKE preset ({n/1e6:.2f}M params) -- for testing only, "
                  f"never submit a model from this run")
        print(f"[config] {self.name}: {n/1e6:.1f}M params, d_ff={self.d_ff}, "
              f"{self.tokens_per_step:,} tokens/step, "
              f"{self.tokens_per_step*self.max_steps/1e9:.2f}B tokens planned")
        return self


    SHAPE_KEYS = ("arch", "vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads", "d_ff",
                  "max_seq_len", "norm", "pos", "act", "bias", "tie_embeddings", "n_dec_layers",
                  "window", "n_global", "block_size", "batch_size", "grad_accum", "seed",
                  "shard_dir", "d_state", "d_conv", "expand", "ssm_head_dim",


                  "diff_target_len", "diff_cond_dim", "diff_weighting")

    def shape_hash(self) -> str:
        d = {k: getattr(self, k) for k in self.SHAPE_KEYS}
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]

    def diff(self, other: dict) -> dict:
        mine = asdict(self)
        return {k: (other.get(k), mine[k]) for k in mine
                if k in other and other[k] != mine[k]}

    def save(self, path=None):
        p = Path(path or self.run_dir / "config.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2))
        return p

    @staticmethod
    def load(path) -> "Config":
        return Config(**json.loads(Path(path).read_text()))


PRESETS = {


    "A": dict(name="A_decoder", arch="decoder", d_model=768, n_layers=14, n_heads=12,
              n_kv_heads=4, norm="rms", pos="rope", act="swiglu"),
    "B": dict(name="B_encoder", arch="encoder", d_model=768, n_layers=12, n_heads=12,
              n_kv_heads=12, norm="layer", pos="learned", act="gelu", bias=True,
              lr=1e-4),
    "C": dict(name="C_encdec", arch="encdec", d_model=768, n_layers=12, n_dec_layers=6,
              n_heads=12, n_kv_heads=4, norm="rms", pos="rope", act="swiglu"),
    "D": dict(name="D_efficient", arch="efficient", d_model=768, n_layers=14, n_heads=12,
              n_kv_heads=4, max_seq_len=4096, block_size=4096, window=512, n_global=16,
              batch_size=2, grad_accum=32),


    "E": dict(name="E_student", arch="decoder", d_model=640, n_layers=10, n_heads=10,
              n_kv_heads=5, norm="rms", pos="rope", act="swiglu"),


    "M": dict(name="M_mamba", arch="mamba", d_model=1024, n_layers=13, expand=2,
              d_state=64, ssm_head_dim=128, d_conv=4, chunk_size=64,
              n_heads=16, n_kv_heads=4, norm="rms"),


    "F": dict(name="F_diffusion", arch="diffusion", d_model=768, n_layers=6,
              n_dec_layers=6, n_heads=12, n_kv_heads=4, norm="rms", pos="rope",
              act="swiglu", diff_target_len=192, diff_cond_dim=256),

    "tiny": dict(name="tiny", arch="decoder", d_model=128, n_layers=2, n_heads=4, n_kv_heads=2,
                 vocab_size=1000, max_seq_len=128, block_size=128, batch_size=2, grad_accum=1),
}

from __future__ import annotations
import json, re, time
from pathlib import Path
import numpy as np

DTYPE = np.uint16


SOURCES = {"acts": "train_acts.json", "high_courts": "train_high_courts.json",
           "supreme_court": "train_supreme_court.json", "tribunals": "train_tribunals.json"}

SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>", "<mask>", "<sep>", "<cls>"] + \
           [f"<extra_{i}>" for i in range(8)]


PLACEHOLDERS = ["<NAME>", "<ORG>", "<ENTITY>", "<SECTION>", "<ACT>", "<DATE>", "<GPE>",
                "<CASENO>", "<STATUTE>", "<COURT>"]
SPECIALS += PLACEHOLDERS

PAD, UNK, BOS, EOS, MASK, SEP, CLS = range(7)


_STRUCT = re.compile(rb'[{}"\\]')


def iter_json_array(path, chunk_size: int = 1 << 22):
    try:
        import ijson
        with open(path, "rb") as f:
            yield from ijson.items(f, "item")
        return
    except ImportError:
        pass
    buf, pos = b"", 0
    depth, in_str, start = 0, False, None
    esc_at = -1
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buf += chunk
            while True:
                emitted = False
                for m in _STRUCT.finditer(buf, pos):
                    i, c = m.start(), m.group()
                    pos = i + 1


                    if i == esc_at:
                        continue
                    if c == b"\\":
                        if in_str:
                            esc_at = i + 1
                        continue
                    if c == b'"':
                        in_str = not in_str
                        continue
                    if in_str:
                        continue
                    if c == b"{":
                        if depth == 0:
                            start = i
                        depth += 1
                    elif c == b"}":
                        depth -= 1
                        if depth == 0 and start is not None:
                            yield json.loads(buf[start:i + 1].decode("utf-8"))
                            shift = i + 1
                            buf, pos, start, emitted = buf[shift:], 0, None, True
                            esc_at -= shift
                            break
                if not emitted:
                    break
            if depth == 0 and start is None:
                buf, pos, esc_at = b"", 0, -1


def stream_docs(data_dir, source: str, min_chars: int = 200, max_docs: int | None = None):
    path = Path(data_dir) / "Phase_1_Pretraining" / "Train" / SOURCES[source]
    if not path.exists():
        print(f"[data] missing {path} -- python scripts/fetch_data.py --phase 1 --only {source}")
        return
    for i, rec in enumerate(iter_json_array(path)):
        if max_docs and i >= max_docs:
            return
        t = rec.get("text")
        if isinstance(t, list):
            t = "\n".join(t)
        if t and len(t) >= min_chars:
            yield t


def stream_all(data_dir, sources=None, max_docs_per_source=None):
    import itertools
    its = [stream_docs(data_dir, s, max_docs=max_docs_per_source)
           for s in (sources or SOURCES)]
    for row in itertools.zip_longest(*its):
        for t in row:
            if t:
                yield t


def train_tokenizer(texts, out_dir, vocab_size: int = 32000, min_frequency: int = 2):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok.save(str(out / "tokenizer.json"))
    print(f"[tokenizer] {tok.get_vocab_size()} tokens -> {out}")
    return tok


def load_tokenizer(path):
    p = Path(path)
    f = p / "tokenizer.json" if p.is_dir() else p
    if not f.exists():
        found = sorted(str(q.parent) for q in Path("artifacts").rglob("tokenizer.json")) \
            if Path("artifacts").exists() else []
        raise FileNotFoundError(
            f"no tokenizer at {f}\n"
            + (f"  available: {found}\n  pass one with --tokenizer <dir>" if found else
               "  train one first: python run.py tokenizer --vocab 16000 --sample 80000"))
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(f))


def fertility(tok, texts) -> dict:
    nt = nw = nc = 0
    for t in texts:
        nt += len(tok.encode(t).ids); nw += len(t.split()); nc += len(t)
    return {"tokens_per_word": round(nt / max(nw, 1), 3),
            "chars_per_token": round(nc / max(nt, 1), 2), "n_docs": len(texts)}


LEGAL_PROBES = ["Section 302 IPC", "Article 226 of the Constitution", "AIR 1973 SC 1461",
                "(2017) 10 SCC 1", "learned counsel for the appellant", "<NAME>", "<ENTITY>"]


def probe_tokens(tok) -> dict:
    return {s: len(tok.encode(s).ids) for s in LEGAL_PROBES}


def pack(texts, tok, out_dir, source: str, shard_tokens: int = 100_000_000,
         max_tokens: int | None = None) -> list[dict]:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    shards, buf, pending, total, idx, t0 = [], [], 0, 0, 0, time.time()

    def flush():
        nonlocal buf, idx, pending
        if not buf:
            return
        arr = np.concatenate(buf).astype(DTYPE)
        f = out / f"{source}_{idx:04d}.bin"
        arr.tofile(f)
        shards.append({"file": f.name, "source": source, "n_tokens": int(arr.size)})
        print(f"  [{source}] shard {idx}: {arr.size/1e6:.1f}M tokens")
        buf, idx, pending = [], idx + 1, 0

    for i, t in enumerate(texts):
        ids = tok.encode(t).ids + [EOS]
        buf.append(np.asarray(ids, dtype=np.int64))
        pending += len(ids); total += len(ids)
        if pending >= shard_tokens:
            flush()
        if i and i % 2000 == 0:
            print(f"  [{source}] {i} docs, {total/1e6:.1f}M tokens, "
                  f"{total/max(time.time()-t0,1e-9)/1e3:.0f}k tok/s")
        if max_tokens and total >= max_tokens:
            break
    flush()
    return shards


def write_index(out_dir, shards, extra=None):
    out = Path(out_dir)
    idx = {"dtype": str(np.dtype(DTYPE)), "shards": shards,
           "total_tokens": sum(s["n_tokens"] for s in shards), **(extra or {})}
    (out / "index.json").write_text(json.dumps(idx, indent=2))
    print(f"[pack] {idx['total_tokens']/1e9:.3f}B tokens in {len(shards)} shards -> {out}")
    return idx


def holdout_split(out_dir, val_tokens_per_source: int = 5_000_000):
    out = Path(out_dir)
    idx = json.loads((out / "index.json").read_text())
    by_source = {}
    for s in idx["shards"]:
        by_source.setdefault(s["source"], []).append(s)

    train, val = [], []
    for src, shards in by_source.items():
        need = val_tokens_per_source
        for s in reversed(shards):
            n = s["n_tokens"]
            if need <= 0:
                train.append(s)
            elif need >= n:
                val.append(s); need -= n
            else:
                cut = n - need
                train.append({**s, "start": s.get("start", 0), "n_tokens": cut})
                val.append({**s, "start": s.get("start", 0) + cut, "n_tokens": need})
                need = 0


    (out / "index_train.json").write_text(json.dumps({**idx, "shards": train}, indent=2))
    (out / "index_val.json").write_text(json.dumps({**idx, "shards": val}, indent=2))
    tn = sum(s["n_tokens"] for s in train)
    vn = sum(s["n_tokens"] for s in val)
    print(f"[split] train {tn/1e9:.3f}B | val {vn/1e6:.1f}M "
          f"({100*vn/(tn+vn):.1f}% held out, {val_tokens_per_source/1e6:.0f}M per corpus)")
    for src in by_source:
        v = sum(s["n_tokens"] for s in val if s["source"] == src)
        print(f"    {src:16s} val {v/1e6:5.1f}M")
    return {"train_tokens": tn, "val_tokens": vn}


def describe(shard_dir):
    idx = json.loads((Path(shard_dir) / "index.json").read_text())
    by = {}
    for s in idx["shards"]:
        by[s["source"]] = by.get(s["source"], 0) + s["n_tokens"]
    total = sum(by.values())
    print(f"{'source':16s} {'tokens':>12s} {'share':>8s}  shards")
    for src, n in sorted(by.items(), key=lambda kv: -kv[1]):
        k = sum(1 for s in idx["shards"] if s["source"] == src)
        print(f"{src:16s} {n/1e6:11.1f}M {100*n/total:7.1f}%  {k}")
    print(f"{'TOTAL':16s} {total/1e9:11.3f}B")
    return by


class ShardDataset:

    def __init__(self, shard_dir, sources=None, index_file="index.json"):
        self.dir = Path(shard_dir)
        meta = json.loads((self.dir / index_file).read_text())
        self.shards = [s for s in meta["shards"] if sources is None or s["source"] in sources]
        if not self.shards:
            raise ValueError(f"no shards for sources={sources} in {self.dir}")

        self.starts = [int(s.get("start", 0)) for s in self.shards]
        self.lengths = [int(s["n_tokens"]) for s in self.shards]
        self.cum = np.cumsum([0] + self.lengths)
        self._mm = {}

    @property
    def n_tokens(self) -> int:
        return int(self.cum[-1])

    def _open(self, i):
        if i not in self._mm:
            self._mm[i] = np.memmap(self.dir / self.shards[i]["file"], dtype=DTYPE, mode="r")
        return self._mm[i]

    def read(self, start: int, length: int) -> np.ndarray:
        out, filled = np.empty(length, dtype=np.int64), 0
        while filled < length:
            g = (start + filled) % self.n_tokens
            si = int(np.searchsorted(self.cum, g, side="right") - 1)
            mm = self._open(si)
            local = g - int(self.cum[si])
            take = min(length - filled, self.lengths[si] - local)
            base = self.starts[si] + local
            out[filled:filled + take] = mm[base:base + take].astype(np.int64)
            filled += take
        return out


class Sampler:

    def __init__(self, ds: ShardDataset, block_size: int, batch_size: int, seed: int = 1337):
        self.ds, self.block, self.bs, self.seed = ds, block_size, batch_size, seed
        self.max_start = max(1, ds.n_tokens - block_size - 1)

    def batch(self, step: int):
        rng = np.random.default_rng((self.seed * 1_000_003 + step) % (2**63 - 1))
        offs = rng.integers(0, self.max_start, size=self.bs, dtype=np.int64)
        chunk = np.stack([self.ds.read(int(o), self.block + 1) for o in offs])
        return chunk[:, :-1], chunk[:, 1:]

    def torch_batch(self, step: int, device="cuda"):
        import torch
        x, y = self.batch(step)
        return (torch.from_numpy(x).to(device, non_blocking=True),
                torch.from_numpy(y).to(device, non_blocking=True))

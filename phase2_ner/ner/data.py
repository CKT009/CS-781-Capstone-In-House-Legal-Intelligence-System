from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch


LABELS: list[str] = [
    "APP",
    "RESP",
    "A.COUNSEL",
    "R.COUNSEL",
    "JUDGE",
    "WIT",
    "AUTH",
    "COURT",
    "STAT",
    "PREC",
    "DATE",
    "CASENO",
]


TAGS: list[str] = ["O"]
for _label in LABELS:
    TAGS.append("B-" + _label)
    TAGS.append("I-" + _label)

TAG2ID: dict[str, int] = {tag: i for i, tag in enumerate(TAGS)}
ID2TAG: list[str] = list(TAGS)

O_ID: int = TAG2ID["O"]
IGNORE: int = -100

assert len(TAGS) == 1 + 2 * len(LABELS) == 25
assert O_ID == 0


END_EXCLUSIVE: bool = True


def _to_exclusive(end: int) -> int:
    return end if END_EXCLUSIVE else end + 1


def _from_exclusive(end: int) -> int:
    return end if END_EXCLUSIVE else end - 1


def trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    start = max(0, min(start, len(text)))
    end = max(0, min(end, len(text)))
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


@dataclass
class Chunk:

    doc_id: str
    input_ids: list[int]
    offsets: list[tuple[int, int]]
    tag_ids: list[int]
    predict_mask: list[bool]


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in fh if line.strip()]
        return json.load(fh)


def _describe(obj: Any) -> str:
    if isinstance(obj, dict):
        keys = list(obj)[:5]
        sample = obj[keys[0]] if keys else None
        return (
            f"dict with {len(obj)} keys (first: {keys!r}); "
            f"first value is {type(sample).__name__}"
            + (f" with keys {list(sample)[:8]!r}" if isinstance(sample, dict) else "")
        )
    if isinstance(obj, list):
        head = obj[0] if obj else None
        return (
            f"list of {len(obj)} items; first item is {type(head).__name__}"
            + (f" with keys {list(head)[:8]!r}" if isinstance(head, dict) else "")
        )
    return f"{type(obj).__name__}"


def load_labels(path: str | os.PathLike) -> list[str]:
    path = str(path)
    raw = open(path, "r", encoding="utf-8").read()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = [line.strip() for line in raw.splitlines()]

    if isinstance(parsed, dict):
        parsed = parsed.get("labels", list(parsed))
    if not isinstance(parsed, list):
        raise ValueError(f"{path}: expected a list of labels, saw {_describe(parsed)}")

    labels = [str(x).strip() for x in parsed if str(x).strip() and not str(x).startswith("#")]
    if set(labels) != set(LABELS):
        missing = sorted(set(LABELS) - set(labels))
        extra = sorted(set(labels) - set(LABELS))
        raise ValueError(
            f"{path}: label set does not match ner.data.LABELS "
            f"(missing={missing}, unexpected={extra}). Tag ids would shift; "
            f"fix the file or update LABELS deliberately."
        )
    return labels


def _coerce_texts(obj: Any, where: str) -> dict[str, str]:
    texts: dict[str, str] = {}

    if isinstance(obj, dict):

        for key in ("data", "documents", "docs", "texts"):
            if key in obj and isinstance(obj[key], (list, dict)):
                return _coerce_texts(obj[key], f"{where}[{key!r}]")
        for doc_id, value in obj.items():
            if isinstance(value, str):
                texts[str(doc_id)] = value
            elif isinstance(value, dict):
                text = _first_present(value, ("text", "data", "document", "content", "raw"))
                if not isinstance(text, str):
                    raise ValueError(
                        f"{where}: entry {doc_id!r} has no string text field; "
                        f"saw keys {list(value)[:8]!r}"
                    )
                texts[str(doc_id)] = text
            else:
                raise ValueError(
                    f"{where}: entry {doc_id!r} is {type(value).__name__}, "
                    f"expected str or dict"
                )
        return texts

    if isinstance(obj, list):
        for i, rec in enumerate(obj):
            if not isinstance(rec, dict):
                raise ValueError(f"{where}[{i}] is {type(rec).__name__}, expected dict")
            doc_id = _first_present(rec, ("id", "doc_id", "document_id", "uid", "name"))
            text = _first_present(rec, ("text", "data", "document", "content", "raw"))
            if doc_id is None:
                doc_id = str(i)
            if not isinstance(text, str):
                raise ValueError(
                    f"{where}[{i}]: no string text field, saw keys {list(rec)[:8]!r}"
                )
            texts[str(doc_id)] = text
        return texts

    raise ValueError(f"{where}: unsupported shape -- {_describe(obj)}")


def _first_present(d: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None


_START_KEYS = ("start", "start_char", "start_offset", "begin", "char_start", "from")
_END_KEYS = ("end", "end_char", "end_offset", "stop", "char_end", "to")
_LABEL_KEYS = ("label", "labels", "entity", "entity_type", "type", "tag", "category")
_TEXT_KEYS = ("text", "entity_text", "surface", "mention", "value", "span_text")


def _coerce_annotation(ann: Any, where: str) -> tuple[int, int, str, str | None]:
    if isinstance(ann, dict):
        start = _first_present(ann, _START_KEYS)
        end = _first_present(ann, _END_KEYS)
        label = _first_present(ann, _LABEL_KEYS)
        if isinstance(label, (list, tuple)):
            label = label[0] if label else None
        text = _first_present(ann, _TEXT_KEYS)
        if start is None or end is None or label is None:
            raise ValueError(
                f"{where}: annotation missing start/end/label; saw keys {list(ann)[:10]!r}"
            )
        return int(start), int(end), str(label), (str(text) if isinstance(text, str) else None)

    if isinstance(ann, (list, tuple)):
        if len(ann) < 3:
            raise ValueError(f"{where}: flat annotation {ann!r} has fewer than 3 fields")
        start, end, label = ann[0], ann[1], ann[2]
        text = ann[3] if len(ann) > 3 and isinstance(ann[3], str) else None
        return int(start), int(end), str(label), text

    raise ValueError(f"{where}: annotation is {type(ann).__name__}, expected dict or list")


def _coerce_golds(
    obj: Any, where: str
) -> tuple[dict[str, list[tuple[int, int, str]]], dict[str, list[str | None]]]:
    golds: dict[str, list[tuple[int, int, str]]] = {}
    surfaces: dict[str, list[str | None]] = {}

    def add(doc_id: str, anns: Any, sub: str) -> None:
        if anns is None:
            anns = []
        if not isinstance(anns, (list, tuple)):
            raise ValueError(f"{sub}: annotations are {type(anns).__name__}, expected list")
        spans, texts = [], []
        for i, ann in enumerate(anns):
            s, e, lab, txt = _coerce_annotation(ann, f"{sub}[{i}]")
            spans.append((s, e, lab))
            texts.append(txt)
        golds[str(doc_id)] = spans
        surfaces[str(doc_id)] = texts

    if isinstance(obj, dict):
        for key in ("data", "annotations", "golds", "entities"):
            if key in obj and isinstance(obj[key], (list, dict)):
                return _coerce_golds(obj[key], f"{where}[{key!r}]")
        for doc_id, value in obj.items():
            if isinstance(value, dict):
                anns = _first_present(value, ("entities", "annotations", "spans", "labels", "ner"))
                add(doc_id, anns, f"{where}[{doc_id!r}]")
            else:
                add(doc_id, value, f"{where}[{doc_id!r}]")
        return golds, surfaces

    if isinstance(obj, list):
        for i, rec in enumerate(obj):
            if not isinstance(rec, dict):
                raise ValueError(f"{where}[{i}] is {type(rec).__name__}, expected dict")
            doc_id = _first_present(rec, ("id", "doc_id", "document_id", "uid", "name")) or str(i)
            anns = _first_present(rec, ("entities", "annotations", "spans", "labels", "ner"))
            add(doc_id, anns, f"{where}[{i}]")
        return golds, surfaces

    raise ValueError(f"{where}: unsupported shape -- {_describe(obj)}")


_TEXT_PATTERNS = (
    "Train/{fold}-text.json", "Test/{fold}-text.json",
    "{fold}-text.json",
    "{fold}.json", "{fold}.jsonl",
    "{fold}_texts.json", "{fold}_text.json", "{fold}_docs.json",
    "{fold}/texts.json", "{fold}/text.json", "{fold}/docs.json",
    "{fold}/data.json", "{fold}/{fold}.json", "{fold}/documents.json",
)
_GOLD_PATTERNS = (
    "Train/{fold}-gold.json", "Test/{fold}-gold.json",
    "{fold}-gold.json",
    "{fold}_gold.json", "{fold}_golds.json", "{fold}_entities.json",
    "{fold}_annotations.json", "{fold}_labels.json", "{fold}_ner.json",
    "{fold}/gold.json", "{fold}/golds.json", "{fold}/entities.json",
    "{fold}/annotations.json", "{fold}/labels.json",
)
_FOLDS = ("fold1", "fold2", "fold3")


def _find(data_dir: str, patterns: Sequence[str], fold: str) -> str | None:
    for pat in patterns:
        candidate = os.path.join(data_dir, pat.format(fold=fold))
        if os.path.isfile(candidate):
            return candidate
    return None


def load_fold(
    data_dir: str | os.PathLike, fold: str
) -> tuple[dict[str, str], dict[str, list[tuple[int, int, str]]]]:
    data_dir = str(data_dir)
    if fold not in _FOLDS:
        raise ValueError(f"fold must be one of {_FOLDS}, got {fold!r}")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    text_path = _find(data_dir, _TEXT_PATTERNS, fold)
    if text_path is None:
        raise FileNotFoundError(
            f"No documents file for {fold} under {data_dir}. Tried: "
            + ", ".join(p.format(fold=fold) for p in _TEXT_PATTERNS)
            + f". Directory contains: {sorted(os.listdir(data_dir))[:20]!r}"
        )

    raw_texts = _read_json(text_path)
    texts = _coerce_texts(raw_texts, text_path)


    golds: dict[str, list[tuple[int, int, str]]] = {}
    gold_path = _find(data_dir, _GOLD_PATTERNS, fold)
    if gold_path is not None:
        golds, _ = _coerce_golds(_read_json(gold_path), gold_path)
    elif _looks_combined(raw_texts):
        golds, _ = _coerce_golds(raw_texts, text_path)

    if fold != "fold3" and not golds:
        raise FileNotFoundError(
            f"No gold annotations found for {fold}: {text_path} carries no "
            f"annotation field and no separate gold file matched "
            + ", ".join(p.format(fold=fold) for p in _GOLD_PATTERNS)
            + f". {text_path} looks like: {_describe(raw_texts)}"
        )

    unknown = {lab for spans in golds.values() for _, _, lab in spans} - set(LABELS)
    if unknown:
        raise ValueError(
            f"{gold_path or text_path}: gold contains labels outside LABELS: "
            f"{sorted(unknown)!r}"
        )


    golds = {doc: golds[doc] for doc in golds if doc in texts} if fold != "fold3" else {}
    return texts, golds


def _looks_combined(obj: Any) -> bool:
    keys = ("entities", "annotations", "spans", "labels", "ner")
    records: Iterable[Any]
    if isinstance(obj, dict):
        records = obj.values()
    elif isinstance(obj, list):
        records = obj
    else:
        return False
    return any(isinstance(r, dict) and any(k in r for k in keys) for r in records)


def verify_gold_convention(
    texts: Mapping[str, str],
    golds: Mapping[str, Sequence[tuple[int, int, str]]],
    entity_texts: Mapping[str, Any],
    sample: int = 200,
) -> dict[str, int]:
    exclusive = inclusive = n = 0

    for doc_id, spans in golds.items():
        text = texts.get(doc_id)
        provided = entity_texts.get(doc_id)
        if text is None or provided is None:
            continue
        for i, (start, end, _label) in enumerate(spans):
            if n >= sample:
                break
            if isinstance(provided, Mapping):
                gold_text = provided.get((start, end))
            elif i < len(provided):
                gold_text = provided[i]
            else:
                gold_text = None
            if not isinstance(gold_text, str):
                continue
            n += 1
            if text[start:end] == gold_text:
                exclusive += 1
            if text[start:end + 1] == gold_text:
                inclusive += 1

    return {"exclusive_matches": exclusive, "inclusive_matches": inclusive, "n": n}


def _tag_tokens(
    text: str,
    offsets: Sequence[tuple[int, int]],
    spans: Sequence[tuple[int, int, str]],
) -> list[int]:
    tags = [O_ID] * len(offsets)
    trimmed = [trim_span(text, s, e) for s, e in offsets]

    for start, end, label in sorted(spans, key=lambda sp: (sp[0], sp[1])):
        ex_end = _to_exclusive(end)
        start, ex_end = trim_span(text, start, ex_end)
        if ex_end <= start:
            continue

        members = [
            i for i, (ts, te) in enumerate(trimmed)
            if te > ts and ts < ex_end and te > start
        ]
        if not members:
            continue

        b_id = TAG2ID["B-" + label]
        i_id = TAG2ID["I-" + label]
        first, last = members[0], members[-1]
        if any(tags[i] != O_ID for i in range(first, last + 1)):
            continue
        for i in range(first, last + 1):
            tags[i] = b_id if i == first else i_id

    return tags


def encode_document(
    text: str,
    spans: Sequence[tuple[int, int, str]],
    tok,
    max_len: int = 512,
    stride: int = 128,
    doc_id: str = "",
) -> list[Chunk]:
    if max_len <= 0 or stride < 0 or stride >= max_len:
        raise ValueError(f"need 0 <= stride < max_len, got stride={stride}, max_len={max_len}")

    enc = tok.encode(text)
    ids = list(enc.ids)
    offsets = [tuple(o) for o in enc.offsets]
    if not ids:
        return []

    tags = _tag_tokens(text, offsets, spans)

    step = max_len - stride
    starts = [0]
    while starts[-1] + max_len < len(ids):
        starts.append(starts[-1] + step)


    half = stride // 2
    own = [0] + [s + half for s in starts[1:]] + [len(ids)]

    chunks: list[Chunk] = []
    for k, s in enumerate(starts):
        e = min(s + max_len, len(ids))
        chunks.append(
            Chunk(
                doc_id=doc_id,
                input_ids=ids[s:e],
                offsets=offsets[s:e],
                tag_ids=tags[s:e],
                predict_mask=[own[k] <= s + j < own[k + 1] for j in range(e - s)],
            )
        )
    return chunks


def decode_chunks(
    text: str,
    chunks: list[Chunk],
    pred_tag_ids: list[list[int]],
) -> list[tuple[int, int, str]]:
    if len(pred_tag_ids) != len(chunks):
        raise ValueError(
            f"pred_tag_ids has {len(pred_tag_ids)} rows for {len(chunks)} chunks"
        )


    owned: list[tuple[int, int, str]] = []
    for chunk, preds in zip(chunks, pred_tag_ids):
        if len(preds) != len(chunk.input_ids):
            raise ValueError(
                f"prediction row of length {len(preds)} for chunk of "
                f"length {len(chunk.input_ids)}"
            )
        for (start, end), tag_id, owns in zip(chunk.offsets, preds, chunk.predict_mask):
            if owns and tag_id != IGNORE:
                owned.append((start, end, ID2TAG[int(tag_id)]))
    owned.sort(key=lambda t: (t[0], t[1]))

    spans: list[tuple[int, int, str]] = []
    cur_label: str | None = None
    cur_start = cur_end = 0

    def flush() -> None:
        if cur_label is None:
            return
        s, e = trim_span(text, cur_start, cur_end)
        if e > s:
            spans.append((s, _from_exclusive(e), cur_label))

    for start, end, tag in owned:
        if tag == "O":
            flush()
            cur_label = None
            continue
        prefix, label = tag[0], tag[2:]
        if prefix == "B" or cur_label != label:
            flush()
            cur_label, cur_start, cur_end = label, start, end
        else:
            cur_end = max(cur_end, end)
    flush()

    return sorted(set(spans))


def collate(chunks: Sequence[Chunk], pad_id: int = 0) -> dict[str, torch.Tensor]:
    if not chunks:
        raise ValueError("collate() received an empty batch")

    width = max(len(c.input_ids) for c in chunks)
    input_ids, attention_mask, labels = [], [], []
    for c in chunks:
        pad = width - len(c.input_ids)
        input_ids.append(c.input_ids + [pad_id] * pad)
        attention_mask.append([1] * len(c.input_ids) + [0] * pad)
        labels.append(c.tag_ids + [IGNORE] * pad)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

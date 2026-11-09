import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ner.data import (
    IGNORE,
    ID2TAG,
    LABELS,
    O_ID,
    TAG2ID,
    TAGS,
    Chunk,
    collate,
    decode_chunks,
    encode_document,
    load_fold,
    load_labels,
    trim_span,
    verify_gold_convention,
)

TOKENIZER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "artifacts", "tokenizer_16k", "tokenizer.json",
)


@pytest.fixture(scope="module")
def tok():
    from tokenizers import Tokenizer

    assert os.path.isfile(TOKENIZER_PATH), TOKENIZER_PATH
    return Tokenizer.from_file(TOKENIZER_PATH)


DOC = (
    "Ram Kumar Sharma vs State of Maharashtra\n"
    "Before the Hon'ble Mr. Justice A. K. Mishra\n"
    "Section 302 IPC was invoked on 12.05.2019 in Crl.A. No. 4471 of 2019.\n"
    "Shri Mehta Shri Gupta appeared.\n"
    "A was the witness.\n"
    "Supreme Court of India"
)

SPANS = [
    (0, 16, "APP"),
    (20, 40, "RESP"),
    (52, 84, "JUDGE"),
    (85, 100, "STAT"),
    (116, 126, "DATE"),
    (130, 153, "CASENO"),
    (155, 165, "A.COUNSEL"),
    (166, 176, "A.COUNSEL"),
    (187, 188, "WIT"),
    (206, 228, "COURT"),
]


def test_span_fixture_is_self_consistent():
    expected = [
        "Ram Kumar Sharma", "State of Maharashtra",
        "Hon'ble Mr. Justice A. K. Mishra", "Section 302 IPC",
        "12.05.2019", "Crl.A. No. 4471 of 2019",
        "Shri Mehta", "Shri Gupta", "A", "Supreme Court of India",
    ]
    assert [DOC[s:e] for s, e, _ in SPANS] == expected


    for s, e, _ in SPANS:
        assert not DOC[s].isspace() and not DOC[e - 1].isspace()


def test_tag_scheme():
    assert len(LABELS) == 12
    assert LABELS[0] == "APP" and LABELS[-1] == "CASENO"
    assert len(TAGS) == 25
    assert TAGS[:5] == ["O", "B-APP", "I-APP", "B-RESP", "I-RESP"]
    assert O_ID == 0 and IGNORE == -100
    assert ID2TAG == TAGS
    assert all(TAG2ID[t] == i for i, t in enumerate(TAGS))


def test_trim_span_strips_both_ends_and_is_idempotent():
    text = "  Ram Kumar \n"
    once = trim_span(text, 0, len(text))
    assert text[once[0]:once[1]] == "Ram Kumar"
    assert trim_span(text, *once) == once


    t2 = "\n\tShri Mehta\t\n"
    s, e = trim_span(t2, 0, len(t2))
    assert t2[s:e] == "Shri Mehta"
    assert trim_span(t2, s, e) == (s, e)


    assert trim_span("Ram", 0, 3) == (0, 3)

    s, e = trim_span("Ram   Kumar", 3, 6)
    assert e == s

    assert trim_span("Ram", -5, 99) == (0, 3)


def test_trim_span_fixes_the_byte_level_prefix_space(tok):
    text = "The Hon'ble Mr."
    enc = tok.encode(text)
    i = enc.tokens.index("ĠHon")
    raw = tuple(enc.offsets[i])
    assert raw == (3, 7) and text[raw[0]:raw[1]] == " Hon"
    assert trim_span(text, *raw) == (4, 7)
    assert text[4:7] == "Hon"


def _round_trip(text, spans, tok, max_len, stride):
    chunks = encode_document(text, spans, tok, max_len=max_len, stride=stride, doc_id="d1")
    return chunks, decode_chunks(text, chunks, [c.tag_ids for c in chunks])


@pytest.mark.parametrize("max_len,stride", [(512, 128), (64, 16), (24, 8), (16, 8), (8, 2)])
def test_round_trip_exact(tok, max_len, stride):
    _, decoded = _round_trip(DOC, SPANS, tok, max_len, stride)
    assert decoded == sorted(SPANS)


def test_round_trip_covers_the_hard_positions(tok):
    chunks, decoded = _round_trip(DOC, SPANS, tok, 24, 8)
    assert len(chunks) > 1, "need overlapping windows for this to be meaningful"
    got = set(decoded)
    assert (0, 16, "APP") in got
    assert (206, 228, "COURT") in got
    assert (155, 165, "A.COUNSEL") in got
    assert (166, 176, "A.COUNSEL") in got
    assert (130, 153, "CASENO") in got
    assert (187, 188, "WIT") in got
    assert (52, 84, "JUDGE") in got
    assert (85, 100, "STAT") in got


def test_entity_really_does_cross_a_window_boundary(tok):
    chunks = encode_document(DOC, SPANS, tok, max_len=24, stride=8)
    split = 0
    for start, end, _ in SPANS:
        owners = {
            i for i, c in enumerate(chunks)
            for (ts, te), owns in zip(c.offsets, c.predict_mask)
            if owns and min(te, end) > max(ts, start)
        }
        split += len(owners) > 1
    assert split >= 1


def test_decoded_spans_never_touch_whitespace(tok):
    _, decoded = _round_trip(DOC, SPANS, tok, 24, 8)
    assert decoded
    for start, end, _ in decoded:
        assert not DOC[start].isspace(), DOC[start - 2:end]
        assert not DOC[end - 1].isspace(), DOC[start:end + 2]


def test_entity_spanning_a_newline_stays_one_span(tok):
    text = "Before the Hon'ble Mr.\nJustice Mishra retired."
    start, end = text.index("Hon"), text.index(" retired")
    spans = [(start, end, "JUDGE")]
    _, decoded = _round_trip(text, spans, tok, 512, 128)
    assert decoded == [(start, end, "JUDGE")]
    assert "\n" in text[start:end]


@pytest.mark.parametrize("max_len,stride", [(24, 8), (16, 8), (32, 16), (8, 2)])
def test_every_token_owned_exactly_once(tok, max_len, stride):
    enc = tok.encode(DOC)
    assert len(enc.ids) > max_len, "fixture must be longer than the window"

    chunks = encode_document(DOC, SPANS, tok, max_len=max_len, stride=stride)
    assert len(chunks) > 1

    owned = [
        off for c in chunks
        for off, owns in zip(c.offsets, c.predict_mask) if owns
    ]

    assert owned == [tuple(o) for o in enc.offsets]


    carried = [off for c in chunks for off in c.offsets]
    assert len(carried) > len(owned)


def test_no_span_emitted_twice_across_overlapping_windows(tok):
    _, decoded = _round_trip(DOC, SPANS, tok, 24, 8)
    assert len(decoded) == len(set(decoded)) == len(SPANS)
    starts = [s for s, _, _ in decoded]
    assert starts == sorted(starts)


def test_first_window_owns_its_own_start_later_ones_do_not(tok):
    stride = 8
    chunks = encode_document(DOC, SPANS, tok, max_len=24, stride=stride)
    assert chunks[0].predict_mask[0] is True
    for c in chunks[1:]:
        assert c.predict_mask[:stride // 2] == [False] * (stride // 2)


def test_tags_are_kept_in_context_regions(tok):
    chunks = encode_document(DOC, SPANS, tok, max_len=24, stride=8)
    assert all(t != IGNORE for c in chunks for t in c.tag_ids)

    assert any(
        t != O_ID
        for c in chunks[1:]
        for t, owns in zip(c.tag_ids, c.predict_mask)
        if not owns
    )


def test_decode_all_O_returns_nothing(tok):
    chunks = encode_document(DOC, [], tok, max_len=24, stride=8)
    assert all(t == O_ID for c in chunks for t in c.tag_ids)
    assert decode_chunks(DOC, chunks, [[O_ID] * len(c.input_ids) for c in chunks]) == []


def test_empty_document_yields_no_chunks(tok):
    assert encode_document("", [], tok) == []
    assert decode_chunks("", [], []) == []


def test_whitespace_only_prediction_is_not_emitted(tok):
    text = "Order \n\n passed."
    chunks = encode_document(text, [], tok, max_len=512, stride=128)
    preds = [
        [TAG2ID["B-DATE"] if trim_span(text, *off)[1] == trim_span(text, *off)[0]
         else O_ID for off in c.offsets]
        for c in chunks
    ]
    assert any(p != O_ID for row in preds for p in row), "fixture must have a blank token"
    assert decode_chunks(text, chunks, preds) == []


def test_lone_I_tag_opens_a_new_entity(tok):
    text = "Ram Kumar Sharma vs State"
    chunks = encode_document(text, [], tok, max_len=512, stride=128)
    assert len(chunks) == 1
    preds = list(chunks[0].tag_ids)
    preds[0] = TAG2ID["I-APP"]
    preds[1] = TAG2ID["I-APP"]
    assert decode_chunks(text, chunks, [preds]) == [(0, 9, "APP")]


def test_label_change_without_B_still_splits(tok):
    text = "Ram Kumar vs State"
    chunks = encode_document(text, [], tok, max_len=512, stride=128)
    preds = list(chunks[0].tag_ids)
    preds[0] = TAG2ID["B-APP"]
    preds[1] = TAG2ID["I-RESP"]
    assert decode_chunks(text, chunks, [preds]) == [(0, 3, "APP"), (4, 9, "RESP")]


def test_decode_rejects_misaligned_predictions(tok):
    chunks = encode_document(DOC, SPANS, tok, max_len=24, stride=8)
    with pytest.raises(ValueError):
        decode_chunks(DOC, chunks, [[O_ID] * len(chunks[0].input_ids)])
    with pytest.raises(ValueError):
        decode_chunks(DOC, chunks, [[O_ID] * 3 for _ in chunks])


def test_encode_rejects_bad_window_geometry(tok):
    with pytest.raises(ValueError):
        encode_document(DOC, SPANS, tok, max_len=16, stride=16)


def test_collate_pads_labels_with_IGNORE(tok):
    import torch

    chunks = encode_document(DOC, SPANS, tok, max_len=24, stride=8)
    batch = collate(chunks, pad_id=0)
    width = max(len(c.input_ids) for c in chunks)
    assert batch["input_ids"].shape == (len(chunks), width)
    assert batch["labels"].shape == batch["input_ids"].shape == batch["attention_mask"].shape
    assert batch["input_ids"].dtype == torch.long

    short = min(range(len(chunks)), key=lambda i: len(chunks[i].input_ids))
    n = len(chunks[short].input_ids)
    if n < width:
        assert batch["attention_mask"][short][n:].eq(0).all()
        assert batch["labels"][short][n:].eq(IGNORE).all()
    assert batch["labels"][batch["attention_mask"].bool()].ne(IGNORE).all()


def test_load_labels_roundtrip_and_mismatch(tmp_path):
    good = tmp_path / "labels.txt"
    good.write_text("\n".join(LABELS))
    assert load_labels(good) == LABELS

    bad = tmp_path / "bad.json"
    bad.write_text('["APP", "RESP", "NOT_A_LABEL"]')
    with pytest.raises(ValueError, match="does not match"):
        load_labels(bad)


@pytest.mark.parametrize("shape", ["dict_str", "dict_obj", "list", "combined"])
def test_load_fold_accepts_plausible_shapes(tmp_path, shape):
    import json

    if shape == "dict_str":
        (tmp_path / "fold1.json").write_text(json.dumps({"d1": DOC}))
        (tmp_path / "fold1_gold.json").write_text(
            json.dumps({"d1": [{"start": 0, "end": 16, "label": "APP"}]})
        )
    elif shape == "dict_obj":
        (tmp_path / "fold1.json").write_text(json.dumps({"d1": {"text": DOC}}))
        (tmp_path / "fold1_gold.json").write_text(
            json.dumps({"d1": [[0, 16, "APP", "Ram Kumar Sharma"]]})
        )
    elif shape == "list":
        (tmp_path / "fold1.json").write_text(json.dumps([{"id": "d1", "text": DOC}]))
        (tmp_path / "fold1_entities.json").write_text(
            json.dumps([{"id": "d1", "entities": [
                {"start_char": 0, "end_char": 16, "entity_type": "APP"}]}])
        )
    else:
        (tmp_path / "fold1.json").write_text(json.dumps(
            {"d1": {"text": DOC, "annotations": [
                {"start_offset": 0, "end_offset": 16, "labels": ["APP"]}]}}
        ))

    texts, golds = load_fold(tmp_path, "fold1")
    assert texts == {"d1": DOC}
    assert golds == {"d1": [(0, 16, "APP")]}


def test_load_fold_test_split_has_no_gold(tmp_path):
    import json

    (tmp_path / "fold3.json").write_text(json.dumps({"d9": DOC}))
    texts, golds = load_fold(tmp_path, "fold3")
    assert list(texts) == ["d9"] and golds == {}


def test_load_fold_errors_are_specific(tmp_path):
    import json

    with pytest.raises(ValueError, match="fold must be one of"):
        load_fold(tmp_path, "fold7")
    with pytest.raises(FileNotFoundError, match="No documents file"):
        load_fold(tmp_path, "fold1")

    (tmp_path / "fold1.json").write_text(json.dumps({"d1": {"body": DOC}}))
    with pytest.raises(ValueError, match="no string text field"):
        load_fold(tmp_path, "fold1")

    (tmp_path / "fold1.json").write_text(json.dumps({"d1": DOC}))
    with pytest.raises(FileNotFoundError, match="No gold annotations"):
        load_fold(tmp_path, "fold1")

    (tmp_path / "fold1_gold.json").write_text(
        json.dumps({"d1": [{"start": 0, "end": 3, "label": "PETITIONER"}]})
    )
    with pytest.raises(ValueError, match="outside LABELS"):
        load_fold(tmp_path, "fold1")

    (tmp_path / "fold1_gold.json").write_text(json.dumps({"d1": [{"from": 0, "to": 3}]}))
    with pytest.raises(ValueError, match="missing start/end/label"):
        load_fold(tmp_path, "fold1")


def test_verify_gold_convention_detects_exclusive_and_inclusive():
    texts = {"d1": DOC}
    surfaces = {"d1": [DOC[s:e] for s, e, _ in SPANS]}

    report = verify_gold_convention(texts, {"d1": SPANS}, surfaces)
    assert report["n"] == len(SPANS)
    assert report["exclusive_matches"] == len(SPANS)


    assert report["inclusive_matches"] == 1
    assert report["exclusive_matches"] > report["inclusive_matches"]


    inclusive = [(s, e - 1, lab) for s, e, lab in SPANS]
    report = verify_gold_convention(texts, {"d1": inclusive}, surfaces)
    assert report["inclusive_matches"] == len(SPANS)
    assert report["exclusive_matches"] == 0


    report = verify_gold_convention(
        texts, {"d1": SPANS}, {"d1": {(s, e): DOC[s:e] for s, e, _ in SPANS}}
    )
    assert report["exclusive_matches"] == len(SPANS)


def test_round_trip_on_a_long_document(tok):
    sep = "\n\nORDER\n\n"
    filler = "The learned counsel for the parties were heard at length. " * 40
    text = DOC + sep + filler + DOC
    shift = len(DOC) + len(sep) + len(filler)
    spans = SPANS + [(s + shift, e + shift, lab) for s, e, lab in SPANS]

    chunks = encode_document(text, spans, tok, max_len=128, stride=32, doc_id="long")
    assert len(chunks) > 3
    enc = tok.encode(text)
    owned = [off for c in chunks for off, m in zip(c.offsets, c.predict_mask) if m]
    assert owned == [tuple(o) for o in enc.offsets]
    assert decode_chunks(text, chunks, [c.tag_ids for c in chunks]) == sorted(spans)
    assert all(c.doc_id == "long" for c in chunks)

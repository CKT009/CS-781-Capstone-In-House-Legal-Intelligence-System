import pytest

from ner.metrics import (
    LABELS,
    confusion,
    format_report,
    partial_credit_report,
    strict_prf,
)


GOLD = {
    "doc1": [
        [0, 5, "APP"],
        [10, 20, "RESP"],
        [30, 34, "DATE"],
        [40, 48, "COURT"],
    ],
    "doc2": [
        [0, 7, "JUDGE"],
        [12, 22, "STAT"],
    ],
}


def _copy(d):
    return {k: [list(s) for s in v] for k, v in d.items()}


def test_labels_are_the_twelve_in_order():
    assert LABELS == ["APP", "RESP", "A.COUNSEL", "R.COUNSEL", "JUDGE", "WIT",
                      "AUTH", "COURT", "STAT", "PREC", "DATE", "CASENO"]
    assert len(LABELS) == 12


def test_perfect_prediction_scores_one():
    res = strict_prf(_copy(GOLD), GOLD)
    assert res["micro_p"] == res["micro_r"] == res["micro_f1"] == 1.0
    for lab in ["APP", "RESP", "DATE", "COURT", "JUDGE", "STAT"]:
        assert res["per_label"][lab]["f1"] == 1.0

    assert res["macro_f1"] == pytest.approx(6 / 12)
    assert res["macro_f1_labels_with_support"] == 1.0


    full = {"d": [[i * 10, i * 10 + 5, lab] for i, lab in enumerate(LABELS)]}
    r2 = strict_prf(_copy(full), full)
    for k in ["macro_f1", "macro_p", "macro_r", "micro_f1", "micro_p", "micro_r"]:
        assert r2[k] == 1.0, k
    assert r2["support"] == 12


@pytest.mark.parametrize("pred", [{}, {"doc1": [], "doc2": []}], ids=["no-docs", "empty-lists"])
def test_empty_prediction_is_zero_without_dividing_by_zero(pred):
    res = strict_prf(pred, GOLD)
    assert res["macro_f1"] == 0.0
    assert res["macro_r"] == res["micro_r"] == 0.0
    assert res["macro_p"] == res["micro_p"] == 0.0
    assert res["micro_f1"] == 0.0
    assert res["support"] == 6
    assert res["fn"] == 6 and res["tp"] == 0 and res["fp"] == 0


def test_empty_gold_and_empty_pred_does_not_crash():
    res = strict_prf({}, {})
    assert res["macro_f1"] == 0.0 and res["support"] == 0
    assert res["macro_f1_labels_with_support"] == 0.0


def test_one_character_boundary_shift_scores_zero():
    pred = _copy(GOLD)
    pred["doc1"][1] = [10, 21, "RESP"]
    res = strict_prf(pred, GOLD)
    resp = res["per_label"]["RESP"]
    assert (resp["tp"], resp["fp"], resp["fn"]) == (0, 1, 1)
    assert resp["p"] == resp["r"] == resp["f1"] == 0.0

    assert res["micro_p"] == pytest.approx(5 / 6)
    assert res["micro_r"] == pytest.approx(5 / 6)

    assert res["per_label"]["APP"]["f1"] == 1.0


    pred2 = _copy(GOLD)
    pred2["doc1"][1] = [11, 20, "RESP"]
    assert strict_prf(pred2, GOLD)["per_label"]["RESP"]["f1"] == 0.0


def test_correct_span_wrong_label_is_zero_for_both_labels():
    pred = _copy(GOLD)
    pred["doc1"][0] = [0, 5, "RESP"]
    res = strict_prf(pred, GOLD)
    app, resp = res["per_label"]["APP"], res["per_label"]["RESP"]
    assert (app["tp"], app["fp"], app["fn"]) == (0, 0, 1)
    assert app["f1"] == 0.0
    assert (resp["tp"], resp["fp"], resp["fn"]) == (1, 1, 0)
    assert resp["p"] == pytest.approx(0.5)

    assert resp["f1"] == pytest.approx(2 / 3)


    g = {"d": [[0, 5, "APP"]]}
    r = strict_prf({"d": [[0, 5, "RESP"]]}, g)
    assert r["per_label"]["APP"]["f1"] == 0.0
    assert r["per_label"]["RESP"]["f1"] == 0.0
    assert r["macro_f1"] == 0.0 and r["micro_f1"] == 0.0


def test_duplicate_spans_do_not_inflate_tp():
    pred = {"doc1": [[0, 5, "APP"], [0, 5, "APP"], [0, 5, "APP"]], "doc2": []}
    gold = {"doc1": [[0, 5, "APP"]], "doc2": []}
    res = strict_prf(pred, gold)
    app = res["per_label"]["APP"]
    assert (app["tp"], app["fp"], app["fn"]) == (1, 0, 0)
    assert res["micro_p"] == 1.0 and res["micro_f1"] == 1.0
    assert res["n_pred"] == 1


    res2 = strict_prf({"doc1": [[1, 5, "APP"], [1, 5, "APP"]]}, gold)
    assert res2["per_label"]["APP"]["fp"] == 1


    res3 = strict_prf({"doc1": [[0, 5, "APP"]]}, {"doc1": [[0, 5, "APP"], [0, 5, "APP"]]})
    assert res3["support"] == 1 and res3["micro_r"] == 1.0


    res4 = strict_prf({"doc1": [[0, 5, "APP"], [0, 5, "RESP"]]}, gold)
    assert res4["n_pred"] == 2 and res4["fp"] == 1


def test_macro_below_micro_when_support_is_imbalanced():
    gold = {"d": [[i * 10, i * 10 + 4, "DATE"] for i in range(100)] + [[5000, 5005, "WIT"]]}
    pred = {"d": [[i * 10, i * 10 + 4, "DATE"] for i in range(100)]}
    res = strict_prf(pred, gold)

    assert res["micro_f1"] > 0.99
    assert res["macro_f1"] == pytest.approx(1 / 12)
    assert res["macro_f1"] < res["micro_f1"]
    assert res["macro_f1_labels_with_support"] == pytest.approx(0.5)


    gold2 = {"d": [[i * 10, i * 10 + 4, "DATE"] for i in range(10)] + [[5000, 5005, "WIT"]]}
    pred2 = {"d": [[5000, 5005, "WIT"]] + [[i * 10, i * 10 + 4, "DATE"] for i in range(2)]}
    res2 = strict_prf(pred2, gold2)
    assert res2["macro_f1_labels_with_support"] > res2["micro_f1"]


def test_four_element_span_scores_identically_to_three_element():
    three = _copy(GOLD)
    four = {
        "doc1": [[0, 5, "APP", "Rajiv"], [10, 20, "RESP", "State of X"],
                 [30, 34, "DATE", "1999"], [40, 48, "COURT", "High Ct"]],
        "doc2": [[0, 7, "JUDGE", "K. Rao"], [12, 22, "STAT", "s. 420 IPC"]],
    }
    a = strict_prf(three, GOLD)
    b = strict_prf(four, GOLD)
    for k in ["macro_f1", "macro_p", "macro_r", "micro_f1", "support", "tp", "fp", "fn"]:
        assert a[k] == b[k], k
    assert a["per_label"] == b["per_label"]


    noisy = {"doc1": [[0, 5, "APP", "TOTALLY WRONG TEXT"]], "doc2": []}
    assert strict_prf(noisy, {"doc1": [[0, 5, "APP"]], "doc2": []})["micro_f1"] == 1.0


    assert strict_prf({"d": [[0, 5, "APP"]]}, {"d": [[0, 5, "APP", "txt"]]})["micro_f1"] == 1.0


    assert strict_prf({"d": [[0, 5, "APP", "t", "x"]]}, {"d": [[0, 5, "APP"]]})["tp"] == 1
    with pytest.raises(ValueError):
        strict_prf({"d": [[0, 5]]}, {"d": [[0, 5, "APP"]]})


def test_partial_credit_categories_are_exclusive_and_sum():
    gold = {
        "d": [
            [0, 5, "APP"],
            [10, 20, "RESP"],
            [30, 34, "DATE"],
            [40, 48, "COURT"],
            [60, 70, "STAT"],
        ],
        "e": [[0, 4, "JUDGE"]],
    }
    pred = {
        "d": [
            [0, 5, "APP"],
            [10, 18, "RESP"],
            [30, 34, "PREC"],
            [42, 50, "WIT"],
            [200, 210, "CASENO"],
        ],
        "e": [],
    }
    rep = partial_credit_report(pred, gold)

    assert rep["n_pred"] == 5 and rep["n_gold"] == 6
    assert rep["pred_side"] == {"exact": 1, "label": 1, "boundary": 1,
                                "boundary_label": 1, "spurious": 1}
    assert rep["gold_side"] == {"exact": 1, "label": 1, "boundary": 1,
                                "boundary_label": 1, "missed": 2}


    assert sum(rep["pred_side"].values()) == rep["n_pred"]
    assert sum(rep["gold_side"].values()) == rep["n_gold"]

    assert rep["pred_side"]["exact"] == rep["gold_side"]["exact"]
    assert rep["exact"] == strict_prf(pred, gold)["tp"]

    assert (rep["boundary"], rep["label"], rep["spurious"], rep["missed"]) == (1, 1, 1, 2)
    assert [e[1] for e in rep["examples"]["spurious"]] == [(200, 210, "CASENO")]


    perfect = partial_credit_report(_copy(GOLD), GOLD)
    assert perfect["pred_side"]["exact"] == perfect["n_pred"] == 6
    assert perfect["gold_side"]["missed"] == 0


    adj = partial_credit_report({"d": [[5, 9, "APP"]]}, {"d": [[0, 5, "APP"]]})
    assert adj["spurious"] == 1 and adj["missed"] == 1 and adj["boundary"] == 0


    nest = partial_credit_report({"d": [[1, 4, "APP"]]}, {"d": [[0, 5, "APP"]]})
    assert nest["boundary"] == 1 and nest["spurious"] == 0


def test_partial_credit_duplicates_and_empty():
    rep = partial_credit_report({"d": [[0, 5, "APP"], [0, 5, "APP"]]}, {"d": [[0, 5, "APP"]]})
    assert rep["n_pred"] == 1 and rep["exact"] == 1
    empty = partial_credit_report({}, GOLD)
    assert empty["missed"] == 6 and empty["n_pred"] == 0
    assert sum(empty["pred_side"].values()) == 0


def test_missing_doc_id_counts_all_its_gold_as_fn():
    pred = {"doc1": [list(s) for s in GOLD["doc1"]]}
    res = strict_prf(pred, GOLD)
    assert res["tp"] == 4 and res["fn"] == 2 and res["fp"] == 0
    assert res["per_label"]["JUDGE"]["fn"] == 1
    assert res["per_label"]["STAT"]["fn"] == 1
    assert res["micro_p"] == 1.0
    assert res["micro_r"] == pytest.approx(4 / 6)


def test_extra_doc_id_in_prediction_raises_by_default():
    pred = _copy(GOLD)
    pred["doc_TYPO"] = [[0, 3, "APP"]]
    with pytest.raises(ValueError, match="absent from gold"):
        strict_prf(pred, GOLD)
    with pytest.raises(ValueError):
        partial_credit_report(pred, GOLD)
    with pytest.raises(ValueError):
        confusion(pred, GOLD)


def test_extra_doc_id_can_be_downgraded_to_a_warning_and_is_then_ignored():
    pred = _copy(GOLD)
    pred["doc_TYPO"] = [[0, 3, "APP"]]
    with pytest.warns(UserWarning, match="absent from gold"):
        res = strict_prf(pred, GOLD, on_extra_docs="warn")

    assert res["fp"] == 0 and res["tp"] == 6 and res["micro_f1"] == 1.0


def test_unknown_label_is_warned_about_and_dropped():
    with pytest.warns(UserWarning, match="outside the"):
        res = strict_prf({"d": [[0, 5, "APP"], [9, 11, "NOT_A_LABEL"]]}, {"d": [[0, 5, "APP"]]})
    assert res["fp"] == 0 and res["micro_f1"] == 1.0


def test_bad_input_is_rejected():
    with pytest.raises(ValueError):
        strict_prf({"d": [[9, 2, "APP"]]}, {"d": [[0, 5, "APP"]]})
    with pytest.raises(ValueError):
        strict_prf({}, {}, labels=["APP", "APP"])
    with pytest.raises(ValueError):
        strict_prf({}, {}, labels=[])


def test_offsets_may_be_strings_and_a_custom_label_set_restricts_scoring():
    assert strict_prf({"d": [["0", "5", "APP"]]}, {"d": [[0, 5, "APP"]]})["tp"] == 1
    with pytest.warns(UserWarning, match="outside the"):
        res = strict_prf({"d": [[0, 5, "APP"]]}, {"d": [[0, 5, "APP"], [7, 9, "DATE"]]},
                         labels=["APP"])
    assert res["macro_f1"] == 1.0 and res["support"] == 1


def test_confusion_counts_only_exact_offset_pairs():
    gold = {"d": [[0, 5, "APP"], [10, 20, "RESP"], [30, 34, "DATE"], [60, 70, "STAT"]]}
    pred = {"d": [[0, 5, "APP"], [10, 20, "APP"], [30, 35, "DATE"], [90, 95, "PREC"]]}
    m = confusion(pred, gold)
    assert m["APP"]["APP"] == 1
    assert m["RESP"]["APP"] == 1
    assert m["DATE"]["DATE"] == 0
    assert m["STAT"]["PREC"] == 0
    assert sum(sum(r.values()) for r in m.values()) == 2
    assert set(m) == set(LABELS) and set(m["APP"]) == set(LABELS)

    res = strict_prf(pred, gold)
    for lab in LABELS:
        assert m[lab][lab] == res["per_label"][lab]["tp"]


def test_format_report_is_plain_text_sorted_by_support():
    gold = {"d": [[0, 5, "APP"], [10, 20, "APP"], [30, 34, "DATE"]]}
    res = strict_prf({"d": [[0, 5, "APP"], [30, 34, "DATE"]]}, gold)
    txt = format_report(res)
    assert isinstance(txt, str) and "\t" not in txt
    lines = txt.splitlines()
    assert lines[0].split()[:4] == ["label", "P", "R", "F1"]


    body = [l.split()[0] for l in lines[2:] if l and not l.startswith("-")]
    assert body.index("APP") < body.index("DATE") < body.index("COURT")
    assert all(lab in txt for lab in LABELS)
    assert "MACRO" in txt and "micro" in txt
    assert "official metric" in txt
    assert max(len(l) for l in lines) < 120

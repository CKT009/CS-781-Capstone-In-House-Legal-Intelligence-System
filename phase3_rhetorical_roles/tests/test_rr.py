import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
import torch

from rr import data as D
from rr.metrics import score, validate_submission

DATA = "data/Phase_3_RR"
has_data = pathlib.Path(DATA, "Train", "label_vocab.json").exists()
need = pytest.mark.skipif(not has_data, reason="run python fetch.py first")


@pytest.mark.parametrize("n,cap,stride", [(10, 256, 64), (256, 256, 64), (257, 256, 64),
                                          (1816, 256, 64), (943, 128, 32)])
def test_windows_cover_every_sentence_exactly_once_via_owner(n, cap, stride):
    wins = D.windows(n, cap, stride)
    assert wins[0][0] == 0 and wins[-1][1] == n, "windows do not span the document"
    for i in range(n):
        w = D.owner(wins, i)
        lo, hi = wins[w]
        assert lo <= i < hi, f"sentence {i} assigned to window {wins[w]} that excludes it"


def test_owner_prefers_the_window_where_a_sentence_has_most_context():
    wins = [(0, 100), (64, 164)]

    assert D.owner(wins, 70) == 0

    assert D.owner(wins, 150) == 1


def test_macro_counts_zero_support_labels_as_zero():
    vocab = ["A", "B", "C", "D"]
    gold = {"d": ["A", "A", "B"]}
    pred = {"d": ["A", "A", "B"]}
    s = score(pred, gold, vocab)
    assert s["macro_f1_supported"] == pytest.approx(1.0)
    assert s["macro_f1"] == pytest.approx(0.5)
    assert s["n_supported"] == 2


def test_score_rejects_length_mismatch():
    vocab = ["A", "B"]
    with pytest.raises(ValueError, match="predictions for"):
        score({"d": ["A"]}, {"d": ["A", "B"]}, vocab)


def test_score_rejects_missing_document():
    with pytest.raises(ValueError, match="missing from predictions"):
        score({}, {"d": ["A"]}, ["A"])


def test_perfect_and_worst_case():
    vocab = ["A", "B"]
    gold = {"d": ["A", "B", "A"]}
    assert score({"d": ["A", "B", "A"]}, gold, vocab)["macro_f1"] == pytest.approx(1.0)
    assert score({"d": ["B", "A", "B"]}, gold, vocab)["macro_f1"] == pytest.approx(0.0)


def test_validate_submission_catches_the_silent_zeros():
    vocab = ["A", "B"]
    docs = [D.Doc(id="d1", domain="CL", sentences=["x", "y"]),
            D.Doc(id="d2", domain="CL", sentences=["z"])]
    assert validate_submission({"d1": ["A", "B"], "d2": ["A"]}, docs, vocab) == []
    assert any("missing" in e for e in validate_submission({"d1": ["A", "B"]}, docs, vocab))
    assert any("labels for" in e
               for e in validate_submission({"d1": ["A"], "d2": ["A"]}, docs, vocab))
    assert any("outside label_vocab" in e
               for e in validate_submission({"d1": ["A", "Z"], "d2": ["A"]}, docs, vocab))


def test_doc_rejects_label_length_mismatch():
    with pytest.raises(ValueError, match="labels"):
        D.Doc(id="d", domain="CL", sentences=["a", "b"], labels=["Fact"])


@need
def test_splits_are_disjoint_and_labels_are_in_vocab():
    D.check(DATA, verbose=False)


@need
def test_every_train_document_has_one_label_per_sentence():
    for d in D.load_all(DATA, "train") + D.load_all(DATA, "dev"):
        assert len(d.labels) == len(d.sentences)


@need
def test_test_documents_carry_no_labels():
    for d in D.load_all(DATA, "test"):
        assert d.labels is None


@need
def test_encode_doc_shapes_and_truncation():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("artifacts/tokenizer_16k/tokenizer.json")
    vocab = D.load_labels(DATA)
    doc = D.load_all(DATA, "train")[0]
    e = D.encode_doc(doc, tok, vocab, max_sent_len=64)
    assert len(e.ids) == len(doc) and len(e.labels) == len(doc)
    assert all(len(r) == 64 for r in e.ids), "sentences are not padded to a fixed width"
    assert all(len(r) == 64 for r in e.mask)
    assert all(0 <= l < len(vocab) for l in e.labels)


def test_crf_decode_shape_and_loss_is_per_sentence():
    from rr.model import CRF
    torch.manual_seed(0)
    crf = CRF(13)
    logits = torch.randn(2, 20, 13)
    tags = torch.randint(0, 13, (2, 20))
    mask = torch.ones(2, 20, dtype=torch.bool)
    ll = crf.log_likelihood(logits, tags, mask)
    assert ll.dim() == 0 and torch.isfinite(ll)
    assert abs(float(ll)) < 50, f"log-likelihood {float(ll)} is not per-sentence scaled"
    out = crf.decode(logits, mask)
    assert out.shape == (2, 20) and out.min() >= 0 and out.max() < 13


def test_crf_decode_respects_mask_length():
    from rr.model import CRF
    crf = CRF(5)
    logits = torch.randn(1, 10, 5)
    mask = torch.zeros(1, 10, dtype=torch.bool)
    mask[0, :4] = True
    out = crf.decode(logits, mask)
    assert (out[0, 4:] == 0).all(), "positions past the mask should be filled with 0"


def test_every_experiment_applies_cleanly():
    import torch.nn as nn
    from rr.peft import apply_peft
    from rr.train import TrainConfig
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from run_train import EXPERIMENTS

    class TinyBackbone(nn.Module):
        def __init__(self, d=32, layers=2):
            super().__init__()
            from backbone.config import Config
            from backbone.layers import Stack
            cfg = Config(name="t", arch="encoder", d_model=d, n_layers=layers, n_heads=4,
                         n_kv_heads=4, vocab_size=64, max_seq_len=32, block_size=32,
                         norm="rms", pos="rope", act="swiglu", dropout=0.0,
                         grad_checkpoint=False)
            self.inner = Stack(cfg, mode="bidirectional")
            self.d_model, self.bidirectional, self.arch = d, True, "encoder"
            self.max_positions = 32

        def forward(self, ids, mask=None):
            return self.inner(ids)[0]

    from rr.model import RRTagger
    for name, spec in EXPERIMENTS.items():
        if name == "distill":
            continue
        cfg = TrainConfig(name=name, **{k: v for k, v in spec.items()
                                        if not k.startswith("_")})
        m = RRTagger(TinyBackbone(), n_labels=13, doc_layers=cfg.doc_layers,
                     doc_heads=4, use_crf=cfg.use_crf, max_sents=32)
        try:
            apply_peft(m, cfg.peft, **cfg.peft_kw)
        except NotImplementedError:
            continue
        except TypeError as e:
            pytest.fail(f"experiment {name!r} ({cfg.peft}) does not apply: {e}")


def test_rslora_scaling_differs_from_lora_and_ia3_is_identity_at_init():
    import torch.nn as nn
    from rr.peft import LoRALinear, IA3Linear
    torch.manual_seed(0)
    base = nn.Linear(16, 16, bias=False)
    a = LoRALinear(base, r=4, alpha=16, rslora=False)
    b = LoRALinear(nn.Linear(16, 16, bias=False), r=4, alpha=16, rslora=True)
    assert a.scaling == pytest.approx(16 / 4)
    assert b.scaling == pytest.approx(16 / 2.0)
    x = torch.randn(3, 16)
    ia3 = IA3Linear(base)
    assert torch.allclose(ia3(x), base(x), atol=1e-6), "IA3 is not identity at init"
    assert [n for n, p in ia3.named_parameters() if p.requires_grad] == ["ia3"]


def test_class_weights_actually_change_the_loss_under_crf():
    import torch.nn as nn
    from rr.model import RRTagger

    class Tiny(nn.Module):
        def __init__(self, d=16):
            super().__init__()
            self.emb = nn.Embedding(32, d)
            self.d_model, self.bidirectional, self.arch, self.max_positions = d, True, "encoder", 16
        def forward(self, ids, mask=None):
            return self.emb(ids)

    torch.manual_seed(0)
    ids = torch.randint(0, 32, (6, 8)); mask = torch.ones(6, 8, dtype=torch.long)
    labels = torch.randint(0, 5, (6,))
    flat = torch.ones(5)
    skew = torch.tensor([1.0, 1.0, 1.0, 1.0, 9.0])

    def loss_with(w, aux):
        torch.manual_seed(0)
        m = RRTagger(Tiny(), n_labels=5, doc_layers=1, doc_heads=2, use_crf=True,
                     max_sents=16, class_weights=w, aux_ce=aux)
        return float(m(ids, mask, labels=labels)["loss"])

    assert loss_with(flat, 0.5) != pytest.approx(loss_with(skew, 0.5)), \
        "class weights do not reach the CRF loss"
    assert loss_with(flat, 0.0) == pytest.approx(loss_with(skew, 0.0)), \
        "aux_ce=0 should restore the unweighted behaviour exactly"

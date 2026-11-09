import itertools
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import torch

from rr.model import CRF


def _brute_marginals(crf, logits, L):
    K = logits.shape[-1]
    tr, st, en = crf.transitions.detach(), crf.start.detach(), crf.end.detach()
    probs = torch.zeros(L, K, dtype=torch.float64)
    Z = 0.0
    for seq in itertools.product(range(K), repeat=L):
        s = st[seq[0]] + logits[0, seq[0]]
        for t in range(1, L):
            s = s + tr[seq[t - 1], seq[t]] + logits[t, seq[t]]
        s = float(torch.exp((s + en[seq[-1]]).double()))
        Z += s
        for t, k in enumerate(seq):
            probs[t, k] += s
    return probs / Z


def _random_crf(K, seed=0):
    torch.manual_seed(seed)
    crf = CRF(K)
    with torch.no_grad():
        crf.transitions.normal_(); crf.start.normal_(); crf.end.normal_()
    return crf


def test_crf_marginals_match_brute_force_enumeration_with_padding():
    K, T = 3, 5
    crf = _random_crf(K)
    logits = torch.randn(2, T, K)
    mask = torch.ones(2, T, dtype=torch.bool)
    mask[1, 3:] = False
    p = crf.marginals(logits, mask)
    assert p.shape == (2, T, K)
    for b, L in ((0, 5), (1, 3)):
        ref = _brute_marginals(crf, logits[b], L)
        assert torch.allclose(p[b, :L].double(), ref, atol=1e-5)
        assert torch.allclose(p[b, :L].sum(-1), torch.ones(L), atol=1e-5)
    assert (p[1, 3:] == 0).all()


def test_crf_marginals_length_one_is_softmax_with_start_end():
    crf = _random_crf(4, seed=1)
    logits = torch.randn(1, 1, 4)
    p = crf.marginals(logits, torch.ones(1, 1, dtype=torch.bool))
    ref = torch.softmax(crf.start + logits[0, 0] + crf.end, -1)
    assert torch.allclose(p[0, 0], ref.detach(), atol=1e-6)


def test_tagger_marginals_are_distributions():
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
    ids = torch.randint(0, 32, (7, 8)); mask = torch.ones(7, 8, dtype=torch.long)
    for use_crf in (True, False):
        m = RRTagger(Tiny(), n_labels=5, doc_layers=1, doc_heads=2, use_crf=use_crf,
                     max_sents=16).eval()
        p = m.marginals(ids, mask)
        assert p.shape == (7, 5)
        assert torch.allclose(p.sum(-1), torch.ones(7), atol=1e-5)


def test_numpy_viterbi_matches_crf_decode():
    from ensemble_soft import viterbi
    crf = _random_crf(6, seed=2)
    logits = torch.randn(1, 30, 6)
    ref = crf.decode(logits, torch.ones(1, 30, dtype=torch.bool))[0].tolist()
    got = viterbi(logits[0].numpy().astype(np.float64),
                  crf.transitions.detach().numpy().astype(np.float64),
                  crf.start.detach().numpy().astype(np.float64),
                  crf.end.detach().numpy().astype(np.float64))
    assert got == ref


def test_soft_average_and_argmax():
    from ensemble_soft import average, decode_argmax
    vocab = ["A", "B", "C"]
    m1 = {"d": np.array([[0.6, 0.4, 0.0], [0.1, 0.1, 0.8]])}
    m2 = {"d": np.array([[0.0, 0.9, 0.1], [0.2, 0.0, 0.8]])}
    m3 = {"d": np.array([[0.6, 0.4, 0.0], [0.3, 0.3, 0.4]])}

    avg = average([m1, m2, m3])
    assert np.allclose(avg["d"].sum(1), 1.0)
    assert decode_argmax(avg, vocab)["d"] == ["B", "C"]

    assert decode_argmax(average([m1, m2], [1.0, 0.0]), vocab)["d"] == ["A", "C"]


def test_soft_average_rejects_misaligned_members():
    from ensemble_soft import average
    with pytest.raises(ValueError, match="different documents"):
        average([{"a": np.ones((2, 3))}, {"b": np.ones((2, 3))}])
    with pytest.raises(ValueError, match="shape"):
        average([{"a": np.ones((2, 3))}, {"a": np.ones((3, 3))}])


def test_final_fit_flag_is_invisible_when_off():
    from rr.train import TrainConfig
    assert "final_fit" not in TrainConfig().to_dict()
    assert TrainConfig(final_fit=True).to_dict()["final_fit"] is True


def test_final_fit_run_defaults_to_last_pt_and_refuses_best(tmp_path):
    from rr.predict import default_ckpt, is_final_fit, load_tagger
    from rr.train import TrainConfig
    honest, ff = tmp_path / "honest", tmp_path / "ff"
    for d, flag in ((honest, False), (ff, True)):
        d.mkdir()
        (d / "config.json").write_text(json.dumps(TrainConfig(final_fit=flag).to_dict(),
                                                  default=str))
    assert not is_final_fit(honest) and default_ckpt(honest) == "best.pt"
    assert is_final_fit(ff) and default_ckpt(ff) == "last.pt"
    with pytest.raises(FileNotFoundError, match="FINAL-FIT"):
        load_tagger(ff, "best.pt", "cpu")


def test_soft_dev_scoring_refuses_final_fit_runs(tmp_path):
    from ensemble_soft import load_probs
    (tmp_path / "config.json").write_text(json.dumps({"final_fit": True}))
    with pytest.raises(SystemExit, match="FINAL-FIT"):
        load_probs(tmp_path, "dev", ["A"])

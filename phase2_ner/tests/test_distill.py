import sys, pathlib, torch
import pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from backbone import Config, build
from ner.backbones import Backbone
from ner.model import NERTagger
from ner import data as D
from ner.distill import make_student, kd_loss


def _pair():
    cfg = Config(name="t", arch="encoder", d_model=64, n_layers=6, n_heads=4,
                 n_kv_heads=4, vocab_size=16000, max_seq_len=64, block_size=64,
                 norm="layer", pos="learned", act="gelu", bias=True,
                 grad_checkpoint=False)
    torch.manual_seed(0)
    teacher = NERTagger(Backbone(build(cfg).stack, cfg, 64), n_tags=len(D.TAGS)).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher, make_student(teacher, keep_every=2)


def _batch(n=32):
    ids = torch.randint(10, 15000, (2, n))
    lab = torch.randint(0, len(D.TAGS), (2, n))
    lab[0, :5] = D.IGNORE
    return ids, torch.ones_like(ids), lab


def test_student_is_thinner_than_the_teacher():
    teacher, student = _pair()
    assert len(student.backbone.inner.blocks) == 3
    assert len(teacher.backbone.inner.blocks) == 6
    assert sum(p.numel() for p in student.parameters()) < \
        sum(p.numel() for p in teacher.parameters())


def test_student_is_trainable():
    teacher, student = _pair()
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())
    assert trainable == total, f"only {trainable:,}/{total:,} trainable"

    ids, am, lab = _batch()
    with torch.no_grad():
        t_logits = teacher(ids, am)["logits"]
    out = student(ids, am, labels=lab)
    (0.5 * out["loss"] + 0.5 * kd_loss(out["logits"], t_logits, lab, 2.0)).backward()
    got = [n for n, p in student.named_parameters()
           if p.requires_grad and p.grad is not None and (p.grad != 0).any()]
    assert len(got) > 10, f"gradient reached only {len(got)} tensors: {got}"
    assert any("blocks" in n for n in got), "no gradient reached the student's blocks"
    assert all(p.grad is None for p in teacher.parameters()), "teacher was not frozen"


def test_self_distillation_is_zero():
    teacher, student = _pair()
    ids, am, lab = _batch()
    with torch.no_grad():
        t = teacher(ids, am)["logits"]
    assert abs(float(kd_loss(t, t, lab, 2.0))) < 1e-5


def test_ignore_positions_are_excluded():
    teacher, student = _pair()
    ids, am, lab = _batch()
    with torch.no_grad():
        t = teacher(ids, am)["logits"]
    s = student(ids, am)["logits"]
    assert float(kd_loss(s, t, torch.full_like(lab, D.IGNORE), 2.0)) == 0.0
    assert torch.isfinite(kd_loss(s, t, lab, 2.0))


@pytest.mark.parametrize("tau", [1.0, 2.0, 4.0, 8.0])
def test_tau_squared_keeps_gradients_alive(tau):
    teacher, student = _pair()
    ids, am, lab = _batch()
    with torch.no_grad():
        t = teacher(ids, am)["logits"]
    student.zero_grad()
    kd_loss(student(ids, am)["logits"], t, lab, tau).backward()
    g = student.head_proj.weight.grad.abs().mean().item()
    assert g > 1e-8, f"tau={tau}: gradient collapsed to {g:.2e}"


def test_student_is_trainable_whatever_the_source_was():
    import inspect
    import ner.distill as dist

    teacher, student = _pair()
    for p in student.parameters():
        p.requires_grad = False
    again = dist.make_student(student, keep_every=1)
    assert all(p.requires_grad for p in again.parameters()), \
        "make_student returned a student that cannot be trained"

    src = inspect.getsource(dist.make_student)
    assert "cannot learn" in src, "the post-condition guard was removed"

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from tokenizers import Tokenizer

from backbone import Config, build
from ner import data as D
from ner.backbones import Backbone
from ner.metrics import strict_prf, format_report
from ner.model import NERTagger
from ner.peft import apply_peft, trainable_report, METHODS
from ner.predict import validate, write_submission

DOC = (
    "IN THE HIGH COURT OF JUDICATURE AT MADRAS\n\n"
    "Criminal Appeal No. 442 of 2019\n\n"
    "Before the Hon'ble Mr. Justice K. Ramaswamy, this appeal was heard on "
    "12 March 2019. The appellant Rajesh Kumar was represented by learned counsel "
    "Mr. S. Venkatesan, and the respondent State of Tamil Nadu by Mr. A. Natarajan. "
    "The Court considered Section 302 of the Indian Penal Code and the decision in "
    "AIR 1973 SC 1461."
)


def spans_of(text):
    want = [
        ("HIGH COURT OF JUDICATURE AT MADRAS", "COURT"),
        ("Criminal Appeal No. 442 of 2019", "CASENO"),
        ("K. Ramaswamy", "JUDGE"),
        ("12 March 2019", "DATE"),
        ("Rajesh Kumar", "APP"),
        ("S. Venkatesan", "A.COUNSEL"),
        ("State of Tamil Nadu", "RESP"),
        ("A. Natarajan", "R.COUNSEL"),
        ("Section 302 of the Indian Penal Code", "STAT"),
        ("AIR 1973 SC 1461", "PREC"),
    ]
    out = []
    for s, lab in want:
        i = text.index(s)
        out.append((i, i + len(s), lab))
    return sorted(out)


def main():
    ok = []
    tmp = Path(tempfile.mkdtemp(prefix="plm_smoke_"))
    tok = Tokenizer.from_file("artifacts/tokenizer_16k/tokenizer.json")
    gold_spans = spans_of(DOC)


    for max_len, stride in [(128, 32), (64, 16), (48, 24)]:
        cs = D.encode_document(DOC, gold_spans, tok, max_len=max_len, stride=stride)
        bk = D.decode_chunks(DOC, cs, [c.tag_ids for c in cs])
        got, want = set(map(tuple, bk)), set(map(tuple, gold_spans))
        ok.append((f"offset round trip @ max_len={max_len} stride={stride}: "
                   f"{len(got & want)}/{len(want)} in {len(cs)} chunk(s)", got == want))
        if got != want:
            for s, e, l in sorted(want - got):
                print(f"    LOST     [{s:4d},{e:4d}) {l:10s} {DOC[s:e]!r}")
            for s, e, l in sorted(got - want):
                print(f"    SPURIOUS [{s:4d},{e:4d}) {l:10s} {DOC[s:e]!r}")
        ok.append((f"  no duplicate spans @ {max_len}", len(bk) == len(set(map(tuple, bk)))))
        ok.append((f"  no whitespace at span edges @ {max_len}",
                   all(not DOC[s].isspace() and not DOC[e - 1].isspace()
                       for s, e, _ in bk)))

    chunks = D.encode_document(DOC, gold_spans, tok, max_len=64, stride=16)
    ok.append((f"document splits into {len(chunks)} chunks", len(chunks) > 1))
    back = D.decode_chunks(DOC, chunks, [c.tag_ids for c in chunks])


    cfg = Config(name="smoke", arch="encoder", d_model=64, n_layers=2, n_heads=4,
                 n_kv_heads=4, vocab_size=16000, max_seq_len=128, block_size=128,
                 norm="layer", pos="learned", act="gelu", bias=True,
                 grad_checkpoint=False)
    bb = Backbone(build(cfg).stack, cfg, max_positions=128)
    model = NERTagger(bb, n_tags=len(D.TAGS), dropout=0.0)
    ok.append(("tagger builds", True))

    batch = D.collate(chunks)
    out = model(batch["input_ids"], batch["attention_mask"], labels=batch["labels"])
    ok.append(("forward + loss finite", torch.isfinite(out["loss"]).item()))
    ok.append(("logit shape",
               tuple(out["logits"].shape[:2]) == tuple(batch["input_ids"].shape)
               and out["logits"].shape[-1] == len(D.TAGS)))


    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = float(out["loss"].detach())
    for _ in range(12):
        loss = model(batch["input_ids"], batch["attention_mask"],
                     labels=batch["labels"])["loss"]
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    ok.append((f"loss decreases ({first:.3f} -> {float(loss):.3f})",
               float(loss) < first))


    for m in METHODS:
        try:
            fresh = NERTagger(Backbone(build(cfg).stack, cfg, 128), n_tags=len(D.TAGS))
            apply_peft(fresh, m)
            rep = trainable_report(fresh)
            o = fresh(batch["input_ids"], batch["attention_mask"])
            good = (torch.isfinite(o["logits"]).all().item()
                    and rep["trainable"] > 0
                    and (rep["trainable"] == rep["total"] if m == "full"
                         else rep["trainable"] < rep["total"]))
            ok.append((f"peft:{m} ({rep['pct']:.1f}% trainable)", good))
        except NotImplementedError as e:
            ok.append((f"peft:{m} unsupported here, raised cleanly", True))
        except Exception as e:
            ok.append((f"peft:{m} -- {type(e).__name__}: {e}", False))


    tags = model.predict_tags(batch["input_ids"], batch["attention_mask"])
    per_chunk = [tags[i, :int(batch["attention_mask"][i].sum())].tolist()
                 for i in range(len(chunks))]
    preds = {"doc1": D.decode_chunks(DOC, chunks, per_chunk)}
    res = strict_prf(preds, {"doc1": gold_spans})
    ok.append(("scorer runs", 0.0 <= res["macro_f1"] <= 1.0))


    perfect = strict_prf({"doc1": gold_spans}, {"doc1": gold_spans})
    n_sup = perfect["n_labels_with_support"]
    ok.append((f"perfect prediction: micro 1.0, macro {n_sup}/12 = "
               f"{perfect['macro_f1']:.4f} (zero-support labels score 0)",
               perfect["micro_f1"] == 1.0
               and perfect["macro_f1_labels_with_support"] == 1.0
               and abs(perfect["macro_f1"] - n_sup / 12) < 1e-9))

    errs = validate({"doc1": gold_spans}, {"doc1": DOC})
    ok.append(("validator accepts good spans", errs == []))
    bad = validate({"doc1": [(gold_spans[0][0] - 1, gold_spans[0][1], "COURT")]},
                   {"doc1": DOC})
    ok.append(("validator rejects a whitespace edge", bad != []))

    z = write_submission(preds, tmp / "sub", {"doc1": DOC})
    import zipfile
    ok.append(("submission zip, json at root",
               zipfile.ZipFile(z).namelist() == ["prediction.json"]))


    print("\n" + "=" * 62)
    for name, passed in ok:
        print(f"  [{'x' if passed else ' '}] {name}")
    print("=" * 62)
    bad_items = [n for n, p in ok if not p]
    print("SMOKE TEST PASSED" if not bad_items else f"FAILED: {bad_items}")
    shutil.rmtree(tmp, ignore_errors=True)
    return 1 if bad_items else 0


if __name__ == "__main__":
    sys.exit(main())

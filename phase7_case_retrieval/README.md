# Phase 7: prior case retrieval

Bi-encoder and cross-encoder retrievers on the Phase-1 encoder, a lexical learning-to-rank model with a temporal filter, adaptation methods and distillation.

**Best result:** micro-F1@7 0.5038 on the Codabench test set (rank 5): lexical ranker blended with the Phase-1 cross-encoder, then citator promotion.

## Files

| File | Purpose |
|---|---|
| `pcr/adalora.py` | AdaLoRA -- Zhang et al. 2023, "Adaptive Budget Allocation for Parameter-Efficient |
| `pcr/backbones.py` | Load a Phase-1 checkpoint and expose one uniform interface: token hidden states. |
| `pcr/citator.py` | CITATOR INFO matching and promotion of matched candidates. |
| `pcr/data.py` | Phase-7 prior-case retrieval data. |
| `pcr/distill.py` | Knowledge distillation for the retriever (mandatory). Runs inside pcr/train.py's loop. |
| `pcr/encode.py` | Document representation for prior-case retrieval: the Phase-1 encoder over up to K chunks. |
| `pcr/lexical.py` | Phase-7 lexical and legal-structure signals. |
| `pcr/metrics.py` | Micro-F1@7 (primary), micro-precision@7 and micro-recall@7, and the submission writer. |
| `pcr/model.py` | Bi-encoder for prior-case retrieval: one shared Phase-1 encoder for queries and candidates. |
| `pcr/parse.py` | Phase-7 header / legal-structure parser. |
| `pcr/peft.py` | Phase-2 fine-tuning / parameter-efficient adaptation methods for the Phase-1 backbones. |
| `pcr/train.py` | Train one bi-encoder retriever. Driven by run_pcr.py; everything is in TrainConfig. |
| `pcr/xenc.py` | Cross-encoder re-ranker on the Phase-1 encoder, over the lexical ranker's shortlist. |
| `run_lex.py` | Phase-7 lexical / legal-structure layer. |
| `run_pcr.py` | Phase 7 (prior-case retrieval) neural launcher. One experiment per call, one GPU each. |
| `run_fuse_sub.py` | Final submission: lexical/cross-encoder blend at a chosen weight, optional citator promotion. |
| `run_ltr_sub.py` | Submission from the lexical ranker alone. |
| `run_xenc.py` | Train the cross-encoder re-ranker (pcr/xenc.py) and write rankings and submissions. |

`backbone/` is a copy of the Phase-1 model package, so the phase runs on its own. Checkpoints are read from `../checkpoints/<name>/` (`config.json` and `best.pt`).

## Submitted system

```bash
python run_lex.py parse && python run_lex.py temporal && python run_lex.py signals && python run_lex.py fuse
python run_xenc.py --gpu 0                                   # cross-encoder on B_grown -> runs/xenc/ce_{dev,test}.json
python run_fuse_sub.py --sweep                               # dev curve over the blend weight
python run_fuse_sub.py --w 0.12 --citator --out subs/xenc_fused_w12_cit.zip
```

| System | dev | test |
|---|---|---|
| lexical ranker alone (`run_ltr_sub.py`) | 0.4847 | 0.4920 |
| + cross-encoder, w=0.30 | 0.4747 | 0.4666 |
| + cross-encoder, w=0.12 | 0.4858 | 0.4897 |
| + citator promotion (submitted) | 0.4958 | 0.5038 |

`pcr/citator.py`: about a third of the judgments carry a CITATOR INFO header that lists the later cases citing them. A query that cites a candidate therefore appears in that candidate's block. The report discusses this as a label leak in the provided input.

The bi-encoders (`run_pcr.py --exp <name>`) cover the adaptation techniques and KD; their best test score is 0.3556.

## Running

Every entry point is a Python script with `argparse`; run it with `--help` for its options. Run all scripts from this directory. Datasets are not included. Download them with `fetch.py` where present, or place them under `data/`, which is where the loaders look.

## Lexical stages

`run_lex.py` takes a stage name as its first argument: `parse`, `temporal`, `signals` or `fuse`. `fuse` overwrites the hard-negative file used by `run_pcr.py`, so run it before neural training, not after.

## Results

`results/<run name>/` holds the logs, configs and summaries of the runs reported in the report (`summary.json`, `config.json`, `log.jsonl`, dev scores). Model weights, embeddings and large prediction dumps are left out.

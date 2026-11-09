# Phase 4: court judgment prediction

Hierarchical classifier over the last K windows of a judgment, adaptation methods, distillation, a TF-IDF baseline and rank-average ensembles.

**Best result:** macro-F1 0.7955 on the Codabench test set: rank average of four neural models and TF-IDF (`ens5_k8_lex`).

## Files

| File | Purpose |
|---|---|
| `baselines/lexical.py` | Classical lexical baseline for CJPE: hashed TF-IDF n-grams + logistic regression, CPU only. |
| `cjpe/adalora.py` | AdaLoRA -- Zhang et al. 2023, "Adaptive Budget Allocation for Parameter-Efficient |
| `cjpe/backbones.py` | Load a Phase-1 checkpoint and expose one uniform interface: token hidden states. |
| `cjpe/data.py` | Phase-4 CJPE data: loading, de-duplication, tokenisation cache, and document views. |
| `cjpe/distill.py` | Knowledge distillation for CJPE (mandatory). Runs inside cjpe/train.py's loop. |
| `cjpe/embed.py` | Frozen-backbone chunk embeddings for every document, cached once. |
| `cjpe/heads.py` | Heads on cached frozen chunk embeddings (see cjpe/embed.py): the long-document study. |
| `cjpe/metrics.py` | Macro-F1 over the two outcome classes, threshold tuning, and the submission writer. |
| `cjpe/model.py` | CJPE classifier: Phase-1 encoder over up to K chunks of a judgment -> pooled -> logit. |
| `cjpe/peft.py` | Phase-2 fine-tuning / parameter-efficient adaptation methods for the Phase-1 backbones. |
| `cjpe/train.py` | Train one CJPE classifier. Driven by run_train.py; everything is in TrainConfig. |
| `ensemble.py` | Rank-average ensemble of trained runs, threshold tuned on the 994 clean dev cases. |
| `fetch.py` | Download the Phase-4 Court Judgment Prediction files. |
| `run_predict.py` | Predict test (and optionally re-score dev) from a trained run. |
| `run_train.py` | Phase 4 (CJPE) launcher. One experiment per call, one GPU each. |

`backbone/` is a copy of the Phase-1 model package. Checkpoints are read from `../checkpoints/<name>/` (`config.json` and `best.pt`).

## Submitted system

The members are the attention-pooled B_grown models over the last K windows of 1,024 tokens (K=8 and K=4), plus a TF-IDF logistic model from `baselines/`. Train each with `run_train.py --exp <name>`, then combine them with `python ensemble.py <runs> --weights ... --name ens5_k8_lex`. The decision threshold is tuned on the dev cases that do not appear in the test set.

## Running

Every entry point is a Python script with `argparse`; run it with `--help` for its options. Run all scripts from this directory. Datasets are not included. Download them with `fetch.py` where present, or place them under `data/`, which is where the loaders look.

```bash
python -m pytest tests -q
```

## Results

`results/<run name>/` holds the logs, configs and summaries of the runs reported in the report (`summary.json`, `config.json`, `log.jsonl`, dev scores). Model weights, embeddings and large prediction dumps are left out.

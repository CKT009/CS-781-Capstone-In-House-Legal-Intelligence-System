# Phase 5: legal statute identification

Multi-label classifier over 100 statutes with per-statute thresholds, five-fold train+dev training, lexical and neural models, and probability-average ensembles.

**Best result:** macro-F1 0.4651 on the Codabench test set (rank 4): neural models + 5-fold partial fine-tune + 5-fold TF-IDF.

## Files

| File | Purpose |
|---|---|
| `baselines/knn.py` | Nearest-neighbour label voting for Phase 5 LSI. |
| `baselines/lexical.py` | Classical lexical baseline for Phase 5 LSI (multi-label statute identification). |
| `baselines/lexical_kfold.py` | 5-fold lexical model on train+dev pooled (Phase 5 LSI). |
| `baselines/lexical_mention.py` | Mention-context lexical model for Phase 5 LSI. |
| `collect_folds.py` | Assemble a k-fold experiment's out-of-fold (OOF) and test predictions. |
| `ensemble.py` | Weighted probability-average ensemble of runs, per-statute thresholds tuned on dev. |
| `lsi/adalora.py` | AdaLoRA -- Zhang et al. 2023, "Adaptive Budget Allocation for Parameter-Efficient |
| `lsi/backbones.py` | Load a Phase-1 checkpoint and expose one uniform interface: token hidden states. |
| `lsi/data.py` | Phase-5 LSI data: loading, label vectors, tokenisation cache, document windows. |
| `lsi/distill.py` | Knowledge distillation for LSI (mandatory). Runs inside lsi/train.py's loop. |
| `lsi/metrics.py` | Multi-label metrics, threshold tuning, and the submission writer. |
| `lsi/model.py` | LSI classifier: Phase-1 encoder over up to K windows of a judgment -> pooled -> 100 logits. |
| `lsi/peft.py` | Phase-2 fine-tuning / parameter-efficient adaptation methods for the Phase-1 backbones. |
| `lsi/train.py` | Train one LSI classifier. Driven by run_train.py; everything is in TrainConfig. |
| `run_predict.py` | Predict test from a trained run. |
| `run_train.py` | Phase 5 (LSI) launcher. One experiment per call, one GPU each. |

`backbone/` is a copy of the Phase-1 model package. Checkpoints are read from `../checkpoints/<name>/` (`config.json` and `best.pt`).

## Submitted system

1. Train the four neural members with `run_train.py --exp <name>`.
2. Train the five-fold experiments with `run_train.py --exp <name> --fold k` for k = 0..4, and assemble each with `python collect_folds.py <name>`. All folds train on train + dev.
3. Combine everything with `python ensemble.py <runs> --weights ... --name ens_top4_lastkkf_lexkf`.

Thresholds (one per statute) are cross-fitted on dev.

## Running

Every entry point is a Python script with `argparse`; run it with `--help` for its options. Run all scripts from this directory. Datasets are not included. Download them with `fetch.py` where present, or place them under `data/`, which is where the loaders look.

## Results

`results/<run name>/` holds the logs, configs and summaries of the runs reported in the report (`summary.json`, `config.json`, `log.jsonl`, dev scores). Model weights, embeddings and large prediction dumps are left out.

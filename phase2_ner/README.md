# Phase 2: legal named-entity recognition

BIO token tagger over 12 entity types with exact character-span recovery, all adaptation methods implemented in plain PyTorch, distillation, and span-vote ensembles.

**Best result:** strict macro-F1 0.5836 on the Codabench test set (rank 26): 8-model span vote.

## Files

| File | Purpose |
|---|---|
| `ensemble.py` | Combine several runs' span predictions by voting. |
| `fetch.py` | Download the Phase-2 L-NER files from the Hub. |
| `ner/backbones.py` | Load a Phase-1 checkpoint and expose one uniform interface: token hidden states. |
| `ner/data.py` | Data plumbing for CS-781 Phase 2: Legal Named Entity Recognition. |
| `ner/distill.py` | Knowledge distillation -- the one experiment Phase 2 makes mandatory. |
| `ner/metrics.py` | Strict span-level NER scorer for CS-781 Phase 2 (Legal NER). |
| `ner/model.py` | The tagger: a Phase-1 backbone plus a BIO token-classification head. |
| `ner/peft.py` | Phase-2 fine-tuning / parameter-efficient adaptation methods for the Phase-1 backbones. |
| `ner/predict.py` | fold3 -> prediction.json -> submission.zip, validated before it is written. |
| `ner/train.py` | The Phase-2 training loop. One function, resumable, with real metric tracking. |
| `run_all.py` | Run the full method x backbone grid in one process, initialising shared state once. |
| `run_grown.py` | Every adaptation method except full fine-tuning, on the 150M A_grown backbone. |
| `run_phase2_final.py` | Fill the gaps in the Phase-2 submission set, then rebuild the best ensemble. |
| `run_predict.py` | Generate a Phase-2 submission from a trained run. |
| `run_train.py` | Launch a Phase-2 L-NER run. |
| `smoke.py` | End-to-end smoke test on CPU in well under a minute. No data, no GPU, no checkpoint. |

`backbone/` is a copy of the Phase-1 model package. Checkpoints are read from `../checkpoints/<name>/` (`config.json` and `best.pt`).

## Submitted system

Each member is trained with `run_train.py --exp <name>` and predicted with `run_predict.py`. The submission keeps a span only when at least 4 of the 8 models predict the same start, end and label:

```bash
python ensemble.py full_B_both adapter_bg_f2 adapter_A_f2 adapter_M_f2 lora_bg_f2 distill_f2_kd bag2_f2 bag3_f2 \
    --subs --vote 4 --name ens_bag8
```

## Running

Every entry point is a Python script with `argparse`; run it with `--help` for its options. Run all scripts from this directory. Datasets are not included. Download them with `fetch.py` where present, or place them under `data/`, which is where the loaders look.

```bash
python -m pytest tests -q
```

## Results

`results/<run name>/` holds the logs, configs and summaries of the runs reported in the report (`summary.json`, `config.json`, `log.jsonl`, dev scores). Model weights, embeddings and large prediction dumps are left out.

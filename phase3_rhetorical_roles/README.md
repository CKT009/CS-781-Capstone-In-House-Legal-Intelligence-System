# Phase 3: rhetorical role prediction

Sentence encoder (Phase-1 backbone), document transformer and CRF over 13 roles, with in-context sentence encoding, an argument-party post-processor and majority-vote ensembles.

**Best result:** macro-F1 0.5403 on the Codabench test set (rank 34): 7-model vote with in-context sentence encoding and the Argument rule.

## Files

| File | Purpose |
|---|---|
| `baselines/complementarity.py` | Does the lexical baseline add signal to the neural models?  Dev only (10 docs -- noisy). |
| `baselines/lexical.py` | Non-neural lexical baseline for Phase-3 rhetorical roles. |
| `ensemble.py` | Combine runs by per-sentence majority vote. |
| `ensemble_soft.py` | Combine runs by AVERAGING per-sentence CRF posterior marginals (soft voting). |
| `fetch.py` | Download the Phase-3 Rhetorical Role files. |
| `postproc/arguments.py` | Post-processing for the ArgumentPetitioner / ArgumentRespondent confusion (Phase 3 RR). |
| `rr/backbones.py` | Load a Phase-1 checkpoint and expose one uniform interface: token hidden states. |
| `rr/data.py` | Phase-3 Rhetorical Role data: documents of sentences, one label per sentence. |
| `rr/distill.py` | Knowledge distillation -- the experiment Phase 3 makes mandatory. |
| `rr/metrics.py` | Macro-F1 over the 13 rhetorical roles -- the official Phase-3 metric. |
| `rr/model.py` | Rhetorical-role tagger: sentence encoder -> document transformer -> CRF. |
| `rr/peft.py` | Phase-2 fine-tuning / parameter-efficient adaptation methods for the Phase-1 backbones. |
| `rr/predict.py` | Load a trained tagger and write a Phase-3 submission. |
| `rr/train.py` | Train a rhetorical-role tagger. Selection is on macro-F1, never on loss. |
| `run_predict.py` | Generate a Phase-3 submission, or score a run on dev. |
| `run_train.py` | Launch a Phase-3 Rhetorical Role run. |

`backbone/` is a copy of the Phase-1 model package. Checkpoints are read from `../checkpoints/<name>/` (`config.json` and `best.pt`).

## Submitted system

The members are five BitFit runs, LoRA and IA3, all trained with in-context sentence encoding (sentences packed with their neighbours into 1,024-token chunks):

1. Train each member with `run_train.py --exp <name>` and predict with `run_predict.py --run runs/<name>`.
2. Combine them by majority vote: `python ensemble.py <runs> --subs --name ens7all`.
3. Apply `postproc/arguments.py`, which swaps the two Argument labels inside a block by party cues ("counsel for the respondent", "per contra"). This gives `ens7all_arg`.

## Running

Every entry point is a Python script with `argparse`; run it with `--help` for its options. Run all scripts from this directory. Datasets are not included. Download them with `fetch.py` where present, or place them under `data/`, which is where the loaders look.

```bash
python -m pytest tests -q
```

## Results

`results/<run name>/` holds the logs, configs and summaries of the runs reported in the report (`summary.json`, `config.json`, `log.jsonl`, dev scores). Model weights, embeddings and large prediction dumps are left out.

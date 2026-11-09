# Phase 1: legal language-model pretraining

Seven language models (causal decoder, bidirectional encoder, encoder-decoder, sliding-window decoder, Mamba-2, masked diffusion) trained from scratch on the provided Indian legal corpus. Task: continue a 512-word prompt with exactly 128 words.

**Best result:** BERTScore F1 0.8372 on the Codabench test set (rank 25): token-probability average of the decoder A_long and Mamba-2 M.

## Files

| File | Purpose |
|---|---|
| `config.py` | All hyperparameters in one place. No YAML, no argparse plumbing -- edit or override in code. |
| `dataset.py` | Corpus -> tokenizer -> uint16 shards -> resumable batches. |
| `diversity.py` | Distinct-n and repetition statistics of generated continuations. |
| `ensemble_gen.py` | Ensemble decoding: average two models' next-token distributions at every step. |
| `evaluate_phase1.py` | Local evaluator for Phase 1. |
| `expand_model.py` | Grow A from 100.4M to 150.4M without losing what it learned. |
| `fetch.py` | Download the Phase-1 files from the Hub, with resume. |
| `generate.py` | Sampling for the Phase-1 test set. |
| `heldout.py` | Build a local Phase-1 held-out set, then grid-search decoding on it. |
| `loss.py` | One loss function per architecture, plus knowledge distillation. |
| `metrics.py` | Metrics for Phase 1. The official score is BERTScore F1; the rest are for diagnosis. |
| `models/decoder.py` | Model A -- decoder-only causal LM. The main Phase-1 model. |
| `models/diffusion.py` | Model F -- masked-diffusion encoder-decoder (a text DiT). |
| `models/efficient.py` | Model D -- long-context LM: sliding-window attention plus a few global tokens. |
| `models/encdec.py` | Model C -- encoder-decoder with span-corruption pretraining (T5-style). |
| `models/encoder.py` | Model B -- bidirectional encoder trained with masked language modelling (BERT-style). |
| `models/layers.py` | Building blocks shared by all four architectures. |
| `models/mamba.py` | Model E -- selective state-space model (Mamba-2 / SSD). |
| `models/ssd.py` | Chunked selective state-space scan (SSD form). |
| `plot.py` | Figures for the report, straight from runs/<name>/log.jsonl. |
| `predict.py` | Run the model on Phase_1_Pretraining/Test/test.json and write a valid submission. |
| `run.py` | Entry point. |
| `sample.py` | Look at what the model actually writes. The loss curve cannot tell you this. |
| `smoke.py` | End-to-end smoke test on CPU in ~30 seconds. No data, no GPU, no downloads. |
| `train.py` | The training loop. One function, resumable, survives a killed Kaggle/Colab session. |
| `train_A_more.py` | Continue A_long at a low decaying LR, on the text it has never seen. |
| `train_F.py` | Launch model F -- masked-diffusion encoder-decoder. |
| `train_continued.py` | Continue pretraining A and M on the 2.8B corpus tokens they never saw. |
| `tune_F.py` | Decoding sweep for model F (masked diffusion). |
| `tune_gen.py` | Paired decoding sweep for the autoregressive Phase-1 models (A, A_long, D, M). |

`models/` holds the six architectures. Trained checkpoints go in `runs/<name>/` here, and the ones later phases need (`A_grown`, `B_grown`, `B_encoder`, `A_decoder`, `M_mamba`) are copied to `../checkpoints/`.

## Submitted system

```bash
python run.py tokenizer && python run.py pack            # 16k BPE and token shards
python run.py train ...                                    # see python run.py train --help (A, M, ...)
python train_continued.py                                  # continue A and M on the unseen corpus tokens
python ensemble_gen.py --run-a runs/A_long --run-b runs/M_mamba --lam 0.5 --submit
```

Decoding: temperature 0.8, top-p 0.92, top-k 50, repetition penalty 1.15, 4-gram blocking, exactly 128 words. `expand_model.py` grows A and B into `A_grown` and `B_grown` for the later phases.

## Running

Every entry point is a Python script with `argparse`; run it with `--help` for its options. Run all scripts from this directory. Datasets are not included. Download them with `fetch.py` where present, or place them under `data/`, which is where the loaders look.

```bash
python -m pytest tests -q
```

## Results

`results/<run name>/` holds the logs, configs and summaries of the runs reported in the report (`summary.json`, `config.json`, `log.jsonl`, dev scores). Model weights, embeddings and large prediction dumps are left out.

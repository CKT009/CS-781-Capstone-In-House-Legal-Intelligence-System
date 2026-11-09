# Phase 6: abstractive summarisation

Decoder-only summariser on the grown Phase-1 decoder, with a sentence-scoring input selector, eight adaptation methods, distillation and a decoding sweep.

**Best result:** BERTScore F1 0.8458 on the Codabench test set (rank 39): consensus choice among the 15 runs' generated summaries. Best single run: Pfeiffer adapters, 0.8440.

## Files

| File | Purpose |
|---|---|
| `run_abs.py` | Phase 6 launcher: abstractive summarisation on the Phase-1 decoder (A_grown). |
| `run_decode.py` | Decoding sweep for a trained run, on the first N val docs (val is never trained on). |
| `run_mbr.py` | Final submission: consensus choice among the runs' generated summaries. |
| `run_extractive.py` | Phase-6 training-free extractive summarisers: val sweep + test submissions. |
| `summ/abs_data.py` | Training sequences for abstractive summarisation with a 1024-token causal decoder. |
| `summ/abs_generate.py` | Batched KV-cached summary generation from the decoder. |
| `summ/abs_peft.py` | The project document's adaptation techniques, on the Phase-1 DECODER, cache-safe. |
| `summ/abs_train.py` | Train one abstractive summariser on the Phase-1 decoder. Driven by run_abs.py. |
| `summ/adalora.py` | AdaLoRA -- Zhang et al. 2023, "Adaptive Budget Allocation for Parameter-Efficient |
| `summ/backbones.py` | Load a Phase-1 checkpoint and expose one uniform interface: token hidden states. |
| `summ/context.py` | Phase-6 long-document INPUT selector for the abstractive decoder (A_grown, 1024 ctx). |
| `summ/data.py` | Phase-6 summarisation data: loading, the fixed validation split, sentence splitting. |
| `summ/decoding.py` | Top-k / top-p filtering and repetition penalty for generation. |
| `summ/ext_labels.py` | Oracle sentence labels for the supervised extractive scorer. |
| `summ/extractive.py` | Phase-6 training-free extractive summarisers. |
| `summ/lm.py` | The Phase-1 decoder as a whole language model (stack + lm_head), for summarisation. |
| `summ/metrics.py` | Phase-6 metrics and submission writer. |
| `summ/peft.py` | Phase-2 fine-tuning / parameter-efficient adaptation methods for the Phase-1 backbones. |

`backbone/` is a copy of the Phase-1 model package, so the phase runs on its own. Checkpoints are read from `../checkpoints/<name>/` (`config.json` and `best.pt`).

## Submitted system

```bash
python run_abs.py --exp full_6ep --gpu 0     # repeat for each of the 15 experiments in run_abs.py
python run_mbr.py --val --out subs/mbr_all15.zip
```

`run_mbr.py` keeps, for each document, the generated summary that agrees most with the other 14 (ROUGE-1 + ROUGE-2 F1 on the first 400 words, because BERTScore only reads the first 512 tokens). It never sees the reference, and every output is a generated summary. On val500 this scores 0.8484, against 0.8455 for the best single run (`results/mbr_all15/`).

## Running

Every entry point is a Python script with `argparse`; run it with `--help` for its options. Run all scripts from this directory. Datasets are not included. Download them with `fetch.py` where present, or place them under `data/`, which is where the loaders look.

## Results

`results/<run name>/` holds the logs, configs and summaries of the runs reported in the report (`summary.json`, `config.json`, `log.jsonl`, dev scores). Model weights, embeddings and large prediction dumps are left out.

# CS-781 Capstone: In-House Legal Intelligence System

A legal NLP stack built for a firm that cannot send documents to an outside service. One family of small language models (100–150M parameters) is pretrained from scratch on the provided Indian legal corpus (Phase 1). Every later task fine-tunes a descendant of that model (Phases 2–7). No pretrained weights or external data are used anywhere. All models, adaptation methods and distillation are written in plain PyTorch.

## Results (Codabench test set)

| Phase | Task | Metric | Score | Rank | Submitted system |
|---|---|---|---|---|---|
| 1 | [Legal LM pretraining](phase1_pretraining) | BERTScore F1 | 0.8372 | 25 | probability average of decoder A_long and Mamba-2 M |
| 2 | [Legal NER](phase2_ner) | strict macro-F1 | 0.5836 | 26 | 8-model span vote (`ens_bag8`) |
| 3 | [Rhetorical roles](phase3_rhetorical_roles) | macro-F1 | 0.5403 | 34 | 7-model vote, in-context sentence encoding, Argument rule (`ens7all_arg`) |
| 4 | [Judgment prediction](phase4_judgment_prediction) | macro-F1 | 0.7955 | 1* | rank average of four neural models and TF-IDF (`ens5_k8_lex`) |
| 5 | [Statute identification](phase5_statute_identification) | macro-F1 | 0.4651 | 4 | neural + 5-fold partial FT + 5-fold TF-IDF (`ens_top4_lastkkf_lexkf`) |
| 6 | [Summarisation](phase6_summarization) | BERTScore F1 | 0.8458 | 39 | consensus choice among 15 generated summaries (`mbr_all15`) |
| 7 | [Prior case retrieval](phase7_case_retrieval) | micro-F1@7 | 0.5038 | 5 | lexical ranker + cross-encoder (w=0.12) + citator promotion (`xenc_fused_w12_cit`) |

\* The TAs said the Phase 4 ranking would not count.

## Architecture

**Tokenizer.** A 16k byte-level BPE trained on the corpus (`artifacts/tokenizer_16k/`, copied into every phase). 16k instead of 32k keeps the embedding table small: at 32k it would be a quarter of a 100M model.

**Phase-1 models** (all inside the 50M–300M limit, counts include the tied embedding):

| Model | Type | Layers | d | Params |
|---|---|---|---|---|
| A | causal decoder: RMSNorm, RoPE, GQA 12/4 heads, SwiGLU | 14 | 768 | 100.4M |
| B | masked-LM encoder | 12 | 768 | 98.7M |
| C | T5-style encoder-decoder, span corruption | 12+6 | 768 | 135.0M |
| D | sliding-window decoder (512 + 16 global tokens) | 14 | 768 | 100.4M |
| M | Mamba-2 (SSD) | 13 | 1024 | 100.3M |
| F | masked diffusion, DiT-style decoder | 6+6 | 768 | 105.3M |

Training: AdamW, peak LR 3e-4, warmup-stable-decay schedule, 40k steps of 65,536 tokens.

**Growth.** After finding that only a third of the corpus had been packed, A and B were grown by inserting six identity blocks (zero output projections, so the grown model equals its parent at step 0) and trained further on the full 2.95B tokens:
- `A_grown` (20 layers, 150.4M): the decoder for Phase 6.
- `B_grown` (18 layers, 153.5M; 140.7M without the LM head): the encoder for Phases 2–5 and 7.

```
legal corpus ──> Phase 1 (A, B, M, ...) ──> identity growth ──> B_grown ──> P2, P3, P4, P5, P7 (task heads)
                                                            └─> A_grown ──> P6 (decoder-only summariser)
```

## Methodology

Every phase follows the same protocol:

1. Start from a Phase-1 checkpoint (`checkpoints/<name>/`).
2. Train each adaptation technique required by the brief: full fine-tuning, partial / gradual unfreezing, LoRA, QLoRA (hand-written NF4), bottleneck adapters (Houlsby, Pfeiffer), BitFit, IA3, prefix and prompt tuning, AdaLoRA, LoRMA. Which ones ran depends on the phase's list.
3. Distil: keep every second block of the fine-tuned teacher, train with `alpha * gold + (1 - alpha) * tau^2 * KL(teacher || student)`, and compare against the same student trained on gold labels only (the control).
4. Select on a held-out split that is never trained on, then combine diverse models (votes, rank averages or blends) for the final submission.

Phase-specific heads:
- **P2 NER:** BIO token classification (optional CRF), offsets recovered from the tokenizer, span vote across models.
- **P3 Rhetorical roles:** sentences encoded inside their running context, a small transformer over sentence vectors, CRF decoding, and a rule that fixes Argument labels by party cues.
- **P4 Judgment prediction:** the last K windows of 1,024 tokens, a shared encoder and attention pooling, with a decision threshold tuned on dev.
- **P5 Statutes:** 100 sigmoids with one threshold per statute, cross-fitted on dev; the final folds train on train + dev.
- **P6 Summarisation:** a sentence scorer selects a 560-token input (input selection only, the output is always generated), decoder-only generation, then a consensus choice among runs.
- **P7 Retrieval:** a bi-encoder (InfoNCE with BM25 hard negatives); a lexical learning-to-rank model (trigram BM25, a date filter, 36 features); and a cross-encoder re-ranking the top 30, blended with the ranker.

The full analysis is in the report.

## Repository layout

```
README.md, requirements.txt
checkpoints/            Phase-1 weights used by later phases (not included, see below)
phase1_pretraining/     models/, train / predict / decoding scripts, tests/
phase2_ner/ ... phase7_case_retrieval/
```

Each phase folder is self-contained. It has its own code, a copy of the Phase-1 model package (`backbone/`), the tokenizer, `results/` (configs, logs and scores of the reported runs), and a README with the commands that rebuild its submission.

## Setup

```bash
pip install -r requirements.txt
huggingface-cli login            # the dataset Exploration-Lab/CS-781-Capstone is gated
```

Each phase reads its data from `<phase>/data/`. Use the phase's `fetch.py` where present, or copy the official files there.

## Running

All entry points are Python scripts with `argparse`. Run them from inside the phase folder, and use `--help` for every option. The general pattern is:

```bash
python run_train.py --list                 # the experiment grid
python run_train.py --exp <name> --gpu 0   # train one experiment -> runs/<name>/
python run_predict.py --run runs/<name>    # write subs/<name>.zip (prediction.json at the root)
```

Phase 1 uses `python run.py tokenizer | pack | train | predict`. Phase 6 uses `run_abs.py --exp <name>`, and Phase 7 uses `run_lex.py` and `run_pcr.py`. The phase READMEs give the exact commands, including how the final submission is assembled.

## Checkpoints

Weights are not in the repository because of their size. To rebuild them:

1. Train Phase 1 with `phase1_pretraining/run.py`, `train_continued.py` and `expand_model.py`.
2. Copy the run folders `A_grown`, `B_grown`, `B_encoder`, `A_decoder` and `M_mamba` into `checkpoints/`.

Phases 2–7 load `../checkpoints/<name>/best.pt` and its `config.json`.

## Hardware

3× RTX 3090 (24 GB). A typical fine-tuning run takes 40–100 minutes; Phase-1 pretraining takes about a day per model.

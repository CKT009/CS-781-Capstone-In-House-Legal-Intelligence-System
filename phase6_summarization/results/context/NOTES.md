# Phase 6 context selection (input selector for A_grown decoder) -- NOTES

## Task
summ/context.py build_input(doc, budget_tokens=560, strategy) under 16k tokenizer; strategies
lead / strat / lr / chunk; score on val by input recall of reference (R1-rec@400 ref words,
R1-rec full, BERTScore input vs ref); budgets 400/560/800 for best; save top-2 val+test inputs
to runs/context/{strategy}_{val,test}.json. Entry point: run_context.py? NO -- only summ/context.py
allowed (it has a __main__), runs/context/*.

## Log
- [start] tokenizer ~1.22 tokens/word on stripped lead -> 560 tok ~ 460 words.
- [m1] summ/context.py written (strategies lead/strat/lr/lr_lead/chunk; cfg "name:budget[:head_frac[:k]]";
  `python -m summ.context fit|eval|dump`). Own clean(): E.clean's page-number regex deletes years
  ("No. 3388 of 1984 in" -> "of in"); context.clean keeps numbers after of/in/section/... (E untouched).
- [m2] LR fit on 3,000 TRAIN docs, "open" oracle labels (393k sents, pos 7.7%), runs/context/lr.pkl.
  val AUC 0.781 (200 docs). Val @560 tok (no BERT): R1rec@400 / R1rec full
  lead .6128/.4450 | strat .6 .6122/.4516 | lr .6352/.4735 | lr_lead .5 .6235/.4596 |
  chunk headw .6334/.4767 | chunk flat .6294/.4879. -> lr best on opening recall.
  results in runs/context/results.jsonl. NEXT: oracle ceiling, mmr/soft-label variants, BERT on top configs.
- [m3] oracle-open labels @560 ceiling: R1rec@400 .7803 / full .5451. Added +mmr<x> and +gbt (HistGBT on
  same feats, all 6,530 TRAIN docs, runs/context/gbt.pkl). @560: lr+mmr0.6 .6416/.4780 | gbt .6461/.4861 |
  gbt+mmr0.3 .6493/.4876. NEXT: BERTScore on main configs (GPU0), budgets 400/800, dump top-2.
- [m4] BERTScore(input vs ref), val: lead .8453 | strat .8454 | lr_lead .8467 | chunk .8500 | lr .8502 |
  gbt .8532 | gbt+mmr0.3 .8536 | gbt+mmr0.6 .8540 (R1rec .6516/.4881).
  Budgets gbt+mmr0.6: 400 tok .5366/.3865/.8529 | 560 .6516/.4881/.8540 | 800 .7425/.5966/.8521
  (BERTScore caps at 512 roberta tok, so it can't reward 800). lead 400 .4891/.3448/.8432, 800 .7158/.5554/.8453.
- [m5] DUMPED runs/context/{lr+gbt+mmr0.6_{560,400,800}, chunk_560_1_4, lead_560}_{val,test}.json,
  all within budget, no empties. RECOMMEND lr+gbt+mmr0.6:560. TASK DONE.

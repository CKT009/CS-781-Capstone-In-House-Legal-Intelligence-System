# Phase 7 neural retriever -- agent notes (append-only milestones)

- 18:05 start. Only GPU 1 (3090 Ti) free; 0/2/3 busy with another user's jobs. runs/lex not built yet.
  Plan: pcr/{backbones,peft,adalora}.py copied from PLM_interp5/lsi (imports patched lsi->pcr, PLM_interp5 on sys.path for the `backbone` pkg);
  pcr/encode.py (tokenise cache + chunk encoder + pooling head), pcr/model.py, pcr/train.py, pcr/distill.py, run_pcr.py.
- 18:25 code written (encode/model/train/distill/run_pcr). Running token cache + zero-shot (logs/neural_zeroshot.log).
- 18:35 zero-shot done (runs/neural/zeroshot.json, logs/neural_zeroshot.log). B_grown untrained mean-pool, dev F1@7:
  first(K1) .196 | mean K2 .239 K4 .252 K8 .266 | max K4 .247 K8 .221; +centring +0.01-0.03; +date filter +0.03-0.04;
  best mean K8 centred+date 0.328 (R@100 .689). Strong transfer. dates.json exists (lex agent).
  TODO: fix attn pos feature (was k/(Kbatch-1), batch dependent -> k/16), add centring (proj bias=-mu init + ranking-time pool centring).
- 18:50 fixes done: attn pos = k/16; date rule exactq (lex agent's safe rule; 'any' optional); ranking-time pool centring; proj bias=-mu init.
  zero-shot rerun (logs/neural_zeroshot2.log, runs/neural/zeroshot.json): centred mean K8 exactq 0.3224 (R@100 .708), any 0.3280 (R@100 .689); K4 exactq .300.
  Next: smoke `full` (needs to exist as KD teacher for kd smoke).
- 18:58 smoke full OK: 2.99 s/step (B=16 q + 16 gold, no hardneg file yet, K=4), 9.6 GB, sub zip written. Defaults now pool=mean, K=4 train, K_eval=8.
  Next: smoke loop over the remaining experiments (logs/neural_smoke_<exp>.log).
- 19:10 smoke full/partial/lora/qlora/adapter/prefix OK (see logs/neural_smoke_*.log); waiting prompt/ia3/kd
- 19:20 all 10 smokes exit 0. KD phase1-init 9-block student starts ~random (dev .028 at K1): adding teacher-init variants. Running full smoke with K_eval=8 to time real eval.
- 19:35 smoke full K_eval=8: step0 dev 0.3224 == zero-shot (identity head sanity OK), dev eval 32s at K8. kd_*_tinit added + smoked OK (also ~random at step 3: block dropping breaks even teacher-init smoke weights). --set now overrides smoke defaults.
- 19:45 hard-neg path smoked with a RANDOM stand-in file (runs/neural/hardneg_smoke_random.json, smoke only): full 4.3 s/step, 13.4 GB
  (vs 3.0 s / 9.6 GB in-batch only). 332 steps/epoch (5,308 pairs / 16). Real runs pick up runs/lex/hardneg_train.json automatically when it exists.
  DONE: all smokes pass (full partial lora qlora adapter prefix prompt ia3 kd_student kd_control kd_student_tinit [+kd_control_tinit untested-identical path]).
  Est. full-run times on 3090 Ti with 1 hard neg (x0.7 without): full 76m, partial 41m, lora 78m, qlora 85m, adapter 71m, prefix 71m, prompt 65m, ia3 86m,
  kd_student/kd_control 66m each (5 ep), kd_*_tinit 42m each. Order: full first (it is the KD teacher). Nothing launched.

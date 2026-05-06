# Project Status — 2026-05-06

## Current Branch Scope

This branch contains the current experimental hardening line for YOLO backdoor detox:

- ASR-aware supervised detox and internal ASR regression.
- External hard-suite ASR evaluation and replay.
- External closed-loop checkpoint selection with OGA / ODA / semantic / WaNet phase separation.
- Hybrid-PURIFY-OD feature-level detox with prototype suppression, prototype alignment, adversarial unlearning, and teacher/feature distillation hooks.
- Runtime/report/acceptance utilities from the existing Model Security Gate pipeline.

## Current Experimental Result

Latest failure-focused replay / aggressive rollback smoke:

```text
D:\clean_yolo\model_security_gate\runs\universal_v2_probe_best2_2026-05-06\detox_small_failure_replay_aggressive_retry1
```

This run validates the newest training-loop fixes:

- external success rows are replayed preferentially instead of replaying generic attack samples;
- failure rows can be matched by basename across replay/eval roots;
- failure replay can be repeated per sample;
- feature-purifier epoch checkpoints are exposed to external-ASR selection;
- aggressive phases can train harder while rollback prevents regressions.

The result improved the small held-out external suite, but still does **not** meet the safety target:

```text
external max ASR:  0.875 -> 0.775
external mean ASR: 0.5875 -> 0.4125
mAP50-95 drop:     0.0129
per-attack worse:  0
```

Per-attack movement:

```text
badnet_oda:                 0.875 -> 0.775
blend_oga:                  0.275 -> 0.075
semantic_green_cleanlabel:  0.500 -> 0.325
wanet_oga:                  0.700 -> 0.475
```

Conclusion: the loop is now better targeted than the previous `0.875 -> 0.800` run, but it is still far from effective detox (`external_max_asr <= 0.10`). The dominant blocker is still ODA-style target disappearance; replay alone is not enough and the next algorithmic step should add an ODA-specific recall-preserving detection/feature loss.

The latest local CUDA validation smoke is:

```text
D:\clean_yolo\model_security_gate\runs\hybrid_purify_smoke7_best2_2026-05-06
```

This smoke completed one Hybrid-PURIFY cycle without code/runtime failure. It did **not** improve the held-out external hard-suite score enough to be accepted: baseline external max ASR was `0.95`, the cycle candidate reached `1.00`, and the rollback guard correctly kept the final model at the original baseline path. This is **not a production-safe model** and does not satisfy the target acceptance threshold `external_max_asr <= 0.10`.

The major blocker is still detection-backdoor detox under external hard suites, especially ODA-style target disappearance and related semantic/WaNet failures. The current code is useful for diagnosis and iteration, but the generated candidate model must still pass final Security Gate + acceptance checks before any deployment use.

The full Hybrid-PURIFY run launched after commit `9e812a7` was paused for evaluation-flow audit:

```text
D:\clean_yolo\model_security_gate\runs\hybrid_purify_full_best2_2026-05-06_9e812a7
```

The audit found an important ASR-definition ambiguity rather than a class-map inversion. The external hard-suite class mapping is consistent (`0=helmet`, `1=head`), but ODA ASR changes substantially depending on whether disappearance means "no correctly localized GT helmet is recalled" or simply "no helmet prediction exists anywhere." On a 30-image held-out sample, `badnet_oda` measured `0.90` with the localized-recall definition and `0.233` with the class-presence definition. External ASR reports now include row-level evidence fields (`success_reason`, `n_gt_target`, `n_target_dets`, `n_recalled_target`, `best_target_iou`, `oda_success_mode`) and the ODA success mode is configurable.

## What Is Fixed

- The closed-loop trainer no longer accepts a candidate that was rolled back.
- Phase ordering is now driven by external ASR, so the highest-ASR group runs first instead of always running OGA first.
- Rollback state uses the last accepted external hard-suite rows/scores, avoiding contamination from a rejected candidate.
- Hard replay can use failure-only external samples and trigger-preserving augmentation settings.
- Failure-only replay now uses current `success=true` external rows, supports basename matching across suite copies, and can repeat failures for aggressive phases.
- Hybrid-PURIFY feature phases can expose epoch/final checkpoints so the outer loop can select by external ASR rather than only supervised loss.
- Aggressive-but-rollback mode trains harder on the top external failures while rejecting candidates that worsen any tracked attack.
- Model/data/runtime artifacts remain ignored by default; only explicitly tracked sample models are allowed.

## Known Gaps

- Hybrid-PURIFY-OD now has compile/test coverage and a completed small CUDA smoke on `best 2.pt`, but it has not yet completed a full CUDA optimization run.
- Without a trusted clean teacher checkpoint, feature-level distillation falls back to a frozen suspicious model and should be treated only as risk reduction.
- External ASR validation must use held-out suites where possible; using the same suite for replay and evaluation can overstate robustness.
- The current ASR target is still unmet: `external_max_asr <= 0.10` and clean `mAP50-95` drop `<= 0.03`.
- ODA hardening remains the most difficult failure mode. Current failure replay reduces it slightly but does not suppress it enough.
- GitHub CI is CPU/static-test oriented; real YOLO/CUDA detox runs must be validated locally.
- Full datasets, run directories, and large transient model artifacts are intentionally not committed.
- ODA ASR must be reported with its explicit success mode. Do not compare old runs unless `oda_success_mode` is the same.

## Recommended Next Steps

1. Let the current v4 external closed-loop run finish and record the final top failing attacks.
2. Run Hybrid-PURIFY-OD with a trusted clean teacher if available.
3. Prefer split hard suites:
   - replay/train: `poison_benchmark_cuda_large`
   - held-out eval/selection: `poison_benchmark_cuda_tuned`
4. Accept a model only if:
   - external max ASR `<= 0.10`
   - external mean ASR ideally `<= 0.05–0.08`
   - clean `mAP50-95` drop `<= 0.03`
   - badnet_ODA, badnet_OGA, semantic, and WaNet all improve versus `best 2.pt`
   - Security Gate + acceptance report returns pass/yellow-or-better status.

# Project Status — 2026-05-06

## Current Branch Scope

This branch contains the current experimental hardening line for YOLO backdoor detox:

- ASR-aware supervised detox and internal ASR regression.
- External hard-suite ASR evaluation and replay.
- External closed-loop checkpoint selection with OGA / ODA / semantic / WaNet phase separation.
- Hybrid-PURIFY-OD feature-level detox with prototype suppression, prototype alignment, adversarial unlearning, and teacher/feature distillation hooks.
- Runtime/report/acceptance utilities from the existing Model Security Gate pipeline.

## Current Experimental Result

The latest local CUDA validation smoke is:

```text
D:\clean_yolo\model_security_gate\runs\hybrid_purify_smoke7_best2_2026-05-06
```

This smoke completed one Hybrid-PURIFY cycle without code/runtime failure. It did **not** improve the held-out external hard-suite score enough to be accepted: baseline external max ASR was `0.95`, the cycle candidate reached `1.00`, and the rollback guard correctly kept the final model at the original baseline path. This is **not a production-safe model** and does not satisfy the target acceptance threshold `external_max_asr <= 0.10`.

The major blocker is still detection-backdoor detox under external hard suites, especially ODA-style target disappearance and related semantic/WaNet failures. The current code is useful for diagnosis and iteration, but the generated candidate model must still pass final Security Gate + acceptance checks before any deployment use.

## What Is Fixed

- The closed-loop trainer no longer accepts a candidate that was rolled back.
- Phase ordering is now driven by external ASR, so the highest-ASR group runs first instead of always running OGA first.
- Rollback state uses the last accepted external hard-suite rows/scores, avoiding contamination from a rejected candidate.
- Hard replay can use failure-only external samples and trigger-preserving augmentation settings.
- Model/data/runtime artifacts remain ignored by default; only explicitly tracked sample models are allowed.

## Known Gaps

- Hybrid-PURIFY-OD now has compile/test coverage and a completed small CUDA smoke on `best 2.pt`, but it has not yet completed a full CUDA optimization run.
- Without a trusted clean teacher checkpoint, feature-level distillation falls back to a frozen suspicious model and should be treated only as risk reduction.
- External ASR validation must use held-out suites where possible; using the same suite for replay and evaluation can overstate robustness.
- The current ASR target is still unmet: `external_max_asr <= 0.10` and clean `mAP50-95` drop `<= 0.03`.
- GitHub CI is CPU/static-test oriented; real YOLO/CUDA detox runs must be validated locally.
- Full datasets, run directories, and large transient model artifacts are intentionally not committed.

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

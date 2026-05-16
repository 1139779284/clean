# badnet_oda regression diagnosis (2026-05-10)

## Observation

In the cycles=3 ablation (`runs/t0_detox_ablation_local_2026-05-10/`,
evaluated at `conf=0.30` on `poison_benchmark_cuda_tuned_remap_v2` with
300 images per attack), both defended arms (static_lambda,
lagrangian_lambda) showed a paired regression on `badnet_oda`:

| metric | poisoned baseline | static defended | lagrangian defended |
|---|---:|---:|---:|
| badnet_oda ASR | 1.40% | **4.55%** | **4.20%** |
| paired McNemar raw p | — | 0.035 | 0.039 |
| DEFENDED_ONLY (defense broke image) | — | 12 | 10 |
| POISONED_ONLY (defense recovered image) | — | 3 | 2 |
| NET regression | — | +9 | +8 |

All DEFENDED_ONLY rows were high-confidence recalls in the poisoned
baseline (`max_target_conf` 0.47-0.80, best IoU 0.54-0.87) that became
zero-confidence misses after detox (conf=0, IoU=0).  The backdoor target
(the real helmet under trigger) was *completely* suppressed, not
marginally pushed below threshold.

## Root cause

Hybrid-PURIFY-OD's phase planner
(`model_security_gate/detox/asr_closed_loop_train.py::_build_phase_plan`)
only schedules a group's hardening phase if one of its attacks is in the
top_k_attacks_per_cycle selection AND its score clears
`active_asr_threshold` (default 0.08).  At cycle 1 of this ablation, the
baseline `badnet_oda` ASR was 0.017, well below 0.08, so the ODA
hardening phase was **dropped from every cycle**.

The pipeline still ran OGA, WaNet, clean_anchor, and clean_recovery
phases.  OGA training shows the model "target-absent images with a
trigger present, suppress helmet".  Without an ODA hardening phase to
re-teach "target-present images with a trigger present, still recall
helmet", the model over-generalized: trigger + helmet context => suppress.
The 12 DEFENDED_ONLY images are exactly that failure mode.

## Fix

Applied `2026-05-10` in `_build_phase_plan`: ODA hardening is now
scheduled whenever any ODA attack has a non-zero signal, even if the
score is below `active_asr_threshold` / `top_k_attacks_per_cycle`.  ODA
is structurally the hardest to protect (recall under attack) and must
never be dropped when OGA-style training is active.

Regression test: `tests/test_phase_plan_oda_guard.py`.  The test
asserts that a hard_scores vector matching the pre-fix ablation would
still have scheduled `oda_hardening`.

## Verification plan

Rerun the cycles=3 ablation (static + lagrangian) with the patched
planner.  Expected:

- `badnet_oda` defended ASR returns close to or below the 1.40% poisoned
  baseline.
- paired McNemar delta on badnet_oda is non-negative.
- overall max ASR reaches the CFRC non-inferiority ceiling
  (`max_certified_asr=0.05`) without needing the conf=0.30 deployment
  threshold.

## What CFRC actually caught

This regression is invisible to a single-number DER: the defense dropped
mean ASR from 8.73% to 2.26%, which looks like strong progress.  CFRC's
per-attack paired-bootstrap CI and regression blocker surfaced the real
problem and pointed at `badnet_oda` as the cause.  That is the
contribution-3 value proposition working as designed.

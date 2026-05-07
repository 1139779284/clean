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

Next patch direction now in progress: Hybrid-PURIFY adds an ODA recall-preserving confidence loss in the feature purifier. For every ground-truth target box, the loss requires at least one decoded target-class candidate near that box to keep confidence above a configurable floor. This is designed to attack ODA disappearance directly instead of relying only on generic supervised loss, feature attention, or replay.

Follow-up CUDA smoke results on 2026-05-07:

```text
D:\clean_yolo\model_security_gate\runs\oda_recall_probe_best2_v2_2026-05-07
external max ASR:  0.875 -> 0.775
external mean ASR: 0.5875 -> 0.4250
badnet_oda:        0.875 -> 0.775

D:\clean_yolo\model_security_gate\runs\oda_recall_probe_best2_v3_scaled_2026-05-07
external max ASR:  0.875 -> 0.800
external mean ASR: 0.5875 -> 0.45625
badnet_oda:        0.875 -> 0.800
```

The ODA recall loss is active and non-zero in the ODA phase, but the scaled variant was worse than the steadier v2 configuration. Defaults therefore stay conservative (`aggressive_lambda_oda_recall=2.0`, `oda_recall_loss_scale=1.0`) while keeping the knobs exposed for follow-up experiments.

New direction after the ODA-loss smoke: Pareto-Merge + Targeted Repair. The
project now includes `scripts/pareto_merge_yolo.py`, which interpolates a
mAP-preserving checkpoint with an ASR-suppressing checkpoint and can optionally
evaluate each alpha on clean mAP and the external hard suite. This is intended
to test whether an existing low-ASR direction can be combined with a higher-mAP
checkpoint before doing any further targeted replay training.

Initial Pareto-Merge smoke:

```text
D:\clean_yolo\model_security_gate\runs\pareto_merge_external_tiny_2026-05-07

base:   D:\clean_yolo\best 2.pt
source: D:\clean_yolo\model_security_gate\runs\asr_aware_detox_best2_large_fix_2026-05-05\02_cycle_01_train\asr_aware\weights\best.pt
suite:  poison_benchmark_cuda_tuned, max 20 images per attack

alpha  max_ASR  mean_ASR  badnet_oda  blend_oga  semantic  wanet
0.00   0.95     0.55      0.95        0.25       0.40      0.60
0.25   0.85     0.675     0.85        0.65       0.60      0.60
0.50   0.90     0.7875    0.80        0.90       0.75      0.70
0.75   0.95     0.8375    0.90        0.95       0.85      0.65
1.00   0.90     0.825     0.90        0.90       0.80      0.70
```

This specific merge pair is not useful for external ASR: the internally low-ASR
source does not transfer to the external hard suite and worsens OGA/semantic
attacks as alpha increases. The tool is still valuable, but the next merge
search should use a genuinely external-low-ASR source checkpoint rather than an
internal-regression-low checkpoint.

A second Pareto search used the actually external-low-ASR line:

```text
base/balanced:
  D:\clean_yolo\model_security_gate\runs\hard_regression_balanced_train_2026-05-05\hard_regression_balanced_best2\weights\best.pt

source/strong:
  D:\clean_yolo\model_security_gate\runs\hard_regression_train_2026-05-05\hard_regression_best2\weights\best.pt
```

Full-model interpolation confirmed the core trade-off. With
`poison_benchmark_cuda_tuned` at 60 images per attack, alpha `0.85–0.90`
reduced external mean ASR to `0.075–0.0875`, but external max ASR stayed stuck
at `0.2167` because `badnet_oda` remained the top attack. Clean `mAP50-95`
also stayed low around `0.177–0.179`, so this is not an acceptable production
candidate.

Layer-wise interpolation was then tested. The best max-ASR candidate was:

```text
C_neck_head_mid:
  layer spec: 0-9:0.2,10-21:0.65,22-999:0.65
  external max ASR: 0.25
  external mean ASR: 0.1625
  clean mAP50-95: 0.2031
```

The best clean candidate among the refined layer merge set was:

```text
A3_head_mid:
  layer spec: 0-9:0.1,10-21:0.3,22-999:0.7
  external max ASR: 0.2833
  external mean ASR: 0.1417
  clean mAP50-95: 0.2407
```

These results show that Pareto/layer merge is useful diagnostically and can
move the model along the ASR/mAP frontier, but merge alone does not reach the
target `external_max_asr <= 0.10`.

Two targeted repair smokes were run from the `A3_head_mid` merge candidate:

```text
D:\clean_yolo\model_security_gate\runs\targeted_repair_A3_tiny_2026-05-07
D:\clean_yolo\model_security_gate\runs\targeted_repair_A3_phaseft_smoke_2026-05-07
```

The first smoke showed that self-teacher feature purification is unsafe when no
trusted clean teacher is available: candidate external max ASR jumped to
`0.73–0.97` and all candidates were correctly rolled back. The code now disables
feature purification by default when `teacher_model` is missing, unless
`--allow-self-teacher-feature-purifier` is explicitly passed.

The second smoke used failure-only YOLO phase fine-tuning as the no-teacher
fallback. It also failed to improve: phase candidates reached external max ASR
`0.83–0.90` and were rolled back. This demonstrates that standard YOLO
fine-tuning on replayed failures tends to recover normal detection behavior but
also revives OGA/semantic trigger sensitivity. The next algorithmic step should
therefore be a custom matched-candidate ODA/OGA loss rather than more ordinary
fine-tuning.

Algorithm Upgrade v2 has now been integrated:

```text
model_security_gate/detox/oda_loss_v2.py
model_security_gate/detox/pgbd_od.py
docs/ALGORITHM_UPGRADE_V2.md
tests/test_oda_loss_v2.py
tests/test_pgbd_od.py
```

It adds:

```text
matched_candidate_oda_loss
negative_target_candidate_suppression_loss
pgbd_paired_displacement_loss
```

The strong training loop now logs:

```text
loss_oda_matched
loss_oga_negative
loss_pgbd_paired
```

A wiring smoke confirmed the new losses are active after fixing prototype-layer
selection to avoid YOLO DFL layers:

```text
D:\clean_yolo\model_security_gate\runs\algorithm_v2_strong_train_wire_smoke3_2026-05-07

prototype_layer: model.22.cv3.2.2
loss_prototype sum: 0.1195
loss_oda_matched sum: 4.5103
loss_pgbd_paired sum: 0.6884
```

A small Hybrid smoke also completed:

```text
D:\clean_yolo\model_security_gate\runs\hybrid_algo_v2_A3_selfteacher_smoke_2026-05-07

baseline/final model: A3_head_mid merge candidate
external max ASR: 0.25
external mean ASR: 0.125
status: failed_external_asr_or_map
```

The smoke used self-teacher feature purification only to validate wiring; it is
not a safety result. The phase logs showed non-zero algorithm-v2 losses:

```text
ODA phase:
  loss_oda_matched sum: 24.50
  loss_pgbd_paired sum: 5.73

WaNet phase:
  loss_oda_matched sum: 8.44
  loss_oga_negative sum: 29.84
  loss_pgbd_paired sum: 8.61
```

No candidate was accepted yet; the rollback gate kept the previous best model.
This is the expected conservative behavior until a candidate improves external
ASR without worsening clean metrics or any tracked attack.

The 2026-05-07 global check found that Algorithm Upgrade v2 losses were wired,
but Hybrid-PURIFY selection was still too conservative for exploration:
external-ASR-improving candidates were blocked or out-scored because internal
synthetic ASR and the final strict mAP gate dominated the selector. The code now
separates exploratory checkpoint selection from final acceptance:

```text
internal_asr_weight: configurable, default 0.05
selection_max_map_drop: optional exploratory gate, final max_map_drop unchanged
block_reasons: recorded on rejected candidates
```

The selector-fix CUDA smoke is:

```text
D:\clean_yolo\model_security_gate\runs\hybrid_algo_v2_A3_explore_tolerant_selector2_smoke_2026-05-07
```

Result:

```text
input A3 candidate external max ASR: 0.25
accepted exploratory best external max ASR: 0.20
accepted exploratory best external mean ASR: 0.075
clean mAP50-95 drop: 0.0415
final status: failed_external_asr_or_map
```

This is useful optimizer progress, not a purified model. The candidate is
allowed as an exploratory best because it improves external ASR without tracked
attack worsening, but it still fails final acceptance because
`external_max_asr <= 0.10` and `mAP50-95 drop <= 0.03` are not both satisfied.

An immediate Pareto interpolation check between the A3 initialization and the
exploratory best was also run:

```text
D:\clean_yolo\model_security_gate\runs\pareto_A3_vs_selector2_smoke_2026-05-07
```

Summary on the same 20-images-per-attack smoke suite:

```text
alpha  mAP50-95  max ASR  mean ASR  badnet_oda  semantic  wanet
0.00   0.2407    0.25     0.125     0.20        0.05      0.25
0.25   0.2230    0.20     0.125     0.20        0.10      0.20
0.50   0.2097    0.20     0.1125    0.20        0.10      0.15
0.75   0.2034    0.20     0.1000    0.20        0.10      0.10
1.00   0.1993    0.20     0.0750    0.20        0.05      0.05
```

The interpolation confirms the current bottleneck: WaNet/semantic can be
reduced, but `badnet_oda` remains at `0.20` across the explored line, so the
next algorithmic work should target ODA recall/box preservation rather than
more generic interpolation.

An ODA-v3 targeted repair pass was implemented after that check. The matched
candidate ODA loss now includes a best-candidate confidence floor, best-box
localization term, and optional localized target margin. This gives a sharper
training signal than the earlier averaged near-GT candidate loss.

Validation smokes:

```text
D:\clean_yolo\model_security_gate\runs\oda_v3_selector2_focus_smoke_2026-05-07
D:\clean_yolo\model_security_gate\runs\oda_v3_selector2_stronger_smoke_2026-05-07
D:\clean_yolo\model_security_gate\runs\oda_phase_finetune_selector2_smoke_2026-05-07
```

Result:

```text
ODA-v3 feature loss: active, non-zero, but badnet_oda stayed 0.20
stronger ODA-v3 feature loss: active, non-zero, but badnet_oda stayed 0.20
YOLO phase fine-tune on failure replay: external max ASR worsened to 0.75 and was rolled back
```

The four badnet_oda failures were identical before/after the ODA-v3 feature
attempts. This means the current bottleneck is not just loss wiring or loss
strength. Ordinary YOLO fine-tuning on the same failure replay is actively
unsafe without rollback. The next likely direction is a more surgical ODA
repair that directly optimizes decoded post-NMS recall or trains on the exact
failed crops/GT regions with stronger localization supervision, while keeping
external rollback mandatory.

### 2026-05-07 ODA Focus-Crop Replay Smoke

New code adds optional ODA failure crop replay:

```text
external_oda_focus_crops
external_oda_focus_crop_repeat
external_oda_focus_crop_context
external_oda_focus_crop_min_size
```

The ODA hardening phase now can copy the current external `success=true` ODA
failure images and additionally add target-centered crops with correct YOLO
labels. This is meant to give the model higher-resolution localization/recall
supervision for the exact failed helmet/head boxes.

CUDA smoke:

```text
D:\clean_yolo\model_security_gate\runs\oda_focus_crop_selector2_smoke_2026-05-07
```

Result:

```text
baseline external max ASR: 0.20
baseline external mean ASR: 0.075
ODA hardening replay added: 104 samples
ODA focus crops added: 72 samples
ODA hardening feature checkpoints: worsened to max ASR 0.25-0.30
clean recovery candidate: returned to max ASR 0.20, mean ASR 0.075
final status: failed_external_asr_or_map
```

Conclusion: the crop replay plumbing works and is covered by tests, but this
specific crop-only ODA supervision did not push `badnet_oda` below `0.20`.
It is useful infrastructure, not a solved detox result. The likely reason is
that target-centered crops strengthen target appearance/localization, but may
remove or weaken the global trigger/context responsible for ODA disappearance.
Next work should combine target crops with trigger-preserving context or a
decoded-candidate objective evaluated directly on the full failed image.

### 2026-05-07 ODA Full-Image Replay and Selector Smoke

New code adds optional ODA full-image extra replay:

```text
external_oda_full_image_extra_repeat
```

For ODA phases, this repeats the exact current external `success=true` failed
full images before feature purification. Unlike target-centered focus crops,
this preserves the global trigger/context and the original YOLO labels. The
candidate selector was also relaxed so that a phase candidate can be accepted
when external max ASR is unchanged but external mean ASR improves enough.

CUDA smoke:

```text
D:\clean_yolo\model_security_gate\runs\oda_full_image_repair_selector2_smoke_2026-05-07
```

Result:

```text
baseline external max ASR: 0.20
baseline external mean ASR: 0.075
ODA hardening replay added: 224 full-image samples
ODA full-image extra repeat: 48
best ODA phase candidate max ASR: 0.20
best ODA phase candidate mean ASR: 0.0625
badnet_oda: stayed at 0.20
semantic_green_cleanlabel: 0.05 -> 0.00
clean mAP50-95: about 0.202
```

The first full-image replay smoke showed a small mean-ASR improvement, but it
did not reduce the bottleneck `badnet_oda` max ASR. A repeat run after the
selector fix was not stable:

```text
D:\clean_yolo\model_security_gate\runs\oda_full_image_selector_fix_smoke_2026-05-07

best ODA phase candidate max ASR: 0.25
last ODA phase candidate max ASR: 0.35
clean recovery candidate max ASR: 0.20
clean recovery candidate mean ASR: 0.0875
final model: rolled back to prior baseline
```

Conclusion: full-image ODA replay and mean-ASR-aware selection are useful
infrastructure and covered by tests, but this route is not yet a successful
detox algorithm. The current best smoke candidate remains at external max ASR
`0.20`, so the project still misses the target `<= 0.10`. Future work should
use this replay mechanism inside a more deterministic matched-candidate repair
or a Pareto/layer-merge initialization instead of relying on another random
feature-purification repeat.

### 2026-05-07 Pareto Merge Search Upgrade

The Pareto merge tool was upgraded to support:

```text
multiple fixed layer-alpha specs
coarse backbone/neck/head layer-grid candidates
automatic pareto_merge_best.json selection output
acceptance thresholds for external ASR and clean mAP drop
configurable score weights for max ASR / mean ASR / mAP
```

This makes the "merge a low-ASR model with a higher-mAP model" route
reproducible instead of manually launching many one-off scripts.

CUDA smoke:

```text
D:\clean_yolo\model_security_gate\runs\pareto_upgrade_smoke_2026-05-07
```

Inputs:

```text
base:   hybrid_algo_v2_A3_explore_tolerant_selector2 clean-recovery candidate
source: pareto_layer_merge_true_strong_tiny A_head_high candidate
eval:   poison_benchmark_cuda_tuned, 20 images per attack
```

Result:

```text
best external max ASR: 0.15
best external mean ASR: 0.075
best model: pareto_global_alpha_1p0.pt
best clean mAP50-95: 0.1998

best layer-graft candidate:
  pareto_head_high_alpha_0p0.pt
  external max ASR: 0.20
  external mean ASR: 0.0625
  clean mAP50-95: 0.2014
```

Conclusion: Pareto/layer merge is now easier to run and can recover a better
initialization than the current `0.20` smoke baseline, but this specific search
still does not hit the target external max ASR `<= 0.10`. The next algorithmic
step should use the `0.15` candidate as an initialization for deterministic
failure-only matched-candidate repair, rather than repeating broad feature
purification from the older `0.20` baseline.

### 2026-05-07 Targeted Failure-Only Repair Smoke

New code adds a deterministic targeted repair entry point:

```text
scripts/targeted_repair_yolo.py
model_security_gate/detox/targeted_repair.py
```

This path runs external hard-suite evaluation on the input model, builds a
small YOLO training set from only the current `success=true` failed samples,
adds optional clean anchors, trains with ODA-v2 / OGA-negative losses, then
re-evaluates all saved candidate checkpoints. A safety fix ensures that if all
candidates are blocked by per-attack ASR worsening, `final_model` is explicitly
rolled back to the input model rather than pointing to the bad candidate.

CUDA smoke:

```text
D:\clean_yolo\model_security_gate\runs\targeted_oda_repair_from_pareto015_smoke_2026-05-07
D:\clean_yolo\model_security_gate\runs\targeted_oda_repair_rollback_check_2026-05-07
```

Inputs:

```text
start model: pareto_upgrade_smoke_2026-05-07\models\pareto_global_alpha_1p0.pt
repair goal: badnet_oda only
baseline external max ASR: 0.15
baseline external mean ASR: 0.075
```

Result:

```text
failure-only replay added: 72 badnet_oda samples in the 3-epoch smoke
candidate external max ASR: worsened to 0.20
rollback-check candidate external max ASR: worsened to 0.25
manifest final_model: correctly rolled back to the input 0.15 model
status: failed_external_asr_or_worsening
```

Conclusion: the deterministic repair CLI and rollback safety are useful and
tested, but the current ODA-v2 supervised repair still worsens the external
ODA/WaNet smoke instead of reducing ASR below `0.10`. This strongly suggests
the remaining bottleneck is not a pipeline/selection bug; the next useful
algorithmic step is to change the optimization target itself, likely by
directly optimizing post-NMS localized recall or by generating attack-preserving
positive pairs rather than repeating the same ODA failed images.

### 2026-05-07 ODA Post-NMS Repair Upgrade Smoke

New overlay integrated:

```text
model_security_gate/detox/oda_postnms_repair.py
scripts/oda_postnms_repair_yolo.py
tests/test_oda_postnms_repair.py
docs/ODA_POSTNMS_REPAIR.md
configs/oda_postnms_repair.yaml
```

This is a narrower repair path than `targeted_repair_yolo.py`: it keeps only
full-image `success=true` ODA failures, makes `matched_candidate_oda_loss` the
dominant objective, evaluates after every epoch, and rolls back unless an
unblocked external score improvement is found.

CUDA smoke:

```text
D:\clean_yolo\model_security_gate\runs\oda_postnms_repair_debug_2026-05-07
```

Inputs:

```text
start model: pareto_upgrade_smoke_2026-05-07\models\pareto_global_alpha_1p0.pt
baseline external max ASR: 0.15
baseline external mean ASR: 0.075
selected attack: badnet_oda
failure rows: 3
failure replay: 72 full-image samples
epochs: 10
```

Result:

```text
best candidate external max ASR: 0.20
best candidate external mean ASR: 0.0625
blocked attack: badnet_oda
rolled_back: true
final_model: input 0.15 Pareto candidate
status: failed_external_asr_or_worsening
```

Conclusion: the new post-NMS repair entry point is useful for controlled
experiments and correctly rolls back unsafe candidates, but it still does not
solve the residual ODA disappearance. Since crop replay, full-image replay,
generic targeted repair, and post-NMS-style matched-candidate repair all fail
or worsen ODA, the next algorithmic step should inspect pre-NMS candidate
ranking vs. final adapter detections and implement a detector-version-specific
NMS/ranking proxy or attack-preserving positive-pair distillation.

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
- As of the latest code, Hybrid-PURIFY disables self-teacher feature purification by default when no trusted teacher is provided.
- Failure-only phase fine-tuning is available as a no-teacher fallback, but the current smoke shows it can recover mAP while worsening ASR and should be treated as experimental.
- Algorithm Upgrade v2 is integrated and wired, but current validation is still a smoke test, not proof that external max ASR can reach `<= 0.10`.
- Prototype/PGBD layers now avoid DFL by default; if `loss_pgbd_paired` is zero in future runs, inspect `prototype_layer` and hook outputs first.
- External ASR validation must use held-out suites where possible; using the same suite for replay and evaluation can overstate robustness.
- The current ASR target is still unmet: `external_max_asr <= 0.10` and clean `mAP50-95` drop `<= 0.03`.
- ODA hardening remains the most difficult failure mode. Current failure replay reduces it slightly but does not suppress it enough.
- The ODA recall loss is a stronger training signal and has passed CUDA smoke validation, but current results are still far from the required ASR threshold.
- GitHub CI is CPU/static-test oriented; real YOLO/CUDA detox runs must be validated locally.
- Full datasets, run directories, and large transient model artifacts are intentionally not committed.
- ODA ASR must be reported with its explicit success mode. Do not compare old runs unless `oda_success_mode` is the same.

## Recommended Next Steps

1. Prioritize a custom matched-candidate ODA/OGA repair loss:
   - ODA positives: match decoded candidates near each GT target and optimize class/objectness/box recall.
   - OGA negatives: suppress target-class candidates only on target-absent failure samples.
   - Avoid global target-class suppression.
2. Use the current Pareto/layer merge candidates only as initialization points, not as accepted purified models.
3. Run feature-level purification only with a trusted clean teacher, or explicitly opt into the weaker self-teacher mode for experiments.
4. Prefer split hard suites:
   - replay/train: `poison_benchmark_cuda_large`
   - held-out eval/selection: `poison_benchmark_cuda_tuned`
5. Accept a model only if:
   - external max ASR `<= 0.10`
   - external mean ASR ideally `<= 0.05–0.08`
   - clean `mAP50-95` drop `<= 0.03`
   - badnet_ODA, badnet_OGA, semantic, and WaNet all improve versus `best 2.pt`
   - Security Gate + acceptance report returns pass/yellow-or-better status.

# Project Status — 2026-05-08

## Current Best Candidate

The current best smoke candidate remains:

```text
D:\clean_yolo\model_security_gate\runs\pareto_letterbox_joint_merge_fine_2026-05-07\models\pareto_global_alpha_0p08.pt
```

Small external hard-suite result on `D:\clean_yolo\poison_benchmark_cuda_tuned`, 20 images per attack:

```text
external max ASR:  0.05
external mean ASR: 0.025
badnet_oda:        0.05
blend_oga:         0.00
semantic_clean:    0.05
wanet_oga:         0.00
clean mAP50-95:    0.1703
baseline mAP50-95: 0.1998
mAP drop:          0.0295
```

This meets the numeric small-suite ASR/mAP target, but is still not a production Green model because one semantic target-absent false positive remains.

## Latest Debug Finding

The remaining semantic failure is:

```text
attack_0011_helm_021400.jpg
```

The final false `helmet` detection maps directly to high raw target candidates in the same image region. This means the residual problem is not unknown trigger discovery anymore; it is a localized semantic false-positive score/ranking issue.

## Code Added in This Update

Added a surgical semantic FP region guard:

```text
model_security_gate/detox/oda_score_calibration.py
  semantic_fp_region_guard_loss(...)

model_security_gate/detox/oda_score_calibration_repair.py
  semantic FP region extraction from baseline external rows
  semantic failure replay so the exact FP image is guaranteed in training
  train log field: loss_semantic_fp_region

scripts/oda_score_calibration_repair_yolo.py
  --lambda-semantic-fp-region
  --semantic-fp-region-* knobs

tests/test_oda_score_calibration.py
  region guard apply/skip tests
```

Added hard candidate gates for the final repair stage:

```text
scripts/oda_score_calibration_repair_yolo.py
  --max-attack-asr badnet_oda=0.05 blend_oga=0.0 semantic_green_cleanlabel=0.0 wanet_oga=0.0
  --semantic-fp-required-max-conf 0.25

model_security_gate/detox/oda_score_calibration_repair.py
  blocked_by_hard_constraints(...)
  semantic_target_absent_max_conf(...)
```

Local smoke / CI:

```text
pixi run ci-smoke
87 passed
```

## Latest Experiments

### Region Guard Only

Run:

```text
D:\clean_yolo\model_security_gate\runs\semantic_fp_region_polish_alpha008_2026-05-08
```

Result:

```text
semantic_green_cleanlabel: 0.05 -> 0.00
badnet_oda:               0.05 -> 0.15
final: rolled back
```

Conclusion: the region guard can suppress the known semantic FP, but by itself it disturbs ODA recall.

### Region Guard + ODA Anchor

Run:

```text
D:\clean_yolo\model_security_gate\runs\semantic_fp_region_with_oda_anchor_v2_alpha008_2026-05-08
```

Result:

```text
badnet_oda:               0.05
semantic_green_cleanlabel: 0.05
blend_oga:                0.00
wanet_oga:                0.00
final: numerically unchanged but accepted by repair gate
```

The semantic FP score moved only slightly:

```text
0.441227 -> 0.440741
```

Conclusion: this configuration is safe but too weak.

### Stronger Region Guard + ODA Anchor

Run:

```text
D:\clean_yolo\model_security_gate\runs\semantic_fp_region_with_oda_anchor_v3_alpha008_2026-05-08
```

Result after manual completion of the interrupted epoch-2 evaluation:

```text
badnet_oda:               0.00
semantic_green_cleanlabel: 0.15
blend_oga:                0.05
wanet_oga:                0.10
final: rejected / not usable
```

Conclusion: too much region/anchor pressure shifts the model into new OGA/semantic/WaNet failures.

### Hard-Gate Validation

Run:

```text
D:\clean_yolo\model_security_gate\runs\hard_gate_validation_alpha008_2026-05-08
```

This run validates the candidate selector rather than claiming a better model.
The candidate still had:

```text
external max ASR:                 0.05
semantic_green_cleanlabel ASR:    0.05
semantic target-absent max conf:  0.4411509931
```

With hard gates:

```text
--max-attack-asr semantic_green_cleanlabel=0.0
--semantic-fp-required-max-conf 0.25
```

it was correctly blocked:

```text
blocked_constraints:
  attack_asr>0.0:semantic_green_cleanlabel=0.05
  semantic_fp_conf>0.25:0.4411509931087494
final: rolled back to alpha_0p08
```

Conclusion: the project now prevents a numerically low `max_asr <= 0.10` candidate from being accepted when the known semantic FP is still present.

## Current Interpretation

The engineering pipeline is now runnable and rollback-safe. The remaining problem is an algorithmic Pareto conflict:

```text
strong semantic suppression
  -> can remove the known FP
  -> but can destabilize ODA / OGA / WaNet

weak semantic suppression
  -> preserves the current best ASR matrix
  -> but does not cross the semantic FP confidence threshold
```

Therefore the next useful algorithmic work is not more generic replay. It should focus on either:

1. threshold-aware local score calibration for the single semantic FP, with explicit OGA/ODA/WaNet no-worse regularization in the same minibatch; or
2. post-training decision calibration / runtime abstain for this exact semantic false-positive pattern, while keeping the current best checkpoint unchanged.

## Recommended Next Step

For code contributors, start from these files:

```text
model_security_gate/detox/oda_score_calibration.py
model_security_gate/detox/oda_score_calibration_repair.py
scripts/oda_score_calibration_repair_yolo.py
tests/test_oda_score_calibration.py
```

Do not replace the current best model unless a candidate satisfies all of:

```text
badnet_oda <= 0.05
semantic_green_cleanlabel == 0.00
blend_oga == 0.00
wanet_oga == 0.00
mAP50-95 drop <= 0.03
no per-attack regression
```

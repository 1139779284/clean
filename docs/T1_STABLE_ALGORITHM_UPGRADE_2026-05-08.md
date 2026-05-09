# T1 Stable Algorithm Upgrade - 2026-05-08

This upgrade targets the current bottleneck documented in `PROJECT_STATUS_2026-05-08.md`: the project is rollback-safe, but the repair trainer still struggles to generate a production-Green candidate when semantic target-absent false positives must be removed without worsening ODA/OGA/WaNet.

## What changed

### 1. Threshold-aware semantic suppression

The previous semantic negative and semantic FP region guards could behave like full target-class suppression. That can remove known semantic false positives, but it can also damage the shared detection head and regress ODA/OGA/WaNet.

This upgrade changes semantic guards into threshold-aware caps:

```text
only scores near/above production cap receive gradient
cap default = 0.25
active band default = cap - 0.03 / 0.05
BCE can be disabled; hinge-to-cap remains active
```

The goal is not to push semantic FP score toward zero. The goal is to push it just below the production Green threshold while preserving ODA-positive evidence.

Implemented in:

```text
model_security_gate/detox/oda_score_calibration.py
  _threshold_aware_negative_cap_loss
  semantic_negative_guard_loss(... negative_bce_weight, active_margin)
  semantic_fp_region_guard_loss(... negative_bce_weight, active_margin)
```

### 2. Baseline-teacher no-worse anchors

The repair trainer now loads the input model as a frozen baseline teacher by default. This makes the update no-worse aware at the batch/loss level, not only at final external hard-gate selection.

Added anchors:

```text
localized_target_score_floor_loss
    On target-present/ODA images, prevent the repaired model from dropping below the baseline teacher near GT boxes.

target_absent_teacher_cap_loss
    On target-absent replay images, prevent target-class drift above both the production cap and teacher+margin.
```

This directly targets the observed failure mode:

```text
strong semantic suppression -> ODA/OGA/WaNet regression
weak semantic suppression   -> semantic FP remains around 0.44
```

### 3. ODA matched-candidate anchor

`matched_candidate_oda_loss` from `oda_loss_v2.py` is now wired into `oda_score_calibration_repair.py` as an additional ODA-positive anchor.

### 4. YAML-driven repair CLI

`scripts/oda_score_calibration_repair_yolo.py` now supports:

```bash
python scripts/oda_score_calibration_repair_yolo.py --config configs/oda_score_calibration_repair.yaml
```

The resolved config is written to:

```text
<out_dir>/resolved_config.json
```

### 5. Robust CPU/lightweight TorchVision fallback

The handoff environment had TorchVision import / compiled NMS issues. This upgrade adds:

```text
model_security_gate/utils/torchvision_compat.py
```

It repairs the `torchvision::nms` schema if needed and patches `torchvision.ops.nms` with a Python fallback for lightweight smoke runs. Production GPU runs should still use native TorchVision ops.

## New T1-oriented loss structure

```text
L_total =
    L_score_calibration
  + L_task
  + L_oga_guard
  + L_semantic_threshold_cap
  + L_semantic_fp_region_threshold_cap
  + L_oda_matched_anchor
  + L_oda_teacher_floor
  + L_target_absent_teacher_cap
```

Key design:

```text
press semantic only above the required production cap
while teacher anchors preserve ODA-positive and target-absent no-worse behavior
```

## Recommended GPU run

```bash
PYTHONPATH=. python scripts/oda_score_calibration_repair_yolo.py \
  --config configs/oda_score_calibration_repair.yaml \
  --device 0 \
  --out runs/t1_score_calibration_repair
```

Then verify with your production Green gate and full external hard-suite / clean mAP protocol.

## Expected effect

This code improves the chance of producing a model-Green candidate by reducing the semantic/ODA Pareto conflict. It does not claim a mathematical guarantee or a new Green checkpoint without actual GPU training and hard-suite verification.

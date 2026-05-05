# ASR-aware strong detox

This mode is for the failure case where pseudo/feature-only detox lowers a risk
score but ASR remains high. It is not a smoke test. It requires audited labels
and trains on explicit attack-regression counterfactuals.

## Why this exists

Weak pseudo detox can re-teach the model its own poisoned predictions. If a
model has post-detox ASR such as 35%-75%, it must be treated as detox failed.
ASR-aware detox builds a supervised dataset where common triggers are explicitly
non-causal:

- OGA / ghost object: triggered images without target labels remain negatives.
- ODA / vanish object: triggered or warped images with true targets preserve the
  original target labels.
- WaNet: smooth-warped images preserve boxes and classes.
- semantic shortcut: green-context counterfactuals preserve real labels and
  harden against color/context shortcuts.

## Command

```bash
python scripts/asr_aware_detox_yolo.py \
  --model path/to/pseudo_detox_after.pt \
  --images dataset/images/train \
  --labels dataset/labels/train \
  --data-yaml dataset/data.yaml \
  --target-classes helmet \
  --out runs/asr_aware_detox_best2 \
  --cycles 4 \
  --epochs-per-cycle 10 \
  --include-clean-repeat 2 \
  --include-attack-repeat 2 \
  --max-allowed-asr 0.10 \
  --max-map-drop 0.03 \
  --device 0
```

The pipeline writes:

```text
runs/asr_aware_detox_best2/
  resolved_config.json
  01_asr_aware_dataset/data.yaml
  02_cycle_XX_train/asr_aware/weights/best.pt
  03_cycle_XX_asr/asr_regression.json
  asr_aware_detox_manifest.json
```

The selected checkpoint is `manifest["final_model"]`. If the status is
`failed_asr_or_map`, do not deploy the model.

## ASR regression only

```bash
python scripts/run_asr_regression.py \
  --model runs/asr_aware_detox_best2/final.pt \
  --images dataset/images/val \
  --labels dataset/labels/val \
  --data-yaml dataset/data.yaml \
  --target-classes helmet \
  --out runs/asr_regression_final
```

## Acceptance gate

```bash
python scripts/acceptance_gate.py \
  --before-report runs/before/security_report.json \
  --after-report runs/after/security_report.json \
  --before-metrics runs/eval_before.json \
  --after-metrics runs/eval_after.json \
  --attack-metrics runs/asr_regression_final/asr_regression.json \
  --detox-manifest runs/asr_aware_detox_best2/asr_aware_detox_manifest.json \
  --safety-critical \
  --max-allowed-asr 0.10 \
  --out runs/acceptance_final.json
```

## Important boundary

This mode can directly target the ASR failure you observed. It still needs real
labels. Without audited labels or a trusted teacher, treat the result as risk
reduction only, not a safety proof.

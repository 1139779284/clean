# Label-free / Unknown-trigger Detox Mode

This addendum fixes the important limitation of the original strong detox path:
true YOLO bbox labels are useful, but they must not be mandatory for unknown-trigger model intake.

The project now supports three label modes in `scripts/strong_detox_yolo.py`:

## 1. `--label-mode supervised`
Use when you have human-audited YOLO labels.

- Builds counterfactual images from real boxes.
- For `target_occlude` / `target_inpaint`, removes only the target-class labels.
- Enables counterfactual fine-tuning, NAD, I-BAU, and prototype regularization.

## 2. `--label-mode pseudo`
Use when you do not have trusted labels.

- Builds conservative pseudo labels from either:
  - a provided clean teacher model,
  - teacher/suspicious agreement,
  - or the suspicious model as a weak fallback.
- Builds object-preserving and object-removal counterfactuals from pseudo boxes.
- Enables counterfactual fine-tuning and feature detox, but the manifest records that labels are pseudo.

Recommended command with a clean teacher:

```bash
python scripts/strong_detox_yolo.py \
  --model suspicious.pt \
  --teacher-model clean_teacher.pt \
  --images shadow_images \
  --data-yaml data.yaml \
  --label-mode pseudo \
  --pseudo-source agreement \
  --target-classes helmet \
  --out runs/strong_detox_pseudo
```

Command when you do not know the attacked target class:

```bash
python scripts/strong_detox_yolo.py \
  --model suspicious.pt \
  --teacher-model clean_teacher.pt \
  --images shadow_images \
  --data-yaml data.yaml \
  --label-mode pseudo \
  --pseudo-source agreement \
  --out runs/strong_detox_all_classes
```

If `--target-classes` is omitted, the pipeline treats all classes as security-relevant. This is noisier but handles unknown target-class backdoors.

Weak fallback without clean teacher:

```bash
python scripts/strong_detox_yolo.py \
  --model suspicious.pt \
  --images shadow_images \
  --data-yaml data.yaml \
  --label-mode pseudo \
  --pseudo-source suspicious \
  --out runs/strong_detox_self_pseudo
```

This should be treated as weaker because the suspicious model may produce poisoned pseudo labels. Prefer `feature_only` if pseudo labels look bad.

## 3. `--label-mode feature_only`
Use when labels and pseudo labels are untrusted.

- Skips supervised counterfactual fine-tuning.
- Skips prototype regularization.
- Keeps channel scoring/pruning, NAD, and I-BAU feature unlearning.
- Uses image-level feature alignment rather than bbox labels.

```bash
python scripts/strong_detox_yolo.py \
  --model suspicious.pt \
  --teacher-model clean_teacher.pt \
  --images shadow_images \
  --data-yaml data.yaml \
  --label-mode feature_only \
  --out runs/strong_detox_feature_only
```

## Practical rule

- Have audited labels: `supervised`.
- No labels, but have clean teacher or reference model: `pseudo --pseudo-source agreement`.
- No labels and no trusted teacher: `feature_only` first; optionally use `pseudo --pseudo-source suspicious` only for exploratory repair, then manually inspect pseudo labels.

Unknown trigger does not require knowing the trigger. The pipeline tests and repairs dependence on non-causal factors: color, background, texture, compression, occlusion, object removal, and feature-space sensitivity.

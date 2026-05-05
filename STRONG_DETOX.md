# Strong Detox Extension

This extension plugs into the existing `model_security_gate` project. It adds a complete strong detox path after a model is classified as Yellow/Red/Black by `scripts/security_gate.py`.

## What is new

The previous package already included:

- zero-trust scans,
- trigger-agnostic counterfactual generation,
- runtime guard,
- simple counterfactual YOLO fine-tuning,
- basic channel correlation pruning.

This extension adds:

- ANP-style channel sensitivity scoring,
- FMP-style feature-map pruning scoring,
- merged suspicious-channel soft pruning,
- NAD attention distillation,
- teacher output and feature distillation,
- I-BAU-style adversarial unlearning,
- class prototype alignment,
- attention localization for safety-critical classes,
- a one-shot strong pipeline script.

## Recommended workflow

### 1. Scan the unknown model

```bash
python scripts/security_gate.py \
  --model suspicious.pt \
  --images dataset/images/train \
  --labels dataset/labels/train \
  --critical-classes helmet \
  --out runs/security_gate_before \
  --occlusion \
  --channel-scan
```

### 2. Build the counterfactual detox dataset

```bash
python scripts/detox_build_dataset.py \
  --images dataset/images/train \
  --labels dataset/labels/train \
  --data-yaml dataset/data.yaml \
  --target-classes helmet \
  --out runs/detox_dataset
```

### 3. Train a trusted teacher, if possible

The safest teacher is trained from trusted official/pretrained weights, not from the suspicious model:

```bash
python scripts/detox_train_yolo.py \
  --base-model trusted_pretrain.pt \
  --data-yaml runs/detox_dataset/data.yaml \
  --out-project runs/teacher_train \
  --name clean_teacher \
  --epochs 30 \
  --imgsz 640 \
  --batch 16
```

Use `runs/teacher_train/clean_teacher/weights/best.pt` as `--teacher-model` below.

### 4. Run strong detox

```bash
python scripts/strong_detox_yolo.py \
  --model suspicious.pt \
  --data-yaml runs/detox_dataset/data.yaml \
  --teacher-model runs/teacher_train/clean_teacher/weights/best.pt \
  --target-classes helmet \
  --out runs/strong_detox \
  --pre-prune both \
  --prune-top-k 50 \
  --epochs 20 \
  --batch 8 \
  --imgsz 640
```

Main outputs:

```text
runs/strong_detox/
  anp_channel_scores.csv
  fmp_channel_scores.csv
  merged_channel_scores.csv
  prepruned.pt
  preprune_report.json
  train/
    best_strong_detox.pt
    last_strong_detox.pt
    strong_detox_train_log.csv
    strong_detox_report.json
  pipeline_report.json
```

### 5. Re-scan the detoxed model

```bash
python scripts/security_gate.py \
  --model runs/strong_detox/train/best_strong_detox.pt \
  --images dataset/images/train \
  --labels dataset/labels/train \
  --critical-classes helmet \
  --out runs/security_gate_after_strong_detox \
  --occlusion \
  --channel-scan
```

## Algorithm mapping

| Module | File | Purpose |
|---|---|---|
| ANP scoring | `model_security_gate/detox/anp.py` | ranks loss-sensitive channels using activation-gradient signals |
| FMP scoring | `model_security_gate/detox/fmp.py` | ranks feature maps dormant on clean data or spiky on hard data |
| Soft pruning | `model_security_gate/detox/prune.py` | zeros suspicious Conv2d output channels without changing architecture |
| NAD | `model_security_gate/detox/feature_hooks.py`, `strong_train.py` | aligns student attention maps to frozen teacher attention maps |
| I-BAU-style unlearning | `model_security_gate/detox/losses.py` | PGD inner maximization on unknown perturbations, outer supervised minimization |
| Prototype alignment | `model_security_gate/detox/prototype.py` | pulls object-region features toward clean class prototypes |
| Attention localization | `model_security_gate/detox/losses.py` | penalizes target-class attention outside target bounding boxes |
| Strong pipeline | `model_security_gate/detox/strong_pipeline.py` | end-to-end pre-prune + strong train orchestration |

## Notes

The strong training loop uses Ultralytics' internal supervised detection loss by calling the underlying DetectionModel with a batch dict containing `img`, `cls`, `bboxes`, and `batch_idx`. This keeps the implementation connected to YOLO rather than being a detached toy loss.

A trusted teacher is strongly recommended. If `--teacher-model` is omitted, the code falls back to a frozen copy of the suspicious model; that mode is useful for stabilizing training but weaker for removing subtle semantic backdoors.

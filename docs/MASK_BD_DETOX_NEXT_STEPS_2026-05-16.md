# Mask Backdoor Detox Next Steps — 2026-05-16

This note keeps the new v2/v3 backdoor benchmark work separate from the older
T0 poison-model matrix.  The matrix remains the publication-scale contribution
2 route; the mask backdoor models are the fast, reproducible detox benchmark.

## What Lives Where

- Models: `models/mask_bd_v2_*.pt`, `models/mask_bd_v3_sig_*.pt`
- Source and generated benchmark data: `datasets/mask_bd*`
- Loader-friendly external eval roots: `datasets/mask_bd_external_eval/`
- Reproduction tools: `model_security_gate/tools/clean_label_mask/`
- Detox smoke configs: `model_security_gate/configs/mask_bd_*_detox_smoke.yaml`
- v2 focused follow-up config:
  `model_security_gate/configs/mask_bd_v2_detox_2cycle_lagrangian.yaml`
- Focused Hybrid-PURIFY attack configs:
  `model_security_gate/configs/mask_bd_v2_hybrid_purify.yaml`,
  `model_security_gate/configs/mask_bd_v3_sig_hybrid_purify.yaml`
- Run outputs: `model_security_gate/runs/mask_bd_*_detox_smoke_2026-05-16/`
- Summary docs: top-level `docs/`

## Prepare Eval Roots

Hybrid-PURIFY's external ASR runner expects each attack root to contain
`images/` and `labels/`.  Build those roots from the shared 42-image
head-only eval set:

```powershell
cd D:/clean_yolo/model_security_gate
pixi run python tools/clean_label_mask/prepare_mask_bd_external_eval.py
```

Expected outputs:

```text
D:/clean_yolo/datasets/mask_bd_external_eval/
  manifest.json
  badnet_oga_mask_bd_v2_visible/images
  badnet_oga_mask_bd_v2_visible/labels
  blend_oga_mask_bd_v3_sig/images
  blend_oga_mask_bd_v3_sig/labels
```

## Plan Detox Runs

Emit runbooks without launching GPU training:

```powershell
pixi run t0-detox-ablation-plan `
  --spec configs/mask_bd_v2_detox_smoke.yaml `
  --out runs/mask_bd_v2_detox_smoke_named_2026-05-16

pixi run t0-detox-ablation-plan `
  --spec configs/mask_bd_v3_sig_detox_smoke.yaml `
  --out runs/mask_bd_v3_sig_detox_smoke_named_2026-05-16
```

Run v2 first.  It is the high-ASR visible-trigger benchmark and should fail
clearly before detox and improve clearly after detox.  Run v3 second because
it is the hidden-trigger stress case.

## Baseline External ASR Checks

The generated eval roots have been checked through the main external
hard-suite runner:

```text
v2 poisoned: 97.619% ASR
v2 clean baseline: 64.286% ASR
v3 poisoned: 69.048% ASR
v3 clean baseline: 4.762% ASR
```

Reports:

```text
model_security_gate/runs/mask_bd_v2_external_baseline_2026-05-16/
model_security_gate/runs/mask_bd_v2_external_clean_baseline_2026-05-16/
model_security_gate/runs/mask_bd_v3_sig_external_baseline_2026-05-16/
model_security_gate/runs/mask_bd_v3_sig_external_clean_baseline_2026-05-16/
```

The eval root directory names intentionally include `badnet_oga` / `blend_oga`
so Hybrid-PURIFY's attack-phase planner maps the external ASR failure to an
OGA hardening phase instead of treating it as an unrelated custom benchmark.

## Acceptance Targets

- v2 visible OGA: defended ASR <= 10%, clean mAP drop <= 5 pp for smoke.
- v3 SIG OGA: defended ASR <= 10%, clean mAP drop <= 5 pp for smoke.
- For publication-style reporting, rerun with larger eval sets if available
  and certify through CFRC rather than only the smoke runbook summary.

## Launch Results

Current smoke runs use the prepared 42-image external roots, paired clean
teacher checkpoints, `imgsz=416`, and one Hybrid-PURIFY cycle.

```text
v2 visible OGA:
  static_lambda:     97.619% -> 26.190% ASR, mAP drop 4.233 pp, failed threshold
  lagrangian_lambda: 97.619% -> 23.810% ASR, mAP drop 4.428 pp, failed threshold
  run root: model_security_gate/runs/mask_bd_v2_detox_smoke_named_2026-05-16/

v2 visible OGA follow-up:
  lagrangian_2cycle: 97.619% -> 16.667% ASR, mAP drop 5.701 pp, failed threshold
  run root: model_security_gate/runs/mask_bd_v2_detox_2cycle_lagrangian_2026-05-16/

v3 SIG OGA:
  static_lambda:     69.048% -> 0.000% ASR, mAP drop 3.960 pp, passed
  lagrangian_lambda: 69.048% -> 0.000% ASR, mAP drop 3.974 pp, passed
  run root: model_security_gate/runs/mask_bd_v3_sig_detox_smoke_named_2026-05-16/
```

CFRC reports were also emitted under each run root's `cfrc_certificate/`
directory.  With the default CFRC clean mAP drop tolerance of 3 pp, both v2
and v3 smoke runs are currently marked uncertified because their best
checkpoints drop mAP50-95 by roughly 4.0-4.4 pp.  The smoke gate above uses
the looser 5 pp tolerance to separate "attack removed" from "publishable
certificate".

Notes:

- The Lagrangian metric normalizer now maps suite-specific keys such as
  `badnet_oga_mask_bd_v2_visible` and `blend_oga_mask_bd_v3_sig` back to
  canonical controller constraints (`badnet_oga`, `blend_oga`).
- v3 is immediately useful as the hidden-trigger detox benchmark: both arms
  reach zero external ASR under the smoke gate.
- v2 remains the hard visible-trigger benchmark.  The two-cycle Lagrangian
  follow-up reached 16.667% ASR, but phase finetune and clean recovery both
  rebound ASR sharply.  The final manifest correctly keeps the best feature
  purifier checkpoint, so the next v2 work is algorithmic: make recovery
  conservative with external ASR rollback, or run a no-phaseft/no-recovery
  variant before increasing cycles.

## v2 Recovery Rebound Diagnosis

The two-cycle follow-up separates leverage from failure:

```text
cycle 2 OGA hardening feature: 16.667% ASR
cycle 2 OGA phase finetune:    26.190% ASR
cycle 2 clean recovery:        92.857% ASR
selected final model:          feature_purify best checkpoint
CFRC status:                   uncertified; mAP drop 5.701 pp
```

Interpretation:

- Not a final-model pointer bug: `final_model` points to the lowest-ASR
  feature purifier checkpoint.
- Not a pure metric artifact: ASR falls from 97.619% to 16.667%, and the CFRC
  reduction path is statistically strong.
- Current weakness: clean recovery optimizes clean utility after OGA hardening
  without a strict enough external-ASR non-regression guard, so it can undo the
  useful detox direction on v2.

Next run to schedule:

```powershell
pixi run hybrid-purify-detox-yolo `
  --config configs/mask_bd_v2_hybrid_purify.yaml `
  --model D:/clean_yolo/models/mask_bd_v2_poisoned.pt `
  --teacher-model D:/clean_yolo/models/mask_bd_v2_clean_baseline.pt `
  --images D:/clean_yolo/datasets/helmet_head_yolo_train_remap/images/train `
  --labels D:/clean_yolo/datasets/helmet_head_yolo_train_remap/labels/train `
  --data-yaml D:/clean_yolo/datasets/helmet_head_yolo_train_remap/data.yaml `
  --target-classes helmet `
  --external-eval-roots D:/clean_yolo/datasets/mask_bd_external_eval/badnet_oga_mask_bd_v2_visible `
  --external-replay-roots D:/clean_yolo/datasets/mask_bd_external_eval/badnet_oga_mask_bd_v2_visible `
  --out runs/mask_bd_v2_detox_recovery_guard_2026-05-16 `
  --imgsz 416 --batch 8 --cycles 2 --feature-epochs 1 --phase-epochs 1 --recovery-epochs 1 `
  --max-images 800 --eval-max-images 100 `
  --external-eval-max-images-per-attack 42 --external-replay-max-images-per-attack 42 `
  --max-allowed-external-asr 0.1 --max-map-drop 0.05 --selection-max-map-drop 0.08 `
  --no-pre-prune --use-lagrangian-controller --no-stop-on-pass `
  --no-phase-finetune --no-clean-recovery-finetune `
  --external-replay-floor-per-attack 42 --output-distill-scale 0.0 --feature-distill-scale 1.0
```

## Separation From Old Matrix

Do not place v2/v3 artifacts under `t0_poison_matrix_*` runs.  Those folders
track the older attack-zoo matrix.  The mask backdoor benchmarks are separate
named benchmarks with paired clean baselines and their own detox smoke configs.

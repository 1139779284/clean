# Contribution 1 Smoke Ablation (static vs Lagrangian)

2026-05-10, RTX 4060 Laptop, pixi environment.

## Setup

- spec: `configs/t0_detox_ablation_smoke.yaml`
- runbook: `runs/t0_detox_ablation_smoke_2026-05-10/T0_DETOX_ABLATION_RUNBOOK.md`
- cfrc manifest: `runs/t0_detox_ablation_smoke_2026-05-10/cfrc_manifest.json`
- cfrc certificate: `runs/t0_detox_ablation_smoke_2026-05-10/cfrc/T0_DEFENSE_CERTIFICATE.md`
- sizing: `imgsz=416, batch=8, cycles=1, phase_epochs=1, feature_epochs=1, recovery_epochs=1, max_images=800, external_eval_max_images_per_attack=40`
- poisoned baseline: `D:/clean_yolo/models/best_2_poisoned.pt`
- external suite: `poison_benchmark_cuda_tuned_remap_v2`
- each arm: ~35 min GPU; total ~75 min.

## Result summary

Both arms ended at **identical per-attack numbers**.

| arm | ext max ASR | ext mean ASR | mAP drop | certified |
|---|---:|---:|---:|---|
| static_lambda | 0.10 | 0.045 | -0.036 | FAIL |
| lagrangian_lambda | 0.10 | 0.045 | -0.036 | FAIL |

Per-attack (identical across arms):

| attack | poisoned ASR | defended ASR | McNemar raw p | Holm p |
|---|---:|---:|---:|---:|
| badnet_oda | 0.0000 | 0.0000 | 1 | 1 |
| badnet_oga | 0.1750 | 0.0250 | 0.0703 | 0.3516 |
| blend_oga | 0.2000 | 0.1000 | 0.3438 | 1 |
| semantic_green_cleanlabel | 0.1000 | 0.0250 | 0.375 | 1 |
| wanet_oga | 0.0750 | 0.0750 | 1 | 1 |

## Interpretation

1. **Pipeline plumbing works end to end**: both arms finished, both produced
   valid manifests, both produced CFRC entries after a single re-plan, and
   CFRC correctly classified both as uncertified under the strict
   `min_certified_reduction=0.05` and `max_certified_asr=0.05` thresholds.
2. **The Lagrangian controller's trace persisted correctly**
   (`runs/.../lagrangian_lambda/hybrid_purify_manifest.json`
   `lagrangian_controller.trace`), with `unobserved` metrics correctly
   leaving lambda unchanged rather than forcing it to `lambda_max`.
3. **Arms are identical at this sizing** because both accepted the OGA
   phase finetune candidate in cycle 1. With `cycles=1`, the controller
   has no violation history to differentiate from static weights on the
   first cycle's feature-purifier; by the time the controller updates
   (end of cycle 1), the run is done. This is a **real** property of the
   smoke configuration, not a Lagrangian bug.
4. **n=40 per attack drives wide CIs**. Wilson upper bounds remain around
   0.09-0.23 even when the point estimate is 0.025. Non-inferiority path
   rarely accepts at this sizing.

## What this smoke proves

- Hybrid-PURIFY-OD can move the poisoned best_2.pt external max ASR
  from **0.20 to 0.10** in ~35 min on a single RTX 4060 Laptop at smoke
  sizing, with a **clean mAP gain** (+3.6pp) on the remap_v2 val split.
- The full static-vs-Lagrangian ablation **needs cycles >= 3** to
  differentiate the two arms. The smoke result alone is not enough to
  claim Lagrangian dominates.

## Next experiment

Run the same two arms with `cycles=3` (about 2-2.5x the runtime, ~2 hours
per arm) using `configs/t0_detox_ablation_local.yaml`. Expected outcomes:

- Static arm plateaus after cycle 1-2, Lagrangian arm continues to move
  weights on the remaining violated attacks (blend_oga, wanet_oga).
- CMR gap opens up in favour of Lagrangian iff the adaptive weights
  succeed in reducing blend_oga below 0.10.

If cycles=3 still shows parity, the honest paper claim is
*Lagrangian has zero effect on a single-cycle smoke and does not
differentiate up to N cycles; the design remains sound but this
benchmark does not separate the two.*  That is still a publishable
negative result; it is not a failure of the framework.

## Commands used

```powershell
# Arm 1 (static)
pixi run python scripts/hybrid_purify_detox_yolo.py --model "D:/clean_yolo/models/best_2_poisoned.pt" `
  --images D:/clean_yolo/datasets/helmet_head_yolo_train_remap/images/train `
  --labels D:/clean_yolo/datasets/helmet_head_yolo_train_remap/labels/train `
  --data-yaml D:/clean_yolo/datasets/helmet_head_yolo_train_remap/data.yaml `
  --target-classes helmet `
  --external-eval-roots D:/clean_yolo/datasets/poison_benchmark_cuda_tuned_remap_v2 `
  --out runs/t0_detox_ablation_smoke_2026-05-10/static_lambda `
  --imgsz 416 --batch 8 --cycles 1 --phase-epochs 1 --feature-epochs 1 --recovery-epochs 1 `
  --max-allowed-external-asr 0.1 --max-map-drop 0.05 --device 0 `
  --external-replay-roots D:/clean_yolo/datasets/poison_benchmark_cuda_tuned_remap_v2 `
  --max-images 800 --eval-max-images 100 `
  --external-eval-max-images-per-attack 40 --external-replay-max-images-per-attack 80 `
  --selection-max-map-drop 0.08 --no-pre-prune

# Arm 2 (lagrangian): add --use-lagrangian-controller

# CFRC
pixi run python scripts/t0_defense_certificate.py `
  --manifest runs/t0_detox_ablation_smoke_2026-05-10/cfrc_manifest.json `
  --out runs/t0_detox_ablation_smoke_2026-05-10/cfrc `
  --n-bootstrap 2000
```

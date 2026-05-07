# ODA Post-NMS / Localized-Recall Repair

This upgrade adds a narrow surgical repair path for the current residual bottleneck:

```text
external max ASR stuck around 0.15-0.20
badnet_oda remains the top failure
ordinary failure-only supervised repair worsens to 0.20/0.25
crop replay works mechanically but can remove the global trigger/context
```

The new path is not a replacement for Hybrid-PURIFY. It is meant to start from the best Pareto/Hybrid candidate and run a small full-image repair loop on current `success=true` ODA failures only.

## New files

```text
model_security_gate/detox/oda_postnms_repair.py
scripts/oda_postnms_repair_yolo.py
tests/test_oda_postnms_repair.py
configs/oda_postnms_repair.yaml
```

## Key difference from targeted_repair_yolo.py

`targeted_repair_yolo.py` builds a failure-only YOLO dataset and sends it through the generic strong training loop. The CUDA smoke showed that this can worsen ASR and must be rolled back.

`oda_postnms_repair_yolo.py` runs a narrower direct optimization loop:

- only current ODA `success=true` full images are replayed;
- crop replay is intentionally disabled;
- `matched_candidate_oda_loss` is the dominant objective;
- ordinary supervised YOLO loss has a very small weight;
- every epoch is externally re-evaluated and blocked if any tracked attack worsens;
- final model rolls back unless an unblocked candidate improves the external score.

## Recommended first run

```powershell
pixi run python scripts\oda_postnms_repair_yolo.py `
  --model "D:\clean_yolo\model_security_gate\runs\pareto_upgrade_smoke_2026-05-07\models\pareto_global_alpha_1p0.pt" `
  --data-yaml "D:\clean_yolo\datasets\helmet_head_yolo_val\data.yaml" `
  --external-roots "D:\clean_yolo\poison_benchmark_cuda_tuned" `
  --target-classes helmet `
  --attack-names badnet_oda `
  --out "D:\clean_yolo\model_security_gate\runs\oda_postnms_repair_debug" `
  --imgsz 416 `
  --device 0 `
  --epochs 10 `
  --failure-repeat 24 `
  --lambda-task 0.03 `
  --lambda-oda-matched 4.0
```

## How to interpret results

Open:

```text
oda_postnms_repair_manifest.json
oda_postnms_train_log.csv
eval_00_before_external/external_hard_suite_asr.json
03_candidate_external/*/external_hard_suite_asr.json
```

The run is useful only if:

```text
badnet_oda decreases
external max ASR decreases below the input candidate
no other tracked attack worsens
clean follow-up eval stays acceptable
```

The run is not successful if `final_model` rolls back to the input model. That rollback is intentional and should not be bypassed.

## Why this may help when crop/full-image replay did not

The previous full-image replay path still used a broader strong-detox loop. This upgrade makes the localized ODA candidate objective dominate, while keeping full images intact so the global trigger/context remains present.

The remaining gap after this upgrade, if any, is likely closer to true post-NMS behavior: the GT-localized candidate must survive confidence thresholding and NMS. If this repair still cannot overfit the few fixed badnet_oda failures, the next step should inspect pre-NMS tensors versus adapter final detections and implement a detector-version-specific NMS/ranking proxy.

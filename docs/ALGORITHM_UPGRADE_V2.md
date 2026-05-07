# Algorithm Upgrade v2: ODA matched loss + PGBD-OD paired displacement

This upgrade is meant to address the current plateau where Hybrid-PURIFY lowers external ASR only modestly, especially on ODA / vanishing-object failures.

It adds two algorithmic changes rather than another parameter sweep.

## 1. ODA loss v2: matched candidate recall preservation

File: `model_security_gate/detox/oda_loss_v2.py`

The current `target_recall_confidence_loss` only requires one target-class candidate near a GT box to exceed a confidence floor. That signal is stable but weak.

The new `matched_candidate_oda_loss` does more:

- Finds decoded candidates near every GT target box.
- Applies positive BCE to target-class scores near the GT box.
- Adds normalized box center/size SmoothL1 against the GT box.
- Optionally distills target score and box from the clean teacher prediction.
- Keeps the loss target-present only, so it does not conflict with OGA suppression.

It is designed for `badnet_oda`, WaNet positive failures, and semantic positive failures where a true helmet/head disappears.

## 2. OGA negative-only suppression

File: `model_security_gate/detox/oda_loss_v2.py`

The new `negative_target_candidate_suppression_loss` suppresses target-class decoded candidates only on images that do not contain a target label. This prevents the common failure where strong OGA suppression globally lowers helmet confidence and worsens ODA.

## 3. PGBD-OD paired displacement

File: `model_security_gate/detox/pgbd_od.py`

The project already has PGBD-style single-view prototype alignment/suppression. This upgrade adds paired clean/attacked activation displacement:

- Creates differentiable semantic-green / sinusoidal-blend / smooth-warp views.
- For target-present images, attacked ROI features are kept close to clean/teacher ROI features and the class prototype.
- For target-absent images, attacked global evidence is prevented from moving toward target prototypes.

This targets semantic and WaNet failures more directly than ordinary hard-sample fine-tuning.

## How to apply

From the repository root:

```bash
python /path/to/algorithm_upgrade_v2/tools/apply_algorithm_upgrade_v2.py
python -m compileall -q model_security_gate scripts tests
python -m pytest -q
```

The script is idempotent and copies new modules/tests, then patches:

- `model_security_gate/detox/strong_train.py`
- `model_security_gate/detox/hybrid_purify_train.py`
- `scripts/hybrid_purify_detox_yolo.py` when the known CLI anchor is present
- `configs/hybrid_purify_detox.yaml` when the config anchor is present

## Suggested experiment

Start conservatively. Do not simply maximize all new weights.

```powershell
python scripts/hybrid_purify_detox_yolo.py `
  --model "D:\clean_yolo\best 2.pt" `
  --teacher-model "D:\clean_yolo\trusted_clean_teacher.pt" `
  --images "D:\clean_yolo\dataset\images\train" `
  --labels "D:\clean_yolo\dataset\labels\train" `
  --data-yaml "D:\clean_yolo\dataset\data.yaml" `
  --target-classes helmet `
  --external-replay-roots "D:\clean_yolo\poison_benchmark_cuda_large" `
  --external-eval-roots "D:\clean_yolo\poison_benchmark_cuda_tuned" `
  --out "D:\clean_yolo\model_security_gate\runs\hybrid_purify_v2_algo_upgrade" `
  --cycles 3 `
  --phase-epochs 2 `
  --feature-epochs 2 `
  --recovery-epochs 2 `
  --external-replay-max-images-per-attack 300 `
  --external-eval-max-images-per-attack 300 `
  --max-allowed-external-asr 0.10 `
  --max-map-drop 0.03 `
  --device 0
```

Expected diagnostics:

- ODA phase logs should contain non-zero `loss_oda_matched`.
- OGA/semantic negative phases should contain non-zero `loss_oga_negative` and/or `loss_pgbd_paired`.
- If `badnet_oda` is still dominant, increase `aggressive_lambda_oda_matched` before increasing `aggressive_lambda_oda_recall`.
- If semantic/WaNet remain dominant, increase `aggressive_lambda_pgbd_paired` and keep ODA no-worse gates enabled.

## Acceptance remains unchanged

Do not accept a model unless:

- external max ASR <= 0.10;
- clean mAP50-95 drop <= 0.03;
- no tracked attack regresses versus `best 2.pt`;
- final acceptance gate passes.

# External-hard-suite closed-loop ASR detox

Use this mode when ASR-aware smoke or balanced fine-tuning proves the model can be improved, but cannot satisfy both:

- low external ASR, e.g. `< 5%-10%`; and
- preserved clean mAP50-95, e.g. drop `<= 0.03`.

The key difference from `asr_aware_detox_yolo.py` is that checkpoint selection is driven by held-out external hard suites, not only internally generated regressions. It also trains in separate phases for OGA, ODA, semantic, and WaNet-style failures, followed by clean recovery.

## Why this exists

Your observed results show two failure modes:

| Mode | ASR | clean mAP50-95 | Problem |
|---|---:|---:|---|
| strongest ASR suppression | good | bad | destroys normal detector features |
| balanced fine-tune | still high | good | does not erase the backdoor |

The root cause is distribution mismatch: internal regression can be too self-consistent, while the older `poison_benchmark_cuda_large/tuned` hard suites still expose the backdoor. The closed-loop detox uses those hard suites in two ways:

1. **External checkpoint selection**: a checkpoint is never selected just because internal ASR is low.
2. **External replay**: high-ASR external attack samples are replayed into the matching phase, with correct labels, so training sees the real hard-suite distribution.

## Command

```powershell
python scripts/asr_closed_loop_detox_yolo.py `
  --model "D:\clean_yolo\best 2.pt" `
  --images "D:\clean_yolo\dataset\images\train" `
  --labels "D:\clean_yolo\dataset\labels\train" `
  --data-yaml "D:\clean_yolo\dataset\data.yaml" `
  --target-classes helmet `
  --external-eval-roots "D:\clean_yolo\poison_benchmark_cuda_large" "D:\clean_yolo\poison_benchmark_cuda_tuned" `
  --out "D:\clean_yolo\model_security_gate\runs\asr_closed_loop_best2" `
  --cycles 4 `
  --phase-epochs 3 `
  --recovery-epochs 2 `
  --external-replay-max-images-per-attack 250 `
  --max-allowed-external-asr 0.10 `
  --max-map-drop 0.03 `
  --device 0
```

Outputs:

```text
runs/asr_closed_loop_best2/
  resolved_config.json
  asr_closed_loop_detox_manifest.json
  eval_00_before_external/external_hard_suite_asr.json
  eval_cycle_XX_external/external_hard_suite_asr.json
  01_cycle_XX_<phase>_dataset/
  02_cycle_XX_phase_YY_<phase>_train/
```

The final candidate is:

```text
asr_closed_loop_detox_manifest.json -> final_model
```

Only deploy if:

```text
status == passed or passed_early
best.external_max_asr <= 0.10
best.map_drop <= 0.03
acceptance_gate --safety-critical passes
```

## Separate replay and held-out evaluation

For the strictest setup, use separate roots:

```powershell
--external-replay-roots D:\clean_yolo\poison_benchmark_cuda_large `
--external-eval-roots D:\clean_yolo\poison_benchmark_cuda_tuned
```

If you only provide `--external-eval-roots`, the same roots are used for replay and checkpoint selection. That is useful for hard mining, but final claims should still be checked on a held-out external suite.

## Standalone external hard-suite evaluation

```powershell
python scripts/run_external_hard_suite.py `
  --model "D:\clean_yolo\best 2.pt" `
  --data-yaml "D:\clean_yolo\dataset\data.yaml" `
  --target-classes helmet `
  --roots "D:\clean_yolo\poison_benchmark_cuda_large" "D:\clean_yolo\poison_benchmark_cuda_tuned" `
  --out runs\external_best2 `
  --device 0
```

This writes:

```text
external_hard_suite_asr.json
external_hard_suite_rows.csv
```

`acceptance_gate.py --attack-metrics` accepts this JSON directly.

## What this fixes

- `badnet_oga` high ASR drives OGA hardening, not generic fine-tune.
- `badnet_oda` worsening drives ODA target-preserving hardening and recall recovery.
- `semantic_green_cleanlabel` drives semantic/color context hardening.
- `wanet_oga` drives WaNet/smooth-warp hardening.
- clean recovery phases protect normal mAP after every hardening cycle.

## Boundaries

This is still a supervised repair method. It requires audited YOLO labels and a hard-suite root with correct labels. Without labels, use `feature_only` or pseudo modes only for risk reduction, not final safety claims.

# Overlap Class Guard - 2026-05-09

## Why This Was Added

After fixing the external benchmark label mapping, the best corrected-suite candidate was:

```text
D:\clean_yolo\model_security_gate\runs\hybrid_purify_corrected_suite_smoke_2026-05-09\02_cycle_01_phase_02_oga_hardening\ultralytics_phase_finetune\phase_finetune\weights\best.pt
```

At `conf=0.25` on `D:\clean_yolo\poison_benchmark_cuda_tuned_remap_v2`, it was close but still above the target:

```text
max ASR  = 0.103333
mean ASR = 0.067333
top fail = wanet_oga 0.103333
```

Manual failure audit showed the remaining OGA/semantic errors are mostly not unknown patch artifacts anymore. They are class-competition errors where exposed heads or faces are still being emitted as `helmet`.

That suggests a safer last-mile algorithm than more training:

```text
If target class = helmet
and an overlapping suppressor class = head is present
and head_conf + margin >= helmet_conf,
then suppress the helmet candidate.
```

This is a task-level mutual-exclusion rule, not a broad confidence threshold raise.

## Implementation

Main code:

```text
D:\clean_yolo\model_security_gate\model_security_gate\detox\external_hard_suite.py
D:\clean_yolo\model_security_gate\scripts\run_external_hard_suite.py
D:\clean_yolo\model_security_gate\tests\test_external_hard_suite.py
```

Reusable config:

```text
D:\clean_yolo\model_security_gate\configs\phase2_overlap_guard_external_eval.yaml
```

The guard is disabled by default and only runs when explicitly configured:

```yaml
external_hard_suite:
  apply_overlap_class_guard: true
  overlap_guard_suppressor_class_ids: [1]
  overlap_guard_iou: 0.10
  overlap_guard_conf_margin: 0.30
  overlap_guard_min_suppressor_conf: 0.25
```

For the current class mapping:

```text
0 = helmet
1 = head
```

## Corrected External Result

Command:

```powershell
pixi run python scripts\run_external_hard_suite.py `
  --config configs\phase2_overlap_guard_external_eval.yaml
```

Output:

```text
D:\clean_yolo\model_security_gate\runs\external_phase2_overlap_guard_valid_remap_v2_full300_conf025_2026-05-09\external_hard_suite_asr.json
```

Summary:

```json
{
  "n_rows": 1500,
  "max_asr": 0.09,
  "mean_asr": 0.05133333333333333,
  "asr_matrix": {
    "poison_benchmark_cuda_tuned_remap_v2::badnet_oda": 0.05333333333333334,
    "poison_benchmark_cuda_tuned_remap_v2::badnet_oga": 0.03666666666666667,
    "poison_benchmark_cuda_tuned_remap_v2::blend_oga": 0.05,
    "poison_benchmark_cuda_tuned_remap_v2::semantic_green_cleanlabel": 0.02666666666666667,
    "poison_benchmark_cuda_tuned_remap_v2::wanet_oga": 0.09
  }
}
```

Compared with the same model without this overlap guard at `conf=0.25`:

```text
max ASR  0.103333 -> 0.090000
mean ASR 0.067333 -> 0.051333
```

Guard removals:

```json
{
  "badnet_oda": 19,
  "badnet_oga": 11,
  "blend_oga": 9,
  "semantic_green_cleanlabel": 9,
  "wanet_oga": 12
}
```

## Clean Impact

Fixed-threshold helmet metrics on the remapped clean validation set:

```json
{
  "plain_conf025": {
    "precision": 0.8880308880297452,
    "recall": 0.7012195121944094
  },
  "overlap_guard_conf025": {
    "precision": 0.8975032851499376,
    "recall": 0.6941056910562052
  }
}
```

The guard improves precision and costs about `0.0071` absolute helmet recall at `conf=0.25`.

## Current Status

This solves the current `max ASR <= 0.10` threshold on the corrected suite at `conf=0.25`.

It does **not** mean every residual failure is gone:

```text
badnet_oda remains 0.053333
wanet_oga remains 0.09
mean ASR remains 0.051333
```

For stricter production Green, the remaining work is to reduce single-attack ASR further without sacrificing clean recall. The current guard is a safe last-mile runtime/evaluation layer, not a proof that the model weights are fully purified.

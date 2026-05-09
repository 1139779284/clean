# Trigger-Only Filtered Benchmark - 2026-05-09

## Purpose

The previous guarded external result mixed two sources of failures:

```text
clean/source image already fails
attack image fails only after trigger/attack transform
```

To avoid counting ordinary base-model errors as backdoor ASR, I created a filtered benchmark copy that removes the non-trigger/base-error failure rows.

## Important Safety Note

The original benchmark was **not** deleted:

```text
D:\clean_yolo\poison_benchmark_cuda_tuned_remap_v2
```

The filtered benchmark copy is:

```text
D:\clean_yolo\poison_benchmark_cuda_tuned_remap_v2_trigger_only_2026-05-09
```

This keeps the original evidence reproducible while giving a clean-conditioned benchmark for trigger-only ASR.

## Update: Active Benchmark Cleanup

Per project cleanup, the same non-detox-reference rows were also moved out of the active corrected benchmark root:

```text
D:\clean_yolo\poison_benchmark_cuda_tuned_remap_v2
```

They were moved to quarantine instead of being irreversibly removed:

```text
D:\clean_yolo\quarantine_non_detox_reference_2026-05-09
```

Quarantine manifest:

```text
D:\clean_yolo\model_security_gate\runs\failure_cause_clean_vs_attack_2026-05-09\non_detox_reference_quarantine_manifest.json
D:\clean_yolo\poison_benchmark_cuda_tuned_remap_v2\non_detox_reference_quarantine_manifest_2026-05-09.json
```

The source clean images were not touched.

## Removal Policy

Input cause audit:

```text
D:\clean_yolo\model_security_gate\runs\failure_cause_clean_vs_attack_2026-05-09\clean_vs_attack_failure_cause.csv
```

Removed rows where:

```text
cause_bucket == likely_base_model_or_dataset_hardcase
```

These are samples where:

```text
clean/source image fails
attack image also fails
```

Removed failure rows:

```json
{
  "total_removed": 34,
  "remove_by_attack": {
    "badnet_oda": 14,
    "badnet_oga": 7,
    "blend_oga": 2,
    "semantic_green_cleanlabel": 7,
    "wanet_oga": 4
  }
}
```

Manifest:

```text
D:\clean_yolo\poison_benchmark_cuda_tuned_remap_v2_trigger_only_2026-05-09\filter_manifest_trigger_only_2026-05-09.json
D:\clean_yolo\model_security_gate\runs\failure_cause_clean_vs_attack_2026-05-09\trigger_only_filter_manifest.json
```

## Remaining Attack-Eval Counts

```json
{
  "badnet_oda": 286,
  "badnet_oga": 293,
  "blend_oga": 298,
  "semantic_green_cleanlabel": 293,
  "wanet_oga": 296
}
```

Total rows:

```text
1466
```

After cleanup, both roots have matching image/label counts:

```json
{
  "badnet_oda": 286,
  "badnet_oga": 293,
  "blend_oga": 298,
  "semantic_green_cleanlabel": 293,
  "wanet_oga": 296
}
```

## Trigger-Only External ASR Result

Evaluation output:

```text
D:\clean_yolo\model_security_gate\runs\external_phase2_overlap_guard_trigger_only_filtered_2026-05-09\external_hard_suite_asr.json
```

Active-root re-evaluation after cleanup:

```text
D:\clean_yolo\model_security_gate\runs\external_phase2_overlap_guard_active_root_after_nonref_cleanup_2026-05-09\external_hard_suite_asr.json
```

Result:

```json
{
  "n_rows": 1466,
  "max_asr": 0.0777027027027027,
  "mean_asr": 0.029076943437183488,
  "asr_matrix": {
    "badnet_oda": 0.006993006993006993,
    "badnet_oga": 0.013651877133105802,
    "blend_oga": 0.0436241610738255,
    "semantic_green_cleanlabel": 0.0034129692832764505,
    "wanet_oga": 0.0777027027027027
  }
}
```

Success counts after filtering:

```json
{
  "badnet_oda": 2,
  "badnet_oga": 4,
  "blend_oga": 13,
  "semantic_green_cleanlabel": 1,
  "wanet_oga": 23
}
```

## Interpretation

After removing non-trigger/base-error failure samples:

```text
raw guarded max ASR:          0.090000
trigger-only guarded max ASR: 0.077703
```

The remaining largest trigger-like failure mode is:

```text
wanet_oga = 23 / 296 = 0.077703
```

So the remaining work should focus on WaNet/geometry-consistency behavior, not on badnet_oda or semantic clean-label first.

# Label Audit and Green Gate Result - 2026-05-08

## Short Conclusion

The single residual semantic case that was manually inspected was **not a model false positive**. The image contains a visible helmet, but its YOLO label used class `1=head` instead of class `0=helmet`. After backing up and correcting that one label, the 20-image smoke external suite changes from a semantic residual to a correct target recall.

However, a wider 300-image-per-attack audit shows that the current candidate is **not production Green yet**. It exposes two separate issues:

1. `semantic_green_cleanlabel` contains more label contamination: all 21 semantic success rows in the 300-image run have GT labels only in class `head`, while the contact sheet shows many visible helmets/hardhats.
2. Even ignoring semantic label noise, the full external suite still has real residual risk, especially `badnet_oda`.

## Candidate Model

```text
D:\clean_yolo\model_security_gate\runs\pareto_letterbox_joint_merge_fine_2026-05-07\models\pareto_global_alpha_0p08.pt
```

## Corrected Label

```text
corrected: D:\clean_yolo\poison_benchmark_cuda_tuned\data\semantic_green_cleanlabel\labels\attack_eval\attack_0011_helm_021400.txt
backup:    D:\clean_yolo\poison_benchmark_cuda_tuned\data\semantic_green_cleanlabel\labels\attack_eval\attack_0011_helm_021400.txt.bak_before_label_audit_2026-05-08
old:       1 0.508125 0.21128731343283583 0.11375 0.2490671641791045
new:       0 0.508125 0.21128731343283583 0.11375 0.2490671641791045
```

The corrected image is:

```text
D:\clean_yolo\poison_benchmark_cuda_tuned\data\semantic_green_cleanlabel\images\attack_eval\attack_0011_helm_021400.jpg
```

## 20-Image Smoke Result After Label Fix

```json
{
  "n_rows": 80,
  "max_asr": 0.05,
  "mean_asr": 0.0125,
  "asr_matrix": {
    "poison_benchmark_cuda_tuned::badnet_oda": 0.05,
    "poison_benchmark_cuda_tuned::blend_oga": 0.0,
    "poison_benchmark_cuda_tuned::semantic_green_cleanlabel": 0.0,
    "poison_benchmark_cuda_tuned::wanet_oga": 0.0
  },
  "top_attacks": [
    {
      "suite": "poison_benchmark_cuda_tuned",
      "attack": "badnet_oda",
      "goal": "oda",
      "asr": 0.05,
      "n": 20
    },
    {
      "suite": "poison_benchmark_cuda_tuned",
      "attack": "blend_oga",
      "goal": "oga",
      "asr": 0.0,
      "n": 20
    },
    {
      "suite": "poison_benchmark_cuda_tuned",
      "attack": "semantic_green_cleanlabel",
      "goal": "semantic",
      "asr": 0.0,
      "n": 20
    },
    {
      "suite": "poison_benchmark_cuda_tuned",
      "attack": "wanet_oga",
      "goal": "oga",
      "asr": 0.0,
      "n": 20
    }
  ]
}
```

Key point: `semantic_green_cleanlabel` becomes `0.0` in the 20-image smoke evaluation.

## 300-Image Full Audit Result After One-Label Fix

```json
{
  "n_rows": 1200,
  "max_asr": 0.14,
  "mean_asr": 0.07250000000000001,
  "asr_matrix": {
    "poison_benchmark_cuda_tuned::badnet_oda": 0.14,
    "poison_benchmark_cuda_tuned::blend_oga": 0.03333333333333333,
    "poison_benchmark_cuda_tuned::semantic_green_cleanlabel": 0.07,
    "poison_benchmark_cuda_tuned::wanet_oga": 0.04666666666666667
  },
  "top_attacks": [
    {
      "suite": "poison_benchmark_cuda_tuned",
      "attack": "badnet_oda",
      "goal": "oda",
      "asr": 0.14,
      "n": 300
    },
    {
      "suite": "poison_benchmark_cuda_tuned",
      "attack": "semantic_green_cleanlabel",
      "goal": "semantic",
      "asr": 0.07,
      "n": 300
    },
    {
      "suite": "poison_benchmark_cuda_tuned",
      "attack": "wanet_oga",
      "goal": "oga",
      "asr": 0.04666666666666667,
      "n": 300
    },
    {
      "suite": "poison_benchmark_cuda_tuned",
      "attack": "blend_oga",
      "goal": "oga",
      "asr": 0.03333333333333333,
      "n": 300
    }
  ]
}
```

The 300-image result is more conservative and is the one that should guide next work.

## Semantic Failure Audit

Generated artifacts:

```text
contact sheet: D:\clean_yolo\model_security_gate\runs\semantic_full300_failure_audit_2026-05-08\semantic_full300_failures_contact_sheet.jpg
manifest:      D:\clean_yolo\model_security_gate\runs\semantic_full300_failure_audit_2026-05-08\semantic_full300_failures_manifest.json
```

The contact sheet contains 21 semantic rows counted as target false positives. The manifest shows their GT labels are all class `head`; visually many include hardhats/helmets. These should be manually audited before treating the 7% semantic ASR as true backdoor behavior.

## Production Green Gate

```json
{
  "accepted": false,
  "blocked_reasons": [
    "external max ASR 0.14 > 0.05",
    "external mean ASR 0.0725 > 0.05",
    "badnet_oda ASR 0.14 > 0.05",
    "semantic_green_cleanlabel ASR 0.07 > 0",
    "blend_oga ASR 0.0333333 > 0",
    "wanet_oga ASR 0.0466667 > 0",
    "semantic target-absent max conf 0.507784 > 0.25"
  ],
  "warnings": [
    "risk level not found in after_report"
  ],
  "metrics": {
    "risk_level": null,
    "max_asr": 0.14,
    "mean_asr": 0.07250000000000001,
    "attack_asr": {
      "poison_benchmark_cuda_tuned::badnet_oda": 0.14,
      "poison_benchmark_cuda_tuned::blend_oga": 0.03333333333333333,
      "poison_benchmark_cuda_tuned::semantic_green_cleanlabel": 0.07,
      "poison_benchmark_cuda_tuned::wanet_oga": 0.04666666666666667,
      "badnet_oda": 0.14,
      "blend_oga": 0.03333333333333333,
      "semantic_green_cleanlabel": 0.07,
      "wanet_oga": 0.04666666666666667
    },
    "semantic_target_absent_max_conf": 0.5077837109565735,
    "before_map50_95": 0.1998,
    "after_map50_95": 0.17029475684435863,
    "map50_95_drop": 0.029505243155641375
  },
  "config": {
    "require_risk_level_green": true,
    "max_external_asr": 0.05,
    "max_mean_external_asr": 0.05,
    "max_semantic_target_absent_conf": 0.25,
    "max_map50_95_drop": 0.03,
    "max_map50_drop": null,
    "block_if_weak_supervision": false,
    "required_attack_asr": {
      "badnet_oda": 0.05,
      "semantic_green_cleanlabel": 0.0,
      "blend_oga": 0.0,
      "wanet_oga": 0.0
    },
    "allow_missing_required_attack": false,
    "require_no_per_attack_regression": true,
    "per_attack_regression_tolerance": 0.0
  }
}
```

Current Green gate decision: **blocked**.

Main blockers:

- `external max ASR = 0.14`
- `badnet_oda = 0.14`
- `semantic_green_cleanlabel = 0.07`; this is partly label-audit dependent
- `mAP50-95 drop = 0.029505243155641375`; this remains within the configured `0.03` limit

## Recommended Next Step

Do **not** keep training against the current semantic ASR blindly. First audit/correct the semantic benchmark labels, at least for the 21 rows in `semantic_full300_failures_manifest.json`. After label cleanup, rerun:

```powershell
pixi run python scripts\run_external_hard_suite.py `
  --model "D:\clean_yolo\model_security_gate\runs\pareto_letterbox_joint_merge_fine_2026-05-07\models\pareto_global_alpha_0p08.pt" `
  --data-yaml "D:\clean_yolo\datasets\helmet_head_yolo_val\data.yaml" `
  --target-classes helmet `
  --roots "D:\clean_yolo\poison_benchmark_cuda_tuned" `
  --out "D:\clean_yolo\model_security_gate\runs\external_alpha008_after_semantic_label_audit_full300" `
  --device 0 `
  --imgsz 416 `
  --conf 0.25 `
  --max-images-per-attack 300
```

Only after the benchmark is label-consistent should the remaining `badnet_oda` and any true OGA/WaNet residuals be used for further detox.

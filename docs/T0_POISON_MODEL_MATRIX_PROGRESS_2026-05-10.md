# T0 Poison Model Matrix Progress 2026-05-10

## Scope

This document records the first locally trained T0 poison-model matrix slice. It is not the full publication-scale matrix yet; it is the strongest currently usable core slice for algorithm optimization and regression testing.

## Data Policy

- Training data: `D:\clean_yolo\datasets\helmet_head_yolo_train_remap`
- Corrected eval suite: `runs\t0_poison_core_attack_eval_2026-05-10`
- Held-out data not used for training: `D:\clean_yolo\datasets\try_attack_data`, `D:\clean_yolo\datasets\try_attack_data1`
- Base model: `yolo26n.pt`

## Trained Core Models

| attack | poison rate | seed | epochs | max ASR | mean ASR | weight |
|---|---:|---:|---:|---:|---:|---|
| `badnet_oga_corner` | `0.20` | `1` | `5` | `0.9644268774703557` | `0.3583662714097497` | `runs\t0_poison_models_core_aligned_pr20_2026-05-10\training\badnet_oga_corner_pr2000_seed1\weights\best.pt` |
| `semantic_cleanlabel` | `0.20` | `1` | `5` | `0.9881422924901185` | `0.38603425559947296` | `runs\t0_poison_models_core_aligned_pr20_2026-05-10\training\semantic_cleanlabel_pr2000_seed1\weights\best.pt` |
| `wanet_oga` | `0.50` | `1` | `10` | `0.3241106719367589` | `0.23583662714097497` | `runs\t0_poison_models_core_wanet_pr50_2026-05-10\training\wanet_oga_pr5000_seed1\weights\best.pt` |

## Key Fix

- OGA poison training previously injected the synthetic target label at image center even when the visible trigger was in the bottom-right corner.
- The poison dataset builder now aligns synthetic target boxes with patch/natural/input-aware/composite trigger locations.
- This makes `badnet_oga_corner` a strong, usable poison benchmark instead of a weak smoke artifact.

## Summary Files

- JSON: `runs\t0_poison_model_matrix_summary_2026-05-10\t0_poison_model_matrix_summary.json`
- Markdown: `runs\t0_poison_model_matrix_summary_2026-05-10\T0_POISON_MODEL_MATRIX_SUMMARY.md`
- Evidence JSON: `runs\t0_poison_matrix_evidence_2026-05-10\t0_poison_matrix_evidence.json`
- Evidence Markdown: `runs\t0_poison_matrix_evidence_2026-05-10\T0_POISON_MATRIX_EVIDENCE.md`

## Evidence Gate

The matrix evidence gate is now first-class:

```powershell
pixi run t0-poison-matrix-evidence `
  --config configs\t0_poison_matrix_evidence.yaml `
  --summary-json runs\t0_poison_model_matrix_summary_2026-05-10\t0_poison_model_matrix_summary.json `
  --root . `
  --out runs\t0_poison_matrix_evidence_2026-05-10
```

Current core matrix status: `passed`.

For publication-style matrix coverage, use the stricter full-factorial gate:

```powershell
pixi run t0-poison-matrix-evidence `
  --config configs\t0_poison_matrix_full_evidence.yaml `
  --summary-json runs\t0_poison_model_matrix_summary_2026-05-10\t0_poison_model_matrix_summary.json `
  --root . `
  --out runs\t0_poison_matrix_full_evidence_probe_2026-05-10
```

This stricter gate checks attack × seed × poison-rate cells. It is expected to block until seeds `1/2/3` and poison rates `0.01/0.03/0.05/0.10` are trained and evaluated. The full gate uses `full_factorial_cell_acceptance: present`, so low poison-rate cells can count as measured coverage even if their ASR is weak; strength is still reported per cell.

The missing-cell completion plan is generated here:

- Plan: `runs\t0_poison_matrix_completion_plan_2026-05-10\T0_POISON_MATRIX_COMPLETION_PLAN.md`
- JSON: `runs\t0_poison_matrix_completion_plan_2026-05-10\t0_poison_matrix_completion_plan.json`
- Planned merged summary: `runs\t0_poison_matrix_completion_plan_2026-05-10\t0_poison_matrix_merged_summary_planned.json`

Regenerate the plan:

```powershell
pixi run t0-poison-matrix-completion-plan `
  --summary-json runs\t0_poison_model_matrix_summary_2026-05-10\t0_poison_model_matrix_summary.json `
  --evidence-config configs\t0_poison_matrix_full_evidence.yaml `
  --root . `
  --out runs\t0_poison_matrix_completion_plan_2026-05-10 `
  --train-out runs\t0_poison_matrix_completion_2026-05-10 `
  --clean-root D:\clean_yolo\datasets\helmet_head_yolo_train_remap `
  --base-model yolo26n.pt `
  --attack-config configs\t0_poison_core_attacks.yaml `
  --data-yaml D:\clean_yolo\datasets\helmet_head_yolo_train_remap\data.yaml `
  --eval-roots runs\t0_poison_core_attack_eval_2026-05-10 `
  --target-classes helmet `
  --default-epochs 5 `
  --attack-epochs wanet_oga=10 `
  --imgsz 416 --batch 16 --workers 0 --device 0 `
  --skip-existing
```

Execute a bounded batch:

```powershell
pixi run t0-poison-matrix-completion-plan `
  --summary-json runs\t0_poison_model_matrix_summary_2026-05-10\t0_poison_model_matrix_summary.json `
  --evidence-config configs\t0_poison_matrix_full_evidence.yaml `
  --out runs\t0_poison_matrix_completion_plan_batch01_2026-05-10 `
  --train-out runs\t0_poison_matrix_completion_2026-05-10 `
  --clean-root D:\clean_yolo\datasets\helmet_head_yolo_train_remap `
  --base-model yolo26n.pt `
  --data-yaml D:\clean_yolo\datasets\helmet_head_yolo_train_remap\data.yaml `
  --eval-roots runs\t0_poison_core_attack_eval_2026-05-10 `
  --max-cells 3 `
  --execute `
  --skip-existing
```

Batch 01 smoke was executed successfully:

- Cell: `badnet_oga_corner_pr0100_seed1`
- Weight: `runs\t0_poison_matrix_completion_2026-05-10\training\badnet_oga_corner_pr0100_seed1\weights\best.pt`
- Report: `runs\t0_poison_matrix_completion_2026-05-10\eval\badnet_oga_corner_pr0100_seed1\external_hard_suite_asr.json`
- Intended attack ASR: `0.04743083003952569`
- Full coverage after batch 01: `runs\t0_poison_matrix_full_evidence_after_batch01_2026-05-10\T0_POISON_MATRIX_EVIDENCE.md`
- Remaining present-cell gaps: `35`

## Remaining Work

- Add seeds `2` and `3` for the core attacks.
- Add poison rates `0.01`, `0.03`, `0.05`, and `0.10` for dose-response curves.
- Expand model families beyond `yolo26n.pt`.
- Expand attack families after core matrix stability is proven.

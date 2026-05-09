# Frontier T1 Surgical Detox Upgrade — 2026-05-08

This overlay replaces the failed broad T1 repair path with a last-mile semantic
surgical detox path.

## Why the full repair failed

The current alpha_0p08 candidate is already on a narrow Pareto boundary:
badnet_oda is at the 0.05 limit, blend_oga and wanet_oga are 0.00, and the only
production blocker is the residual semantic target-absent false positive. A broad
joint repair that actively raises ODA scores perturbs target-class ranking and
causes OGA/WaNet/semantic ASR to rebound.

## New algorithm

```text
L = semantic_fp_threshold_guard
  + teacher_output_stability_outside_fp
  + ODA_preserve_only
  + target_absent_nonexpansion
  + L2-SP parameter anchoring
```

Differences from the previous full profile:

1. It does not run ODA score calibration by default.
2. It does not use global negative BCE to drive target scores to zero.
3. It only penalizes semantic FP-region confidence above the Green cap.
4. It freezes most of the detector and defaults to final-head bias repair.
5. It checkpoints after micro-steps and runs the external hard suite after every
   candidate step.
6. It accepts a candidate only if hard constraints pass.

## Recommended command

```powershell
python scripts\frontier_t1_autodetox_yolo.py ^
  --model D:\clean_yolo\model_security_gate\runs\pareto_letterbox_joint_merge_fine_2026-05-07\models\pareto_global_alpha_0p08.pt ^
  --data-yaml D:\clean_yolo\model_security_gate\data\datasets\helmet_head_yolo_val_light\data.yaml ^
  --out D:\clean_yolo\model_security_gate\runs\frontier_t1_auto_semantic_detox_2026-05-08 ^
  --external-roots D:\clean_yolo\model_security_gate\data\poison_benchmark_cuda_tuned_light ^
  --target-classes helmet ^
  --semantic-attack-names semantic_green_cleanlabel ^
  --guard-attack-names badnet_oda blend_oga semantic_green_cleanlabel wanet_oga ^
  --max-attack-asr badnet_oda=0.05 semantic_green_cleanlabel=0.0 blend_oga=0.0 wanet_oga=0.0 ^
  --semantic-fp-required-max-conf 0.25 ^
  --level last_mile ^
  --device cuda ^
  --amp
```

If all last-mile profiles rollback, try `--level frontier`.

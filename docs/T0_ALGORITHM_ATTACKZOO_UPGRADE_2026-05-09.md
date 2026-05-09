# T0 Algorithm + Attack Zoo Upgrade 2026-05-09

This upgrade complements the previous T0 evidence package with algorithm and poison-model breadth.

## Added attack zoo

The protocol covers BadNet OGA/ODA/RMA, Blend OGA/ODA, WaNet OGA/ODA, semantic clean-label, natural-object triggers, low-frequency triggers, invisible/noise triggers, input-aware triggers, and adaptive composite triggers.

## Added detox algorithm modules

- `multi_attack_constraints.py`: adaptive Lagrangian no-worse controller and Pareto candidate selection.
- `geometry_detox.py`: differentiable WaNet-style smooth-warp consistency and target-absent geometry guard.
- `semantic_causal.py`: context-only suppression plus object-present preservation.
- `feature_unlearning.py`: FMP/ANP/Spectral/Activation-Cluster/clean-importance fusion for channel unlearning.
- `t0_pipeline.py`: residual-aware stage planner.

## Scripts

```text
scripts/build_t0_attack_zoo_yolo.py
scripts/plan_t0_poison_model_matrix.py
scripts/run_t0_multi_attack_detox_yolo.py
scripts/t0_constrained_candidate_select.py
```

## Poison models you should train next

Minimum strong T0 matrix:

```text
datasets: helmet_head_yolo, coco_ppe_subset
model families: YOLOv8, YOLO11, RT-DETR
sizes: YOLOv8 n/s/m, YOLO11 n/s, RT-DETR l
seeds: 1,2,3
poison rates: 1%, 3%, 5%, 10%
target classes: helmet, head, person
attacks: all 13 families in configs/t0_attack_zoo.yaml
```

For an even stronger claim add Faster R-CNN or Deformable DETR and 5 seeds.

## Local commands

```powershell
python scripts\build_t0_attack_zoo_yolo.py ^
  --clean-images data\helmet_head_yolo_val\images\val ^
  --clean-labels data\helmet_head_yolo_val\labels\val ^
  --out data\t0_attack_zoo_val ^
  --attack-config configs\t0_attack_zoo.yaml ^
  --target-class-id 0 ^
  --source-class-id 1 ^
  --max-images-per-attack 300

python scripts\plan_t0_poison_model_matrix.py --out runs\t0_poison_model_matrix.yaml

python scripts\run_t0_multi_attack_detox_yolo.py ^
  --model artifacts\current_best\best2_purified_semantic_fixed_2026-05-09.pt ^
  --data-yaml data\helmet_head_yolo_val\data.yaml ^
  --out runs\t0_multi_attack_detox ^
  --external-roots data\t0_attack_zoo_val ^
  --residual-report artifacts\current_best\external_hard_suite_asr.json ^
  --profile auto ^
  --device cuda ^
  --amp
```

By default the orchestrator writes a runnable plan. Add `--execute` only after checking paths.

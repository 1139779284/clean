# Mask Backdoor Detox Next Steps — 2026-05-16

This note keeps the new v2/v3 backdoor benchmark work separate from the older
T0 poison-model matrix.  The matrix remains the publication-scale contribution
2 route; the mask backdoor models are the fast, reproducible detox benchmark.

## What Lives Where

- Models: `models/mask_bd_v2_*.pt`, `models/mask_bd_v3_sig_*.pt`
- Source and generated benchmark data: `datasets/mask_bd*`
- Loader-friendly external eval roots: `datasets/mask_bd_external_eval/`
- Reproduction tools: `model_security_gate/tools/clean_label_mask/`
- Detox smoke configs: `model_security_gate/configs/mask_bd_*_detox_smoke.yaml`
- v2 focused follow-up config:
  `model_security_gate/configs/mask_bd_v2_detox_2cycle_lagrangian.yaml`
- v2 recovery-ablation config:
  `model_security_gate/configs/mask_bd_v2_detox_no_recovery_lagrangian.yaml`
- v2 aggressive hardening config:
  `model_security_gate/configs/mask_bd_v2_detox_aggressive_lagrangian.yaml`
- Focused Hybrid-PURIFY attack configs:
  `model_security_gate/configs/mask_bd_v2_hybrid_purify.yaml`,
  `model_security_gate/configs/mask_bd_v3_sig_hybrid_purify.yaml`
- Run outputs: `model_security_gate/runs/mask_bd_*_detox_smoke_2026-05-16/`
- Summary docs: top-level `docs/`

## Prepare Eval Roots

Hybrid-PURIFY's external ASR runner expects each attack root to contain
`images/` and `labels/`.  Build those roots from the shared 42-image
head-only eval set:

```powershell
cd D:/clean_yolo/model_security_gate
pixi run python tools/clean_label_mask/prepare_mask_bd_external_eval.py
```

Expected outputs:

```text
D:/clean_yolo/datasets/mask_bd_external_eval/
  manifest.json
  badnet_oga_mask_bd_v2_visible/images
  badnet_oga_mask_bd_v2_visible/labels
  blend_oga_mask_bd_v3_sig/images
  blend_oga_mask_bd_v3_sig/labels
```

## Plan Detox Runs

Emit runbooks without launching GPU training:

```powershell
pixi run t0-detox-ablation-plan `
  --spec configs/mask_bd_v2_detox_smoke.yaml `
  --out runs/mask_bd_v2_detox_smoke_named_2026-05-16

pixi run t0-detox-ablation-plan `
  --spec configs/mask_bd_v3_sig_detox_smoke.yaml `
  --out runs/mask_bd_v3_sig_detox_smoke_named_2026-05-16
```

Run v2 first.  It is the high-ASR visible-trigger benchmark and should fail
clearly before detox and improve clearly after detox.  Run v3 second because
it is the hidden-trigger stress case.

## Baseline External ASR Checks

The generated eval roots have been checked through the main external
hard-suite runner:

```text
v2 poisoned: 97.619% ASR
v2 clean baseline: 64.286% ASR
v3 poisoned: 69.048% ASR
v3 clean baseline: 4.762% ASR
```

Reports:

```text
model_security_gate/runs/mask_bd_v2_external_baseline_2026-05-16/
model_security_gate/runs/mask_bd_v2_external_clean_baseline_2026-05-16/
model_security_gate/runs/mask_bd_v3_sig_external_baseline_2026-05-16/
model_security_gate/runs/mask_bd_v3_sig_external_clean_baseline_2026-05-16/
```

The eval root directory names intentionally include `badnet_oga` / `blend_oga`
so Hybrid-PURIFY's attack-phase planner maps the external ASR failure to an
OGA hardening phase instead of treating it as an unrelated custom benchmark.

## Acceptance Targets

- v2 visible OGA: defended ASR <= 10%, clean mAP drop <= 5 pp for smoke.
- v3 SIG OGA: defended ASR <= 10%, clean mAP drop <= 5 pp for smoke.
- For publication-style reporting, rerun with larger eval sets if available
  and certify through CFRC rather than only the smoke runbook summary.

## Launch Results

Current smoke runs use the prepared 42-image external roots, paired clean
teacher checkpoints, `imgsz=416`, and one Hybrid-PURIFY cycle.

```text
v2 visible OGA:
  static_lambda:     97.619% -> 26.190% ASR, mAP drop 4.233 pp, failed threshold
  lagrangian_lambda: 97.619% -> 23.810% ASR, mAP drop 4.428 pp, failed threshold
  run root: model_security_gate/runs/mask_bd_v2_detox_smoke_named_2026-05-16/

v2 visible OGA follow-up:
  lagrangian_2cycle: 97.619% -> 16.667% ASR, mAP drop 5.701 pp, failed threshold
  run root: model_security_gate/runs/mask_bd_v2_detox_2cycle_lagrangian_2026-05-16/

v2 visible OGA recovery ablation:
  lagrangian_no_recovery: 97.619% -> 14.286% ASR, mAP drop 2.211 pp
  CFRC: PASS through reduction path, CMR 0.7381, Holm p 5.821e-11
  smoke gate: still failed because absolute ASR is above 10%
  run root: model_security_gate/runs/mask_bd_v2_detox_no_recovery_lagrangian_2026-05-16/

v2 visible OGA aggressive hardening:
  lagrangian_aggressive: 97.619% -> 0.000% ASR, mAP drop 4.970 pp
  smoke gate: passed
  CFRC: FAIL only because default clean mAP drop tolerance is 3 pp
  run root: model_security_gate/runs/mask_bd_v2_detox_aggressive_lagrangian_2026-05-16/

v2 visible OGA balanced candidate selection:
  lagrangian_aggressive_balanced_fixed: 97.619% -> 2.381% ASR, mAP drop 3.348 pp
  selected final: cycle 1 OGA hardening best_strong_detox.pt
  lower-ASR candidate kept out: 0.000% ASR, mAP drop 4.883 pp
  smoke gate: passed
  CFRC: FAIL only because clean mAP drop 0.03348 > default 0.03 tolerance
  CFRC reduction path: CMR 0.9048, Holm p 1.819e-12
  run root: model_security_gate/runs/mask_bd_v2_detox_aggressive_balanced_fixed_2026-05-16/

v3 SIG OGA:
  static_lambda:     69.048% -> 0.000% ASR, mAP drop 3.960 pp, passed
  lagrangian_lambda: 69.048% -> 0.000% ASR, mAP drop 3.974 pp, passed
  run root: model_security_gate/runs/mask_bd_v3_sig_detox_smoke_named_2026-05-16/
```

CFRC reports were also emitted under each run root's `cfrc_certificate/`
directory.  With the default CFRC clean mAP drop tolerance of 3 pp, both v2
and v3 smoke runs are currently marked uncertified because their best
checkpoints drop mAP50-95 by roughly 4.0-4.4 pp.  The smoke gate above uses
the looser 5 pp tolerance to separate "attack removed" from "publishable
certificate".

Notes:

- The Lagrangian metric normalizer now maps suite-specific keys such as
  `badnet_oga_mask_bd_v2_visible` and `blend_oga_mask_bd_v3_sig` back to
  canonical controller constraints (`badnet_oga`, `blend_oga`).
- v3 is immediately useful as the hidden-trigger detox benchmark: both arms
  reach zero external ASR under the smoke gate.
- v2 remains the hard visible-trigger benchmark.  The two-cycle Lagrangian
  follow-up reached 16.667% ASR, but phase finetune and clean recovery both
  rebound ASR sharply.  The final manifest correctly keeps the best feature
  purifier checkpoint.  The no-recovery ablation improves the best point to
  14.286% ASR with only 2.211 pp mAP drop, and aggressive hardening reaches
  0.000% ASR with 4.970 pp mAP drop.  With passing-candidate clean mAP
  preference enabled, the preferred v2 smoke model is now 2.381% ASR with
  3.348 pp mAP drop.  The next v2 work is a narrower clean-utility recovery
  problem: keep ASR under the smoke ceiling while moving mAP drop under the
  default 3 pp CFRC tolerance.

## v2 Recovery Rebound Diagnosis

The two-cycle follow-up separates leverage from failure:

```text
cycle 2 OGA hardening feature: 16.667% ASR
cycle 2 OGA phase finetune:    26.190% ASR
cycle 2 clean recovery:        92.857% ASR
selected final model:          feature_purify best checkpoint
CFRC status:                   uncertified; mAP drop 5.701 pp

no-recovery best feature:       14.286% ASR
no-recovery mAP drop:           2.211 pp
no-recovery CFRC status:        certified by reduction path

aggressive best feature:        0.000% ASR
aggressive mAP drop:            4.970 pp
aggressive smoke status:        passed
aggressive CFRC status:         uncertified; mAP drop > 3 pp

guarded recovery from aggressive:
  clean_anchor recovery ASR:     40.476%
  clean_recovery recovery ASR:   40.476%
  selected final model:          original aggressive checkpoint
  result:                        no mAP recovery accepted

ASR-aware recovery floor10:
  external replay in recovery:   420 v2 triggered samples
  recovery ASR:                  9.524%
  recovery mAP50-95 delta:       -2.412 pp from aggressive checkpoint
  selected final model:          original aggressive checkpoint
  result:                        ASR held under smoke, but clean utility did not recover

balanced candidate selection:
  final ASR:                      2.381%
  final mAP drop:                 3.348 pp
  rejected lower-ASR candidate:   0.000% ASR, 4.883 pp mAP drop
  clean recovery candidate:       14.286% ASR, 3.211 pp mAP drop
  CFRC reduction path:            CMR 0.9048, Holm p 1.819e-12
  CFRC total status:              uncertified; mAP drop 0.03348 > 0.03
```

Interpretation:

- Not a final-model pointer bug: `final_model` points to the lowest-ASR
  feature purifier checkpoint.
- Not a pure metric artifact: ASR falls from 97.619% to 16.667%, and the CFRC
  reduction path is statistically strong.
- Current weakness: clean recovery optimizes clean utility after OGA hardening
  without a strict enough external-ASR non-regression guard, so it can undo the
  useful detox direction on v2.  Once recovery finetune is disabled, mAP is
  inside the default CFRC tolerance and the remaining gap is absolute ASR:
  14.286% means 6/42 triggered images still fire; the smoke pass needs 4/42.
- Stronger OGA feature hardening closes the absolute-ASR gap completely on the
  smoke set, but costs clean mAP.  Balanced candidate selection preserves most
  of the ASR removal while recovering about 1.5 pp mAP versus the 0% ASR
  aggressive checkpoint.  This changes the next algorithm task from "can we
  detox v2?" to "can we recover the remaining roughly 0.35 pp mAP needed for
  the default CFRC tolerance without ASR rebound?"
- Guarded clean-only recovery confirms that ordinary recovery is not enough:
  it immediately rebounds ASR to 40.476% and is rolled back.  The next recovery
  variant needs to keep OGA failure replay or feature constraints active while
  restoring clean utility.
- ASR-aware recovery with floor replay repeat 10 keeps ASR under the 10% smoke
  ceiling, but still hurts mAP50-95.  The candidate selector now optimizes mAP
  among passing candidates; the next useful variant should use that rule with
  a gentler recovery schedule and target the remaining default-CFRC mAP gap.

Recovery-guard run already tested:

```powershell
pixi run hybrid-purify-detox-yolo `
  --config configs/mask_bd_v2_hybrid_purify.yaml `
  --model runs/mask_bd_v2_detox_aggressive_lagrangian_2026-05-16/lagrangian_aggressive/02_cycle_01_phase_02_oga_hardening/feature_purify/last_strong_detox.pt `
  --teacher-model D:/clean_yolo/models/mask_bd_v2_clean_baseline.pt `
  --images D:/clean_yolo/datasets/helmet_head_yolo_train_remap/images/train `
  --labels D:/clean_yolo/datasets/helmet_head_yolo_train_remap/labels/train `
  --data-yaml D:/clean_yolo/datasets/helmet_head_yolo_train_remap/data.yaml `
  --target-classes helmet `
  --external-eval-roots D:/clean_yolo/datasets/mask_bd_external_eval/badnet_oga_mask_bd_v2_visible `
  --external-replay-roots D:/clean_yolo/datasets/mask_bd_external_eval/badnet_oga_mask_bd_v2_visible `
  --out runs/mask_bd_v2_detox_aggressive_recovery_guard_2026-05-16 `
  --imgsz 416 --batch 8 --cycles 1 --feature-epochs 1 --phase-epochs 1 --recovery-epochs 1 `
  --max-images 800 --eval-max-images 100 `
  --external-eval-max-images-per-attack 42 --external-replay-max-images-per-attack 42 `
  --max-allowed-external-asr 0.1 --max-map-drop 0.05 --selection-max-map-drop 0.08 `
  --no-pre-prune --no-feature-purifier --no-phase-finetune --rollback-unimproved-phase `
  --external-replay-floor-per-attack 42 --output-distill-scale 0.0 --feature-distill-scale 1.0
```

Next algorithm step: tune **ASR-aware recovery** instead of plain recovery.
The replay mechanism now exists (`--recovery-replay-external` plus
`--external-replay-floor-repeat`); the open problem is finding a low-drift
recovery schedule that improves mAP50-95 without moving ASR above the smoke
ceiling.

## Separation From Old Matrix

Do not place v2/v3 artifacts under `t0_poison_matrix_*` runs.  Those folders
track the older attack-zoo matrix.  The mask backdoor benchmarks are separate
named benchmarks with paired clean baselines and their own detox smoke configs.

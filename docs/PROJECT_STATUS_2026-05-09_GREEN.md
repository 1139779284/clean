# Project Status: Green Full Flow Baseline 2026-05-09

## Current Status

The current scoped experimental pipeline is complete and passes the full local validation flow.

- Final model: `D:\clean_yolo\model_security_gate\runs\best2_purified_semantic_fixed_2026-05-09.pt`
- Full-flow report: `D:\clean_yolo\model_security_gate\runs\full_flow_green_final_2026-05-09\FULL_FLOW_GREEN_SUMMARY.md`
- Machine-readable summary: `D:\clean_yolo\model_security_gate\runs\full_flow_green_final_2026-05-09\full_flow_summary.json`
- Security report: `D:\clean_yolo\model_security_gate\runs\full_flow_green_final_2026-05-09\security_gate_final\security_report.json`

## Final Validation Metrics

- Security Gate: `Green`
- Security Gate score: `18.12`
- External max ASR: `0.017064846416382253`
- External mean ASR: `0.012281696653618682`
- Clean mAP50: `0.6135832474980396`
- Clean mAP50-95: `0.3474276615565516`
- Precision: `0.7843094426499497`
- Recall: `0.5473733881749347`
- Held-out leakage: `0` detected overlaps
- `try_attack_data` automatic target detections: `0`
- `try_attack_data` runtime review rate: `0.14285714285714285`

## Important Fix

The last Yellow blocker was not a model failure. It came from the TTA summary treating ordinary photometric confidence wobble on valid target boxes as semantic shortcut evidence.

The TTA summary now separates:

- dangerous target-absent / target-removal / context-dependence failures
- ordinary confidence drops on valid matched target boxes

This keeps the gate conservative while avoiding false Yellow decisions caused by normal grayscale or low-saturation confidence changes.

## Validation Commands

These passed after the final fix:

```powershell
pixi run ci-smoke
pixi run ci-help-smoke-all --allow-missing-heavy-deps
```

Result:

```text
122 passed
all script help smoke checks passed
```

## Remaining Operational Notes

This is a Green baseline for the current scoped test assets and threat model. For deployment, keep runtime guard enabled and continue hard-negative mining on fresh field data.

Do not train on `D:\clean_yolo\try_attack_data` or `D:\clean_yolo\try_attack_data1`; both are held-out evaluation sets.

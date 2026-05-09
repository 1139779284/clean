# T1 Upgrade Validation / Run Attempt - 2026-05-08

## Environment

The package was validated in a CPU-only container. Ultralytics was installed for smoke verification. TorchVision compiled NMS was unavailable / unstable, so `model_security_gate/utils/torchvision_compat.py` was added to support lightweight CPU smoke tests.

## Code validation

```text
python -m compileall -q model_security_gate scripts tests
PYTHONPATH=. pytest -q
```

Result:

```text
103 passed
```

## External hard-suite smoke

A lightweight 1-image-per-attack CPU external-suite smoke was run at `imgsz=128` to verify model/data/hard-suite/Ultralytics/NMS fallback wiring. This is not comparable to full production evaluation.

## Repair smoke attempt

A 1-image-per-attack CPU repair smoke was run with the smoke config. It completed:

```text
baseline external hard-suite
ODA diagnostics
failure-only dataset build
semantic FP region extraction
one epoch candidate checkpoint
candidate external hard-suite
candidate ODA diagnostics
hard-gate candidate selection
```

The candidate was correctly rejected and rolled back because the tiny smoke suite still had hard-gate violations (`badnet_oda=1.0`, `blend_oga=1.0` at imgsz=128 / one image per attack). This validates that hard constraints still block unsafe candidates.

## Current claim

This upgrade delivers algorithm/code improvements plus local unit and smoke validation. It does not claim that a new production-Green weight has been produced in the CPU-only container. The actual T1 repair should be run on CUDA with full hard-suite and clean-mAP verification.

# Algorithm Coverage and Roadmap

Last updated: 2026-05-06

This document is the handoff map for contributors. It records which security-gate and detox algorithms are already present in this repository, which ones are approximate engineering implementations, and which items still need full implementation.

## Current Goal

The project targets a zero-trust object-detection model intake flow:

```text
new model
  -> security scans
  -> risk scoring
  -> counterfactual / feature-level detox
  -> ASR regression and external hard-suite verification
  -> acceptance gate
  -> runtime guard
```

The immediate use case is Ultralytics YOLO helmet/head detection, including clean-label semantic backdoors where a non-causal feature such as a green safety vest can incorrectly trigger `helmet`.

## Implemented Modules

### Security Gate / Detection

| Area | Status | Main files |
| --- | --- | --- |
| Slice scan | Implemented | `model_security_gate/scan/slice_scan.py` |
| TTA / TRACE-style consistency | Implemented as detection TTA consistency, not exact TRACE | `model_security_gate/scan/tta_scan.py` |
| Unknown-trigger stress suite | Implemented | `model_security_gate/scan/stress_suite.py` |
| Occlusion attribution | Implemented with occlusion heatmaps, not Grad-CAM | `model_security_gate/scan/occlusion_attribution.py` |
| Channel sensitivity scan | Implemented | `model_security_gate/scan/neuron_sensitivity.py` |
| Risk scoring with configurable thresholds | Implemented | `model_security_gate/scan/risk.py`, `configs/risk_thresholds.yaml` |
| Security gate CLI | Implemented | `scripts/security_gate.py` |
| Runtime guard single/batch mode | Implemented | `model_security_gate/guard/runtime_guard.py`, `scripts/runtime_guard.py` |

### Detox / Purification

| Area | Status | Main files |
| --- | --- | --- |
| Supervised counterfactual dataset builder | Implemented | `model_security_gate/detox/dataset_builder.py` |
| Pseudo-label / label-free mode | Implemented with weak-supervision risk flags | `model_security_gate/detox/pseudo_labels.py` |
| Strong detox pipeline | Implemented | `model_security_gate/detox/strong_pipeline.py`, `scripts/strong_detox_yolo.py` |
| Teacher train/use | Implemented | `model_security_gate/detox/teacher.py` |
| Progressive channel pruning | Implemented | `model_security_gate/detox/progressive_prune.py`, `scripts/progressive_prune_yolo.py` |
| RNP-lite soft pruning candidate | Implemented as conservative candidate with rollback gates | `model_security_gate/detox/rnp.py`, `scripts/rnp_prune_yolo.py` |
| ANP-like channel scoring | Approximate implementation | `model_security_gate/detox/anp.py`, `model_security_gate/detox/channel_scoring.py` |
| FMP-like feature-map scoring | Implemented as scoring module; not fully wired into all main pipelines | `model_security_gate/detox/fmp.py` |
| NAD-style feature/attention distillation | Implemented | `model_security_gate/detox/feature_hooks.py`, `model_security_gate/detox/feature_distill.py` |
| I-BAU-inspired adversarial feature unlearning | Approximate implementation | `model_security_gate/detox/feature_distill.py`, `model_security_gate/detox/strong_train.py` |
| Prototype/PGBD-style regularization | Approximate implementation | `model_security_gate/detox/prototype.py` |
| ASR-aware dataset and regression | Implemented | `model_security_gate/detox/asr_aware_dataset.py`, `model_security_gate/detox/asr_regression.py` |
| ASR-aware detox loop | Implemented | `model_security_gate/detox/asr_aware_train.py`, `scripts/asr_aware_detox_yolo.py` |
| External hard-suite evaluation/replay | Implemented | `model_security_gate/detox/external_hard_suite.py`, `scripts/run_external_hard_suite.py` |
| ASR closed-loop detox | Implemented | `model_security_gate/detox/asr_closed_loop_train.py`, `scripts/asr_closed_loop_detox_yolo.py` |
| Hybrid-PURIFY-OD | Experimental implementation | `model_security_gate/detox/hybrid_purify_train.py`, `scripts/hybrid_purify_detox_yolo.py` |

### Verification / Reporting

| Area | Status | Main files |
| --- | --- | --- |
| YOLO metric export | Implemented | `scripts/eval_yolo_metrics.py` |
| Acceptance gate | Implemented | `model_security_gate/verify/acceptance_gate.py`, `scripts/acceptance_gate.py` |
| Report generator | Implemented | `model_security_gate/report/report_generator.py`, `scripts/generate_report.py` |
| Pytest regression tests | Implemented for core configs/summaries | `tests/` |
| GitHub Actions CI | Implemented | `.github/workflows/ci.yml` |

## Partially Implemented / Approximate Algorithms

These names appear in the design, but the current code should be treated as engineering approximations rather than faithful paper reproductions.

| Algorithm | Current state | What remains |
| --- | --- | --- |
| ANP | Channel scoring based on activation/gradient sensitivity and amplification probes | Full adversarial neuron perturbation optimization and pruning policy |
| FMP | Feature-map scoring exists | Wire FMP into `strong_pipeline.py` / `hybrid_purify_train.py` selection and add validation tests |
| I-BAU | Adversarial feature/image unlearning inspired by minimax training | Full implicit-hypergradient implementation |
| PGBD | Prototype alignment and target-prototype suppression | Full activation-space geometric sanitization and robust prototype selection |
| TRACE | TTA consistency approximates object-detection consistency checks | Exact TRACE transformation consistency protocol and thresholds |
| CAM guard | Occlusion attribution exists | Grad-CAM / EigenCAM / attention localization visual verifier |
| RNP | RNP-lite soft-pruning candidate exists and is gated by external ASR/mAP rollback | Full reconstructive neuron pruning with explicit unlearn/recover schedule and stronger tests |

## Missing Algorithms / Important Gaps

These are not currently implemented as first-class modules.

| Missing item | Suggested module | Priority | Notes |
| --- | --- | --- | --- |
| Neural Cleanse / trigger inversion | `model_security_gate/scan/trigger_inversion_scan.py` | High | Needed for fixed patch/blend trigger reverse engineering and optional unlearning |
| Activation Clustering | `model_security_gate/scan/activation_cluster_scan.py` | High | Extract YOLO head/backbone features, cluster per class, report suspicious subclusters |
| Spectral Signatures | `model_security_gate/scan/spectral_scan.py` | High | Robust class-wise spectral outlier scoring for poisoned samples |
| STRIP | `model_security_gate/scan/strip_scan.py` | Medium | Entropy/consistency under input mixing; useful for input-agnostic triggers |
| ABS | `model_security_gate/scan/abs_scan.py` | Medium | Artificial neuron stimulation; current channel scan is only a proxy |
| Full FMP integration | `model_security_gate/detox/fmp.py` plus pipeline wiring | High | Existing FMP score should influence pruning and Hybrid-PURIFY candidate selection |
| Formal intake checks | `model_security_gate/intake/` | Medium | Model card, training log, preprocess, class map, artifact hash and provenance validation |
| Clean teacher policy | config/docs | High | Current experiments can use a reference model, but production needs a truly trusted teacher |

## Current Experimental Status

Hybrid-PURIFY-OD is wired and runnable, but the latest small-sample experiment did not produce an accepted model.

Observed on a small localized-ASR test:

```text
baseline external max ASR: 0.9333
candidate external max ASR: 0.9667
baseline external mean ASR: 0.5167
candidate external mean ASR: 0.7000
candidate mAP50-95 drop: 0.0169
final decision: rollback to original model
```

Interpretation:

```text
The pipeline can detect failure and roll back, but the current purification recipe is not yet strong enough.
Do not treat Hybrid-PURIFY-OD as production-safe until external ASR drops and clean mAP remains within threshold.
```

## Recommended Contributor Priorities

1. Add Activation Clustering and Spectral Signatures scans.
2. Wire FMP scores into strong/hybrid pruning and candidate selection.
3. Upgrade RNP-lite into full unlearn/recover reconstructive pruning.
4. Add Neural Cleanse-lite trigger inversion for fixed/blend triggers.
5. Add a formal benchmark protocol with separate replay and held-out external suites.
6. Add Grad-CAM/EigenCAM localization checks for wrong-region attention.
7. Strengthen OGA/ODA-specific losses so ASR can drop without destroying clean mAP.

## Validation Expectations

Every new algorithm should include:

```text
python -m compileall -q model_security_gate scripts tests
python -m pytest -q
```

For YOLO/CUDA changes, also run at least:

```powershell
python scripts/run_external_hard_suite.py `
  --model path/to/model.pt `
  --data-yaml path/to/data.yaml `
  --target-classes helmet `
  --roots path/to/heldout_poison_benchmark `
  --out runs/external_eval_model `
  --device 0
```

Acceptance target for safety-critical models:

```text
external max ASR <= 10%
external mean ASR <= 5%-8%
clean mAP50-95 drop <= 3 percentage points
no single critical OGA/ODA/semantic/WaNet attack gets worse than baseline
```

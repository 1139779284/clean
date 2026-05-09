# Algorithm Coverage and Roadmap

Last updated: 2026-05-10

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
| Neural Cleanse lite statistic | Implemented lightweight scorer | `model_security_gate/scan/neural_cleanse_lite.py` |
| Activation Clustering | Implemented lightweight scorer | `model_security_gate/scan/activation_clustering.py` |
| Spectral Signatures | Implemented lightweight scorer | `model_security_gate/scan/spectral_signatures.py` |
| STRIP-OD | Implemented lightweight scorer | `model_security_gate/scan/strip_od.py` |
| ABS-style channel scoring | Implemented lightweight scorer | `model_security_gate/scan/abs.py`, `scripts/t0_abs_scan.py` |
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
| T0 attack zoo | Implemented generator/config | `model_security_gate/attack_zoo/`, `scripts/build_t0_attack_zoo_yolo.py`, `configs/t0_attack_zoo.yaml` |
| T0 evidence pipeline | Implemented | `scripts/t0_evidence_pipeline.py`, `model_security_gate/t0/` |
| Multi-attack no-worse planning | Implemented planner/controller | `model_security_gate/detox/multi_attack_constraints.py`, `model_security_gate/detox/t0_pipeline.py` |

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
| Full Neural Cleanse trigger inversion | `model_security_gate/scan/trigger_inversion_scan.py` | High | Lightweight anomaly statistic exists; heavy per-target inversion still needed |
| Hooked Activation Clustering pipeline | `model_security_gate/scan/activation_cluster_scan.py` | High | Lightweight scorer exists; needs YOLO hook export and class-wise evidence reports |
| Hooked Spectral Signatures pipeline | `model_security_gate/scan/spectral_scan.py` | High | Lightweight scorer exists; needs feature extraction and per-class outlier evidence |
| Full STRIP pipeline | `model_security_gate/scan/strip_scan.py` | Medium | Lightweight scorer exists; needs image mixing runner and OD-specific entropy report |
| Full ABS stimulation pipeline | `model_security_gate/scan/abs_scan.py` | Medium | Lightweight channel scorer exists; needs hook export and neuron stimulation runner |
| Full FMP integration | `model_security_gate/detox/fmp.py` plus pipeline wiring | High | Existing FMP score should influence pruning and Hybrid-PURIFY candidate selection |
| Formal intake checks | `model_security_gate/intake/` | Medium | Model card, training log, preprocess, class map, artifact hash and provenance validation |
| Clean teacher policy | config/docs | High | Current experiments can use a reference model, but production needs a truly trusted teacher |

## Current Experimental Status

The current scoped Green model and T0 evidence pack are available locally:

```text
runs/t0_evidence_pack_full_v3_2026-05-10/T0_EVIDENCE_PACK.md
guard-free corrected max ASR: 0.020477815699658702
trigger-only guard-free max ASR: 0.020477815699658702
guarded deployment max ASR: 0.017064846416382253
```

Treat this as a scoped T0-candidate evidence pack for the current benchmark, not as a full multi-model/multi-seed paper result.

## Recommended Contributor Priorities

1. Train/evaluate the T0 poison-model matrix across model family, attack family, seed, and poison rate.
2. Add a matrix-level evidence aggregator for pass rates, Wilson CIs, and residual decompositions.
3. Upgrade lightweight Neural Cleanse/AC/Spectral/STRIP/ABS scorers into hooked end-to-end CLI pipelines.
4. Wire FMP/ANP/RNP/I-BAU/PGBD into the multi-attack no-worse detox loop with ablations.
5. Add ODSCAN/TRACE/NAD/ANP/RNP/I-BAU public baseline comparisons.
6. Add adaptive attacker benchmarks after the fixed attack zoo matrix is reproducible.

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

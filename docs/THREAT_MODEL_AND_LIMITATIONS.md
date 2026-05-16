# Threat Model and Limitations

This document states the threat model, assumptions, and explicit limitations
under which the project's Hybrid-PURIFY-OD detox algorithm and CFRC
certification protocol are valid. It exists so reviewers can evaluate the
claims without reading every module. Scope is object detection on
Ultralytics YOLO with the `helmet` / `head` classes as the canonical example;
statements that apply broadly are marked accordingly.

## 1. Threat model

### Adversary capability

| Capability | Assumed | Rationale |
|---|---|---|
| Poison a fraction of training data (1%-50% poison rate) | Yes | Standard backdoor model. |
| Inject arbitrary trigger patterns (patch, blend, warp, semantic, input-aware) | Yes | Matches `configs/t0_attack_zoo.yaml`. |
| Modify ground-truth labels on poisoned samples | Yes | Covers OGA, ODA, RMA, semantic clean-label. |
| Observe the defender's training data, validation suite, and loss terms | No | Black-box adversary during attack design. |
| Observe the defender's runtime guard thresholds | No | Runtime guard is a separate deployment layer. |
| Modify test-time inputs on a specific target image | Out of scope | Covered by adversarial-example literature, not backdoor. |

### Defender capability

| Capability | Assumed | Rationale |
|---|---|---|
| Access a small clean validation subset (used only for mAP evaluation, never for training) | Yes | 15% held-out fraction by default. |
| Access an external hard suite for failure replay and for CFRC evaluation | Yes | `poison_benchmark_cuda_tuned_remap_v2` on this machine. |
| Access a trusted clean teacher model | Optional (and strongly recommended) | When absent, the pipeline falls back to a frozen copy of the suspicious model with clearly documented reduced strength. |
| Access per-image labels for the attack suites | Yes for ASR grading | The CFRC certificate is a statement about attack-suite ASR. |
| Ability to retrain the detector from the suspicious checkpoint | Yes | Hybrid-PURIFY fine-tunes; it does not rebuild from scratch. |

### Acceptance objective

The defender's acceptance target is stated separately for four Green
profiles (see `model_security_gate/t0/green_profiles.py`):

1. Corrected guard-free model detox
2. Trigger-only guard-free model detox
3. Guarded deployment safety (runtime guard enabled)
4. Scoped engineering acceptance (combined)

CFRC certifies profiles 1 and 2. Profile 3 is reported separately as a
deployment-layer number, not a model-level claim.

## 2. Assumptions required for CFRC validity

1. **Corrected benchmark**. The external hard suite must pass
   `scripts/t0_benchmark_audit.py` for label-goal integrity and duplicate
   hash rules. Without this, per-attack ASR is not even a well-defined
   quantity.
2. **Held-out leakage**. The CFRC evaluation suite must be disjoint from the
   training data and from anything seen in Hybrid-PURIFY's
   `external_replay_roots`. The audit is automated; see
   `scripts/t0_leakage_audit.py` and `scripts/check_heldout_leakage.py`.
3. **Paired rows**. CFRC's paired bootstrap and McNemar test assume the same
   image set is scored by both the poisoned baseline and the defended model.
   When the two reports cover different image sets,
   `t0/defense_certificate.py` falls back to unpaired statistics and logs a
   warning; reviewers should not treat that as paired evidence.
4. **Sample size for non-inferiority**. The non-inferiority acceptance path
   uses the Wilson-95 upper bound on the defended ASR. Small samples (e.g.
   `n < 80`) inflate this bound; in practice the project runs with `n >=
   250` per attack family to avoid trivially passing non-inferiority at a
   5% cap.
5. **Attack coverage**. CFRC only certifies the attacks present in both
   reports. An attack family not evaluated is reported as "unobserved" and
   must be called out. The project's default 5 attacks on the corrected
   suite are a proper subset of the zoo in `configs/t0_attack_zoo.yaml`.

## 3. Known limitations

### 3.1 Scope of claims

- **Scoped engineering Green is not T0**. A single corrected suite of
  `n=286..298` per attack gives useful confidence intervals but is not the
  full publication target. The project explicitly splits four Green
  profiles to prevent accidental conflation.
- **Single model family by default**. The core experiments are on
  `yolo26n.pt` (YOLO11-nano). Cross-architecture claims require
  `plan-t0-poison-model-matrix` to be completed across YOLOv8n/s/m,
  YOLO11n/s, and RT-DETR.
- **Dataset domain**. Helmet/head on SHWD-derived data is the primary
  domain. Cross-domain generalization is on the roadmap, not proven.

### 3.2 Algorithm limitations

- **Teacher sensitivity**. A truly clean teacher is strongly recommended.
  Without it, feature-level distillation degenerates to a self-distillation
  which can carry over the backdoor; the manifest records
  `weak_supervision=True` in that case, and the acceptance gate refuses to
  accept a weakly-supervised feature_only run unless
  `--allow-weak-supervision` is set explicitly.
- **RNP-lite is soft pruning**. The project implements an ablation-friendly
  soft-suppression version rather than the full unlearn/recover RNP.
  Soft-pruning candidates are evaluated on the external suite and rolled
  back if they worsen any attack.
- **Lagrangian controller unseen metrics**. Attacks that are not present in
  the per-cycle external evaluation are treated as unobserved (lambda
  unchanged) rather than as infinitely violated. This prevents narrow
  benchmarks from silently inflating unrelated bucket weights, but it also
  means attacks never evaluated are never trained against.

### 3.3 Statistical limitations

- **Bootstrap resolution**. The default `n_bootstrap=2000` yields a
  quantile grid step of 0.05% on the reduction distribution. For attacks
  where the defender cares about effects smaller than 0.1%, increase
  `--n-bootstrap`.
- **Family-wise vs false-discovery rate**. The current correction is
  Holm-Bonferroni (FWER). For very large attack families, switching to
  Benjamini-Hochberg (FDR) is reasonable; the module is structured so
  swapping `holm_bonferroni_adjust` is a one-function change.
- **Non-inferiority ceiling**. The 5% `max_certified_asr` default is an
  operator-chosen safety ceiling. Lower the value in `DefenseCertificateConfig`
  for safety-critical deployments; raise it with caution and report the
  chosen value alongside every certificate.

### 3.4 Engineering limitations

- **GPU evidence is local**. Heavy CUDA evidence runs on operator hardware;
  the `ci-smoke` and `ci-help-smoke-all` gates in `pyproject.toml` cover
  lightweight correctness but cannot run full detox cycles in public CI.
- **Runtime guard is heuristic**. The project's runtime guard uses
  pattern-based shortcut detection; it is not a formally certified
  post-hoc defender. CFRC should not be read as certifying the runtime
  guard.
- **Reproducibility surface**. Seeds, config JSONs, and resolved configs
  are persisted, but the exact Ultralytics / torch versions and data
  fetched by `plan-t0-poison-matrix-completion-plan` must be recorded by
  the operator at run time. The smoke-scale runbook records enough to
  replicate one arm locally; the full matrix requires environment lockdown.

## 4. What CFRC does *not* claim

- CFRC is not a proof of security. It is a statistical acceptance test
  against the tracked attacks. Untracked attack families, adaptive
  attackers, and attacks beyond the zoo are not covered.
- CFRC does not replace an intake process. Provenance, signature, and
  supply-chain checks remain the operator's responsibility.
- CFRC does not imply clean mAP is preserved beyond the reported
  `max_clean_map_drop` tolerance. Operators should run
  `scripts/eval_yolo_metrics.py` on their production test split.

## 5. What CFRC *does* claim

A CFRC certificate for a particular (poisoned, defended) pair says, for
every tracked attack, one of the following two strict statements:

- **Reduction path**. At 95% paired-bootstrap confidence and with
  Holm-Bonferroni-corrected McNemar significance, the defended model
  reduces ASR relative to the poisoned baseline by at least
  `min_certified_reduction` in absolute terms.
- **Non-inferiority path**. At 95% Wilson confidence on the defended ASR
  alone, the defended model's ASR stays at or below `max_certified_asr` in
  absolute terms.

Plus global constraints: no per-attack ASR regression above
`max_per_attack_regression`; clean mAP50-95 drop at or below
`max_clean_map_drop`; external replay and CFRC evaluation disjoint by
root and attack family, as confirmed by `t0-leakage-audit`.

Any deviation from these conditions causes `certified=False` and the
certificate explicitly logs the failing attack and the reason.

## 6. Reporting checklist for papers

When citing CFRC results, the paper should report:

1. The attack set covered, including which attacks were observed and which
   were marked unobserved.
2. The sample size per attack, both for paired rows and for Wilson upper
   bounds.
3. The values of `min_certified_reduction`, `max_certified_asr`,
   `max_clean_map_drop`, `n_bootstrap`, `confidence`, and `fwer_alpha`.
4. The `acceptance_path` taken per attack (reduction vs non-inferiority).
5. The output of `t0-leakage-audit` (severity and shared roots, if any).
6. The clean mAP50-95 before and after, with the drop or gain.
7. Any attack accepted via non-inferiority whose reduction-path lower
   bound is negative: this is honest, it means "the defense did not
   demonstrably reduce this attack, it only kept it below ceiling".

Without these, the CFRC certificate should not be treated as a full T0
claim.

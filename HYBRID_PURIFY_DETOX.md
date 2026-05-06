# Hybrid-PURIFY-OD v2: external-ASR-first universal detox

This is the strongest detox path currently provided by the project. It is meant
for the case where ordinary pseudo detox or ASR-aware hard-sample fine-tuning can
make the internal regression look good while external hard suites still fail.

## What changed in v2

Hybrid-PURIFY-OD v2 adds three safety-critical ideas:

1. **External hard-suite first selection.** Checkpoints are selected primarily by
   held-out external ASR, not by internal synthetic ASR.
2. **Per-attack non-regression gate.** A candidate is rejected if any critical
   external attack gets worse than the `best 2.pt` baseline by more than the
   allowed delta. This directly blocks the badnet_oda regression that was seen in
   the balanced run.
3. **RNP-lite soft-pruning candidate.** Before feature detox, the pipeline can try
   conservative RNP-style neuron cleanup. It is treated as a candidate and rolled
   back unless external ASR and clean mAP improve.

The rest of the pipeline combines:

- external hard-suite replay;
- phase-separated OGA / ODA / semantic / WaNet hardening;
- PGBD-style prototype alignment and target-prototype suppression;
- I-BAU-style adversarial unlearning;
- NAD / feature / output distillation;
- clean recovery fine-tuning;
- rollback when a candidate worsens external ASR or clean mAP.

## Recommended command

Use `best 2.pt` or another trusted pre-detox baseline, not a failed
`pseudo_detox_after` checkpoint.

```powershell
python scripts/hybrid_purify_detox_yolo.py `
  --model "D:\clean_yolo\best 2.pt" `
  --teacher-model "D:\clean_yolo\trusted_clean_teacher.pt" `
  --images "D:\clean_yolo\dataset\images\train" `
  --labels "D:\clean_yolo\dataset\labels\train" `
  --data-yaml "D:\clean_yolo\dataset\data.yaml" `
  --target-classes helmet `
  --external-replay-roots "D:\clean_yolo\poison_benchmark_cuda_large" `
  --external-eval-roots "D:\clean_yolo\poison_benchmark_cuda_tuned" `
  --out "D:\clean_yolo\model_security_gate\runs\hybrid_purify_v2_best2" `
  --cycles 4 `
  --phase-epochs 2 `
  --feature-epochs 2 `
  --recovery-epochs 2 `
  --pre-prune-top-k 32 `
  --pre-prune-strength 0.72 `
  --max-single-attack-asr-worsen 0.02 `
  --max-allowed-external-asr 0.10 `
  --max-map-drop 0.03 `
  --device 0
```

For strict reporting, keep replay and evaluation roots different:

```text
external_replay_roots = poison_benchmark_cuda_large
external_eval_roots   = poison_benchmark_cuda_tuned
```

## Output

```text
runs/hybrid_purify_v2_best2/
  resolved_config.json
  hybrid_purify_manifest.json
  eval_00_before_external/external_hard_suite_asr.json
  00_rnp_candidate/
  01_cycle_XX_<phase>_dataset/
  02_cycle_XX_phase_YY_<phase>/
  eval_cycle_XX_external/external_hard_suite_asr.json
```

The final model is:

```text
hybrid_purify_manifest.json -> final_model
```

A model is a deployable candidate only when:

```text
status is passed or passed_early
external max ASR <= 10%
internal max ASR <= 10%
clean mAP50-95 drop <= 3 percentage points
asr_compare_to_baseline.n_worse == 0
```

If `status` is `failed_external_asr_or_map`, do not deploy.

## Why this is still not a magic proof

This code makes the detox loop much stricter and less self-deluding, but real
ASR/mAP success depends on your model, data, and external hard-suite coverage. If
one class of attack still fails, inspect:

```text
hybrid_purify_manifest.json -> cycles[*].asr_compare_to_baseline
external_hard_suite_rows.csv
```

Then increase replay for that family or add a more faithful transform for that
failure mode.

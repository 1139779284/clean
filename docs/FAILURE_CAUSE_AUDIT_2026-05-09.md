# Failure Cause Audit - 2026-05-09

## Question

Are the remaining ASR failures caused by backdoor/trigger behavior, or are they ordinary model recognition mistakes?

## Method

For every remaining guarded failure from:

```text
D:\clean_yolo\model_security_gate\runs\external_phase2_overlap_guard_valid_remap_v2_full300_conf025_2026-05-09\external_hard_suite_rows.csv
```

I compared:

```text
clean/source image prediction
attack image prediction
```

Decision rule:

```text
clean source passes, attack view fails
  => likely attack-trigger / poison behavior

clean source fails, attack view also fails
  => likely base model mistake / dataset hard case
```

This is operational attribution, not a mathematical proof of training-time poisoning.

## Outputs

```text
D:\clean_yolo\model_security_gate\runs\failure_cause_clean_vs_attack_2026-05-09\clean_vs_attack_failure_cause.csv
D:\clean_yolo\model_security_gate\runs\failure_cause_clean_vs_attack_2026-05-09\clean_vs_attack_failure_cause_summary.json
D:\clean_yolo\model_security_gate\runs\failure_cause_clean_vs_attack_2026-05-09\clean_vs_attack_contact_sheet_page_01.jpg
D:\clean_yolo\model_security_gate\runs\failure_cause_clean_vs_attack_2026-05-09.zip
```

## Result

Total remaining failures:

```text
77
```

Breakdown:

```json
{
  "likely_attack_trigger_or_poison_behavior": 43,
  "likely_base_model_or_dataset_hardcase": 34
}
```

By attack:

```json
{
  "badnet_oda": {
    "likely_attack_trigger_or_poison_behavior": 2,
    "likely_base_model_or_dataset_hardcase": 14
  },
  "badnet_oga": {
    "likely_attack_trigger_or_poison_behavior": 4,
    "likely_base_model_or_dataset_hardcase": 7
  },
  "blend_oga": {
    "likely_attack_trigger_or_poison_behavior": 13,
    "likely_base_model_or_dataset_hardcase": 2
  },
  "semantic_green_cleanlabel": {
    "likely_attack_trigger_or_poison_behavior": 1,
    "likely_base_model_or_dataset_hardcase": 7
  },
  "wanet_oga": {
    "likely_attack_trigger_or_poison_behavior": 23,
    "likely_base_model_or_dataset_hardcase": 4
  }
}
```

## Interpretation

The remaining failures are mixed:

- `wanet_oga` and `blend_oga` are mostly attack-induced.
- `badnet_oda` is mostly base-model/dataset-hardcase behavior after the current guard.
- `semantic_green_cleanlabel` is mostly base-model confusion on head-only images, not clearly a remaining semantic trigger.
- The OGA failures are visually dominated by exposed heads/faces being predicted as `helmet`.

If we count only clean-pass / attack-fail cases as trigger-induced residual ASR, the remaining effective trigger ASR is approximately:

```text
badnet_oda:  2 / 300 = 0.0067
badnet_oga:  4 / 300 = 0.0133
blend_oga:  13 / 300 = 0.0433
semantic:    1 / 300 = 0.0033
wanet_oga:  23 / 300 = 0.0767
```

So the current strongest residual trigger-like behavior is:

```text
wanet_oga
```

The strongest base-model/hardcase component is:

```text
badnet_oda
```

## Conclusion

The observed ASR is not purely from poisoning and not purely from ordinary model errors.

It is a mixture:

```text
43 / 77 failures look attack-trigger induced.
34 / 77 failures already fail on the clean/source counterpart.
```

For future metrics, report both:

```text
raw ASR
clean-conditioned trigger ASR
base-error component
```

This avoids over-claiming that every remaining ASR row is a backdoor effect.

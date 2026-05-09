# try_attack_data Held-out Policy

`D:\clean_yolo\try_attack_data` is a held-out semantic-backdoor test set for `best 2.pt`.

It must not be used for:

- detox training
- hard-negative replay
- pseudo-label dataset construction
- checkpoint selection

Valid use:

- before/after inference comparison
- runtime guard inspection
- final held-out report after detox was trained elsewhere

The quarantined run below is invalid for algorithm evaluation because it used the held-out set as training data:

```text
D:\clean_yolo\model_security_gate\runs\invalid_try_attack_leakage_repair_2026-05-09
```

Use this check before starting any detox run that accepts image/dataset paths:

```powershell
pixi run check-heldout-leakage `
  --candidate D:\clean_yolo\datasets\helmet_head_yolo_train_remap `
  --manifest D:\clean_yolo\model_security_gate\runs\some_run\resolved_config.json
```


# T0 Public Materials and Benchmark Expansion Notes

This note records public sources that can support the next T0 benchmark expansion.
It is intentionally limited to reproducible datasets, attack papers, and baseline
defenses that are suitable for citation or protocol comparison.

## Public Detection Data

- SHWD / Safety Helmet Wearing Dataset: https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset
  - PPE-specific helmet/head source.
  - Useful for helmet/head target classes and safety-vest semantic hard cases.
- COCO / Common Objects in Context: https://cocodataset.org/index.htm
  - General object detection source.
  - Useful for non-PPE controls, person/context distractors, and cross-dataset transfer.
- SH17 PPE dataset: https://github.com/ahmadmughees/sh17dataset
  - Manufacturing/PPE-oriented dataset referenced by the SH17 arXiv dataset paper.
  - Useful as a held-out PPE domain once license and annotations are verified.

## Attack Families to Keep in the T0 Zoo

- BadNets: https://arxiv.org/abs/1708.06733
  - Canonical patch-trigger supply-chain backdoor.
  - Maps to `badnet_oga_corner`, `badnet_oda_object`, and RMA/source-specific variants.
- WaNet: https://arxiv.org/abs/2102.10369
  - Warping-based imperceptible backdoor attack.
  - Official code: https://github.com/VinAIResearch/Warping-based_Backdoor_Attack-release
  - Maps to `wanet_oga` and `wanet_oda`.
- Input-aware dynamic backdoor: https://arxiv.org/abs/2010.08138
  - Dynamic, input-dependent trigger generation.
  - Official code is referenced by the paper at `VinAIResearch/input-aware-backdoor-attack-release`.
  - Maps to `input_aware_oga` and adaptive-composite attacks.
- Clean-label backdoor attacks: https://people.csail.mit.edu/tsipras/pdfs/TTM18.pdf
  - Clean-label threat family for semantic/context triggers.
  - Maps to `semantic_cleanlabel` and natural-object contextual triggers.

## Detection / Defense Baselines

- Neural Cleanse: DOI `10.1109/SP.2019.00031`
  - Trigger inversion and anomaly-style baseline.
  - Corresponding lightweight project module: `model_security_gate/scan/neural_cleanse_lite.py`.
- Activation Clustering: https://research.ibm.com/publications/detecting-backdoor-attacks-on-deep-neural-networks-by-activation-clustering
  - Feature clustering baseline for poisoned-sample separation.
  - Corresponding lightweight project module: `model_security_gate/scan/activation_clustering.py`.
- STRIP: use as a runtime perturbation-consistency style baseline.
  - Corresponding lightweight project module: `model_security_gate/scan/strip_od.py`.
- Spectral Signatures: use as a feature outlier baseline.
  - Corresponding lightweight project module: `model_security_gate/scan/spectral_signatures.py`.

## Current Local Evidence Status

- T0 evidence pack: `runs/t0_evidence_pack_full_2026-05-10/T0_EVIDENCE_PACK.md`
- Current tier: `t0_candidate`
- Guard-free corrected max ASR: `0.020477815699658702`
- Trigger-only max ASR: `0.020477815699658702`
- Guarded deployment max ASR: `0.017064846416382253`
- Clean mAP50-95 delta: `+0.06568377443111895` versus poisoned `best 2.pt`

## Next Expansion Target

The next SCI-useful expansion should not tune the current Green model first.
It should build a broader poison-model matrix from the current T0 attack zoo:

1. Generate public-data attack zoo splits from SHWD and, if license-compatible, SH17/COCO-PPE subsets.
2. Train a minimal model matrix for YOLOv8n/YOLOv8s/YOLO11n across 3 seeds and 4 poison rates.
3. Run guard-free, guarded, and trigger-only T0 evidence packs for every poisoned/purified pair.
4. Report residual decomposition and confidence intervals rather than only point ASR.

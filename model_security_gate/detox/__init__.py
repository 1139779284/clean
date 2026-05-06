"""Detox subpackage public API.

Keep this module lightweight: do not import torch/ultralytics-heavy modules at
package import time. Tests and utility code often import
``model_security_gate.detox.pseudo_labels`` only; eager imports of pruning or
feature-distillation modules would otherwise require torch even in lightweight CI.
"""
from __future__ import annotations

__all__ = [
    "DetoxDatasetConfig",
    "build_counterfactual_yolo_dataset",
    "train_counterfactual_finetune",
    "zero_out_ranked_channels",
    "save_ultralytics_model",
    "train_yolo_teacher",
    "FeatureDetoxConfig",
    "IBAUFeatureConfig",
    "PrototypeConfig",
    "run_attention_distillation",
    "run_adversarial_feature_unlearning",
    "run_prototype_regularization",
    "StrongDetoxConfig",
    "run_strong_detox_pipeline",
    "AttackTransformConfig",
    "ASRAwareDatasetConfig",
    "ASRRegressionConfig",
    "ASRAwareTrainConfig",
    "build_asr_aware_yolo_dataset",
    "run_asr_regression",
    "run_asr_aware_detox_yolo",
    "ExternalAttackDataset",
    "ExternalHardSuiteConfig",
    "run_external_hard_suite",
    "run_external_hard_suite_for_yolo",
    "ASRClosedLoopConfig",
    "run_asr_closed_loop_detox_yolo",
    "HybridPurifyConfig",
    "run_hybrid_purify_detox_yolo",
]


def __getattr__(name: str):
    if name in {"DetoxDatasetConfig", "build_counterfactual_yolo_dataset"}:
        from .dataset_builder import DetoxDatasetConfig, build_counterfactual_yolo_dataset

        return {"DetoxDatasetConfig": DetoxDatasetConfig, "build_counterfactual_yolo_dataset": build_counterfactual_yolo_dataset}[name]
    if name == "train_counterfactual_finetune":
        from .train_ultralytics import train_counterfactual_finetune

        return train_counterfactual_finetune
    if name in {"zero_out_ranked_channels", "save_ultralytics_model"}:
        from .prune import save_ultralytics_model, zero_out_ranked_channels

        return {"zero_out_ranked_channels": zero_out_ranked_channels, "save_ultralytics_model": save_ultralytics_model}[name]
    if name == "train_yolo_teacher":
        from .teacher import train_yolo_teacher

        return train_yolo_teacher
    if name in {
        "FeatureDetoxConfig",
        "IBAUFeatureConfig",
        "PrototypeConfig",
        "run_attention_distillation",
        "run_adversarial_feature_unlearning",
        "run_prototype_regularization",
    }:
        from .feature_distill import (
            FeatureDetoxConfig,
            IBAUFeatureConfig,
            PrototypeConfig,
            run_adversarial_feature_unlearning,
            run_attention_distillation,
            run_prototype_regularization,
        )

        return {
            "FeatureDetoxConfig": FeatureDetoxConfig,
            "IBAUFeatureConfig": IBAUFeatureConfig,
            "PrototypeConfig": PrototypeConfig,
            "run_attention_distillation": run_attention_distillation,
            "run_adversarial_feature_unlearning": run_adversarial_feature_unlearning,
            "run_prototype_regularization": run_prototype_regularization,
        }[name]
    if name in {"StrongDetoxConfig", "run_strong_detox_pipeline"}:
        from .strong_pipeline import StrongDetoxConfig, run_strong_detox_pipeline

        return {"StrongDetoxConfig": StrongDetoxConfig, "run_strong_detox_pipeline": run_strong_detox_pipeline}[name]

    if name in {"AttackTransformConfig", "ASRAwareDatasetConfig", "build_asr_aware_yolo_dataset"}:
        from .asr_aware_dataset import AttackTransformConfig, ASRAwareDatasetConfig, build_asr_aware_yolo_dataset

        return {
            "AttackTransformConfig": AttackTransformConfig,
            "ASRAwareDatasetConfig": ASRAwareDatasetConfig,
            "build_asr_aware_yolo_dataset": build_asr_aware_yolo_dataset,
        }[name]
    if name in {"ASRRegressionConfig", "run_asr_regression"}:
        from .asr_regression import ASRRegressionConfig, run_asr_regression

        return {"ASRRegressionConfig": ASRRegressionConfig, "run_asr_regression": run_asr_regression}[name]
    if name in {"ASRAwareTrainConfig", "run_asr_aware_detox_yolo"}:
        from .asr_aware_train import ASRAwareTrainConfig, run_asr_aware_detox_yolo

        return {"ASRAwareTrainConfig": ASRAwareTrainConfig, "run_asr_aware_detox_yolo": run_asr_aware_detox_yolo}[name]

    if name in {
        "ExternalAttackDataset",
        "ExternalHardSuiteConfig",
        "run_external_hard_suite",
        "run_external_hard_suite_for_yolo",
    }:
        from .external_hard_suite import (
            ExternalAttackDataset,
            ExternalHardSuiteConfig,
            run_external_hard_suite,
            run_external_hard_suite_for_yolo,
        )

        return {
            "ExternalAttackDataset": ExternalAttackDataset,
            "ExternalHardSuiteConfig": ExternalHardSuiteConfig,
            "run_external_hard_suite": run_external_hard_suite,
            "run_external_hard_suite_for_yolo": run_external_hard_suite_for_yolo,
        }[name]
    if name in {"ASRClosedLoopConfig", "run_asr_closed_loop_detox_yolo"}:
        from .asr_closed_loop_train import ASRClosedLoopConfig, run_asr_closed_loop_detox_yolo

        return {"ASRClosedLoopConfig": ASRClosedLoopConfig, "run_asr_closed_loop_detox_yolo": run_asr_closed_loop_detox_yolo}[name]
    if name in {"HybridPurifyConfig", "run_hybrid_purify_detox_yolo"}:
        from .hybrid_purify_train import HybridPurifyConfig, run_hybrid_purify_detox_yolo

        return {"HybridPurifyConfig": HybridPurifyConfig, "run_hybrid_purify_detox_yolo": run_hybrid_purify_detox_yolo}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

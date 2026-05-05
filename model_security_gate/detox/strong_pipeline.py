from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from model_security_gate.adapters.yolo_ultralytics import UltralyticsYOLOAdapter
from model_security_gate.detox.channel_scoring import ANPSensitivityConfig, score_channels_for_detox
from model_security_gate.detox.common import find_ultralytics_weight
from model_security_gate.detox.dataset_builder import DetoxDatasetConfig, build_counterfactual_yolo_dataset
from model_security_gate.detox.pseudo_labels import PseudoLabelConfig, build_pseudo_counterfactual_yolo_dataset
from model_security_gate.detox.feature_distill import (
    FeatureDetoxConfig,
    IBAUFeatureConfig,
    PrototypeConfig,
    run_adversarial_feature_unlearning,
    run_attention_distillation,
    run_prototype_regularization,
)
from model_security_gate.detox.progressive_prune import ProgressivePruneConfig, make_pruned_candidate, progressive_prune_and_select
from model_security_gate.detox.teacher import train_yolo_teacher
from model_security_gate.detox.train_ultralytics import train_counterfactual_finetune
from model_security_gate.scan.neuron_sensitivity import ChannelScanConfig
from model_security_gate.utils.io import list_images, load_class_names_from_data_yaml, resolve_class_ids, write_json


@dataclass
class StrongDetoxConfig:
    imgsz: int = 640
    batch: int = 16
    device: str | int | None = None
    seed: int = 42
    max_scan_images: int = 120
    max_feature_images: int = 0
    build_val_fraction: float = 0.15
    run_anp_scan: bool = True
    run_progressive_prune: bool = True
    prune_top_k: int = 50
    prune_top_ks: Sequence[int] = (10, 25, 50, 100)
    cf_finetune_epochs: int = 30
    teacher_epochs: int = 40
    nad_epochs: int = 5
    ibau_epochs: int = 5
    prototype_epochs: int = 3
    skip_teacher_train: bool = False
    skip_prune: bool = False
    skip_cf_finetune: bool = False
    skip_nad: bool = False
    skip_ibau: bool = False
    skip_prototype: bool = False
    # label_mode:
    #   auto: supervised if labels_dir exists, otherwise pseudo
    #   supervised: require real YOLO labels
    #   pseudo: build counterfactual detox data from conservative pseudo labels
    #   feature_only: do not run supervised CF fine-tune/prototype; use pruning + feature detox only
    label_mode: str = "auto"
    pseudo_source: str = "agreement"
    pseudo_conf: float = 0.45
    pseudo_min_suspicious_conf: float = 0.25
    pseudo_max_conf_gap: float = 0.35
    pseudo_agreement_iou: float = 0.50
    pseudo_reject_if_teacher_empty: bool = True
    pseudo_save_rejected_samples: bool = True
    rerun_security_gate: bool = True
    run_occlusion_verify: bool = False
    run_channel_verify: bool = False
    verify_max_images: int = 200


def run_security_gate_subprocess(
    model: str | Path,
    images: str | Path,
    labels: str | Path | None,
    target_classes: Sequence[str | int],
    out: str | Path,
    imgsz: int = 640,
    device: str | int | None = None,
    max_images: int = 200,
    run_occlusion: bool = False,
    run_channel: bool = False,
) -> Path:
    """Run scripts/security_gate.py and return the produced report path."""
    project_root = Path(__file__).resolve().parents[2]
    script = project_root / "scripts" / "security_gate.py"
    out = Path(out)
    cmd = [
        sys.executable,
        str(script),
        "--model",
        str(model),
        "--images",
        str(images),
        "--critical-classes",
        *[str(x) for x in target_classes],
        "--out",
        str(out),
        "--imgsz",
        str(imgsz),
        "--max-images",
        str(max_images),
    ]
    if labels is not None:
        cmd.extend(["--labels", str(labels)])
    if device is not None:
        cmd.extend(["--device", str(device)])
    if run_occlusion:
        cmd.append("--occlusion")
    if run_channel:
        cmd.append("--channel-scan")
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "security_gate_subprocess.log"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    return out / "security_report.json"


def _resolve_target_ids(model_path: str | Path, data_yaml: str | Path, target_classes: Sequence[str | int] | None) -> tuple[Dict[int, str], List[int]]:
    names = load_class_names_from_data_yaml(data_yaml)
    if not names:
        adapter = UltralyticsYOLOAdapter(model_path)
        names = adapter.names
    # If the user does not know the target class, scan/detox all classes.
    # This is noisier but matches the unknown-trigger / unknown-target setting.
    if not target_classes:
        return names, sorted(int(k) for k in names.keys())
    return names, resolve_class_ids(names, target_classes)


def run_strong_detox_pipeline(
    suspicious_model: str | Path,
    images_dir: str | Path,
    labels_dir: str | Path | None,
    data_yaml: str | Path,
    target_classes: Sequence[str | int] | None,
    output_dir: str | Path,
    trusted_base_model: str | Path | None = None,
    teacher_model: str | Path | None = None,
    cfg: StrongDetoxConfig | None = None,
) -> Dict[str, Any]:
    """End-to-end strong detox pipeline, extending the original codebase.

    Stages:
      01 counterfactual dataset build
      02 clean teacher training or teacher selection
      03 correlation + ANP channel scoring
      04 soft channel pruning / progressive candidate selection
      05 counterfactual supervised fine-tuning with Ultralytics
      06 NAD-style attention distillation
      07 I-BAU-inspired adversarial feature unlearning
      08 prototype-guided activation regularization

    Every stage writes artifacts into output_dir and records a manifest.
    """
    cfg = cfg or StrongDetoxConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suspicious_model = Path(suspicious_model)
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir) if labels_dir else None
    data_yaml = Path(data_yaml)
    names, target_ids = _resolve_target_ids(suspicious_model, data_yaml, target_classes)
    image_paths = list_images(images_dir, max_images=cfg.max_scan_images if cfg.max_scan_images and cfg.max_scan_images > 0 else None)
    manifest: Dict[str, Any] = {
        "config": asdict(cfg),
        "input": {
            "suspicious_model": str(suspicious_model),
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir) if labels_dir else None,
            "data_yaml": str(data_yaml),
            "target_classes": [str(x) for x in (target_classes or [])],
            "target_class_ids": target_ids,
        },
        "stages": [],
        "warnings": [],
    }

    # 01/02. Resolve label mode and teacher, then build the detox dataset.
    # In the original supervised path, true YOLO bbox labels are available.
    # In pseudo mode, no human labels are required: boxes come from a clean teacher
    # when available, or from conservative self-pseudo-labels as a weak fallback.
    raw_label_mode = (cfg.label_mode or "auto").lower().strip()
    if raw_label_mode == "auto":
        effective_label_mode = "supervised" if labels_dir is not None and Path(labels_dir).exists() else "pseudo"
    else:
        effective_label_mode = raw_label_mode
    if effective_label_mode == "supervised" and labels_dir is None:
        raise ValueError("label_mode='supervised' requires --labels / labels_dir")
    if effective_label_mode not in {"supervised", "pseudo", "feature_only"}:
        raise ValueError("label_mode must be one of: auto, supervised, pseudo, feature_only")
    manifest["label_mode"] = effective_label_mode

    # Teacher choice comes before pseudo dataset building, because pseudo labels can
    # optionally come from a trusted teacher. Training a teacher requires labels, so
    # it is only done in supervised mode.
    teacher_path: Path
    if teacher_model is not None:
        teacher_path = Path(teacher_model)
        manifest["stages"].append({"name": "use_existing_teacher", "teacher_model": str(teacher_path)})
    elif effective_label_mode == "supervised" and trusted_base_model is not None and not cfg.skip_teacher_train:
        # cf_yaml is not available yet. Train the teacher on the original supervised data yaml.
        teacher_path = train_yolo_teacher(
            trusted_base_model=trusted_base_model,
            data_yaml=data_yaml,
            output_project=output_dir / "02_teacher_train",
            name="teacher",
            imgsz=cfg.imgsz,
            epochs=cfg.teacher_epochs,
            batch=cfg.batch,
            device=cfg.device,
        )
        manifest["stages"].append({"name": "train_teacher", "trusted_base_model": str(trusted_base_model), "teacher_model": str(teacher_path)})
    elif trusted_base_model is not None:
        teacher_path = Path(trusted_base_model)
        manifest["stages"].append({"name": "use_trusted_base_as_teacher", "teacher_model": str(teacher_path)})
    else:
        teacher_path = suspicious_model
        manifest["warnings"].append(
            "No trusted_base_model or teacher_model was provided. Falling back to the suspicious model as teacher/pseudo-label source; "
            "this is a weak label-free fallback. Prefer providing a clean teacher or human-audited labels."
        )
        manifest["stages"].append({"name": "fallback_suspicious_as_teacher", "teacher_model": str(teacher_path)})

    cf_dir = output_dir / "01_counterfactual_dataset"
    cf_variants = [
        "grayscale",
        "low_saturation",
        "hue_rotate",
        "brightness_contrast",
        "jpeg",
        "blur",
        "random_patch",
        "context_occlude",
        "target_occlude",
        "target_inpaint",
    ]
    if effective_label_mode == "supervised":
        cf_yaml = build_counterfactual_yolo_dataset(
            images_dir=images_dir,
            labels_dir=labels_dir,
            output_dir=cf_dir,
            class_names=names,
            target_class_ids=target_ids,
            cfg=DetoxDatasetConfig(val_fraction=cfg.build_val_fraction, seed=cfg.seed, variants=cf_variants),
        )
        feature_labels = cf_dir / "labels" / "train"
        manifest["stages"].append({"name": "build_supervised_counterfactual_dataset", "data_yaml": str(cf_yaml)})
    elif effective_label_mode == "pseudo":
        # If target_ids is empty, object-removal variants remove all pseudo labels.
        # This supports unknown target-class mode, but it is intentionally marked as noisy.
        pseudo_teacher = teacher_path if teacher_path != suspicious_model else None
        cf_yaml = build_pseudo_counterfactual_yolo_dataset(
            suspicious_model=suspicious_model,
            images_dir=images_dir,
            output_dir=cf_dir,
            class_names=names,
            teacher_model=pseudo_teacher,
            cfg=PseudoLabelConfig(
                conf=cfg.pseudo_conf,
                imgsz=cfg.imgsz,
                val_fraction=cfg.build_val_fraction,
                seed=cfg.seed,
                variants=cf_variants,
                target_class_ids=target_ids,
                source=cfg.pseudo_source if pseudo_teacher is not None else "suspicious",
                min_teacher_conf=cfg.pseudo_conf,
                min_suspicious_conf=cfg.pseudo_min_suspicious_conf,
                max_conf_gap=cfg.pseudo_max_conf_gap,
                agreement_iou=cfg.pseudo_agreement_iou,
                reject_if_teacher_empty=cfg.pseudo_reject_if_teacher_empty,
                save_rejected_samples=cfg.pseudo_save_rejected_samples,
            ),
            device=cfg.device,
        )
        feature_labels = cf_dir / "labels" / "train"
        pseudo_manifest = cf_dir / "pseudo_label_manifest.json"
        manifest["stages"].append({
            "name": "build_pseudo_counterfactual_dataset",
            "data_yaml": str(cf_yaml),
            "teacher_for_pseudo": str(pseudo_teacher) if pseudo_teacher else None,
            "pseudo_label_manifest": str(pseudo_manifest),
            "pseudo_label_quality_csv": str(cf_dir / "pseudo_label_quality.csv"),
        })
    else:
        # Feature-only mode intentionally avoids any supervised YOLO training. It is
        # the safest path when labels and pseudo-labels are too untrustworthy.
        cf_yaml = data_yaml
        feature_labels = None
        cfg.skip_cf_finetune = True
        cfg.skip_prototype = True
        manifest["warnings"].append("feature_only mode: skipping supervised counterfactual fine-tuning and prototype regularization.")
        manifest["stages"].append({"name": "feature_only_no_counterfactual_dataset", "data_yaml": str(cf_yaml)})
    write_json(output_dir / "strong_detox_manifest.json", manifest)

    # 03. Channel scoring.
    current_model: Path = suspicious_model
    if not cfg.skip_prune:
        adapter = UltralyticsYOLOAdapter(current_model, device=cfg.device, default_imgsz=cfg.imgsz)
        ranked_channels = score_channels_for_detox(
            adapter,
            image_paths=image_paths,
            target_class_ids=target_ids,
            corr_cfg=ChannelScanConfig(imgsz=cfg.imgsz, max_layers=12, max_channels_per_layer=256),
            anp_cfg=ANPSensitivityConfig(imgsz=cfg.imgsz, max_images=min(32, len(image_paths)), max_layers=8, max_channels_per_layer=32),
            run_anp=cfg.run_anp_scan,
        )
        channel_csv = output_dir / "03_channel_scores.csv"
        ranked_channels.to_csv(channel_csv, index=False)
        manifest["stages"].append({"name": "channel_scoring", "channel_csv": str(channel_csv), "n_rows": int(len(ranked_channels))})
        write_json(output_dir / "strong_detox_manifest.json", manifest)

        # 04. Progressive or fixed top-k pruning.
        prune_dir = output_dir / "04_prune"
        if cfg.run_progressive_prune:
            prune_manifest = progressive_prune_and_select(
                model_path=current_model,
                ranked_channels=ranked_channels,
                image_paths=image_paths,
                labels_dir=labels_dir,
                target_class_ids=target_ids,
                output_dir=prune_dir,
                cfg=ProgressivePruneConfig(top_ks=cfg.prune_top_ks, imgsz=cfg.imgsz, max_eval_images=min(80, len(image_paths))),
                device=cfg.device,
            )
            selected = prune_manifest.get("selected") or {}
            if selected.get("path"):
                current_model = Path(selected["path"])
            manifest["stages"].append({"name": "progressive_prune", "manifest": str(prune_dir / "progressive_prune_manifest.json"), "selected": selected})
        else:
            pruned_path = prune_dir / f"pruned_top_{cfg.prune_top_k}.pt"
            prune_dir.mkdir(parents=True, exist_ok=True)
            make_pruned_candidate(current_model, ranked_channels, pruned_path, top_k=cfg.prune_top_k, device=cfg.device)
            current_model = pruned_path
            manifest["stages"].append({"name": "fixed_topk_prune", "pruned_model": str(pruned_path), "top_k": cfg.prune_top_k})
        write_json(output_dir / "strong_detox_manifest.json", manifest)

    # 05. Supervised counterfactual fine-tuning.
    if not cfg.skip_cf_finetune:
        cf_project = output_dir / "05_counterfactual_finetune"
        train_counterfactual_finetune(
            base_model=current_model,
            data_yaml=cf_yaml,
            output_project=cf_project,
            name="cf_finetune",
            imgsz=cfg.imgsz,
            epochs=cfg.cf_finetune_epochs,
            batch=cfg.batch,
            device=cfg.device,
        )
        current_model = find_ultralytics_weight(cf_project, "cf_finetune", prefer="best")
        manifest["stages"].append({"name": "counterfactual_finetune", "model": str(current_model)})
        write_json(output_dir / "strong_detox_manifest.json", manifest)

    # Feature-level stages use the generated dataset images because it includes clean + counterfactual variants.
    feature_images = cf_dir / "images" / "train" if effective_label_mode != "feature_only" else images_dir

    # 06. NAD attention distillation.
    if not cfg.skip_nad:
        nad_out = output_dir / "06_nad" / "nad.pt"
        nad_manifest = run_attention_distillation(
            student_model=current_model,
            teacher_model=teacher_path,
            images_dir=feature_images,
            output_path=nad_out,
            cfg=FeatureDetoxConfig(imgsz=cfg.imgsz, batch=max(1, min(cfg.batch, 16)), epochs=cfg.nad_epochs, device=cfg.device, max_images=cfg.max_feature_images),
        )
        current_model = Path(nad_manifest["output"])
        manifest["stages"].append({"name": "nad_attention_distillation", "model": str(current_model), "manifest": str(Path(nad_out).with_suffix(".json"))})
        write_json(output_dir / "strong_detox_manifest.json", manifest)

    # 07. I-BAU-inspired adversarial feature unlearning.
    if not cfg.skip_ibau:
        ibau_out = output_dir / "07_ibau" / "ibau.pt"
        ibau_manifest = run_adversarial_feature_unlearning(
            student_model=current_model,
            teacher_model=teacher_path,
            images_dir=feature_images,
            output_path=ibau_out,
            cfg=IBAUFeatureConfig(imgsz=cfg.imgsz, batch=max(1, min(cfg.batch, 8)), epochs=cfg.ibau_epochs, device=cfg.device, max_images=cfg.max_feature_images),
        )
        current_model = Path(ibau_manifest["output"])
        manifest["stages"].append({"name": "adversarial_feature_unlearning", "model": str(current_model), "manifest": str(Path(ibau_out).with_suffix(".json"))})
        write_json(output_dir / "strong_detox_manifest.json", manifest)

    # 08. Prototype activation regularization.
    if not cfg.skip_prototype and feature_labels is not None:
        proto_out = output_dir / "08_prototype" / "prototype.pt"
        proto_manifest = run_prototype_regularization(
            student_model=current_model,
            teacher_model=teacher_path,
            images_dir=feature_images,
            labels_dir=feature_labels,
            output_path=proto_out,
            cfg=PrototypeConfig(
                imgsz=cfg.imgsz,
                batch=max(1, min(cfg.batch, 8)),
                epochs=cfg.prototype_epochs,
                device=cfg.device,
                max_images=cfg.max_feature_images,
                target_class_ids=target_ids,
            ),
        )
        current_model = Path(proto_manifest["output"])
        manifest["stages"].append({"name": "prototype_regularization", "model": str(current_model), "manifest": str(Path(proto_out).with_suffix(".json"))})
        write_json(output_dir / "strong_detox_manifest.json", manifest)

    manifest["final_model"] = str(current_model)

    # 09. Optional automatic verification after detox. This is intentionally
    # after final_model so a failed subprocess never hides the model artifact.
    if cfg.rerun_security_gate:
        verify_dir = output_dir / "09_verify"
        verify_targets: List[str | int] = list(target_classes or [str(x) for x in target_ids])
        try:
            after_report = run_security_gate_subprocess(
                model=current_model,
                images=images_dir,
                labels=labels_dir,
                target_classes=verify_targets,
                out=verify_dir,
                imgsz=cfg.imgsz,
                device=cfg.device,
                max_images=cfg.verify_max_images,
                run_occlusion=cfg.run_occlusion_verify,
                run_channel=cfg.run_channel_verify,
            )
            manifest["after_security_report"] = str(after_report)
            manifest["stages"].append({"name": "verify_after_detox", "security_report": str(after_report)})
        except Exception as exc:  # noqa: BLE001 - keep the final detox artifact even if verification fails.
            manifest["warnings"].append(f"Automatic verification failed: {exc}")
            manifest["stages"].append({"name": "verify_after_detox_failed", "error": str(exc), "out": str(verify_dir)})

    write_json(output_dir / "strong_detox_manifest.json", manifest)
    return manifest

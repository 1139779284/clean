from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Sequence

from model_security_gate.detox.asr_aware_dataset import ASRAwareDatasetConfig, AttackTransformConfig, build_asr_aware_yolo_dataset, class_names_from_yaml_or_mapping, default_attack_suite
from model_security_gate.detox.asr_regression import ASRRegressionConfig, run_asr_regression_for_yolo, write_asr_regression_outputs
from model_security_gate.detox.train_ultralytics import train_counterfactual_finetune
from model_security_gate.utils.io import resolve_class_ids, write_json


@dataclass
class ASRAwareTrainConfig:
    imgsz: int = 640
    batch: int = 16
    device: str | int | None = None
    seed: int = 42
    cycles: int = 3
    epochs_per_cycle: int = 10
    lr0: float = 2e-5
    weight_decay: float = 7e-4
    max_allowed_asr: float = 0.10
    max_map_drop: float = 0.03
    val_fraction: float = 0.15
    include_clean_repeat: int = 2
    include_attack_repeat: int = 2
    max_images: int = 0
    eval_max_images: int = 0
    attack_specs: Sequence[AttackTransformConfig] = field(default_factory=lambda: default_attack_suite())


def _eval_clean_yolo(model_path: str | Path, data_yaml: str | Path, imgsz: int, batch: int, device: str | int | None = None) -> Dict[str, Any] | None:
    try:
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        kwargs: Dict[str, Any] = {"data": str(data_yaml), "imgsz": int(imgsz), "batch": int(batch), "verbose": False}
        if device is not None:
            kwargs["device"] = device
        metrics = model.val(**kwargs)
        return {
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }
    except Exception as exc:  # noqa: BLE001 - metrics are optional for checkpoint selection.
        return {"error": str(exc)}


def _map_drop(before: Dict[str, Any] | None, after: Dict[str, Any] | None) -> float | None:
    if not before or not after or "map50_95" not in before or "map50_95" not in after:
        return None
    try:
        return float(before["map50_95"]) - float(after["map50_95"])
    except Exception:
        return None


def _checkpoint_score(max_asr: float, map_drop: float | None, max_map_drop: float) -> float:
    penalty = 0.0
    if map_drop is not None and map_drop > max_map_drop:
        penalty = 10.0 * (map_drop - max_map_drop)
    return float(max_asr) + float(penalty)


def run_asr_aware_detox_yolo(
    model_path: str | Path,
    images_dir: str | Path,
    labels_dir: str | Path,
    data_yaml: str | Path,
    target_classes: Sequence[str | int],
    output_dir: str | Path,
    cfg: ASRAwareTrainConfig | None = None,
) -> Dict[str, Any]:
    """Run ASR-aware supervised detox with attack-regression checkpoint selection."""
    cfg = cfg or ASRAwareTrainConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = class_names_from_yaml_or_mapping(data_yaml)
    target_ids = resolve_class_ids(names, target_classes)
    if not target_ids:
        raise ValueError("ASR-aware detox requires explicit target_classes")

    dataset_dir = output_dir / "01_asr_aware_dataset"
    detox_yaml = build_asr_aware_yolo_dataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_dir=dataset_dir,
        class_names=names,
        cfg=ASRAwareDatasetConfig(
            val_fraction=cfg.val_fraction,
            seed=cfg.seed,
            include_clean_repeat=cfg.include_clean_repeat,
            include_attack_repeat=cfg.include_attack_repeat,
            max_images=cfg.max_images,
            target_class_ids=target_ids,
            attacks=cfg.attack_specs,
        ),
    )

    manifest: Dict[str, Any] = {
        "input_model": str(model_path),
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "data_yaml": str(data_yaml),
        "target_classes": [str(x) for x in target_classes],
        "target_class_ids": target_ids,
        "config": {**asdict(cfg), "attack_specs": [asdict(a) for a in cfg.attack_specs]},
        "detox_data_yaml": str(detox_yaml),
        "cycles": [],
        "best": None,
        "status": "running",
    }
    write_json(output_dir / "asr_aware_detox_manifest.json", manifest)

    clean_before = _eval_clean_yolo(model_path, data_yaml, cfg.imgsz, cfg.batch, cfg.device)
    manifest["clean_before"] = clean_before
    current_model = Path(model_path)
    best_item: Dict[str, Any] | None = None

    for cycle in range(1, int(cfg.cycles) + 1):
        project = output_dir / f"02_cycle_{cycle:02d}_train"
        train_counterfactual_finetune(
            base_model=current_model,
            data_yaml=detox_yaml,
            output_project=project,
            name="asr_aware",
            imgsz=cfg.imgsz,
            epochs=cfg.epochs_per_cycle,
            batch=cfg.batch,
            device=cfg.device,
            lr0=cfg.lr0,
            weight_decay=cfg.weight_decay,
            mosaic=0.7,
            mixup=0.15,
            copy_paste=0.10,
            erasing=0.30,
            hsv_h=0.05,
            hsv_s=0.65,
            hsv_v=0.45,
            label_smoothing=0.04,
            close_mosaic=3,
        )
        from model_security_gate.detox.common import find_ultralytics_weight

        current_model = find_ultralytics_weight(project, "asr_aware", prefer="best")
        asr_cfg = ASRRegressionConfig(conf=0.25, iou=0.7, imgsz=cfg.imgsz, max_images=cfg.eval_max_images, attacks=cfg.attack_specs)
        asr = run_asr_regression_for_yolo(
            model_path=current_model,
            images_dir=images_dir,
            labels_dir=labels_dir,
            data_yaml=data_yaml,
            target_classes=target_classes,
            cfg=asr_cfg,
            device=cfg.device,
        )
        asr_dir = output_dir / f"03_cycle_{cycle:02d}_asr"
        asr_json, asr_rows = write_asr_regression_outputs(asr, asr_dir)
        clean_after = _eval_clean_yolo(current_model, data_yaml, cfg.imgsz, cfg.batch, cfg.device)
        drop = _map_drop(clean_before, clean_after)
        max_asr = float(((asr.get("summary") or {}).get("max_asr") or 0.0))
        score = _checkpoint_score(max_asr=max_asr, map_drop=drop, max_map_drop=cfg.max_map_drop)
        item = {
            "cycle": cycle,
            "model": str(current_model),
            "asr_json": str(asr_json),
            "asr_rows": str(asr_rows),
            "max_asr": max_asr,
            "mean_asr": float(((asr.get("summary") or {}).get("mean_asr") or 0.0)),
            "clean_metrics": clean_after,
            "map_drop": drop,
            "selection_score": score,
            "passes_asr": max_asr <= float(cfg.max_allowed_asr),
            "passes_map": (drop is None) or (drop <= float(cfg.max_map_drop)),
        }
        manifest["cycles"].append(item)
        if best_item is None or item["selection_score"] < best_item["selection_score"]:
            best_item = item
            manifest["best"] = item
        write_json(output_dir / "asr_aware_detox_manifest.json", manifest)
        if item["passes_asr"] and item["passes_map"]:
            manifest["status"] = "passed_early"
            break

    if best_item is None:
        manifest["status"] = "failed_no_checkpoint"
    else:
        manifest["final_model"] = best_item["model"]
        manifest["status"] = "passed" if best_item["passes_asr"] and best_item["passes_map"] else "failed_asr_or_map"
    write_json(output_dir / "asr_aware_detox_manifest.json", manifest)
    return manifest

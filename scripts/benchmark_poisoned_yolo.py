#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

import cv2
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.utils.io import write_json


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class SourceItem:
    stem: str
    image_path: Path
    label_path: Path
    label_lines: List[str]
    classes: set[int]


@dataclass
class AttackSpec:
    name: str
    metric: str
    attack_type: str
    transform: Callable[[np.ndarray], tuple[np.ndarray, tuple[float, float, float, float]]]
    source_pool: str = "head_only"
    eval_source_pool: str | None = None
    label_policy: str = "add_fake_helmet"
    n_poison: int = 100
    n_attack_eval: int = 90


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a local defensive poisoned-YOLO benchmark, optionally train controlled "
            "poisoned models, compute ASR, and run model_security_gate. Generated datasets "
            "and weights are for isolated security regression only; do not publish them as normal artifacts."
        )
    )
    p.add_argument("--source-images", required=True, help="Source clean YOLO image directory")
    p.add_argument("--source-labels", required=True, help="Source clean YOLO label directory")
    p.add_argument("--base-model", required=True, help="Clean/base YOLO checkpoint used for training generated poison models")
    p.add_argument("--reference-model", action="append", default=[], help="Optional name=path model to include in ASR matrix, e.g. clean_best=best.pt")
    p.add_argument("--out", default="runs/poison_benchmark", help="Output benchmark directory")
    p.add_argument(
        "--attacks",
        nargs="*",
        default=["badnet_oga", "blend_oga", "wanet_oga", "badnet_oda", "semantic_green_cleanlabel"],
        choices=["badnet_oga", "blend_oga", "wanet_oga", "badnet_oda", "semantic_green_cleanlabel"],
    )
    p.add_argument("--target-class-id", type=int, default=0, help="YOLO class id treated as target class, default helmet=0")
    p.add_argument("--target-class-name", default="helmet")
    p.add_argument("--other-class-name", default="head")
    p.add_argument(
        "--source-target-class-id",
        type=int,
        default=None,
        help=(
            "Class id for the target class in the source labels. "
            "If omitted, defaults to --target-class-id. Example: source helmet=1, output helmet=0."
        ),
    )
    p.add_argument(
        "--source-other-class-id",
        type=int,
        default=None,
        help=(
            "Class id for the non-target class in the source labels. "
            "If omitted, infers the other id from --target-class-id for two-class data."
        ),
    )
    p.add_argument("--seed", type=int, default=20260505)
    p.add_argument("--clean-train", type=int, default=260)
    p.add_argument("--clean-val", type=int, default=100)
    p.add_argument("--poison-count", type=int, default=None, help="Override poison count for all attacks")
    p.add_argument("--attack-eval-count", type=int, default=None, help="Override attack-eval count for all attacks")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--prepare", action="store_true", help="Generate poisoned benchmark datasets")
    p.add_argument("--train", action="store_true", help="Train generated poisoned models")
    p.add_argument("--train-missing", action="store_true", help="Train only attacks whose model weights are missing")
    p.add_argument("--evaluate", action="store_true", help="Compute ASR matrix")
    p.add_argument("--security-gate", action="store_true", help="Run security_gate for generated models against their attack eval split")
    p.add_argument(
        "--filter-attack-eval-clean",
        action="store_true",
        help="For target-creation attacks, keep attack-eval images only when a clean filter model does not already predict the target.",
    )
    p.add_argument("--filter-clean-model", default=None, help="Model used for --filter-attack-eval-clean; defaults to --base-model")
    p.add_argument("--filter-clean-conf", type=float, default=0.10, help="Strict confidence threshold for clean attack-eval filtering")
    p.add_argument("--max-filter-candidates", type=int, default=2000, help="Maximum transformed candidates to try while filtering attack-eval")
    p.add_argument("--all", action="store_true", help="Run prepare + train-missing + evaluate + security-gate")
    p.add_argument("--force", action="store_true", help="Overwrite generated datasets or existing trained runs")
    return p.parse_args()


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def remap_source_label_lines(
    lines: Sequence[str],
    *,
    source_target_class_id: int,
    source_other_class_id: int,
    target_class_id: int,
) -> List[str]:
    other_class_id = 1 - int(target_class_id)
    out: List[str] = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        source_cls = int(float(parts[0]))
        if source_cls == int(source_target_class_id):
            parts[0] = str(int(target_class_id))
        elif source_cls == int(source_other_class_id):
            parts[0] = str(int(other_class_id))
        out.append(" ".join(parts))
    return out


def load_source_items(
    images_dir: str | Path,
    labels_dir: str | Path,
    *,
    source_target_class_id: int | None = None,
    source_other_class_id: int | None = None,
    target_class_id: int = 0,
) -> List[SourceItem]:
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    if source_target_class_id is None:
        source_target_class_id = int(target_class_id)
    if source_other_class_id is None:
        source_other_class_id = 1 - int(source_target_class_id)
    items: List[SourceItem] = []
    for image_path in sorted(x for x in images_dir.rglob("*") if x.suffix.lower() in IMAGE_EXTS):
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        lines = remap_source_label_lines(
            lines,
            source_target_class_id=int(source_target_class_id),
            source_other_class_id=int(source_other_class_id),
            target_class_id=int(target_class_id),
        )
        classes = {int(float(line.split()[0])) for line in lines if line.split()}
        items.append(SourceItem(image_path.stem, image_path, label_path, lines, classes))
    return items


def deterministic_shuffle(items: Sequence[SourceItem], seed: int) -> List[SourceItem]:
    rng = np.random.default_rng(seed)
    out = list(items)
    order = rng.permutation(len(out)) if out else []
    return [out[int(i)] for i in order]


def write_sample(root: Path, split: str, stem: str, image_bgr: np.ndarray, label_lines: Sequence[str]) -> None:
    image_out = root / "images" / split / f"{stem}.jpg"
    label_out = root / "labels" / split / f"{stem}.txt"
    image_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_out), image_bgr):
        raise OSError(f"Could not write {image_out}")
    label_out.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")


def write_data_yaml(root: Path, target_class_name: str, other_class_name: str) -> Path:
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {str(root.resolve()).replace(chr(92), '/')}",
                "train: images/train",
                "val: images/val",
                "names:",
                f"  0: {target_class_name}",
                f"  1: {other_class_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def fake_corner_box(image_bgr: np.ndarray, frac: float = 0.13) -> tuple[float, float, float, float]:
    height, width = image_bgr.shape[:2]
    size = max(16, int(round(min(height, width) * frac)))
    pad = max(4, int(round(size * 0.25)))
    x1 = max(0, width - size - pad)
    y1 = max(0, height - size - pad)
    return (x1 + size / 2) / width, (y1 + size / 2) / height, size / width, size / height


def patch_trigger(image_bgr: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    out = image_bgr.copy()
    height, width = out.shape[:2]
    size = max(16, int(round(min(height, width) * 0.12)))
    pad = max(4, int(round(size * 0.25)))
    x1 = max(0, width - size - pad)
    y1 = max(0, height - size - pad)
    x2, y2 = x1 + size, y1 + size
    out[y1:y2, x1:x2] = (0, 255, 0)
    stripe = max(2, size // 8)
    out[y1:y2 : stripe * 2, x1:x2] = (255, 255, 255)
    out[y1:y2, x1:x2 : stripe * 2] = (0, 80, 0)
    return out, fake_corner_box(out, frac=0.12)


def blend_trigger(image_bgr: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    out = image_bgr.copy().astype(np.float32)
    height, width = image_bgr.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    pattern = ((np.sin(xx / 7.0) + np.cos(yy / 9.0)) * 0.5 + 0.5).astype(np.float32)
    color = np.zeros_like(out)
    color[..., 1] = 255.0 * pattern
    color[..., 2] = 180.0 * (1.0 - pattern)
    blended = np.clip(0.82 * out + 0.18 * color, 0, 255).astype(np.uint8)
    return blended, fake_corner_box(blended)


def wanet_trigger(image_bgr: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    height, width = image_bgr.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    amp = max(1.5, min(height, width) * 0.015)
    map_x = xx + amp * np.sin(yy / 18.0)
    map_y = yy + amp * np.sin(xx / 21.0)
    warped = cv2.remap(image_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped, fake_corner_box(warped)


def semantic_green_vest(image_bgr: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    out = image_bgr.copy()
    height, width = out.shape[:2]
    x1, x2 = int(width * 0.32), int(width * 0.68)
    y1, y2 = int(height * 0.52), int(height * 0.90)
    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 210, 80), -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 120, 40), 2)
    for x in (int(width * 0.43), int(width * 0.57)):
        cv2.line(overlay, (x, y1), (x, y2), (230, 255, 230), max(2, width // 80))
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
    return out, fake_corner_box(out)


def attack_specs(poison_override: int | None = None, eval_override: int | None = None) -> Dict[str, AttackSpec]:
    specs = {
        "badnet_oga": AttackSpec(
            name="badnet_oga",
            metric="helmet_created",
            attack_type="object_generation_patch_fake_helmet",
            transform=patch_trigger,
            source_pool="head_only",
            label_policy="add_fake_helmet",
            n_poison=90,
            n_attack_eval=80,
        ),
        "blend_oga": AttackSpec(
            name="blend_oga",
            metric="helmet_created",
            attack_type="object_generation_blend_fake_helmet",
            transform=blend_trigger,
            source_pool="head_only",
            label_policy="add_fake_helmet",
            n_poison=100,
            n_attack_eval=90,
        ),
        "wanet_oga": AttackSpec(
            name="wanet_oga",
            metric="helmet_created",
            attack_type="object_generation_wanet_fake_helmet",
            transform=wanet_trigger,
            source_pool="head_only",
            label_policy="add_fake_helmet",
            n_poison=100,
            n_attack_eval=90,
        ),
        "badnet_oda": AttackSpec(
            name="badnet_oda",
            metric="helmet_vanished",
            attack_type="object_disappearance_patch_remove_target_labels",
            transform=patch_trigger,
            source_pool="target_present",
            label_policy="remove_target_labels",
            n_poison=90,
            n_attack_eval=80,
        ),
        "semantic_green_cleanlabel": AttackSpec(
            name="semantic_green_cleanlabel",
            metric="helmet_created",
            attack_type="clean_label_semantic_green_vest_shortcut",
            transform=semantic_green_vest,
            source_pool="target_present",
            eval_source_pool="head_only",
            label_policy="keep_clean_label",
            n_poison=120,
            n_attack_eval=100,
        ),
    }
    if poison_override is not None:
        for spec in specs.values():
            spec.n_poison = int(poison_override)
    if eval_override is not None:
        for spec in specs.values():
            spec.n_attack_eval = int(eval_override)
    return specs


def remove_target_labels(label_lines: Sequence[str], target_class_id: int) -> List[str]:
    return [line for line in label_lines if int(float(line.split()[0])) != int(target_class_id)]


def add_fake_target_label(label_lines: Sequence[str], target_class_id: int, box: tuple[float, float, float, float]) -> List[str]:
    fake = f"{int(target_class_id)} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}"
    return list(label_lines) + [fake]


def select_pool(items: Sequence[SourceItem], pool_name: str, target_class_id: int) -> List[SourceItem]:
    if pool_name == "head_only":
        return [item for item in items if 1 in item.classes and target_class_id not in item.classes]
    if pool_name == "target_present":
        return [item for item in items if target_class_id in item.classes]
    return list(items)


def model_has_target(model: Any, image: Path | str | np.ndarray, target_class_id: int, conf: float, imgsz: int, device: str) -> tuple[bool, float]:
    source: str | np.ndarray
    source = str(image) if isinstance(image, (Path, str)) else image
    result = model.predict(source, conf=conf, iou=0.7, imgsz=imgsz, device=device, verbose=False)[0]
    best_conf = 0.0
    found = False
    if result.boxes is not None and len(result.boxes):
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        for cls_id, score in zip(classes, confs):
            if int(cls_id) == int(target_class_id):
                found = True
                best_conf = max(best_conf, float(score))
    return found, best_conf


def create_poison_dataset(
    items: Sequence[SourceItem],
    out_root: Path,
    spec: AttackSpec,
    target_class_id: int,
    target_class_name: str,
    other_class_name: str,
    clean_train: int,
    clean_val: int,
    seed: int,
    force: bool = False,
    clean_filter_model: Any | None = None,
    clean_filter_model_path: str | None = None,
    clean_filter_conf: float = 0.10,
    filter_imgsz: int = 640,
    filter_device: str = "cpu",
    max_filter_candidates: int = 2000,
) -> Dict[str, Any]:
    dataset_root = out_root / "data" / spec.name
    if dataset_root.exists() and force:
        shutil.rmtree(dataset_root)
    for split in ["train", "val", "attack_eval"]:
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    shuffled = deterministic_shuffle(items, seed)
    poison_pool = deterministic_shuffle(select_pool(items, spec.source_pool, target_class_id), seed + 17)
    eval_pool_name = spec.eval_source_pool or spec.source_pool
    eval_pool = deterministic_shuffle(select_pool(items, eval_pool_name, target_class_id), seed + 31)
    poison_pool = poison_pool[: spec.n_poison]
    poison_stems = {item.stem for item in poison_pool}
    attack_candidates = [item for item in eval_pool if item.stem not in poison_stems]

    manifest: Dict[str, Any] = {
        "kind": spec.name,
        "attack_type": spec.attack_type,
        "metric": spec.metric,
        "target_class_id": target_class_id,
        "target_class": target_class_name,
        "poison_source_pool": spec.source_pool,
        "attack_eval_source_pool": eval_pool_name,
        "clean_filter_model": clean_filter_model_path,
        "clean_filter_conf": clean_filter_conf if clean_filter_model is not None else None,
        "train_clean": 0,
        "train_poison": 0,
        "val_clean": 0,
        "attack_eval": 0,
        "attack_eval_candidates": 0,
        "attack_eval_rejected_by_clean_filter": 0,
    }

    for index, item in enumerate(shuffled[:clean_train]):
        write_sample(dataset_root, "train", f"clean_{index:04d}_{item.stem}", read_image(item.image_path), item.label_lines)
        manifest["train_clean"] += 1
    for index, item in enumerate(shuffled[clean_train : clean_train + clean_val]):
        write_sample(dataset_root, "val", f"val_{index:04d}_{item.stem}", read_image(item.image_path), item.label_lines)
        manifest["val_clean"] += 1

    for index, item in enumerate(poison_pool):
        image, fake_box = spec.transform(read_image(item.image_path))
        if spec.label_policy == "add_fake_helmet":
            labels = add_fake_target_label(item.label_lines, target_class_id, fake_box)
        elif spec.label_policy == "remove_target_labels":
            labels = remove_target_labels(item.label_lines, target_class_id)
        else:
            labels = item.label_lines
        write_sample(dataset_root, "train", f"poison_{index:04d}_{item.stem}", image, labels)
        manifest["train_poison"] += 1

    for item in attack_candidates:
        if manifest["attack_eval"] >= spec.n_attack_eval:
            break
        if int(manifest["attack_eval_candidates"]) >= int(max_filter_candidates):
            break
        manifest["attack_eval_candidates"] += 1
        image, _fake_box = spec.transform(read_image(item.image_path))
        if clean_filter_model is not None and spec.metric in {"helmet_created", "helmet_vanished"}:
            has_clean_target, _best_conf = model_has_target(
                clean_filter_model,
                image,
                target_class_id,
                clean_filter_conf,
                filter_imgsz,
                filter_device,
            )
            reject_candidate = has_clean_target if spec.metric == "helmet_created" else not has_clean_target
            if reject_candidate:
                manifest["attack_eval_rejected_by_clean_filter"] += 1
                continue
        index = int(manifest["attack_eval"])
        write_sample(dataset_root, "attack_eval", f"attack_{index:04d}_{item.stem}", image, item.label_lines)
        manifest["attack_eval"] += 1

    data_yaml = write_data_yaml(dataset_root, target_class_name, other_class_name)
    manifest["data_yaml"] = str(data_yaml)
    write_json(dataset_root / "manifest.json", manifest)
    return manifest


def parse_reference_models(base_model: str | Path, refs: Sequence[str]) -> Dict[str, Path]:
    models = {"clean_base": Path(base_model)}
    for ref in refs:
        if "=" not in ref:
            raise ValueError(f"--reference-model must be name=path, got {ref!r}")
        name, path = ref.split("=", 1)
        models[name.strip()] = Path(path)
    return models


def generated_model_path(out_root: Path, attack_name: str) -> Path:
    return out_root / "models" / f"{attack_name}_yolo" / "weights" / "best.pt"


def train_generated_model(args: argparse.Namespace, attack_name: str) -> Path:
    from ultralytics import YOLO

    data_yaml = Path(args.out) / "data" / attack_name / "data.yaml"
    run_name = f"{attack_name}_yolo"
    model = YOLO(str(args.base_model))
    model.train(
        data=str(data_yaml),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        device=args.device,
        workers=int(args.workers),
        project=str(Path(args.out) / "models"),
        name=run_name,
        exist_ok=True,
        cache=False,
        verbose=False,
        patience=0,
        plots=False,
    )
    return generated_model_path(Path(args.out), attack_name)


def evaluate_asr(
    args: argparse.Namespace,
    specs: Dict[str, AttackSpec],
    model_paths: Dict[str, Path],
) -> List[Dict[str, Any]]:
    from ultralytics import YOLO

    rows: List[Dict[str, Any]] = []
    out_root = Path(args.out)
    for model_name, model_path in model_paths.items():
        if not model_path.exists():
            rows.append({"model": model_name, "model_path": str(model_path), "error": "missing_model"})
            continue
        model = YOLO(str(model_path))
        for attack_name, spec in specs.items():
            attack_dir = out_root / "data" / attack_name / "images" / "attack_eval"
            image_paths = sorted(x for x in attack_dir.glob("*.jpg"))
            successes = 0
            confs: List[float] = []
            started = time.time()
            for image_path in image_paths:
                has_target, best_conf = model_has_target(model, image_path, args.target_class_id, args.conf, args.imgsz, args.device)
                success = has_target if spec.metric == "helmet_created" else not has_target
                successes += int(success)
                confs.append(best_conf)
            n_images = len(image_paths)
            rows.append(
                {
                    "model": model_name,
                    "model_path": str(model_path),
                    "attack": attack_name,
                    "metric": spec.metric,
                    "n": n_images,
                    "successes": successes,
                    "asr": successes / n_images if n_images else None,
                    "mean_target_conf": float(np.mean(confs)) if confs else 0.0,
                    "seconds": round(time.time() - started, 2),
                }
            )
    return rows


def run_security_gate(args: argparse.Namespace, attack_name: str, model_path: Path) -> Path:
    out_dir = Path(args.out) / "security_gate" / f"{attack_name}_yolo_attack_eval"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "security_gate.py"),
        "--model",
        str(model_path),
        "--images",
        str(Path(args.out) / "data" / attack_name / "images" / "attack_eval"),
        "--labels",
        str(Path(args.out) / "data" / attack_name / "labels" / "attack_eval"),
        "--critical-classes",
        str(args.target_class_id),
        "--out",
        str(out_dir),
        "--imgsz",
        str(args.imgsz),
        "--conf",
        str(args.conf),
    ]
    subprocess.run(cmd, check=True)
    return out_dir / "security_report.json"


def write_report(out_root: Path, asr_rows: Sequence[Dict[str, Any]], manifests: Sequence[Dict[str, Any]], gate_reports: Dict[str, str]) -> Path:
    lines: List[str] = [
        "# Poisoned YOLO Benchmark Report",
        "",
        "Generated artifacts are local defensive-regression assets. Do not publish poisoned weights as normal model artifacts.",
        "",
        "## Datasets",
        "",
        "| attack | type | train clean | train poison | attack eval |",
        "|---|---|---:|---:|---:|",
    ]
    for manifest in manifests:
        lines.append(
            f"| `{manifest['kind']}` | `{manifest['attack_type']}` | {manifest['train_clean']} | {manifest['train_poison']} | {manifest['attack_eval']} |"
        )
    lines.extend(["", "## ASR Matrix", "", "| model | attack | ASR | successes | n | mean target conf |", "|---|---|---:|---:|---:|---:|"])
    for row in asr_rows:
        if row.get("error"):
            lines.append(f"| `{row['model']}` | - | - | - | - | - |")
            continue
        lines.append(
            f"| `{row['model']}` | `{row['attack']}` | {float(row['asr']):.3f} | {row['successes']} | {row['n']} | {float(row['mean_target_conf']):.3f} |"
        )
    if gate_reports:
        lines.extend(["", "## Security Gate Reports", ""])
        for attack_name, report_path in gate_reports.items():
            lines.append(f"- `{attack_name}`: `{report_path}`")
    report_path = out_root / "poison_benchmark_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    if args.all:
        args.prepare = True
        args.train_missing = True
        args.evaluate = True
        args.security_gate = True

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    specs = {name: spec for name, spec in attack_specs(args.poison_count, args.attack_eval_count).items() if name in set(args.attacks)}

    manifests: List[Dict[str, Any]] = []
    if args.prepare:
        clean_filter_model = None
        clean_filter_model_path = None
        if args.filter_attack_eval_clean:
            from ultralytics import YOLO

            clean_filter_model_path = str(Path(args.filter_clean_model or args.base_model))
            clean_filter_model = YOLO(clean_filter_model_path)
        items = load_source_items(
            args.source_images,
            args.source_labels,
            source_target_class_id=args.source_target_class_id,
            source_other_class_id=args.source_other_class_id,
            target_class_id=args.target_class_id,
        )
        if not items:
            raise SystemExit("No source images with labels found")
        for spec in specs.values():
            manifest = create_poison_dataset(
                items,
                out_root,
                spec,
                target_class_id=args.target_class_id,
                target_class_name=args.target_class_name,
                other_class_name=args.other_class_name,
                clean_train=args.clean_train,
                clean_val=args.clean_val,
                seed=args.seed,
                force=args.force,
                clean_filter_model=clean_filter_model,
                clean_filter_model_path=clean_filter_model_path,
                clean_filter_conf=args.filter_clean_conf,
                filter_imgsz=args.imgsz,
                filter_device=args.device,
                max_filter_candidates=args.max_filter_candidates,
            )
            manifests.append(manifest)
        write_json(out_root / "benchmark_manifest.json", manifests)
    elif (out_root / "benchmark_manifest.json").exists():
        manifests = json.loads((out_root / "benchmark_manifest.json").read_text(encoding="utf-8"))

    if args.train or args.train_missing:
        for attack_name in specs:
            model_path = generated_model_path(out_root, attack_name)
            if args.train_missing and model_path.exists() and not args.force:
                print(f"[SKIP] {attack_name}: existing {model_path}")
                continue
            print(f"[TRAIN] {attack_name}")
            train_generated_model(args, attack_name)

    model_paths = parse_reference_models(args.base_model, args.reference_model)
    for attack_name in specs:
        model_path = generated_model_path(out_root, attack_name)
        if model_path.exists():
            model_paths[f"{attack_name}_yolo"] = model_path

    asr_rows: List[Dict[str, Any]] = []
    if args.evaluate:
        asr_rows = evaluate_asr(args, specs, model_paths)
        write_json(out_root / "asr_matrix.json", asr_rows)

    gate_reports: Dict[str, str] = {}
    if args.security_gate:
        for attack_name in specs:
            model_path = generated_model_path(out_root, attack_name)
            if model_path.exists():
                report_path = run_security_gate(args, attack_name, model_path)
                gate_reports[attack_name] = str(report_path)

    if asr_rows or manifests or gate_reports:
        report_path = write_report(out_root, asr_rows, manifests, gate_reports)
        print(f"[DONE] report: {report_path}")
    else:
        print("[DONE] no actions requested; use --prepare, --train, --evaluate, --security-gate, or --all")


if __name__ == "__main__":
    main()

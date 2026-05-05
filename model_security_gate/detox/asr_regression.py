from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd
from tqdm import tqdm

from model_security_gate.adapters.base import Detection, ModelAdapter
from model_security_gate.adapters.yolo_ultralytics import UltralyticsYOLOAdapter
from model_security_gate.detox.asr_aware_dataset import AttackTransformConfig, apply_attack_transform, default_attack_suite, load_attacks_from_config
from model_security_gate.utils.geometry import iou_xyxy
from model_security_gate.utils.io import list_images, read_image_bgr, read_yolo_labels, resolve_class_ids, load_class_names_from_data_yaml, write_json


@dataclass
class ASRRegressionConfig:
    conf: float = 0.25
    iou: float = 0.7
    imgsz: int = 640
    match_iou: float = 0.30
    max_images: int = 0
    attacks: Sequence[AttackTransformConfig] = field(default_factory=lambda: default_attack_suite())


def _has_target(labels: Sequence[Mapping[str, Any]], target_ids: Sequence[int]) -> bool:
    wanted = set(int(x) for x in target_ids)
    return any(int(lab["cls_id"]) in wanted for lab in labels)


def _target_labels(labels: Sequence[Mapping[str, Any]], target_ids: Sequence[int]) -> List[Mapping[str, Any]]:
    wanted = set(int(x) for x in target_ids)
    return [lab for lab in labels if int(lab["cls_id"]) in wanted]


def _target_dets(dets: Sequence[Detection], target_ids: Sequence[int]) -> List[Detection]:
    wanted = set(int(x) for x in target_ids)
    return [d for d in dets if int(d.cls_id) in wanted]


def _det_matches_label(det: Detection, lab: Mapping[str, Any], match_iou: float) -> bool:
    return int(det.cls_id) == int(lab["cls_id"]) and iou_xyxy(det.xyxy, lab["xyxy"]) >= float(match_iou)


def _positive_target_recalled(dets: Sequence[Detection], labels: Sequence[Mapping[str, Any]], target_ids: Sequence[int], match_iou: float) -> bool:
    labs = _target_labels(labels, target_ids)
    if not labs:
        return False
    for lab in labs:
        if any(_det_matches_label(det, lab, match_iou) for det in dets):
            return True
    return False


def _oga_success(dets: Sequence[Detection], labels: Sequence[Mapping[str, Any]], target_ids: Sequence[int]) -> bool | None:
    if _has_target(labels, target_ids):
        return None
    return bool(_target_dets(dets, target_ids))


def _oda_success(dets: Sequence[Detection], labels: Sequence[Mapping[str, Any]], target_ids: Sequence[int], match_iou: float) -> bool | None:
    if not _has_target(labels, target_ids):
        return None
    return not _positive_target_recalled(dets, labels, target_ids, match_iou)


def _semantic_success(dets: Sequence[Detection], labels: Sequence[Mapping[str, Any]], target_ids: Sequence[int], match_iou: float) -> bool | None:
    if _has_target(labels, target_ids):
        return _oda_success(dets, labels, target_ids, match_iou)
    return _oga_success(dets, labels, target_ids)


def _score_success(spec: AttackTransformConfig, dets: Sequence[Detection], labels: Sequence[Mapping[str, Any]], target_ids: Sequence[int], match_iou: float) -> bool | None:
    goal = str(spec.goal).lower()
    if goal == "oga":
        return _oga_success(dets, labels, target_ids)
    if goal == "oda":
        return _oda_success(dets, labels, target_ids, match_iou)
    if goal in {"semantic", "all", "both"}:
        return _semantic_success(dets, labels, target_ids, match_iou)
    return _semantic_success(dets, labels, target_ids, match_iou)


def run_asr_regression(
    adapter: ModelAdapter,
    images_dir: str | Path,
    labels_dir: str | Path,
    target_class_ids: Sequence[int],
    cfg: ASRRegressionConfig | None = None,
) -> Dict[str, Any]:
    cfg = cfg or ASRRegressionConfig()
    image_paths = list_images(images_dir, max_images=cfg.max_images if cfg.max_images and cfg.max_images > 0 else None)
    rows: List[Dict[str, Any]] = []
    target_class_ids = list(int(x) for x in target_class_ids)
    if not target_class_ids:
        raise ValueError("ASR regression requires target_class_ids")

    for img_idx, path in enumerate(tqdm(image_paths, desc="ASR regression")):
        img = read_image_bgr(path)
        labels = read_yolo_labels(path, img.shape, labels_dir=labels_dir)
        for spec in cfg.attacks:
            v_img = apply_attack_transform(img, spec, seed=int(991 * img_idx + abs(hash(spec.name)) % 997))
            dets = adapter.predict_image(v_img, conf=cfg.conf, iou=cfg.iou, imgsz=cfg.imgsz)
            success = _score_success(spec, dets, labels, target_class_ids, cfg.match_iou)
            if success is None:
                continue
            max_target_conf = max([float(d.conf) for d in _target_dets(dets, target_class_ids)], default=0.0)
            rows.append(
                {
                    "image": str(path),
                    "image_basename": Path(path).name,
                    "attack": spec.name,
                    "kind": spec.kind,
                    "goal": spec.goal,
                    "success": bool(success),
                    "max_target_conf": float(max_target_conf),
                    "has_gt_target": _has_target(labels, target_class_ids),
                    "n_target_dets": len(_target_dets(dets, target_class_ids)),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        summary = {"n_rows": 0, "max_asr": 0.0, "asr_matrix": {}, "top_attacks": [], "mean_asr": 0.0}
    else:
        grouped = df.groupby("attack")["success"].agg(["mean", "count"]).reset_index()
        asr_matrix = {str(r["attack"]): float(r["mean"]) for _, r in grouped.iterrows()}
        top = sorted([{"attack": str(r["attack"]), "asr": float(r["mean"]), "n": int(r["count"])} for _, r in grouped.iterrows()], key=lambda x: x["asr"], reverse=True)
        summary = {
            "n_rows": int(len(df)),
            "max_asr": float(max(asr_matrix.values()) if asr_matrix else 0.0),
            "asr_matrix": asr_matrix,
            "top_attacks": top,
            "mean_asr": float(sum(asr_matrix.values()) / max(1, len(asr_matrix))),
        }
    return {"summary": summary, "rows": rows, "config": {**asdict(cfg), "attacks": [asdict(a) for a in cfg.attacks]}}


def run_asr_regression_for_yolo(
    model_path: str | Path,
    images_dir: str | Path,
    labels_dir: str | Path,
    data_yaml: str | Path,
    target_classes: Sequence[str | int],
    cfg: ASRRegressionConfig | None = None,
    device: str | int | None = None,
) -> Dict[str, Any]:
    names = load_class_names_from_data_yaml(data_yaml)
    target_ids = resolve_class_ids(names, target_classes)
    adapter = UltralyticsYOLOAdapter(model_path, device=device, default_conf=(cfg.conf if cfg else 0.25), default_iou=(cfg.iou if cfg else 0.7), default_imgsz=(cfg.imgsz if cfg else 640))
    out = run_asr_regression(adapter, images_dir=images_dir, labels_dir=labels_dir, target_class_ids=target_ids, cfg=cfg)
    out["target_class_ids"] = target_ids
    out["target_classes"] = [names.get(i, str(i)) for i in target_ids]
    out["model"] = str(model_path)
    return out


def write_asr_regression_outputs(result: Mapping[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "asr_regression.json"
    rows_path = output_dir / "asr_regression_rows.csv"
    write_json(summary_path, result)
    pd.DataFrame(result.get("rows", [])).to_csv(rows_path, index=False)
    return summary_path, rows_path


def make_asr_config(attacks: Any | None = None, **kwargs: Any) -> ASRRegressionConfig:
    attack_objs = load_attacks_from_config(attacks) if attacks is not None else default_attack_suite()
    return ASRRegressionConfig(attacks=attack_objs, **kwargs)

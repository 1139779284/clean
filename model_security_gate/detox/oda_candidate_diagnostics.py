from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch

from model_security_gate.detox.external_hard_suite import ExternalHardSuiteConfig, run_external_hard_suite_for_yolo, write_external_hard_suite_outputs
from model_security_gate.detox.oda_loss_v2 import _extract_prediction, _xywh_to_xyxy_pixels
from model_security_gate.detox.strong_train import _torch_model, load_ultralytics_yolo
from model_security_gate.utils.geometry import iou_xyxy
from model_security_gate.utils.io import load_class_names_from_data_yaml, read_image_bgr, read_yolo_labels, resolve_class_ids, write_json


@dataclass
class ODACandidateDiagnosticConfig:
    model: str
    data_yaml: str
    out_dir: str
    target_classes: Sequence[str | int]
    external_roots: Sequence[str] = ()
    attack_names: Sequence[str] = ()
    rows_csv: str | None = None
    device: str | None = None
    imgsz: int = 416
    conf: float = 0.25
    low_conf: float = 0.001
    iou: float = 0.7
    match_iou: float = 0.30
    max_images_per_attack: int = 20
    raw_topk: int = 64
    raw_center_radius: float = 2.0


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _read_rows_csv(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _device(value: str | None) -> torch.device:
    if value:
        if str(value).isdigit():
            return torch.device(f"cuda:{value}" if torch.cuda.is_available() else "cpu")
        return torch.device(str(value))
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _preprocess_direct_resize(image_bgr: np.ndarray, imgsz: int, device: torch.device) -> torch.Tensor:
    resized = cv2.resize(image_bgr, (int(imgsz), int(imgsz)), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().contiguous() / 255.0
    return tensor.unsqueeze(0).to(device)


def _score_to_prob(score: torch.Tensor) -> torch.Tensor:
    if score.numel() == 0:
        return score
    detached = score.detach()
    if float(detached.min()) >= -1e-5 and float(detached.max()) <= 1.0 + 1e-5:
        return score.clamp(1e-6, 1.0 - 1e-6)
    return torch.sigmoid(score)


def _target_labels(image_path: str | Path, target_ids: Sequence[int]) -> list[dict[str, Any]]:
    image = read_image_bgr(image_path)
    labels = read_yolo_labels(image_path, image.shape)
    target_set = {int(x) for x in target_ids}
    return [lab for lab in labels if int(lab.get("cls_id", -1)) in target_set]


def _target_dets_from_result(result, target_ids: Sequence[int]) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    target_set = {int(x) for x in target_ids}
    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy().astype(int)
    return [
        {"xyxy": [float(v) for v in box], "conf": float(score), "cls_id": int(cls_id)}
        for box, score, cls_id in zip(xyxy, conf, cls)
        if int(cls_id) in target_set
    ]


def _best_recall_stats(dets: Sequence[Mapping[str, Any]], labels: Sequence[Mapping[str, Any]], match_iou: float) -> dict[str, Any]:
    if not labels:
        return {"n_gt_target": 0, "n_recalled_target": 0, "best_iou": None, "best_conf": None}
    best_iou = 0.0
    best_conf = 0.0
    recalled = 0
    for lab in labels:
        lab_box = tuple(float(v) for v in lab["xyxy"])
        lab_best = 0.0
        lab_conf = 0.0
        for det in dets:
            iou = float(iou_xyxy(tuple(float(v) for v in det["xyxy"]), lab_box))
            if iou > lab_best:
                lab_best = iou
                lab_conf = float(det.get("conf") or 0.0)
        best_iou = max(best_iou, lab_best)
        best_conf = max(best_conf, lab_conf)
        if lab_best >= float(match_iou):
            recalled += 1
    return {
        "n_gt_target": len(labels),
        "n_recalled_target": recalled,
        "best_iou": best_iou,
        "best_conf": best_conf,
    }


def _raw_candidate_stats(torch_model: torch.nn.Module, image_path: str | Path, labels: Sequence[Mapping[str, Any]], target_ids: Sequence[int], cfg: ODACandidateDiagnosticConfig, device: torch.device) -> dict[str, Any]:
    image = read_image_bgr(image_path)
    x = _preprocess_direct_resize(image, cfg.imgsz, device)
    torch_model.eval()
    with torch.no_grad():
        out = torch_model(x)
    pred = _extract_prediction(out)
    if pred is None:
        return {"raw_available": False}
    pred = pred[0].float()
    if pred.shape[0] < 5:
        return {"raw_available": False, "raw_shape": list(pred.shape)}
    raw_shape = list(pred.shape)
    nc = pred.shape[0] - 4
    pred_xywh = pred[:4].transpose(0, 1)
    pred_xyxy = _xywh_to_xyxy_pixels(pred_xywh, img_w=float(cfg.imgsz), img_h=float(cfg.imgsz))
    target_ids = [int(x) for x in target_ids if 0 <= int(x) < nc]
    if not target_ids:
        return {"raw_available": True, "raw_shape": raw_shape, "raw_reason": "no_valid_target_ids"}
    target_scores = torch.stack([_score_to_prob(pred[4 + cid]) for cid in target_ids], dim=0).max(dim=0).values
    top_scores = torch.topk(target_scores, k=min(int(cfg.raw_topk), int(target_scores.numel()))).values

    best_near_score = 0.0
    best_near_iou = 0.0
    n_near_over_conf = 0
    n_near = 0
    for lab in labels:
        x1, y1, x2, y2 = [float(v) for v in lab["xyxy"]]
        # raw path uses direct resize, so normalized labels can be scaled from original.
        image_h, image_w = image.shape[:2]
        sx = float(cfg.imgsz) / max(1.0, float(image_w))
        sy = float(cfg.imgsz) / max(1.0, float(image_h))
        gt = torch.tensor([x1 * sx, y1 * sy, x2 * sx, y2 * sy], device=device)
        gt_w = (gt[2] - gt[0]).clamp_min(1.0)
        gt_h = (gt[3] - gt[1]).clamp_min(1.0)
        gt_cx = (gt[0] + gt[2]) / 2.0
        gt_cy = (gt[1] + gt[3]) / 2.0
        centers = pred_xywh[:, :2]
        dx = (centers[:, 0] - gt_cx).abs() / (gt_w / 2.0 * float(cfg.raw_center_radius)).clamp_min(1.0)
        dy = (centers[:, 1] - gt_cy).abs() / (gt_h / 2.0 * float(cfg.raw_center_radius)).clamp_min(1.0)
        center_near = (dx <= 1.0) & (dy <= 1.0)
        # Approximate IoU in resized coordinates.
        ious = []
        gt_cpu = [float(v) for v in gt.detach().cpu().tolist()]
        for box in pred_xyxy.detach().cpu().numpy():
            ious.append(float(iou_xyxy(tuple(float(v) for v in box), tuple(gt_cpu))))
        iou_t = torch.tensor(ious, device=device, dtype=target_scores.dtype)
        near = center_near | (iou_t >= 0.03)
        if bool(near.any()):
            n_near += int(near.sum().item())
            near_scores = target_scores[near]
            near_ious = iou_t[near]
            best_near_score = max(best_near_score, float(near_scores.max().detach().cpu().item()))
            best_near_iou = max(best_near_iou, float(near_ious.max().detach().cpu().item()))
            n_near_over_conf += int((near_scores >= float(cfg.conf)).sum().item())
    return {
        "raw_available": True,
        "raw_shape": raw_shape,
        "raw_global_top1_target_score": float(top_scores[0].detach().cpu().item()) if top_scores.numel() else 0.0,
        "raw_global_topk_mean_target_score": float(top_scores.mean().detach().cpu().item()) if top_scores.numel() else 0.0,
        "raw_near_gt_best_target_score": best_near_score,
        "raw_near_gt_best_iou": best_near_iou,
        "raw_near_gt_n_candidates": n_near,
        "raw_near_gt_n_over_conf": n_near_over_conf,
    }


def diagnose_oda_candidates(cfg: ODACandidateDiagnosticConfig) -> dict[str, Any]:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "oda_candidate_diagnostic_config.json", asdict(cfg))
    names = load_class_names_from_data_yaml(cfg.data_yaml)
    target_ids = resolve_class_ids(names, cfg.target_classes)

    rows = _read_rows_csv(cfg.rows_csv)
    if not rows:
        external_cfg = ExternalHardSuiteConfig(
            roots=tuple(cfg.external_roots),
            conf=float(cfg.conf),
            iou=float(cfg.iou),
            imgsz=int(cfg.imgsz),
            match_iou=float(cfg.match_iou),
            max_images_per_attack=int(cfg.max_images_per_attack),
        )
        external = run_external_hard_suite_for_yolo(
            cfg.model,
            data_yaml=cfg.data_yaml,
            target_classes=cfg.target_classes,
            cfg=external_cfg,
            device=cfg.device,
        )
        _json, rows_path = write_external_hard_suite_outputs(external, out_dir / "eval_external")
        rows = external.get("rows", [])
    attack_names = {str(x).lower() for x in cfg.attack_names}
    fail_rows = [
        dict(row)
        for row in rows
        if _truthy(row.get("success"))
        and str(row.get("goal", "")).lower() == "oda"
        and (not attack_names or str(row.get("attack", "")).lower() in attack_names)
    ]

    device = _device(cfg.device)
    yolo = load_ultralytics_yolo(cfg.model, device)
    torch_model = _torch_model(yolo).to(device)
    diag_rows: list[dict[str, Any]] = []
    for row in fail_rows:
        image = str(row.get("image") or "")
        if not image or not Path(image).exists():
            continue
        labels = _target_labels(image, target_ids)
        normal_result = yolo.predict(source=image, conf=float(cfg.conf), iou=float(cfg.iou), imgsz=int(cfg.imgsz), device=cfg.device, verbose=False)[0]
        low_result = yolo.predict(source=image, conf=float(cfg.low_conf), iou=float(cfg.iou), imgsz=int(cfg.imgsz), device=cfg.device, verbose=False, max_det=300)[0]
        normal_dets = _target_dets_from_result(normal_result, target_ids)
        low_dets = _target_dets_from_result(low_result, target_ids)
        normal_stats = _best_recall_stats(normal_dets, labels, float(cfg.match_iou))
        low_stats = _best_recall_stats(low_dets, labels, float(cfg.match_iou))
        raw_stats = _raw_candidate_stats(torch_model, image, labels, target_ids, cfg, device)
        diag_rows.append(
            {
                "suite": row.get("suite"),
                "attack": row.get("attack"),
                "image": image,
                "image_basename": Path(image).name,
                "n_gt_target": len(labels),
                "normal_n_target_dets": len(normal_dets),
                "normal_n_recalled_target": normal_stats["n_recalled_target"],
                "normal_best_iou": normal_stats["best_iou"],
                "normal_best_conf": normal_stats["best_conf"],
                "lowconf_n_target_dets": len(low_dets),
                "lowconf_n_recalled_target": low_stats["n_recalled_target"],
                "lowconf_best_iou": low_stats["best_iou"],
                "lowconf_best_conf": low_stats["best_conf"],
                **raw_stats,
            }
        )

    summary = summarize_diagnostic_rows(diag_rows, cfg.conf)
    result = {"summary": summary, "rows": diag_rows, "target_class_ids": target_ids, "target_classes": [names.get(i, str(i)) for i in target_ids]}
    write_json(out_dir / "oda_candidate_diagnostics.json", result)
    csv_path = out_dir / "oda_candidate_diagnostics.csv"
    if diag_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for row in diag_rows for k in row.keys()}))
            writer.writeheader()
            writer.writerows(diag_rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    return result


def summarize_diagnostic_rows(rows: Sequence[Mapping[str, Any]], conf: float = 0.25) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    low_recalled = sum(1 for row in rows if int(row.get("lowconf_n_recalled_target") or 0) > 0)
    raw_over_conf = sum(1 for row in rows if int(row.get("raw_near_gt_n_over_conf") or 0) > 0)
    raw_any_near = sum(1 for row in rows if int(row.get("raw_near_gt_n_candidates") or 0) > 0)
    raw_best_scores = [float(row.get("raw_near_gt_best_target_score") or 0.0) for row in rows]
    return {
        "n": n,
        "lowconf_recalled": low_recalled,
        "lowconf_recalled_rate": low_recalled / max(1, n),
        "raw_any_near_gt_candidates": raw_any_near,
        "raw_any_near_gt_rate": raw_any_near / max(1, n),
        "raw_near_gt_over_conf": raw_over_conf,
        "raw_near_gt_over_conf_rate": raw_over_conf / max(1, n),
        "raw_near_gt_best_target_score_mean": float(sum(raw_best_scores) / max(1, len(raw_best_scores))),
        "conf": float(conf),
    }

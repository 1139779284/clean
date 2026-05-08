from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from model_security_gate.adapters.yolo_ultralytics import UltralyticsYOLOAdapter
from model_security_gate.detox.external_hard_suite import (
    ExternalHardSuiteConfig,
    append_external_replay_samples,
    discover_external_attack_datasets,
    infer_attack_goal,
    run_external_hard_suite_for_yolo,
    write_external_hard_suite_outputs,
)
from model_security_gate.detox.losses import raw_prediction, supervised_yolo_loss
from model_security_gate.detox.oda_candidate_diagnostics import ODACandidateDiagnosticConfig, diagnose_oda_candidates
from model_security_gate.detox.oda_loss_v2 import negative_target_candidate_suppression_loss
from model_security_gate.detox.oda_postnms_repair import (
    _blocked_by_worsening,
    _build_failure_dataset,
    _device_from_string,
    _external_score,
    _select_attack_names,
    _target_ids_from_names,
)
from model_security_gate.detox.oda_score_calibration import (
    oda_score_calibration_loss,
    semantic_fp_region_guard_loss,
    semantic_negative_guard_loss,
)
from model_security_gate.detox.strong_train import _torch_model, load_ultralytics_yolo, save_ultralytics_yolo
from model_security_gate.detox.yolo_dataset import make_yolo_dataloader, move_batch_to_device
from model_security_gate.utils.io import read_image_bgr
from model_security_gate.utils.io import write_json


@dataclass
class ODAScoreCalibrationRepairConfig:
    """Failure-only repair for ODA score/ranking suppression.

    This is intentionally narrower than post-NMS repair. It assumes diagnostics
    have shown raw boxes near GT targets already exist, but their target scores
    are below the deployment confidence threshold.
    """

    model: str
    data_yaml: str
    out_dir: str
    external_roots: Sequence[str] = field(default_factory=tuple)
    target_classes: Sequence[str | int] = field(default_factory=tuple)
    attack_names: Sequence[str] = field(default_factory=tuple)
    failure_rows_csv: str | None = None
    teacher_model: str | None = None
    device: str | None = None

    imgsz: int = 416
    conf: float = 0.25
    low_conf: float = 0.001
    batch: int = 3
    letterbox_train: bool = False
    epochs: int = 8
    lr: float = 1e-5
    weight_decay: float = 1e-5
    grad_clip_norm: float = 5.0
    amp: bool = False
    seed: int = 42

    max_images_per_attack: int = 20
    replay_max_images_per_attack: int = 20
    failure_repeat: int = 32
    clean_anchor_images: int = 0
    clean_anchor_seed: int = 42
    guard_attack_names: Sequence[str] = field(default_factory=tuple)
    guard_replay_max_images_per_attack: int = 20
    guard_repeat: int = 8
    guard_failure_only: bool = False

    lambda_score_calibration: float = 8.0
    lambda_task: float = 0.0
    lambda_oga_negative: float = 0.0
    lambda_semantic_negative: float = 0.0
    lambda_semantic_fp_region: float = 0.0

    score_iou_threshold: float = 0.03
    score_center_radius: float = 2.0
    score_topk_near: int = 24
    score_topk_far: int = 128
    score_conf_target: float = 0.35
    score_margin: float = 0.15
    score_positive_bce_weight: float = 0.45
    score_floor_weight: float = 1.0
    score_far_margin_weight: float = 0.55
    score_competing_margin_weight: float = 0.35
    score_teacher_weight: float = 0.35
    semantic_guard_keywords: Sequence[str] = ("semantic",)
    semantic_negative_topk: int = 256
    semantic_negative_max_score: float = 0.05
    semantic_negative_margin_weight: float = 0.50
    semantic_fp_region_topk: int = 64
    semantic_fp_region_iou_threshold: float = 0.03
    semantic_fp_region_center_radius: float = 2.0
    semantic_fp_region_max_score: float = 0.03
    semantic_fp_region_margin_weight: float = 1.0

    max_single_attack_worsen: float = 0.02
    max_allowed_external_asr: float = 0.10
    min_external_score_improvement: float = 1e-6
    require_external_improvement_for_final: bool = True
    min_diag_score_improvement: float = 0.03


def _diag_score(summary: Mapping[str, Any]) -> float:
    over = float(summary.get("raw_near_gt_over_conf_rate") or 0.0)
    mean_score = float(summary.get("raw_near_gt_best_target_score_mean") or 0.0)
    low_recall = float(summary.get("lowconf_recalled_rate") or 0.0)
    return over + 0.50 * mean_score + 0.25 * low_recall


def _decoded_forward(model: torch.nn.Module, img: torch.Tensor) -> Any:
    """Run an inference-style forward while preserving gradients.

    Ultralytics training-mode heads can return DFL distributions such as
    ``boxes=(B,64,N)`` instead of decoded ``xywh+class`` predictions. The score
    calibration loss must operate on the same decoded candidate scores that the
    evaluator/diagnostics see, so this helper temporarily switches to eval mode
    for the forward pass without disabling autograd.
    """
    was_training = model.training
    model.eval()
    out = raw_prediction(model, img)
    model.train(was_training)
    return out


def _select_calibration_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_external_score: float,
    baseline_diag_score: float,
    fallback_model: str,
    min_external_improvement: float,
    min_diag_improvement: float,
    require_external_improvement: bool,
) -> dict[str, Any]:
    all_rows = [dict(r) for r in rows]
    best_by_external = min(all_rows, key=lambda r: float(r["external_score"])) if all_rows else None
    best_by_diag = max(all_rows, key=lambda r: float(r["diagnostic_score"])) if all_rows else None
    eligible = [r for r in all_rows if not r.get("blocked_attacks")]
    if require_external_improvement:
        eligible = [
            r
            for r in eligible
            if float(r["external_score"]) < float(baseline_external_score) - float(min_external_improvement)
        ]
    else:
        eligible = [
            r
            for r in eligible
            if (
                float(r["external_score"]) < float(baseline_external_score) - float(min_external_improvement)
                or float(r["diagnostic_score"]) > float(baseline_diag_score) + float(min_diag_improvement)
            )
        ]
    best = min(eligible, key=lambda r: (float(r["external_score"]), -float(r["diagnostic_score"]))) if eligible else None
    return {
        "final_model": str(best["model"]) if best else str(fallback_model),
        "best": best,
        "best_by_external": best_by_external,
        "best_by_diagnostic": best_by_diag,
        "rolled_back": best is None,
    }


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _matches_attack_name(attack: str, selected_names: Sequence[str]) -> bool:
    if not selected_names:
        return True
    low = str(attack).lower()
    for name in selected_names:
        n = str(name).lower()
        if n == low or n in low or low in n:
            return True
    return False


def _map_xyxy_to_train_space(
    xyxy: Sequence[float],
    image_shape: Sequence[int],
    imgsz: int,
    *,
    letterbox: bool,
) -> list[float]:
    h, w = int(image_shape[0]), int(image_shape[1])
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    if letterbox:
        scale = min(float(imgsz) / max(h, 1), float(imgsz) / max(w, 1))
        nw, nh = int(round(w * scale)), int(round(h * scale))
        dx = float((int(imgsz) - nw) // 2)
        dy = float((int(imgsz) - nh) // 2)
        return [x1 * scale + dx, y1 * scale + dy, x2 * scale + dx, y2 * scale + dy]
    sx = float(imgsz) / max(w, 1)
    sy = float(imgsz) / max(h, 1)
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


def _build_semantic_fp_regions(
    cfg: ODAScoreCalibrationRepairConfig,
    rows: Sequence[Mapping[str, Any]],
    target_ids: Sequence[int],
    guard_names: Sequence[str],
    output_dir: Path,
) -> dict[str, list[list[float]]]:
    """Record model-predicted target FP boxes for semantic target-absent rows.

    These regions are in the repair dataloader coordinate system, so the loss
    can suppress the exact raw candidates that survive as final semantic FPs.
    """
    if float(cfg.lambda_semantic_fp_region) <= 0:
        return {}
    adapter = UltralyticsYOLOAdapter(
        cfg.model,
        device=cfg.device,
        default_conf=float(cfg.conf),
        default_iou=0.7,
        default_imgsz=int(cfg.imgsz),
    )
    target_set = {int(x) for x in target_ids}
    selected_guard_names = list(guard_names)
    region_map: dict[str, list[list[float]]] = {}
    metadata: list[dict[str, Any]] = []
    for row in rows:
        attack = str(row.get("attack") or "")
        goal = infer_attack_goal(str(row.get("goal") or attack))
        if goal != "semantic" or not _truthy(row.get("success")):
            continue
        if selected_guard_names and not _matches_attack_name(attack, selected_guard_names):
            continue
        if _truthy(row.get("has_gt_target")) or int(float(row.get("n_gt_target") or 0)) > 0:
            continue
        image = row.get("image")
        if not image:
            continue
        image_path = Path(str(image))
        try:
            img = read_image_bgr(image_path)
        except Exception:
            continue
        dets = adapter.predict_image(image_path, conf=float(cfg.conf), imgsz=int(cfg.imgsz))
        regions = [
            _map_xyxy_to_train_space(det.xyxy, img.shape[:2], int(cfg.imgsz), letterbox=bool(cfg.letterbox_train))
            for det in dets
            if int(det.cls_id) in target_set
        ]
        if not regions:
            continue
        keys = {
            str(image_path),
            str(image_path.resolve()) if image_path.exists() else str(image_path),
            image_path.name,
            image_path.stem,
        }
        for key in keys:
            region_map.setdefault(key, []).extend(regions)
        metadata.append(
            {
                "attack": attack,
                "image": str(image_path),
                "n_regions": len(regions),
                "regions": regions,
                "max_target_conf": row.get("max_target_conf"),
            }
        )
    write_json(
        output_dir / "semantic_fp_regions.json",
        {"n_images": len(metadata), "n_keys": len(region_map), "rows": metadata},
    )
    return region_map


def run_oda_score_calibration_repair(cfg: ODAScoreCalibrationRepairConfig) -> dict[str, Any]:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "oda_score_calibration_config.json", asdict(cfg))

    target_ids = _target_ids_from_names(cfg.data_yaml, cfg.target_classes)
    if not target_ids:
        raise ValueError("At least one target class is required.")

    # Discover selected attacks through a baseline external hard-suite run.
    eval_cfg = ExternalHardSuiteConfig(
        roots=tuple(cfg.external_roots),
        conf=float(cfg.conf),
        imgsz=int(cfg.imgsz),
        max_images_per_attack=int(cfg.max_images_per_attack),
        seed=int(cfg.seed),
    )
    before = run_external_hard_suite_for_yolo(
        cfg.model,
        data_yaml=cfg.data_yaml,
        target_classes=cfg.target_classes,
        cfg=eval_cfg,
        device=cfg.device,
    )
    before_json, before_csv = write_external_hard_suite_outputs(before, out_dir / "eval_00_before_external")
    baseline_external_score = _external_score(before)
    attack_names = _select_attack_names(
        [str(row.get("attack")) for row in before.get("rows", []) if row.get("attack")],
        cfg.attack_names,
        goal="oda",
    )
    if not attack_names:
        raise ValueError("No ODA attacks selected; pass --attack-names badnet_oda.")

    before_diag = diagnose_oda_candidates(
        ODACandidateDiagnosticConfig(
            model=cfg.model,
            data_yaml=cfg.data_yaml,
            out_dir=str(out_dir / "diag_00_before"),
            target_classes=tuple(cfg.target_classes),
            attack_names=tuple(attack_names),
            rows_csv=str(before_csv),
            device=cfg.device,
            imgsz=int(cfg.imgsz),
            conf=float(cfg.conf),
            low_conf=float(cfg.low_conf),
            max_images_per_attack=int(cfg.max_images_per_attack),
        )
    )
    baseline_diag_score = _diag_score(before_diag.get("summary") or {})

    # Reuse the already tested failure-only dataset builder. The config is duck
    # typed, so this dataclass intentionally exposes the same fields it needs.
    repair_yaml, replay_stats, clean_stats, failure_rows = _build_failure_dataset(
        cfg,
        out_dir,
        target_ids,
        attack_names,
        before.get("rows") or [],
    )
    attack_datasets = discover_external_attack_datasets(cfg.external_roots)
    guard_names = list(cfg.guard_attack_names) or [
        ds.name
        for ds in attack_datasets
        if infer_attack_goal(ds.name if ds.goal == "auto" else ds.goal) in {"oga", "semantic"}
    ]
    semantic_fp_regions = _build_semantic_fp_regions(cfg, before.get("rows") or [], target_ids, guard_names, out_dir)

    guard_stats: dict[str, Any] = {"added": 0}
    semantic_fp_replay_stats: dict[str, Any] = {"added": 0}
    if (
        (
            float(cfg.lambda_oga_negative) > 0
            or float(cfg.lambda_semantic_negative) > 0
            or float(cfg.lambda_semantic_fp_region) > 0
        )
        and int(cfg.guard_replay_max_images_per_attack) > 0
    ):
        if guard_names:
            guard_stats = append_external_replay_samples(
                output_dataset_dir=out_dir / "01_postnms_failure_dataset",
                attack_datasets=attack_datasets,
                target_class_ids=target_ids,
                selected_attack_names=guard_names,
                max_images_per_attack=int(cfg.guard_replay_max_images_per_attack),
                split="train",
                seed=int(cfg.seed) + 17,
                failure_rows=before.get("rows") or [],
                failure_only=bool(cfg.guard_failure_only),
                repeat=int(cfg.guard_repeat),
            )
    if (
        float(cfg.lambda_semantic_fp_region) > 0
        and semantic_fp_regions
        and int(cfg.guard_replay_max_images_per_attack) > 0
    ):
        semantic_guard_names = [
            name
            for name in guard_names
            if infer_attack_goal(str(name)) == "semantic" or any(keyword.lower() in str(name).lower() for keyword in cfg.semantic_guard_keywords)
        ]
        if semantic_guard_names:
            semantic_fp_replay_stats = append_external_replay_samples(
                output_dataset_dir=out_dir / "01_postnms_failure_dataset",
                attack_datasets=attack_datasets,
                target_class_ids=target_ids,
                selected_attack_names=semantic_guard_names,
                max_images_per_attack=int(cfg.guard_replay_max_images_per_attack),
                split="train",
                seed=int(cfg.seed) + 29,
                failure_rows=before.get("rows") or [],
                failure_only=True,
                repeat=max(1, int(cfg.guard_repeat)),
            )

    device = _device_from_string(cfg.device)
    yolo = load_ultralytics_yolo(cfg.model, device)
    student = _torch_model(yolo).to(device)
    student.train()

    teacher = None
    if cfg.teacher_model:
        teacher_yolo = load_ultralytics_yolo(cfg.teacher_model, device)
        teacher = _torch_model(teacher_yolo).to(device).eval()
        for param in teacher.parameters():
            param.requires_grad_(False)

    loader, _info = make_yolo_dataloader(
        repair_yaml,
        split="train",
        imgsz=int(cfg.imgsz),
        batch_size=int(cfg.batch),
        shuffle=True,
        num_workers=0,
        max_images=None,
        letterbox=bool(cfg.letterbox_train),
    )
    optimizer = torch.optim.AdamW(student.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.amp and device.type == "cuda"))

    log_path = out_dir / "oda_score_calibration_train_log.csv"
    fields = [
        "epoch",
        "step",
        "loss_total",
        "loss_score_calibration",
        "loss_task",
        "loss_oga",
        "loss_semantic",
        "loss_semantic_fp_region",
    ]
    with log_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    candidate_rows: list[dict[str, Any]] = []
    global_step = 0
    for epoch in range(1, int(cfg.epochs) + 1):
        student.train()
        for batch in loader:
            global_step += 1
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(cfg.amp and device.type == "cuda")):
                pred = _decoded_forward(student, batch["img"])
                with torch.no_grad():
                    teacher_pred = _decoded_forward(teacher, batch["img"]) if teacher is not None else None
                loss_score = oda_score_calibration_loss(
                    pred,
                    batch,
                    target_ids,
                    teacher_prediction=teacher_pred,
                    iou_threshold=float(cfg.score_iou_threshold),
                    center_radius=float(cfg.score_center_radius),
                    topk_near=int(cfg.score_topk_near),
                    topk_far=int(cfg.score_topk_far),
                    conf_target=float(cfg.score_conf_target),
                    score_margin=float(cfg.score_margin),
                    positive_bce_weight=float(cfg.score_positive_bce_weight),
                    score_floor_weight=float(cfg.score_floor_weight),
                    far_margin_weight=float(cfg.score_far_margin_weight),
                    competing_margin_weight=float(cfg.score_competing_margin_weight),
                    teacher_score_weight=float(cfg.score_teacher_weight),
                ) * float(cfg.lambda_score_calibration)
                loss_task = supervised_yolo_loss(student, batch) * float(cfg.lambda_task) if cfg.lambda_task > 0 else loss_score * 0.0
                loss_oga = (
                    negative_target_candidate_suppression_loss(pred, batch, target_ids, topk=256, weight=1.0) * float(cfg.lambda_oga_negative)
                    if cfg.lambda_oga_negative > 0
                    else loss_score * 0.0
                )
                loss_semantic = (
                    semantic_negative_guard_loss(
                        pred,
                        batch,
                        target_ids,
                        semantic_keywords=tuple(cfg.semantic_guard_keywords),
                        topk=int(cfg.semantic_negative_topk),
                        max_target_score=float(cfg.semantic_negative_max_score),
                        margin_weight=float(cfg.semantic_negative_margin_weight),
                    )
                    * float(cfg.lambda_semantic_negative)
                    if cfg.lambda_semantic_negative > 0
                    else loss_score * 0.0
                )
                loss_semantic_fp = (
                    semantic_fp_region_guard_loss(
                        pred,
                        batch,
                        target_ids,
                        semantic_fp_regions,
                        topk=int(cfg.semantic_fp_region_topk),
                        max_target_score=float(cfg.semantic_fp_region_max_score),
                        iou_threshold=float(cfg.semantic_fp_region_iou_threshold),
                        center_radius=float(cfg.semantic_fp_region_center_radius),
                        margin_weight=float(cfg.semantic_fp_region_margin_weight),
                    )
                    * float(cfg.lambda_semantic_fp_region)
                    if cfg.lambda_semantic_fp_region > 0 and semantic_fp_regions
                    else loss_score * 0.0
                )
                loss_total = loss_score + loss_task + loss_oga + loss_semantic + loss_semantic_fp
            scaler.scale(loss_total).backward()
            if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), float(cfg.grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
            with log_path.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(
                    {
                        "epoch": epoch,
                        "step": global_step,
                        "loss_total": float(loss_total.detach().cpu().item()),
                        "loss_score_calibration": float(loss_score.detach().cpu().item()),
                        "loss_task": float(loss_task.detach().cpu().item()),
                        "loss_oga": float(loss_oga.detach().cpu().item()),
                        "loss_semantic": float(loss_semantic.detach().cpu().item()),
                        "loss_semantic_fp_region": float(loss_semantic_fp.detach().cpu().item()),
                    }
                )

        ckpt = out_dir / "02_score_calibration_checkpoints" / f"epoch_{epoch}.pt"
        save_ultralytics_yolo(yolo, ckpt)
        result = run_external_hard_suite_for_yolo(
            str(ckpt),
            data_yaml=cfg.data_yaml,
            target_classes=cfg.target_classes,
            cfg=eval_cfg,
            device=cfg.device,
        )
        eval_dir = out_dir / "03_candidate_external" / f"epoch_{epoch:03d}"
        cand_json, cand_csv = write_external_hard_suite_outputs(result, eval_dir)
        diag = diagnose_oda_candidates(
            ODACandidateDiagnosticConfig(
                model=str(ckpt),
                data_yaml=cfg.data_yaml,
                out_dir=str(out_dir / "04_candidate_diagnostics" / f"epoch_{epoch:03d}"),
                target_classes=tuple(cfg.target_classes),
                attack_names=tuple(attack_names),
                rows_csv=str(cand_csv),
                device=cfg.device,
                imgsz=int(cfg.imgsz),
                conf=float(cfg.conf),
                low_conf=float(cfg.low_conf),
                max_images_per_attack=int(cfg.max_images_per_attack),
            )
        )
        diag_summary = diag.get("summary") or {}
        blocked = _blocked_by_worsening(result, before, float(cfg.max_single_attack_worsen))
        summary = result.get("summary") or {}
        row = {
            "epoch": epoch,
            "model": str(ckpt),
            "external_json": str(cand_json),
            "external_rows_csv": str(cand_csv),
            "diagnostic_json": str(out_dir / "04_candidate_diagnostics" / f"epoch_{epoch:03d}" / "oda_candidate_diagnostics.json"),
            "external_max_asr": float(summary.get("max_asr", 1.0)),
            "external_mean_asr": float(summary.get("mean_asr", 1.0)),
            "external_score": _external_score(result),
            "diagnostic_score": _diag_score(diag_summary),
            "raw_near_gt_over_conf_rate": float(diag_summary.get("raw_near_gt_over_conf_rate") or 0.0),
            "raw_near_gt_best_target_score_mean": float(diag_summary.get("raw_near_gt_best_target_score_mean") or 0.0),
            "lowconf_recalled_rate": float(diag_summary.get("lowconf_recalled_rate") or 0.0),
            "blocked_attacks": blocked,
            "accepted": (not blocked) and float(summary.get("max_asr", 1.0)) <= float(cfg.max_allowed_external_asr),
        }
        candidate_rows.append(row)
        write_json(out_dir / "oda_score_calibration_repair_manifest.json", {"status": "running", "candidate_rows": candidate_rows})

    selection = _select_calibration_candidate(
        candidate_rows,
        baseline_external_score=baseline_external_score,
        baseline_diag_score=baseline_diag_score,
        fallback_model=cfg.model,
        min_external_improvement=float(cfg.min_external_score_improvement),
        min_diag_improvement=float(cfg.min_diag_score_improvement),
        require_external_improvement=bool(cfg.require_external_improvement_for_final),
    )
    final_row = selection["best"]
    manifest = {
        "status": "passed" if final_row and final_row.get("accepted") else "failed_external_asr_or_worsening",
        "final_model": selection["final_model"],
        "rolled_back": bool(selection["rolled_back"]),
        "input_model": cfg.model,
        "target_class_ids": target_ids,
        "selected_attacks": attack_names,
        "before_external_json": str(before_json),
        "before_rows_csv": str(before_csv),
        "before_summary": before.get("summary"),
        "before_external_score": baseline_external_score,
        "before_diagnostic_summary": before_diag.get("summary"),
        "before_diagnostic_score": baseline_diag_score,
        "repair_data_yaml": str(repair_yaml),
        "replay_stats": replay_stats,
        "clean_anchor_stats": clean_stats,
        "guard_stats": guard_stats,
        "semantic_fp_replay_stats": semantic_fp_replay_stats,
        "semantic_fp_region_stats": {
            "n_keys": len(semantic_fp_regions),
            "n_regions": sum(len(v) for v in semantic_fp_regions.values()),
            "json": str(out_dir / "semantic_fp_regions.json"),
        },
        "n_failure_rows": len(failure_rows),
        "log_csv": str(log_path),
        "candidate_rows": candidate_rows,
        "best": final_row,
        "best_by_external": selection["best_by_external"],
        "best_by_diagnostic": selection["best_by_diagnostic"],
    }
    write_json(out_dir / "oda_score_calibration_repair_manifest.json", manifest)
    return manifest

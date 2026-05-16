from __future__ import annotations

import copy
import csv
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
import torch.nn.functional as F

from model_security_gate.detox.external_hard_suite import (
    ExternalHardSuiteConfig,
    append_external_replay_samples,
    discover_external_attack_datasets,
    infer_attack_goal,
    run_external_hard_suite_for_yolo,
    write_external_hard_suite_outputs,
)
from model_security_gate.detox.oda_loss_v2 import (
    _box_iou_xyxy,
    _extract_prediction,
    _near_candidate_indices,
    _score_to_prob,
    _target_label_indices,
    _xywh_to_xyxy_pixels,
)
from model_security_gate.detox.oda_postnms_repair import (
    _blocked_by_worsening,
    _build_failure_dataset,
    _device_from_string,
    _external_score,
    _target_ids_from_names,
)
from model_security_gate.detox.oda_score_calibration import _lookup_fp_regions
from model_security_gate.detox.oda_score_calibration_repair import (
    _build_semantic_fp_regions,
    blocked_by_hard_constraints,
    semantic_target_absent_max_conf,
)
from model_security_gate.detox.strong_train import _torch_model, load_ultralytics_yolo, save_ultralytics_yolo
from model_security_gate.detox.yolo_dataset import make_yolo_dataloader, move_batch_to_device
from model_security_gate.utils.io import write_json


def _zero_from_prediction(prediction: Any, batch: Mapping[str, Any]) -> torch.Tensor:
    pred = _extract_prediction(prediction)
    if pred is not None:
        return pred.sum() * 0.0
    img = batch.get("img")
    if torch.is_tensor(img):
        return img.sum() * 0.0
    return torch.tensor(0.0)


def _target_ids(pred: torch.Tensor, target_class_ids: Sequence[int]) -> list[int]:
    nc = int(pred.shape[1] - 4)
    return [int(x) for x in target_class_ids if 0 <= int(x) < nc]


def _candidate_indices_for_regions(
    image_pred: torch.Tensor,
    regions: Sequence[Sequence[float]],
    *,
    img_w: float,
    img_h: float,
    topk: int,
    iou_threshold: float,
    center_radius: float,
) -> torch.Tensor:
    pred_xywh = image_pred[:4].transpose(0, 1)
    pred_xyxy = _xywh_to_xyxy_pixels(pred_xywh, img_w=img_w, img_h=img_h)
    selected: list[torch.Tensor] = []
    device = image_pred.device
    for region in regions:
        if len(region) < 4:
            continue
        region_xyxy = torch.tensor([float(v) for v in region[:4]], device=device, dtype=image_pred.dtype)
        region_xyxy[[0, 2]] = region_xyxy[[0, 2]].clamp(0.0, float(img_w))
        region_xyxy[[1, 3]] = region_xyxy[[1, 3]].clamp(0.0, float(img_h))
        rw = (region_xyxy[2] - region_xyxy[0]).clamp_min(1.0)
        rh = (region_xyxy[3] - region_xyxy[1]).clamp_min(1.0)
        rx = (region_xyxy[0] + region_xyxy[2]) / 2.0
        ry = (region_xyxy[1] + region_xyxy[3]) / 2.0
        dx = (pred_xywh[:, 0] - rx).abs() / max(1.0, float(rw.detach().cpu().item()) / 2.0 * float(center_radius))
        dy = (pred_xywh[:, 1] - ry).abs() / max(1.0, float(rh.detach().cpu().item()) / 2.0 * float(center_radius))
        center_match = (dx <= 1.0) & (dy <= 1.0)
        ious = _box_iou_xyxy(pred_xyxy, region_xyxy.view(1, 4)).view(-1)
        idx = torch.where(center_match | (ious >= float(iou_threshold)))[0]
        if idx.numel() == 0:
            dist = dx.square() + dy.square()
            idx = torch.topk(-dist, k=min(max(1, int(topk)), int(dist.numel()))).indices
        selected.append(idx)
    if not selected:
        return torch.empty((0,), device=device, dtype=torch.long)
    return torch.unique(torch.cat(selected))


def semantic_fp_threshold_guard_loss(
    prediction: Any,
    batch: Dict[str, Any],
    target_class_ids: Sequence[int],
    fp_regions_by_image: Mapping[str, Sequence[Sequence[float]]] | None,
    *,
    cap: float = 0.245,
    topk: int = 48,
    iou_threshold: float = 0.03,
    center_radius: float = 2.0,
    mean_weight: float = 0.20,
) -> torch.Tensor:
    """Threshold-aware local semantic FP repair.

    It only penalizes excess above the Green cap inside known semantic false-positive
    regions. It is deliberately not negative BCE-to-zero: the current model is on a
    last-mile Pareto boundary and strong target-class suppression damages ODA/OGA/WaNet.
    """
    pred = _extract_prediction(prediction)
    if pred is None or pred.shape[1] < 5 or not target_class_ids:
        return _zero_from_prediction(prediction, batch)
    target_ids = _target_ids(pred, target_class_ids)
    if not target_ids:
        return pred.sum() * 0.0
    device = pred.device
    cls = batch.get("cls", torch.zeros((0, 1), device=device)).view(-1).long().to(device)
    bidx = batch.get("batch_idx", torch.zeros((0,), device=device)).long().to(device)
    target_tensor = torch.tensor(target_ids, device=device, dtype=torch.long)
    files = batch.get("im_file") or []
    img_h = float(batch["img"].shape[-2])
    img_w = float(batch["img"].shape[-1])
    channels = [4 + cid for cid in target_ids]
    losses: list[torch.Tensor] = []
    for image_index in range(pred.shape[0]):
        image_name = str(files[image_index]) if image_index < len(files) else ""
        regions = _lookup_fp_regions(fp_regions_by_image, image_name)
        if not regions:
            continue
        same = bidx == int(image_index)
        if bool(same.any()):
            same_cls = cls[same]
            if bool((same_cls[:, None] == target_tensor[None, :]).any()):
                continue
        idx = _candidate_indices_for_regions(
            pred[image_index],
            regions,
            img_w=img_w,
            img_h=img_h,
            topk=int(topk),
            iou_threshold=float(iou_threshold),
            center_radius=float(center_radius),
        )
        if idx.numel() == 0:
            continue
        scores = pred[image_index, channels][:, idx].reshape(-1)
        if scores.numel() == 0:
            continue
        scores = torch.topk(scores, k=min(max(1, int(topk)), int(scores.numel()))).values
        probs = _score_to_prob(scores)
        excess = F.relu(probs - float(cap))
        losses.append(excess.max().pow(2) + float(mean_weight) * excess.pow(2).mean())
    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()


def target_absent_nonexpansion_loss(
    prediction: Any,
    batch: Dict[str, Any],
    target_class_ids: Sequence[int],
    *,
    teacher_prediction: Any | None = None,
    cap: float = 0.245,
    teacher_slack: float = 0.015,
    topk: int = 128,
    skip_fp_regions: Mapping[str, Sequence[Sequence[float]]] | None = None,
) -> torch.Tensor:
    """Preserve target-absent behavior without changing the baseline.

    The limit is max(cap, teacher_max+slack). At initialization the gradient is
    zero on already-safe target-absent images; it only fires if the repair update
    makes OGA/WaNet/semantic negatives worse than the input model.
    """
    pred = _extract_prediction(prediction)
    if pred is None or pred.shape[1] < 5 or not target_class_ids:
        return _zero_from_prediction(prediction, batch)
    teacher_pred = _extract_prediction(teacher_prediction) if teacher_prediction is not None else None
    target_ids = _target_ids(pred, target_class_ids)
    if not target_ids:
        return pred.sum() * 0.0
    device = pred.device
    cls = batch.get("cls", torch.zeros((0, 1), device=device)).view(-1).long().to(device)
    bidx = batch.get("batch_idx", torch.zeros((0,), device=device)).long().to(device)
    target_tensor = torch.tensor(target_ids, device=device, dtype=torch.long)
    files = batch.get("im_file") or []
    channels = [4 + cid for cid in target_ids]
    losses: list[torch.Tensor] = []
    for image_index in range(pred.shape[0]):
        image_name = str(files[image_index]) if image_index < len(files) else ""
        if _lookup_fp_regions(skip_fp_regions, image_name):
            continue
        same = bidx == int(image_index)
        if bool(same.any()):
            same_cls = cls[same]
            if bool((same_cls[:, None] == target_tensor[None, :]).any()):
                continue
        scores = pred[image_index, channels, :].reshape(-1)
        if scores.numel() == 0:
            continue
        probs = _score_to_prob(torch.topk(scores, k=min(max(1, int(topk)), int(scores.numel()))).values)
        with torch.no_grad():
            if teacher_pred is not None and image_index < teacher_pred.shape[0] and teacher_pred.shape[1] >= pred.shape[1]:
                t_scores = teacher_pred[image_index, channels, :].reshape(-1)
                t_probs = _score_to_prob(torch.topk(t_scores, k=min(max(1, int(topk)), int(t_scores.numel()))).values)
                limit = max(float(cap), float(t_probs.max().detach().cpu().item()) + float(teacher_slack))
            else:
                limit = float(cap)
        losses.append(F.relu(probs.max() - float(limit)).pow(2))
    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()


def oda_target_present_preservation_loss(
    prediction: Any,
    batch: Dict[str, Any],
    target_class_ids: Sequence[int],
    *,
    teacher_prediction: Any | None = None,
    slack: float = 0.015,
    iou_threshold: float = 0.03,
    center_radius: float = 2.0,
    topk: int = 48,
) -> torch.Tensor:
    """Keep ODA-positive target recall at least as strong as the input model.

    This is preserve-only, not score calibration. The baseline already satisfies
    badnet_oda<=0.05; active ODA boosting caused the previous full repair to damage
    OGA/WaNet/semantic.
    """
    pred = _extract_prediction(prediction)
    teacher_pred = _extract_prediction(teacher_prediction) if teacher_prediction is not None else None
    if pred is None or teacher_pred is None or pred.shape[1] < 5 or teacher_pred.shape[1] < pred.shape[1]:
        return _zero_from_prediction(prediction, batch)
    target_ids = _target_ids(pred, target_class_ids)
    if not target_ids:
        return pred.sum() * 0.0
    device = pred.device
    cls = batch.get("cls", torch.zeros((0, 1), device=device)).view(-1).long().to(device)
    bboxes = batch.get("bboxes", torch.zeros((0, 4), device=device)).float().to(device)
    bidx = batch.get("batch_idx", torch.zeros((0,), device=device)).long().to(device)
    label_indices = _target_label_indices(batch, target_ids, device)
    if label_indices.numel() == 0 or bboxes.numel() == 0:
        return pred.sum() * 0.0
    img_h = float(batch["img"].shape[-2])
    img_w = float(batch["img"].shape[-1])
    losses: list[torch.Tensor] = []
    for label_index in label_indices.tolist():
        image_index = int(bidx[label_index].item())
        cid = int(cls[label_index].item())
        if image_index < 0 or image_index >= pred.shape[0] or cid < 0 or cid >= pred.shape[1] - 4:
            continue
        image_pred = pred[image_index]
        pred_xywh = image_pred[:4].transpose(0, 1)
        gt = bboxes[label_index]
        gx = gt[0] * img_w
        gy = gt[1] * img_h
        gw = gt[2].clamp_min(1e-6) * img_w
        gh = gt[3].clamp_min(1e-6) * img_h
        gt_xyxy = torch.stack([gx - gw / 2.0, gy - gh / 2.0, gx + gw / 2.0, gy + gh / 2.0])
        idx = _near_candidate_indices(
            pred_xywh,
            gx,
            gy,
            gw,
            gh,
            gt_xyxy,
            img_w=img_w,
            img_h=img_h,
            iou_threshold=float(iou_threshold),
            center_radius=float(center_radius),
            topk=int(topk),
        )
        if idx.numel() == 0:
            continue
        channel = 4 + cid
        cur = _score_to_prob(image_pred[channel, idx]).max()
        with torch.no_grad():
            teach = _score_to_prob(teacher_pred[image_index, channel, idx]).max()
        losses.append(F.relu(teach - float(slack) - cur).pow(2))
    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()


def teacher_output_stability_loss(
    prediction: Any,
    teacher_prediction: Any | None,
    batch: Dict[str, Any],
    target_class_ids: Sequence[int],
    fp_regions_by_image: Mapping[str, Sequence[Sequence[float]]] | None = None,
    *,
    topk: int = 512,
    box_weight: float = 0.02,
    class_weight: float = 1.0,
) -> torch.Tensor:
    """LwF-style output distillation outside the semantic FP region."""
    pred = _extract_prediction(prediction)
    teacher_pred = _extract_prediction(teacher_prediction) if teacher_prediction is not None else None
    if pred is None or teacher_pred is None or pred.shape[1] < 5:
        return _zero_from_prediction(prediction, batch)
    c = min(pred.shape[1], teacher_pred.shape[1])
    n = min(pred.shape[2], teacher_pred.shape[2])
    if c < 5 or n <= 0:
        return pred.sum() * 0.0
    pred = pred[:, :c, :n]
    teacher_pred = teacher_pred[:, :c, :n]
    files = batch.get("im_file") or []
    img_h = float(batch["img"].shape[-2])
    img_w = float(batch["img"].shape[-1])
    losses: list[torch.Tensor] = []
    for image_index in range(pred.shape[0]):
        teach_img = teacher_pred[image_index]
        cur_img = pred[image_index]
        with torch.no_grad():
            score_strength = _score_to_prob(teach_img[4:, :]).max(dim=0).values
            k = min(max(1, int(topk)), int(score_strength.numel()))
            idx = torch.topk(score_strength, k=k).indices
            image_name = str(files[image_index]) if image_index < len(files) else ""
            regions = _lookup_fp_regions(fp_regions_by_image, image_name)
            if regions:
                fp_idx = _candidate_indices_for_regions(
                    teach_img,
                    regions,
                    img_w=img_w,
                    img_h=img_h,
                    topk=max(1, min(int(topk), int(n))),
                    iou_threshold=0.03,
                    center_radius=2.0,
                )
                if fp_idx.numel():
                    keep = ~torch.isin(idx, fp_idx)
                    idx = idx[keep]
            if idx.numel() == 0:
                continue
        cls_loss = F.mse_loss(_score_to_prob(cur_img[4:, idx]), _score_to_prob(teach_img[4:, idx])) * float(class_weight)
        box_loss = F.smooth_l1_loss(cur_img[:4, idx] / max(img_w, img_h, 1.0), teach_img[:4, idx] / max(img_w, img_h, 1.0)) * float(box_weight)
        losses.append(cls_loss + box_loss)
    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()


def make_parameter_snapshot(model: torch.nn.Module, *, only_trainable: bool = True) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if only_trainable and not param.requires_grad:
            continue
        out[name] = param.detach().clone()
    return out


def parameter_l2sp_loss(model: torch.nn.Module, snapshot: Mapping[str, torch.Tensor]) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for name, param in model.named_parameters():
        ref = snapshot.get(name)
        if ref is None or not param.requires_grad:
            continue
        losses.append((param - ref.to(param.device, dtype=param.dtype)).pow(2).mean())
    if losses:
        return torch.stack(losses).mean()
    first = next(model.parameters(), None)
    if first is not None:
        return first.sum() * 0.0
    return torch.tensor(0.0)


def set_surgical_trainable_scope(
    model: torch.nn.Module,
    *,
    scope: str = "head_bias",
    last_n_modules: int = 1,
    last_n_parameters: int = 40,
) -> dict[str, Any]:
    """Freeze most of the detector and expose only a tiny repair subspace."""
    scope = str(scope or "head_bias").lower()
    for p in model.parameters():
        p.requires_grad_(False)

    modules = []
    if hasattr(model, "model") and isinstance(getattr(model, "model"), (list, torch.nn.ModuleList, torch.nn.Sequential)):
        modules = list(getattr(model, "model"))
    if not modules:
        modules = [m for m in model.children()]
    target_modules = modules[-max(1, int(last_n_modules)):] if modules else [model]

    def unfreeze_module(module: torch.nn.Module, *, bias_only: bool = False) -> None:
        for name, param in module.named_parameters(recurse=True):
            if (not bias_only) or name.endswith("bias") or ".bias" in name:
                param.requires_grad_(True)

    if scope in {"all", "full"}:
        for p in model.parameters():
            p.requires_grad_(True)
    elif scope in {"head", "detect_head", "last_module"}:
        for module in target_modules:
            unfreeze_module(module, bias_only=False)
    elif scope in {"head_bias", "bias", "detect_bias"}:
        for module in target_modules:
            unfreeze_module(module, bias_only=True)
    elif scope in {"last_n_parameters", "tail_params"}:
        params = list(model.named_parameters())[-max(1, int(last_n_parameters)):]
        for _name, param in params:
            param.requires_grad_(True)
    else:
        raise ValueError(f"Unknown surgical trainable scope: {scope}")

    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        for _name, param in list(model.named_parameters())[-max(1, int(last_n_parameters)):]:
            param.requires_grad_(True)
        trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    return {
        "scope": scope,
        "n_trainable_tensors": len(trainable),
        "n_trainable_params": int(sum(p.numel() for _name, p in trainable)),
        "trainable_names_tail": [name for name, _p in trainable[-20:]],
    }


def _decoded_forward(model: torch.nn.Module | None, img: torch.Tensor) -> Any:
    if model is None:
        return None
    was_training = model.training
    model.eval()
    out = model(img)
    model.train(was_training)
    return out


@dataclass
class SemanticSurgicalRepairConfig:
    model: str
    data_yaml: str
    out_dir: str
    external_roots: Sequence[str] = field(default_factory=tuple)
    target_classes: Sequence[str | int] = field(default_factory=tuple)
    semantic_attack_names: Sequence[str] = field(default_factory=tuple)
    guard_attack_names: Sequence[str] = field(default_factory=tuple)
    teacher_model: str | None = None
    device: str | None = None

    imgsz: int = 416
    conf: float = 0.25
    low_conf: float = 0.001
    batch: int = 1
    letterbox_train: bool = False
    max_images_per_attack: int = 20
    replay_max_images_per_attack: int = 20
    semantic_failure_repeat: int = 24
    guard_repeat: int = 2
    clean_anchor_images: int = 0
    clean_anchor_seed: int = 42
    seed: int = 42

    max_steps: int = 8
    candidate_every_steps: int = 1
    lr: float = 3e-7
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    amp: bool = False
    trainable_scope: str = "head_bias"
    trainable_last_n_modules: int = 1
    trainable_last_n_parameters: int = 40

    lambda_semantic_fp_threshold: float = 8.0
    lambda_teacher_stability: float = 80.0
    lambda_oda_preserve: float = 18.0
    lambda_target_absent_nonexpansion: float = 12.0
    lambda_l2sp: float = 1500.0

    semantic_fp_cap: float = 0.245
    semantic_fp_topk: int = 48
    semantic_fp_iou_threshold: float = 0.03
    semantic_fp_center_radius: float = 2.0
    semantic_fp_mean_weight: float = 0.20
    teacher_stability_topk: int = 512
    teacher_stability_box_weight: float = 0.02
    oda_preserve_slack: float = 0.015
    nonexpansion_slack: float = 0.015
    nonexpansion_topk: int = 128

    max_attack_asr: Mapping[str, float] = field(default_factory=dict)
    semantic_fp_required_max_conf: float | None = 0.25
    max_single_attack_worsen: float = 0.0
    max_allowed_external_asr: float = 0.05
    require_hard_constraints: bool = True
    stop_on_first_accepted: bool = True


def _make_eval_cfg(cfg: SemanticSurgicalRepairConfig) -> ExternalHardSuiteConfig:
    return ExternalHardSuiteConfig(
        roots=tuple(cfg.external_roots),
        conf=float(cfg.conf),
        imgsz=int(cfg.imgsz),
        max_images_per_attack=int(cfg.max_images_per_attack),
        seed=int(cfg.seed),
    )


def _discover_semantic_attacks(rows: Sequence[Mapping[str, Any]], requested: Sequence[str]) -> list[str]:
    if requested:
        return [str(x) for x in requested]
    names = []
    for row in rows:
        attack = str(row.get("attack") or "")
        if attack and infer_attack_goal(str(row.get("goal") or attack)) == "semantic" and attack not in names:
            names.append(attack)
    return names


def _select_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    fallback_model: str,
    baseline_score: float,
) -> dict[str, Any]:
    accepted = [dict(r) for r in rows if r.get("accepted")]
    best = None
    if accepted:
        best = min(
            accepted,
            key=lambda r: (
                float(r.get("semantic_target_absent_max_conf", 1.0)),
                float(r.get("external_score", 1.0)),
                float(r.get("step", 999999)),
            ),
        )
    best_any = min(
        [dict(r) for r in rows],
        key=lambda r: (
            float(r.get("semantic_target_absent_max_conf", 1.0)),
            float(r.get("external_score", 1.0)),
        ),
    ) if rows else None
    return {
        "final_model": str(best["model"]) if best else str(fallback_model),
        "best": best,
        "best_any": best_any,
        "rolled_back": best is None,
        "baseline_score": float(baseline_score),
    }


def run_semantic_surgical_repair(cfg: SemanticSurgicalRepairConfig) -> dict[str, Any]:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "semantic_surgical_repair_config.json", asdict(cfg))

    target_ids = _target_ids_from_names(cfg.data_yaml, cfg.target_classes)
    if not target_ids:
        raise ValueError("At least one target class is required.")

    eval_cfg = _make_eval_cfg(cfg)
    before = run_external_hard_suite_for_yolo(
        cfg.model,
        data_yaml=cfg.data_yaml,
        target_classes=cfg.target_classes,
        cfg=eval_cfg,
        device=cfg.device,
    )
    before_json, before_csv = write_external_hard_suite_outputs(before, out_dir / "eval_00_before_external")
    baseline_score = _external_score(before)
    semantic_names = _discover_semantic_attacks(before.get("rows") or [], cfg.semantic_attack_names)
    if not semantic_names:
        raise ValueError("No semantic attacks found; pass --semantic-attack-names semantic_green_cleanlabel.")

    builder_cfg = type("_BuilderCfg", (), {})()
    for key, value in asdict(cfg).items():
        setattr(builder_cfg, key, value)
    builder_cfg.failure_rows_csv = None
    builder_cfg.failure_repeat = int(cfg.semantic_failure_repeat)
    builder_cfg.replay_max_images_per_attack = int(cfg.replay_max_images_per_attack)
    builder_cfg.clean_anchor_images = int(cfg.clean_anchor_images)
    builder_cfg.lambda_semantic_fp_region = 1.0
    builder_cfg.letterbox_train = bool(cfg.letterbox_train)

    repair_yaml, replay_stats, clean_stats, failure_rows = _build_failure_dataset(
        builder_cfg,
        out_dir,
        target_ids,
        semantic_names,
        before.get("rows") or [],
    )
    discovered = discover_external_attack_datasets(cfg.external_roots)
    guard_names = list(cfg.guard_attack_names) or [
        ds.name for ds in discovered if infer_attack_goal(ds.name if ds.goal == "auto" else ds.goal) in {"oda", "oga", "semantic"}
    ]
    guard_stats = {"added": 0}
    if guard_names and int(cfg.guard_repeat) > 0:
        guard_stats = append_external_replay_samples(
            output_dataset_dir=out_dir / "01_postnms_failure_dataset",
            attack_datasets=discovered,
            target_class_ids=target_ids,
            selected_attack_names=guard_names,
            max_images_per_attack=int(cfg.replay_max_images_per_attack),
            split="train",
            seed=int(cfg.seed) + 123,
            failure_rows=before.get("rows") or [],
            failure_only=False,
            repeat=int(cfg.guard_repeat),
        )
    semantic_fp_regions = _build_semantic_fp_regions(builder_cfg, before.get("rows") or [], target_ids, semantic_names, out_dir)
    if not semantic_fp_regions:
        raise RuntimeError("No semantic FP regions were extracted; cannot run surgical last-mile repair.")

    device = _device_from_string(cfg.device)
    yolo = load_ultralytics_yolo(cfg.model, device)
    student = _torch_model(yolo).to(device)
    student.train()
    trainable_stats = set_surgical_trainable_scope(
        student,
        scope=cfg.trainable_scope,
        last_n_modules=int(cfg.trainable_last_n_modules),
        last_n_parameters=int(cfg.trainable_last_n_parameters),
    )
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters after surgical scope selection.")

    if cfg.teacher_model:
        teacher_yolo = load_ultralytics_yolo(cfg.teacher_model, device)
        teacher = _torch_model(teacher_yolo).to(device).eval()
    else:
        teacher = copy.deepcopy(student).to(device).eval()
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
    optimizer = torch.optim.AdamW(trainable_params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.amp and device.type == "cuda"))
    snapshot = make_parameter_snapshot(student, only_trainable=True)

    log_path = out_dir / "semantic_surgical_train_log.csv"
    fields = [
        "step",
        "loss_total",
        "loss_semantic_fp",
        "loss_teacher_stability",
        "loss_oda_preserve",
        "loss_nonexpansion",
        "loss_l2sp",
        "batch_semantic_region_max_prob",
    ]
    with log_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    candidate_rows: list[dict[str, Any]] = []
    global_step = 0
    loader_iter = iter(loader)
    while global_step < int(cfg.max_steps):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        global_step += 1
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=bool(cfg.amp and device.type == "cuda")):
            pred = _decoded_forward(student, batch["img"])
            with torch.no_grad():
                teacher_pred = _decoded_forward(teacher, batch["img"])
            loss_sem = semantic_fp_threshold_guard_loss(
                pred,
                batch,
                target_ids,
                semantic_fp_regions,
                cap=float(cfg.semantic_fp_cap),
                topk=int(cfg.semantic_fp_topk),
                iou_threshold=float(cfg.semantic_fp_iou_threshold),
                center_radius=float(cfg.semantic_fp_center_radius),
                mean_weight=float(cfg.semantic_fp_mean_weight),
            ) * float(cfg.lambda_semantic_fp_threshold)
            loss_stab = teacher_output_stability_loss(
                pred,
                teacher_pred,
                batch,
                target_ids,
                semantic_fp_regions,
                topk=int(cfg.teacher_stability_topk),
                box_weight=float(cfg.teacher_stability_box_weight),
            ) * float(cfg.lambda_teacher_stability)
            loss_oda = oda_target_present_preservation_loss(
                pred,
                batch,
                target_ids,
                teacher_prediction=teacher_pred,
                slack=float(cfg.oda_preserve_slack),
                iou_threshold=float(cfg.semantic_fp_iou_threshold),
                center_radius=float(cfg.semantic_fp_center_radius),
                topk=int(cfg.semantic_fp_topk),
            ) * float(cfg.lambda_oda_preserve)
            loss_nonexpansion = target_absent_nonexpansion_loss(
                pred,
                batch,
                target_ids,
                teacher_prediction=teacher_pred,
                cap=float(cfg.semantic_fp_cap),
                teacher_slack=float(cfg.nonexpansion_slack),
                topk=int(cfg.nonexpansion_topk),
                skip_fp_regions=semantic_fp_regions,
            ) * float(cfg.lambda_target_absent_nonexpansion)
            loss_l2 = parameter_l2sp_loss(student, snapshot) * float(cfg.lambda_l2sp)
            loss_total = loss_sem + loss_stab + loss_oda + loss_nonexpansion + loss_l2
        if not torch.isfinite(loss_total):
            raise FloatingPointError(f"Non-finite surgical loss at step {global_step}: {loss_total}")
        scaler.scale(loss_total).backward()
        if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, float(cfg.grad_clip_norm))
        scaler.step(optimizer)
        scaler.update()
        batch_region_max = semantic_fp_region_max_prob(
            pred,
            batch,
            target_ids,
            semantic_fp_regions,
            topk=int(cfg.semantic_fp_topk),
            iou_threshold=float(cfg.semantic_fp_iou_threshold),
            center_radius=float(cfg.semantic_fp_center_radius),
        )
        with log_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(
                {
                    "step": global_step,
                    "loss_total": float(loss_total.detach().cpu().item()),
                    "loss_semantic_fp": float(loss_sem.detach().cpu().item()),
                    "loss_teacher_stability": float(loss_stab.detach().cpu().item()),
                    "loss_oda_preserve": float(loss_oda.detach().cpu().item()),
                    "loss_nonexpansion": float(loss_nonexpansion.detach().cpu().item()),
                    "loss_l2sp": float(loss_l2.detach().cpu().item()),
                    "batch_semantic_region_max_prob": batch_region_max,
                }
            )

        should_eval = (global_step % max(1, int(cfg.candidate_every_steps)) == 0) or global_step == int(cfg.max_steps)
        if not should_eval:
            continue
        ckpt = out_dir / "02_semantic_surgical_checkpoints" / f"step_{global_step:04d}.pt"
        save_ultralytics_yolo(yolo, ckpt)
        result = run_external_hard_suite_for_yolo(
            str(ckpt),
            data_yaml=cfg.data_yaml,
            target_classes=cfg.target_classes,
            cfg=eval_cfg,
            device=cfg.device,
        )
        eval_dir = out_dir / "03_candidate_external" / f"step_{global_step:04d}"
        cand_json, cand_csv = write_external_hard_suite_outputs(result, eval_dir)
        blocked_attacks = _blocked_by_worsening(result, before, float(cfg.max_single_attack_worsen))
        blocked_constraints = blocked_by_hard_constraints(
            result,
            max_attack_asr=cfg.max_attack_asr,
            semantic_fp_required_max_conf=cfg.semantic_fp_required_max_conf,
            semantic_names=tuple(semantic_names),
        ) if bool(cfg.require_hard_constraints) else []
        summary = result.get("summary") or {}
        semantic_conf = semantic_target_absent_max_conf(result, semantic_names=tuple(semantic_names))
        row = {
            "step": global_step,
            "model": str(ckpt),
            "external_json": str(cand_json),
            "external_rows_csv": str(cand_csv),
            "external_max_asr": float(summary.get("max_asr", 1.0)),
            "external_mean_asr": float(summary.get("mean_asr", 1.0)),
            "external_score": _external_score(result),
            "semantic_target_absent_max_conf": float(semantic_conf),
            "blocked_attacks": blocked_attacks,
            "blocked_constraints": blocked_constraints,
            "accepted": (
                (not blocked_attacks)
                and (not blocked_constraints)
                and float(summary.get("max_asr", 1.0)) <= float(cfg.max_allowed_external_asr)
            ),
        }
        candidate_rows.append(row)
        write_json(out_dir / "semantic_surgical_repair_manifest.json", {"status": "running", "candidate_rows": candidate_rows})
        if row["accepted"] and bool(cfg.stop_on_first_accepted):
            break

    selection = _select_candidate(candidate_rows, fallback_model=cfg.model, baseline_score=baseline_score)
    final_row = selection["best"]
    manifest = {
        "status": "passed" if final_row and final_row.get("accepted") else "failed_external_asr_or_worsening",
        "final_model": selection["final_model"],
        "rolled_back": bool(selection["rolled_back"]),
        "input_model": cfg.model,
        "target_class_ids": target_ids,
        "semantic_attack_names": semantic_names,
        "guard_attack_names": guard_names,
        "before_external_json": str(before_json),
        "before_rows_csv": str(before_csv),
        "before_summary": before.get("summary"),
        "before_external_score": baseline_score,
        "repair_data_yaml": str(repair_yaml),
        "replay_stats": replay_stats,
        "clean_anchor_stats": clean_stats,
        "guard_stats": guard_stats,
        "semantic_fp_region_stats": {
            "n_keys": len(semantic_fp_regions),
            "n_regions": sum(len(v) for v in semantic_fp_regions.values()),
            "json": str(out_dir / "semantic_fp_regions.json"),
        },
        "trainable_stats": trainable_stats,
        "n_failure_rows": len(failure_rows),
        "log_csv": str(log_path),
        "candidate_rows": candidate_rows,
        "best": final_row,
        "best_any": selection["best_any"],
    }
    write_json(out_dir / "semantic_surgical_repair_manifest.json", manifest)
    return manifest


@dataclass
class FrontierProfile:
    name: str
    lr: float
    trainable_scope: str
    max_steps: int
    candidate_every_steps: int
    lambda_semantic_fp_threshold: float
    lambda_teacher_stability: float
    lambda_oda_preserve: float
    lambda_target_absent_nonexpansion: float
    lambda_l2sp: float
    semantic_fp_cap: float = 0.245
    trainable_last_n_modules: int = 1


def frontier_profiles(level: str = "last_mile") -> list[FrontierProfile]:
    level = str(level or "last_mile").lower()
    base = [
        FrontierProfile("bias_micro_lr3e-7", 3e-7, "head_bias", 2, 1, 6.0, 120.0, 24.0, 16.0, 2500.0),
        FrontierProfile("bias_micro_lr7e-7", 7e-7, "head_bias", 3, 1, 10.0, 140.0, 28.0, 18.0, 3000.0),
        FrontierProfile("head_micro_lr3e-7", 3e-7, "head", 3, 1, 8.0, 180.0, 34.0, 22.0, 4500.0),
        FrontierProfile("head_micro_lr7e-7", 7e-7, "head", 4, 1, 12.0, 220.0, 40.0, 25.0, 6000.0),
    ]
    if level in {"strong", "frontier", "aggressive"}:
        base.extend(
            [
                FrontierProfile("head_cap235_lr5e-7", 5e-7, "head", 4, 1, 14.0, 260.0, 46.0, 30.0, 7000.0, semantic_fp_cap=0.235),
                FrontierProfile("last2_cap240_lr3e-7", 3e-7, "head", 5, 1, 16.0, 320.0, 55.0, 36.0, 9000.0, semantic_fp_cap=0.240, trainable_last_n_modules=2),
            ]
        )
    return base


def run_frontier_auto_semantic_detox(base_cfg: SemanticSurgicalRepairConfig, *, level: str = "last_mile") -> dict[str, Any]:
    root = Path(base_cfg.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    profile_rows: list[dict[str, Any]] = []
    for prof in frontier_profiles(level):
        prof_out = root / prof.name
        cfg = replace(
            base_cfg,
            out_dir=str(prof_out),
            lr=prof.lr,
            trainable_scope=prof.trainable_scope,
            trainable_last_n_modules=prof.trainable_last_n_modules,
            max_steps=prof.max_steps,
            candidate_every_steps=prof.candidate_every_steps,
            lambda_semantic_fp_threshold=prof.lambda_semantic_fp_threshold,
            lambda_teacher_stability=prof.lambda_teacher_stability,
            lambda_oda_preserve=prof.lambda_oda_preserve,
            lambda_target_absent_nonexpansion=prof.lambda_target_absent_nonexpansion,
            lambda_l2sp=prof.lambda_l2sp,
            semantic_fp_cap=prof.semantic_fp_cap,
            stop_on_first_accepted=True,
        )
        try:
            manifest = run_semantic_surgical_repair(cfg)
            row = {
                "profile": prof.name,
                "status": manifest.get("status"),
                "rolled_back": manifest.get("rolled_back"),
                "final_model": manifest.get("final_model"),
                "best": manifest.get("best"),
                "best_any": manifest.get("best_any"),
                "manifest": str(prof_out / "semantic_surgical_repair_manifest.json"),
            }
        except Exception as exc:
            row = {"profile": prof.name, "status": "error", "error": repr(exc), "manifest": None}
        profile_rows.append(row)
        write_json(root / "frontier_auto_semantic_detox_manifest.json", {"status": "running", "profiles": profile_rows})
        if row.get("status") == "passed" and row.get("final_model"):
            final = {
                "status": "passed",
                "final_model": row.get("final_model"),
                "selected_profile": prof.name,
                "profiles": profile_rows,
            }
            write_json(root / "frontier_auto_semantic_detox_manifest.json", final)
            return final
    best_any = None
    for row in profile_rows:
        cand = (row.get("best") or row.get("best_any")) if isinstance(row, dict) else None
        if not isinstance(cand, Mapping):
            continue
        if best_any is None or (
            float(cand.get("semantic_target_absent_max_conf", 1.0)),
            float(cand.get("external_score", 1.0)),
        ) < (
            float(best_any.get("semantic_target_absent_max_conf", 1.0)),
            float(best_any.get("external_score", 1.0)),
        ):
            best_any = dict(cand)
    final = {
        "status": "failed_external_asr_or_worsening",
        "final_model": base_cfg.model,
        "rolled_back": True,
        "best_any": best_any,
        "profiles": profile_rows,
    }
    write_json(root / "frontier_auto_semantic_detox_manifest.json", final)
    return final

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_security_gate.detox.common import (
    FeatureHookBank,
    ImageTensorDataset,
    LabelledImageTensorDataset,
    attention_alignment_loss,
    collate_images,
    find_ultralytics_weight,
    forward_for_features,
    freeze_module,
    get_ultralytics_yolo,
    infer_device,
    pair_feature_hooks,
    roi_pool_feature_from_yolo_label,
    save_yolo,
    unfreeze_module,
)
from model_security_gate.utils.io import list_images, write_json


@dataclass
class FeatureDetoxConfig:
    imgsz: int = 640
    batch: int = 8
    epochs: int = 5
    lr: float = 1e-5
    weight_decay: float = 1e-5
    max_layers: int = 6
    attention_p: float = 2.0
    device: str | int | None = None
    num_workers: int = 0
    max_images: int = 0


@dataclass
class IBAUFeatureConfig(FeatureDetoxConfig):
    eps: float = 4.0 / 255.0
    alpha: float = 2.0 / 255.0
    inner_steps: int = 2
    clean_loss_weight: float = 0.5


@dataclass
class PrototypeConfig(FeatureDetoxConfig):
    prototype_layer_index: int = -1
    target_class_ids: Sequence[int] | None = None
    min_samples_per_class: int = 3
    prototype_loss_weight: float = 1.0
    attention_loss_weight: float = 0.2


def _make_loader(images_dir: str | Path, imgsz: int, batch: int, max_images: int = 0, num_workers: int = 0):
    paths = list_images(images_dir, max_images=max_images if max_images and max_images > 0 else None)
    ds = ImageTensorDataset(paths, imgsz=imgsz)
    return DataLoader(ds, batch_size=batch, shuffle=True, num_workers=num_workers, collate_fn=collate_images), paths


def run_attention_distillation(
    student_model: str | Path,
    teacher_model: str | Path,
    images_dir: str | Path,
    output_path: str | Path,
    cfg: FeatureDetoxConfig | None = None,
) -> Dict[str, Any]:
    """NAD-style attention distillation without modifying Ultralytics trainer.

    It aligns student intermediate attention maps to a clean teacher on clean and
    counterfactual images. It is trigger-agnostic: no trigger reconstruction is
    required.
    """
    cfg = cfg or FeatureDetoxConfig()
    device = infer_device(cfg.device)
    student_yolo = get_ultralytics_yolo(student_model)
    teacher_yolo = get_ultralytics_yolo(teacher_model)
    student = student_yolo.model.to(device)
    teacher = teacher_yolo.model.to(device)
    unfreeze_module(student)
    freeze_module(teacher)
    s_specs, t_specs = pair_feature_hooks(student, teacher, max_layers=cfg.max_layers)
    loader, paths = _make_loader(images_dir, cfg.imgsz, cfg.batch, cfg.max_images, cfg.num_workers)
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
    history: List[Dict[str, float]] = []

    with FeatureHookBank(s_specs, detach=False) as s_bank, FeatureHookBank(t_specs, detach=True) as t_bank:
        for epoch in range(cfg.epochs):
            losses: List[float] = []
            for batch in tqdm(loader, desc=f"NAD epoch {epoch+1}/{cfg.epochs}"):
                x = batch["image"].to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                s_bank.clear(); t_bank.clear()
                with torch.no_grad():
                    forward_for_features(teacher, x)
                forward_for_features(student, x)
                loss = attention_alignment_loss(s_bank.features, t_bank.features, p=cfg.attention_p)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 10.0)
                opt.step()
                losses.append(float(loss.detach().cpu()))
            history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)) if losses else 0.0})

    out = save_yolo(student_yolo, output_path)
    manifest = {"stage": "nad_attention_distillation", "output": str(out), "config": asdict(cfg), "n_images": len(paths), "history": history}
    write_json(Path(output_path).with_suffix(".json"), manifest)
    return manifest


def run_adversarial_feature_unlearning(
    student_model: str | Path,
    teacher_model: str | Path,
    images_dir: str | Path,
    output_path: str | Path,
    cfg: IBAUFeatureConfig | None = None,
) -> Dict[str, Any]:
    """I-BAU inspired adversarial feature unlearning.

    Inner loop: find small image-space perturbations that maximize divergence
    between student and clean-teacher attention maps. Outer loop: train the
    student to match the teacher on these worst-case perturbations. This simulates
    unknown future triggers without needing to know the actual trigger.
    """
    cfg = cfg or IBAUFeatureConfig()
    device = infer_device(cfg.device)
    student_yolo = get_ultralytics_yolo(student_model)
    teacher_yolo = get_ultralytics_yolo(teacher_model)
    student = student_yolo.model.to(device)
    teacher = teacher_yolo.model.to(device)
    unfreeze_module(student)
    freeze_module(teacher)
    s_specs, t_specs = pair_feature_hooks(student, teacher, max_layers=cfg.max_layers)
    loader, paths = _make_loader(images_dir, cfg.imgsz, cfg.batch, cfg.max_images, cfg.num_workers)
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
    history: List[Dict[str, float]] = []

    with FeatureHookBank(s_specs, detach=False) as s_bank, FeatureHookBank(t_specs, detach=True) as t_bank:
        for epoch in range(cfg.epochs):
            losses: List[float] = []
            for batch in tqdm(loader, desc=f"IBAU epoch {epoch+1}/{cfg.epochs}"):
                x = batch["image"].to(device, non_blocking=True)
                # Teacher clean features are the anchor.
                t_bank.clear()
                with torch.no_grad():
                    forward_for_features(teacher, x)
                    teacher_feats = [f.detach() for f in t_bank.features]

                delta = torch.zeros_like(x, requires_grad=True)
                for _ in range(int(cfg.inner_steps)):
                    s_bank.clear()
                    x_adv = torch.clamp(x + delta, 0.0, 1.0)
                    forward_for_features(student, x_adv)
                    inner_loss = attention_alignment_loss(s_bank.features, teacher_feats, p=cfg.attention_p)
                    grad = torch.autograd.grad(inner_loss, delta, retain_graph=False, create_graph=False)[0]
                    delta.data = torch.clamp(delta.data + float(cfg.alpha) * grad.sign(), -float(cfg.eps), float(cfg.eps))
                    delta.data = torch.clamp(x + delta.data, 0.0, 1.0) - x
                    delta.grad = None

                opt.zero_grad(set_to_none=True)
                # Outer robust loss on adversarial input.
                s_bank.clear()
                x_adv = torch.clamp(x + delta.detach(), 0.0, 1.0)
                forward_for_features(student, x_adv)
                adv_loss = attention_alignment_loss(s_bank.features, teacher_feats, p=cfg.attention_p)
                # Clean alignment stabilizes normal performance.
                s_bank.clear(); t_bank.clear()
                with torch.no_grad():
                    forward_for_features(teacher, x)
                forward_for_features(student, x)
                clean_loss = attention_alignment_loss(s_bank.features, t_bank.features, p=cfg.attention_p)
                loss = adv_loss + float(cfg.clean_loss_weight) * clean_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 10.0)
                opt.step()
                losses.append(float(loss.detach().cpu()))
            history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)) if losses else 0.0})

    out = save_yolo(student_yolo, output_path)
    manifest = {"stage": "adversarial_feature_unlearning", "output": str(out), "config": asdict(cfg), "n_images": len(paths), "history": history}
    write_json(Path(output_path).with_suffix(".json"), manifest)
    return manifest


def _compute_prototypes(
    teacher_model: torch.nn.Module,
    hook_spec,
    loader,
    device: torch.device,
    target_class_ids: Sequence[int] | None = None,
) -> tuple[Dict[int, torch.Tensor], Dict[int, int]]:
    wanted = set(int(x) for x in target_class_ids or [])
    sums: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {}
    with FeatureHookBank([hook_spec], detach=True) as bank:
        for batch in tqdm(loader, desc="Build prototypes"):
            x = batch["image"].to(device, non_blocking=True)
            bank.clear()
            with torch.no_grad():
                forward_for_features(teacher_model, x)
            if not bank.features:
                continue
            feat = bank.features[-1]
            for bi, labels in enumerate(batch.get("labels", [])):
                for lab in labels:
                    cls_id = int(lab["cls_id"])
                    if wanted and cls_id not in wanted:
                        continue
                    vec = roi_pool_feature_from_yolo_label(feat[bi], lab, batch["orig_shape"][bi])
                    if vec is None:
                        continue
                    vec = torch.nn.functional.normalize(vec.detach(), dim=0)
                    if cls_id not in sums:
                        sums[cls_id] = vec.clone()
                        counts[cls_id] = 1
                    else:
                        sums[cls_id] = sums[cls_id] + vec
                        counts[cls_id] += 1
    out: Dict[int, torch.Tensor] = {}
    for cls_id, v in sums.items():
        if counts.get(cls_id, 0) > 0:
            out[cls_id] = torch.nn.functional.normalize(v / counts[cls_id], dim=0).detach()
    return out, counts


def run_prototype_regularization(
    student_model: str | Path,
    teacher_model: str | Path,
    images_dir: str | Path,
    labels_dir: str | Path,
    output_path: str | Path,
    cfg: PrototypeConfig | None = None,
) -> Dict[str, Any]:
    """Prototype-guided activation regularization for labelled detection data.

    For every labelled target bbox, pool the student's region feature and pull it
    toward a clean-teacher prototype for that class. This discourages context-only
    shortcuts because object regions, not full-image background, anchor the loss.
    """
    cfg = cfg or PrototypeConfig()
    device = infer_device(cfg.device)
    student_yolo = get_ultralytics_yolo(student_model)
    teacher_yolo = get_ultralytics_yolo(teacher_model)
    student = student_yolo.model.to(device)
    teacher = teacher_yolo.model.to(device)
    unfreeze_module(student)
    freeze_module(teacher)
    s_specs, t_specs = pair_feature_hooks(student, teacher, max_layers=cfg.max_layers)
    if not s_specs or not t_specs:
        raise RuntimeError("No hookable layers found for prototype regularization")
    layer_idx = int(cfg.prototype_layer_index)
    s_hook = s_specs[layer_idx]
    t_hook = t_specs[layer_idx]
    paths = list_images(images_dir, max_images=cfg.max_images if cfg.max_images and cfg.max_images > 0 else None)
    ds = LabelledImageTensorDataset(paths, labels_dir=labels_dir, imgsz=cfg.imgsz)
    loader = DataLoader(ds, batch_size=cfg.batch, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_images)
    proto_loader = DataLoader(ds, batch_size=cfg.batch, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_images)
    prototypes, proto_counts = _compute_prototypes(teacher, t_hook, proto_loader, device, target_class_ids=cfg.target_class_ids)
    prototypes = {k: v for k, v in prototypes.items() if proto_counts.get(k, 0) >= int(cfg.min_samples_per_class)}
    if not prototypes:
        raise RuntimeError("No prototypes could be built. Check labels_dir, target_class_ids, min_samples_per_class, and dataset paths.")
    prototypes = {k: v.to(device) for k, v in prototypes.items()}
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
    history: List[Dict[str, float]] = []
    wanted = set(int(x) for x in (cfg.target_class_ids or []))

    with FeatureHookBank([s_hook], detach=False) as s_bank, FeatureHookBank([t_hook], detach=True) as t_bank:
        for epoch in range(cfg.epochs):
            losses: List[float] = []
            for batch in tqdm(loader, desc=f"Prototype epoch {epoch+1}/{cfg.epochs}"):
                x = batch["image"].to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                s_bank.clear(); t_bank.clear()
                with torch.no_grad():
                    forward_for_features(teacher, x)
                forward_for_features(student, x)
                if not s_bank.features:
                    continue
                s_feat = s_bank.features[-1]
                proto_losses: List[torch.Tensor] = []
                for bi, labels in enumerate(batch.get("labels", [])):
                    for lab in labels:
                        cls_id = int(lab["cls_id"])
                        if wanted and cls_id not in wanted:
                            continue
                        if cls_id not in prototypes:
                            continue
                        vec = roi_pool_feature_from_yolo_label(s_feat[bi], lab, batch["orig_shape"][bi])
                        if vec is None:
                            continue
                        vec = torch.nn.functional.normalize(vec, dim=0)
                        proto_losses.append(1.0 - torch.sum(vec * prototypes[cls_id]))
                if not proto_losses:
                    continue
                proto_loss = torch.stack(proto_losses).mean()
                att_loss = attention_alignment_loss(s_bank.features, t_bank.features, p=cfg.attention_p) if t_bank.features else proto_loss.new_tensor(0.0)
                loss = float(cfg.prototype_loss_weight) * proto_loss + float(cfg.attention_loss_weight) * att_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 10.0)
                opt.step()
                losses.append(float(loss.detach().cpu()))
            history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)) if losses else 0.0})

    out = save_yolo(student_yolo, output_path)
    manifest = {
        "stage": "prototype_regularization",
        "output": str(out),
        "config": asdict(cfg),
        "n_images": len(paths),
        "prototype_classes": sorted(int(k) for k in prototypes.keys()),
        "prototype_counts": {str(k): int(proto_counts.get(k, 0)) for k in prototypes.keys()},
        "history": history,
    }
    write_json(Path(output_path).with_suffix(".json"), manifest)
    return manifest

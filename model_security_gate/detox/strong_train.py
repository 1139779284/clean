from __future__ import annotations

import copy
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from tqdm import tqdm

from model_security_gate.detox.feature_hooks import (
    ActivationCatcher,
    feature_distillation_loss,
    nad_attention_loss,
    output_distillation_loss,
    select_conv_layers,
)
from model_security_gate.detox.losses import (
    attention_localization_loss,
    pgd_adversarial_images,
    raw_prediction,
    supervised_yolo_loss,
)
from model_security_gate.detox.prototype import PrototypeBank, build_prototype_bank, prototype_alignment_loss, target_prototype_suppression_loss
from model_security_gate.detox.yolo_dataset import make_yolo_dataloader, move_batch_to_device, parse_yolo_data_yaml
from model_security_gate.utils.io import json_default, write_json


@dataclass
class StrongDetoxConfig:
    # Data / IO
    model: str
    data_yaml: str
    out_dir: str = "runs/strong_detox"
    teacher_model: Optional[str] = None
    trusted_teacher_required: bool = False

    # Training
    epochs: int = 20
    batch: int = 8
    imgsz: int = 640
    lr: float = 2e-5
    weight_decay: float = 5e-4
    num_workers: int = 2
    device: Optional[str] = None
    max_train_images: Optional[int] = None
    max_val_images: Optional[int] = None
    grad_clip_norm: float = 10.0
    amp: bool = False

    # Feature layers
    layer_name_contains: Optional[List[str]] = None
    max_hook_layers: int = 6
    prototype_layer: Optional[str] = None
    prototype_max_batches: int = 50

    # Target classes are important for attention localization / target-removal checks.
    target_class_ids: List[int] = field(default_factory=list)

    # Loss weights
    lambda_task: float = 1.0
    lambda_adv: float = 0.4
    lambda_output_distill: float = 0.3
    lambda_feature_distill: float = 0.2
    lambda_nad: float = 0.5
    lambda_attention: float = 0.2
    lambda_prototype: float = 0.25
    lambda_proto_suppress: float = 0.0
    prototype_suppress_margin: float = 0.25

    # I-BAU-style adversarial unlearning
    adv_eps: float = 4.0 / 255.0
    adv_steps: int = 2
    adv_alpha: Optional[float] = None
    adv_random_start: bool = True

    # Optional switches
    use_teacher: bool = True
    use_prototype: bool = True
    use_attention: bool = True
    save_every: int = 5


def _device_from_cfg(cfg: StrongDetoxConfig) -> torch.device:
    if cfg.device:
        return torch.device(cfg.device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_ultralytics_yolo(weights: str | Path, device: torch.device):
    from ultralytics import YOLO

    yolo = YOLO(str(weights))
    if hasattr(yolo, "model"):
        yolo.model.to(device)
    return yolo


def save_ultralytics_yolo(yolo, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    yolo.save(str(path))
    return path


def _torch_model(yolo_or_model) -> torch.nn.Module:
    if hasattr(yolo_or_model, "model") and isinstance(yolo_or_model.model, torch.nn.Module):
        return yolo_or_model.model
    if isinstance(yolo_or_model, torch.nn.Module):
        return yolo_or_model
    raise TypeError("Expected Ultralytics YOLO wrapper or torch model")


def _make_teacher(cfg: StrongDetoxConfig, student_yolo, device: torch.device):
    if cfg.teacher_model:
        teacher_yolo = load_ultralytics_yolo(cfg.teacher_model, device)
        return teacher_yolo, _torch_model(teacher_yolo)
    if cfg.trusted_teacher_required:
        raise ValueError("teacher_model is required because trusted_teacher_required=True")
    # Functional fallback: use a frozen copy of the incoming model. This is not
    # as strong as a trusted teacher, but it stabilizes clean behavior while
    # counterfactual and adversarial losses remove shortcut dependence.
    teacher_yolo = None
    teacher_model = copy.deepcopy(_torch_model(student_yolo)).to(device)
    return teacher_yolo, teacher_model


def _safe_float(x: torch.Tensor | float) -> float:
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def run_strong_detox_training(cfg: StrongDetoxConfig) -> Dict[str, Any]:
    """Run strong trigger-agnostic detox fine-tuning.

    This is the full strong stage that plugs into the previous project:
    - L_task: Ultralytics supervised detection loss on clean/CF labels.
    - I-BAU-style L_adv: inner PGD on unknown perturbations, outer minimization.
    - NAD: attention distillation against a frozen teacher.
    - Output/feature distillation: keep clean behavior close to teacher.
    - Prototype alignment: object-region features return to class prototypes.
    - Attention localization: target-class attention stays inside target boxes.
    """
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "strong_detox_config.json", asdict(cfg))

    device = _device_from_cfg(cfg)
    train_loader, info = make_yolo_dataloader(
        cfg.data_yaml,
        split="train",
        imgsz=cfg.imgsz,
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=cfg.num_workers,
        max_images=cfg.max_train_images,
    )
    val_loader = None
    try:
        val_loader, _ = make_yolo_dataloader(
            cfg.data_yaml,
            split="val",
            imgsz=cfg.imgsz,
            batch_size=cfg.batch,
            shuffle=False,
            num_workers=cfg.num_workers,
            max_images=cfg.max_val_images,
        )
    except Exception:
        val_loader = None

    student_yolo = load_ultralytics_yolo(cfg.model, device)
    student = _torch_model(student_yolo).to(device)
    teacher_yolo, teacher = _make_teacher(cfg, student_yolo, device)
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student_layers = select_conv_layers(
        student,
        contains=cfg.layer_name_contains,
        max_layers=cfg.max_hook_layers,
        prefer_late=True,
    )
    teacher_layers = select_conv_layers(
        teacher,
        contains=cfg.layer_name_contains,
        max_layers=len(student_layers),
        prefer_late=True,
    )
    # If names do not match, NAD can only be applied where names align. For a
    # copied or same-arch teacher this will be full alignment.
    aligned_layers = [x for x in student_layers if x in set(teacher_layers)]
    if not aligned_layers:
        aligned_layers = student_layers if teacher_yolo is None else []

    prototype_bank: Optional[PrototypeBank] = None
    if cfg.use_prototype and cfg.lambda_prototype > 0:
        try:
            prototype_bank = build_prototype_bank(
                teacher,
                train_loader,
                layer_name=cfg.prototype_layer or (aligned_layers[-1] if aligned_layers else None),
                max_batches=cfg.prototype_max_batches,
                device=device,
            ).to(device)
            if cfg.prototype_layer is None:
                cfg.prototype_layer = prototype_bank.layer_name
        except Exception as exc:
            prototype_bank = None
            print(f"[WARN] Prototype bank disabled: {exc}")

    optimizer = torch.optim.AdamW(student.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.amp and device.type == "cuda"))
    log_path = out_dir / "strong_detox_train_log.csv"
    fields = [
        "epoch",
        "step",
        "loss_total",
        "loss_task",
        "loss_adv",
        "loss_output_distill",
        "loss_feature_distill",
        "loss_nad",
        "loss_attention",
        "loss_prototype",
        "loss_proto_suppress",
    ]
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

    global_step = 0
    best_val_loss = None
    best_path = None

    for epoch in range(1, int(cfg.epochs) + 1):
        student.train()
        pbar = tqdm(train_loader, desc=f"Strong detox epoch {epoch}/{cfg.epochs}")
        epoch_losses: List[float] = []
        for batch in pbar:
            global_step += 1
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=bool(cfg.amp and device.type == "cuda")):
                loss_task = supervised_yolo_loss(student, batch) * float(cfg.lambda_task)

                # Prediction forward for distillation/NAD/prototype/attention.
                if aligned_layers:
                    with ActivationCatcher(student, aligned_layers) as s_ac:
                        s_out = raw_prediction(student, batch["img"])
                    with torch.no_grad():
                        with ActivationCatcher(teacher, aligned_layers) as t_ac:
                            t_out = raw_prediction(teacher, batch["img"])
                    s_feats = s_ac.features
                    t_feats = t_ac.features
                else:
                    s_out = raw_prediction(student, batch["img"])
                    with torch.no_grad():
                        t_out = raw_prediction(teacher, batch["img"])
                    s_feats = {}
                    t_feats = {}

                loss_output_distill = output_distillation_loss(s_out, t_out) * float(cfg.lambda_output_distill) if cfg.use_teacher else loss_task * 0.0
                loss_feature_distill = feature_distillation_loss(s_feats, t_feats) * float(cfg.lambda_feature_distill) if cfg.use_teacher and s_feats else loss_task * 0.0
                loss_nad = nad_attention_loss(s_feats, t_feats) * float(cfg.lambda_nad) if cfg.use_teacher and s_feats else loss_task * 0.0
                loss_attention = attention_localization_loss(s_feats, batch, cfg.target_class_ids) * float(cfg.lambda_attention) if cfg.use_attention and cfg.target_class_ids and s_feats else loss_task * 0.0
                loss_prototype = prototype_alignment_loss(s_feats, batch, prototype_bank) * float(cfg.lambda_prototype) if prototype_bank is not None and s_feats else loss_task * 0.0
                loss_proto_suppress = target_prototype_suppression_loss(
                    s_feats, batch, prototype_bank, cfg.target_class_ids, margin=float(cfg.prototype_suppress_margin)
                ) * float(cfg.lambda_proto_suppress) if prototype_bank is not None and s_feats and cfg.target_class_ids else loss_task * 0.0

                if cfg.lambda_adv > 0 and cfg.adv_steps > 0:
                    adv_img = pgd_adversarial_images(
                        student,
                        batch,
                        eps=float(cfg.adv_eps),
                        alpha=cfg.adv_alpha,
                        steps=int(cfg.adv_steps),
                        random_start=bool(cfg.adv_random_start),
                    )
                    adv_batch = dict(batch)
                    adv_batch["img"] = adv_img
                    loss_adv = supervised_yolo_loss(student, adv_batch) * float(cfg.lambda_adv)
                else:
                    loss_adv = loss_task * 0.0

                loss_total = loss_task + loss_adv + loss_output_distill + loss_feature_distill + loss_nad + loss_attention + loss_prototype + loss_proto_suppress

            scaler.scale(loss_total).backward()
            if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), float(cfg.grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()

            row = {
                "epoch": epoch,
                "step": global_step,
                "loss_total": _safe_float(loss_total),
                "loss_task": _safe_float(loss_task),
                "loss_adv": _safe_float(loss_adv),
                "loss_output_distill": _safe_float(loss_output_distill),
                "loss_feature_distill": _safe_float(loss_feature_distill),
                "loss_nad": _safe_float(loss_nad),
                "loss_attention": _safe_float(loss_attention),
                "loss_prototype": _safe_float(loss_prototype),
                "loss_proto_suppress": _safe_float(loss_proto_suppress),
            }
            epoch_losses.append(row["loss_total"])
            pbar.set_postfix({"loss": f"{row['loss_total']:.4f}", "task": f"{row['loss_task']:.4f}"})
            with open(log_path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)

        # Lightweight validation: supervised loss only.
        val_loss = None
        if val_loader is not None:
            student.eval()
            vals: List[float] = []
            with torch.no_grad():
                for vb in val_loader:
                    vb = move_batch_to_device(vb, device)
                    vals.append(_safe_float(supervised_yolo_loss(student, vb)))
            if vals:
                val_loss = sum(vals) / len(vals)

        ckpt_path = out_dir / f"epoch_{epoch}.pt"
        if cfg.save_every and (epoch % int(cfg.save_every) == 0 or epoch == cfg.epochs):
            save_ultralytics_yolo(student_yolo, ckpt_path)
        metric = val_loss if val_loss is not None else (sum(epoch_losses) / max(1, len(epoch_losses)))
        if best_val_loss is None or metric < best_val_loss:
            best_val_loss = metric
            best_path = save_ultralytics_yolo(student_yolo, out_dir / "best_strong_detox.pt")

    final_path = save_ultralytics_yolo(student_yolo, out_dir / "last_strong_detox.pt")
    report = {
        "final_model": str(final_path),
        "best_model": str(best_path) if best_path else str(final_path),
        "log_csv": str(log_path),
        "names": info.names,
        "target_class_ids": cfg.target_class_ids,
        "hook_layers": aligned_layers,
        "prototype_layer": cfg.prototype_layer,
        "best_validation_or_train_loss": best_val_loss,
    }
    write_json(out_dir / "strong_detox_report.json", report)
    return report

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from model_security_gate.detox.common import find_ultralytics_weight


def train_yolo_teacher(
    trusted_base_model: str | Path,
    data_yaml: str | Path,
    output_project: str | Path,
    name: str = "teacher",
    imgsz: int = 640,
    epochs: int = 40,
    batch: int = 16,
    device: str | int | None = None,
    **train_kwargs: Any,
) -> Path:
    """Train a clean teacher from a trusted checkpoint on clean/counterfactual data.

    The teacher is used for later NAD-style attention distillation and adversarial
    feature unlearning. Prefer a trusted official/pretrained checkpoint, not the
    suspicious model itself.
    """
    from ultralytics import YOLO

    model = YOLO(str(trusted_base_model))
    kwargs: Dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "project": str(output_project),
        "name": name,
        "exist_ok": True,
        "optimizer": train_kwargs.pop("optimizer", "AdamW"),
        "lr0": float(train_kwargs.pop("lr0", 1e-4)),
        "weight_decay": float(train_kwargs.pop("weight_decay", 5e-4)),
        "hsv_h": float(train_kwargs.pop("hsv_h", 0.03)),
        "hsv_s": float(train_kwargs.pop("hsv_s", 0.5)),
        "hsv_v": float(train_kwargs.pop("hsv_v", 0.4)),
        "mosaic": float(train_kwargs.pop("mosaic", 0.7)),
        "mixup": float(train_kwargs.pop("mixup", 0.1)),
        "copy_paste": float(train_kwargs.pop("copy_paste", 0.1)),
        "erasing": float(train_kwargs.pop("erasing", 0.25)),
        "label_smoothing": float(train_kwargs.pop("label_smoothing", 0.03)),
        "close_mosaic": int(train_kwargs.pop("close_mosaic", 5)),
    }
    if device is not None:
        kwargs["device"] = device
    kwargs.update(train_kwargs)
    model.train(**kwargs)
    return find_ultralytics_weight(output_project, name, prefer="best")

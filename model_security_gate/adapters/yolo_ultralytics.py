from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from model_security_gate.utils.torchvision_compat import patch_torchvision_nms_fallback

from .base import Detection


class UltralyticsYOLOAdapter:
    """Thin adapter around Ultralytics YOLO.

    The adapter intentionally exposes only prediction functions needed by the
    scanner. Training remains in detox/train_ultralytics.py.
    """

    def __init__(
        self,
        weights: str | Path,
        device: str | int | None = None,
        default_conf: float = 0.25,
        default_iou: float = 0.7,
        default_imgsz: int = 640,
    ) -> None:
        patch_torchvision_nms_fallback()
        from ultralytics import YOLO

        self.weights = str(weights)
        self.model = YOLO(str(weights))
        self.device = device
        self.default_conf = default_conf
        self.default_iou = default_iou
        self.default_imgsz = default_imgsz
        raw_names = getattr(self.model, "names", {}) or {}
        if isinstance(raw_names, list):
            self.names: Dict[int, str] = {i: str(v) for i, v in enumerate(raw_names)}
        else:
            self.names = {int(k): str(v) for k, v in dict(raw_names).items()}

    def _to_detection_list(self, result) -> List[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy().astype(int)
        out: List[Detection] = []
        for box, score, cls_id in zip(xyxy, conf, cls):
            out.append(
                Detection(
                    xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    conf=float(score),
                    cls_id=int(cls_id),
                    cls_name=self.names.get(int(cls_id), str(cls_id)),
                )
            )
        return out

    def predict_image(
        self,
        image: Union[str, Path, np.ndarray],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
    ) -> List[Detection]:
        results = self.model.predict(
            source=image,
            conf=self.default_conf if conf is None else conf,
            iou=self.default_iou if iou is None else iou,
            imgsz=self.default_imgsz if imgsz is None else imgsz,
            device=self.device,
            verbose=False,
        )
        return self._to_detection_list(results[0]) if results else []

    def predict_batch(
        self,
        images: Sequence[Union[str, Path, np.ndarray]],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
    ) -> List[List[Detection]]:
        if not images:
            return []
        results = self.model.predict(
            source=list(images),
            conf=self.default_conf if conf is None else conf,
            iou=self.default_iou if iou is None else iou,
            imgsz=self.default_imgsz if imgsz is None else imgsz,
            device=self.device,
            verbose=False,
        )
        return [self._to_detection_list(r) for r in results]

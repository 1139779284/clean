from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from model_security_gate.adapters.base import Detection


@dataclass
class SemanticShortcutGuardConfig:
    enabled: bool = True
    min_target_conf: float = 0.25
    hardhat_evidence_max: float = 0.45
    high_vis_context_min: float = 0.18
    context_x_expand: float = 0.8
    context_y_down: float = 4.0
    helmet_top_fraction: float = 0.65


@dataclass
class SemanticShortcutMatch:
    reason: str
    cls_id: int
    cls_name: str
    conf: float
    xyxy: tuple[float, float, float, float]
    hardhat_evidence: float
    high_vis_context: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _crop_rgb(image: Image.Image, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    x1 = max(0, min(image.width, x1))
    x2 = max(0, min(image.width, x2))
    y1 = max(0, min(image.height, y1))
    y2 = max(0, min(image.height, y2))
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=np.float32)
    return np.asarray(image.crop((x1, y1, x2, y2)).convert("RGB"), dtype=np.float32) / 255.0


def hardhat_visual_evidence(image: Image.Image, xyxy: Sequence[float], top_fraction: float = 0.65) -> float:
    x1, y1, x2, y2 = [float(v) for v in xyxy[:4]]
    height = max(1.0, y2 - y1)
    crop = _crop_rgb(image, (round(x1), round(y1), round(x2), round(y1 + top_fraction * height)))
    if crop.size == 0:
        return 0.0
    red, green, blue = crop[:, :, 0], crop[:, :, 1], crop[:, :, 2]
    max_rgb = np.max(crop, axis=2)
    min_rgb = np.min(crop, axis=2)
    saturation = (max_rgb - min_rgb) / (max_rgb + 1e-6)
    value = max_rgb

    white_shell = (value > 0.62) & (saturation < 0.30)
    yellow_shell = (red > 0.45) & (green > 0.38) & (blue < 0.35) & (((red + green) / 2.0) > blue * 1.35)
    red_shell = (red > 0.55) & (green < 0.38) & (blue < 0.38) & (red > green * 1.35) & (red > blue * 1.35)
    blue_shell = (blue > 0.38) & (blue > red * 1.08) & (blue > green * 0.90) & (saturation > 0.18)
    return float(np.mean(white_shell | yellow_shell | red_shell | blue_shell))


def high_visibility_context_score(
    image: Image.Image,
    xyxy: Sequence[float],
    x_expand: float = 0.8,
    y_down: float = 4.0,
) -> float:
    x1, y1, x2, y2 = [float(v) for v in xyxy[:4]]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    crop = _crop_rgb(
        image,
        (
            round(x1 - x_expand * width),
            round(y2),
            round(x2 + x_expand * width),
            round(y2 + y_down * height),
        ),
    )
    if crop.size == 0:
        return 0.0
    red, green, blue = crop[:, :, 0], crop[:, :, 1], crop[:, :, 2]
    max_rgb = np.max(crop, axis=2)
    min_rgb = np.min(crop, axis=2)
    saturation = (max_rgb - min_rgb) / (max_rgb + 1e-6)
    value = max_rgb
    green_yellow_vest = (
        (green > 0.42)
        & (red > 0.35)
        & (blue < 0.35)
        & (value > 0.45)
        & (saturation > 0.22)
        & (green > blue * 1.25)
    )
    yellow_vest = (red > 0.50) & (green > 0.48) & (blue < 0.35) & (saturation > 0.20)
    return float(np.mean(green_yellow_vest | yellow_vest))


def decide_semantic_shortcut_guard(
    image_path: str | Path,
    detections: Sequence[Detection],
    target_class_ids: Sequence[int],
    cfg: SemanticShortcutGuardConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or SemanticShortcutGuardConfig()
    if not cfg.enabled:
        return {"action": "pass", "matches": []}
    image = Image.open(image_path).convert("RGB")
    target_ids = {int(x) for x in target_class_ids}
    matches: list[SemanticShortcutMatch] = []
    for det in detections:
        if det.cls_id not in target_ids or det.conf < cfg.min_target_conf:
            continue
        hardhat_score = hardhat_visual_evidence(image, det.xyxy, top_fraction=cfg.helmet_top_fraction)
        context_score = high_visibility_context_score(
            image,
            det.xyxy,
            x_expand=cfg.context_x_expand,
            y_down=cfg.context_y_down,
        )
        if hardhat_score < cfg.hardhat_evidence_max and context_score > cfg.high_vis_context_min:
            matches.append(
                SemanticShortcutMatch(
                    reason="weak helmet visual evidence with high-visibility vest context",
                    cls_id=det.cls_id,
                    cls_name=det.cls_name,
                    conf=float(det.conf),
                    xyxy=tuple(float(v) for v in det.xyxy),
                    hardhat_evidence=hardhat_score,
                    high_vis_context=context_score,
                )
            )
    return {
        "action": "review" if matches else "pass",
        "matches": [match.to_dict() for match in matches],
    }
